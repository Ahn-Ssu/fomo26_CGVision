# 05. 다운스트림 파인튜닝 파이프라인 — teacher-specific-only finetuning (2026-08-08)

**목적**: 대회가 요구하는 5개 다운스트림 태스크에 `ver3`/`ver4`/`ver5` 사전학습 체크포인트를 파인튜닝하되, shared backbone은 얼리고 teacher-specific parameter(Convpass adapter)만 학습하는 방식으로 진행. 상세 설계 배경 및 근거는 승인된 계획 파일(`/root/.claude/plans/mossy-wibbling-rabbit.md`) 참고, 이 문서는 실제 구현·검증 결과를 기록한다.

---

## 0. 요약 (TL;DR)

| 질문 | 답 |
|---|---|
| 대회 문서의 리터럴 레시피(`asp_finetune_* +model=resenc_unet_b(_clsreg)`)를 그대로 쓸 수 있는가? | **아니오** — 우리 체크포인트와 Asparagus의 stock 모델(`gardening_tools.ResidualUNetEncoder`)은 서브모듈 이름이 근본적으로 다르고(§1), Convpass 자체가 stock 모델에 없어서 그대로 쓰면 pretrained 가중치가 사실상 로드되지 않음 |
| 그럼 어떻게 했는가? | Asparagus의 Hydra 모델 슬롯에 우리 자체 아키텍처(`StudentResEncUNet`)를 새로 등록(`fomo26_student.py` + 신규 model config) — 데이터 파이프라인/Trainer/지표/체크포인트 저장은 Asparagus 원본 그대로, 모델만 교체 |
| 검증됐는가? | **예** — ver5(280K) + Task 1(Infarct 분류) + teacher `brats`로 pretrained-init/from-scratch-init 두 조합 모두 end-to-end 완주(학습 3 epoch + 검증 + 테스트) 확인 |

---

## 1. 핵심 발견: 왜 대회 리터럴 레시피가 안 통하는가

`networks/student.py`(`StudentResEncUNet`)의 체크포인트 키(`model.encoder.stages.0.block.conv1.weight`, `model.encoder.stages.0.convpass.<teacher>.down.weight` 등)와 Asparagus의 `resenc_unet_b`(`gardening_tools.ResidualUNetEncoder`)를 직접 소스 대조:

- `gardening_tools`는 별도 `self.stem` 모듈이 있음(우리는 stage 0이 stem 역할) — 구조 자체가 다름.
- `gardening_tools`의 스테이지는 `StackedResidualBlocks`로 블록이 여러 개면 `blocks.<i>...`(복수형+인덱스), 우리는 `.block`(단수) — 이름 불일치.
- 블록 내부도 `ConvDropoutNormNonlin`으로 한 겹 더 감싸져 있어 leaf tensor 경로 자체가 다름.
- **Convpass adapter는 `gardening_tools`에 아예 존재하지 않음.**

Asparagus의 실제 체크포인트 전이 함수(`BaseModule.load_state_dict` → `should_load_key`, `base_module.py`)는 **완전 동일한 dotted-name + shape 매치만** 허용하고 `assert successful > 0`만 확인한다. 즉 대회 문서 그대로 `+model=resenc_unet_b_clsreg checkpoint_path=our_ckpt.pt`를 실행하면 pretrained encoder/decoder/Convpass 가중치가 사실상 로드되지 않고 random-init에서 시작하게 된다 — 실험 목적 자체가 무효화됨. (이건 `run_pretrain.py`의 기존 docstring이 가정했던 "no shim needed"보다 한 단계 더 깊은 문제였음 — prefix는 이미 맞았지만 서브모듈 이름까지는 안 맞았다.)

## 2. 구현: Asparagus 로컬 fork + 커스텀 모델 등록

### 신규 파일

- **`/root/asparagus/asparagus/modules/networks/fomo26_student.py`**
  - `_load_checkpoint_state_dict`: `"model."` prefix 제거해 `StudentResEncUNet` 고유 키로 복원.
  - `_infer_student_config`: 체크포인트 키만 보고 teacher roster / `convpass_encoder` / `skip_alpha` 플래그를 역추론 — ver3(decoder-only convpass)/ver4/ver5(encoder+decoder convpass) 전부 하드코딩 없이 동일 코드로 처리.
  - `_load_with_channel_repeat`: 우리 사전학습은 단일 모달리티(`in_channels=1`)인데 Task 1은 3모달리티(FLAIR+ADC+DWI) 입력이라 stem conv(및 stage-0 Convpass의 `down` conv)의 입력 채널 수가 안 맞음 — Asparagus 자체의 `repeat_stem_weights` 관례와 동일한 방식(반복+평균)으로 일반화해서 처리.
  - `FomoStudentBackboneMixin._load_and_configure`: 체크포인트 로드 → `backbone_parameters()` 전체 freeze → 지정된 teacher의 Convpass/skip_alpha 파라미터만 `requires_grad=True`로 복원 → `from_scratch=True`면 그 teacher의 Convpass만 재초기화(`Convpass3D.__init__`과 동일한 Kaiming 스킴, gate=0/skip_alpha=1).
  - `FomoStudentClsRegNet`: encoder 마지막 스테이지(bottleneck, 320채널) 출력에 `gardening_tools.ClsRegHead`(재사용, 새로 안 만듦)를 붙인 분류/회귀 헤드.
  - Asparagus의 top-level `checkpoint_path=`/`weights=` 메커니즘은 **의도적으로 사용하지 않음**(`resolve_checkpoint()`가 `None` 반환하도록 비워둠) — 우리 체크포인트 로딩은 전부 이 파일 안에서 직접 수행, `should_load_key`의 exact-name-matching 리스크를 원천 차단.
- **`configs/model/core/fomo26_student.yaml`**, **`configs/model/fomo26_student_clsreg.yaml`** — `_target_`을 위 클래스로 연결하는 Hydra config. `model.checkpoint_path`/`model.teacher_name`/`model.from_scratch`가 CLI 오버라이드 가능한 필드.

### 그대로 재사용

`asp_process`/`asp_split`(별도 GitHub repo `asparagus_preprocessing`, FOMO26 5개 태스크 전처리 스크립트가 이미 포함돼 있어 새로 작성 불필요), `asp_finetune_cls/seg/reg`, `ClsRegBase`, 지표, Trainer 루프, 체크포인트 저장 — 전부 원본.

## 3. 환경 이슈 및 해결 (재현 시 참고)

1. **`/opt/conda` base env의 numpy1/2 충돌** — `01_ASPARAGUS_ANALYSIS.md`가 이미 문서화한 이슈, 이번에도 그대로 재현(`asp_process` 실행 시 pandas가 `numpy.core.multiarray failed to import`). → 격리된 venv(`/root/asparagus_venv`)에 `asparagus`+`asparagus_preprocessing`을 새로 설치해서 해결(numpy/pandas가 서로 호환되는 버전으로 새로 resolve됨).
2. **`monai`/`timm`/`lightly` 암묵적 필수 의존성** — `01_ASPARAGUS_ANALYSIS.md` E.7에 이미 문서화된 이슈(`DinoV2.py`가 무조건 `import monai`). 설치로 해결하되, `lightly`가 딸려오는 `torch`를 2.13.0으로 끌어올려 CUDA 드라이버 비호환(구동 드라이버가 CUDA 13 지원 안 함) 유발 → `torch==2.6.0+cu124` 재고정, 이 과정에서 `nvidia-cudnn-cu12`가 버전 불일치 상태로 남아 `CUDNN_STATUS_NOT_INITIALIZED` 발생 → `--force-reinstall nvidia-cudnn-cu12==9.1.0.70`로 해결.
3. **`torch.compile`(하드웨어 프리셋 기본값 `compile_mode: default`)와의 상호작용은 미확인** — cuDNN 문제 해결 전에 컴파일 경로에서도 같은 에러가 났었는데, 근본 원인(cuDNN)을 고친 뒤에는 `compile_mode=null`로 끄고 검증했다. 커스텀 아키텍처(teacher별 ModuleDict/ParameterDict 동적 인덱싱)가 `torch.compile`과 실제로 잘 맞는지는 별도 확인 필요 — 스케일업 시 성능이 중요해지면 재검토 권장.
4. **`asp_finetune_cls`의 `data.test_split` 필수** — 대회 문서 예시 커맨드에는 있지만 생략하면 `cfg.data.test_split`이 `None`이라 `finetune_cls.py`에서 `TypeError`(기존 코드 자체의 사소한 버그, 우리 쪽 이슈 아님) — `data.test_split=TEST_80_10_10` 항상 포함할 것.

## 4. 검증 결과

체크포인트: `ver5`의 `step_280000.pt`. 태스크: Task 1(`CLS002_FOMO26_Infarct`, 21 subjects, FLAIR+ADC+DWI 3모달리티, 2-class). Teacher: `brats`(임의 선택, 파이프라인 검증 목적).

| 항목 | 확인 결과 |
|---|---|
| 체크포인트 전이 | **329/329 backbone 텐서 전부 로드**(missing=0, unexpected=0) — 채널 반복 로직 포함 정상 동작 |
| Freeze 정확성 | teacher `brats` 기준 **72,237 / 32,160,633 (0.225%)** 파라미터만 trainable, 나머지 전부 frozen |
| `from_scratch=True` 재초기화 | 11개 Convpass 어댑터(encoder 6 + decoder 5 스테이지) 재초기화 확인, `from_scratch=True` vs `False` 간 38/44 trainable 텐서 값이 실제로 다름을 직접 대조 확인 |
| 학습 루프 | 3 epoch 전부 정상 완주(pretrained-init: train/loss 0.668→0.654, train/AUROC 0.458→0.645; from-scratch-init: train/AUROC 0.488→0.623 — pretrained-init이 약간 더 빠르게 학습, 기대와 일치) |
| 검증/테스트 | val/train 지표(loss, accuracy, AUROC) 정상 로깅, `trainer.test()`의 best-checkpoint 복원 + Precision/Recall 집계까지 정상 동작 |

3 epoch(전체 250 step/epoch 제한 내)만 돌린 스모크 테스트라 두 조건의 테스트셋 성능(Precision/Recall)이 이번엔 동일하게 나왔음 — 21-subject의 극소 데이터셋 + class imbalance + 짧은 학습 때문으로, 파이프라인 자체의 문제는 아님(train-side AUROC 궤적에서는 이미 차이가 보임). 실제 비교 실험은 대회 문서 권장 epoch 수(Task 1: 50 epoch)로 늘려서 재실행 필요.

## 5. 다음 단계

1. 이번 검증에 쓴 두 커맨드를 `training.epochs=50`(대회 문서 권장값) 등 실제 하이퍼파라미터로 늘려서 재실행, `brats` teacher에 대한 진짜 비교 결과 확보.
2. 나머지 teacher(anatomix+brains/vesselfm/voco/vjepa)로 동일 조합 확장.
3. `FomoStudentSegNet`(Task 2/4, segmentation) 구현 — decoder 마지막 스테이지 출력 + 1x1x1 conv head, `deep_supervision=False` 고정.
4. Task 3(Brain Age, regression)은 `FomoStudentClsRegNet`을 `output_channels=1`로 그대로 재사용 가능(회귀·분류 모두 `ClsRegHead`/`ClsRegBase` 공유 구조).
5. `ver3`/`ver4` 체크포인트로도 동일 파이프라인이 통하는지 확인(`_infer_student_config`가 두 버전 다 처리하도록 설계돼 있으나 실제 실행 검증은 아직 안 함).

## 6. 실행 예시 (재현용)

```bash
source /root/asparagus_env.sh
cd /root/asparagus
/root/asparagus_venv/bin/asp_finetune_cls \
  task=CLS002_FOMO26_Infarct \
  +model=fomo26_student_clsreg \
  model.checkpoint_path=/root/FOMO26/expr/ver5/checkpoints/step_280000.pt \
  model.teacher_name=brats \
  model.from_scratch=false \
  data.train_split=split_80_10_10 \
  data.test_split=TEST_80_10_10 \
  data.fold=0 \
  hardware.num_workers=8 \
  hardware.compile_mode=null \
  training.batch_size=2 \
  training.epochs=50 \
  logger.wandb_logging=false
```
