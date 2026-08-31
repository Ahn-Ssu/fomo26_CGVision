# Asparagus 코드베이스 분석 — FOMO26 멀티티처 episodic distillation 통합 가능성

조사 대상: `Sllambias/asparagus` (commit at clone time, 2026-07-13), `Sllambias/asparagus_preprocessing`.
방법: 소스 직접 열람(에이전트 4개 병렬 조사 + 직접 열람) + `pip install -e .` 실제 설치 + `pytest tests/` 전체 실행 + `asp_pretrain` CLI를 synthetic 데이터로 실제 실행.

---

## A. 구조 요약

### A.1 디렉토리 트리 + 역할

```
asparagus/
├── functional/               # torch.nn.functional 스타일 — 순수 함수
│   └── metrics/               # SSL 사전학습 진단용 지표 (재구성 SSIM, feature-collapse 등).
│                               # 주의: 다운스트림 태스크 지표(Dice/AUROC/NSD)는 여기 없음
├── modules/                  # torch.nn 스타일 — 상태를 갖는 클래스
│   ├── callbacks/             # Lightning Callback (ssl_training.py의 OnlineSegmentationPlugin 등)
│   ├── data_modules/          # LightningDataModule (pretraining.py: PretrainDataModule)
│   ├── datasets/              # Dataset 클래스 (PretrainDataset.py, TrainDataset.py)
│   ├── dataclasses/            # config 데이터클래스 + presets/
│   ├── hydra/plugins/         # 커스텀 Hydra searchpath/resolver 플러그인
│   ├── lightning_modules/     # LightningModule (base_module, self_supervised, dinov2,
│   │                           # segmentation_module, clsreg_module, linear_probe_module)
│   ├── losses/                # dinov2.py(DINOv2Loss), ibot.py(IBOTPatchLoss3D) — SelfSupervisedModule은 미사용
│   ├── networks/               # 모델 팩토리 (unet.py, resenc_unet.py, primus.py, dinov2.py,
│   │                           # vision_transformer.py) — 실제 아키텍처 바디는 외부 패키지 gardening_tools에 있음
│   └── transforms/             # crop.py, pad.py, clamp.py, blockmask.py, DinoV2.py, presets/
├── notebooks/                 # ⚠ 디버그/탐색용 노트북 6개뿐. 학습 로직 없음 (아래 E 참고)
├── pipeline/
│   ├── auto_configuration/    # checkpoint.py, experiment_setup.py, versioning.py 등
│   ├── evaluation/             # EvalBox 관련 헬퍼
│   ├── run/                    # CLI 엔트리포인트 스크립트 (pretrain.py, finetune_*.py 등)
│   └── shell/                  # Slurm/쉘 스크립트 생성
└── scripts/
    ├── FOMO26/                 # ⚠ Task1~5, Task6_and_7 predict 스크립트만 존재 (추론 전용, 학습/지표 코드 없음)
    └── testing/

configs/
├── core/base.yaml             # 모든 스크립트가 공유하는 Hydra 배선 (run_dir 템플릿, lightning/_target_, transforms 배선)
├── model/core/{unet,resenc_unet,primus,ultradino}.yaml   # 아키텍처 계열별 _target_ 정의
├── model/*.yaml                # 크기 프리셋 (unet_tiny/s/m/b_lw_dec, resenc_unet_b(_clsreg), primus_s/m, dinov2_test)
├── hardware/*.yaml             # 디바이스/precision/strategy 프리셋
├── evaluation/{dev_box,dev_box_dtu}.yaml   # EvalBox 설정
├── plugins/main_segmentation.yaml
├── default_{pretrain,train_*,finetune_*,test,predict,linear_probe}.yaml   # 스크립트별 최상위 config
└── projects/{datapaper,development}/...    # 실제 실험 config 모음
```

### A.2 Entry point 매핑표 (`pyproject.toml`)

| CLI 명령 | 매핑 함수 | 비고 |
|---|---|---|
| `asp_getid` | `pipeline.run.get_id:main` | |
| `asp_pretrain` | `pipeline.run.pretrain:main` | 기본 config: `default_pretrain.yaml` → `SelfSupervisedModule` + `PretrainDataModule` |
| `asp_train_cls/seg/reg` | `pipeline.run.train_{cls,seg,reg}:main` | scratch 학습 |
| `asp_test_cls` | `pipeline.run.test_cls:main` | |
| `asp_test_seg` | `pipeline.run.test_seg:main` | |
| `asp_test_reg` | `pipeline.run.test_cls:main` | ⚠ **`test_cls`를 가리킴 — 사실상 복붙 버그로 보임** (E 참고) |
| `asp_finetune_seg/cls/reg` | `pipeline.run.finetune_{seg,cls,reg}:main` | |
| `asp_linear_probe` | `pipeline.run.linear_probe:main` | |
| `asp_eval_box_run` | `pipeline.run.eval_box:main` | |
| `asp_eval_box_prepare_data` | `pipeline.run.eval_box:prepare_data` | |
| `asp_eval_box_collect_results` | `pipeline.run.eval_box:collect_results` | |

### A.3 Notebook 89.3% 문제 — 확인 결과

`notebooks/`에는 `padding.ipynb`, `Test augmentations.ipynb`, `test_downsample_label.ipynb`, `model_sizes.ipynb`, `ttUnet_pp_comparison.ipynb`, `debug/hydracfgs.ipynb` 6개만 존재하며, 모두 디버그/탐색용이다. `asp_pretrain` 등 모든 entry point는 100% `.py` 파일(`pipeline/run/*.py` → `modules/lightning_modules/*.py`)로 구현되어 있다. GitHub 언어 통계의 "Jupyter Notebook 89.3%"는 `.ipynb`가 JSON+출력 셀 형태로 바이트 수가 부풀려지는 잘 알려진 통계 아티팩트이며, **실제 학습 로직에 대한 신뢰도 문제는 아님**.

---

## B. 통합 가능성 판정표

| 컴포넌트 | 판정 | 근거 (파일:라인) | 필요한 작업 |
|---|---|---|---|
| **Data preprocessing** | **수정 후 재사용** | `asparagus_preprocessing/utils/resample.py:35-39`, `utils/normalize.py:6-75` — 정규화가 전처리 시점에 모달리티별 1개 스킴으로 디스크에 baked-in됨 (`torch.save` 전에 `normalizer()` 호출) | teacher별 다른 정규화(BraTS z-norm vs VesselFM percentile-01)를 위해 raw intensity 보존 프리셋(`normalization_operation=no_norm`) 추가, 정규화를 로드 시점으로 이연 |
| **Dataset / DataLoader** | **수정 후 재사용** | `PretrainDataset.py:21-31`이 반환하는 dict은 `{file_path, image, transforms_applied}` 뿐, modality/subject_id/spacing 없음; `pretraining.py:58-87` DataLoader에 custom `collate_fn` 없음(`default_collate`) | metadata 필드를 담는 Dataset 서브클래스 + custom collate_fn 신규 작성. `crop.py`/`pad.py`/spatial transform은 그대로 재사용 가능 |
| **Model (Primus/ResEnc)** | **재사용 가능** | `configs/model/core/{primus,resenc_unet}.yaml` → `asparagus/modules/networks/{primus,resenc_unet}.py` → 실제 구현은 `gardening_tools`(PyPI, 별도 레포) `modules/networks/{primus,resunet,unet}.py`. 로컬 재현으로 100% 확인(`gardening_tools/modules/networks/components/decoders.py:8-172`, `encoders.py:144-216`) | 그대로 사용. LoRA/hook 삽입 지점은 이미 named submodule로 노출됨 (C.2 참고) |
| **Pretrain LightningModule** | **신규 작성 (템플릿 재사용)** | 기본 경로 `SelfSupervisedModule`(`self_supervised.py:64-110`)은 단일 MSE reconstruction 전용, teacher 개념 없음. `DINOv2Module`(`dinov2.py`)이 유일한 frozen-teacher 선례 | `DINOv2Module` 구조를 템플릿 삼아 신규 `EpisodicDistillModule` 작성 |
| **Loss 구조** | **신규 작성 (템플릿 재사용)** | `self_supervised.py:58`는 `nn.MSELoss` 하드코딩; `modules/losses/ibot.py:68-88`이 patch-level distillation(centered softmax CE + masked-token 선택) 템플릿으로 적합 | decoder-stage feature loss 신규 구현 |
| **Checkpoint 저장/로드** | **재사용 가능** | 표준 Lightning `ModelCheckpoint`(`pipeline/run/pretrain.py:58-68`); 로드는 `BaseModule.load_state_dict`(`base_module.py:159-250`)가 `strict=False` 하드코딩, `"model."` prefix 요구, `load_decoder` 플래그로 디코더만 스킵 가능 | 우리 커스텀 pretrain 스크립트가 저장하는 state_dict 키에 `"model."` prefix만 붙이면 그대로 로드됨(1줄 shim) |
| **Finetune 파이프라인** | **재사용** | `finetune_seg/cls/reg.py` + `segmentation_module.py`/`clsreg_module.py` — 실제로 `asp_finetune_seg`까지 pytest로 통과 확인(`tests/test_finetune_seg.py`, Primus/ResEnc 포함) | 그대로 사용 |
| **Evaluation / metrics** | **수정 후 재사용** | EvalBox 수집 지표: Dice+volume_similarity(seg), Precision/Recall(cls), MAE/MSE(reg) (`pipeline/run/eval_box.py:207-236`). **NSD, test-time AUROC, Pearson/Spearman은 전체 repo grep 0건.** `scripts/FOMO26/`은 추론 전용 스크립트 6개뿐, 학습 config/지표 코드 없음 | FOMO26 5개 태스크에 필요한 NSD(seg), test-time AUROC(cls), Pearson/Spearman(reg) 신규 구현 필수 |

---

## C. 핵심 질문 4개에 대한 답

### C.1 `training_step()`에 frozen teacher forward를 삽입할 수 있는가? → **가능**

`LightningModule.__init__`은 일반 `nn.Module` 서브클래스이므로 `self.teachers = nn.ModuleList([...])`를 추가하는 데 프레임워크 제약이 없다. 이미 정확히 같은 패턴의 선례가 존재한다:

```python
# asparagus/modules/networks/dinov2.py:17-21
def freeze_eval_module(module: nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad = False
    module.eval()
```
```python
# asparagus/modules/networks/dinov2.py:144-147 (forward 안에서)
with torch.no_grad():
    teacher_cls_token, teacher_patch_tokens = self.forward_teacher(global_views)
```

옵티마이저에서 frozen teacher 파라미터를 제외하는 로직도 이미 구현되어 있다 (`modules/lightning_modules/dinov2.py:111-122`, `requires_grad`/이름 기반 필터링). 4~5개 teacher로 확장하는 것은 기계적 작업이다. 단, **기본 경로인 `SelfSupervisedModule`은 batch 계약이 `{"image","label","mask"}`뿐이라 너무 좁으므로, `SelfSupervisedModule`을 고치기보다 `DINOv2Module` 구조를 본뜬 새 LightningModule을 작성**하는 편이 안전하다.

### C.2 모델에서 decoder stage별 intermediate feature를 hook으로 뽑을 수 있는가? → **가능**

`gardening_tools` 소스를 직접 열람해 확인(로컬 pip 설치본, `/opt/conda/lib/python3.12/site-packages/gardening_tools/modules/networks/`):

```python
# gardening_tools/modules/networks/components/decoders.py — UNetDecoder
self.decoder_conv1 = self.basic_block(...)   # line 56
self.decoder_conv2 = self.basic_block(...)   # line 72
self.decoder_conv3 = self.basic_block(...)   # line 88
self.decoder_conv4 = self.basic_block(...)   # line 104

def forward(self, xs):                        # line 137
    x5 = self.decoder_conv1(...)
    x6 = self.decoder_conv2(...)
    x7 = self.decoder_conv3(...)
    x8 = self.decoder_conv4(...)
    ...
    return logits   # 기본은 최종 출력만 반환
```

`ResidualUNetDecoder`도 동일 패턴으로 `decoder_conv1~5` (`decoders.py:228-313`). Encoder 쪽은:

```python
# gardening_tools/modules/networks/components/encoders.py:210
self.stages = nn.Sequential(*stages)   # → model.encoder.stages[i] 로 인덱싱 가능
```

즉 **hook 대상 경로는 `model.decoder.decoder_conv1` ~ `decoder_conv4`(또는 ResEnc는 `_conv5`), `model.encoder.stages[i]`** 로 명확하다. 추가로 `deep_supervision=True`일 때는 decoder.forward가 이미 다중 해상도 리스트 `[ds4,ds3,ds2,ds1,ds0]`를 반환하는 코드 경로가 존재한다(`decoders.py:159-166`) — 단, 기본 UNet/ResEnc config는 `deep_supervision: False`이고, **Primus는 `configs/model/core/primus.yaml:39`에 `deep_supervision: False # not supported`로 명시적으로 비활성화**되어 있다. Primus의 decoder는 단일 `ClsRegHead`(`primus.py`)라 UNet/ResEnc처럼 자연스러운 multi-stage decoder가 아니므로, Primus에서 stage-feature distillation을 하려면 decoder가 아니라 **Eva transformer block 단위 hook**을 써야 한다(E 참고).

기존 non-hook API인 `forward_with_features()`(`unet.py:70-73`)는 `(output, skips[-1])`만 반환 — bottleneck 하나뿐이라 다단계 decoder distillation에는 부족하고, forward hook 등록이 필요하다.

### C.3 DataLoader가 modality/metadata를 배치에 함께 전달하는가? → **아니오**

```python
# asparagus/modules/datasets/PretrainDataset.py:21-31
def __getitem__(self, idx):
    file = self.files[idx]
    data = load_image_file(file)
    data_dict = {"file_path": file, "image": data, "transforms_applied": {}}
    data_dict = self._transform(data_dict)
    return data_dict
```

`file_path`(문자열) 외에는 modality/spacing/subject_id 필드가 전혀 없다. `PretrainDataModule.train_dataloader/val_dataloader`(`data_modules/pretraining.py:58-87`)는 custom `collate_fn` 없이 `default_collate`를 그대로 쓴다. episode 샘플링 시 teacher-confidence weighting을 위한 modality 정보는 **Dataset을 서브클래싱해 새로 추가**해야 한다.

### C.4 자체 pretrain checkpoint를 `asp_finetune_*`이 로드할 수 있는가? → **예, 조건부**

로딩 경로는 `torch.load(path, map_location="cpu", weights_only=False)` → `state_dict` 또는 `network_weights` 키 확인(`pipeline/auto_configuration/checkpoint.py:8-19`) → `BaseModule.load_state_dict(weights, strict=False)`(`base_module.py:159-250`, `strict=False`가 하드코딩됨, 220행).

**요구되는 최소 구조:**
1. `dict`이고 `"state_dict"` 또는 `"network_weights"` 키를 가짐
2. 그 안의 flat `{str: Tensor}` 딕셔너리 키는 반드시 **`"model."` prefix**로 시작 (target `_seg_net`/`_cls_net`의 named submodule과 동일한 dotted-name이어야 함) — `should_load_key`(`base_module.py:203-208`)가 이 prefix로 매칭·필터링하기 때문
3. `hyper_parameters`/`epoch`/`optimizer_states` 등은 finetune 로딩에는 불필요(로깅용)
4. shape 불일치·존재하지 않는 키는 자동으로 스킵되며, `load_decoder=False`면 `"model.decoder"`로 시작하는 키를 명시적으로 무시
5. 최소 1개 이상의 텐서가 실제로 로드되어야 함(`assert successful > 0`, 250행) — 안 그러면 즉시 예외

**결론: 우리가 커스텀 pretrain에서 저장하는 encoder-only state_dict에 `{"model." + k: v for k, v in encoder_sd.items()}`로 prefix만 붙여서 `checkpoint_path`로 넘기면 별도 리매핑 없이 로드된다.** 단, 텐서 이름이 target 아키텍처(`configs/model/core/*.yaml`이 가리키는 `_seg_net`/`_cls_net`)의 submodule 이름과 정확히 일치해야 한다.

---

## D. 권장 아키텍처

### 옵션 1: Asparagus fork + 새 entry point 추가 (권장)

```
asparagus/
  modules/
    lightning_modules/
      episodic_distill_module.py   [NEW, DINOv2Module 템플릿 기반]
    losses/
      distillation.py              [NEW, ibot.py 템플릿 기반 — decoder-stage feature loss]
    models/
      lora.py                      [NEW — post-construction Conv3d/Linear 치환 유틸]
      teacher_registry.py          [NEW — 4-5개 frozen teacher 로드/정규화 래퍼]
    datasets/
      episodic_pretrain_dataset.py [NEW — modality/subject_id 포함 Dataset]
    transforms/
      per_teacher_normalize.py     [NEW]
configs/
  model/core/episodic_distill.yaml [NEW]
  pretrain_multiteacher.yaml       [NEW]
```

- **예상 재사용 비율**: Hydra 설정 배선, 모델 팩토리(Primus/ResEnc), checkpoint 포맷, finetune/eval 전체 파이프라인 — 실측상 60~70% 재사용. 신규 작성은 LightningModule 1개, loss 1개, LoRA 유틸, teacher 레지스트리, metadata-aware Dataset/collate에 집중됨.
- **리스크**: `docs/advanced/hacking_asparagus.md`가 공식적으로 지원하는 확장 지점은 "다른 LightningModule/DataModule 클래스 경로를 config에서 교체" 정도로 얕음 — 새 entry point(`asp_pretrain_multiteacher`) 자체는 `pyproject.toml [project.scripts]`에 직접 추가하거나 그냥 `python my_pretrain.py`로 `@hydra.main(config_path=get_config_path(), config_name="pretrain_multiteacher")` 패턴을 따르면 됨(문서에 명시된 패턴, `docs/advanced/hacking_asparagus.md:16-27`). 대회 규격과의 불일치 리스크는 낮음 — finetune/eval을 원본 그대로 쓰기 때문.
- **유지보수성**: 좋음 — 업스트림 Asparagus 업데이트(예: 새 모델, 버그 수정)를 계속 받을 수 있음. 단 `gardening_tools`라는 두 번째 외부 의존성(별도 레포, 별도 버전 관리)에 묶임.

### 옵션 2: 자체 파이프라인 + Asparagus는 finetune/eval만

- **예상 재사용 비율**: crop/pad/spatial transform 클래스는 라이브러리로 import해서 재사용 가능(30~40%), 나머지(Hydra 설정, Dataset, DataModule, Trainer 루프)는 새로 작성.
- **리스크**: 체크포인트 포맷을 수작업으로 `"model."` prefix 규칙에 맞춰 저장해야 함(C.4에서 확인된 대로 어렵지 않음). Hydra 기반 실험 재현성(`hydra/config.yaml`)을 잃으므로 자체적으로 다시 만들어야 함. **대회가 pretrain 단계 자체의 Asparagus 호환/재현성을 요구한다면 규격 불일치 가능성 — 이 부분은 저장소만으로는 확인 불가, FOMO26 챌린지 룰을 별도 확인 필요.**
- **유지보수성**: 초반 속도는 빠르나 장기적으로 Asparagus와 별개로 두 파이프라인을 유지해야 함.

**권장: 옵션 1.** C.1~C.4에서 확인했듯 frozen teacher forward, decoder hook, checkpoint 호환은 모두 "가능"으로 판정되었고, finetune/eval을 원본 그대로 재사용할 수 있다는 점이 대회 규격 준수에 유리하다. 유일한 진짜 신규 작업은 LightningModule 1개 + loss 1개 + LoRA 유틸 + metadata Dataset이며, 이는 어느 옵션을 택하든 어차피 새로 짜야 하는 부분이다.

---

## E. 발견된 blocker / 주의사항

1. **전처리 시점 정규화가 디스크에 baked-in됨** (`asparagus_preprocessing/utils/resample.py:35-39`) — teacher별 다른 정규화(BraTS z-norm vs VesselFM percentile-01)를 지원하려면 raw intensity를 보존하는 별도 전처리 프리셋이 필요. 구조적으로 우리 설계와 정면 충돌하는 지점이므로 가장 먼저 해결해야 함.

2. **`gardening_tools`가 핵심 아키텍처의 실제 소유자** — `UNet`, `ResidualEncoderUNet`, `Primus`, 모든 conv/attention 블록(`components/blocks.py`, `eva.py` 등)이 별도 PyPI 패키지(`gardening_tools>=0.3.4`, 별도 GitHub 레포)에 있다. LoRA를 conv-block 생성 시점에 깊게 끼워 넣으려면(post-hoc 모듈 치환이 아니라) `gardening_tools`도 포크해야 한다.

3. **Primus는 deep supervision 미지원** (`configs/model/core/primus.yaml:39`, `# not supported`) — decoder가 단일 `ClsRegHead`라 UNet/ResEnc처럼 자연스러운 multi-stage decoder 구조가 아님. Primus 기반 teacher/student에서 "decoder stage별" distillation을 하려면 Eva transformer block 단위 hook으로 대체 설계 필요.

4. **Pretraining Dataset/DataLoader에 metadata 없음** — modality/subject_id/spacing 없이는 teacher-confidence weighting이 불가능. Dataset 서브클래싱 + custom collate_fn 필수(현재는 `default_collate`, 비균일 타입 배치 시 깨질 수 있음).

5. **EvalBox/FOMO26 스크립트에 필요한 지표 다수 누락** — NSD(surface distance), test-time AUROC, Pearson/Spearman correlation이 저장소 전체에서 grep 0건. FOMO26의 5개 태스크(Infarct cls=AUROC, Meningioma seg=Dice+NSD, Brain Age reg=MAE+corr, Trigeminal seg=Dice+NSD, Polymicrogyria cls=AUROC) 평가를 위해 신규 구현 필수.

6. **`asp_test_reg`가 `test_cls:main`을 가리킴** (`pyproject.toml`) — 업스트림 복붙 버그로 보임. regression test 스크립트가 필요하면 직접 고치거나 우회 필요.

7. **환경 의존성 문제 (직접 재현 확인)**:
   - 기본 `pip install -e .`만으로는 `modules/transforms/DinoV2.py`가 무조건 `import monai`(선택 의존성 취급되지만 실제로는 `SelfSupervisedModule`/`PretrainDataModule` import 체인에서 강제로 필요)하기 때문에 즉시 `ModuleNotFoundError`. `monai`, `timm`, `lightly`를 별도 설치해야 `pytest tests/` 전체가 통과함. `pyproject.toml`의 `optional-dependencies.extras`/`dependency-groups.dcai`로 분리되어 있지만 core import 경로에서 실질적으로 필수임 — 문서화되지 않은 암묵적 가정.
   - `ASPARAGUS_DATA`, `ASPARAGUS_MODELS`, `ASPARAGUS_CONFIGS` 세 환경변수가 없으면 CLI가 즉시 `ValueError`로 죽음(`asparagus/paths.py:15`) — README에 언급은 있으나 실행 전 필수 설정임을 재확인.
   - **`configs/hardware/cpu.yaml:5`가 `precision: "16-mixed"`로 설정되어 있어, CPU에서 `asp_pretrain`을 그대로 실행하면 `self_supervised.py`의 `assert not torch.isnan(loss)`가 실제로 발동해 즉시 죽는다.** 직접 재현 확인: `hardware.precision=32`로 오버라이드하면 정상 동작. CPU 스모크 테스트/디버깅 시 반드시 필요한 오버라이드이며 문서에 언급 없음.
   - **run 디렉토리 이름에 제외되지 않은 모든 CLI override가 그대로 baked-in됨** (`configs/core/base.yaml:9-30`의 `exclude_keys`에 없는 키, 예: `training.*`, `logger.*`)되어, override를 7~8개 이상 주면 `OSError: File name too long`(Linux 255자 제한)으로 실제로 실패함을 직접 재현. 하이퍼파라미터 스윕을 많이 돌릴 계획이면 `stem=` 파라미터로 짧은 별칭을 주는 습관이 필요.

8. **`Task` 개념은 클래스/레지스트리가 아니라 디렉토리 네이밍 컨벤션**(`<TYPE><3자리>_<Name>`) — 존재하지 않는 task를 지정해도 config 단계가 아니라 `dataset.json`/`split_*.json` 파일을 못 찾는 시점에야 `FileNotFoundError`로 실패함(직접 재현: `Task998_LauritSyn` 더미 데이터를 만들기 전엔 `/tmp/asp_data/Task998_LauritSyn/dataset.json` 없음 에러). 오타에 취약.

### 직접 실행 검증 로그 (요약)

- `pip install -e .` (별도 venv, base conda env는 numpy1/2 충돌로 실패 → 격리 필요) — 성공.
- `pytest tests/` (monai/timm/lightly 추가 설치, `CUDA_VISIBLE_DEVICES=""`, `ASPARAGUS_DATA/MODELS` 더미 설정 후) — **20개 테스트 전체 통과**, DINOv2 teacher-student pretrain, Primus/ResEnc finetune 포함.
- `asp_pretrain task=Task998_LauritSyn +model=unet_b_lw_dec data.train_split=split_75_15_10 +hardware=cpu ...` — synthetic `.pt` 파일 4개 + `dataset.json`/`split_75_15_10.json` 수동 생성 후, `hardware.precision=32` 오버라이드와 함께 **실제로 2 pseudo-epoch, 10 step까지 완주**(`Trainer.fit stopped: max_steps=10 reached.`), `hydra/config.yaml` 등 재현성 아티팩트 정상 생성 확인.

---

## 확인 불가 항목

- `gardening_tools`의 `ResidualUNetEncoder`/`ClsRegHead`/`Eva`의 완전한 내부 forward 로직(패키지가 로컬에 설치되어 소스를 읽었으나, 시간 관계상 attention block 단위까지는 미조사) — LoRA를 Eva 내부 `nn.Linear`(QKV projection)에 깊게 끼우려면 `gardening_tools/modules/networks/components/eva.py`/`transformer.py` 추가 조사 필요.
- FOMO26 챌린지 규정이 "pretrain 단계도 Asparagus 파이프라인이어야 한다"는 요구사항을 포함하는지 여부 — 저장소만으로는 확인 불가, 대회 공식 룰 확인 필요(옵션 2 채택 시 리스크에 직결).
- `asparagus_preprocessing`가 FOMO300K 원본 데이터에 대해 실제로 어떤 `normalization_operation`을 기본값으로 쓰는지(코드상 설정 가능하다는 것만 확인, 실제 FOMO300K 실행 시 사용된 정확한 프리셋 값은 `asparagus_preprocessing/configs/preprocessing_presets.py`의 `get_FOMO300K_saving_config`/`get_noresampling_preprocessing_config` 호출부까지 더 깊이 봐야 함).
