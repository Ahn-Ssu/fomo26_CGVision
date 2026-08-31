# Task 3 (Brain Age Regression) 작업 핸드오프 — MK-4 → mk8, 2026-08-13

이 문서는 `MK5_HANDOFF.md`(segmentation, MK-4→MK-5)와 같은 성격의 핸드오프 문서다.
차이점: 이번엔 별도 스펙 문서가 없다 — **이 문서 자체가 스펙 + 실측치**다. 대상 머신은
**mk8, RTX 5090(32GB) x2** (MK-4는 RTX 6000 Ada 49GB x4 — GPU 세대/개수가 다르므로
아래 §2의 환경 문제를 반드시 먼저 처리할 것).

**핵심 요약**: Task 1(Infarct classification)에서 검증한 head-architecture 축(§0)을
그대로 Task 3(Brain Age, single-modality regression)에 적용해 stratified 10-fold CV를
돌리는 게 목적. 전처리·split·코드는 이미 다 준비되어 `/root/fomo26_task3/`(이 문서가
있는 폴더) 안에 들어있다 — mk8에서 할 일은 **환경 세팅 + 정확한 경로에 파일 배치 +
VRAM 재실측 + orchestrator 실행**뿐이다.

---

## 0. 배경 — 무엇을, 왜

Task 1 head-architecture 실험(`/root/task1_head_arch/`, 별도 세션)에서 두 가지 head를
구현/검증함:
- **`head_mode=base`**: 기존 방식, 마지막 encoder stage + GAP + Linear.
- **`head_mode=multistage_gem`**: encoder stage 1~5(stage 0 제외)를 각각 독립적으로
  signed GeM pooling + LayerNorm 한 뒤 concat — 작년 FOMO25 1위 팀의 head 구조에서
  착안 (단, backbone full-finetune는 하지 않음, PEFT 유지).

Task 3는 **단일 모달리티(T1w)**이므로 Task 1의 `fusion_mode`(early/late) 축은 의미가
없다 — 항상 `fusion_mode=early`, `stem_mode=frozen`으로 고정 (아래 §3.2 참조). 즉
Task 3의 search space는 사용자가 표현한 그대로 **"single vs multi stage features"**
두 가지뿐이다: `head_mode ∈ {base, multistage_gem}` x 10 fold = **20 jobs**.

---

## 1. 파일 매니페스트 — 이미 이 폴더 안에 다 있음

```
fomo26_task3/
├── TASK3_HANDOFF.md                              (이 문서)
├── asparagus/                                     (fork, 로컬 변경 포함 -- 아래 git status 참조)
├── asparagus_preprocessing/                       (fork)
├── task1_head_arch/
│   ├── gem_pooling.py                             (signed GeM, MUST 배치 -- §3.1)
│   ├── multistage_head.py                         (MultiStageGeMHead, MUST 배치 -- §3.1)
│   └── measure_task3_vram.py                      (VRAM 재실측용, §2.3)
├── FOMO26/
│   ├── networks/student.py                        (StudentResEncUNet, Convpass3D 원본)
│   ├── data/normalize.py                          (asparagus_volume_wise_znorm, 전처리에 이미 사용됨)
│   ├── run_pretrain.py                             (참고용, 체크포인트 포맷)
│   └── expr/pretraining/ver5/checkpoints/step_280000.pt   (400MB, ver5 최종 체크포인트)
├── data/
│   └── REGR002_FOMO26_BrainAge_iso1mm/            (22GB -- 전처리 완료된 494명 .pt + split json, §4)
├── asparagus_env.sh
├── asparagus_orchestrate_common.py                (GPU orchestrator, 그대로 재사용)
├── asparagus_preprocess_task3_brainage.py         (전처리 스크립트, 이미 실행 완료 -- 참고/재실행용)
├── asparagus_build_stratified_split_task3.py      (split 빌더, 이미 실행 완료 -- 참고/재실행용)
├── asparagus_orchestrate_task3_head_arch.py       (mk8에서 실행할 20-job orchestrator)
└── asparagus_aggregate_task3_regression.py        (전부 끝난 후 실행할 집계 스크립트)
```

### 1.1 `asparagus` git 상태 (MK-4 시점, `git status --short`)

```
 M asparagus/modules/lightning_modules/__init__.py
 M configs/core/base.yaml
?? asparagus/modules/lightning_modules/fomo_classification_module.py
?? asparagus/modules/lightning_modules/fomo_regression_module.py     <- 이번에 새로 추가
?? asparagus/modules/networks/fomo26_student.py
?? configs/model/core/fomo26_student.yaml
?? configs/model/fomo26_student_clsreg.yaml
?? configs/model/fomo26_student_seg.yaml
```
`fomo_regression_module.py`(`FomoRegressionModule`)는 **이번 핸드오프 준비 중 새로 작성**한
파일 — `FomoClassificationModule`이 test-AUROC + best_epoch/best_val_loss를 기록하는 것과
동일한 패턴을 regression에 적용 (`MSE`/`MAE` + best_epoch/best_val_loss). 이미 `asparagus/`
폴더 안에 포함되어 있고 `__init__.py`에도 등록되어 있음 — 별도 작업 불필요, **이 체크리스트는
"파일이 실제로 존재하는지" 검증용**.

이 폴더는 `cp -r`로 통째로 복사된 것이라(git clone 아님) 커밋 안 된 변경사항도 전부
포함되어 있다. 그대로 `/root/asparagus`(또는 원하는 경로)에 두고 pip install -e 하면 됨.

### 1.2 가져올 필요 없는 것

- `asparagus_venv` — 재설치 권장 (아래 §2, **특히 mk8은 cu124를 그대로 쓰면 안 됨**).
- `/root/FOMO26` 전체(59GB, MK-4 기준) — 위 §1의 subset만 이미 추려서 넣어놨음.
- 원본 raw Task 3 데이터(`/root/data/FOMO2026_downstream/Task_3/`, T1w nifti 원본) —
  **전처리가 이미 끝난 `.pt` 파일만 가져가면 됨**, nifti 원본은 불필요 (재전처리하고
  싶은 경우가 아니면).

---

## 2. 환경 — mk8은 MK-4/MK-5와 다르다, 반드시 재확인

### 2.1 【최우선, 반드시 먼저 확인】 RTX 5090은 Blackwell (compute capability sm_120)

MK-4(이 문서를 작성한 머신)는 `torch==2.6.0+cu124`를 쓴다 (RTX 6000 Ada, compute
capability 8.9). **RTX 5090은 Blackwell 아키텍처(sm_120)이고, torch 2.6.0+cu124는
sm_120용 커널이 컴파일되어 있지 않다** — 그대로 쓰면 최악의 경우 "no kernel image is
available for execution on the device" 같은 에러로 즉시 죽거나, 최선의 경우에도 PTX
JIT 컴파일 오버헤드로 매 실행마다 느려질 수 있다. **레포 전체를 검색했지만 이 프로젝트
어디에도 Blackwell/sm_120 대응 코드나 메모가 없다** — mk8이 처음이다.

**mk8에서 반드시 할 것**: torch를 **2.7.0 이상 + cu128**(또는 그 이상, sm_120을
공식 지원하는 버전)으로 새로 설치. 정확한 버전은 mk8의 nvidia driver가 지원하는
CUDA 버전을 `nvidia-smi`로 먼저 확인한 뒤 맞출 것. `pip install -e /root/asparagus`가
`gardening_tools==0.3.5`, `lightning`, `monai` 등과의 버전 호환성을 깨지 않는지도
확인 필요 (MK-4 고정 버전은 §2.2 참조, 그대로 안 될 가능성 있음 — 특히 monai/torch
버전 조합).

```bash
nvidia-smi   # driver/CUDA 버전 확인
python3.12 -m venv asparagus_venv
source asparagus_venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu128   # 버전 명시 없이 최신 받고, 이후 compute capability 실측
python -c "import torch; print(torch.__version__, torch.cuda.get_device_capability(0))"  # sm_120 = (12, 0) 나오는지 확인
pip install -e /root/fomo26_task3/asparagus
pip install -e /root/fomo26_task3/asparagus_preprocessing
pip install gardening_tools==0.3.5   # 버전 충돌나면 이 고정을 풀고 최신으로
```

### 2.2 MK-4 참고 버전 (그대로 안 맞을 수 있음, 참고용)

```
torch 2.6.0+cu124, lightning 2.4.0, hydra-core 1.3.5, monai 1.6.0,
nibabel 5.4.2, simpleitk 2.5.6, numpy 2.5.1, scipy 1.18.0,
torchmetrics 1.9.0, torchvision 0.21.0+cu124
```

### 2.3 VRAM — 32GB 카드 기준 재실측 필수, MK-4 수치는 참고치일 뿐

MK-4(RTX 6000 Ada, 49GB)에서 실측한 Task 3 수치 (`task1_head_arch/measure_task3_vram.py`,
`TASK3_SHAPE=(176,256,256)`, bf16-mixed autocast, 실제 AdamW step):

| head | batch | peak reserved (MK-4, 49GB 카드) |
|---|---|---|
| base | 1 | 6.2GB |
| base | 2 | 12.7GB |
| multistage_gem | 1 | 8.6GB |

**이 숫자를 mk8에 그대로 외삽하지 말 것.** Blackwell의 bf16 텐서코어/메모리 allocator
동작이 Ada와 다를 수 있고, 애초에 다른 GPU에서 측정한 값이다. mk8에서 반드시 재실행:

```bash
cd /root/fomo26_task3/task1_head_arch
# BUDGETS_MB를 24GB/32GB 대신 32GB(mk8 실제 카드) 기준으로 확인
python measure_task3_vram.py
```
(`sys.path`가 `/root/asparagus`, `/root/task1_head_arch`를 가리키므로 §3.1의 경로
요구사항을 먼저 맞춰야 이 스크립트도 돌아간다.)

`asparagus_orchestrate_task3_head_arch.py`의 `training.batch_size=2`,
`jobs_per_gpu=2`는 MK-4 수치에서 보수적으로 추정한 **시작값일 뿐**이다 — 위 재실측
후 조정할 것. `hardware.num_workers`/동시성 상한도 CPU 코어 수·RAM에 종속적이므로
mk8에서 처음부터 다시 스윕해야 한다 (MK5_HANDOFF §4.4와 동일한 원칙 — MK-4의
"16 workers가 최적" 같은 숫자를 그대로 재사용하지 말 것, mk8은 GPU가 2장뿐이라
CPU 코어 배분도 다를 것).

---

## 3. 코드 — 이미 구현·검증됨, 배치만 정확히 할 것

### 3.1 【중요】 `/root/task1_head_arch/`는 정확히 이 절대경로에 있어야 함

`asparagus/asparagus/modules/networks/fomo26_student.py`가 다음처럼 **하드코딩된
절대경로**로 `gem_pooling.py`/`multistage_head.py`를 import한다:
```python
sys.path.insert(0, "/root/FOMO26")
sys.path.insert(0, "/root/task1_head_arch")
...
from multistage_head import MultiStageGeMHead
```
즉 **mk8에서도 `/root/task1_head_arch/gem_pooling.py`,
`/root/task1_head_arch/multistage_head.py`가 정확히 이 경로에 있어야 한다** — 다른
경로에 두면 import가 그냥 실패한다(상대경로/설정 가능한 경로가 아님). 이 폴더
(`fomo26_task3/task1_head_arch/`)의 두 파일을 `/root/task1_head_arch/`로 복사할 것.
(원래는 asparagus 패키지 내부로 옮기는 게 더 견고하지만, 소스 머신에서 현재 다른
실험이 이 경로에 의존해 돌고 있어서 이번엔 건드리지 않았다 — 향후 정리 과제.)

같은 이유로 **`/root/FOMO26/networks/student.py`, `/root/FOMO26/data/normalize.py`도
정확히 이 경로**에 있어야 한다 (`fomo26_student.py`가 `sys.path.insert(0,
"/root/FOMO26")` 후 `from networks.student import StudentResEncUNet`,
전처리 스크립트가 `from data.normalize import asparagus_volume_wise_znorm` 하드코딩).
`fomo26_task3/FOMO26/`을 `/root/FOMO26/`으로 복사(또는 심볼릭 링크)할 것.

### 3.2 head_mode / fusion_mode / stem_mode — Task 3는 2개 축만 유효

`FomoStudentClsRegNet.__init__`(`fomo26_student.py`)는 `head_mode`,
`fusion_mode`, `stem_mode` 3개 축을 받지만, Task 3는 **단일 모달리티**(T1w 1채널)라
사전학습된 stem의 native 채널 수(1)와 정확히 일치 — 즉:
- `fusion_mode="early"` 고정 (late fusion은 다중 모달리티 전용, 1채널이면 의미 없음).
- `stem_mode="frozen"` 고정. `"learnable"`을 쓰면 채널-repeat이 필요 없어
  `repeated_keys`가 비어서 `_load_and_configure`의 assert(
  "unfreeze_stem=True but no non-Convpass channel-repeated (stem) tensors were
  found")에 걸려 **에러가 난다** — 실수로 바꾸지 말 것. `"mixer"`도 1→1 항등
  변환이라 무의미(에러는 안 나지만 파라미터 낭비).
- **유효한 유일한 축은 `head_mode ∈ {"base", "multistage_gem"}`.**

`asparagus_orchestrate_task3_head_arch.py`가 이미 이 3개를 정확히 고정해서 job을
만든다 — 직접 CLI를 짤 때도 이 조합을 지킬 것.

### 3.3 `teacher_name=brats` — Task 1에서 가져온 기본값, Task 3용으로 검증된 건 아님

Task 1(뇌경색)은 brats teacher가 병변 관련이라 자연스러운 선택이었지만, Task 3(범용
뇌연령)는 특정 병변과 무관한 태스크다. `brats`를 기본값으로 남겨뒀지만 **다른
teacher(anatomix+brains, vesselfm, vjepa, voco)가 더 잘 맞을 수 있음** — GPU 시간이
남으면 teacher 축도 ablation 대상으로 고려할 것 (현재 20-job 계획에는 포함 안 됨).

### 3.4 학습 레시피 — Task 1 것을 그대로 가져옴, regression 수렴은 미검증

```
LR=1e-4, warmup=3 epoch, epochs=25, batch_size=2(§2.3에서 재조정), dropout_rate=0.0(기본값)
```
Task 1 classification에서 실측된 값을 그대로 시작점으로 썼다 — **regression(MSE
loss)의 수렴 속도/필요 epoch 수는 독립적으로 검증된 적 없음.** mk8에서 첫 1~2 fold
학습 시 `val/MSE`/`val/loss` 곡선이 25 epoch 안에 그럴듯하게 수렴하는지 반드시
확인하고, 필요하면 epoch 수나 warmup을 조정할 것.

---

## 4. 데이터 — 이미 전처리·split 완료, 배치 경로만 맞추면 됨

### 4.1 【중요】 데이터도 정확한 절대경로 필요

Split JSON(`split_stratified10.json`, `TEST_stratified10_fold{0..9}.json`) 안에
각 subject의 `.pt` 파일 절대경로가 **이미 하드코딩되어 저장되어 있다**
(`/root/asparagus_data/REGR002_FOMO26_BrainAge_iso1mm/preprocessed/sub-XXX/ses-01/t1w.pt`
형태). `asparagus_orchestrate_task3_head_arch.py`의 `DATA_PATH` 상수도 이 경로를
가리킨다. **`fomo26_task3/data/REGR002_FOMO26_BrainAge_iso1mm/`를 정확히
`/root/asparagus_data/REGR002_FOMO26_BrainAge_iso1mm/`로 복사할 것** (경로가 다르면
JSON 안의 절대경로가 다 어긋나서 파일을 못 찾는다). 다른 경로를 쓰고 싶으면
`asparagus_build_stratified_split_task3.py`를 그 경로로 재실행해서 split JSON을
다시 만들어야 한다 (전처리는 재실행할 필요 없음, `.pt` 파일 자체는 경로 무관).

### 4.2 전처리 내용 (이미 완료, 재실행 불필요 — 참고용)

- 원본: `/root/data/FOMO2026_downstream/Task_3/preprocessed/sub-XXX/ses-01/t1w.nii.gz`
  (494명, 전원 native shape (176,256,256) @ (1,1,1)mm — 측정 결과 **전 subject 분산 0**,
  이미 표준 템플릿 그리드에 정합되어 있음 — resample/crop/pad 불필요).
- `asparagus_volume_wise_znorm`(FOMO26/data/normalize.py, verbatim reuse)으로
  foreground-masked z-norm — **REGR002_FOMO26_BrainAge.py(asparagus_preprocessing 기본
  제공 스크립트)의 `no_norm` 프리셋을 쓰지 않고 이걸 쓴 이유**: 사전학습된 backbone이
  이 정규화 분포로 학습됐기 때문 — Task 1에서 정규화 불일치 때문에 "신호 없음"으로
  잘못 결론났던 전례를 반복하지 않기 위함(프로젝트 메모리 참조).
  (참고: `REGR002_FOMO26_BrainAge.py` 자체에도 `subdir="Task_3/Task_3"` 기본값 버그가
  있음 — 실제 경로는 `Task_3/`뿐, 이 프로젝트에서는 그 스크립트를 안 쓰고
  `asparagus_preprocess_task3_brainage.py`를 새로 작성해서 우회했으므로 영향 없음.)
- 저장 형식: subject당 `.pt` 파일 하나, `torch.save([image_tensor, label_tensor], ...)`
  — `image_tensor` shape `[1,176,256,256]` (float32, [0,1] 범위로 rescale됨),
  `label_tensor` shape `[1]` (나이, float32). `ClsRegDataset.__getitem__`이 기대하는
  포맷과 정확히 일치 (`data[0]`=image, `data[1]`=label).
- `dataset.json`: `{"metadata": {"n_classes": 1, "n_modalities": 1}}` — 이게 있어야
  `finetune_reg.py`가 `input_channels`/`output_channels`를 올바르게 잡는다.

### 4.3 Split 통계 (age-quantile 10-fold stratification, 이미 검증됨)

494명 전원, 10 fold, fold당 46~53명. 나이 분포: 전체 평균 45.2세(std 17.4), fold별
평균 44.4~45.7세(std 17.1~17.9) — 폴드 간 나이 분포 거의 동일하게 잘 맞춰짐.
Split 무결성(전원 커버, 중복 없음, train/val에 test subject 누출 없음, fold 간
중복 없음) assert 전부 통과 확인됨 (`asparagus_build_stratified_split_task3.py`
실행 로그 참고).

---

## 5. 이미 검증된 것 vs 아직 검증 안 된 것 (혼동 방지용 명시적 구분)

**검증됨 (MK-4에서 직접 실행, 2026-08-13)**:
- 전처리 494명 전원 성공 (실패 0건), 예상 shape 그대로.
- Split 무결성 100% 통과, 나이 분포 균형 확인.
- Hydra config dry-run(`--cfg job`)으로 `head_mode`/`fusion_mode`/`stem_mode`/
  `lightning_module=FomoRegressionModule` 배선 정상 확인.
- **실제 GPU에서 2-epoch smoke test 완주** (`head_mode=multistage_gem`, fold0,
  batch=2, RTX 6000 Ada): NaN/크래시 없음, `.pt` 데이터 로딩 정상, checkpoint
  저장/best-checkpoint 복원 정상, `FomoRegressionModule`이 predictions JSON을
  `{prediction, label, best_epoch, best_val_loss}` 스키마로 정확히 저장 —
  `asparagus_aggregate_task3_regression.py`의 glob 패턴을 이 실제 출력 경로 문자열로
  직접 검증 완료.

**검증 안 됨 (mk8에서 확인 필요)**:
- Blackwell(sm_120) 호환성 — 전혀 테스트 안 됨, §2.1 최우선 처리.
- 32GB 카드에서의 실제 VRAM 여유 — §2.3, 재실측 필수.
- 20-job 전체 sweep의 실제 학습 수렴/최종 MAE — smoke test는 2 epoch뿐이라 유의미한
  성능 숫자 아님 (예: MAE 24.6세는 warmup 단계라 사실상 평균 예측 수준).
- `teacher_name=brats` 선택이 Task 3에 최선인지.
- `asparagus_aggregate_task3_regression.py`의 glob 패턴 — smoke test(`root=task3_smoketest`)
  기준으로 검증했지만 실제 sweep은 `root=task3_head_arch`로 도는 것만 다름(패턴
  자체는 root 값과 무관하게 정확함) — 그래도 mk8의 hydra/lightning 버전이 달라
  override_dirname 포맷팅이 바뀌면 안 맞을 수 있음, 그 경우를 대비해 loose-glob
  fallback + 경고 메시지를 이미 넣어뒀다.

---

## 6. mk8에서 즉시 할 일 (순서대로)

1. `nvidia-smi`로 driver 확인 → torch/cu128(or 최신) 새로 설치 (§2.1).
2. `/root/fomo26_task3/`를 mk8로 전송 (22GB, scp/rsync).
3. 정확한 절대경로에 배치:
   - `fomo26_task3/task1_head_arch/*` → `/root/task1_head_arch/`
   - `fomo26_task3/FOMO26/*` → `/root/FOMO26/` (또는 그 subset만)
   - `fomo26_task3/data/REGR002_FOMO26_BrainAge_iso1mm` → `/root/asparagus_data/REGR002_FOMO26_BrainAge_iso1mm`
   - `fomo26_task3/asparagus`, `fomo26_task3/asparagus_preprocessing` → 원하는 경로(pip install -e)
   - 나머지 루트 스크립트들 → `/root/` (orchestrator가 `/root/asparagus_orchestrate_common.py`를 import하고, `source /root/asparagus_env.sh`를 실행하므로 이 경로여야 함, 또는 스크립트 내 경로 상수를 조정)
4. `asp_finetune_reg --cfg job ...`으로 1개 config dry-run (본 문서 §3의 CLI 패턴,
   `asparagus_orchestrate_task3_head_arch.py`의 `build_cmd`가 실제 예시).
5. `measure_task3_vram.py`로 32GB 카드 기준 VRAM 재실측 → `batch_size`/`jobs_per_gpu`
   확정 (§2.3).
6. 1 fold, 2~3 epoch짜리 실제 GPU smoke test 한 번 더 (mk8/Blackwell 환경 자체
   검증용 — MK-4에서 이미 통과했다고 mk8에서도 자동으로 통과한다고 가정하지 말 것).
7. `asparagus_orchestrate_task3_head_arch.py` 실행 (20 jobs).
8. 전부 끝나면 `asparagus_aggregate_task3_regression.py` 실행 → pooled MAE/RMSE +
   base vs multistage_gem paired ΔMAE 95% CI 확인.
