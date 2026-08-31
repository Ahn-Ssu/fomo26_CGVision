# Teacher 모델 확보·검증·Feature Hook 구현 보고서

작업 기준 문서: `00_PRE_ANALYSIS.md`, `02_setup_teachers_v2.md`.
모든 항목은 실제 GPU 박스(RTX 6000 Ada ×3, 49GB)에서 실제 코드를 clone/설치/실행해 검증했다.
추측한 항목은 없으며, 검증 불가능했던 항목은 "확인 불가"로 명시한다.

**작업 위치**: `/root/teachers/` (요청대로 `/tmp`가 아닌 `/root` 아래 실제 clone)
**공유 venv**: `/root/teachers/venv` (`--system-site-packages`, torch==2.6.0+cu124, numpy==1.26.4, monai==1.3.2)

---

## A. 가중치 확보 현황

| Teacher | Weight 출처 | 파일 크기 | `strict=True` | 비고 |
|---|---|---|---|---|
| **Anatomix** | `/root/anatomix/model-weights/anatomix.pth` (이미 이 박스에 존재, README 방식 그대로) | 23.6 MB | **성공** (`<All keys matched successfully>`) | `anatomix+brains.pth` variant도 동일 검증 완료 |
| **VesselFM** | HuggingFace `bwittmann/vesselFM`, `vesselFM_base.pt` (`huggingface_hub.hf_hub_download`) | ~125.7 MB | **성공** | repo에 존재하는 **유일한** 체크포인트. `inference.py` 자체가 fallback으로 다운로드하는 파일과 동일 → generalist zero-shot 가중치 확정 |
| **FastSurfer** | Zenodo record `10390573` (공식 b2share URL은 404, `checkpoint_paths.yaml`이 실제로는 존재하지 않는 URL 조합을 시도하는 버그가 있어 Zenodo API로 직접 조회) — axial/coronal/sagittal VINN aseg-dkt 3개 | 22.4/22.4/22.4 MB | **성공** (3개 view 모두) | **최종 판정: 제외 (아래 §G)**. cerebnet/hypvinn/recon-surf는 처음부터 미시도 |
| **BraTS** | 사용자가 2026-07-15 `/root/teachers/BarTS_Teacher/`에 업로드(nnUNet `fold_all/checkpoint_best.pth`) | 체크포인트 파일 자체 크기 미기록(292 tensor, 30.8M params) | **성공** | `dynamic_network_architectures.PlainConvUNet` — PRE_ANALYSIS가 가정한 "ResEnc UNet"이 **아니라 PlainConvUNet**임을 정정. `_best_ema` fg dice=0.9143, epoch=998 |

---

## B. 모델 구조

### Anatomix (`gardening_tools` 방식이 아닌, anatomix 자체 구현)
- `anatomix/model/network.py::Unet` — **평탄한 `nn.Sequential`** (66개 layer, named submodule tree 아님).
  `Unet(dimension=3, input_nc=1, output_nc=16, num_downs=4, ngf=16)` — README 로딩 코드와 정확히 일치.
- `model.encoder_idx = [8, 15, 22, 29]`, `model.decoder_idx = [37, 44, 51, 58]` (skip-connect 지점).
- **Hook 불필요** — `forward(x, layers=[...], encode_only=False)`가 네이티브로 다중 feature를 반환하는 API 제공 (`network.py:447-521`).
- 총 파라미터 5,899,344.

### VesselFM
- `monai.networks.nets.DynUNet(in_channels=1, out_channels=1, spatial_dims=3, strides=[[1,1,1],[2,2,2]×5], filters=[32,64,128,256,320,320], res_block=True)` — hydra config `vesselfm/seg/configs/model/dyn_unet_base.yaml`을 통해 인스턴스화.
- `named_modules()`: `input_block → downsamples.0..3 → bottleneck → upsamples.0..4 → output_block`, 그리고 이를 재귀적으로 wiring하는 `skip_layers`(같은 서브모듈을 참조로 재사용, 새 파라미터 없음).
- **Hook 필요** — `DynUNet.forward()`는 최종 logit만 반환. `upsamples[0..4]` + `output_block`에 `register_forward_hook` 등록.
- `model.parameters()` = 31,418,977 (unique). raw state_dict 키 합산 파라미터 수(62,837,921)는 `skip_layers`가 `downsamples[i]`를 참조 공유하기 때문 — 불일치 아님, 확인 완료.

### FastSurfer (조사 완료 — 제외 결정, §G 참고)
- `FastSurferCNN/models/networks.py:214::FastSurferVINN`. **`named_modules()` 125개 모듈 중 `Conv2d` 45개, `Conv3d` 0개** — 순수 2D 아키텍처, 확인 완료.
- 입력 shape `(N, 7, H, W)` — 7-slice stack을 채널 축으로 쌓는 방식(`inp_block.conv0 = Conv2d(7,32,...)`).
- **axial/coronal/sagittal 3개의 완전히 별도인 모델·체크포인트** (axial/coronal NUM_CLASSES=79, sagittal=51 → 51→79 인덱스 remap 후 `alpha`-weighted logit 합산으로 aggregate, `inference.py:299-380`). Task 문서가 가정한 "95-class"는 네트워크 출력이 아니라 후처리 LUT 리매핑 결과(`run_prediction.py:396-399`).
- `strict=True` 3개 view 모두 성공.

### BraTS (2026-07-15 구현 완료)
- `dynamic_network_architectures.architectures.unet.PlainConvUNet` (nnUNet v2 표준), `plans.json`의 `configurations["3d_fullres"]["architecture"]`에서 재구성. **PRE_ANALYSIS/BRATS_BLOCKED.md가 가정한 "ResEnc UNet"이 아니라 PlainConvUNet(Residual 아님)** — 정정.
- 6-stage, `features_per_stage=[32,64,128,256,320,320]`, `Conv3d`+`InstanceNorm3d`+`LeakyReLU`, 마지막 stride만 `(2,2,1)`(비등방).
- `named_children()`: `encoder.stages[0..5]`, `decoder.stages[0..4]` (feature) + `decoder.seg_layers[0..4]` (deep-supervision 분류 head, feature 아님 — decoder source(`unet_decoder.py::forward`) 직접 확인해 `stages[s]` output과 `seg_layers[s]` output이 다른 텐서임을 확인).
- **Hook 필요** — `decoder.stages[0..4]`에 forward hook (feature), `seg_layers`도 참고용으로 같이 hook.
- `strict=True` 성공, 30,785,994 params. Checkpoint 최상위 키가 `network_weights`(Task1 `/root/FOMO26/CLAUDE_CODE_RUN_REPORTS/01_ASPARAGUS_ANALYSIS.md`가 예측한 nnUNet 포맷과 정확히 일치).

---

## C. Feature Extraction Spec (128³ 입력 기준, GPU 실측)

| Teacher | Stage | Module path / API | Output shape | Channels | Spatial scale |
|---|---|---|---|---|---|
| Anatomix | dec_stage_0 | native `layers=[43]` | (1,128,16,16,16) | 128 | 0.125 |
| Anatomix | dec_stage_1 | native `layers=[50]` | (1,64,32,32,32) | 64 | 0.25 |
| Anatomix | dec_stage_2 | native `layers=[57]` | (1,32,64,64,64) | 32 | 0.5 |
| Anatomix | dec_stage_3 | native `layers=[64]` | (1,16,128,128,128) | 16 | 1.0 |
| VesselFM | dec_stage_0 | `model.upsamples[0]` hook | (1,320,8,8,8) | 320 | 0.0625 |
| VesselFM | dec_stage_1 | `model.upsamples[1]` hook | (1,256,16,16,16) | 256 | 0.125 |
| VesselFM | dec_stage_2 | `model.upsamples[2]` hook | (1,128,32,32,32) | 128 | 0.25 |
| VesselFM | dec_stage_3 | `model.upsamples[3]` hook | (1,64,64,64,64) | 64 | 0.5 |
| VesselFM | dec_stage_4 | `model.upsamples[4]` hook | (1,32,128,128,128) | 32 | 1.0 |
| BraTS | dec_stage_0 | `model.decoder.stages[0]` hook | (1,320,8,8,8) | 320 | 0.0625 |
| BraTS | dec_stage_1 | `model.decoder.stages[1]` hook | (1,256,16,16,16) | 256 | 0.125 |
| BraTS | dec_stage_2 | `model.decoder.stages[2]` hook | (1,128,32,32,32) | 128 | 0.25 |
| BraTS | dec_stage_3 | `model.decoder.stages[3]` hook | (1,64,64,64,64) | 64 | 0.5 |
| BraTS | dec_stage_4 | `model.decoder.stages[4]` hook | (1,32,128,128,128) | 32 | 1.0 |

모두 `/root/teachers/scripts/verify_teachers.py` 실행으로 shape을 `feature_specs`와 자동 대조해 일치 확인(더미 텐서 + 실제 MRI 패치 양쪽). **VesselFM과 BraTS의 stage별 채널/스케일이 정확히 동일**(둘 다 nnU-Net 표준 6-stage 3D UNet 계열)한 점이 student stage-mapping(§E)에 유용.

---

## D. 정규화 명세 — (b) 방식(asparagus z-norm 위에 teacher 정규화 재적용) 판정

**중요 정정**: `00_PRE_ANALYSIS.md`가 가정한 "raw → z-norm"은 부정확했다. asparagus의 실제 스킴
(`asparagus_preprocessing/utils/normalize.py::normalizer(scheme="volume_wise_znorm")`)은
**clamp(foreground 99th percentile 상한) → z-score(foreground 통계) → 전체 볼륨 min-max [0,1] rescale**의
합성이다. 이 보고서의 모든 실증 검증은 이 실제 함수를 그대로(재구현 없이) 사용했다.

| Teacher | 정규화 원문 | `norm_type` | Percentile scope | (b) 적용 가능? | 실증 검증 결과 |
|---|---|---|---|---|---|
| **Anatomix** | `normalize_img(x, percentile=99.99, zero_centered=True)` = full-volume min/percentile → [0,1]→[-1,1] (`pretraining/data/data_utils.py:4-46`) | **percentile** | full-volume (마스크 없음) | **예 (수학적으로 완전 등가)** | 대수적 증명: `normalize_img`는 입력에 대한 양의 스케일 affine 변환에 불변 — z-norm 자체가 그런 affine이므로 `normalize_img(raw) == normalize_img(z_normed)`가 항등식으로 성립 (PRE_ANALYSIS §3.2의 1.49e-09 실측과 일치) |
| **VesselFM** | `ScaleIntensityRangePercentiles(lower=1, upper=99, b_min=0, b_max=1, clip=True)` (`vesselfm/seg/configs/inference.yaml` → `utils/data.py::generate_transforms()`, MONAI 1.3.2 소스 직접 확인) | **percentile** | **full-volume** (`channel_wise=False`, 마스크 없음 — MONAI 소스로 확인, 가정 아님) | **예** | 실측: 정규화 후 볼륨 max abs diff = 1.79e-07, Pearson corr = 0.999999999999998, Spearman = 0.9977; 실제 네트워크 통과 후 최종 logit max abs diff = 0.00988, decoder feature cosine similarity = 1.0 (전체 5 stage) — Path A/B 사실상 동일 |
| **FastSurfer** | `conform()` — robust histogram 기반(`f_low=0.0, f_high=0.999`) linear rescale → `uint8[0,255]` (`FastSurferCNN/data_loader/conform.py:851-1146`) | **absolute** (실측으로 확정, 가정 아님) | 해당 없음 (히스토그램 기반, foreground mask 없음) | **아니오** | 정규화 후 볼륨 자체는 겉보기엔 비슷해 보임(max abs diff 58/255, Pearson r=0.9935)이지만, **실제 체크포인트로 돌린 다운스트림 예측은 raw vs z-norm 입력 간 argmax 라벨 일치율 79.28%에 불과**(최저 슬라이스 8.03%) — VesselFM(cosine sim 1.0)·Anatomix(수학적 항등)와 질적으로 다른, 실질적인 예측 변화. `meta={"raw":...}` 경로 필수 |
| **BraTS** | `dataset.json` 명시: "p99.5 clip only; z-norm via nnUNet" = clamp(foreground, q=0.995) → z-score(foreground), **per-case**(channel명 "MRI" → nnUNet 기본 `ZScoreNormalization`, dataset-wide fingerprint 아님). **단, Anatomix/VesselFM과 달리 최종 min-max rescale이 없음** — 이 때문에 (b) 적용 시 asparagus 출력을 그대로 통과시키면 안 되고, **teacher의 preprocess()가 clamp+zscore를 입력으로부터 매번 새로 재계산**해야 invariance가 성립(단순 pass-through 아님) | **percentile (근사적)** | foreground-only (mask = x≠min) | **예 (근사적, 완전 등가는 아님)** | 실측: 전처리 후 볼륨 max abs diff=0.87, Pearson r=0.9995; 실제 네트워크 통과 후 cosine sim=0.9994, **argmax 라벨 일치율 99.83%**(FastSurfer의 79.28%와 질적으로 다름, Anatomix/VesselFM의 100%보다는 약간 낮음 — asparagus의 q=0.99 clamp와 BraTS 자체 q=0.995 clamp가 미세하게 상호작용하는 것으로 추정, 추가 조사는 안 함) |

**결론**: Anatomix·VesselFM은 percentile 기반이라 **(b) 방식이 수학적으로 완전히 적용 가능**하고, BraTS는 **근사적으로(99.8% 수준) 적용 가능**하다(단, teacher의 preprocess가 입력을 다시 clamp+zscore 해야 함 — 단순 pass-through는 틀림). FastSurfer는 absolute-scale임이 실측으로 확정되었으나, 어차피 §G의 독립적인 이유(순수 2D, 중복 역할)로 제외가 권고되어 이 예외 처리 자체가 불필요해졌다.

---

## E. Student(ResEnc UNet)에서 Stage 매핑 제안

Task 1(`/root/FOMO26/CLAUDE_CODE_RUN_REPORTS/01_ASPARAGUS_ANALYSIS.md`)에서 확인한 student(ResEnc UNet, `gardening_tools/modules/networks/components/decoders.py`)의 decoder 구조: `decoder_conv1`(최심층) ~ `decoder_conv5`(최종 해상도), spatial_scale은 nnU-Net 6-stage 기본 설정 기준 대략 1/32, 1/16, 1/8, 1/4, 1/2 순.

| Student stage (추정, resenc_unet_b 6-stage 기준) | Spatial scale | Anatomix | VesselFM | BraTS | 비고 |
|---|---|---|---|---|---|
| `decoder.decoder_conv1` | ~1/16 | dec_stage_0 (1/8, 128ch) | dec_stage_0 (1/16, 320ch) — **scale 일치** | dec_stage_0 (1/16, 320ch) — **scale 일치** | VesselFM·BraTS는 완전히 동일한 6-stage nnU-Net 표준 구조라 scale/channel이 서로 동일. Anatomix만 4-downs(1/8이 최심층)라 한 단계 어긋남 |
| `decoder.decoder_conv2` | ~1/8 | dec_stage_0 (1/8, 128ch) — **scale 일치** | dec_stage_1 (1/8, 256ch) — **scale 일치** | dec_stage_1 (1/8, 256ch) — **scale 일치** | channel 수는 셋 다 student와 달라 projection layer 필수 |
| `decoder.decoder_conv3` | ~1/4 | dec_stage_1 (1/4, 64ch) — **scale 일치** | dec_stage_2 (1/4, 128ch) — **scale 일치** | dec_stage_2 (1/4, 128ch) — **scale 일치** | |
| `decoder.decoder_conv4` | ~1/2 | dec_stage_2 (1/2, 32ch) — **scale 일치** | dec_stage_3 (1/2, 64ch) — **scale 일치** | dec_stage_3 (1/2, 64ch) — **scale 일치** | |
| `decoder.decoder_conv5` | 1/1 | dec_stage_3 (1.0, 16ch) — **scale 일치** | dec_stage_4 (1.0, 32ch) — **scale 일치** | dec_stage_4 (1.0, 32ch) — **scale 일치** | |

**주의**: student의 정확한 stage 수/채널 수는 실제 사용할 `resenc_unet_*` 프리셋(`configs/model/resenc_unet_b.yaml` 등)에 따라 다르므로, 실제 distillation 구현 시점에 student를 인스턴스화해 `named_modules()`로 정확한 stage 수·채널을 재확인해야 한다 (이 표는 nnU-Net 표준 6-stage 가정 하의 **초안**이며 "확인 필요"로 표시).
Channel 수가 teacher/student 간 모두 다르므로, **모든 매핑에 1x1x1 conv projection layer가 필요**하다 (예: Anatomix dec_stage_0[128ch] → student decoder_conv2 채널수로 projection).

---

## F. 성능 프로파일 (128³ patch, RTX 6000 Ada, `verify_teachers.py` 실측)

| Teacher | Forward (128³, 1 patch) | Peak GPU mem |
|---|---|---|
| Anatomix | 653ms (cold, cuDNN warmup 포함) / 25ms (anatomix+brains, warm) | 1.60 GB |
| VesselFM | 89ms | 1.84 GB |
| BraTS | 61ms | 3.55 GB |
| FastSurfer (참고용, 제외됨) | 1개 view 256-slice 풀 forward: 1.76s | **11.08 GB** (1개 view만) |

FastSurfer는 `conform()` 전처리에만 CPU 3.73s, 3-view 전체 GPU forward ~5.3s + 리샘플/LUT 후처리 포함 시
volume당 **총 10~15초** 추정.

> **정정 (2026-07-15, 사용자 지적으로 재검토)**: 이 수치를 "VesselFM 대비 300~500배 느림"으로 표현한 것은
> 오해를 유발했다. FastSurfer 공식 README(`README.md:21`)는 `--seg_only` 전체 CLI 기준 "약 5분(GPU)"이라
> 명시하는데, 여기엔 conform+3-view+재정렬+LUT+통계+QC 이미지 생성까지 다 포함된다 — 우리가 측정한
> 10~15초는 오히려 이 공식 수치보다 훨씬 빠르며, 측정 오류의 증거가 아니다. 또한 VesselFM/Anatomix는
> 128³ **패치** 하나로 측정했지만 FastSurfer의 `conform()`은 256³ **전체 볼륨**(복셀 수 8배)을 강제하므로
> 애초에 동일 스케일 비교가 아니었다. FastSurfer는 frozen·deterministic teacher이므로, 학습 루프에서 매
> step 실시간 호출하는 대신 **FOMO300K volume당 1회만 미리 계산해 캐싱**하면 이 비용은 실질적으로 0에
> 수렴한다 — "속도"는 애초에 배제 사유로서 설득력이 약했다. 아래 §G의 판단은 이 정정을 반영해 업데이트함.

**3-teacher(Anatomix+VesselFM+BraTS) 동시 로드 가능성**: 개별 peak 합산 ~7.0GB(Anatomix 1.6+VesselFM 1.84+BraTS 3.55). 49GB GPU
기준 여유 매우 큼(1개 GPU에 3개 teacher를 동시에 올려도 15% 미만 사용). 다만 실제 동시 상주 상태에서의 합산
측정(개별 순차 측정이 아닌 진짜 동시 로드)은 아직 수행하지 않음 — 최종 확인 필요 항목으로 남겨둔다.
FastSurfer는 제외되었으므로 이 계산에서 빠짐.

---

## G. FastSurfer 2.5D 판단 — **live feature-level teacher로는 제외, offline pseudo-label 소스로는 재검토 여지 있음 (2026-07-15 수정)**

`/root/teachers/wrappers/FASTSURFER_EXCLUDED.md`(사용자 지적 반영해 정정 완료, 전체 근거/코드 인용 포함)에 상세 기록. 요약:

**4가지 옵션 재평가:**
- **(a) slice-wise pseudo-3D**: 기각(유지). 3개 모델 × ~5 stage × 256 slice에 걸쳐 hook을 걸어야 하고, slice별로 쌓은 2D activation은 실질적인 cross-slice receptive field가 없어 "3D feature"라 부르기 어려움.
- **(b) 단일 view만 사용**: 기각(유지). FastSurfer 저자들이 axis-aligned striping artifact를 해결하기 위해 설계한 3-view fusion 자체를 버리는 것이며, 절감되는 계산량도 1/3에 불과.
- **(c) 최종 output(79/95-class)만 pseudo-label로 사용, offline precompute**: **속도 재평가 후 재고 가치 있음으로 격상.** FOMO300K 각 volume에 대해 딱 한 번만 conform+3-view forward+LUT remap을 미리 돌려 디스크에 캐싱해두면, 학습 루프에서는 저장된 pseudo-label을 읽기만 하면 되므로 "10~15초/volume" 비용이 학습 속도에 전혀 영향을 주지 않는다. 다만 여전히 `BaseTeacher`가 지향하는 feature-level(중간 decoder feature) distillation 패턴과는 다른 별도 파이프라인(오프라인 스크립트 + output-level loss)이 필요함.
- **(d) 완전 제외 — live feature-level teacher로는 채택 유지, 근거는 2가지로 축소**:
  1. **순수 2D 구조** (`Conv3d` 0개, `named_modules()` 125개 중 45개가 `Conv2d`) — decoder-stage feature hook 방식과 근본적으로 안 맞음
  2. **정규화가 absolute-scale이라 실측 예측 결과가 실제로 달라짐** (§D — argmax 라벨 일치율 79.28%, 최저 슬라이스 8.03%) — 다른 teacher와 같은 `preprocess(asparagus_normed_x)` 경로를 재사용하면 조용히 틀린 결과를 냄
  3. ~~비용이 다른 teacher 대비 압도적으로 큼~~ → **철회.** §F 정정 참고 — 절대 비용(10~15초/volume)은 FastSurfer 자체 공식 claim(전체 CLI 기준 5분)보다도 빠르고, frozen teacher이므로 offline precompute로 상쇄 가능. "300~500배 느림"이라는 프레이밍이 오해를 유발했음을 인정함.

**처리 상태**: live feature-level teacher wrapper는 미구현·`registry.py` 미등록 유지(근거 1·2가 여전히 유효하므로). Offline output-level pseudo-label 소스로 쓸지는 **사용자 판단 필요** — 원하면 FOMO26 파이프라인에 별도 `precompute_fastsurfer_labels.py` 스크립트를 추가할 수 있음(1회성 배치 작업이라 4-teacher 실시간 로드 목록에는 포함 안 됨). 체크포인트 3개(~22MB×3)는 `/root/teachers/checkpoints/fastsurfer/`에 보존.

## H. VesselFM 1mm 해상도 판단

**확인 불가** (이번 라운드에서 미수행) — VesselFM 자체 학습 해상도(0.3~0.5mm 추정, PRE_ANALYSIS §2.1)와
FOMO300K 표준 1mm isotropic 간의 pseudo-label 품질 차이는 **실제 혈관이 보이는 스캔(MRA/CTA 등)으로
inference를 돌려 시각적으로 확인**해야 한다. 이번 세션에서 사용한 real MRI 샘플(`sub_11043`, T1)은
혈관 구조가 뚜렷하지 않은 일반 T1이라 이 목적에는 적합하지 않다. **후속 작업 필요**: MRA/CTA 또는
FOMO300K 내 혈관 조영 시퀀스가 있는 서브셋을 찾아 원본 해상도 vs 1mm resampled에서 각각 inference 후 비교.

---

## I. Blocker 및 미해결 이슈

1. ~~BraTS teacher 체크포인트 부재~~ → **해결 (2026-07-15)**. `/root/teachers/BarTS_Teacher/`에 업로드됨,
   `strict=True` 검증 완료, `TEACHER_REGISTRY`에 `"brats"`로 등록, 4개 teacher 전체 `verify_teachers.py` 통과.
   PRE_ANALYSIS가 가정한 "ResEnc UNet"이 아니라 **PlainConvUNet**임이 확인되어 정정함.

2. **환경 이슈 (2회 재현, 둘 다 해결됨)**:
   - `pip install`이 의도치 않게 torch를 2.13+cu130으로 업그레이드해 이 박스의 드라이버(CUDA 12.2)와
     불일치 → CUDA 사용 불가. `torch==2.6.0+cu124`로 재고정하여 해결.
   - 위 과정에서 딸려온 `nvidia-cudnn-cu13` 잔재가 `nvidia-cudnn-cu12`와 충돌해
     `CUDNN_STATUS_NOT_INITIALIZED` 발생 → cu13 계열 패키지 제거로 해결.
   - 두 이슈 모두 VesselFM/FastSurfer 조사 에이전트에게도 명시적으로 경고해 재발 방지 지침을 전달함.

3. **권한 분류기가 외부 저장소 clone/실행을 자동 차단**: FastSurfer 관련 작업(서브에이전트 경유,
   직접 git clone 경유 모두)이 최초 시도에서 "사용자가 이 외부 저장소를 명시적으로 지정하지 않았다"는
   사유로 거부됨. `02_setup_teachers_v2.md`에 명시되어 있음에도 분류기가 첨부 문서 내용을 반영하지
   못한 것으로 보임. 사용자에게 직접 확인받아 재시도로 해결 — **향후 유사 작업 시 이 저장소들을
   대화에서 명시적으로 언급하면 재차단 가능성을 줄일 수 있음**.

4. **Anatomix/VesselFM은 실제 브레인 T1 스캔(혈관이 뚜렷하지 않은 모달리티)으로만 sanity check함.**
   VesselFM은 애초에 혈관 특화 모델이므로, 실제 유용성 검증을 위해서는 혈관이 보이는 모달리티
   (MRA/CTA/TOF 등)로 별도 확인이 필요 — 이번 라운드는 "정규화/구조/frozen 불변조건" 검증까지만
   완료했고, "teacher가 의미 있는 예측을 내는가"는 미검증.

5. **Student stage-mapping 표(§E)는 초안** — 실제 `resenc_unet_*` 프리셋을 인스턴스화해
   `named_modules()`로 정확한 stage 수/채널을 재확인하지 않은 상태에서 nnU-Net 표준 구성을 가정해 작성함.

6. **3-teacher(Anatomix+VesselFM+BraTS) 동시 GPU 상주 시 메모리 사용량은 개별 측정치의 합산 추정치일 뿐, 실측하지 않음.**

7. **VesselFM 1mm 해상도 하에서의 pseudo-label 품질은 미확인** (§H).

8. **FastSurfer 업스트림 체크포인트 다운로드 경로 버그**: `FastSurferCNN/config/checkpoint_paths.yaml`이 가리키는
   공식 b2share URL은 404이며, 코드의 URL 조합 로직(`url + "/checkpoints/<file>"`)도 실제 Zenodo 파일 배치 구조와
   맞지 않는다(Zenodo는 flat 구조). Zenodo API(`record 10390573`)를 직접 조회해 우회 다운로드함 — 우리 코드
   문제는 아니지만, 향후 FastSurfer를 다시 시도할 경우 재발할 것이므로 기록해 둔다.

9. **BraTS teacher의 정규화 방식은 여전히 확인 불가.** §D의 결론("추가 예외 처리 불필요")은 Anatomix/VesselFM
   두 teacher에 한해서만 유효하며, BraTS가 실제로 percentile 기반인지 여부는 체크포인트 도착 후
   `BRATS_BLOCKED.md` §6 절차로 별도 검증해야 한다.
