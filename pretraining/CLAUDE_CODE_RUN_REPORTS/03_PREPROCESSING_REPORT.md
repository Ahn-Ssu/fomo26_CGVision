# Raw-Preserving 전처리 + Teacher별 정규화 이연 — 보고서

작업 기준 문서: `00_PRE_ANALYSIS.md`, `/root/FOMO26/CLAUDE_CODE_RUN_REPORTS/02_TEACHER_REPORT.md`, `/root/FOMO26/CLAUDE_CODE_RUN_REPORTS/01_ASPARAGUS_ANALYSIS.md`.
모든 수치는 실제 GPU 박스에서 실제 FOMO300K 데이터(zip)로 실행한 결과다. 추측 없음.

**코드 위치**: `/root/FOMO26/data/preprocess_raw.py` (전처리), `/root/FOMO26/teacher/brats_teacher.py`
(mask 연결), `/root/FOMO26/data/fomo300k_dataset.py` (index 재사용).

---

## A. 이슈 C 검증 결과 (raw 보존 필요성 최종 판단)

BraTS teacher checkpoint(`/root/teachers/BarTS_Teacher/fold_all/checkpoint_best.pth`, `fold_all` 확인됨 —
`checkpoint_final.pth`도 존재하나 Task 2부터 일관되게 `checkpoint_best.pth` 사용)로 실제 전처리된 pilot
샘플(T1w, `PT001_ClevelandCCF`) 기준 검증:

| | argmax agreement | feature cosine (stage별) | raw 보존 필요? |
|---|---|---|---|
| BraTS: asparagus-norm(raw) 위에 z-norm 재적용 vs raw 직접 z-norm (**명시적 mask 연결 후**) | **99.65%** | dec_stage_0~4: 0.9995~0.9997, final: 0.9997 | **필수는 아니지만 권장** |

**해석**: FastSurfer(79.28%, Task 2)처럼 크게 어긋나지는 않는다 — BraTS는 asparagus 정규화 위에서도
"거의" 작동한다. 하지만 Anatomix(수학적으로 완전 등가)·VesselFM(cosine 1.0)만큼 완벽하지도 않다 —
0.3~0.4%의 argmax 불일치가 남는다(주로 종양 경계 등 boundary 영역으로 추정, 개별 voxel 단위 추가 조사는
하지 않음). **판단**: 이미 raw-preserving 파이프라인을 구축했고(§B), 디스크 여유도 확보했으므로
(§B 용량 이슈 참고) 근사치(99.65%)에 안주하지 않고 raw 보존 방식을 채택하는 것이 안전하다 — 특히
향후 teacher가 추가될 때마다 이 검증을 반복하지 않아도 되는 구조적 이점이 크다.

---

## B. 전처리 사양 확정

### B.1 Resampling
- Interpolation: 이미지 **order=3 (cubic B-spline, SimpleITK `sitkBSpline`)**, mask는 **order=0
  (nearest-neighbor)** — asparagus_preprocessing(`resample.py:44-55`, `order=3`/`order=0` 구분)와
  nnU-Net 관례 그대로 따름 (BraTS teacher와의 기하학적 정합성 유지).
- Target spacing: 1mm isotropic (전체 프로젝트 표준, PRE_ANALYSIS §2.1).

### B.2 이슈 A (background 오염) — 실제로 발생함을 확인, (a)+(b) 조합으로 해결
- **(a)**: foreground mask를 **resample 이전 원본 해상도**에서 `x > 1e-6`으로 계산 후, mask 자체를
  order=0으로 별도 resample — cubic interpolation이 mask 자체를 흐리지 않도록 함.
- **(b)**: resample된 intensity 이미지에서 resample된 mask 기준 배경을 **강제로 0**으로 재클린.
- **검증**: pilot 샘플에서 `data[mask==0]`이 전부 0임을 실측 확인(`np.all(data[mask==0]==0) == True`).
- **부작용 확인**: mask 내부에서도 cubic B-spline 특유의 ringing으로 미세 음수(예: -32.6, 범위
  [-32.6, 621.5], mean 51.7)가 나타남 — 이는 order=3의 잘 알려진 특성이며(asparagus/nnU-Net도 동일
  order=3을 쓰므로 이 특성을 이미 공유), 별도 조치는 하지 않음.
- **Mask 정의 정정**: "brain mask"가 아니라 asparagus의 `x != x.min()`, BraTS teacher의
  `use_mask_for_norm` 관례와 동일한 **"foreground(비배경) mask"**다 — 두개골/scalp 등도 포함될 수
  있음. Pilot 샘플에서 mask 비율 83.8% (잘 크롭된 head FOV 기준 타당한 수치).

### B.3 저장 포맷/용량 (실측 — 2026-07-16 최종)
```
<out_dir>/<dataset>/<dataset>__[<group>__]<subject>__<session>__<원본파일명>.npz
    data     : float16, (D,H,W), 1mm iso, raw intensity, 배경=0
    mask     : uint8,   (D,H,W), 1mm iso, foreground mask
    spacing  : (1.0,1.0,1.0)
    orig_spacing, orig_shape, affine : 원본 provenance
    intensity_scale : float, 저장 시 곱해진 양의 스케일 (기본 1.0, §B.5)
    modality, dataset, subject, session, member : 문자열 메타데이터
```
(`<group>__`은 OpenNeuro 같은 중첩 디렉토리 구조에서만 존재 — §F item 1의 세 번째 버그 참고)

- 저장: `np.savez_compressed`(zlib).
- **최종 전체 실행 결과 (§F item 1 참고, 두 번의 재실행을 거침)**:
  - 원본 165,790건 (2026-07-15, `preprocess_log.csv`) + PT030_OpenNeuro 140,300건
    (2026-07-16, `preprocess_log_openneuro.csv`, `--datasets PT030_OpenNeuro`로 별도 실행) =
    **총 306,090 scan** (`geo_survey`의 306,202/306,203과 거의 정확히 일치 — 나머지 ~100여 건
    차이는 `skipped`(전체 112건: 23+89) 및 zip 자체 read 실패 등으로 설명됨, 추가 조사 안 함).
  - **총 용량 2.5TB** (원본 1.272TB + OpenNeuro 1.370TB), **디스크 여유 7.3TB**.
  - **CSV의 "ok" 카운트와 실제 디스크 파일 수가 정확히 일치함을 두 로그 모두에서 확인**
    (165,790=165,790, 140,300=140,300) — 파일명 충돌 버그(§F) 재발 없음을 최종 검증.
  - 처리 속도: 64 workers 기준 두 실행 모두 ~8.3~9.5 scans/sec, 각각 약 247~257분(4.1~4.3시간).
- **디스크 여유 검증**: float32였다면 전체 약 12.56TB로 원래 9.7TB 예산을 초과해 애초에
  불가능했을 것 — float16 + 압축이 필수적인 선택이었음을 사전 계산(`geo_survey`의 306,202개
  전체 scan shape/spacing 기준: float32 raw=10.05TB+mask 2.51TB=12.56TB, float16
  raw=5.03TB+mask 2.51TB=7.54TB)과 실측(2.5TB, 압축 효과로 더 낮아짐) 양쪽으로 확인.

### B.4 asparagus_preprocessing 코드 재확인 결과
- `normalizer()`에 **`no_norm` 스킴이 이미 존재**(`normalize.py`, `accepted_schemes`에 포함, `if
  scheme == "no_norm": return array`) — "경로 A(Asparagus 확장)"를 택했다면 정규화 스킵 자체는
  설정만으로 가능했을 것.
- 그러나 **foreground mask를 계산·저장하는 로직은 asparagus_preprocessing에 없음** — 이슈 A 해결을
  위해서는 어차피 새 코드가 필요했음.
- **최종 선택: 독립 스크립트(경로 B에 가까움)**, 단 asparagus_preprocessing의 검증된 관례(order=3/0
  구분)는 그대로 재사용. 이유: 전처리된 raw 코퍼스의 소비자는 asparagus의 `PretrainDataset`이 아니라
  우리 자체 `FOMO26/data/fomo300k_dataset.py`이므로, asparagus dataset-script 등록 체계
  (`datasets_pretraining/PT9xx_*.py`)를 따를 필요가 없음. **Asparagus와의 실질적 호환 지점은
  finetune 체크포인트뿐**(`"model."` prefix, Task 2에서 이미 해결) — 전처리된 중간 데이터 포맷 자체는
  호환 요구사항이 없음을 확인함(§E).

### B.5 실제 전체 실행 중 발견된 버그: float16 overflow — scale factor 방식으로 해결

사용자가 실제 전체 실행(64 workers)을 돌리던 중 `RuntimeWarning: overflow encountered in cast`를
직접 발견해 보고했다 — **파일럿(825 entry)에는 없던 문제가 전체 코퍼스에는 있었다**(파일럿의 한계,
§F에 추가 기록).

- **원인**: float16 표현 범위는 약 ±65504인데, 일부 raw MRI intensity는 이를 초과한다(BraTS
  foreground 통계 자체에 `max=3,683,626` 사례가 있음, Task 2). 초과 값을 `.astype(float16)`하면
  예외 없이 조용히 `inf`로 overflow된다 — 이미 처리된 4400여 개 스캔 중 일부가 손상된 채 저장됐을
  가능성이 있어 **사용자가 즉시 실행을 중단**했다.
- **1차 수정(clip)**: 범위를 `[-65000, 65000]`으로 clip. 그러나 실제 데이터로 검증하는 과정에서
  `PT009_BraTS-GEN`(synthetic) 데이터셋의 **T1c 모달리티 전체가 다른 스케일**임을 발견했다 —
  raw 값 자체가 mean=59940, 99.9th percentile=731816, max=1,462,942로, "몇 개 outlier voxel"이
  아니라 **분포 전체**가 float16 범위 위에 있었다. 이 경우 단순 clip은 해당 스캔 voxel의
  16~18%를 단일 상한값으로 뭉개버려 실질적인 신호 손실이었다(파일럿 재현 시 스캔당
  1.4~1.6백만 voxel clip 확인).
- **최종 수정(scale factor)**: clip 대신 **스캔별 양의 스케일 팩터**를 적용하도록 변경
  (`preprocess_raw.py::process_one`, foreground 99.9th percentile 기준으로 필요한 경우만 축소,
  `.npz`에 `intensity_scale`로 기록). 이 방식이 무손실인 이유: 이 프로젝트가 쓰는 정규화 3종
  (Anatomix/VesselFM의 percentile-rescale, BraTS의 z-score) **전부 양의 스케일 변환에 불변**임이
  이미 `/root/FOMO26/CLAUDE_CODE_RUN_REPORTS/02_TEACHER_REPORT.md` §D에서 증명됨 — `normalize(a·x) == normalize(x)` (a>0) — 따라서 저장
  시점에 스케일을 얼마로 잡든 최종 teacher/student 입력은 동일하다. 잔여 clip은 "스케일을
  결정하는 극소수 단일 voxel spike"에 대한 안전장치로만 남겨둠(scale은 percentile 기반이라
  스파이크 하나 때문에 전체 스캔이 과도하게 축소되지 않도록).
- **최종 결과(전체 165,790개 성공 스캔 기준)**: 646건(0.39%)이 scale factor 적용, 5,920건(3.57%)에
  잔여 clip(총 2,992,721 voxel), NaN 0건. `PT009_BraTS-GEN/sub-0027`의 T1c로 개별 검증:
  `intensity_scale=0.0699`, `finite everywhere=True`, mask 밖 완전히 0, 잔여 clip 비율 0.016%만
  — 정상.
- **전수 집계 결과 (사후 분석, §F item 7)**: 최초 서술한 "PT009_BraTS-GEN이 원인"은 **불완전한
  설명이었다** — scale factor가 필요했던 646건 중 실제로는 **`PT010_BrainLat`이 559건으로
  다수**, `PT009_BraTS-GEN`은 87건뿐이었다(`PT010_BrainLat`의 구체적 원인 모달리티/스캔 특성은
  추가 조사하지 않음). 반면 잔여 clip 5,920건 중 4,901건(82.8%)은 `PT020_HCP_Wu_Minn`의 dwi
  모달리티였는데, 이쪽은 확인 결과 **`intensity_scale`이 전부 1.0으로 유지되고 clip된 voxel
  수도 스캔당 median 7개(최대 199개, 전체 500만+ voxel 중)** — 설계 의도대로 "진짜 고립된
  단일 voxel spike"에 대한 안전장치가 정상 동작한 것이지 새로운 문제가 아니다.
- **교훈**: 파일럿(825 entry, 200 zip)이 이 문제를 잡아내지 못한 것은 표본이 작아서였다 —
  `PT009_BraTS-GEN`/`PT010_BrainLat`처럼 소수 데이터셋에 국한된 이상치는 소규모 무작위 파일럿으로는
  발견하기 어렵다. **작업 원칙 5("소규모 파일럿으로 검증 후 전체 실행")를 지켰음에도 이 버그를
  사전에 잡지 못했다** — 사용자의 실시간 관찰이 없었다면 전체 코퍼스가 조용히 손상된 채로 완료됐을
  것이다(§F에 이 한계를 명시).

---

## C. Teacher 정규화 검증

| Teacher | 정규화 수식 (원본 코드 재확인) | 원본 대비 cosine sim | 통과? |
|---|---|---|---|
| BraTS | `clamp(fg, q=0.995) → z-score(fg)`, **명시적 mask 사용**(이번에 연결) | 0.9995~0.9997 (stage별) | **예 (근사)** |
| VesselFM | `ScaleIntensityRangePercentiles(1,99,[0,1],clip=True)`, full-volume, mask 없음 (Task 2, MONAI 소스 직접 확인) | 1.0 | **예 (정확)** |
| Anatomix | `normalize_img(percentile=99.99,zero_centered=True)`, full-volume (Task 2, 대수적 증명) | 1.0 (수학적 항등) | **예 (정확)** |

VesselFM/Anatomix는 Task 2에서 이미 검증되어 재확인만 했다(수식이 원본 코드와 정확히 일치함을
`vesselfm_teacher.py`/`anatomix_teacher.py`의 인용 docstring으로 재대조). **BraTS만 이번에 처음으로
mask를 명시적으로 연결**했고, 그 결과가 §A의 수치다.

---

## D. 극단 shape 필터링 결과 — 버그 발견 및 수정

**최초 구현에 실제 버그가 있었다** (200-zip pilot, 825 entry 기준):

| 버전 | ok | skipped | 비고 |
|---|---|---|---|
| 최초(voxel-grid 기준) | 143 (17.3%) | 682 (82.7%) | **버그** |
| 수정 후(physical-extent mm 기준) | 825 (100%) | 0 | 정상 |

**버그 원인**: `shape[2] > 500`로 "슬라이스 축"을 index 2로 가정했으나, nibabel 배열 축 순서는 파일마다
다르다 — 실제로 `(18, 512, 512)`, `(512, 20, 512)` 같은 **정상적인 고해상도 임상 스캔**(512×512
in-plane, 18~20 slice thick)이 "손상 파일"로 오분류되고 있었다. 마찬가지로 `axis < 32`도 **native
voxel 개수**로 체크해 `(256,256,19)` 같은 정상 thick-slice 스캔(geometry survey의 "25.8%
thick-slice"에 해당)이 "패치 미달"로 오분류됐다.

**수정**: voxel 개수가 아니라 **물리적 크기(voxel_count × spacing, mm)**로 판정하도록 재설계
(`preprocess_raw.py::check_extreme_shape`). 축 순서에 무관하고, 실제 resample 후 patch 적합성을
올바르게 반영한다. 수정 후 pilot(825 entries)에서 **skip 0, error 0**.

이 버그가 그대로 306K 전체에 적용됐다면 **82.7%의 스캔을 부당하게 버렸을 것** — 소규모 파일럿을
먼저 돌려본 것이 결정적이었다(작업 원칙 5).

`skipped_scans.csv`: pilot 규모(825건)에서는 스킵된 건이 없어 별도 파일 없음 — 실제 극단치가
나타나면 `<out_dir>/preprocess_log.csv`의 `status=skipped` 행에 사유와 함께 기록되도록 구현됨.

---

## E. Asparagus finetune 호환성

전처리 출력(raw+mask `.npz`)은 asparagus의 `PretrainDataset`/`asp_pretrain`이 아니라 우리 자체
`FOMO26/run_pretrain.py`(accelerate)가 소비하므로, **이 중간 데이터 포맷 자체에는 asparagus 호환
요구사항이 없다**(§B.4). 실질적 호환 지점은 **pretrain 완료 후 저장하는 체크포인트**뿐이며, 이는
Task 2의 `run_pretrain.py::save_checkpoint()`에서 이미 `{"state_dict": {"model.<key>": tensor}}`
형식으로 저장하도록 구현·검증되어 있다(`asp_finetune_seg/cls/reg`가 요구하는 형식과 정확히 일치,
`/root/FOMO26/CLAUDE_CODE_RUN_REPORTS/01_ASPARAGUS_ANALYSIS.md` §C.4). 이번 작업으로 추가 변경 없음.

---

## F. Blocker 및 미해결 이슈

1. ~~전체 306,202-scan 실행은 아직 하지 않았다~~ → **완료 (2026-07-15), 그러나 불완전했음이
   2026-07-16에 밝혀짐.** 사용자가 64 workers로 전체 실행, 257.2분 소요, `ok=165790 skipped=23
   errors=0`, 총 1.272TB (`/root/data/FOMO300K_preprocessed/`). 최초 실행 중 float16 overflow
   버그를 사용자가 실시간으로 발견해 중단 → 수정(§B.5) → 재실행한 결과였다.

   **추가 버그 발견 (2026-07-16, 사용자가 `mri_info.tsv`의 스캔 수와 우리 npz 수가 안 맞는다고
   지적하여 재조사)**: `build_index()`의 glob 패턴 `*/sub-*/ses-*.zip`(경로 3단계 고정)이
   `PT030_OpenNeuro/<accession-id>/sub-*/ses-*.zip`(OpenNeuro는 여러 독립 데이터셋을
   accession-번호 하위 폴더로 한 단계 더 묶어놓음, 예: `ds004271/`)를 **통째로 놓치고 있었다.**
   실측: 전체 zip 81,190개 중 45,377개(56%)가 이 구조였고, `geo_survey`(재귀 glob으로 처음부터
   정확했음, 306,203 rows)와 대조한 결과 **PT030_OpenNeuro만 140,389 scan(전체의 45.8%)**으로,
   실행이 끝난 165,790건보다도 많았다. 부수적으로 `dataset_name = zp.parts[-3]`도 경로 깊이가
   섞이면서 OpenNeuro의 각 accession-id를 별개 "데이터셋"으로 잘못 취급할 뻔한 버그였다.
   (참고: `mri_info.tsv`의 306,207 rows는 PT030_OpenNeuro가 **0건** 포함된 카탈로그라, geo_survey의
   306,203과 우연히 숫자만 비슷할 뿐 서로 다른 대상이다 — mri_info.tsv는 OpenNeuro 편입 이전에
   만들어진 것으로 보인다.)

   **수정**: `build_index()`를 재귀 glob(`**/sub-*/ses-*.zip`) + `relative_to(root).parts[0]`
   방식으로 교체, 재검증 결과 306,202건(geo_survey와 거의 정확히 일치) 확인. 낡은 인덱스 캐시
   삭제. `preprocess_raw.py`에 `--datasets` 필터를 추가해 기존 165,790건은 그대로 두고
   PT030_OpenNeuro 140,389건만 별도 로그(`preprocess_log_openneuro.csv`)로 추가 실행 **진행
   중**(백그라운드, 64 workers). 완료되면 두 로그를 하나로 합치고 이 보고서를 최종 수치로
   업데이트할 것.

   **교훈**: 소규모 파일럿(200 zip, 순차 인덱스 앞부분)도, 사용자의 float16 overflow 실시간
   관찰도 이 버그는 잡지 못했다 — 인덱싱 자체가 특정 데이터셋을 "존재하지 않는 것처럼" 조용히
   건너뛰는 종류의 버그는, 처리된 숫자가 예상보다 적다는 것을 **외부 소스(mri_info.tsv)와
   대조하지 않는 한** 발견하기 어렵다. 이번에도 사용자의 직접 대조가 없었다면 전체 코퍼스의
   거의 절반이 조용히 빠진 채로 "완료"로 보고될 뻔했다.

   ~~`geo_survey`의 306,202 record와 다른 이유는 미조사~~ → **위에서 규명 완료.**

   **두 번째 버그 (같은 날 2026-07-16, OpenNeuro 첫 재실행 직후 발견): 파일명 충돌로 인한 데이터
   유실.** `preprocess_raw.py`의 출력 파일명이 `{dataset}__{subject}__{session}__{원본파일명}`
   형식이었는데, OpenNeuro의 각 accession 하위 데이터셋(`ds000001`, `ds004271`...)이 서로 독립적으로
   `sub-01`, `ses-01` 같은 일반적인 BIDS 이름을 재사용하기 때문에, **`dataset`이 전부
   "PT030_OpenNeuro"로 같고 accession id가 파일명에 없어서** 서로 다른 accession의 스캔이 같은
   출력 경로로 충돌했다. 실측: OpenNeuro 첫 실행에서 `ok=140,300`으로 보고됐으나 실제 디스크에는
   **68,583개 파일만 존재**(51% 유실, 71,717개가 조용히 덮어써짐 — 어느 스캔이 "살아남았는지"는
   64-worker 병렬 실행의 완료 순서에 의존하는 사실상 임의적 결과라 신뢰 불가). 비-OpenNeuro
   165,790건은 별도로 전수 대조해 **충돌 0건, CSV·디스크 파일 수 정확히 일치** 확인 — 이 버그는
   중첩 디렉토리 구조(OpenNeuro)에만 해당.

   **수정**: `build_index()`가 dataset과 subject 사이의 중간 경로 성분을 `group` 필드로 캡처하도록
   추가(예: `ds004271`), `preprocess_raw.py`의 출력 파일명과 `load_preprocessed_index()`의 역산
   로직 양쪽에 `group`을 포함시켜 유일성을 복원. 손상된 첫 실행 결과(604GB)는 삭제.

   **세 번째 버그 (수정 직후 재실행에서 발견): 캐시가 코드 수정을 무시함.** `group` 필드를 추가한
   직후 `build_index()`만 독립적으로 재검증했을 땐 140,389/140,389 유니크로 정상이었는데, 실제
   `preprocess_raw.py`를 다시 돌리자 **똑같은 충돌(140,300건 중 68,583개만 실존)이 그대로
   재현**됐다. 원인: `load_or_build_index()`가 `group` 필드 추가 *이전*에 생성된 캐시 파일
   (`fomo300k_index.json`)을 스키마 검증 없이 그대로 재사용하고 있었다 — 코드는 고쳐졌지만
   캐시가 낡아서 실행에는 반영되지 않은 것. `load_or_build_index()`에 캐시 항목의 키 집합이
   현재 스키마(`group` 포함)와 일치하는지 확인하고, 불일치 시 자동으로 재빌드하도록 방어 로직을
   추가. 낡은 캐시 삭제 후 `process_one()`을 실제 충돌 사례(726개가 몰릴 뻔했던 케이스)에 직접
   적용해 `ds000001`/`ds000002`로 정확히 분리됨을 확인하고 나서 재실행.

   **최종 결과 (2026-07-16 완료)**: `ok=140,300 skipped=89 errors=0`, 1.370TB. **CSV의 유니크
   파일명 수(140,300) = 실제 디스크 파일 수(140,300) — 완전 일치 확인.** 기존 165,790건과 합쳐
   **총 306,090 scan, 2.5TB** (`/root/data/FOMO300K_preprocessed/`, 로그는
   `preprocess_log.csv` + `preprocess_log_openneuro.csv` 두 파일로 분리 유지). 디스크 여유
   7.3TB. `FOMO300KPreprocessedDataset`이 두 로그를 모두 읽도록 연결, `run_pretrain.py`
   end-to-end 스모크 테스트로 최종 확인 완료.

   **교훈**: "처리 결과 수가 이상하다"는 신호(§ item 1)를 좇다가 재실행했는데, **재실행 자체가
   또 다른 잠재적 데이터 무결성 버그를 갖고 있었다** — 첫 버그를 고치는 과정에서 두 번째 버그를
   우연히 발견한 셈이다(파일 수를 실제로 세어보지 않았다면 "ok=140300"이라는 숫자만 보고 정상
   완료로 오인했을 것). **"성공"으로 보고된 카운트와 실제 디스크의 파일 수를 항상 대조 검증해야
   한다**는 게 이번 작업 전체를 관통하는 가장 중요한 교훈이다. 세 번째 버그(스테일 캐시)는 한
   겹 더한 교훈을 남겼다: **코드를 고쳤다고 해서 다음 실행이 그 수정을 실제로 쓴다는 보장은
   없다** — 캐시/중간 산출물이 하나라도 남아있으면 조용히 옛 동작을 재현할 수 있으므로, 버그
   수정 후 재실행 전에는 관련 캐시를 항상 명시적으로 지우거나(이번처럼) 캐시 자체에 스키마
   검증을 넣어야 한다.

2. **BraTS 정규화는 "근사적으로만" 등가**(§A, 99.65%) — Anatomix/VesselFM처럼 완벽하지 않다. 남은
   0.35% 불일치의 정체(경계 영역인지, 다른 원인인지)는 voxel 단위로 추가 조사하지 않았다.

3. ~~Pilot 표본(200 zip)이 전체 306K 코퍼스를 대표하는지 검증 안 됨~~ → **실제로 대표하지 못했음이
   확인됨.** 파일럿(825 entry)에서는 float16 overflow가 0건이었지만, 전체 실행에서는 646건이
   scale factor가 필요했다(§B.5) — 원인이 `PT009_BraTS-GEN`이라는 **단일 데이터셋에 국한된
   이상치**였기 때문에, 무작위/순차 소규모 표본으로는 애초에 잡을 수 없는 종류의 문제였다.
   **교훈**: 이후 유사 작업에서는 "각 데이터셋에서 최소 N개씩" 식의 층화 표본(stratified
   sampling) 파일럿이 순수 무작위/순차 표본보다 이런 데이터셋별 이상치를 잡는 데 더 효과적일
   것 — 이번엔 사용자의 실시간 관찰로 대신 잡혔다.

4. ~~FOMO26/data/fomo300k_dataset.py(학습용 Dataset)는 아직 zip 직접 읽기 방식 그대로다~~ —
   전체 전처리가 완료된 지금은 **우선순위가 높아진 다음 작업**이다. `run_pretrain.py`가
   `/root/data/FOMO300K_preprocessed/*.npz`를 직접 읽도록 `FOMO300KDataset`을 교체하면
   (zip 추출+resample 비용이 사라지므로) 학습 데이터 로딩이 크게 빨라질 것으로 예상되나
   실측하지 않았다. 통합 시 `run_pretrain.py`가 각 teacher episode에 mask를 함께 전달하도록
   확장해야 함(현재는 이미지만 전달 — `brats_teacher.py`는 mask 없이도 폴백 동작하므로 당장
   깨지지는 않지만, 정확도를 위해선 연결이 필요).

5. **VesselFM/Anatomix wrapper 코드 자체는 수정하지 않았다** — mask 불필요한 정규화라 Task 2
   구현 그대로 유효함을 재확인만 했음(§C).

7. ~~`preprocess_log.csv`를 전수 분석하지 않았다~~ → **집계 완료(§B.5)**. Scale factor 필요
   646건 중 `PT010_BrainLat` 559건, `PT009_BraTS-GEN` 87건(최초 서술은 BraTS-GEN만 원인으로
   잘못 단정했었음 — 정정). 잔여 clip 5,920건 중 82.8%(`PT020_HCP_Wu_Minn` dwi)는 스캔당
   median 7 voxel 수준의 정상적인 안전장치 동작. **미해결로 남는 것**: `PT010_BrainLat`이
   왜 scale factor가 필요한 스캔 대부분을 차지하는지 근본 원인은 조사하지 않음(PT009처럼
   raw 값을 직접 로드해 확인하지 않았음).

6. **압축 방식(zlib/npz)이 최적인지는 검증하지 않았다** — blosc2/zstd 등 더 빠른 압축 라이브러리를
   비교하지 않고 numpy 표준 `savez_compressed`만 사용함. 처리 속도(8.5 scans/sec)의 병목이
   압축인지 I/O인지 resample 연산인지 프로파일링하지 않았다.
