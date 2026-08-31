# 04. Teacher 로스터 확장 (VoCo-OpenMind, V-JEPA 2.1) — 중복 검증 및 distillation 통합 리포트

**목적**: 기존 3-teacher 로스터(anatomix+brains/vesselFM/BraTS)에 VoCo-OpenMind, V-JEPA 2.1을 추가했을 때 (1) 서로 겹치지 않는 정보를 주는지 정량/정성적으로 확인하고, (2) 실제 `run_pretrain.py` 학습 파이프라인에 두 teacher를 어떻게 distillation 대상으로 연결했는지 기록한다.

**관련 산출물**: `/root/external_teacher_probe/`(독립 탐색, PCA 분석 전체), 이 문서(실제 학습 코드 통합 부분).

---

## 0. 요약 (TL;DR)

| 질문 | 답 |
|---|---|
| VoCo가 기존 3개 teacher와 겹치는가? | **아니오** — PR/C depth별 궤적이 구조적으로 다르고(§1.1), 동일 stride/동일 volume에서 시각적으로도 확연히 다른 특징을 보임(§1.2) |
| V-JEPA가 기존 teacher들과 겹치는가? | **아키텍처/데이터/objective 세 축 모두 이질적** — CNN이 아닌 ViT, 의료영상이 아닌 자연영상/비디오로 사전학습, random-init 대비 효과크기가 CNN 계열과 자릿수 자체가 다름(§1.3). 단, VoCo처럼 동일 stride에서의 직접 PR/C 대조표는 아키텍처가 근본적으로 달라(ViT는 depth 무관 고정 해상도) 만들지 않았고 이 한계를 §1.3에 명시함 |
| V-JEPA feature를 어느 student stage에 매핑했는가? | `layer2→enc_stage2, layer5→enc_stage3, layer8→dec_stage1, layer11→dec_stage2` — 요청하신 매핑과 **정확히 일치**(§2) |
| VoCo는 어떻게 distillation에 연결했는가? | 순수 encoder-to-encoder, `enc_stage_0..4` 1:1 매핑, student도 decoder를 아예 건너뛰는 `forward_encoder_only()`로 계산 효율화(§3) |
| 지금 학습 코드에 실제로 반영됐는가? | 네 — 4개 파일 수정 + 2개 파일 신규, 실제 ver4 체크포인트로 backward-compat 재확인, VoCo/V-JEPA 각각 mini-run으로 loss 하강 확인(§4) |
| 코드 리뷰(augmentation 정합성 + 11개 항목)에서 발견된 문제는? | **실제 버그 1개 발견·수정**(캐시 padding 값 불일치, 30K 중 20% 영향, §6.2) + **완화 조치 1개 추가**(V-JEPA MSE 항 스케일, `--mse_weight_overrides`, §6.6). 나머지 9개는 코드/실측으로 안전함을 확인했거나 설계상 알려진 트레이드오프로 문서화(§6) |

---

## 1. 모든 teacher가 서로 겹치지 않는다는 확인

### 1.1 VoCo vs 기존 3개 teacher — PR/C 직접 대조 (정량)

`/root/external_teacher_probe/probe_existing_teachers_enc.py`로 BraTS/vesselFM/anatomix+brains의 **encoder측**을 VoCo와 **완전히 동일한 방법론**(30-volume pooled PCA, 채널 표준화, 전체 고유값분해, sign 고정)으로 재실행해서 직접 비교했다 (이전 조사에서 "미완료"로 남겨뒀던 항목, 이번에 완료).

| stride | VoCo | BraTS | vesselFM | anatomix+brains |
|---|---|---|---|---|
| 1 | 0.223 | 0.255 | 0.163 | 0.372 |
| 2 | 0.317 | 0.280 | 0.296 | 0.735 |
| 4 | 0.415 | 0.382 | 0.312 | 0.463 |
| **8** | **0.450** | 0.179 | 0.273 | 0.312 |
| 16 | 0.210 | 0.082 | 0.157 | (4-stage라 없음) |

**핵심 차이**: 기존 3개는 전부 stride 4 이후 PR/C가 감소한다(BraTS는 0.382→0.179→0.082로 단조 감소). **VoCo만 stride 8에서 자기 자신의 최댓값(0.450)을 찍는다** — 다른 셋이 정보를 압축해 들어가는 지점에서 VoCo는 오히려 유효 차원이 가장 높다. Depth에 따른 압축 패턴 자체가 구조적으로 다르다는 뜻이다.

### 1.2 VoCo vs 기존 3개 teacher — 시각적 대조 (정성)

같은 T1 volume, 같은 stride 4 지점에서 PC1-3 RGB 합성을 비교하면:
- **BraTS**: 뇌실 나비 모양이 PC2 하나만으로도 아주 선명 (`output/existing_teachers_enc_viz/pca_brats_enc_1_T1_1mm_stride4.png`)
- **vesselFM**: 중간 정도로 뇌실 윤곽 인식 가능
- **anatomix**: 흐릿하게만 보임
- **VoCo**: 이 stride에서는 **뚜렷한 구조가 전혀 안 보임** — VoCo의 명확한 신호는 stride 8의 PC7(=PC4-10 범위, RGB 3장만 봐선 안 보이던 지점)에서야 나타난다 (`output/cnn_voco/pca_ResEncL-VoCo_1_T1_1mm_stride8.png`)

같은 위치·같은 해상도에서 "구조가 보이기 시작하는 지점"이 다르다는 것은 단순히 "신호가 세다/약하다"의 문제가 아니라 **다른 종류의 표현을 인코딩하고 있다**는 정성적 증거다.

### 1.3 V-JEPA vs 모든 CNN teacher (VoCo 포함) — 아키텍처/데이터 축 (정성 + 부분 정량)

V-JEPA는 ViT라서 VoCo처럼 "같은 stride에서 직접 PR/C 대조"가 애초에 성립하지 않는다 — attention은 depth와 무관하게 항상 같은 해상도(512px 기준 32×32×64)를 유지하므로, BraTS/VoCo 같은 CNN의 "depth가 깊어질수록 해상도가 줄어든다"는 축 자체가 없다. 대신 세 가지 독립적인 근거로 비중첩을 뒷받침한다:

1. **아키텍처**: CNN(국소 receptive field, depth와 결합된 spatial downsampling) vs Transformer(전역 attention, 모든 layer가 같은 해상도) — 로스터에서 유일한 non-CNN.
2. **사전학습 데이터/목적**: 나머지 넷(BraTS/vesselFM/anatomix/VoCo)은 전부 의료영상(실제 뇌 MRI 또는 그것을 흉내낸 synthetic) 기반인데, V-JEPA 2.1은 **자연 이미지/비디오**로 사전학습된 모델이다 — 의료 도메인을 한 번도 본 적이 없다는 게 오히려 상보성의 근거([[project_vjepa_teacher_roadmap]] 메모 참고).
3. **Random-init 대비 효과크기의 자릿수**: 같은 방법론으로 측정한 Cohen's d가 VoCo는 stage별 -2.85~+15.5인데 V-JEPA는 +150~+342로, **CNN 계열 어떤 teacher와도 자릿수가 다르다** (random ViT 자체가 random CNN보다 PR/C가 53배 낮은 극단적으로 붕괴된 기준선을 갖기 때문 — `/root/external_teacher_probe/ANCHOR_RESULTS.md`).

**한계 명시**: VoCo처럼 "동일 volume, 동일 stride에서 PR/C 숫자를 나란히 놓고 비교"하는 표는 만들지 않았다 — 위 이유(해상도 축 자체가 다름) 때문에 만들어도 직접 비교 가능한 숫자가 아니다. 필요하면 V-JEPA의 512px(stride4 매칭)과 BraTS/VoCo의 stride4를 "같은 stride 라벨"로만 나란히 놓고 참고용으로 볼 수는 있으나, 이는 이번 리포트에서 수행하지 않았다.

---

## 2. V-JEPA feature distillation 매핑 재확인

요청하신 매핑을 실제 구현 코드에서 직접 추적해서 재확인했다:

**`networks/projections.py` STAGE_MAP["vjepa"]**:
```python
"vjepa": {("enc", 2): 0, ("enc", 3): 1, ("dec", 1): 2, ("dec", 2): 3}
```
(student_key → teacher_idx: student `enc_stage_2`↔teacher_idx 0, `enc_stage_3`↔1, `dec_stage_1`↔2, `dec_stage_2`↔3)

**`data/vjepa_cache_dataset.py` LAYER_TO_TEACHER_STAGE**:
```python
LAYER_TO_TEACHER_STAGE = {2: 0, 5: 1, 8: 2, 11: 3}
```
(V-JEPA layer → teacher_idx: layer2↔0, layer5↔1, layer8↔2, layer11↔3)

**두 매핑을 합치면**:

| V-JEPA layer | student stage | 근거 |
|---|---|---|
| layer2 (가장 primitive) | **enc_stage_2** (32³, bottleneck 이전) | student 자체 처리 순서상 이른 지점 ↔ teacher의 가장 얕은 layer |
| layer5 | **enc_stage_3** (16³, bottleneck 이전) | |
| layer8 | **dec_stage_1** (16³, bottleneck 이후) | |
| layer11 (가장 semantic) | **dec_stage_2** (32³, bottleneck 이후) | student 자체 처리 순서상 가장 늦은 지점 ↔ teacher의 가장 깊은 layer |

**요청하신 대로 정확히 일치함을 확인.** 8³ bottleneck(enc_stage_4/dec_stage_0)은 의도적으로 제외 — masking 후 유효 voxel이 너무 적어(외부 조사에서 반복 확인된 rank-deficiency) 그 지점의 gradient 신호가 희박하다고 판단했고, 어차피 BraTS/vesselFM이 `dec_stage_0`을 이미 커버하고 있어 중복 우려도 없다.

### 구현 상세

- **캐싱 방식**: 300K 전체가 아니라 1/20 규모(30,000 volume, 데이터셋 38개에 water-fill로 균등 배분)를 512px에서 미리 추출·저장. 온라인으로 매 step V-JEPA를 돌리면 step당 평균 ~375ms가 추가로 얹혀(캐싱 안 하면 학습 전체 시간이 2배 이상 증가) 캐싱이 필수였음.
- **저장 형식**: volume당 student-format 이미지(asparagus z-norm 적용된 128³) + 4개 pooled feature(`layer2/11`→32³, `layer5/8`→16³), 총 30,000개 `.npz` (~3.3TiB), `/root/external_teacher_probe/vjepa_cache/`.
- **버그 발견·수정**: 캐시 저장 시 feature의 공간축 순서가 `(T,H,W,D)`였는데 student/다른 teacher는 전부 `(X,Y,Z,D)`(Z가 마지막) 관례 — 그대로 썼다면 teacher-student가 서로 다른 물리적 위치를 비교하는 셈이 되어 distillation이 조용히 깨졌을 것. 재추출(17시간) 없이 로더(`VJEPACachedDataset`)에서 permute로 수정, 수치로 직접 검증함(재계산값과 캐시값이 정확히 일치).
- **학습 루프 연결**: `run_pretrain.py`에서 episodic sampler가 `"vjepa"`를 뽑으면, 일반 dataloader 대신 `VJEPACachedDataset` 기반 별도 loader에서 **이미지+teacher_feats 쌍**을 통째로 가져옴 — 살아있는 forward pass가 전혀 없음. `student.forward_with_features(images, "vjepa", include_encoder=True)`로 enc/dec feature를 모두 뽑아 4개 stage에 맞춰 투영.
- `--vjepa_cache_dir`를 지정하면 `--convpass_peft --convpass_encoder`가 강제된다(enc_stage 타깃이 있으므로 encoder-side Convpass 용량이 반드시 필요).

---

## 3. VoCo encoder-to-encoder distillation 구현

### 3.1 왜 encoder-only인가

VoCo 체크포인트 자체가 encoder만 있고 decoder가 없다(`state_dict`에 decoder 키 자체가 없음, `external_teacher_probe/NOTES.md` §1에서 실측 확인). 그리고 student·VoCo 둘 다 **완전히 같은 nnU-Net ResEnc plan**(channels `[32,64,128,256,320,320]`, stride `[1,2,2,2,2,2]`)이라 별도 매핑 로직 없이 `enc_stage_i ↔ VoCo stage i` 1:1로 그대로 맞아떨어진다.

**`networks/projections.py` STAGE_MAP["voco"]**:
```python
"voco": {("enc", 0): 0, ("enc", 1): 1, ("enc", 2): 2, ("enc", 3): 3, ("enc", 4): 4}
```
`enc_stage_5`(stride 32)는 양쪽 다 제외 — VoCo 쪽은 PCA 조사에서 30-volume pooling을 해도 rank-deficient였던 지점이라, student도 굳이 거기까지 맞출 이유가 없다고 판단.

### 3.2 계산 효율화 — student encoder-only forward

요청하신 대로, VoCo가 선택된 step에서는 **student의 decoder를 아예 실행하지 않는다.** `networks/student.py`에 새 메서드를 추가:

```python
def forward_encoder_only(self, x, teacher_name=None):
    skips = self.encoder(x, teacher_name)
    return {f"enc_stage_{i}": f for i, f in enumerate(skips)}
```

`self.decoder(...)` 호출 자체가 없으므로 decoder forward는 물론 backward(gradient 계산)까지 스킵된다 — VoCo는 애초에 decoder 타깃이 없으니 이건 순수하게 낭비였던 계산이다. 이 분기는 하드코딩이 아니라 `STAGE_MAP`에서 자동으로 유도된다:

```python
def is_encoder_only_teacher(teacher_name):
    """STAGE_MAP의 모든 항목이 enc side면 True"""
    smap = STAGE_MAP.get(teacher_name, {})
    return bool(smap) and all(_parse_stage_key(k)[0] == "enc" for k in smap)
```

**실측 효과** (mini-run, batch_size=1): VoCo step **~134ms**, BraTS(살아있는 teacher forward + student 전체 forward/backward) **~346~1237ms** — VoCo는 3~9배 빠르다. 캐시가 필요 없는 살아있는 teacher이면서(입력 해상도가 student와 동일한 128³라 별도 upscale/캐싱 파이프라인 자체가 불필요) decoder까지 건너뛰니 로스터 중 가장 저렴한 teacher가 됐다.

### 3.3 정규화 — 별도로 raw intensity를 흘려보내야 했던 이유

VoCo의 사전학습 정규화는 `ZScoreNormalization`, **`use_mask_for_norm=False`**(마스크 없이 전체 crop 기준 z-score) — 반면 이 코드베이스의 기존 관례는 asparagus의 mask 기반 z-score(게다가 clamp 포함)를 먼저 적용한 뒤 그 결과를 teacher에 넘긴다. Anatomix처럼 percentile 정규화는 affine 불변이라 이중 정규화가 문제 없지만, **clamp가 낀 z-score는 두 번 적용해도 원래 VoCo가 학습 때 본 분포로 돌아가지 않는다** — 이 점을 놓치면 "VoCo가 별로다"라는 잘못된 결론으로 이어질 뻔했다.

그래서 `data/fomo300k_dataset.py`를 수정해 **raw(전처리 전) crop도 asparagus-normed crop과 같은 위치에서 같이 잘라 반환**하도록 했다(`_random_crop_or_pad`에 `extra2` 인자 추가). `teacher/voco_teacher.py`의 `preprocess()`는 `meta["raw"]`를 받아 그 위에서 자체 whole-volume z-score를 계산한다(`norm_type="absolute"`로 선언, 일반 `x` 인자는 아예 쓰지 않음). 다른 teacher들은 `raw` 메타를 그냥 무시하므로 영향 없음.

---

## 4. 구현 요약 및 검증

### 변경/신규 파일

| 파일 | 내용 |
|---|---|
| `networks/student.py` | `forward_with_features(..., include_encoder=False)` (기본값 유지, additive), `forward_encoder_only()` 신규 |
| `networks/projections.py` | `STAGE_MAP`이 `(side, idx)` 키 지원하도록 확장, `"vjepa"`/`"voco"` 항목 추가, `is_encoder_only_teacher()`/`teacher_needs_encoder_features()` 헬퍼, **기존 teacher들의 head 이름은 완전히 그대로 유지**(체크포인트 호환성) |
| `data/fomo300k_dataset.py` | `_random_crop_or_pad`에 `extra2`(raw crop) 지원 추가, 두 Dataset 클래스 모두 `raw` 필드 반환 |
| `data/vjepa_cache_dataset.py` | **신규**: `VJEPACachedDataset` — 캐시 로드 + 축 순서 버그 수정 |
| `teacher/voco_teacher.py` | **신규**: `VoCoTeacher` — encoder 재구성, `strict=True` 로드, raw 기반 whole-volume z-score |
| `teacher/registry.py` | `"voco"` 등록 (V-JEPA는 살아있는 모델이 없어 registry에 없음, `--vjepa_cache_dir`로 별도 처리) |
| `run_pretrain.py` | `--vjepa_cache_dir` 플래그, teacher 샘플링 순서 재구성(teacher 먼저 선택 후 해당 로더에서 batch 획득), encoder-only 분기, `raw` meta 전달, `evaluate()`도 동일 로직 반영 |

### 검증

1. **VoCo mini-run** (8 step, teacher_weights로 voco 가중치↑): loss 1.02→0.94로 정상 하강, decoder 스킵으로 실제 속도 향상 확인.
2. **V-JEPA mini-run** (6 step): loss 1.13→1.03 정상 하강, 캐시라서 살아있는 teacher(brats)보다 빠름.
3. **VoCo eval 경로**: `evaluate()`가 VoCo를 포함해 4개 teacher 전부에 대해 정상적으로 holdout loss 계산 (V-JEPA는 캐시가 holdout 이미지와 무관해 구조적으로 eval 루프에 못 들어감 — 이건 알려진 제약으로 문서화해둠).
4. **체크포인트 재개**: 저장된 checkpoint에 `heads.vjepa.stage_enc_2/3`, `heads.voco.stage_enc_0..4` 키가 정확한 이름으로 저장됨을 확인, resume 정상 동작.
5. **하위 호환성**: 실제 학습 중이던 **ver4의 진짜 체크포인트**(`step_200000.pt`, VoCo/V-JEPA 없이 학습됨)를 이번 변경된 코드로 resume — 100% 정상 동작, loss가 원래 수렴 구간(0.05~0.1)과 일치. proj_heads 로딩을 `strict=False`로 바꿔서 향후 teacher 로스터가 달라져도(예: 처음엔 3-teacher로 학습하다 나중에 voco/vjepa 추가) resume이 깨지지 않도록 함.
6. `examples/sanity_check_train_step.py`도 시그니처 변경에 맞춰 함께 수정 후 통과 확인.

---

## 5. 실행 예시 및 다음 단계 (2026-08-02 확정)

말씀하신 계획대로, "encoder+decoder 모두에 Convpass가 있는 구조"(`--convpass_peft --convpass_encoder --convpass_no_skip_alpha`, ver4와 동일 arch)를 고정하고 teacher 다양성만 바꿔가며 비교한다.

### 5.1 사전 검증 (완료)

실제 launch 전에 두 단계로 재검토했다:

1. **4-GPU preflight** (`expr/ver5_preflight/`, `--limit 300 --steps 20`): voco/vesselfm/brats/vjepa가 정상적으로 샘플링되어 크래시 없이 20 step 완주. `anatomix+brains`는 uniform-weight(4.15개 teacher 중 실질 weight 1/4.15=24%) 기준으로도 20 step 안에 안 뽑힐 확률이 ~0.4%로 낮지만 있는 사건이라(실측 시드 재현으로 확인, RNG 버그 아님), 별도로 anatomix+brains만 단독 forward+backward를 돌려 정상 동작을 재확인했다.
2. **실제 corpus 5K-step 버그체크 런** (`expr/ver5_bugcheck/`, `--limit` 없음, 5개 teacher 전체): 크래시/NCCL watchdog/loss NaN 여부를 확인하는 용도로 실행.

### 5.2 Teacher 샘플링 가중치 — vjepa=0.2로 최종 확정 (균등안 재검토 후 변경)

처음엔 5개 teacher 모두 균등(각 1)으로 정했으나("기존 3-teacher와 step 수를 얼추 비슷하게" 목표에 350K 시점 +5%로 가장 근접), V-JEPA 캐시(3만 개 고정 subset)가 350K까지 평균 ~9.3회 반복 노출된다는 부담이 재검토 대상이 됐다. 최종적으로 **`vjepa=0.2`, 나머지 4개(anatomix+brains/vesselfm/brats/voco)=1**로 확정하고, "step-parity 목표 total step"도 그에 맞춰 **350K → 280,000**으로 재계산했다(가중치가 바뀌면 CNN teacher 4개가 ver4와 같은 66,667 step/teacher에 도달하는 total step도 같이 바뀌기 때문 — 350K는 균등 가중치 기준 계산이라 vjepa=0.2에는 더 이상 맞지 않음):

| 구성 | teacher당(CNN 4개) step | vjepa step | vjepa 캐시 반복(batch=4 기준) |
|---|---|---|---|
| ver4 (3-teacher, 균등, 기준선, 200K) | 66,667 | — | — |
| ver5 @ 200K (vjepa=0.2) | 47,619 (ver4 대비 -29%, 의도된 확장 초기 단계) | 9,524 | 1.27x |
| **ver5 @ 280K (vjepa=0.2, 최종 목표)** | **66,667 (ver4와 정확히 일치)** | 13,333 | **1.78x** |
| (기각) 균등(5-teacher) @ 350K | 70,000 | 70,000 | 9.33x |
| (기각) vjepa=0.15 @ 350K | 84,337 | 12,651 | 1.69x |

**확정 이유**: `vjepa=0.2`는 캐시 반복을 350K-균등안의 9.3x에서 1.27~1.78x 수준(사실상 "1회 안팎")으로 크게 낮추면서도, 280K까지 늘리면 CNN teacher 4개는 ver4와 **정확히 동일한 step 수**를 받는다 — "V-JEPA 캐시 과다 반복 방지"와 "기존 3-teacher와의 step-parity" 두 목표를 모두 만족하는 지점. 다만 200K 시점(1차 관찰 지점)에서는 CNN teacher가 ver4 대비 -29% 적은 step을 받는다는 점은 감안해야 함 — 이건 "teacher가 5개로 늘어난 초기 확장 단계"로 명시적으로 받아들인 것이지, 결함이 아니다.

**실행 순서**: (1) `ver5`를 200K까지 학습해 추이 관찰 → (2) 문제 없으면 같은 run을 STEPS=280000으로 이어서 재실행(자동 resume) → **ver4(200K) vs ver5(200K) vs ver5-280k(280K, CNN teacher step-parity)** 세 체크포인트로 최종 비교.

### 5.3 확정 실행 커맨드

```bash
# 기존 3-teacher 기준선 (이미 완료, 비교 대조군) -- ver4, step_200000.pt 존재
# (참고용, 재실행 불필요)

# ver5: teacher 5개, vjepa=0.2, 우선 200K까지 실행해 추이 관찰
NUM_GPUS=4 STEPS=200000 BATCH_SIZE=4 LOG_EVERY=250 EVAL_EVERY=5000 CKPT_EVERY=10000 \
./run_train.sh ver5 \
    --teachers anatomix+brains vesselfm brats voco \
    --vjepa_cache_dir /root/external_teacher_probe/vjepa_cache \
    --teacher_weights '{"anatomix+brains":1,"vesselfm":1,"brats":1,"voco":1,"vjepa":0.2}' \
    --mse_weight_overrides '{"vjepa":0.02}' \
    --convpass_peft --convpass_encoder --convpass_no_skip_alpha

# ver5 200K 결과 이상 없으면: 같은 run_name/가중치로 STEPS만 280000으로 올려서 재실행 -> 체크포인트에서 자동 resume
# (별도 run_name/디렉터리 불필요 -- step_200000.pt와 step_280000.pt가 같은 expr/ver5/checkpoints/ 아래 둘 다 남으므로
#  "ver5"와 "ver5-280k" 비교는 같은 run의 서로 다른 체크포인트 스냅샷을 가리키는 것으로 충분함)
NUM_GPUS=4 STEPS=280000 BATCH_SIZE=4 LOG_EVERY=250 EVAL_EVERY=5000 CKPT_EVERY=10000 \
./run_train.sh ver5 \
    --teachers anatomix+brains vesselfm brats voco \
    --vjepa_cache_dir /root/external_teacher_probe/vjepa_cache \
    --teacher_weights '{"anatomix+brains":1,"vesselfm":1,"brats":1,"voco":1,"vjepa":0.2}' \
    --mse_weight_overrides '{"vjepa":0.02}' \
    --convpass_peft --convpass_encoder --convpass_no_skip_alpha
```

`--mse_weight_overrides '{"vjepa":0.02}'`는 §6.6에서 측정한 V-JEPA feature norm 스케일(다른 teacher 대비 5~40배) 보정용으로 가중치 결정과 무관하게 유지한다. `--teacher_weights`의 `vjepa` 값(0.2)이 바뀌면 §5.2의 280,000이라는 목표 step 수도 같이 재계산해야 한다는 점에 유의 — 이 두 숫자는 서로 묶여 있다.

### 5.4 비교 계획

**ver4**(3-teacher, 200K) vs **ver5**(5-teacher, vjepa=0.2, 200K) vs **ver5-280k**(5-teacher, vjepa=0.2, 280K, `step_280000.pt`, CNN teacher 4개 step-parity) 세 체크포인트로 비교한다 — ver4 vs ver5(200K)는 "5-teacher 확장 초기 단계(CNN teacher는 아직 ver4보다 -29% 적은 step)에서 teacher 다양성 자체의 효과"를, ver5 vs ver5-280k는 "CNN teacher의 step 희석을 정확히 보정했을 때 효과가 유지/강화되는지"를 각각 보여주는 별도 축이므로 두 비교 모두 의미가 있다.

이 비교 실험 결과를 보고 step 수 확장 및 downstream task 튜닝으로 넘어가는 흐름은 이 리포트 범위 밖이며, 결과가 나오는 대로 별도로 다뤄야 합니다.

---

## 6. 코드 리뷰 대응 (2026-08-02 추가)

다른 Claude 인스턴스가 이 통합 작업을 검토하며 "axis-order 버그처럼 loss는 멀쩡해 보이는데 결과만 조용히 나빠지는" 종류의 문제가 더 있을 수 있다고 지적한 12개 항목(augmentation 정합성 1개 + 번호 매겨진 11개)을 전부 코드 추적·실측·실제 4-GPU mini-run으로 하나씩 검증했다. 요약표:

| # | 항목 | 결과 |
|---|---|---|
| — | augmentation 정합성 | ✅ 문제 없음 — 코드베이스 전체에 기하학적 augmentation이 아예 없음(§6.1) |
| 1 | 캐시 crop과 student 입력의 위치 일치 | ✅ 문제 없음 — 같은 npz에서 같이 로드, 재-crop 없음 |
| 2 | 캐시 padding 값 | 🔴 **실제 버그, 수정 완료** — 30K 중 20.0%(5,990개) 영향, 재추출 진행 중(§6.2) |
| 3 | pooling anisotropy | ✅ 문제 없음 — 의도된 물리 공간 등방화(§6.3) |
| 4 | float16 표현 범위 | ✅ 문제 없음 — 샘플 검사에서 이상 없음(§6.4) |
| 5 | modality 분포 편향 | ⚠️ 실재하지만 설계상 트레이드오프로 문서화(§6.5) |
| 6 | teacher 간 loss 스케일 | 🟡 **완화 조치 추가** — `--mse_weight_overrides`(§6.6) |
| 7 | projection head 초기화 | ✅ 문제 없음 — teacher별 특별 취급 없음(§6.7) |
| 8 | episodic sampler DDP 동기화 | ✅ 문제 없음 — 코드+실측 모두 확인(§6.8) |
| 9 | VoCo encoder-only DDP 오버헤드 | ✅ 문제 없음 — 오히려 더 빠름, 실측 확인(§6.9) |
| 10 | teacher 5개로 늘어나며 step 희석 | ⚠️ 실재하는 트레이드오프, 수치화만 함(§6.10) |
| 11 | Convpass 파라미터 비율 | ✅ 문제 없음 — 1.1%로 여전히 작음, 수치 확인(§6.11) |

### 6.1 Augmentation 정합성

캐시된 V-JEPA feature는 특정 시점(추출 당시)의 128³ crop에 고정된 결과물이라, "학습 중 student에게 들어가는 이미지가 그 시점의 crop과 다르게 변형됐다면" teacher-student가 서로 다른 입력을 놓고 비교하는 셈이 되어 조용히 깨질 수 있다는 우려였다. `flip`/`rotate`/`elastic`/`augment` 패턴으로 전체 `.py`를 훑었으나 실제 기하학적 augmentation은 어디에도 없음을 확인(오탐 2건 — docstring 참조 1건, vesselFM의 inference-config 문서가 "training augmentation과 대조적으로"라고 명시한 문구 1건). 구조적으로도 안전: 살아있는 teacher는 student와 **완전히 같은 `images` 텐서**를 공유하므로 애초에 어긋날 수 없고, V-JEPA는 이미지+feature가 추출 시점에 함께 저장되고 이후 어떤 후처리도 없이 그대로 로드되므로 self-consistent.

### 6.2 캐시 padding 값 불일치 — 실제 버그, 수정

`extract_vjepa_cache.py`의 `crop_at()`이 out-of-bounds 영역을 `cropped.min()`으로 채우고 있었는데, 실제 학습 파이프라인의 `FOMO300KPreprocessedDataset._random_crop_or_pad()`는 `0`으로 채운다. 두 값이 다르면, **같은 step에서도 어느 teacher가 뽑혔는지에 따라 student가 보는 배경-채움 관례가 달라지는** 실제 정합성 문제가 된다.

- 30,000개 전량 스캔 결과 소스 volume이 128보다 작은 축을 가진 경우가 **정확히 5,990개(20.0%)** — 이 비율만큼 padding이 실제로 발동해서 영향을 받음.
- **수정**: `constant_values=cropped.min()` → `constant_values=0`으로 변경. 영향받은 5,990개 캐시 파일을 삭제하고, 기존 4-GPU sharded 추출 스크립트(resumable, skip-if-exists)를 재실행해 해당 파일만 재생성 중 — 이 addendum 작성 시점 기준 **358/5,990 완료**, 나머지는 백그라운드에서 계속 진행 중(예상 총 소요 ~3.4시간, 기존 30K 전체 추출 처리율 기준 역산).
- **다음 확인 필요**: 재추출이 30,000/30,000 완료되면 5,990개 파일에서 padding 경계 픽셀이 더 이상 `cropped.min()` 아티팩트를 보이지 않는지 재점검할 것 (§7 Pending 참고).

### 6.3 Pooling anisotropy — 의도된 설계, 버그 아님

V-JEPA의 네이티브 토큰화 자체가 이미 물리 공간상 비등방적이다: 512px로 4배 업스케일된 H/W축은 토큰 하나가 물리적으로 4복셀, 업스케일이 없는 Z/tubelet축은 토큰 하나가 물리적으로 2복셀에 대응한다. `extract_vjepa_cache.py`의 비대칭 pooling 커널(`(2,1,1)` / `(4,2,2)`)은 이 기존 비등방성을 **상쇄해서 세 축 모두 물리적으로 동일한 stride(4 또는 8복셀/토큰)를 만드는** 계산이다 — 직접 산술로 재확인, 왜곡이 아니라 등방화가 목적임을 확인.

### 6.4 float16 표현 범위 — 이상 없음

30개 캐시 파일 × 4개 layer를 샘플링해 `isfinite`, min/max, per-channel std를 확인. 전부 finite, 죽은 채널(std=0) 없음, 값 범위도 대략 -26~+12로 fp16이 표현 가능한 범위에 여유 있게 들어감.

### 6.5 Modality 분포 편향 — 실재하지만 설계상 트레이드오프

- 전체 corpus(약 3,000-sample 추정): DWI 39.2% > T1w 28.9% > ... > FLAIR 7.8%
- 30K 캐시: T1w 40.5% > DWI 32.7%(순위 역전), FLAIR ~4.9%(거의 절반으로 감소)

water-fill stratified sampling이 **dataset 단위**로 균형을 맞추도록 설계돼 있어서(dataset마다 modality 구성비가 다름) 나타나는 부작용이다. modality 단위까지 균형을 맞추려면 dataset×modality 이중 stratification으로 재설계해야 하는데, 이건 독자적으로 바꿀 사안이 아니라 사용자 판단이 필요한 설계 결정이라 이번엔 "알려진 특성"으로 문서화만 하고 코드는 바꾸지 않았다.

### 6.6 Teacher 간 loss 스케일 — 완화 조치 추가

동일한 실제 데이터에 대해 실측한 teacher별 feature L2 norm(voxel당 평균):

| teacher | L2 norm 범위 |
|---|---|
| anatomix+brains | 2.8 ~ 9.4 |
| vesselfm | 4.0 ~ 8.0 |
| brats | 1.4 ~ 3.6 |
| voco | 0.7 ~ 2.6 |
| **vjepa** | **25 ~ 31** |

기존 CNN teacher 4개 사이의 ~2-6배 스케일 차는 손실 함수 설계(`(1-cosine).mean() + mse_weight*MSE`, cosine 항은 scale-invariant)가 이미 감내하도록 되어 있었지만, V-JEPA는 **다른 CNN teacher 대비 5~40배** 커서 MSE 항(scale-invariant 아님)이 V-JEPA가 뽑힌 step에서 과도하게 지배할 위험이 있었다. `run_pretrain.py`에 `--mse_weight_overrides`(JSON dict, 기본값 빈 dict = 기존 동작과 100% 동일) 플래그를 추가해 teacher별로 `mse_weight`를 오버라이드할 수 있게 했다 — 학습 루프와 `evaluate()` 양쪽에 동일하게 반영, 예: `--mse_weight_overrides '{"vjepa": 0.02}'`. (`evaluate()`는 현재 살아있는 teacher만 순회하므로 V-JEPA 자체의 holdout eval loss는 구조적으로 계산되지 않는다 — §4 검증 3번과 동일한 기존 제약, §6.6 fix와는 별개.)

### 6.7 Projection head 초기화 — 문제 없음

`ProjectionHeads.__init__`을 코드로 추적한 결과 teacher별 특별 취급 없이 단일 `nn.Conv3d` 인스턴스화 지점 하나만 존재 — 모든 teacher가 동일한 (기본 PyTorch) 초기화를 거친다.

### 6.8 Episodic sampler의 DDP 동기화 — 문제 없음, 실측으로 재확인

`EpisodicTeacherSampler`는 모든 rank에서 **동일한 `seed=args.seed`**로 생성되고, `sampler.sample()`은 메인 학습 루프에서 rank 조건 없이 매 step 무조건 호출된다(`run_pretrain.py` 주석에도 "Same seed on every rank -> every rank samples the SAME teacher per step"으로 이미 명시돼 있었음) — `random.Random`은 순수하게 호출 횟수에 의해 결정되는 결정적 시퀀스이므로, 모든 rank가 매 step 정확히 같은 teacher를 뽑는다는 걸 코드로 재확인.

실측: `voco`/`brats`/`vesselfm` 3-teacher 구성으로 4-GPU 40-step mini-run을 직접 돌려봤고(`expr/ddp_smoke/`), NCCL watchdog/hang/desync 없이 40/40 정상 종료·체크포인트 저장까지 확인. 실현된 샘플링 빈도는 brats 13/40(32.5%), vesselfm 19/40(47.5%), voco 8/40(20%) — 균등 가중치 기준 기대값(각 33.3%)과의 편차는 n=40이라는 표본 크기에서 나올 수 있는 정상적인 이항분포 노이즈로, 20만 step 규모의 실제 학습에서는 대수의 법칙으로 수렴한다.

### 6.9 VoCo encoder-only teacher의 multi-GPU DDP 오버헤드 — 문제 없음

같은 4-GPU mini-run 로그에서 teacher별 step 시간을 비교: `voco`(decoder skip, `find_unused_parameters=True`로 큰 미사용 파라미터 집합 존재) **269~758ms**, `brats`/`vesselfm`(전체 encoder+decoder forward/backward) **494~1505ms**(간헐적으로 최대 4396ms, 단 이 구간은 동시에 돌아가던 V-JEPA 캐시 재추출 작업과 GPU를 공유한 영향이 섞여 있어 절대값은 참고용). VoCo가 단일 프로세스 스모크 테스트(§3.2, ~134ms)보다는 느리지만(멀티프로세스 DDP 오버헤드 + GPU 공유 영향), **decoder를 도는 teacher들보다 항상 더 빠르다** — `find_unused_parameters=True`의 동기화 비용이 decoder skip으로 얻는 계산 절감을 상쇄하지 않음을 실측으로 확인.

### 6.10 Teacher 5개로 늘어나며 생기는 step 희석 — 수치화

균등 가중치 기준, teacher 1개가 받는 기대 step 수는 `총 step / teacher 수`다:

| teacher 수 | 200,000 step 기준 teacher당 기대 step |
|---|---|
| 3 (기존) | ~66,667 |
| 5 (VoCo+V-JEPA 추가) | 40,000 |

즉 총 step 예산이 그대로면 teacher 1개가 받는 업데이트 횟수가 **40% 감소**한다. "5-teacher가 3-teacher보다 낫다/못하다"를 비교할 때 이 희석 효과를 반드시 감안해야 한다 — 공정한 비교를 하려면 (a) 5-teacher 쪽 총 step을 비례해서 늘리거나, (b) 최소한 이 confound를 명시하고 해석할 것을 권장. `--teacher_weights`로 특정 teacher에 가중치를 더 줄 수도 있지만 그건 희석을 없애는 게 아니라 재분배하는 것이므로 별개 문제.

### 6.11 Convpass 파라미터 비율 — 실측 확인

`build_student(teachers=..., convpass=True, convpass_encoder=True, skip_alpha=False)`로 실제 인스턴스화해서 비교:

| teacher 수 | 전체 파라미터 | Convpass 파라미터 | 비율 |
|---|---|---|---|
| 3 (기존, ver4 기준선) | 32,014,985 | 214,761 | 0.671% |
| 5 (VoCo+V-JEPA 추가) | 32,158,159 | 357,935 | 1.113% |

teacher가 늘면서 teacher별 Convpass adapter가 그만큼 추가되지만(3→5는 파라미터 수로는 +67%), 전체 대비로는 여전히 1.1%대로 작다 — 학습 불안정이나 최적화 부담을 걱정할 수준은 아님.

### 6.12 남은 작업

1. ~~백그라운드에서 진행 중인 padding-fix 재추출(§6.2) 완료 확인~~ **완료** — 4개 shard 전부 정상 종료(`done` 합계 1515+1496+1547+1432=**5,990/5,990**, `FAILED` 로그 0건), `vjepa_cache/` 디렉터리 파일 수도 정확히 30,000개로 확인.
2. ~~재추출된 5,990개 파일에 대해 §6.4 스타일의 샘플 무결성 재점검~~ **완료** — 영향받은 인덱스 중 8개를 무작위 샘플링해 재확인. 모든 샘플에서 image 값이 `[0, 1]` 범위(asparagus z-norm의 `rescale_intensity(out_range=(0,1))` 관례와 일치)이고, padding이 발동한 경계 영역은 정확히 `0.0`으로 채워져 있음을 확인(과거처럼 `cropped.min()`이 남아있는 흔적 없음). `crop_start`가 0/경계 근처인 샘플일수록 0-비율이 높게 나타나는 것도 예상과 일치(예: idx=1628, 19802는 zero_frac≈0.40, idx=10519, 12288은 경계에서 덜 벗어나 zero_frac≈0.047).
3. ~~`--mse_weight_overrides`를 실제 5-teacher run에서 켜서 V-JEPA-step의 total loss가 다른 teacher들과 같은 자릿수인지 한 번 더 확인~~ **완료** — §7의 실제 `ver5` 200K 런에서 `--mse_weight_overrides '{"vjepa":0.02}'`를 켜고 학습, V-JEPA step의 loss가 다른 teacher와 같은 범위(0.09대)로 유지됨을 확인.

---

## 7. `ver5` 200K 실행 — 결과 및 크래시 조사 (2026-08-07)

### 7.1 최종 결과

`ver5`(5-teacher, `vjepa=0.2` 가중치, `--convpass_peft --convpass_encoder --convpass_no_skip_alpha`) 200,000 step 완주, `step_200000.pt` 저장 완료.

| step | anatomix+brains | vesselfm | brats | voco |
|---|---|---|---|---|
| 5,000 | 0.230 | 0.253 | 0.203 | 0.341 |
| 50,000 | 0.091 | 0.115 | 0.092 | 0.169 |
| 100,000 | 0.088 | 0.106 | 0.077 | 0.154 |
| 150,000 | (진행 중 기록, 아래 크래시 조사 참고) | | | |
| **200,000 (최종)** | **0.078** | **0.098** | **0.063** | **0.142** |

4개 teacher 전부 학습 전 구간에 걸쳐 단조 감소, 200K 시점까지 발산/붕괴 징후 없음. V-JEPA는 살아있는 teacher가 아니라 `evaluate()`에서 구조적으로 제외되지만(§6.6 참고), train-time loss는 다른 teacher와 동일 자릿수(0.09대)를 유지.

### 7.2 반복되는 크래시 원인 조사

200K 도달까지 총 **21회** 크래시(`run_train.sh`의 `MAX_RETRIES` 재시도로 전부 자동 복구, 매번 `checkpoints/last.pt`에서 재개해 실질적 진행 손실 없음 — 재개 직후 eval 수치가 크래시 직전과 거의 동일함을 반복 확인). 사용자 요청으로 근본 원인을 추적:

**주 원인 (크래시 대부분, `OpType=BROADCAST` + `Invalid mt19937 state` 시그니처)**: `accelerate` 1.14.0 소스 직접 추적 결과, `Accelerator.__init__`의 기본값 `rng_types=["generator"]` 때문에 **accelerate가 prepare한 모든 DataLoader가 `iter()`로 새로 시작할 때마다(즉 매 epoch 경계마다) `torch.distributed.broadcast`로 각 rank의 sampler generator 상태를 동기화**한다(`accelerate/data_loader.py` `DataLoaderShard.__iter__` → `accelerate/utils/random.py` `synchronize_rng_states`). 이 broadcast는 4개 rank 전부가 정확히 같은 시점에 호출해야 하는 collective인데, 한 rank라도 그 시점에 다른 코드를 실행 중이면 나머지가 NCCL의 600초 watchdog에 걸려 죽는다. `ver5`는 메인 코퍼스 loader와 V-JEPA 캐시 loader **2개**를 모두 accelerate로 prepare하기 때문에(기존 3-teacher 버전엔 1개뿐), epoch 경계 이벤트 자체가 더 많다 — 실측: 200K step 기준 메인 loader ~10회, vjepa loader ~5회, 합계 ~15회의 재시작 이벤트 vs 관측된 크래시 21회로 같은 자릿수.

**수정**: 이 동기화가 우리 학습 루프의 정합성에 필요 없다는 점(각 rank는 애초에 서로 다른 데이터 shard를 보도록 설계돼 있고, 크래시-재개 후 loss/eval 값이 정확히 일치하지 않는 것에서도 이미 cross-rank RNG 연속성에 의존하지 않는다는 게 실증됨)을 확인하고, `run_pretrain.py`의 `Accelerator(...)` 생성자에 `rng_types=[]`를 추가해 이 broadcast 자체를 제거(`synchronize_rng_states([])`는 빈 리스트를 순회하는 no-op, 코드로 확인). 라이브 run에 step 190,000에서 재적용, 이후 200,000 완주까지 **크래시 0회**.

**부 원인 (별도 메커니즘, `OpType=ALLREDUCE`)**: 수정 적용 직전 마지막 2회 크래시는 BROADCAST가 아니라 **그래디언트 동기화용 ALLREDUCE**(매 step 정상적으로 호출되는 collective) 타임아웃이었고, 공교롭게도 둘 다 정확히 step 199,750(재개 지점 190,000의 +9,750)에서 발생 — episodic sampler가 매 프로세스 재시작마다 `seed=0`으로 새로 초기화돼(체크포인트에 RNG 상태가 저장되지 않음) 같은 체크포인트에서 재개하면 teacher/데이터 선택 시퀀스가 결정론적으로 재생되므로, "특정 위치의 느린 샘플"이 매번 같은 지점에서 걸릴 수 있다는 가설을 세우고 재검증. `rng_types=[]` 수정 후 같은 step_190000.pt에서 다시 재생했을 때는 199,750을 무사히 통과해 200,000까지 완주 — **재현 안 됨, 일회성 우연(다른 rank의 일시적 느려짐 등)으로 결론**, 결정론적 "불량 샘플" 가설은 기각.

**참고**: 200K 완주 직후 `training complete` 로그는 찍혔으나 프로세스 자체가 GPU 100% 점유 상태로 멈춰 정상 종료되지 않는 현상 발견(체크포인트 파일 자체는 손상 없이 완전히 저장됨, md5 확인) — accelerate/NCCL의 프로세스그룹 정리(teardown) 단계에서 멈춘 것으로 추정, 수동으로 프로세스 트리를 종료해 정리. 다음 `--steps 280000` 재개에는 영향 없음(체크포인트 기반 재개라 프로세스 종료 방식과 무관).

### 7.3 다음 단계

§5.3의 계획대로 같은 `ver5` run을 `--steps 280000`으로 이어서 재실행 — CNN teacher 4개(anatomix+brains/vesselfm/brats/voco)가 ver4와 동일한 66,667 step/teacher에 도달하는 지점까지 확장.
