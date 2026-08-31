# Task 6/7 Pre-registration (커밋 전 첫 실행 없음)

작성일: 2026-08-11, 실제 embedding 추출 실행 전에 작성됨.

## 0. 로컬 프록시의 한계 (실행 전에 명시)

Task 3(뇌연령, 494명 — 스펙의 200명 추정은 부정확했음, 실측치로 정정)에는 **나이 라벨만
있고 성별/기관 등 범주형 인구통계가 전혀 없음** (Task 1~4 원본 데이터 전체를 확인했으나
없음). 따라서:

- **Task 6(cohort 분리도)은 로컬 프록시로 전부 검증한다** — `selected_cohort`를 나이
  사분위(q1~q4, 4클래스, 균형 잡힘: 128/124/122/120명)로 구성.
- **Task 7(fairness)은 로컬 검증을 스킵한다** (사용자 결정, 2026-08-11). FEATURE2로 쓸
  실제 변수가 없어 억지로 대리 변수를 만들면 신뢰도 낮은 숫자만 나옴. Task 7 예측은
  아래 §2에 기록하되, **실제 검증은 validation 제출로만** 한다.
- 부작용: `selected_cohort`(나이 사분위)와 FEATURE1(나이)이 사실상 같은 변수라, 이
  프록시는 "나이 관련 구조적 변화를 얼마나 잘 인코딩하는가"를 재는 것이지, 실제
  Task 6의 (미공개) cohort 정의와 반드시 일치한다는 보장은 없음. **분리도의 상대적
  순위(경로 간 비교)는 유효하지만, 절대 수치는 실제 Task 6과 다를 수 있음.**

## 1. Task 6 (로컬 프록시: 나이 사분위 분리도) 예측 순위

예상 순위 (강함 → 약함), 근거 포함:

1. **anatomix+brains** — 정상 해부학 구조 인코딩에 특화된 teacher. 뇌 나이는 피질 두께,
   뇌실 크기 등 "정상 해부학적 변이"로 나타나므로 이 teacher의 표현 공간과 직접 일치할
   것으로 예상. **1위 예상.**
2. **voco** — contrastive/global objective 기반(OpenMind). FOMO25에서 hybrid/global
   objective가 classification에 유리했던 경향과 부합. 전역적 구조 요약이 age quartile
   분류 같은 global classification 태스크에 잘 맞을 것. **1~2위 예상.**
3. **neutral** (gate=0, teacher-비특이적 공유 표현) — 특정 teacher에 편향되지 않은
   범용 표현. 나이 관련 신호를 어느 정도는 담고 있겠지만 어떤 teacher에도 최적화되지
   않았으므로 중위권 예상.
4. **brats** — 병변/종양 국소화에 최적화. 정상 노화 패턴(전역적, 미만성)과는 다른
   국소적 이상 탐지 방향이라 age quartile 분리에는 약할 것으로 예상.
5. **vesselfm** — 혈관/관형 구조 특화. 혈관 노화(굴곡, 석회화)가 나이와 상관은 있지만
   전체 뇌 구조 대비 지나치게 국소적인 특징이라 quartile 분류엔 약할 것.
6. **vjepa** — 자연영상 사전학습 + distillation weight가 낮았던 이력([[project_vjepa_teacher_roadmap]]
   참조: "제한적 No-go"로 유보됨). 의료영상 특이적 신호를 가장 적게 담았을 것으로 예상,
   **최하위 예상.**

**불확실한 부분**: 스펙 문서에 인용된 PCA spread 수치("ver3 12.75 < ver2 18.42 <
ver4 25.64")는 스펙 §3.1의 "ver3는 IN affine이 전 stage에 걸려 있어 차이가 훨씬
크다"는 서술과 **표면적으로 모순**된다 (수치상 ver3의 spread가 가장 작은데, 서술은
ver3의 teacher 간 차이가 가장 크다고 함). 이 프로젝트의 실제 PCA 분석 산출물을 다시
확인하기 전까지는 이 불일치를 그대로 두고, **결과 해석 시 이 수치를 맹신하지 않는다.**

## 2. Task 7 (fairness) 예측 — 로컬 미검증, validation 제출로만 확인

로컬 프록시가 없으므로 방향성만 기록 (검증은 validation 제출 이후):

- **anatomix**: 해부학 구조를 잘 담을수록 인구통계(특히 연령)와 상관된 정보도 잘
  담을 가능성이 높음 → Task 6 유리 예상과 반대로 **Task 7 불리 예상** (스펙 §4의
  일반 원칙과 동일).
- **neutral**: 특정 teacher에 편향되지 않으므로 인구통계 편향도 상대적으로 적을 것
  → **Task 7 유리 예상**, 가장 방어 가능한 경로.
- 나머지 teacher는 로컬 신호가 없어 강한 예측을 하지 않음 — validation 결과를 보고
  사후에 §REPORT.md에 기록.

## 3. 전처리 A(native) vs B(iso) 예측

**B(iso, 사전학습과 정합)가 A(native)보다 나을 것으로 예측.** 근거: frozen backbone이
사전학습 때 본 입력 분포(1mm iso spacing, foreground-masked znorm)에서 벗어나면 표현
품질이 저하된다는 것이 이 프로젝트의 반복된 가설이며, Task 1에서 이 문제를 발견하고
전체 전처리를 재작업했다(iso1mm redo, 진행 중). **단, Task 1의 현재 진행 중인 실험은
"pretrained-init vs from-scratch Convpass" 축을 보는 것이지 "native vs iso 전처리"를
직접 대조하는 control이 아니므로, 이번 Task 6/7 실험이 "사전학습 정합" 가설의 첫 번째
독립적/직접적 검증이 된다.** Task 1의 이전 결과(무신호, [[project_fomo26_task1_pretrain_comparison]])와
이번 결과가 다르게 나올 수 있다는 점을 열어둔다 — backbone이 완전히 frozen인 Task 6/7은
Task 1(Convpass도 학습되는)보다 분포 불일치 효과가 더 크게 나타날 조건이라는 게 스펙
§3의 주장이다.

## 4. Test 선택 규칙 (스펙 §5 그대로 채택, 여기 재확정)

```
1. 로컬 프록시와 validation에서 1위가 일치하고, 2위와 명확히 차이(ovr_auroc 0.02 초과)
   → 그 경로 선택
2. 차이가 작거나(0.02 이내) 로컬과 validation의 1위가 불일치
   → neutral (gate=0) 선택
3. Task 6 최적과 Task 7 최적이 다르면 → Task 6 우선
```

## 5. 실행 전 체크리스트

- [ ] `gate=0` forward 출력이 ver1 구조 forward와 bit-identical한지 `torch.equal`로 검증
      (`torch.allclose` 아님 — 정확히 같아야 함)
- [ ] 이 파일이 첫 embedding 추출 실행 전에 커밋되어 있는지 확인
