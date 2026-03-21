# RUNNER UI를 FairGRAPE 가중치 프루닝에 적용 가능성 검증 실험 설계

## 연구 질문

> **"RUNNER의 UI(Unfairness Importance)를 FairGRAPE의 가중치 프루닝 기준으로 그대로 적용할 수 있는가?"**

RUNNER는 뉴런(채널) 단위로 불공정 중요도를 계산하고,
FairGRAPE는 가중치(연결선) 단위로 프루닝을 수행합니다.

이 실험은 RUNNER의 UI 수식을 가중치 단위로 분해했을 때,
실제 ΔEO에 미치는 영향과 상관관계가 있는지를 측정합니다.
추가로 UI와 성능 중요도(I_w)의 상관관계도 측정하여,
UI가 성능과 독립적으로 공정성을 잡아내는지 확인합니다.

---

## 배경

### RUNNER UI의 원래 정의 (뉴런/채널 단위)

논문 수식 (6):

UI_채널c = (∂GapEO/∂w_c × w_c)² = sum over (in_ch, kH, kW) of (grad[i,c,h,w] × w[i,c,h,w])²

채널 내 모든 가중치의 (gradient × weight)²를 합산한 값입니다.
RUNNER는 이 값을 채널 단위로 집계해서 top-k 불공정 뉴런을 선택합니다.

### 가중치 단위로 분해하면

sum하기 전 개별 값이 곧 가중치 단위 UI입니다:

UI_w(i,c,h,w) = (∂GapEO/∂w × w)²

즉 RUNNER 수식 안에 이미 가중치 단위 중요도가 내재되어 있습니다.
채널로 sum하지 않으면 FairGRAPE처럼 가중치 단위 중요도로 쓸 수 있습니다.

### 성능 중요도 (I_w) 정의

FairGRAPE 수식 (5), (6) 기반:

I_w = (L(D, θ) - L(D, θ'|w=0))² ≈ (g_w × w)²

전체 데이터셋 D 기준으로 가중치 w를 제거했을 때 loss 변화량의 제곱입니다.
그룹 구분 없이 전체 loss에 대한 gradient를 사용합니다.

RUNNER UI와 수식 형태는 동일하지만 gradient의 출처가 다릅니다:
- RUNNER UI: EO gap(공정성)에 대한 gradient
- 성능 중요도 I_w: 전체 loss(성능)에 대한 gradient

---

## 가설

**H1 (주 가설):**
가중치 단위 RUNNER UI와 해당 가중치 제거 후 실제 ΔEO 변화량 사이에
유의미한 양의 상관관계가 존재한다.
→ RUNNER UI를 FairGRAPE 프루닝 기준으로 사용할 수 있음

**H2 (독립성 가설):**
RUNNER UI와 성능 중요도 I_w 사이의 상관관계가 낮다.
→ UI가 성능과 독립적으로 공정성만을 잡아내고 있음
→ UI 기반 프루닝 시 성능 손실 없이 공정성 개선 가능하다는 근거

---

## 실험 대상

| 항목 | 값 |
|------|-----|
| 모델 | ResNet-18 (CelebA, pretrained) |
| 데이터셋 | CelebA (test split 사용) |
| 태스크 | 얼굴 속성 분류 (wavy hair 또는 attractive) |
| Sensitive attribute | Gender (Male=1, Female=0) |
| 분석 레이어 | Layer 1 (1차 검증), Layer 4 (2차 본 분석) |
| 분석 단위 | 개별 가중치 w[out_c, in_c, kH, kW] |
| 가중치 수 | Layer 1: 64×64×3×3 = 36,864개 |

---

## 실험 절차

### Phase 0: 사전 준비

모델을 준비하고 가중치 제거 없이 원본 모델의 기준값을 측정합니다.
- 기준 ΔEO: sum 기반으로 계산 (|TPR0 - TPR1| + |FPR0 - FPR1|)
- 기준 Loss: 전체 데이터셋 기준 cross entropy loss

---

### Phase 1: Layer 1 검증 실험 (약 2~3시간)

#### Step 1. 가중치 단위 RUNNER UI 계산 (공정성 중요도)

서브그룹별 배치를 구성합니다 (EO 기준):
- gender=0, label=0 → X00
- gender=0, label=1 → X01
- gender=1, label=0 → X10
- gender=1, label=1 → X11

GapEO를 sum 기반으로 계산한 뒤 backward를 수행합니다.
Layer 1 첫 번째 conv의 각 가중치에 대해 (∂GapEO/∂w × w)²를 계산합니다.
채널로 sum하지 않고 개별 가중치 단위 값을 그대로 사용합니다.
결과: Layer 1의 RUNNER UI 배열 (64×64×3×3 = 36,864개)

#### Step 2. 성능 중요도 I_w 계산

전체 데이터셋 기준 loss에 대해 backward를 수행합니다.
동일한 레이어의 각 가중치에 대해 (∂L/∂w × w)²를 계산합니다.
결과: Layer 1의 성능 중요도 배열 (동일 shape)

#### Step 3. 샘플링 전략

36,864개를 전부 하기에는 너무 오래 걸리므로 아래 전략으로 샘플링합니다:
- UI 상위 500개 + UI 하위 500개 + 랜덤 500개 = 총 약 1,500개

이렇게 하면 상위/하위/중간 분포를 고루 커버해서 상관관계를 보기에 충분합니다.

#### Step 4. 가중치 단위 실제 ΔEO 변화량 계산

샘플링된 가중치를 하나씩 0으로 마스킹한 뒤 ΔEO를 측정합니다.
마스킹 전후 ΔEO 차이가 그 가중치의 실제 공정성 영향도입니다.
마스킹 후 반드시 원래 값으로 원복합니다.

evaluate_eo에서 전체 test set 대신 랜덤 샘플 2,000장을 사용하면
속도를 10배 단축할 수 있으며 상관관계 경향성 확인에는 충분합니다.

#### Step 5. 상관관계 분석 (두 가지)

**A. RUNNER UI vs 실제 ΔEO 변화량 (H1 검증)**
- Pearson 상관계수 (선형 관계)
- Spearman 상관계수 (순위 기반)
- p-value로 통계적 유의성 확인

**B. RUNNER UI vs 성능 중요도 I_w (H2 검증)**
- Pearson 상관계수 (선형 관계)
- Spearman 상관계수 (순위 기반)
- p-value로 통계적 유의성 확인
- 상관관계가 낮을수록 UI가 성능과 독립적으로 공정성을 잡아낸다는 증거

#### Step 6. 시각화

아래 플롯을 생성합니다:
- 산점도 A: RUNNER UI vs 실제 ΔEO 변화량
- 산점도 B: RUNNER UI vs 성능 중요도 I_w
- 분위수별 bar plot: UI 구간(D1~D10)별 평균 ΔEO 영향도

---

### Phase 2: Layer 4 본 분석 (Phase 1 결과 유의미할 때)

Phase 1과 동일한 절차를 Layer 4 (512×512×3×3 = 2,359,296개)에 적용합니다.
샘플링 전략: UI 상위 1,000개 + 하위 1,000개 + 랜덤 1,000개 = 3,000개

---

## 예상 소요 시간

| 단계 | 소요 시간 |
|------|-----------|
| Phase 0: 기준 ΔEO + Loss 측정 | 약 5분 |
| Phase 1 Step 1~2: UI + I_w 계산 | 약 2분 |
| Phase 1 Step 3~4: ΔEO 변화량 (1,500개 × 샘플 2,000장) | 약 1~2시간 |
| Phase 1 Step 5~6: 분석 및 시각화 | 약 10분 |
| **Phase 1 합계** | **약 2~3시간** |
| Phase 2 (Layer 4) | 약 8~12시간 |

---

## 결과 해석 기준

### H1: RUNNER UI vs 실제 ΔEO 상관관계

| Spearman r | 해석 |
|:----------:|------|
| 0.7 이상 | RUNNER UI가 강한 공정성 프루닝 기준 ✅ |
| 0.4 ~ 0.7 | 중간 수준, 보완적으로 활용 가능 |
| 0.2 ~ 0.4 | 약한 상관, 단독 사용은 한계 |
| 0.2 미만 | FairGRAPE 대체로 부적합 |

### H2: RUNNER UI vs 성능 중요도 I_w 상관관계

| Spearman r | 해석 |
|:----------:|------|
| 0.2 미만 | UI가 성능과 독립적 → 공정성만 선택적으로 잡아냄 ✅ |
| 0.2 ~ 0.5 | 부분적 겹침, 성능 손실 일부 발생 가능 |
| 0.5 이상 | UI가 성능 중요도와 유사 → 프루닝 시 성능 손실 우려 |

---

## 연구 의의

**H1 높음 + H2 낮음 (이상적인 결과):**
- RUNNER UI가 공정성을 잘 잡으면서 성능과 독립적
- FairGRAPE에 RUNNER UI를 적용하면 성능 손실 없이 공정성 개선 가능
- "RUNNER-guided FairGRAPE" 방법론 제안 가능

**H1 높음 + H2 높음:**
- UI가 공정성과 성능을 동시에 잡음
- 프루닝 시 성능 손실 우려 → 추가 보정 필요

**H1 낮음:**
- UI가 ΔEO를 잘 근사하지 못함
- 집계 방식 변형(채널 평균 vs 최대값 등) 또는 다른 proxy 탐색 필요

---

## 파일 구조

```
project_root/
├── correlation_weight_analysis.py    # 메인 분석 스크립트
└── correlation_results/
    ├── runner_ui_vs_eo_layer1.png        # UI vs ΔEO 산점도 (Layer 1)
    ├── runner_ui_vs_perf_layer1.png      # UI vs I_w 산점도 (Layer 1)
    ├── runner_ui_vs_eo_layer4.png        # UI vs ΔEO 산점도 (Layer 4)
    ├── runner_ui_vs_perf_layer4.png      # UI vs I_w 산점도 (Layer 4)
    └── correlation_results.json         # 수치 결과
```

---

## 주의사항

1. gradient 누적 방지: UI 및 I_w 계산 전 반드시 model.zero_grad() 호출
2. 가중치 원복 필수: 마스킹 후 반드시 원래 값으로 복원
3. EO 계산 일관성: UI 계산과 ΔEO 측정 모두 sum 기반 EO 사용
4. 샘플링 재현성: 랜덤 seed 고정하여 재현 가능하게
5. 모델 eval 모드: 측정 중 eval + no_grad 사용 (단, UI/I_w 계산 시 grad 필요하므로 구분)

---

## 진행 순서 요약

```
[Phase 0] 기준 ΔEO + 기준 Loss 측정
        ↓
[Phase 1] Layer 1 상관관계 분석 (약 2~3시간)
  1. RUNNER UI 계산 (EO gap 기반 gradient)
  2. 성능 중요도 I_w 계산 (전체 loss 기반 gradient)
  3. 샘플링 (상위 500 + 하위 500 + 랜덤 500)
  4. 가중치별 실제 ΔEO 변화량 측정
  5-A. RUNNER UI vs ΔEO 상관관계 (H1)
  5-B. RUNNER UI vs I_w 상관관계 (H2)
  6. 시각화
        ↓ 결과 확인
[Phase 2] Layer 4 본 분석 (Phase 1 유의미할 때)
```
