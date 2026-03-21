# Layer 1 vs Layer 4 비교 요약

## 실험 개요

- 설계서: RUNNER UI를 FairGRAPE 가중치 프루닝 기준으로 적용 가능성 검증
- 모델: ResNet-18, CelebA, wavy_hair, pretrained
- checkpoint: `paper_reproduction_results/phase0_wavy_van_seed0/checkpoints/celeba_wavy_van_eo_seed0_best.pt`
- 공통 baseline:
  - BCE loss = 0.39070937127343713
  - AP = 0.8679287748838791
  - EO gap = 0.3720535040882176
- 실제 ΔEO 측정 subset:
  - 샘플 수 = 2000
  - subset baseline EO gap = 0.3978916990899409

## 분석 대상

| 항목 | Layer 1 | Layer 4 |
|------|---------|---------|
| 대상 가중치 | backbone.layer1[0].conv1.weight | backbone.layer4[1].conv2.weight |
| shape | 64 x 64 x 3 x 3 | 512 x 512 x 3 x 3 |
| 전체 가중치 수 | 36,864 | 2,359,296 |
| 샘플링 | 상위 500 + 하위 500 + 랜덤 500 | 상위 1000 + 하위 1000 + 랜덤 1000 |
| 샘플 수 | 1,500 | 3,000 |

## H1: RUNNER UI vs 실제 ΔEO 변화량

설계서 기준:

- Spearman 0.7 이상: 강한 공정성 프루닝 기준
- 0.4 ~ 0.7: 중간 수준
- 0.2 ~ 0.4: 약한 상관
- 0.2 미만: 대체로 부적합

| 지표 | Layer 1 | Layer 4 |
|------|---------|---------|
| Pearson r | 0.6639930935134672 | 0.20876369581706133 |
| Pearson p | 2.338162620281096e-191 | 6.7839868443776155e-31 |
| Spearman r | 0.8776776324390261 | 0.13424014267346127 |
| Spearman p | 0.0 | 1.5475048627692597e-13 |
| 해석 | 강한 상관 | 매우 약한 상관 |

### H1 해석

- Layer 1:
  - Spearman 0.8777로 매우 높음
  - RUNNER UI가 실제 ΔEO 변화량을 강하게 포착함
  - 설계서 기준에서 RUNNER UI가 강한 공정성 프루닝 기준이라는 해석이 가능함
- Layer 4:
  - Spearman 0.1342로 매우 낮음
  - 개별 가중치 수준에서는 RUNNER UI가 실제 ΔEO 변화량을 잘 근사하지 못함
  - 설계서 기준에서 FairGRAPE 대체로 부적합에 가까움

## H2: RUNNER UI vs 성능 중요도 I_w

설계서 기준:

- Spearman 0.2 미만: 성능과 독립적
- 0.2 ~ 0.5: 부분적 겹침
- 0.5 이상: 성능 중요도와 유사

### 전체 가중치 기준

| 지표 | Layer 1 | Layer 4 |
|------|---------|---------|
| Pearson r | 0.6511258693566261 | 0.39453480618235603 |
| Pearson p | 0.0 | 0.0 |
| Spearman r | 0.7872056820932648 | 0.7917207354625893 |
| Spearman p | 0.0 | 0.0 |
| 해석 | 성능 중요도와 매우 유사 | 성능 중요도와 매우 유사 |

### 샘플링 가중치 기준

| 지표 | Layer 1 | Layer 4 |
|------|---------|---------|
| Pearson r | 0.6660718136850223 | 0.31521414293953237 |
| Pearson p | 5.688245710138594e-193 | 3.4348478226288973e-70 |
| Spearman r | 0.9440424788565194 | 0.9380055211117245 |
| Spearman p | 0.0 | 0.0 |
| 해석 | 성능 중요도와 매우 강하게 겹침 | 성능 중요도와 매우 강하게 겹침 |

### H2 해석

- Layer 1과 Layer 4 모두 H2는 지지되지 않음
- RUNNER UI가 성능 중요도 I_w와 강하게 겹치므로, 현재 결과만으로는 UI가 성능과 독립적인 공정성 지표라고 보기 어려움
- 따라서 UI 기반 프루닝이 성능 손실 없이 공정성만 선택적으로 개선한다는 근거는 확보되지 않음

## 종합 비교

| 질문 | Layer 1 | Layer 4 |
|------|---------|---------|
| UI가 실제 ΔEO를 잘 설명하는가 | 예, 매우 강함 | 아니오, 매우 약함 |
| UI가 성능 중요도와 독립적인가 | 아니오 | 아니오 |
| 설계서 기준 결론 | 공정성 프루닝 기준으로는 유망하지만 성능 독립성은 부족 | 공정성 프루닝 기준으로도 약하고 성능 독립성도 부족 |

## 결론

1. Layer 1은 H1이 매우 강하게 성립한다.
   - RUNNER UI가 실제 ΔEO 변화량과 높은 상관을 보인다.
   - 따라서 Layer 1 수준에서는 RUNNER UI를 공정성 중심 프루닝 기준으로 검토할 가치가 있다.

2. Layer 4는 H1이 약하다.
   - RUNNER UI와 실제 ΔEO 변화량의 연결이 약하다.
   - 깊은 레이어의 개별 가중치 수준에서는 UI가 직접적인 공정성 영향도를 잘 반영하지 못할 수 있다.

3. 두 레이어 모두 H2는 성립하지 않는다.
   - UI와 성능 중요도 I_w가 강하게 겹친다.
   - 즉 현재 실험 결과는 RUNNER UI가 공정성과 성능을 분리해서 포착한다고 보기 어렵다.

4. 현재 실험이 시사하는 가장 보수적인 해석은 다음과 같다.
   - RUNNER UI는 얕은 레이어에서는 실제 공정성 영향도를 잘 포착할 수 있다.
   - 하지만 그 신호는 성능 중요도와도 강하게 얽혀 있다.
   - 깊은 레이어로 갈수록 가중치 단위 proxy로서의 품질은 크게 약화된다.

## 관련 결과 파일

- Layer 1 JSON: [correlation_results.json](correlation_results.json)
- Layer 1 UI vs ΔEO 플롯: [runner_ui_vs_eo_layer1.png](runner_ui_vs_eo_layer1.png)
- Layer 1 UI vs I_w 플롯: [runner_ui_vs_perf_layer1.png](runner_ui_vs_perf_layer1.png)
- Layer 1 decile 플롯: [runner_ui_decile_deltaeo_layer1.png](runner_ui_decile_deltaeo_layer1.png)
- Layer 4 JSON: [../correlation_results_layer4/correlation_results.json](../correlation_results_layer4/correlation_results.json)
- Layer 4 UI vs ΔEO 플롯: [../correlation_results_layer4/runner_ui_vs_eo_layer4.png](../correlation_results_layer4/runner_ui_vs_eo_layer4.png)
- Layer 4 UI vs I_w 플롯: [../correlation_results_layer4/runner_ui_vs_perf_layer4.png](../correlation_results_layer4/runner_ui_vs_perf_layer4.png)
- Layer 4 decile 플롯: [../correlation_results_layer4/runner_ui_decile_deltaeo_layer4.png](../correlation_results_layer4/runner_ui_decile_deltaeo_layer4.png)