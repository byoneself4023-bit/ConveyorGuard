# ConveyorGuard 앙상블 모델 결과

## 1. 목표

> **ML 4종 + DL 2종 앙상블로 최고 성능 달성**

### 핵심 질문
- 앙상블이 개별 모델보다 성능이 좋은가?
- Soft Voting vs Weighted Voting vs Stacking 중 어떤 전략이 최적인가?
- ML과 DL의 앙상블이 시너지를 내는가?

---

## 2. 앙상블 구성

### 개별 모델 6종

| 유형 | 모델 | Test Acc | 비고 |
|------|------|----------|------|
| ML | LightGBM | 96.89% | 🥇 최고 |
| ML | XGBoost | 96.70% | |
| ML | CatBoost | 96.46% | |
| ML | RandomForest | 95.58% | |
| DL | Baseline CNN | 92.72% | 30프레임 |
| DL | Tuned CNN | 87.75% | 10프레임 (속도 최적화) |

### 앙상블 전략 3종

| 전략 | 설명 |
|------|------|
| **Soft Voting** | 모든 모델 확률 균등 평균 |
| **Weighted Voting** | Val Acc 비례 가중 평균 |
| **Stacking** | Meta-learner (Logistic Regression) |

---

## 3. 앙상블 결과

### 전체 순위

| 순위 | 모델 | Val Acc | Test Acc | Test F1 |
|------|------|---------|----------|---------|
| 🥇 | LightGBM | 96.98% | **96.89%** | 96.89% |
| 🥇 | **Stacking** | 96.91% | **96.89%** | 96.89% |
| 🥉 | XGBoost | 97.04% | 96.70% | 96.70% |
| 4 | Weighted Voting | - | 96.70% | 96.71% |
| 5 | Soft Voting | - | 96.64% | 96.64% |
| 6 | CatBoost | 96.59% | 96.46% | 96.46% |
| 7 | RandomForest | 96.53% | 95.58% | 95.63% |
| 8 | Baseline CNN | 93.24% | 92.72% | 92.80% |
| 9 | Tuned CNN | 88.10% | 87.75% | 88.13% |

### 가중치 분포 (Weighted Voting)

```
xgboost:      0.1707
lightgbm:     0.1706
randomforest: 0.1698
catboost:     0.1699
baseline:     0.1640
tuned:        0.1550  ← 가장 낮음
```

**ML 4종이 비슷한 가중치, DL 2종은 낮은 가중치**

---

## 4. 앙상블 효과 분석

### 결론: 앙상블 효과 거의 없음

| 비교 | 결과 |
|------|------|
| LightGBM 단독 | 96.89% |
| Stacking | 96.89% (동일) |
| Weighted Voting | 96.70% (하락) |
| Soft Voting | 96.64% (하락) |

### 원인 분석

| 원인 | 설명 |
|------|------|
| **ML 성능 포화** | ML 4종이 이미 95~97%로 충분히 높음 |
| **DL 성능 부족** | DL 2종이 87~93%로 ML 대비 낮음 |
| **다양성 부족** | DL이 ML과 다른 패턴을 잡지 못함 |

### 앙상블이 효과적이려면?

```
좋은 앙상블 조건:
1. 개별 모델 성능이 비슷해야 함
2. 모델들이 서로 다른 패턴을 학습해야 함
3. 틀리는 샘플이 달라야 상호 보완 가능

현재 상황:
- ML: 96~97% (비슷)
- DL: 87~93% (낮음)
→ DL이 보완하기보다 성능을 깎아먹음
```

---

## 5. Stacking 상세

### Meta-features 구성

```
6개 모델 × 4개 클래스 = 24개 meta-features
```

| 모델 | 클래스 0 | 클래스 1 | 클래스 2 | 클래스 3 |
|------|----------|----------|----------|----------|
| XGBoost | prob_0 | prob_1 | prob_2 | prob_3 |
| LightGBM | prob_0 | prob_1 | prob_2 | prob_3 |
| ... | ... | ... | ... | ... |

### Meta-Learner

```python
LogisticRegression(max_iter=1000, random_state=42)
```

### Stacking 결과

| 지표 | 값 |
|------|-----|
| Val Accuracy | 96.91% |
| Test Accuracy | 96.89% |
| Test F1 | 96.89% |

**LightGBM 단독과 동일 → Stacking도 효과 없음**

---

## 6. 클래스별 성능 (Stacking 기준)

| 클래스 | Precision | Recall | F1-Score | Support |
|--------|-----------|--------|----------|---------|
| 정상(0) | 0.99 | 0.99 | 0.99 | 788 |
| 경미(1) | 0.93 | 0.94 | 0.94 | 371 |
| 중간(2) | 0.96 | 0.94 | 0.95 | 361 |
| **심각(3)** | 0.98 | **0.98** | 0.98 | 88 |

### 심각(3) 탐지 성능 🎯

```
88개 중 86개 정확히 탐지 (Recall 98%)
→ 단 2개만 놓침!
```

---

## 7. Confusion Matrix (LightGBM)

```
              Predicted
           정상  경미  중간  심각
Actual
  정상      781    7    0    0
  경미        8  349   14    0
  중간        0   17  342    2
  심각        0    0    2   86
```

### 주요 혼동 패턴

| 혼동 | 빈도 | 해석 |
|------|------|------|
| 중간 → 경미 | 17 | 경계 불확실 |
| 경미 → 중간 | 14 | 경계 불확실 |
| 정상 → 경미 | 7 | 초기 열화 과탐지 (안전) |
| **심각 → 중간** | **2** | 거의 없음 ✅ |

---

## 8. 인사이트

### 이 프로젝트에서 배운 점

| 항목 | 내용 |
|------|------|
| **앙상블은 만능 아님** | 개별 모델 다양성이 부족하면 효과 없음 |
| **ML vs DL** | 정형 데이터에서 ML이 여전히 강력 |
| **피처 엔지니어링** | 시계열 통계 피처가 핵심 |
| **최적 모델** | LightGBM 단독이 최고 (간단 + 빠름 + 정확) |

### 면접 포인트

> "앙상블을 시도했지만, ML이 이미 충분히 좋고 DL이 다른 패턴을 보완하지 못해서 성능 향상이 없었습니다. 이를 통해 앙상블이 만능이 아니며, 개별 모델의 다양성이 중요하다는 것을 배웠습니다."

---

## 9. 저장된 파일

| 파일 | 용도 |
|------|------|
| ensemble_results.csv | 전체 모델 성능 비교 |
| ensemble_model.pkl | Stacking meta-learner |

---

## 10. 파이프라인 연결

```
00_EDA
  └── 클래스 불균형 발견
        ↓
01_preprocess
  └── 윈도우 생성 + 세션 Split
        ↓
02_baseline_cnn
  └── DL Baseline → Test 92.72%
        ↓
03_ml_baseline
  └── LightGBM → Test 96.89%
        ↓
04_dl_tuning
  └── Optuna + Ablation Study
        ↓
05_llm_comparison
  └── LLM 3종 + LangGraph
        ↓
06_ensemble (현재)
  ├── Soft Voting → 96.64%
  ├── Weighted Voting → 96.70%
  └── Stacking → 96.89% (LightGBM과 동일)
        ↓
07_final_comparison
  └── 다차원 비교 + 최종 결론
```

---

## 핵심 요약

### 🏆 최종 결과

| 항목 | 값 |
|------|-----|
| Best 모델 | LightGBM (단독) |
| Test Accuracy | **96.89%** |
| 심각 Recall | **98%** |
| 앙상블 효과 | 없음 (동일 성능) |

### 📊 앙상블 전략 비교

| 전략 | Test Acc | vs LightGBM |
|------|----------|-------------|
| Soft Voting | 96.64% | -0.25%p |
| Weighted Voting | 96.70% | -0.19%p |
| Stacking | 96.89% | 동일 |

### 💡 핵심 교훈

1. **앙상블 ≠ 무조건 성능 향상**
2. **모델 다양성이 핵심** (성능 차이만으론 부족)
3. **단순한 모델이 최고일 수 있음** (LightGBM)
4. **실험하고 분석하는 과정 자체가 가치**

### 다음 단계

- 07_final_comparison: 전체 파이프라인 종합 비교 + 최종 결론
