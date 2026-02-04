# ConveyorGuard ML Baseline 모델 결과

## 1. 피처 엔지니어링

### 멀티모달 → 64개 통계 피처로 변환

| 모달리티 | 원본 Shape | 추출 피처 | 개수 |
|----------|-----------|----------|------|
| 센서 | (30, 8) | mean, std, max, min, last, diff | 48개 |
| 열화상 | (30, 60, 80) | mean, std, max, max_mean, last_mean, last_max, trend | 7개 |
| 외부환경 | (30, 3) | mean, std, last | 9개 |
| 총 | - | - | 64개 |

### 핵심 아이디어

- ML은 이미지를 직접 처리 못함 → 통계값(온도 max, mean 등)으로 요약
- EDA에서 "열화상 max 온도가 상태와 상관관계 높음" 확인 → 피처로 활용

### ML의 한계

- 통계값만 사용 → 공간적 패턴(어디가 뜨거운지) 학습 불가
- 열화상의 열 분포 패턴은 CNN만 학습 가능

---

## 2. 모델 성능 비교

### 전체 결과 (Test Accuracy 기준 정렬)

| Model | Val Acc | Test Acc | Val F1 | Test F1 | Train Time |
|-------|---------|----------|--------|---------|------------|
| LightGBM | 96.98% | 96.89% | 96.69% | 96.43% | 2.7초 |
| XGBoost | 97.04% | 96.70% | 96.98% | 96.28% | 3.8초 |
| CatBoost | 96.59% | 96.46% | 96.37% | 96.01% | 21.5초 |
| RandomForest | 96.53% | 95.58% | 96.36% | 95.19% | 3.7초 |
| DecisionTree | 93.44% | 92.97% | 93.21% | 92.00% | 0.6초 |
| CNN+Transformer (DL Baseline) | 93.24% | 92.72% | 93.00% | 93.00% | 2178초 |
| KNN (k=5) | 89.90% | 89.12% | 90.20% | 88.28% | 0.0초 |
| SVM (RBF) | 86.87% | 87.75% | 87.07% | 87.01% | 1.0초 |
| LogisticRegression | 86.42% | 87.31% | 86.08% | 86.81% | 2.0초 |

---

## 3. Baseline 비교: ML vs DL (튜닝 전)

### 성능 비교 (Baseline 기준)

| 모델 | Test Acc | 학습 시간 | 비고 |
|------|----------|----------|------|
| LightGBM | 96.89% | 2.7초 | ML Best |
| CNN+Transformer | 92.72% | 2178초 | DL Baseline (튜닝 전) |

### 현재 ML이 높은 이유

| 이유 | 설명 |
|------|------|
| 피처 엔지니어링 성공 | 64개 통계 피처가 핵심 정보 잘 요약 |
| 데이터 규모 | 7,311개 → ML에 충분 |
| 열화상 특성 | max/mean 온도가 핵심 → 통계값만으로도 유효 |
| NTC 상관도 0.79 | 센서 통계만으로도 높은 예측력 |

### DL 튜닝 필요성

| ML | DL |
|----|-----|
| 하이퍼파라미터 설정됨 | Baseline (튜닝 안 함) |
| 통계 피처만 사용 | 이미지 공간 패턴 학습 가능 |

→ 04_dl_tuning에서 DL 최적화 후 재비교 필요

---

## 4. Best Model: LightGBM

### Classification Report (Test Set)

| 클래스 | Precision | Recall | F1-Score | Support |
|--------|-----------|--------|----------|---------|
| 정상(0) | 0.99 | 0.99 | 0.99 | 788 |
| 경미(1) | 0.94 | 0.94 | 0.94 | 371 |
| 중간(2) | 0.96 | 0.95 | 0.95 | 361 |
| 심각(3) | 0.98 | 0.98 | 0.98 | 88 |

### Confusion Matrix

```
              Predicted
           정상  경미  중간  심각
Actual
  정상      781    7    0    0
  경미        8  349   14    0
  중간        0   17  342    2
  심각        0    0    2   86
```

### 핵심 성과

- 심각 Recall 98%: 88개 중 86개 정확히 탐지
- 정상→경미, 경미→중간 혼동 약간 있지만 심각 탐지 거의 완벽

---

## 5. Feature Importance (XGBoost)

### Top 10 피처

| 순위 | 피처 | Importance | 해석 |
|------|------|------------|------|
| 1 | sensor_NTC_last | 0.23 | 마지막 온도 (핵심) |
| 2 | sensor_CT2_diff | 0.085 | 전류 변화량 |
| 3 | sensor_CT2_std | 0.067 | 전류 변동성 |
| 4 | sensor_NTC_max | 0.057 | 최고 온도 |
| 5 | sensor_CT2_max | 0.045 | 전류 최대값 |
| 6 | sensor_CT2_mean | 0.042 | 전류 평균 |
| 7 | sensor_CT1_max | 0.041 | 전류 최대값 |
| 8 | sensor_CT1_diff | 0.035 | 전류 변화량 |
| 9 | sensor_PM10_min | 0.035 | 미세먼지 |
| 10 | sensor_PM10_std | 0.030 | 미세먼지 변동 |

### EDA와 일치 확인

| EDA 상관도 | Feature Importance |
|------------|-------------------|
| NTC 0.79 (1위) | NTC_last 0.23 (1위) |
| CT2 0.38 (2위) | CT2_diff 0.09 (2위) |
| CT1 0.34 (3위) | CT1_max, CT1_diff (상위) |

EDA 인사이트가 모델에서도 검증됨

---

## 6. ML vs DL 비교 요약

| 모델 유형 | 장점 | 단점 | 적합한 상황 |
|-----------|------|------|-------------|
| Traditional ML | 초고속, 해석 용이 | 정확도 제한적 | 빠른 POC, 비교 기준선 |
| 앙상블 ML | 빠름, 해석 가능, 가벼움 | 이미지 직접 처리 불가 | 센서 위주 분석, 빠른 추론 |
| DL | 이미지 직접 처리, 공간 패턴 학습 | 학습 시간 김, GPU 필요 | 멀티모달 융합, 이미지 패턴 분석 |

### 이 프로젝트의 특성

- 멀티모달 예지보전 → 이미지 직접 처리하는 DL이 프로젝트 취지에 맞음
- 열화상의 공간적 패턴(어디가 뜨거운지)은 CNN만 학습 가능
- ML은 통계값만 사용 → 이미지 정보 손실

### 다음 단계: DL 튜닝 후 재비교

- DL 튜닝 후 ML보다 높아지면 → 멀티모달 융합의 가치 입증
- DL 튜닝 후에도 ML이 높으면 → 이 데이터셋에서는 피처 엔지니어링이 정답
- 어느 쪽이든 의미 있는 결론 도출 가능

---

## 7. 저장된 파일

| 파일 | 크기 | 용도 |
|------|------|------|
| lightgbm_model.pkl | 3.13 MB | Best 모델 |
| xgboost_model.pkl | 2.11 MB | 2등 모델 |
| catboost_model.pkl | 2.97 MB | 3등 모델 |
| randomforest_model.pkl | 11.64 MB | 4등 모델 |
| ml_scaler.pkl | 2 KB | 피처 스케일러 |
| ml_comparison_results.csv | 411 B | 성능 비교표 |
| feature_importance.csv | 1.7 KB | 피처 중요도 |
| ml_comparison.html | 4.36 MB | 성능 비교 시각화 |
| ml_confusion_matrix.html | 4.36 MB | Confusion Matrix |
| feature_importance.html | 4.36 MB | 피처 중요도 시각화 |

---

## 8. 파이프라인 연결

```
00_EDA
  └── NTC 상관도 0.79 발견 → 핵심 피처 확인
        ↓
01_preprocess
  └── 윈도우 생성 + 세션 Split
        ↓
02_baseline_cnn
  └── DL Baseline → Test 92.72%
        ↓
03_ml_baseline (현재)
  ├── 피처 엔지니어링 (64개)
  ├── LightGBM → Test 96.89%
  └── Feature Importance로 EDA 검증
        ↓
04_dl_tuning (다음 단계)
  └── Optuna로 DL 하이퍼파라미터 최적화
        ↓
06_ensemble / 07_final_comparison
  └── ML vs DL (튜닝 후) 최종 비교
```

---

## 핵심 요약

### 현재 Best Model (Baseline 기준)

- LightGBM: Test Accuracy 96.89%, 학습 시간 2.7초

### 심각 탐지 성능

- Recall 98%: 88개 중 86개 탐지

### 현재까지의 인사이트

- EDA 기반 피처 엔지니어링의 힘
- 잘 설계된 64개 피처로 높은 성능 달성
- Feature Importance가 EDA 결과와 일치 → 검증됨

### 다음 단계

- DL Baseline은 튜닝 안 된 상태
- 멀티모달 프로젝트 취지 → 이미지 공간 패턴 학습하는 DL 최적화 필요
- 튜닝 후 ML vs DL 재비교로 최종 결론 도출