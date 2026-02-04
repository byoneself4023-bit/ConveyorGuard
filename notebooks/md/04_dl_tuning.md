# ConveyorGuard DL Tuning (Optuna 최적화) 결과

## 1. 목표

> **DL Baseline (92.72%)을 Optuna로 튜닝하여 ML Baseline (96.89%)에 근접/초과**

### 핵심 질문
- Optuna 하이퍼파라미터 최적화로 DL 성능 개선 가능한가?
- 멀티모달 융합이 정말 효과가 있는가? (Ablation Study)
- FiLM(외부환경 조건화)이 성능에 기여하는가?

---

## 2. Optuna 하이퍼파라미터 탐색

### 탐색 공간

| 파라미터 | 범위 | 최적값 |
|----------|------|--------|
| embed_dim | [128, 256] | **256** |
| num_heads | [4, 8] | **4** |
| num_layers | [1, 2] | **2** |
| dropout | [0.1, 0.2, 0.3] | **0.1** |
| lr | 1e-4 ~ 1e-3 | **0.000196** |
| weight_decay | 1e-5 ~ 1e-3 | **5.4e-05** |

### 탐색 설정

| 항목 | 값 |
|------|-----|
| Sampler | TPE (Tree-structured Parzen Estimator) |
| Pruner | MedianPruner |
| Trial 수 | 15 (타임아웃 30분) |
| 완료 Trial | 8개 (4개 Pruned) |

### Trial 결과

| Trial | Val Acc | 상태 |
|-------|---------|------|
| 0 | 88.74% | 완료 |
| 1 | 89.83% | 완료 |
| **2** | **90.48%** | **Best** |
| 3 | 90.09% | 완료 |
| 4-7 | - | Pruned (조기 종료) |

**Pruning 효과**: 비효율적인 Trial 조기 종료 → GPU 시간 절약

---

## 3. 최적 모델 학습 결과

### 학습 설정

| 항목 | 값 |
|------|-----|
| Batch Size | 64 (GPU당 32) |
| Optimizer | AdamW |
| Scheduler | CosineAnnealing |
| Loss | CrossEntropy (weighted) |
| Early Stopping | Patience 7 |
| Max Epochs | 30 |

### 학습 곡선

```
Epoch  1/30 │ Val: 81.0% ✅ Best!
Epoch  2/30 │ Val: 84.3% ✅ Best!
Epoch  4/30 │ Val: 86.2% ✅ Best!
Epoch  6/30 │ Val: 89.1% ✅ Best!
Epoch  8/30 │ Val: 89.3% ✅ Best!
Epoch 10/30 │ Val: 90.0% ✅ Best!
Epoch 11/30 │ Val: 90.3% ✅ Best! ← 최고 성능
Epoch 18/30 │ Early stopping (no improve: 7)
```

### 최종 성능

| 지표 | 값 |
|------|-----|
| **Best Val Accuracy** | **90.35%** |
| **Test Accuracy** | **87.75%** |
| 학습 시간 | 10.0분 |

---

## 4. Ablation Study (핵심!) 🔬

### 멀티모달 효과 검증

| 구성 | 센서 | 이미지 | FiLM | Val Acc |
|------|:----:|:------:|:----:|---------|
| Sensor Only | ✅ | - | - | 89.12% |
| Image Only | - | ✅ | - | 69.56% |
| Sensor + Image | ✅ | ✅ | - | 89.64% |
| **Full (+ FiLM)** | ✅ | ✅ | ✅ | **90.35%** |

### 시각화

```
Sensor Only                    → 89.12%  ████████████████████████████████████████████
Image Only                     → 69.56%  ██████████████████████████████████
Sensor + Image                 → 89.64%  ████████████████████████████████████████████
Sensor + Image + FiLM (Full)   → 90.35%  █████████████████████████████████████████████
```

### 인사이트

| 발견 | 의미 |
|------|------|
| Image Only 69.56% | 열화상만으론 불충분 |
| Sensor Only 89.12% | **센서가 핵심 정보원** |
| +Image → 89.64% | 멀티모달 효과 +0.5%p |
| +FiLM → 90.35% | 외부환경 조건화 +0.7%p |

**→ 멀티모달 + FiLM 효과 실험적으로 입증!**

---

## 5. 클래스별 상세 평가 (Test Set)

| 클래스 | Precision | Recall | F1-Score | Support |
|--------|-----------|--------|----------|---------|
| 정상(0) | 0.97 | 0.85 | 0.90 | 788 |
| 경미(1) | 0.70 | 0.91 | 0.79 | 371 |
| 중간(2) | 0.95 | 0.90 | 0.92 | 361 |
| **심각(3)** | 0.87 | **0.97** | 0.91 | 88 |

### 핵심 성과: 심각(3) 탐지 🎯

```
실제 심각 88개 중 85개 정확히 탐지 (Recall 97%)
→ 단 3개만 놓침!
```

---

## 6. Confusion Matrix 분석

```
              Predicted
           정상  경미  중간  심각
Actual
  정상      666  122    0    0
  경미       20  336   15    0
  중간        3   21  324   13
  심각        1    0    2   85
```

### 주요 혼동 패턴

| 혼동 | 빈도 | 해석 |
|------|------|------|
| 정상 → 경미 | 122 | 초기 열화 과탐지 (안전한 방향) |
| 경미 → 정상 | 20 | 경미 일부 미탐지 |
| 중간 → 심각 | 13 | 중간-심각 경계 불확실 |
| **심각 → 중간** | **2** | 거의 없음 ✅ |

**심각을 놓치는 경우가 거의 없음 → 안전한 예지보전 가능!**

---

## 7. GPU 최적화 기법

| 기법 | 적용 | 효과 |
|------|------|------|
| DataParallel | T4 x2 | 학습 속도 향상 |
| AMP (Mixed Precision) | autocast + GradScaler | 메모리 절약 + 속도 향상 |
| cuDNN benchmark | enabled | 연산 최적화 |
| 이미지 서브샘플링 | 30 → 10 프레임 | 3배 속도 향상 |
| Optuna Pruning | MedianPruner | 비효율 Trial 조기 종료 |

---

## 8. DL Baseline vs DL Tuned 비교

| 항목 | Baseline (02) | Tuned (04) | 변화 |
|------|---------------|------------|------|
| Val Accuracy | 93.24% | 90.35% | -2.9%p |
| Test Accuracy | 92.72% | 87.75% | -5.0%p |
| 심각 Recall | 98% | 97% | -1%p |
| 학습 시간 | 36.3분 | 10.0분 | **3.6배 빠름** |
| 최적화 방식 | 수동 | Optuna | 자동화 |

### 성능 차이 분석

| 원인 | 설명 |
|------|------|
| 이미지 서브샘플링 | 30 → 10 프레임 (속도↑, 정보↓) |
| Optuna 탐색 범위 | 제한된 범위 내 탐색 |
| Trial 수 | 8개 완료 (더 많으면 개선 가능) |

---

## 9. ML vs DL 비교 (현재까지)

| 모델 | Test Acc | 심각 Recall | 학습 시간 |
|------|----------|-------------|----------|
| LightGBM (ML) | **96.89%** | 98% | 2.7초 |
| DL Baseline | 92.72% | 98% | 36.3분 |
| DL Tuned | 87.75% | 97% | 10.0분 |

### 현재 결론

- ML이 여전히 높음 (피처 엔지니어링 효과)
- DL Tuned는 속도 최적화에 집중 → 정확도 trade-off
- 앙상블(06)에서 ML + DL 결합 시 시너지 기대

---

## 10. 저장된 파일

| 파일 | 크기 | 용도 |
|------|------|------|
| tuned_model.pt | 16 MB | 튜닝된 모델 + Ablation 결과 |
| optuna_study.pkl | 16 KB | Optuna 탐색 히스토리 |
| tuned_confusion_matrix.html | 4 MB | Confusion Matrix 시각화 |

### tuned_model.pt 내용

```python
{
    'model_state_dict': ...,
    'best_val_acc': 90.35,
    'test_acc': 87.75,
    'best_params': {...},
    'ablation_results': {
        'Sensor Only': 89.12,
        'Image Only': 69.56,
        'Sensor + Image': 89.64,
        'Sensor + Image + FiLM (Full)': 90.35
    },
    'model_config': {...}
}
```

---

## 11. 파이프라인 연결

```
00_EDA
  └── 클래스 불균형 발견 → 가중치 필요성
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
04_dl_tuning (현재)
  ├── Optuna 최적화 → Test 87.75%
  ├── Ablation Study → 멀티모달 효과 입증
  └── GPU 최적화 (속도 3.6배 향상)
        ↓
06_ensemble
  └── ML + DL 앙상블
        ↓
07_final_comparison
  └── 다차원 비교 + 최종 결론
```

---

## 핵심 요약

### 🔬 Ablation Study 결론

| 구성 | 효과 |
|------|------|
| 센서만 | 89.12% (핵심 정보원) |
| 이미지만 | 69.56% (단독 사용 부적합) |
| 센서+이미지 | +0.5%p (멀티모달 효과) |
| +FiLM | +0.7%p (외부환경 조건화) |

**→ "왜 멀티모달인가?" 실험적으로 답변 완료!**

### 🎯 심각 탐지 성능

| 지표 | 값 | 의미 |
|------|-----|------|
| 심각 Recall | **97%** | 88개 중 85개 탐지 |
| 심각 → 중간 오분류 | 2개 | 거의 놓치지 않음 |

### ⚡ GPU 최적화 성과

| 지표 | Baseline | Tuned |
|------|----------|-------|
| 학습 시간 | 36.3분 | 10.0분 |
| 속도 향상 | - | **3.6배** |

### 다음 단계

- 06_ensemble: ML + DL 결합으로 최종 성능 극대화
- 07_final_comparison: 다차원 비교표 + 레이더 차트 + 최종 결론
