# ConveyorGuard Baseline CNN 모델 결과

## 1. 모델 아키텍처

```
센서 (30, 8)    → MLP → Transformer
                              ↓
                        Cross-Attention → FiLM(External) → Classifier → 4-Class
                              ↑
열화상 (30, 60, 80) → CNN → Transformer
```

### 구성 요소
| 모듈 | 역할 |
|------|------|
| SensorEncoder (MLP) | 센서 8채널 → 128차원 임베딩 |
| ImageEncoder (CNN) | 열화상 60×80 → 128차원 임베딩 |
| TemporalEncoder (Transformer) | 시계열 패턴 학습 |
| CrossAttentionFusion | 센서-이미지 융합 |
| FiLM | 외부환경으로 조건화 |

### 모델 스펙
| 항목 | 값 |
|------|-----|
| Total Parameters | 1,168,772 |
| Embed Dimension | 128 |
| Transformer Layers | 2 |
| Attention Heads | 4 |

---

## 2. 학습 설정

| 항목 | 값 |
|------|-----|
| Batch Size | 64 (GPU당 32) |
| Optimizer | AdamW (lr=1e-3) |
| Scheduler | CosineAnnealing |
| Loss | CrossEntropy (weighted) |
| Early Stopping | Patience 7 |
| Max Epochs | 30 |

### 클래스 가중치 (불균형 대응)
| 클래스 | 가중치 |
|--------|--------|
| 정상(0) | 0.28 |
| 경미(1) | 0.60 |
| 중간(2) | 0.61 |
| 심각(3) | **2.51** |

---

## 3. 학습 결과

### 학습 곡선
```
Epoch  1/30 │ Val: 83.1% ✅ Best!
Epoch  2/30 │ Val: 88.5% ✅ Best!
Epoch  8/30 │ Val: 89.6% ✅ Best!
Epoch 14/30 │ Val: 90.5% ✅ Best!
Epoch 16/30 │ Val: 92.0% ✅ Best!
Epoch 19/30 │ Val: 92.7% ✅ Best!
Epoch 22/30 │ Val: 92.7% ✅ Best!
Epoch 23/30 │ Val: 93.2% ✅ Best! ← 최고 성능
Epoch 30/30 │ Early stopping (no improve: 7)
```

### 최종 성능
| 지표 | 값 |
|------|-----|
| **Best Val Accuracy** | **93.24%** |
| **Test Accuracy** | **92.72%** |
| 총 학습 시간 | 36.3분 |

---

## 4. 클래스별 상세 평가 (Test Set)

| 클래스 | Precision | Recall | F1-Score | Support |
|--------|-----------|--------|----------|---------|
| 정상(0) | 0.98 | 0.93 | 0.95 | 788 |
| 경미(1) | 0.84 | 0.90 | 0.87 | 371 |
| 중간(2) | 0.91 | 0.95 | 0.93 | 361 |
| **심각(3)** | 0.92 | **0.98** | 0.95 | 88 |

### 핵심 성과: 심각(3) 탐지 🎯
```
실제 심각 88개 중 86개 정확히 탐지 (Recall 98%)
→ 단 2개만 놓침!
```

**EDA 인사이트 + 클래스 가중치 = 심각 탐지 성공!**

---

## 5. Confusion Matrix 분석

```
              Predicted
           정상  경미  중간  심각
Actual
  정상      729   51    8    0
  경미       13  333   25    0
  중간        0   11  343    7
  심각        0    0    2   86
```

### 주요 혼동 패턴
| 혼동 | 빈도 | 해석 |
|------|------|------|
| 정상 → 경미 | 51 | 초기 열화 징후 미감지 |
| 경미 → 중간 | 25 | 경계 구간 불확실 |
| 중간 → 심각 | 7 | 중간-심각 경계 |
| **심각 → 중간** | **2** | 거의 없음 ✅ |

**심각을 놓치는 경우가 거의 없음 → 안전한 예지보전 가능!**

---

## 6. 최적화 기법

| 기법 | 적용 | 효과 |
|------|------|------|
| DataParallel | T4 x2 | 학습 속도 2배 |
| AMP (Mixed Precision) | autocast + GradScaler | 메모리 절약 + 속도 향상 |
| cuDNN benchmark | enabled | 연산 최적화 |
| pin_memory | True | CPU→GPU 전송 가속 |
| persistent_workers | True | DataLoader 오버헤드 감소 |

---

## 7. 저장된 파일

| 파일 | 위치 | 용도 |
|------|------|------|
| baseline_cnn_model.pt | conveyorguard-baseline | 학습된 모델 |
| confusion_matrix.html | conveyorguard-baseline | 평가 시각화 |

---

## 8. 파이프라인 연결

```
00_EDA
  └── 클래스 불균형 발견 → 가중치 필요성
        ↓
01_preprocess
  └── 윈도우 생성 + 세션 Split + 가중치 계산
        ↓
02_baseline_cnn (현재)
  ├── 3-modal 융합 모델
  ├── 클래스 가중치 적용
  └── Test Acc 92.72% 달성
        ↓
03_ml_baseline / 04_dl_tuning
  └── 비교 실험 + 하이퍼파라미터 최적화
```

---

## 핵심 요약

### 🎯 목표 달성
> **"심각(3) 클래스를 놓치지 않는 예지보전 모델"**

| 지표 | 결과 | 의미 |
|------|------|------|
| 심각 Recall | **98%** | 88개 중 86개 탐지 |
| 심각 Precision | 92% | 오탐 낮음 |
| Overall Accuracy | 92.72% | 높은 전체 성능 |

### 성공 요인
1. **EDA 인사이트** → 윈도우 단위 분류
2. **세션 단위 Split** → 데이터 누출 방지
3. **클래스 가중치** → 심각 탐지 강화
4. **멀티모달 융합** → 센서 + 열화상 + 외부환경
