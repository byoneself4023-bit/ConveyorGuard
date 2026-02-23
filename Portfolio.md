# ConveyorGuard — 이송장치 열화 예측 AI

> 반도체 제조 라인 이송장치의 열화 상태를 예측하는 예지보전 AI 시스템

------

## 프로젝트 개요

| 항목   | 내용                                                  |
| ------ | ----------------------------------------------------- |
| 유형   | 개인 프로젝트                                         |
| 역할   | 전체 설계·구현 (데이터 분석 → 모델링 → API → 데모)    |
| 도메인 | 반도체 제조, 예지보전 (Predictive Maintenance)         |
| 데이터 | AIHub #71802 — 센서(8ch) + 열화상(60×80) + 환경       |
| 규모   | API 19개, 노트북 8개, 모델 13개 비교, ~5,500줄        |

------

## 기술 스택

| 레이어             | 기술                                          |
| ------------------ | --------------------------------------------- |
| Demo UI            | Streamlit, Plotly                              |
| ML API             | FastAPI, Uvicorn (Port 8000)                   |
| LLM Service        | FastAPI, Uvicorn (Port 8001)                   |
| Deep Learning      | PyTorch (Multimodal CNN + Transformer)         |
| LLM 진단           | LangGraph, Gemini 2.5 Flash                   |
| RAG                | FAISS, Sentence-Transformers                   |
| ML Models (실험)   | LightGBM, XGBoost, CatBoost, RandomForest     |
| 인프라             | Docker, Docker Compose, Git                    |

------

## 핵심 어필 포인트

### 1. 가설 검증 실패를 인정한 정직한 연구

"멀티모달 딥러닝이 이길 것이다" → **LightGBM이 3.65%p 앞섬**

- Ablation Study: 열화상은 +0.52%p만 기여
- Optuna 튜닝해도 DL은 90.48% 한계
- 결론을 데이터로 증명하고 ML 채택
- 면접 포인트: *"최신 기술이 항상 정답은 아니다"*

### 2. LangGraph 멀티에이전트 진단 시스템

```
Analyzer → Diagnostician → Reviewer → Finalize
                ↑____________REVISE___|
```

- 품질 검토(APPROVE/REVISE) 루프 포함
- Gemini 2.5 Flash + JSON 스키마 강제 출력
- FAISS 기반 유사 케이스 RAG (정비 이력 5건)
- 전체 파이프라인 91.5초 (REVISE 1회 포함)

### 3. End-to-End 자동화 파이프라인

```
센서 삽입 → ML 예측 → 열화 ≥ 2면 LLM 호출 → 진단 저장 → Alert 생성
```

- 조건부 LLM 호출로 비용 최적화 (정상/경미는 ML만)
- Fallback 휴리스틱: 모델 미로드 시 NTC/PM2.5/CT 기반 점수제 분류

### 4. 64-Feature 멀티모달 피처 엔지니어링

| Feature 유형                       | 수  |
| ---------------------------------- | --- |
| 센서 (mean/std/max/min/last/diff × 8) | 48  |
| 열화상 (frame mean/max/trend)      | 7   |
| 환경 (mean/std/last × 3)           | 9   |

### 5. 연구 스토리텔링 (Streamlit 데모)

- 8단계 실험 여정을 인터랙티브 Plotly 시각화
- "DL → ML 전환" 과정을 데이터로 설명
- 면접관이 30초 만에 연구 맥락 파악 가능

------

## 아키텍처

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Demo UI    │     │   ML API    │◀───▶│ LLM Service │
│ (Streamlit) │     │  (FastAPI)  │     │ (LangGraph) │
│  Port 8501  │     │  Port 8000  │     │  Port 8001  │
└─────────────┘     └──────┬──────┘     └──────┬──────┘
                          │                    │
                   ┌──────▼──────┐      ┌──────▼──────┐
                   │  LightGBM   │      │ Gemini 2.5  │
                   │  (96.89%)   │      │ + FAISS RAG │
                   └─────────────┘      └─────────────┘
```

------

## 성과 숫자

### 모델 성능

| 지표                          | 값                              |
| ----------------------------- | ------------------------------- |
| Best Accuracy (LightGBM)      | 96.89%                          |
| Best F1 Score                 | 0.9689                          |
| 심각 Recall                   | 98%                             |
| DL Baseline (CNN+Transformer) | 93.24%                          |
| 속도 차이                     | 800× (LightGBM 2.7s vs DL 2,178s) |
| 모델 크기                     | 3.1MB (LightGBM) vs 13.5MB (DL) |

### 코드베이스

| 항목              | 수량                            |
| ----------------- | ------------------------------- |
| Python 파일       | 33개                            |
| Jupyter 노트북    | 8개                             |
| 비교 실험 모델 수 | 13개 (ML 9 + DL 2 + Ensemble 2) |
| API 엔드포인트    | 19개 (ML 14 + LLM 5)           |
| 총 코드           | ~5,500줄                        |

### 데이터셋

| 항목               | 값        |
| ------------------ | --------- |
| 전체 프레임        | 111,870개 |
| 세션               | 341개     |
| 학습 윈도우        | 10,473개  |
| 클래스 불균형 비율 | 6.34:1    |

------

## 실험 노트북

| 노트북              | 핵심 실험                                |
| ------------------- | ---------------------------------------- |
| 00_eda              | 111,870 프레임 분석, NTC 상관계수 0.793  |
| 01_preprocess       | 30-frame 슬라이딩 윈도우, 세션 기반 분리 |
| 02_baseline_cnn     | CNN+Transformer 93.24% 달성              |
| 03_ml_baseline      | 8개 ML 모델 비교 → LightGBM 96.89%      |
| 04_dl_tuning        | Optuna 8 trials + 모달리티 Ablation      |
| 05_llm_comparison   | Gemini vs Gemma-3 vs Qwen2.5 비교       |
| 06_ensemble         | Stacking/Voting (개선 없음 증명)         |
| 07_final_comparison | 13개 모델 최종 비교                      |

------

## 시연 및 링크

- [시연 영상 + 프로젝트 소개](https://byoneself4023-bit.github.io)
- [GitHub Repository](GitHub URL 입력)