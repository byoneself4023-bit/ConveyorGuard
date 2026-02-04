# ConveyorGuard

> AI 기반 제조현장 이송장치(OHT/AGV) 예지보전 시스템

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red)
![Next.js](https://img.shields.io/badge/Next.js-15-black)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100-green)

## 프로젝트 개요

ConveyorGuard는 제조현장의 이송장치(OHT/AGV) 열화 상태를 **멀티모달 AI**로 예측하여 예방 정비를 지원하는 B2B 시스템입니다.

### 핵심 기능
- **멀티모달 예측**: 센서(8종) + 열화상 + 환경 데이터 융합 분석
- **4-Class 분류**: 정상 / 경미 / 중간 / 심각 열화 상태 예측
- **LLM 진단**: 자연어 기반 원인 분석 및 조치 권고
- **실시간 모니터링**: 센서 트렌드 시각화 + 알림 시스템

### 차별화 포인트
- **멀티모달 AI**: 센서 + 열화상 + 환경 데이터 융합 → 복합 고장 패턴 감지
- **LLM 진단**: 숫자가 아닌 "왜"를 설명하는 자연어 리포트
- **저비용**: 상용 솔루션 대비 98% 비용 절감 (오픈소스 + API)

---

## 시스템 아키텍처

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Frontend   │────▶│   ML API    │◀───▶│ LLM Service │
│  (Next.js)  │     │  (FastAPI)  │     │  (Gemini)   │
│  Port 3000  │     │  Port 8000  │     │  Port 8001  │
└─────────────┘     └──────┬──────┘     └─────────────┘
                          │
                   ┌──────▼──────┐
                   │  ML Model   │
                   │   XGBoost   │
                   └─────────────┘
```

---

## 기술 스택

| 분류 | 기술 |
|------|------|
| **Frontend** | Next.js 15, TypeScript, TailwindCSS, Recharts |
| **ML API** | FastAPI, PyTorch, Python 3.11 |
| **LLM** | Google Gemini 2.5 Flash, LangChain, LangGraph |
| **ML Models** | XGBoost, LightGBM, RandomForest, CatBoost |
| **Deep Learning** | Multimodal CNN + Transformer |
| **Database** | Supabase (PostgreSQL) |
| **Data** | AIHub #71802 (센서 + 열화상 + 환경) |
| **Infra** | Docker, Git |

---

## 데이터셋

**AIHub #71802**: 이송장치 열화 예지를 위한 멀티모달 데이터

| 데이터 | 형식 | 설명 |
|--------|------|------|
| 센서 | CSV | NTC, PM1.0, PM2.5, PM10, CT1~4 (8개) |
| 열화상 | BIN | 120×160 픽셀 |
| 환경 | JSON | 온도, 습도, 조도 |
| 라벨 | JSON | 열화 상태 (0~3) |

### 데이터 규모 (EDA 기반)

| 항목 | 값 |
|------|-----|
| 전체 세션 | 341개 |
| 전체 프레임 | 111,870개 |
| 클래스 불균형 | 6.34:1 (정상:심각) |

| 클래스 | 프레임 수 | 비율 |
|--------|-----------|------|
| 정상 | 54,928 | 49.1% |
| 경미 | 24,081 | 21.5% |
| 중간 | 24,191 | 21.6% |
| 심각 | 8,670 | 7.8% |

### 핵심 인사이트
- **NTC 센서** 상관도 0.793으로 가장 중요한 피처
- **외부환경**(온도/습도/조도)은 상관도 <0.06 → 보조 피처로만 사용
- **심각(3)** 클래스는 세션 전체가 아닌 특정 구간에서만 발생 → 윈도우 기반 분류 필수

---

## 모델 아키텍처

### ML Pipeline (XGBoost)
```
Raw Data (30 timesteps)
    │
    ▼
┌─────────────────────────────────────────┐
│  Feature Engineering (64 features)      │
│  ├── Sensor: mean, std, max, min, last, diff (48개)  │
│  ├── Image: mean, std, max, trend (7개)              │
│  └── External: mean, std, last (9개)                 │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────┐
│    XGBoost      │ → 4-Class 분류
│   Classifier    │
└─────────────────┘
```

### DL Pipeline (CNN + Transformer)
```
Sensor (30, 8)   → MLP Encoder → TemporalEncoder ─┐
                                                   ├→ CrossAttention → Classifier → 4-Class
Image (30,60,80) → CNN Encoder → TemporalEncoder ─┘
```

---

## 프로젝트 구조

```
ConveyorGuard/
├── conveyorguard-api/     # ML API (FastAPI)
│   ├── app/
│   │   ├── main.py        # FastAPI 엔트리포인트
│   │   ├── db.py          # Supabase 클라이언트
│   │   ├── routers/       # 장비, 센서, 진단, 알림, 통계 API
│   │   ├── core/          # 모델 로더, 전처리
│   │   └── services/      # ML 파이프라인, 시뮬레이터
│   └── Dockerfile
│
├── llm-service/           # LLM 진단 서비스
│   └── app/
│       ├── main.py
│       ├── agents/        # LangGraph 진단 에이전트
│       ├── core/          # Gemini, 프롬프트, RAG
│       └── tools/         # 진단 도구
│
├── frontend/              # Next.js 프론트엔드
│   └── src/
│       ├── app/
│       │   ├── page.tsx           # 대시보드
│       │   └── equipment/[id]/    # 장비 상세
│       ├── components/
│       │   ├── dashboard/         # StatusCard, SensorGauge 등
│       │   ├── layout/            # AppShell, Header, Sidebar
│       │   └── ui/                # Toast, Skeleton
│       └── lib/api.ts
│
├── ml-service/            # 모델 학습 코드
│   ├── models/            # 학습된 모델 (.pkl)
│   └── outputs/           # 시각화 결과
│
├── notebooks/             # Kaggle 학습 노트북
│   ├── 00_eda.ipynb               # 탐색적 데이터 분석
│   ├── 01_preprocess.ipynb        # EDA 기반 전처리
│   ├── 02_baseline_cnn.ipynb      # CNN+Transformer 베이스라인
│   ├── 03_ml_baseline.ipynb       # ML 모델 비교 (XGBoost 등)
│   ├── 04_dl_tuning.ipynb         # DL 하이퍼파라미터 튜닝
│   ├── 05_llm_comparison.ipynb    # LLM 진단 비교
│   ├── 06_ensemble.ipynb          # 앙상블 모델
│   └── 07_final_comparison.ipynb  # 최종 비교 분석
│
├── data/
│   ├── models/            # 공유 모델 파일
│   └── processed/         # EDA 결과 (eda_metadata, eda_summary)
│
├── docs/
│   ├── ARCHITECTURE.md
│   ├── DEVELOPMENT.md
│   └── eda/               # EDA 시각화 결과
│
├── docker-compose.yml
└── README.md
```

---

## 실행 방법

### 1. ML API 실행
```bash
cd conveyorguard-api
# .env 파일에 SUPABASE_URL, SUPABASE_KEY 설정 필요
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 2. LLM Service 실행
```bash
cd llm-service
# .env 파일에 GEMINI_API_KEY 설정 필요
uvicorn app.main:app --host 0.0.0.0 --port 8001
```

### 3. Frontend 실행
```bash
cd frontend
npm install
npm run dev
```

### 4. 접속
- Frontend: http://localhost:3000
- ML API Docs: http://localhost:8000/docs
- LLM API Docs: http://localhost:8001/docs

---

## 학습 노트북

| 노트북 | 설명 | 환경 |
|--------|------|------|
| `00_eda.ipynb` | 탐색적 데이터 분석 | Kaggle GPU T4 x2 |
| `01_preprocess.ipynb` | EDA 기반 전처리 + 윈도우 생성 | Kaggle GPU T4 x2 |
| `02_baseline_cnn.ipynb` | CNN+Transformer 베이스라인 | Kaggle GPU T4 x2 |
| `03_ml_baseline.ipynb` | ML 모델 4종 비교 | Kaggle GPU T4 x2 |
| `04_dl_tuning.ipynb` | Optuna DL 하이퍼파라미터 튜닝 | Kaggle GPU T4 x2 |
| `05_llm_comparison.ipynb` | Gemini vs Gemma-3 vs Qwen2.5 LLM 진단 비교 | Kaggle GPU T4 x2 |
| `06_ensemble.ipynb` | ML+DL 앙상블 (Voting, Stacking) | Kaggle GPU T4 x2 |
| `07_final_comparison.ipynb` | 전체 모델 최종 비교 분석 | Kaggle GPU T4 x2 |

---

## 향후 계획

- [ ] 인증 서비스 (Spring Boot / NextAuth)
- [ ] RAG 유사 사례 검색 고도화
- [ ] WebSocket 실시간 센서 연결
- [ ] Docker Compose 통합 배포
- [ ] 실제 데이터로 DL 모델 재학습
- [ ] MLFlow 실험 관리 통합

---

## 개발자

**쿠카** - 아주대학교 AI융합전공

---

## 라이선스

This project is licensed under the MIT License.
