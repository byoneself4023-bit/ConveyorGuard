# ConveyorGuard 개발 문서

> 최종 업데이트: 2026-02-01

---

## 1. 프로젝트 현황

### 1.1 구현 상태 요약

| 구분 | 계획 | 실제 구현 | 상태 |
|------|------|----------|------|
| **Frontend** | Next.js 15 + TailwindCSS | 1,171줄 | :green_circle: 완료 |
| **ML API** | FastAPI + 추론 | 784줄 | :green_circle: 완료 |
| **LLM Service** | Gemini + RAG | 884줄 | :green_circle: 완료 |
| **ML Service** | 모델 아키텍처 + 학습 | 3,411줄 | :yellow_circle: 재학습 중 |
| **Infra** | Docker Compose | docker-compose.yml | :green_circle: 완료 |

### 1.2 코드 통계

| 서비스 | 줄 수 | 비율 |
|--------|-------|------|
| ml-service | 3,411 | 54.6% |
| frontend | 1,171 | 18.7% |
| llm-service | 884 | 14.1% |
| conveyorguard-api | 784 | 12.5% |
| **총계** | **6,250** | 100% |

---

## 2. 프로젝트 구조

```
ConveyorGuard/
├── frontend/                    # Next.js 프론트엔드
│   └── src/
│       ├── app/
│       │   ├── page.tsx                # 대시보드
│       │   └── equipment/[id]/page.tsx # 장비 상세 + AI 진단
│       ├── components/
│       │   ├── dashboard/              # StatusCard, SensorGauge 등
│       │   └── ui/                     # Toast, Skeleton
│       └── lib/api.ts                  # API 클라이언트
│
├── llm-service/                 # LLM 진단 서비스 (Port 8001)
│   ├── .env                     # GEMINI_API_KEY
│   └── app/
│       ├── main.py              # FastAPI 앱
│       ├── api/router.py        # /api/v1/diagnose
│       ├── core/                # gemini.py, prompts.py, rag.py
│       ├── agents/              # LangGraph 멀티에이전트 (미연동)
│       ├── tools/               # LangChain Tools (미연동)
│       └── rag/                 # FAISS 벡터 DB (미연동)
│
├── conveyorguard-api/           # ML API (Port 8000)
│   └── app/
│       ├── main.py              # FastAPI 앱
│       ├── api/                 # router.py, schemas.py
│       └── core/                # model.py, loader.py
│
├── ml-service/                  # 모델 학습 코드
│   ├── models/                  # 학습된 모델 (.pkl)
│   └── outputs/                 # 시각화 결과
│
├── notebooks/                   # Kaggle 학습 노트북
│   ├── 01_preprocess.ipynb      # 전처리
│   ├── 02_baseline_cnn.ipynb    # CNN 학습
│   ├── 03_ml_baseline.ipynb     # ML 모델 비교
│   ├── 04_dl_tuning.ipynb       # DL 튜닝 (TPU)
│   └── 05_llm_comparison.ipynb  # LLM 비교
│
└── docker-compose.yml
```

---

## 3. Frontend 상세

### 3.1 파일 구조

| 파일 | 설명 | 줄 수 |
|------|------|-------|
| `app/page.tsx` | 대시보드 메인 | 206 |
| `app/equipment/[id]/page.tsx` | 장비 상세 + AI 진단 | 302 |
| `components/dashboard/*` | StatusCard, SensorGauge 등 | 348 |
| `components/ui/*` | Toast, Skeleton | 125 |
| `lib/api.ts` | ML + LLM API 클라이언트 | 93 |
| `types/index.ts` | TypeScript 타입 정의 | 44 |

### 3.2 주요 기능

- 장비 상태 카드 (실시간)
- 센서 게이지 + 트렌드 차트
- 열화상 히트맵
- AI 진단 리포트 표시
- 진단 히스토리 (localStorage, 최대 100건)
- 토스트 알림

### 3.3 API 클라이언트

```typescript
// ML API (Port 8000)
const API_BASE = "http://localhost:8000";
- fetchHealth()
- fetchModelInfo()
- predictTest()
- predict(data)

// LLM API (Port 8001)
const LLM_API_BASE = "http://localhost:8001";
- requestDiagnosis(request)
- checkLLMHealth()
```

---

## 4. LLM Service 상세

### 4.1 파일 구조

| 파일 | 설명 | 줄 수 |
|------|------|-------|
| `main.py` | FastAPI 앱 (Port 8001) | 23 |
| `api/router.py` | POST /api/v1/diagnose | 90 |
| `core/gemini.py` | Gemini 2.5 Flash 연동 | 100 |
| `core/prompts.py` | 진단 프롬프트 템플릿 | 41 |
| `core/rag.py` | 코사인 유사도 검색 | 65 |
| `agents/diagnosis_graph.py` | LangGraph 멀티에이전트 | 339 |
| `tools/diagnosis_tools.py` | LangChain Tools (5개) | 116 |
| `rag/case_retriever.py` | FAISS 벡터 DB RAG | 110 |

### 4.2 API 스키마

**Request:**
```python
class DiagnosisRequest:
    equipment_id: str           # "OHT-003"
    prediction: str             # "정상" | "경미" | "중간" | "심각"
    confidence: float           # 0.85
    sensors: SensorData         # ntc, pm1_0, ..., ct4
    thermal_max_temp: float     # 열화상 최고온도
```

**Response:**
```python
class DiagnosisResponse:
    equipment_id: str
    severity: str               # 심각도
    anomalies: List[str]        # ["NTC(78)가 정상범위 초과", ...]
    probable_cause: str         # "모터 과부하 추정"
    recommended_action: str     # "즉시 점검 필요"
    similar_cases: List[...]    # 유사 사례
```

### 4.3 진단 플로우

```
POST /api/v1/diagnose
        │
        ├──► generate_diagnosis()  (gemini.py)
        │         │
        │         ├─► analyze_sensors() - 이상치 검출
        │         ├─► get_diagnosis_prompt() - 프롬프트 생성
        │         └─► Gemini 2.5 Flash API 호출
        │
        └──► find_similar_cases()  (rag.py)
                  └─► 코사인 유사도로 유사 사례 검색
```

### 4.4 고급 기능 (미연동)

| 파일 | 기능 | 상태 |
|------|------|------|
| `diagnosis_graph.py` | LangGraph 멀티에이전트 | 미연동 |
| `diagnosis_tools.py` | LangChain Tools | 미연동 |
| `case_retriever.py` | FAISS 벡터 DB | 미연동 |

---

## 5. ML API 상세

### 5.1 엔드포인트

| Method | Endpoint | 설명 |
|--------|----------|------|
| GET | `/` | 서비스 정보 |
| GET | `/api/v1/health` | 헬스체크 |
| GET | `/api/v1/model/info` | 모델 정보 |
| GET | `/api/v1/predict/test` | 테스트 예측 |
| POST | `/api/v1/predict` | 단일 예측 |
| POST | `/api/v1/predict/batch` | 배치 예측 |

### 5.2 예측 응답

```python
{
    "prediction": 3,           # 0~3
    "label": "심각한 열화",
    "confidence": 0.92,
    "probabilities": [0.02, 0.03, 0.03, 0.92]
}
```

---

## 6. 모델 아키텍처

### 6.1 ML Pipeline (XGBoost) - 메인

```
Raw Data (30 timesteps)
    │
    ▼
┌─────────────────────────────────────────┐
│  Feature Engineering (64 features)      │
│  ├── Sensor: mean, std, max, min, last, diff (48개)
│  ├── Image: mean, std, max, trend (7개)
│  └── External: mean, std, last (9개)
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────┐
│    XGBoost      │ → 4-Class 분류
│   Classifier    │
└─────────────────┘
```

### 6.2 DL Pipeline (CNN + Transformer)

```
┌─────────────────────────────────────────────────────────────┐
│                  MultimodalTransformer                      │
│                  (582,405 파라미터)                          │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────┐   ┌──────────┐   ┌─────────┐                  │
│  │ Sensor  │   │  Image   │   │External │                  │
│  │(B,30,8) │   │(B,30,    │   │(B,30,3) │                  │
│  │         │   │ 60,80)   │   │         │                  │
│  └────┬────┘   └────┬─────┘   └────┬────┘                  │
│       │             │              │                        │
│       ▼             ▼              ▼                        │
│  ┌─────────┐   ┌──────────┐   ┌─────────┐                  │
│  │MLP      │   │CNN       │   │Linear   │                  │
│  │8→64→128 │   │4-layer   │   │3→128    │                  │
│  └────┬────┘   └────┬─────┘   └────┬────┘                  │
│       ▼             ▼              │                        │
│  ┌──────────────────────┐         │                        │
│  │  TemporalEncoder     │         │                        │
│  │  (Transformer 2-layer)│         │                        │
│  └──────────┬───────────┘         │                        │
│             └──────────┬───────────┘                        │
│                        │                                    │
│               ┌────────▼────────┐                          │
│               │ Fusion (384→128)│                          │
│               └────────┬────────┘                          │
│               ┌────────▼────────┐                          │
│               │ Attention Pool  │                          │
│               └────────┬────────┘                          │
│               ┌────────▼────────┐                          │
│               │Classifier (128→4)│                          │
│               └─────────────────┘                          │
└─────────────────────────────────────────────────────────────┘
```

---

## 7. 학습 노트북 현황

| 노트북 | 설명 | 환경 | 상태 |
|--------|------|------|------|
| `00_eda.ipynb` | 탐색적 데이터 분석 | GPU T4 x2 | :green_circle: 완료 |
| `01_preprocess.ipynb` | EDA 기반 전처리 + 윈도우 생성 | GPU T4 x2 | :green_circle: 완료 |
| `02_baseline_cnn.ipynb` | CNN+Transformer 베이스라인 | GPU T4 x2 | :green_circle: 완료 |
| `03_ml_baseline.ipynb` | ML 모델 4종 비교 | GPU T4 x2 | :green_circle: 완료 |
| `04_dl_tuning.ipynb` | Optuna DL 하이퍼파라미터 튜닝 | GPU T4 x2 | :yellow_circle: 작업 중 |
| `05_llm_comparison.ipynb` | LLM 비교 (Gemini vs Gemma-3 vs Qwen2.5) | GPU T4 x2 | :yellow_circle: 작업 중 |
| `06_ensemble.ipynb` | ML+DL 앙상블 (Voting, Stacking) | GPU T4 x2 | :yellow_circle: 작업 중 |
| `07_final_comparison.ipynb` | 전체 모델 최종 비교 분석 | GPU T4 x2 | :yellow_circle: 작업 중 |

---

## 8. 설정 정보

### 8.1 데모용 장비 (5대)

| ID | Type | Location | 초기 상태 |
|----|------|----------|----------|
| OHT-001 | OHT | FAB1-Zone A | 정상 |
| OHT-002 | OHT | FAB1-Zone B | 경미 |
| OHT-003 | OHT | FAB2-Zone A | 심각 |
| OHT-004 | OHT | FAB2-Zone B | 중간 |
| OHT-005 | OHT | FAB2-Zone C | 정상 |

### 8.2 진단 이력 설정

- **저장 위치**: localStorage (브라우저)
- **최대 보관 건수**: 100건 (장비별)
- **저장 키 형식**: `diagnosis_history_{equipmentId}`

### 8.3 사용자 관리

- **현재**: 단일 사용자 (인증 없음)
- **향후 계획**: Spring Boot 인증 서비스

### 8.4 알림 기능

- **현재**: Toast 팝업만 구현 (브라우저 내)
- **향후 계획**: WebSocket 실시간 알림

---

## 9. 실행 방법

### ML API (Port 8000)
```bash
cd conveyorguard-api
uvicorn app.main:app --port 8000
```

### LLM API (Port 8001)
```bash
cd llm-service
# .env에 GEMINI_API_KEY 필요
uvicorn app.main:app --port 8001
```

### Frontend (Port 3000)
```bash
cd frontend
npm run dev
```

---

## 10. Phase별 진행 상황

| Phase | 내용 | 완성도 | 상태 |
|-------|------|--------|------|
| Phase 1 | 인프라 기반 (Docker, DB, Auth) | 0% | :red_circle: 미착수 |
| Phase 2 | 데이터 파이프라인 | 100% | :green_circle: 완료 |
| Phase 3 | ML 모델 학습 | 70% | :yellow_circle: 재학습 중 |
| Phase 4 | API 서버 구현 | 100% | :green_circle: 완료 |
| Phase 5 | LLM 진단 | 100% | :green_circle: 완료 |
| Phase 6 | Frontend | 90% | :green_circle: 완료 |

---

*Last Updated: 2025-01-16*
