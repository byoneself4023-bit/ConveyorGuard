# ConveyorGuard 프로젝트 아키텍처 설계도

> 다른 프로젝트에 변형 적용할 수 있도록 작성된 상세 참조 문서

---

## 1. 시스템 전체 구조

```
┌─────────────────────────────────────────────────────────┐
│       Frontend (Next.js 15 + React 19)                   │
│  /  (Dashboard)    /equipment/[id]  (상세/진단)          │
└─────────────────────────────────────────────────────────┘
              ↓ fetch (REST)           ↓ fetch (REST)
┌────────────────────────┐   ┌────────────────────────────┐
│   ML API (FastAPI)     │   │   LLM Service (FastAPI)    │
│   Port 8000            │   │   Port 8001                │
│  - 열화 예측           │   │  - AI 자연어 진단          │
│  - 배치 추론           │   │  - LangGraph 에이전트      │
│  - 모델 정보           │   │  - RAG 유사 사례 검색      │
└────────────────────────┘   └────────────────────────────┘
         ↓                            ↓
┌────────────────────────┐   ┌────────────────────────────┐
│   Core Layer           │   │   External Services        │
│  - PyTorch Model       │   │  - Google Gemini 2.5 Flash │
│  - Preprocessing       │   │  - FAISS Vector Store      │
│  - Singleton Loader    │   │  - LangSmith Monitoring    │
└────────────────────────┘   └────────────────────────────┘
         ↓                            ↓
┌─────────────────────────────────────────────────────────┐
│                 Data / Model Layer                        │
│  - PyTorch Checkpoint (.pt)                              │
│  - XGBoost/LightGBM (.pkl) - ML Service                 │
│  - In-memory Case DB (JSON) - RAG                       │
│  - localStorage (진단 히스토리) - Frontend               │
│  - MLFlow (실험 추적)                                    │
└─────────────────────────────────────────────────────────┘
```

### Listify와의 핵심 차이점

| 항목 | Listify | ConveyorGuard |
|------|---------|---------------|
| **프론트엔드** | React (Vite, SPA) | Next.js 15 (App Router, SSR가능) |
| **백엔드** | Flask (단일 서버) | FastAPI × 2 (ML + LLM 마이크로서비스) |
| **DB** | MySQL + Connection Pool | 없음 (모델 파일 + Mock 데이터 + localStorage) |
| **인증** | JWT + bcrypt + Role | 없음 (인증 미구현) |
| **외부 API** | Spotify API | Google Gemini API + Spotify-like 없음 |
| **상태관리** | React Hooks (Props drilling) | React Hooks + Next.js useRouter/useParams |
| **라우팅** | view 상태 전환 | Next.js App Router (파일 기반 라우팅) |
| **핵심 도메인** | 음악 플레이리스트 CRUD | AI 예측 + 실시간 모니터링 + 자연어 진단 |

---

# PART 1: ML API 백엔드 (FastAPI - Port 8000)

---

## 2. 진입점 (conveyorguard-api/app/main.py)

```python
# 1. Lifespan Context Manager (모델 로드/해제)
@asynccontextmanager
async def lifespan(app: FastAPI):
    """앱 시작 시 모델 로드"""
    model_path = os.getenv("MODEL_PATH", "../data/models/best_model.pt")
    try:
        model_loader.load(model_path)     # 싱글톤 로더
    except FileNotFoundError:
        logger.warning(f"Model not found: {model_path}")
    yield                                  # 앱 실행
    logger.info("Shutting down...")        # 종료

# 2. FastAPI 애플리케이션 초기화
app = FastAPI(
    title="ConveyorGuard ML API",
    description="이송장치 열화 예측 AI API",
    version="1.0.0",
    lifespan=lifespan,     # ← Flask의 @app.before_request 대신
    docs_url="/docs"       # Swagger UI 자동 생성
)

# 3. CORS 설정 (모든 origin 허용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 4. 라우터 등록 (Blueprint 대신 APIRouter)
app.include_router(router, prefix="/api/v1", tags=["Prediction"])

# 5. Root 엔드포인트
@app.get("/")
async def root():
    return {"service": "ConveyorGuard ML API", "version": "1.0.0", "docs": "/docs"}

# 6. 서버 시작
if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
```

### Listify(Flask)와 비교

```
Flask (Listify)                    FastAPI (ConveyorGuard)
─────────────────────              ───────────────────────
app = Flask(__name__)              app = FastAPI(lifespan=lifespan)
CORS(app, ...)                     app.add_middleware(CORSMiddleware, ...)
app.register_blueprint(bp)         app.include_router(router, prefix=...)
@app.route('/health')              @app.get("/")
app.run(host, port, debug)         uvicorn.run("app.main:app", ...)

데이터 검증: 수동 (if not data)     자동 (Pydantic BaseModel)
API 문서: 없음 (직접 작성)          자동 (Swagger UI at /docs)
비동기: 불가                        기본 (async/await)
```

---

## 3. Model Loader Layer (core/loader.py) — DB Layer 대체

> ConveyorGuard에는 DB가 없으므로, Listify의 `db.py` (Connection Pool)에 해당하는 역할을
> **ModelLoader 싱글톤**이 수행합니다.

### 싱글톤 패턴 (Listify의 DatabaseManager와 동일 패턴)

```python
class ModelLoader:
    """모델 로더 싱글톤 — Listify의 DatabaseManager에 대응"""

    _instance: Optional['ModelLoader'] = None     # 클래스 변수 (싱글톤)
    _model: Optional[MultimodalTransformer] = None
    _device: torch.device = None
    _model_info: dict = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load(self, model_path: str) -> MultimodalTransformer:
        """모델 로드 (1회만 실행)"""
        if self._model is not None:
            return self._model

        # Device 설정
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Checkpoint 로드
        checkpoint = torch.load(model_path, map_location=self._device, weights_only=False)

        # Config 복원
        config = Config()
        if 'config' in checkpoint:
            for key, value in checkpoint['config'].items():
                if hasattr(config, key):
                    setattr(config, key, value)

        # 모델 생성 및 가중치 로드
        self._model = MultimodalTransformer(config)
        self._model.load_state_dict(checkpoint['model_state_dict'])
        self._model.to(self._device)
        self._model.eval()                        # 추론 모드 고정

        # 메타 정보 저장
        self._model_info = {
            'best_accuracy': checkpoint.get('best_acc', 'N/A'),
            'parameters': sum(p.numel() for p in self._model.parameters()),
        }
        return self._model

    @property
    def model(self) -> MultimodalTransformer: ...
    @property
    def device(self) -> torch.device: ...
    @property
    def info(self) -> dict: ...
    def is_loaded(self) -> bool: ...

# 전역 인스턴스 (Listify의 get_connection()에 대응)
model_loader = ModelLoader()
```

### 사용 패턴 비교

```
Listify (DB)                       ConveyorGuard (Model)
──────────                         ─────────────────────
conn = get_connection()            model = model_loader.model
try:                               model.eval()
    cursor.execute(SQL, params)    with torch.no_grad():
    result = cursor.fetchone()         result = model.predict(s, i, e)
    conn.commit()                  return result
finally:
    conn.close()
```

---

## 4. Schemas Layer (api/schemas.py) — Pydantic 자동 검증

> Listify에서 Controller가 수동으로 하던 `if not data.get('email')` 검증을
> Pydantic이 자동으로 처리합니다.

```python
# === 라벨 매핑 ===
STATE_LABELS = {
    0: "정상",
    1: "경미한 열화",
    2: "중간 열화",
    3: "심각한 열화"
}

# === 요청 스키마 (자동 검증) ===
class PredictRequest(BaseModel):
    """단일 예측 요청"""
    sensors: List[List[float]]          # (30, 8) - 자동 타입 검증
    images: List[List[List[float]]]     # (30, 60, 80)
    external: List[List[float]]         # (30, 3)
    normalize: bool = True              # 기본값

class BatchPredictRequest(BaseModel):
    """배치 예측 요청"""
    items: List[PredictRequest] = Field(..., max_length=32)

# === 응답 스키마 (자동 직렬화) ===
class PredictResponse(BaseModel):
    prediction: int                     # 0~3
    label: str                          # "정상" / "경미한 열화" / ...
    confidence: float                   # 0~1
    probabilities: List[float]          # [0.9, 0.05, 0.03, 0.02]

class BatchPredictResponse(BaseModel):
    results: List[PredictResponse]
    total: int

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str

class ModelInfoResponse(BaseModel):
    name: str = "MultimodalTransformer"
    version: str = "1.0.0"
    accuracy: float
    parameters: int
    device: str
    input_shape: dict
    output_classes: dict
```

### Listify Controller 검증 vs ConveyorGuard Pydantic

```
Listify (수동 검증)                 ConveyorGuard (Pydantic 자동)
─────────────────                   ─────────────────────────────
data = request.get_json()           # FastAPI가 자동으로:
if not data:                        #   1. JSON 파싱
    return error(400)               #   2. 타입 변환
if not data.get('email'):           #   3. 필수 필드 체크
    return error(400, "이메일 필요") #   4. 검증 실패 시 422 응답
                                    #   5. Swagger UI 자동 생성
```

---

## 5. Router Layer (api/router.py) — Routes + Controller 통합

> FastAPI에서는 Routes와 Controller가 하나의 Router 파일에 통합됩니다.
> Listify의 `routes/` + `controllers/` 역할을 합친 구조입니다.

### 5.1 ML API 엔드포인트

| 메서드 | 엔드포인트 | 역할 | 인증 |
|--------|-----------|------|------|
| `GET` | `/api/v1/health` | 헬스체크 (모델 로드 상태, 디바이스) | 공개 |
| `GET` | `/api/v1/model/info` | 모델 메타데이터 (정확도, 파라미터 수) | 공개 |
| `POST` | `/api/v1/predict` | 단일 예측 (센서+이미지+환경) | 공개 |
| `POST` | `/api/v1/predict/batch` | 배치 예측 (최대 32건) | 공개 |
| `GET` | `/api/v1/predict/test` | 정상 상태 더미 테스트 | 공개 |
| `GET` | `/api/v1/predict/test/degraded` | 열화 상태 더미 테스트 | 공개 |

### 5.2 입력 검증 함수 (shape 검증)

```python
def validate_input_shape(sensors, images, external):
    """입력 데이터 shape 검증 — Listify의 validate_email() 등에 대응"""
    errors = []

    # sensors: (30, 8)
    if len(sensors) != 30:
        errors.append(f"sensors: 시퀀스 길이가 30이어야 합니다 (현재: {len(sensors)})")
    elif len(sensors[0]) != 8:
        errors.append(f"sensors: 각 타임스텝은 8개 센서값이어야 합니다 (현재: {len(sensors[0])})")

    # images: (30, 60, 80)
    if len(images) != 30:
        errors.append(f"images: 시퀀스 길이가 30이어야 합니다 (현재: {len(images)})")
    elif len(images[0]) != 60:
        errors.append(f"images: 이미지 높이가 60이어야 합니다 (현재: {len(images[0])})")
    elif len(images[0][0]) != 80:
        errors.append(f"images: 이미지 너비가 80이어야 합니다 (현재: {len(images[0][0])})")

    # external: (30, 3)
    if len(external) != 30:
        errors.append(f"external: 시퀀스 길이가 30이어야 합니다 (현재: {len(external)})")
    elif len(external[0]) != 3:
        errors.append(f"external: 각 타임스텝은 3개 값이어야 합니다 (현재: {len(external[0])})")

    if errors:
        raise HTTPException(status_code=422, detail={
            "message": "입력 데이터 shape이 올바르지 않습니다",
            "errors": errors,
            "expected_shape": {"sensors": [30, 8], "images": [30, 60, 80], "external": [30, 3]}
        })
```

### 5.3 핵심 예측 엔드포인트

```python
@router.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """단일 예측 — Listify의 Controller+Service 역할 통합"""

    # 1. 모델 로드 확인 (Listify: 사용자 존재 확인)
    if not model_loader.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")

    # 2. 입력 검증 (Listify: validate_email, validate_password)
    validate_input_shape(request.sensors, request.images, request.external)

    # 3. 전처리 (Listify: hash_password)
    processed = preprocess_input(
        request.sensors, request.images, request.external, request.normalize
    )

    # 4. 텐서 변환 + 추론 (Listify: DB 쿼리)
    device = model_loader.device
    sensors = torch.tensor(processed['sensors'], dtype=torch.float32).unsqueeze(0).to(device)
    images = torch.tensor(processed['images'], dtype=torch.float32).unsqueeze(0).to(device)
    external = torch.tensor(processed['external'], dtype=torch.float32).unsqueeze(0).to(device)

    result = model_loader.model.predict(sensors, images, external)

    # 5. 응답 (Listify: jsonify)
    pred = result['prediction'].item()
    return PredictResponse(
        prediction=pred,
        label=STATE_LABELS[pred],
        confidence=round(result['confidence'].item(), 4),
        probabilities=[round(p, 4) for p in result['probabilities'].squeeze().tolist()]
    )
```

### 5.4 배치 예측

```python
@router.post("/predict/batch", response_model=BatchPredictResponse)
async def predict_batch(request: BatchPredictRequest):
    """배치 예측 (최대 32건)"""
    # 각 아이템 shape 검증
    for i, item in enumerate(request.items):
        try:
            validate_input_shape(item.sensors, item.images, item.external)
        except HTTPException as e:
            raise HTTPException(status_code=422, detail={"message": f"items[{i}] 검증 실패", "errors": e.detail})

    # 전체 배치 전처리 → 텐서 스택 → 한번에 추론
    all_sensors, all_images, all_external = [], [], []
    for item in request.items:
        processed = preprocess_input(item.sensors, item.images, item.external, item.normalize)
        all_sensors.append(processed['sensors'])
        # ...

    sensors = torch.tensor(np.stack(all_sensors), dtype=torch.float32).to(device)
    result = model_loader.model.predict(sensors, images, external)
    # ... 개별 결과로 분리하여 반환
```

### 5.5 테스트 엔드포인트 (정상/열화 더미)

```python
@router.get("/predict/test", response_model=PredictResponse)
async def predict_test():
    """정상 상태 더미 데이터 테스트"""
    dummy_sensors = [[25, 10, 15, 20, 50, 50, 50, 50]] * 30   # NTC=25°C, 정상 전류
    dummy_images = [[[30.0] * 80 for _ in range(60)] for _ in range(30)]  # 30°C 균일
    dummy_external = [[25, 50, 500]] * 30                       # 25°C, 50%, 500lux
    # → preprocess → predict → return

@router.get("/predict/test/degraded", response_model=PredictResponse)
async def predict_test_degraded():
    """열화 상태 더미 데이터 테스트"""
    dummy_sensors = [[80, 200, 300, 400, 150, 150, 150, 150]] * 30  # 높은 온도/전류
    dummy_images = [[[80.0] * 80 for _ in range(60)] for _ in range(30)]  # 80°C
    dummy_external = [[40, 80, 200]] * 30                             # 고온/고습도
```

---

## 6. Preprocessing Layer (core/preprocessing.py) — Service 계층 대체

> Listify에서 `services/*.py`가 하던 데이터 변환/검증 역할을
> Preprocessing 모듈이 수행합니다.

### 정규화 범위 정의

```python
# 센서 정규화 범위
SENSOR_RANGES = {
    0: (0, 200),    # NTC (온도) — °C
    1: (0, 1000),   # PM1.0 — μg/m³
    2: (0, 1000),   # PM2.5
    3: (0, 1000),   # PM10
    4: (0, 200),    # CT1 (전류) — A
    5: (0, 200),    # CT2
    6: (0, 200),    # CT3
    7: (0, 200),    # CT4
}

# 외부환경 정규화 범위
EXTERNAL_RANGES = np.array([
    [0, 50],    # 온도 (°C)
    [0, 100],   # 습도 (%)
    [0, 1000],  # 조도 (lux)
])
```

### 정규화 함수

```python
def normalize_sensors(data: np.ndarray) -> np.ndarray:
    """센서 Min-Max 정규화 → [0, 1]"""
    result = data.copy().astype(np.float32)
    for i, (min_v, max_v) in SENSOR_RANGES.items():
        result[..., i] = (data[..., i] - min_v) / (max_v - min_v + 1e-8)
    return np.clip(result, 0, 1)

def normalize_images(data: np.ndarray) -> np.ndarray:
    """이미지 Min-Max 정규화 (프레임별)"""
    # 각 프레임마다 독립적으로 min-max

def normalize_external(data: np.ndarray) -> np.ndarray:
    """외부환경 Min-Max 정규화"""

def preprocess_input(sensors, images, external, normalize=True) -> Dict[str, np.ndarray]:
    """전체 전처리 파이프라인 — Listify의 Service 함수에 대응"""
    sensors = np.array(sensors, dtype=np.float32)
    images = np.array(images, dtype=np.float32)
    external = np.array(external, dtype=np.float32)

    if normalize:
        sensors = normalize_sensors(sensors)
        images = normalize_images(images)
        external = normalize_external(external)

    return {'sensors': sensors, 'images': images, 'external': external}
```

---

## 7. Model Layer (core/model.py) — AI 모델 아키텍처

> Listify에는 없는 ConveyorGuard만의 핵심 계층입니다.

### 모델 설정

```python
class Config:
    seq_len: int = 30           # 30 타임스텝
    sensor_dim: int = 8         # 8채널 센서
    external_dim: int = 3       # 3채널 환경
    image_height: int = 60      # 열화상 높이
    image_width: int = 80       # 열화상 너비
    num_classes: int = 4        # 4단계 열화
    embed_dim: int = 128        # 임베딩 차원
    num_heads: int = 4          # 어텐션 헤드
    num_layers: int = 2         # Transformer 레이어
    dropout: float = 0.1
```

### MultimodalTransformer 구조

```
입력 (3가지 모달리티)
──────────────────────────────────────────────────────
센서 (B, 30, 8)                SensorEncoder: Linear(8→64)→LN→GELU→Linear(64→128)→LN
                                    ↓
                               TemporalEncoder: PositionalEmbed + TransformerEncoder(2L, 4H)
                                    ↓
열화상 (B, 30, 60, 80)        ImageEncoder: Conv2d(1→32→64→128) + AdaptiveAvgPool2d(1) → Linear(128)
                                    ↓
                               TemporalEncoder: 동일 Transformer 공유
                                    ↓
환경 (B, 30, 3)               ExternalEncoder: Linear(3→128)→LN→GELU
──────────────────────────────────────────────────────

융합 (Fusion)
──────────────────────────────────────────────────────
Concat: [sensor(128) | image(128) | external(128)] = (B, 30, 384)
    ↓
FusionMLP: Linear(384→128) → LN → GELU → Dropout
    ↓
AttentionPooling: Linear(128→32)→Tanh→Linear(32→1) → Softmax → WeightedSum
    ↓
Output: (B, 128)
──────────────────────────────────────────────────────

분류 (Classifier)
──────────────────────────────────────────────────────
Linear(128→64) → LN → GELU → Dropout → Linear(64→4)
    ↓
Softmax → 4-class 확률 분포
──────────────────────────────────────────────────────
```

### predict 메서드 (추론 전용)

```python
def predict(self, sensor, image, external) -> dict:
    """추론 + 확률 반환"""
    self.eval()
    with torch.no_grad():
        logits = self.forward(sensor, image, external)
        probs = F.softmax(logits, dim=-1)
        pred = logits.argmax(dim=-1)
        confidence = probs.max(dim=-1).values

    return {
        'prediction': pred,           # 클래스 (0~3)
        'confidence': confidence,     # 최대 확률
        'probabilities': probs        # 전체 분포
    }
```

---

# PART 2: LLM Service 백엔드 (FastAPI - Port 8001)

---

## 8. LLM Service 진입점 (llm-service/app/main.py)

```python
app = FastAPI(
    title="ConveyorGuard LLM Service",
    description="AI 진단 리포트 생성 API",
    version="1.0.0"
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], ...)
app.include_router(router, prefix="/api/v1")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "llm-service"}
```

---

## 9. LLM Service 엔드포인트 (llm-service/app/api/router.py)

### 9.1 엔드포인트 목록

| 메서드 | 엔드포인트 | 역할 | 기술 |
|--------|-----------|------|------|
| `POST` | `/api/v1/diagnose` | 표준 AI 진단 | Gemini + 코사인 유사도 RAG |
| `POST` | `/api/v1/diagnose/graph` | LangGraph 에이전트 진단 | LangGraph StateGraph |
| `POST` | `/api/v1/diagnose/rag` | FAISS 유사 사례 검색 | HuggingFace + FAISS |
| `GET` | `/api/v1/metrics` | 모니터링 메트릭 | LangSmith Logger |
| `GET` | `/api/v1/test` | 테스트 진단 | 더미 데이터 |

### 9.2 요청/응답 스키마

```python
class SensorData(BaseModel):
    ntc: float          # NTC 온도
    pm1_0: float        # PM1.0 미세먼지
    pm2_5: float        # PM2.5
    pm10: float         # PM10
    ct1: float          # 전류 1
    ct2: float          # 전류 2
    ct3: float          # 전류 3
    ct4: float          # 전류 4

class DiagnosisRequest(BaseModel):
    equipment_id: str                    # "OHT-003"
    prediction: str                      # "심각"
    confidence: float                    # 0.92
    sensors: SensorData                  # 센서 데이터
    thermal_max_temp: Optional[float]    # 열화상 최고온도

class SimilarCase(BaseModel):
    date: str = ""
    equipment_id: str = ""
    issue: str = ""
    action: str = ""
    similarity: float = 0.0

class DiagnosisResponse(BaseModel):
    equipment_id: str
    severity: str                        # "심각"
    anomalies: List[str]                 # ["NTC 78°C - 임계초과", ...]
    probable_cause: str                  # "베어링 마모로 인한 과열"
    recommended_action: str              # "즉시 가동 중단 후 베어링 교체"
    similar_cases: List[SimilarCase]     # 유사 과거 사례 Top-3
```

### 9.3 표준 진단 처리 흐름

```python
@router.post("/diagnose", response_model=DiagnosisResponse)
async def diagnose(request: DiagnosisRequest):
    """
    처리 흐름:
    1. Gemini API로 자연어 진단 생성
    2. 코사인 유사도로 과거 사례 검색
    3. 결과 통합 응답
    """
    # 1. LLM 진단 생성
    diagnosis = await generate_diagnosis(
        equipment_id=request.equipment_id,
        prediction=request.prediction,
        confidence=request.confidence,
        sensors=request.sensors.model_dump(),
        thermal_max_temp=request.thermal_max_temp
    )

    # 2. 유사 사례 검색
    similar_cases = find_similar_cases(
        sensors=request.sensors.model_dump(),
        prediction=request.prediction
    )

    # 3. 응답 조합
    return DiagnosisResponse(
        equipment_id=request.equipment_id,
        severity=request.prediction,
        anomalies=diagnosis["anomalies"],
        probable_cause=diagnosis["probable_cause"],
        recommended_action=diagnosis["recommended_action"],
        similar_cases=similar_cases
    )
```

---

## 10. Gemini Integration (core/gemini.py)

> Listify의 Spotify 통합에 대응하는 외부 API 통합 계층

```python
import google.generativeai as genai

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-2.5-flash")

def generate_diagnosis(prediction_result: dict, sensor_data: dict) -> str:
    """
    ML 예측 결과 + 센서 데이터 → 자연어 진단 리포트

    프롬프트 구조:
    1. 역할 정의: "제조설비 예지보전 전문가"
    2. 데이터 주입: 예측 결과 + 센서 8채널 + 임계값 기준
    3. 출력 지시:
       - 현재 상태 요약
       - 이상 징후 분석
       - 예상 원인
       - 권장 조치 사항
       - 긴급도 (즉시/24시간내/1주일내/정기점검시)
    """
    prompt = f"""당신은 제조설비 예지보전 전문가입니다.

## 예측 결과
- 열화 상태: {prediction_result.get('label')}
- 신뢰도: {prediction_result.get('confidence'):.1%}

## 센서 데이터
- NTC (온도): {sensor_data.get('NTC')}°C
- PM2.5: {sensor_data.get('PM2_5')} μg/m³
- CT1~CT4: {sensor_data.get('CT1')}A ~ {sensor_data.get('CT4')}A

## 임계값 기준
- NTC: 50°C 이상 주의, 70°C 이상 위험
- PM2.5: 35 μg/m³ 이상 주의
- CT: 5A 이상 주의

분석 후 진단 결과를 작성하세요."""

    response = model.generate_content(prompt)
    return response.text
```

### Listify(Spotify) vs ConveyorGuard(Gemini) 외부 API 비교

```
Listify (Spotify)                   ConveyorGuard (Gemini)
──────────────                      ──────────────────────
sp = spotipy.Spotify(...)           genai.configure(api_key=...)
sp.search(q=keyword, type="track")  model.generate_content(prompt)
sp.audio_features([track_id])       # 단일 호출로 진단 완료
sp.artist(artist_id)                # 별도 호출 없음
→ DB에 저장 (upsert)               → 응답으로 바로 반환
```

---

## 11. Prompt Templates (core/prompts.py)

```python
def get_diagnosis_prompt(
    equipment_id: str,
    prediction: str,
    confidence: float,
    sensors: Dict,
    anomalies: List[str],
    thermal_max_temp: Optional[float] = None
) -> str:
    """구조화된 진단 프롬프트 생성"""

    anomaly_text = "\n".join(f"- {a}" for a in anomalies) if anomalies else "- 없음"

    return f"""당신은 제조 설비 예지보전 전문가입니다.

## 장비 정보
- 장비 ID: {equipment_id}
- AI 예측: {prediction} 열화 (신뢰도: {confidence*100:.1f}%)

## 센서 데이터
- NTC: {sensors.get('ntc', 0)}°C
- PM1.0: {sensors.get('pm1_0', 0)} μg/m³
- PM2.5: {sensors.get('pm2_5', 0)} μg/m³
- PM10: {sensors.get('pm10', 0)} μg/m³
- CT1~CT4: {sensors.get('ct1', 0)}A ~ {sensors.get('ct4', 0)}A
- 열화상 최고온도: {thermal_max_temp if thermal_max_temp else 'N/A'}°C

## 감지된 이상
{anomaly_text}

## 응답 형식 (JSON만 출력)
{{
    "probable_cause": "추정되는 고장 원인 (한 문장)",
    "recommended_action": "권장 조치사항 (한 문장)"
}}"""
```

---

## 12. RAG Layer (core/rag.py + rag/case_retriever.py)

### 12.1 기본 RAG (코사인 유사도)

```python
def calculate_similarity(sensors1: Dict, sensors2: Dict) -> float:
    """센서 데이터 코사인 유사도"""
    keys = ['ntc', 'pm1_0', 'pm2_5', 'pm10', 'ct1', 'ct2', 'ct3', 'ct4']
    vec1 = np.array([sensors1.get(k, 0) for k in keys])
    vec2 = np.array([sensors2.get(k, 0) for k in keys])
    similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    return float(similarity)

def find_similar_cases(sensors, prediction, top_k=3) -> List[Dict]:
    """유사 과거 사례 검색 (JSON 파일 기반)"""
    cases = load_cases()                     # cases.json 로드
    scored = []
    for case in cases:
        sim = calculate_similarity(sensors, case.get("sensors", {}))
        if case.get("severity") == prediction:
            sim = min(sim * 1.2, 1.0)        # 같은 심각도면 가산점
        scored.append({...})
    scored.sort(key=lambda x: x["similarity"], reverse=True)
    return scored[:top_k]
```

### 12.2 고급 RAG (FAISS + Sentence Transformers)

```python
# 사전 정의된 사례 DB
CASES = [
    {"id": "CASE-001", "symptom": "모터 과열 NTC 75°C+", "cause": "베어링 마모", "solution": "베어링 교체"},
    {"id": "CASE-002", "symptom": "전류 급증 CT 6A+",     "cause": "벨트 장력 과다", "solution": "장력 조정"},
    {"id": "CASE-003", "symptom": "PM2.5 급증",           "cause": "필터 막힘",     "solution": "필터 교체"},
    {"id": "CASE-004", "symptom": "열화상 핫스팟 80°C+",  "cause": "접촉 불량",     "solution": "커넥터 점검"},
    {"id": "CASE-005", "symptom": "온도+전류 동시 상승",   "cause": "기어박스 오일 부족", "solution": "오일 보충"},
]

class CaseRetriever:
    def __init__(self):
        # Sentence Transformer로 임베딩 → FAISS 인덱스 구축
        self.embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        docs = [Document(page_content=f"{c['symptom']} {c['cause']}", metadata=c) for c in CASES]
        self.vectorstore = FAISS.from_documents(docs, self.embeddings)

    def search(self, query: str, k: int = 3) -> List[Dict]:
        results = self.vectorstore.similarity_search_with_score(query, k=k)
        return [{"case_id": d.metadata["id"], "cause": d.metadata["cause"],
                 "solution": d.metadata["solution"], "similarity": round(1-s, 2)}
                for d, s in results]

retriever = CaseRetriever()  # 모듈 로드 시 초기화 (싱글톤)
```

---

## 13. LangGraph Agent (agents/diagnosis_graph.py)

> Listify에는 없는 ConveyorGuard만의 멀티 에이전트 시스템

### 13.1 상태 정의

```python
class DiagnosisState(TypedDict):
    equipment_id: str
    prediction_result: dict
    sensor_data: dict
    analysis: str               # 분석가 결과
    diagnosis: str              # 진단자 결과
    review: str                 # 검토자 결과
    review_count: int           # 검토 횟수 (최대 2회)
    final_report: str           # 최종 리포트
    status: str                 # "approved" / "unknown"
```

### 13.2 그래프 구조

```
┌───────────┐     ┌──────────────┐     ┌────────────┐
│ analyzer  │ ──→ │diagnostician │ ──→ │  reviewer   │
│ (분석가)  │     │  (진단자)    │     │ (검토자)    │
└───────────┘     └──────────────┘     └────────────┘
                        ↑                     │
                        │  REVISE             │
                        └─────────────────────┤
                                              │ APPROVE
                                              ↓
                                       ┌────────────┐
                                       │  finalize   │
                                       │ (최종 리포트)│
                                       └────────────┘
```

### 13.3 노드 함수

```python
def analyzer_node(state) -> dict:
    """센서 분석 전문가 — 임계값 초과/이상 징후 분석"""
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
    prompt = f"장비: {state['equipment_id']}, 센서: NTC={sensor.get('ntc')}°C, ..."
    return {"analysis": llm.invoke(prompt).content}

def diagnostician_node(state) -> dict:
    """진단 전문가 — 원인, 긴급도, 조치방안 제시 (피드백 반영)"""
    review = state.get("review", "")
    prompt = f"분석: {state['analysis']}\n{'피드백: ' + review if 'REVISE' in review else ''}"
    return {"diagnosis": llm.invoke(prompt).content}

def reviewer_node(state) -> dict:
    """품질관리 책임자 — APPROVE 또는 REVISE 판정"""
    prompt = f"분석: {state['analysis']}\n진단: {state['diagnosis']}\n검토 후 APPROVE/REVISE"
    return {"review": llm.invoke(prompt).content, "review_count": state.get("review_count", 0) + 1}

def finalize_node(state) -> dict:
    """최종 리포트 생성"""
    report = f"# 진단 리포트\n## 장비: {state['equipment_id']}\n### 분석\n{state['analysis']}..."
    return {"final_report": report, "status": "approved"}

def should_continue(state) -> str:
    """조건부 엣지: 2회 초과 또는 APPROVE면 종료"""
    if state.get("review_count", 0) >= 2 or "APPROVE" in state.get("review", "").upper():
        return "finalize"
    return "revise"
```

---

## 14. LangChain Tools (tools/diagnosis_tools.py)

```python
def get_diagnosis_tools():
    """진단에 사용할 5개 Tool"""
    return [
        EquipmentInfoTool(),           # 장비 기본 정보 조회 (Mock DB)
        MaintenanceHistoryTool(),      # 정비 이력 조회 (Mock DB)
        SensorDataTool(),              # 현재 센서 데이터 조회
        MLPredictionTool(),            # ML API 호출 (/predict/test)
        SimilarCaseTool(),             # RAG 유사 사례 검색
    ]

def create_diagnosis_agent():
    """Tool-calling Agent 생성"""
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
    tools = get_diagnosis_tools()

    prompt = ChatPromptTemplate.from_messages([
        ("system", "진단 절차: 1.장비정보 2.정비이력 3.센서 4.ML예측 5.유사사례 6.종합진단"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=10)
```

---

## 15. Monitoring (monitoring/langsmith_config.py)

```python
class Logger:
    """진단 요청 로깅 + 메트릭"""

    def log(self, request_id, equipment_id, latency_ms, success, error=None):
        # JSONL 파일로 로그 기록

    def metrics(self, n=100) -> dict:
        # 최근 n건의 성공률, 평균 레이턴시
        return {"total": ..., "success_rate": ..., "avg_latency_ms": ...}

@track  # 데코레이터
async def some_function(**kwargs):
    # 자동으로 latency, success/failure 로깅
```

---

# PART 3: ML Service (모델 학습)

---

## 16. 학습 파이프라인 (ml-service/)

### 16.1 모델 구성요소

| 파일 | 역할 |
|------|------|
| `models/sensor_encoder.py` | 센서 임베딩 (Linear 기반) |
| `models/image_encoder.py` | 열화상 임베딩 (CNN 기반) |
| `models/temporal_encoder.py` | 시계열 인코딩 (Transformer) |
| `models/fusion.py` | 멀티모달 융합 (4종: Cross-Attention, FiLM, Gated, Hierarchical) |
| `models/classifier.py` | 4-class 분류기 |
| `training/train.py` | MLFlow 연동 학습 스크립트 |
| `training/evaluate.py` | 평가 (Accuracy, F1, Confusion Matrix) |
| `training/tracking.py` | MLFlow 트래커 래퍼 |
| `training/trainer.py` | 학습 루프 |
| `training/dataset.py` | PyTorch Dataset |
| `configs/model_config.yaml` | 모델/학습/데이터 설정 |

### 16.2 Fusion 모듈 4종

```python
# 1. CrossAttention 융합 (기본)
class MultimodalFusion:
    # 양방향 Cross-Attention: 센서→이미지 + 이미지→센서 → Concat → Linear

# 2. FiLM 조건화 (외부환경)
class MultimodalFusionWithExternal:
    # 센서+이미지 → CrossAttention → 외부환경으로 γ,β 계산 → Scale+Shift

# 3. 게이트 기반 융합
class GatedFusion:
    # 각 모달리티의 기여도를 동적 Softmax 게이트로 조절

# 4. 계층적 융합
class HierarchicalFusion:
    # 1단계: 센서+외부환경 → Concat → Linear
    # 2단계: (결과)+이미지 → CrossAttention
```

### 16.3 학습 설정 (model_config.yaml 핵심)

```yaml
model:
  embed_dim: 256
  num_heads: 8
  sensor_depth: 2
  image_depth: 4
  temporal_depth: 4
  fusion_type: "cross_attention"
  pooling_type: "attention"

training:
  epochs: 50
  batch_size: 16
  learning_rate: 0.001
  scheduler: "cosine"
  warmup_epochs: 5
  early_stopping: true
  patience: 10

data:
  training_samples: 99476
  validation_samples: 12394
  source: "AIHub #71802"
  equipment_types: [AGV, OHT]
```

### 16.4 노트북 파이프라인 (notebooks/)

| 노트북 | 역할 | 핵심 기술 |
|--------|------|-----------|
| `00_eda.ipynb` | 탐색적 데이터 분석 (EDA) | Plotly, 클래스 분포, 센서/열화상/외부환경 분석 |
| `01_preprocess.ipynb` | 데이터 전처리 + 윈도우 생성 | GroupShuffleSplit (세션 단위 split) |
| `02_baseline_cnn.ipynb` | DL 기준 모델 학습 | Transformer + FiLM 외부환경 융합 + Early Stopping |
| `03_ml_baseline.ipynb` | ML 기준 모델 비교 | XGBoost, LightGBM, CatBoost, RandomForest |
| `04_dl_tuning.ipynb` | DL 하이퍼파라미터 튜닝 | Optuna + Transformer |
| `05_llm_comparison.ipynb` | LLM 진단 파이프라인 비교 | Gemini + LangGraph 멀티에이전트 |

#### 00_eda.ipynb — 탐색적 데이터 분석

AIHub #71802 데이터셋의 구조와 분포를 시각화합니다.

```
분석 항목 (9개 섹션):
1. 데이터셋 구조 파악 (ZIP 내 CSV/BIN/JSON 구성)
2. 클래스 불균형 분석 (4-class 분포)
3. 센서 채널별 분포 (NTC, PM, CT × 정상/열화)
4. 센서 상관관계 히트맵
5. 시계열 패턴 분석 (정상 vs 열화 세션)
6. 열화상 이미지 시각화 (평균/최대온도 분포)
7. 외부환경 분석 (온도, 습도, 조도)
8. 모달리티 간 상관관계
9. 분석 요약 및 인사이트
```

#### 01_preprocess.ipynb — 세션 단위 데이터 분할

> **핵심 변경**: `train_test_split` → `GroupShuffleSplit`로 데이터 누수 방지

```python
# 문제: 윈도우 단위 split → 같은 세션의 윈도우가 train/test 양쪽에 출현 (데이터 누수)
# 해결: 세션 단위 split → GroupShuffleSplit(groups=session_ids)

from sklearn.model_selection import GroupShuffleSplit

# 1단계: train(70%) vs temp(30%)
gss1 = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
train_idx, temp_idx = next(gss1.split(all_labels, all_labels, groups=all_session_ids))

# 2단계: temp → val(50%) + test(50%)
gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
val_idx, test_idx = next(gss2.split(temp_labels, temp_labels, groups=temp_session_ids))

# 검증: 세션 누수 없음 확인
assert len(train_sessions & val_sessions) == 0
assert len(train_sessions & test_sessions) == 0
assert len(val_sessions & test_sessions) == 0
```

#### 02_baseline_cnn.ipynb — FiLM 외부환경 융합

> **핵심 변경**: 미사용이었던 `external` 데이터를 FiLM(Feature-wise Linear Modulation)으로 융합

```python
# FiLM 조건화: 외부환경(온도, 습도, 조도)이 센서+이미지 융합 결과를 조절
self.external_encoder = nn.Sequential(nn.Linear(3, embed_dim), nn.LayerNorm(embed_dim), nn.GELU())
self.film_gamma = nn.Linear(embed_dim, embed_dim)  # Scale
self.film_beta = nn.Linear(embed_dim, embed_dim)   # Shift

# forward():
ext_feat = self.external_encoder(externals)     # (B, T, D)
ext_pooled = ext_feat.mean(dim=1)               # (B, D)
gamma = self.film_gamma(ext_pooled)             # (B, D)
beta = self.film_beta(ext_pooled)               # (B, D)
pooled = gamma * pooled + beta                  # FiLM 적용
```

추가 변경:
- **Early Stopping** (PATIENCE=7) 적용
- **Test 세트 평가** 추가 (기존 Val만 평가)
- **Confusion Matrix** Test 세트 기준으로 변경

#### 05_llm_comparison.ipynb — 버그 수정

```python
# 수정 전 (BUG): review가 정의되지 않은 변수
if 'APPROVE' in str(review).upper():

# 수정 후 (FIXED): state에서 review 읽기
review = state.get('review', '')
if 'APPROVE' in str(review).upper():
```

### 16.5 MLFlow 학습 실행

```python
def train_with_mlflow(model_name="xgboost", experiment_name="conveyorguard-ml"):
    """
    사용법: python train.py --model xgboost --experiment xgboost-tuning

    1. 데이터 로드 (.joblib)
    2. 피처 추출 (센서 통계 + 이미지 통계)
       - sensor_mean, std, max, min, last → (N, 40)
       - img_mean, max, std → (N, 3)
       - 총 43개 피처
    3. MLFlow 실험 시작
    4. 모델 학습 (XGBoost/LightGBM/CatBoost/RandomForest)
    5. 평가 (Accuracy, F1)
    6. 모델 저장 (.pkl + MLFlow artifact)
    """
    tracker = MLFlowTracker(experiment_name)
    with tracker.start_run(run_name):
        tracker.log_params(model_params)
        model.fit(X_train, y_train)
        tracker.log_metrics({"test_accuracy": accuracy_score(y_test, y_pred)})
        tracker.log_model_sklearn(model, f"{model_name}_model")
```

### 16.5 모델 성능

| 모델 | 테스트 정확도 | F1 | 학습 시간 |
|------|-------------|-----|----------|
| **XGBoost** | **97.90%** | 97.85% | 2.9초 |
| LightGBM | 97.43% | - | 0.7초 |
| CatBoost | 96.44% | - | 15.0초 |
| RandomForest | 96.44% | - | 1.3초 |
| DL Transformer | 93.89% | - | 1,764초 |

> **주의**: 위 수치는 윈도우 단위 랜덤 split 기준입니다. `01_preprocess.ipynb`에서 세션 단위
> `GroupShuffleSplit`으로 변경한 후에는 데이터 누수가 제거되므로 성능 수치가 달라질 수 있습니다.
> 세션 단위 split이 실제 운영 환경의 일반화 성능을 더 정확하게 반영합니다.

---

# PART 4: 프론트엔드 (Next.js 15 + React 19 + TypeScript)

---

## 17. 프론트엔드 개요

```
프레임워크: Next.js 15 (App Router)
React:      19.2.3
TypeScript: 5
스타일:     TailwindCSS 4 + CSS Variables (다크 테마)
차트:       Recharts 3.6.0
아이콘:     Lucide React
클래스:     clsx (조건부 클래스)
라우팅:     파일 기반 (App Router)
상태관리:   React Hooks (useState, useEffect)
데이터:     Mock Data (하드코딩) + localStorage
```

### Listify와 비교

```
Listify                             ConveyorGuard
───────                             ─────────────
React (Vite, SPA)                   Next.js 15 (App Router)
view 상태로 페이지 전환               파일 기반 라우팅 (/equipment/[id])
Props drilling                      Props drilling (동일)
fetch (api.ts 래퍼)                  fetch (api.ts 직접 호출)
localStorage (토큰)                 localStorage (진단 히스토리)
Tailwind + Lucide                   Tailwind + Lucide + Recharts (동일)
```

---

## 18. 타입 정의 (types/index.ts)

```typescript
// === 열화 상태 ===
export type DegradationState = 0 | 1 | 2 | 3;

// === 장비 ===
export interface Equipment {
    id: string;                    // "OHT-001"
    name: string;                  // "OHT-001"
    type: "OHT" | "AGV";
    state: DegradationState;       // 0=정상, 1=경미, 2=중간, 3=심각
    confidence: number;
    sensors: SensorData;
    lastInspection: string;
    updatedAt: string;
}

// === 센서 데이터 ===
export interface SensorData {
    ntc: number;       // NTC 온도 (°C)
    pm1_0: number;     // PM1.0 (μg/m³)
    pm2_5: number;     // PM2.5
    pm10: number;      // PM10
    ct1: number;       // 전류1 (A)
    ct2: number;       // 전류2
    ct3: number;       // 전류3
    ct4: number;       // 전류4
}

// === 예측 응답 ===
export interface PredictResponse {
    prediction: DegradationState;
    label: string;               // "정상" / "경미한 열화" / ...
    confidence: number;          // 0~1
    probabilities: number[];     // [0.9, 0.05, 0.03, 0.02]
}

// === 상수 매핑 ===
export const STATE_LABELS: Record<DegradationState, string> = {
    0: "정상",
    1: "경미한 열화",
    2: "중간 열화",
    3: "심각한 열화",
};

export const STATE_COLORS: Record<DegradationState, string> = {
    0: "status-normal",     // #10B981
    1: "status-minor",      // #F59E0B
    2: "status-moderate",   // #F97316
    3: "status-severe",     // #EF4444
};
```

---

## 19. API 서비스 계층 (lib/api.ts)

### 19.1 환경변수

```typescript
const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
const LLM_API_BASE = process.env.NEXT_PUBLIC_LLM_API_URL || "http://localhost:8001";
```

### 19.2 전체 API 매핑

| 함수 | HTTP | 엔드포인트 | 대상 서비스 |
|------|------|-----------|------------|
| `fetchHealth()` | GET | `/api/v1/health` | ML API |
| `fetchModelInfo()` | GET | `/api/v1/model/info` | ML API |
| `predictTest()` | GET | `/api/v1/predict/test` | ML API |
| `predictTestDegraded()` | GET | `/api/v1/predict/test/degraded` | ML API |
| `predict(data)` | POST | `/api/v1/predict` | ML API |
| `requestDiagnosis(req)` | POST | `/api/v1/diagnose` | LLM Service |
| `checkLLMHealth()` | GET | `/health` | LLM Service |

### 19.3 API 인터페이스 (api.ts에 정의)

```typescript
export interface SensorData {
    ntc: number; pm1_0: number; pm2_5: number; pm10: number;
    ct1: number; ct2: number; ct3: number; ct4: number;
}

export interface DiagnosisRequest {
    equipment_id: string;
    prediction: string;
    confidence: number;
    sensors: SensorData;
    thermal_max_temp?: number;
}

export interface DiagnosisResponse {
    equipment_id: string;
    severity: string;
    anomalies: string[];
    probable_cause: string;
    recommended_action: string;
    similar_cases: SimilarCase[];
}
```

### Listify API 래퍼와 비교

```
Listify (api.ts 공통 래퍼)           ConveyorGuard (개별 함수)
──────────────────────               ─────────────────────────
async function request<T>(           // 공통 래퍼 없음
  method, endpoint, data             // 각 함수가 직접 fetch 호출
): Promise<T>
→ 자동 Authorization 헤더            // 인증 없으므로 불필요
→ 자동 Content-Type                  → 필요한 함수만 Content-Type 설정
```

---

## 20. 페이지 구조 (App Router)

```
src/app/
├── layout.tsx          # 루트 레이아웃 (html, body, 메타데이터)
├── globals.css         # CSS Variables + 애니메이션
├── page.tsx            # / → 대시보드 (메인 페이지)
└── equipment/
    └── [id]/
        └── page.tsx    # /equipment/OHT-001 → 장비 상세/진단
```

---

## 21. 대시보드 페이지 (page.tsx)

### 21.1 상태 변수

| 상태 | 타입 | 설명 |
|------|------|------|
| `apiStatus` | `"loading" \| "connected" \| "error"` | ML API 연결 상태 |
| `testResult` | `PredictResponse \| null` | API 테스트 예측 결과 |
| `search` | `string` | 장비 검색어 |
| `filter` | `FilterType` | 필터 (all/oht/agv/warning) |
| `sort` | `SortType` | 정렬 (state/name/temp) |

### 21.2 Mock 데이터

```typescript
const mockEquipment = [
    { id: "OHT-001", name: "OHT-001", type: "OHT", state: 0, temperature: 32, current: 45 },
    { id: "OHT-002", name: "OHT-002", type: "OHT", state: 1, temperature: 48, current: 67 },
    { id: "OHT-003", name: "OHT-003", type: "OHT", state: 3, temperature: 78, current: 142 },
    { id: "OHT-004", name: "OHT-004", type: "OHT", state: 2, temperature: 58, current: 89 },
    { id: "OHT-005", name: "OHT-005", type: "OHT", state: 0, temperature: 30, current: 48 },
];
```

### 21.3 useEffect

```typescript
useEffect(() => {
    // 마운트 시 ML API 테스트 호출
    predictTest()
        .then((res) => { setTestResult(res); setApiStatus("connected"); })
        .catch(() => setApiStatus("error"));
}, []);
```

### 21.4 필터/정렬 로직

```typescript
// 필터링: 검색어 + 타입 + 상태
const filtered = mockEquipment.filter((eq) => {
    if (search && !eq.name.toLowerCase().includes(search.toLowerCase())) return false;
    if (filter === "oht" && eq.type !== "OHT") return false;
    if (filter === "agv" && eq.type !== "AGV") return false;
    if (filter === "warning" && eq.state === 0) return false;
    return true;
});

// 정렬: 상태순(심각→정상) / 이름순 / 온도순
const sorted = [...filtered].sort((a, b) => {
    if (sort === "state") return b.state - a.state;
    if (sort === "name") return a.name.localeCompare(b.name);
    if (sort === "temp") return b.temperature - a.temperature;
    return 0;
});
```

### 21.5 레이아웃

```
┌────────────────────────────────────────────────────────┐
│ Header (로고, 네비게이션, 알림, 프로필)                  │
├────────────────────────────────────────────────────────┤
│ 대시보드                               API: 연결됨 ●   │
├──────┬──────┬──────┬──────────────────────────────────┤
│전체 5│정상 2│주의 2│위험 1                              │ ← SummaryCard × 4
├──────┴──────┴──────┴──────────────────────────────────┤
│                                                        │
│  실시간 장비 상태          │  AI 예측 테스트            │
│  [검색] [전체|OHT|AGV|이상] [정렬]│                    │
│  ┌──────┐ ┌──────┐ ┌──────┐ │  예측: 정상              │
│  │OHT-01│ │OHT-02│ │OHT-03│ │  신뢰도: 97.2%          │
│  │정상   │ │경미   │ │심각   │ │  ┌정상  ████████ 97%│  │
│  │32°C   │ │48°C   │ │78°C   │ │  ├경미  █       2% │  │
│  └──────┘ └──────┘ └──────┘ │  ├중간  ▏       0% │  │
│  ┌──────┐ ┌──────┐          │  └심각  ▏       1% │  │
│  │OHT-04│ │OHT-05│          │                      │
│  │중간   │ │정상   │          │                      │
│  └──────┘ └──────┘          │                      │
└────────────────────────────────────────────────────────┘
```

---

## 22. 장비 상세 페이지 (equipment/[id]/page.tsx)

### 22.1 상태 변수

| 상태 | 타입 | 설명 |
|------|------|------|
| `diagnosis` | `DiagnosisResponse \| null` | AI 진단 결과 |
| `loading` | `boolean` | 진단 로딩 |
| `dataLoading` | `boolean` | 초기 데이터 로딩 (1초 딜레이) |
| `error` | `string \| null` | 에러 메시지 |
| `lastUpdate` | `Date` | 마지막 업데이트 시각 |
| `historyKey` | `number` | DiagnosisHistory 리렌더 트리거 |
| `isRefreshing` | `boolean` | 새로고침 아이콘 회전 |
| `trendData` | `Record<string, TrendData[]>` | 센서 트렌드 차트 데이터 |
| `toasts` | `ToastData[]` | 토스트 알림 목록 |

### 22.2 센서 임계값 설정

```typescript
const sensorThresholds = {
    ntc:  { min: 20, max: 80,  threshold: 50, unit: "°C" },
    pm1_0:{ min: 0,  max: 100, threshold: 35, unit: "μg/m³" },
    pm2_5:{ min: 0,  max: 100, threshold: 35, unit: "μg/m³" },
    pm10: { min: 0,  max: 150, threshold: 75, unit: "μg/m³" },
    ct1:  { min: 0,  max: 10,  threshold: 5,  unit: "A" },
    ct2:  { min: 0,  max: 10,  threshold: 5,  unit: "A" },
    ct3:  { min: 0,  max: 10,  threshold: 5,  unit: "A" },
    ct4:  { min: 0,  max: 10,  threshold: 5,  unit: "A" },
};
```

### 22.3 useEffect 실행 순서

```
1. 마운트 → 1초 딜레이 후 dataLoading=false + 트렌드 데이터 생성
2. 10초 간격 자동 업데이트 (setInterval)
   → lastUpdate 갱신 + 트렌드 데이터 재생성
```

### 22.4 AI 진단 핸들러

```typescript
const handleDiagnosis = async () => {
    setLoading(true);
    setError(null);
    try {
        // 1. LLM 서비스 호출
        const res = await requestDiagnosis({
            equipment_id: id,
            prediction: label,        // "심각" 등
            confidence: 0.85,
            sensors: eq.sensors,
            thermal_max_temp: eq.thermalMax,
        });

        // 2. 결과 저장
        setDiagnosis(res);

        // 3. 히스토리 localStorage에 저장
        saveDiagnosisToHistory(id, res.severity, res.probable_cause);
        setHistoryKey((prev) => prev + 1);  // 리렌더 트리거

        // 4. 심각도별 토스트 알림
        if (res.severity === "심각") {
            addToast(`⚠️ ${eq.name}: 심각한 이상 감지!`, "error");
        } else if (res.severity === "중간" || res.severity === "경미") {
            addToast(`${eq.name}: ${res.severity} 수준 이상 감지`, "warning");
        } else {
            addToast(`${eq.name}: 정상 상태입니다`, "success");
        }
    } catch (e) {
        setError(e instanceof Error ? e.message : "진단 실패");
        addToast("AI 진단 실패. LLM 서비스를 확인하세요.", "error");
    } finally {
        setLoading(false);
    }
};
```

### 22.5 레이아웃

```
┌────────────────────────────────────────────────────────────┐
│ ← OHT-003                          업데이트: 방금 전 ● 심각│ ← 헤더
│    OHT · FAB2-Zone A                         [🔄]         │
├────────────────────────────────────────────────────────────┤
│ 열화상 최고온도                                 85°C       │ ← TempHeatBar
│ ═══════════════════════════════════════●═══════════════    │
│ 20°C        40°C        60°C        80°C        100°C     │
├────────────────────────────────────────────────────────────┤
│ 센서 데이터 (실시간)                                       │
│ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐          │ ← SensorGauge × 8
│ │NTC 78°C │ │PM1.0 85 │ │PM2.5 102│ │PM10 128 │          │
│ │████████▒│ │████████▒│ │████████▒│ │████████▒│          │
│ │0  임:50 80│0  임:35 100│0  임:35 100│0 임:75 150│        │
│ ├─────────┤ ├─────────┤ ├─────────┤ ├─────────┤          │
│ │CT1 5.8A │ │CT2 6.2A │ │CT3 5.5A │ │CT4 6.8A │          │
│ │████████▒│ │████████▒│ │████████▒│ │████████▒│          │
│ └─────────┘ └─────────┘ └─────────┘ └─────────┘          │
├──────────────────┬─────────────────┬──────────────────────┤
│ NTC 추이         │ PM2.5 추이      │ CT1 추이             │ ← SensorTrendChart × 3
│  ╱╲    --- 임계50│      --- 임계35 │        --- 임계5     │
│ ╱  ╲╱╲          │    ╱╲           │   ╱╲                 │
├──────────────────┴─────────────────┴──────────────────────┤
│                              │                            │
│  AI 진단 리포트    [AI진단실행] │  진단 히스토리             │
│                              │  최근 5건                   │
│  심각도: 심각                 │  심각│베어링 마모│1/15 11:00 │
│                              │  중간│필터 막힘  │1/14 15:30 │
│  이상 징후                    │  정상│정상      │1/14 09:00 │
│  ⚠️ NTC 78°C - 임계초과       │                            │
│  ⚠️ CT1~CT4 전류 임계초과     │  ┌───┬───┬───┬───┐        │
│                              │  │ 2 │ 1 │ 3 │ 1 │        │
│  추정 원인                    │  │정상│경미│중간│심각│        │
│  ┌────────────────────┐      │  └───┴───┴───┴───┘        │
│  │ 베어링 마모로 인한   │      │                            │
│  │ 과열 및 전류 증가    │      │                            │
│  └────────────────────┘      │                            │
│                              │                            │
│  권장 조치                    │                            │
│  █ 즉시 가동 중단 후          │                            │
│  █ 베어링 교체 필요           │                            │
│                              │                            │
└──────────────────────────────┴────────────────────────────┘
```

---

## 23. 컴포넌트 트리

```
App Router
│
├── layout.tsx (루트 레이아웃: <html lang="ko">)
│
├── page.tsx (대시보드)
│   ├── Header.tsx (상단 네비게이션)
│   │   └── Lucide: Activity, Bell, User
│   ├── SummaryCard.tsx × 4 (전체/정상/주의/위험)
│   └── StatusCard.tsx × N (장비 카드)
│       └── Lucide: Thermometer, Zap
│
└── equipment/[id]/page.tsx (장비 상세)
    ├── ToastContainer.tsx (우상단 알림)
    │   └── Toast.tsx × N
    ├── TempHeatBar.tsx (열화상 그라디언트)
    ├── SensorGauge.tsx × 8 (센서 게이지)
    │   └── [또는] SensorGaugeSkeleton × 8
    ├── SensorTrendChart.tsx × 3 (Recharts 라인차트)
    │   └── [또는] CardSkeleton × 3
    ├── AI 진단 리포트 (인라인)
    │   └── 심각도 + 이상징후 + 추정원인 + 권장조치
    └── DiagnosisHistory.tsx (localStorage 기반)
```

---

## 24. 컴포넌트 Props 상세

### Header

```typescript
// Props: 없음 (자체 렌더링)
// 네비게이션: 대시보드, 장비목록, 알림내역, 설정
// 아이콘: Activity (로고), Bell (알림배지=3), User (프로필)
```

### SummaryCard

```typescript
interface SummaryCardProps {
    title: string;                                    // "전체 장비", "정상", ...
    value: number;                                    // 5, 2, 2, 1
    color?: "normal" | "minor" | "moderate" | "severe" | "default";
}
// 렌더: 라벨 + 4xl 볼드 숫자 (color별 텍스트 색상)
```

### StatusCard

```typescript
interface StatusCardProps {
    id: string;                      // "OHT-001"
    name: string;                    // "OHT-001"
    type: "OHT" | "AGV";
    state: DegradationState;         // 0~3
    temperature: number;             // °C
    current: number;                 // A
    onClick?: () => void;            // → router.push(`/equipment/${id}`)
}
// 렌더: 상태 컬러 상단바 + 이름/타입 + 상태라벨 + 온도/전류
// 심각(state=3) 시 status-pulse 애니메이션
```

### SensorGauge

```typescript
interface SensorGaugeProps {
    label: string;        // "NTC", "PM2.5", "CT1" 등
    value: number;        // 현재값
    min: number;          // 최소값
    max: number;          // 최대값
    unit: string;         // "°C", "μg/m³", "A"
    threshold: number;    // 임계값 (초과 시 빨강)
}
// 렌더: 라벨 + 현재값 + 프로그레스바 + min/임계/max 표시
// isWarning(value > threshold) 시 빨간색
```

### SensorTrendChart

```typescript
interface SensorTrendChartProps {
    label: string;        // "NTC (온도)"
    data: TrendData[];    // [{time: "11:00", value: 78.2}, ...]
    threshold: number;    // 임계 참조선
    unit: string;         // "°C"
    color?: string;       // 기본 "#10B981"
}
// Recharts: LineChart + XAxis + YAxis + Tooltip + ReferenceLine(임계)
// 마지막 값이 임계 초과 시 라인 빨간색
```

### TempHeatBar

```typescript
interface TempHeatBarProps {
    temp: number;         // 열화상 최고온도
}
// 렌더: 그라디언트 바 (파랑→초록→주황→빨강) + 흰색 마커
// percent = (temp - 20) / (100 - 20) * 100
```

### DiagnosisHistory

```typescript
interface DiagnosisHistoryProps {
    equipmentId: string;  // "OHT-003"
}
// localStorage 키: `diagnosis_history_${equipmentId}`
// 렌더: 최근 10건 리스트 + 심각도별 카운트 (4그리드)
```

### Toast / ToastContainer / useToast

```typescript
interface ToastData {
    id: string;
    message: string;
    type: "success" | "error" | "warning" | "info";
}

// useToast 훅: addToast(message, type) / removeToast(id)
// 자동 4초 후 사라짐
// 우상단 슬라이드-인 애니메이션
```

### Skeleton 변형

```typescript
// Skeleton: 기본 사각형 (animate-pulse)
// SensorGaugeSkeleton: 게이지 모양 플레이스홀더
// CardSkeleton: 카드 모양 플레이스홀더
// StatusCardSkeleton: 상태 카드 모양 플레이스홀더
```

---

## 25. 디자인 시스템 (globals.css + tailwind.config.ts)

### CSS Variables (다크 테마 — GitHub 기반)

```css
@theme {
    --color-bg-primary: #0D1117;       /* 최상위 배경 */
    --color-bg-secondary: #161B22;     /* 카드/섹션 배경 */
    --color-bg-tertiary: #21262D;      /* 입력/호버 배경 */
    --color-border: #30363D;           /* 테두리 */
    --color-status-normal: #10B981;    /* 정상 (초록) */
    --color-status-minor: #F59E0B;     /* 경미 (노랑) */
    --color-status-moderate: #F97316;  /* 중간 (주황) */
    --color-status-severe: #EF4444;    /* 심각 (빨강) */
    --color-brand-primary: #3B82F6;    /* 주 브랜드 (파랑) */
    --color-brand-accent: #06B6D4;     /* 보조 (시안) */
    --color-text-primary: #E6EDF3;     /* 주 텍스트 */
    --color-text-muted: #8B949E;       /* 보조 텍스트 */
    --font-sans: 'Pretendard', -apple-system, sans-serif;
    --font-mono: 'JetBrains Mono', monospace;
}
```

### 커스텀 애니메이션

```css
/* 심각 상태 펄스 */
@utility status-pulse {
    animation: status-pulse 2s ease-in-out infinite;
}
@keyframes status-pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.7; } }

/* 토스트 슬라이드-인 */
.animate-slide-in { animation: slide-in 0.3s ease-out; }
@keyframes slide-in { from { transform: translateX(100%); opacity: 0; } to { ... } }
```

---

## 26. 데이터 저장 (localStorage)

```
localStorage:
└── diagnosis_history_{equipmentId}   # JSON 배열 (최대 100건)
    [{
        id: "1706345600000",          # Date.now() 문자열
        date: "1. 15. 오전 11:00",    # 한국어 포맷
        severity: "심각",
        cause: "베어링 마모로 인한 과열"
    }, ...]
```

### 저장 함수

```typescript
export function saveDiagnosisToHistory(equipmentId: string, severity: string, cause: string) {
    const key = `diagnosis_history_${equipmentId}`;
    const history: HistoryItem[] = JSON.parse(localStorage.getItem(key) || "[]");

    history.unshift({
        id: Date.now().toString(),
        date: new Date().toLocaleDateString("ko-KR", { month: "numeric", day: "numeric", hour: "2-digit", minute: "2-digit" }),
        severity,
        cause,
    });

    localStorage.setItem(key, JSON.stringify(history.slice(0, 100)));  // 최대 100건
}
```

---

# PART 5: 배포 (Docker)

---

## 27. Docker 구성

### 27.1 docker-compose.yml

```yaml
version: '3.8'

services:
  ml-api:
    build:
      context: ./conveyorguard-api
      dockerfile: Dockerfile
    container_name: conveyorguard-ml-api
    ports:
      - "8000:8000"
    volumes:
      - ./data/models:/app/models:ro       # 모델 파일 (읽기전용)
    environment:
      - MODEL_PATH=/app/models/best_model.pt
      - LOG_LEVEL=INFO
      - MLFLOW_TRACKING_URI=http://mlflow:5000
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/api/v1/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    restart: unless-stopped
    depends_on:
      - mlflow

  mlflow:
    image: ghcr.io/mlflow/mlflow:v2.9.2
    container_name: conveyorguard-mlflow
    ports:
      - "5000:5000"
    volumes:
      - ./mlruns:/mlruns
      - ./data/models:/models
    command: >
      mlflow server --host 0.0.0.0 --port 5000
      --backend-store-uri sqlite:///mlruns/mlflow.db
      --default-artifact-root /mlruns
    restart: unless-stopped

  # llm-service:                          # 선택적 활성화
  #   build: ./llm-service
  #   ports: ["8001:8001"]
  #   environment: [GEMINI_API_KEY=${GEMINI_API_KEY}]
```

### 27.2 ML API Dockerfile

```dockerfile
FROM python:3.10-slim
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

ENV MODEL_PATH=/app/models/best_model.pt
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Listify Docker와 비교

```
Listify                              ConveyorGuard
───────                              ─────────────
Frontend: Node→Build→Nginx(80)       Frontend: Next.js (개발서버, Docker 미설정)
Backend: Python Flask(5001)          ML API: Python FastAPI(8000)
DB: host.docker.internal MySQL       DB: 없음 (모델 파일만 마운트)
                                     MLFlow: SQLite(5000)
                                     LLM: 선택적(8001)
```

---

# PART 6: 요청 처리 시나리오

---

## 28. 시나리오 1: 대시보드 로드

```
브라우저 → GET /
  → page.tsx 마운트
    → useEffect: predictTest()
      → GET http://localhost:8000/api/v1/predict/test
      → ML API router.py: predict_test()
        → model_loader.is_loaded() 확인
        → 더미 데이터 생성 (정상 범위)
        → preprocess_input(normalize=True)
        → model.predict(sensors, images, external)
      ← PredictResponse { prediction:0, label:"정상", confidence:0.97, probabilities:[...] }
    → setTestResult(res)
    → setApiStatus("connected")
  → 렌더: SummaryCard×4 + StatusCard×5 + 예측 결과 패널
```

## 29. 시나리오 2: 장비 상세 진입 + 자동 업데이트

```
StatusCard 클릭 → router.push("/equipment/OHT-003")
  → equipment/[id]/page.tsx 마운트
    → params.id = "OHT-003"
    → mockData["OHT-003"] → { state:3, sensors:{ntc:78, ...}, thermalMax:85 }

    → useEffect[1]: 1초 딜레이 → setDataLoading(false)
      → generateMockTrendData(78, 3)  → ntc 트렌드 12포인트
      → generateMockTrendData(102, 5) → pm2_5 트렌드
      → generateMockTrendData(5.8, 0.5) → ct1 트렌드

    → useEffect[2]: setInterval(10000)
      → 매 10초: lastUpdate 갱신 + 트렌드 재생성

  → 렌더: TempHeatBar + SensorGauge×8 + SensorTrendChart×3 + DiagnosisHistory
```

## 30. 시나리오 3: AI 진단 실행

```
"AI 진단 실행" 클릭 → handleDiagnosis()
  → setLoading(true)
  → requestDiagnosis({
      equipment_id: "OHT-003",
      prediction: "심각한 열화",
      confidence: 0.85,
      sensors: { ntc:78, pm1_0:85, pm2_5:102, pm10:128, ct1:5.8, ct2:6.2, ct3:5.5, ct4:6.8 },
      thermal_max_temp: 85
    })
    → POST http://localhost:8001/api/v1/diagnose
    → LLM router.py: diagnose()
      → generate_diagnosis()
        → Gemini API 호출 (프롬프트: 센서 + 임계값 + 분석 지시)
        ← { anomalies: [...], probable_cause: "...", recommended_action: "..." }
      → find_similar_cases()
        → 코사인 유사도 계산 (8채널 벡터)
        → 유사도 순 Top-3 반환
    ← DiagnosisResponse { severity:"심각", anomalies, probable_cause, recommended_action, similar_cases }

  → setDiagnosis(res)
  → saveDiagnosisToHistory("OHT-003", "심각", "베어링 마모...")
    → localStorage 저장
  → setHistoryKey(prev+1) → DiagnosisHistory 리렌더
  → addToast("⚠️ OHT-003: 심각한 이상 감지!", "error")
  → setLoading(false)
```

## 31. 시나리오 4: LangGraph 진단 (고급)

```
POST /api/v1/diagnose/graph
  → run_diagnosis(equipment_id, prediction_result, sensor_data)
    → StateGraph 생성

    → [1단계] analyzer_node
      → Gemini: "NTC=78°C 임계초과, CT1~4 전류 임계초과 분석"
      ← analysis: "온도 및 전류 동시 상승으로 기계적 마모 의심..."

    → [2단계] diagnostician_node
      → Gemini: "원인/긴급도/조치방안 제시"
      ← diagnosis: "베어링 마모 → 즉시 가동 중단 → 교체 필요"

    → [3단계] reviewer_node
      → Gemini: "분석+진단 검토 → APPROVE/REVISE"
      ← review: "분석 적절함. APPROVE", review_count: 1

    → should_continue: "APPROVE" 포함 → "finalize"

    → [4단계] finalize_node
      ← final_report: "# 진단 리포트\n## 장비: OHT-003\n..."
      ← status: "approved"
```

---

# PART 7: 다른 프로젝트 적용 가이드

---

## 32. 재사용 가능한 모듈

### 32.1 그대로 복사 가능 (수정 최소)

| 파일 | 대상 | 수정 사항 |
|------|------|-----------|
| `conveyorguard-api/app/main.py` | FastAPI 진입점 | 서비스명, 모델 경로 변경 |
| `conveyorguard-api/app/core/loader.py` | 모델 로더 싱글톤 | 모델 클래스 변경 |
| `frontend/src/components/ui/Toast.tsx` | 토스트 알림 시스템 | 그대로 사용 |
| `frontend/src/components/ui/Skeleton.tsx` | 로딩 플레이스홀더 | 그대로 사용 |
| `frontend/src/app/globals.css` | 다크 테마 CSS Variables | 색상만 변경 |
| `frontend/tailwind.config.ts` | Tailwind 확장 설정 | 색상/폰트만 변경 |
| `llm-service/app/monitoring/langsmith_config.py` | 로깅/메트릭 | 그대로 사용 |
| `docker-compose.yml` | 멀티서비스 오케스트레이션 | 서비스명/포트 변경 |
| `Dockerfile` | FastAPI 컨테이너 | 그대로 사용 |

### 32.2 프로젝트별 새로 작성

| 항목 | 설명 |
|------|------|
| `api/schemas.py` | Pydantic 요청/응답 모델 (도메인별) |
| `api/router.py` | API 엔드포인트 + 비즈니스 로직 |
| `core/model.py` | AI 모델 아키텍처 |
| `core/preprocessing.py` | 데이터 전처리 파이프라인 |
| `core/prompts.py` | LLM 프롬프트 템플릿 |
| `core/rag.py` | RAG 사례 DB + 유사도 함수 |
| `agents/*.py` | LangGraph 에이전트 그래프 |
| `tools/*.py` | LangChain Tool 정의 |
| `types/index.ts` | TypeScript 타입 정의 |
| `lib/api.ts` | API 클라이언트 함수 |
| `app/page.tsx` | 메인 페이지 |
| `components/dashboard/*.tsx` | 도메인 특화 컴포넌트 |

### 32.3 새 프로젝트 셋업 순서

```
1단계: ML API 기반 인프라
  ✓ app/main.py          → FastAPI + Lifespan
  ✓ app/core/loader.py   → 모델 로더 싱글톤
  ✓ app/core/model.py    → AI 모델 정의
  ✓ app/core/preprocessing.py → 전처리
  ✓ app/api/schemas.py   → Pydantic 스키마
  ✓ app/api/router.py    → API 엔드포인트
  ✓ Dockerfile           → 컨테이너

2단계: LLM 서비스 (선택)
  ✓ app/main.py          → FastAPI
  ✓ app/core/gemini.py   → Gemini 연동
  ✓ app/core/prompts.py  → 프롬프트 템플릿
  ✓ app/core/rag.py      → RAG 유사 사례
  ✓ app/agents/*.py      → LangGraph 에이전트
  ✓ app/tools/*.py       → LangChain Tools

3단계: 프론트엔드
  ✓ src/types/index.ts   → 타입 정의
  ✓ src/lib/api.ts       → API 클라이언트
  ✓ src/app/page.tsx     → 대시보드
  ✓ src/app/{detail}/page.tsx → 상세 페이지
  ✓ src/components/      → UI 컴포넌트

4단계: 배포
  ✓ docker-compose.yml
  ✓ Dockerfile
```

---

## 33. 환경변수 목록

```bash
# ML API
MODEL_PATH=./data/models/best_model.pt
LOG_LEVEL=INFO
MLFLOW_TRACKING_URI=http://localhost:5000

# LLM Service
GEMINI_API_KEY=your-gemini-api-key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your-langsmith-key    # 선택
LANGCHAIN_PROJECT=conveyorguard

# Frontend (빌드 시)
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_LLM_API_URL=http://localhost:8001
```

---

## 34. Import 의존성 맵

```
conveyorguard-api/
├── app/main.py
│   ├── app/api/router.py
│   │   ├── app/api/schemas.py              (STATE_LABELS, 6개 모델)
│   │   ├── app/core/loader.py → model_loader (싱글톤)
│   │   │   └── app/core/model.py           (MultimodalTransformer, Config)
│   │   └── app/core/preprocessing.py       (preprocess_input)
│   └── app/core/loader.py                  (lifespan에서 load)

llm-service/
├── app/main.py
│   └── app/api/router.py
│       ├── app/core/gemini.py              (generate_diagnosis)
│       ├── app/core/rag.py                 (find_similar_cases)
│       ├── app/core/prompts.py             (get_diagnosis_prompt)
│       ├── app/agents/diagnosis_graph.py   (LangGraph)
│       │   └── langchain_google_genai
│       ├── app/rag/case_retriever.py       (FAISS)
│       │   └── langchain_community (HuggingFace + FAISS)
│       ├── app/tools/diagnosis_tools.py    (5개 LangChain Tools)
│       └── app/monitoring/langsmith_config.py (Logger)

frontend/
├── src/app/page.tsx
│   ├── src/components/layout/Header.tsx
│   ├── src/components/dashboard/SummaryCard.tsx
│   ├── src/components/dashboard/StatusCard.tsx
│   ├── src/lib/api.ts → ML API (8000)
│   └── src/types/index.ts
│
└── src/app/equipment/[id]/page.tsx
    ├── src/lib/api.ts → LLM Service (8001)
    ├── src/components/dashboard/SensorGauge.tsx
    ├── src/components/dashboard/SensorTrendChart.tsx (Recharts)
    ├── src/components/dashboard/TempHeatBar.tsx
    ├── src/components/dashboard/DiagnosisHistory.tsx (localStorage)
    ├── src/components/ui/Toast.tsx (useToast 훅)
    └── src/components/ui/Skeleton.tsx
```

---

## 35. Listify → ConveyorGuard 아키텍처 전환 요약

| Listify 계층 | ConveyorGuard 대응 | 차이점 |
|-------------|-------------------|--------|
| `app.py` (Flask) | `main.py` (FastAPI × 2) | 마이크로서비스, Lifespan, 비동기 |
| `db.py` (MySQL Pool) | `loader.py` (Model 싱글톤) | DB 대신 모델 파일 관리 |
| `middleware/auth_utils.py` | 없음 | 인증 미구현 |
| `routes/*.py` | `router.py` 통합 | FastAPI APIRouter (Routes+Controller 합침) |
| `controllers/*.py` | `router.py` 통합 | Pydantic 자동 검증으로 Controller 역할 축소 |
| `services/*.py` | `preprocessing.py` + `gemini.py` + `rag.py` | 도메인 특화 서비스 |
| `model/*.py` (SQL) | `model.py` (PyTorch) | SQL → 텐서 연산 |
| `types.ts` | `types/index.ts` | 도메인 타입 변경 |
| `api.ts` (공통 래퍼) | `api.ts` (개별 함수) | 인증 헤더 불필요 |
| `App.tsx` (SPA 전체) | `page.tsx` × 2 (App Router) | 파일 기반 라우팅 |
| `Login/Register.tsx` | 없음 | 인증 미구현 |
| `MySQL` | 없음 | localStorage + 모델 파일 |
| `Spotify API` | `Gemini API` | 음악 검색 → AI 진단 |
