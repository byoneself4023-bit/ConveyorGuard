"""
ConveyorGuard ML API Server
이송장치 열화 예측 API
"""

import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from app.api.router import router
from app.core.loader import model_loader
from app.routers import equipment, sensors, diagnosis, alerts, stats, auth

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """앱 시작 시 모델 로드"""
    logger.info("🚀 Starting ConveyorGuard ML API...")
    
    model_path = os.getenv("MODEL_PATH", "../data/models/baseline_cnn_model.pt")
    
    try:
        model_loader.load(model_path)
        logger.info("✅ Model loaded successfully")
    except FileNotFoundError:
        logger.warning(f"⚠️ Model not found: {model_path}")
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
    
    yield
    logger.info("👋 Shutting down...")


app = FastAPI(
    title="ConveyorGuard ML API",
    description="이송장치 열화 예측 AI API (CNN+Transformer 3-modal, 정확도: 93.24%)",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1", tags=["Prediction"])
app.include_router(equipment.router, prefix="/api/v1")
app.include_router(sensors.router, prefix="/api/v1")
app.include_router(diagnosis.router, prefix="/api/v1")
app.include_router(alerts.router, prefix="/api/v1")
app.include_router(stats.router, prefix="/api/v1")
app.include_router(auth.router, prefix="/api/v1")


@app.get("/")
async def root():
    return {
        "service": "ConveyorGuard ML API",
        "version": "1.0.0",
        "docs": "/docs"
    }


# ============================================
# WebSocket + Simulator
# ============================================
from app.services.simulator import simulator


@app.post("/api/v1/simulator/start")
async def start_simulator():
    """시뮬레이터 시작"""
    simulator.start()
    return {"success": True, "message": "Simulator started"}


@app.post("/api/v1/simulator/stop")
async def stop_simulator():
    """시뮬레이터 정지"""
    simulator.stop()
    return {"success": True, "message": "Simulator stopped"}


@app.get("/api/v1/simulator/status")
async def simulator_status():
    """시뮬레이터 상태"""
    return {"success": True, "running": simulator.running, "clients": len(simulator.clients)}


@app.websocket("/ws/sensors")
async def sensor_stream(websocket: WebSocket):
    """센서 데이터 실시간 스트림"""
    await simulator.connect(websocket)
    try:
        while True:
            # 클라이언트로부터 메시지 대기 (연결 유지용)
            await websocket.receive_text()
    except WebSocketDisconnect:
        simulator.disconnect(websocket)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
