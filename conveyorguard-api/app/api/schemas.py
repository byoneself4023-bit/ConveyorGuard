"""
ConveyorGuard - API Schemas
"""

from pydantic import BaseModel, Field
from typing import List, Optional

# 라벨 정본은 app.config (SSoT). 기존 import 경로 호환을 위해 re-export.
from app.config import STATE_LABELS


class PredictRequest(BaseModel):
    """단일 예측 요청"""
    sensors: List[List[float]] = Field(..., description="센서 시퀀스 (30, 8)")
    images: List[List[List[float]]] = Field(..., description="열화상 시퀀스 (30, 60, 80)")
    external: List[List[float]] = Field(..., description="외부환경 시퀀스 (30, 3)")
    normalize: bool = Field(default=True, description="정규화 여부")


class BatchPredictRequest(BaseModel):
    """배치 예측 요청"""
    items: List[PredictRequest] = Field(..., description="예측할 데이터 목록", max_length=32)


class PredictResponse(BaseModel):
    """예측 응답"""
    prediction: int = Field(..., description="예측 클래스 (0~3)")
    label: str = Field(..., description="예측 라벨")
    confidence: float = Field(..., description="예측 신뢰도")
    probabilities: List[float] = Field(..., description="클래스별 확률")


class BatchPredictResponse(BaseModel):
    """배치 예측 응답"""
    results: List[PredictResponse]
    total: int


class HealthResponse(BaseModel):
    """헬스 체크 응답"""
    status: str
    model_loaded: bool
    device: str


class ModelInfoResponse(BaseModel):
    """모델 정보 응답"""
    name: str = "MultimodalTransformer"
    version: str = "1.0.0"
    accuracy: float
    parameters: int
    device: str
    input_shape: dict
    output_classes: dict
