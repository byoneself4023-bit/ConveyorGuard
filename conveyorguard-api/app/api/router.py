"""
ConveyorGuard - API Router
"""

import numpy as np
import torch
from fastapi import APIRouter, HTTPException, status
import logging

from app.api.schemas import (
    PredictRequest, PredictResponse,
    BatchPredictRequest, BatchPredictResponse,
    HealthResponse, ModelInfoResponse, STATE_LABELS
)
from app.core.loader import model_loader
from app.core.preprocessing import preprocess_input

logger = logging.getLogger(__name__)
router = APIRouter()


# ============== 입력 검증 함수 ==============

def validate_input_shape(sensors, images, external):
    """입력 데이터 shape 검증"""
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
        raise HTTPException(
            status_code=422,
            detail={
                "message": "입력 데이터 shape이 올바르지 않습니다",
                "errors": errors,
                "expected_shape": {
                    "sensors": [30, 8],
                    "images": [30, 60, 80],
                    "external": [30, 3]
                }
            }
        )


# ============== 기존 엔드포인트 ==============

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """서버 헬스 체크"""
    return HealthResponse(
        status="healthy",
        model_loaded=model_loader.is_loaded(),
        device=str(model_loader.device) if model_loader.is_loaded() else "not loaded"
    )


@router.get("/model/info", response_model=ModelInfoResponse)
async def get_model_info():
    """모델 정보 조회"""
    if not model_loader.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    info = model_loader.info
    return ModelInfoResponse(
        accuracy=info.get('best_accuracy', 0),
        parameters=info.get('parameters', 0),
        device=str(model_loader.device),
        input_shape={"sensors": [30, 8], "images": [30, 60, 80], "external": [30, 3]},
        output_classes=STATE_LABELS
    )


@router.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """단일 예측"""
    if not model_loader.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # 입력 검증 추가
    validate_input_shape(request.sensors, request.images, request.external)
    
    try:
        processed = preprocess_input(
            request.sensors, request.images, request.external, request.normalize
        )
        
        device = model_loader.device
        sensors = torch.tensor(processed['sensors'], dtype=torch.float32).unsqueeze(0).to(device)
        images = torch.tensor(processed['images'], dtype=torch.float32).unsqueeze(0).to(device)
        external = torch.tensor(processed['external'], dtype=torch.float32).unsqueeze(0).to(device)
        
        result = model_loader.model.predict(sensors, images, external)
        
        pred = result['prediction'].item()
        conf = result['confidence'].item()
        probs = result['probabilities'].squeeze().tolist()
        
        return PredictResponse(
            prediction=pred,
            label=STATE_LABELS[pred],
            confidence=round(conf, 4),
            probabilities=[round(p, 4) for p in probs]
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict/batch", response_model=BatchPredictResponse)
async def predict_batch(request: BatchPredictRequest):
    """배치 예측"""
    if not model_loader.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # 각 아이템 입력 검증
    for i, item in enumerate(request.items):
        try:
            validate_input_shape(item.sensors, item.images, item.external)
        except HTTPException as e:
            raise HTTPException(
                status_code=422,
                detail={"message": f"items[{i}] 검증 실패", "errors": e.detail}
            )
    
    try:
        device = model_loader.device
        all_sensors, all_images, all_external = [], [], []
        
        for item in request.items:
            processed = preprocess_input(
                item.sensors, item.images, item.external, item.normalize
            )
            all_sensors.append(processed['sensors'])
            all_images.append(processed['images'])
            all_external.append(processed['external'])
        
        sensors = torch.tensor(np.stack(all_sensors), dtype=torch.float32).to(device)
        images = torch.tensor(np.stack(all_images), dtype=torch.float32).to(device)
        external = torch.tensor(np.stack(all_external), dtype=torch.float32).to(device)
        
        result = model_loader.model.predict(sensors, images, external)
        
        results = []
        for i in range(len(request.items)):
            pred = result['prediction'][i].item()
            results.append(PredictResponse(
                prediction=pred,
                label=STATE_LABELS[pred],
                confidence=round(result['confidence'][i].item(), 4),
                probabilities=[round(p, 4) for p in result['probabilities'][i].tolist()]
            ))
        
        return BatchPredictResponse(results=results, total=len(results))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============== 테스트 엔드포인트 (NEW) ==============

@router.get("/predict/test", response_model=PredictResponse)
async def predict_test():
    """
    🧪 더미 데이터로 예측 테스트
    
    Swagger UI에서 쉽게 테스트할 수 있도록 더미 데이터를 사용합니다.
    - sensors: 정상 범위의 센서값 (NTC=25℃, PM=10~20, CT=50A)
    - images: 균일한 열화상 (30℃)
    - external: 정상 환경 (25℃, 50%, 500lux)
    """
    if not model_loader.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # 더미 데이터 생성 (정상 상태 시뮬레이션)
    dummy_sensors = [[25, 10, 15, 20, 50, 50, 50, 50]] * 30  # (30, 8)
    dummy_images = [[[30.0] * 80 for _ in range(60)] for _ in range(30)]  # (30, 60, 80)
    dummy_external = [[25, 50, 500]] * 30  # (30, 3)
    
    try:
        processed = preprocess_input(dummy_sensors, dummy_images, dummy_external, normalize=True)
        
        device = model_loader.device
        sensors = torch.tensor(processed['sensors'], dtype=torch.float32).unsqueeze(0).to(device)
        images = torch.tensor(processed['images'], dtype=torch.float32).unsqueeze(0).to(device)
        external = torch.tensor(processed['external'], dtype=torch.float32).unsqueeze(0).to(device)
        
        result = model_loader.model.predict(sensors, images, external)
        
        pred = result['prediction'].item()
        conf = result['confidence'].item()
        probs = result['probabilities'].squeeze().tolist()
        
        return PredictResponse(
            prediction=pred,
            label=STATE_LABELS[pred],
            confidence=round(conf, 4),
            probabilities=[round(p, 4) for p in probs]
        )
    except Exception as e:
        logger.error(f"Test prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/predict/test/degraded", response_model=PredictResponse)
async def predict_test_degraded():
    """
    🧪 열화 상태 시뮬레이션 테스트
    
    비정상적인 센서값으로 열화 상태를 시뮬레이션합니다.
    - sensors: 높은 온도(80℃), 높은 미세먼지, 높은 전류
    - images: 고온 열화상 (80℃)
    - external: 고온 환경 (40℃)
    """
    if not model_loader.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # 열화 상태 시뮬레이션
    dummy_sensors = [[80, 200, 300, 400, 150, 150, 150, 150]] * 30  # 높은 값들
    dummy_images = [[[80.0] * 80 for _ in range(60)] for _ in range(30)]  # 고온
    dummy_external = [[40, 80, 200]] * 30  # 고온/고습도
    
    try:
        processed = preprocess_input(dummy_sensors, dummy_images, dummy_external, normalize=True)
        
        device = model_loader.device
        sensors = torch.tensor(processed['sensors'], dtype=torch.float32).unsqueeze(0).to(device)
        images = torch.tensor(processed['images'], dtype=torch.float32).unsqueeze(0).to(device)
        external = torch.tensor(processed['external'], dtype=torch.float32).unsqueeze(0).to(device)
        
        result = model_loader.model.predict(sensors, images, external)
        
        pred = result['prediction'].item()
        conf = result['confidence'].item()
        probs = result['probabilities'].squeeze().tolist()
        
        return PredictResponse(
            prediction=pred,
            label=STATE_LABELS[pred],
            confidence=round(conf, 4),
            probabilities=[round(p, 4) for p in probs]
        )
    except Exception as e:
        logger.error(f"Degraded test prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))