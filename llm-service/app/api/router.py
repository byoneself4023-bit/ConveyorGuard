from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from app.agents.diagnosis_graph import run_diagnosis

router = APIRouter()


class SensorData(BaseModel):
    ntc: float
    pm1_0: float
    pm2_5: float
    pm10: float
    ct1: float
    ct2: float
    ct3: float
    ct4: float


class DiagnosisRequest(BaseModel):
    equipment_id: str
    prediction: str
    confidence: float
    sensors: SensorData
    thermal_max_temp: Optional[float] = None


class SimilarCase(BaseModel):
    date: str = ""
    equipment_id: str = ""
    issue: str = ""
    action: str = ""
    similarity: float = 0.0


class DiagnosisResponse(BaseModel):
    equipment_id: str
    severity: str
    anomalies: List[str]
    probable_cause: str
    recommended_action: str
    similar_cases: List[SimilarCase]


# ========== 단일 진단 엔드포인트 (게이트 통과 시 4-에이전트 그래프 + 통합 RAG) ==========
@router.post("/diagnose", response_model=DiagnosisResponse)
async def diagnose(request: DiagnosisRequest):
    """4-에이전트 그래프(통합 FAISS RAG 참조)를 실행해 구조화 진단을 반환.

    게이트(예측>=2)는 호출자(conveyorguard-api)에 있으므로, 이 엔드포인트는
    게이트 통과 입력에만 도달한다(E 비용 게이트). 응답 스키마는 기존 계약 유지.
    """
    try:
        sensor_data = request.sensors.model_dump()
        if request.thermal_max_temp is not None:
            sensor_data["thermal_max_temp"] = request.thermal_max_temp
        result = await run_diagnosis(
            equipment_id=request.equipment_id,
            prediction_result={"label": request.prediction, "confidence": request.confidence},
            sensor_data=sensor_data,
        )
        structured = result.get("structured", {})
        return DiagnosisResponse(
            equipment_id=request.equipment_id,
            severity=request.prediction,
            anomalies=structured.get("anomalies", []),
            probable_cause=structured.get("probable_cause", ""),
            recommended_action=structured.get("recommended_action", ""),
            similar_cases=result.get("similar_cases", []),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ========== 메트릭 ==========
@router.get("/metrics")
async def get_metrics():
    try:
        from app.monitoring.langsmith_config import logger
        return logger.metrics()
    except Exception as e:
        return {"error": str(e)}


# ========== 기존 테스트 ==========
@router.get("/test")
async def test_diagnose():
    test_request = DiagnosisRequest(
        equipment_id="OHT-007",
        prediction="심각",
        confidence=0.942,
        sensors=SensorData(ntc=45.2, pm1_0=120, pm2_5=180, pm10=250, ct1=142, ct2=85, ct3=78, ct4=92),
        thermal_max_temp=78.5
    )
    return await diagnose(test_request)
