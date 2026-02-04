from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from app.core.gemini import generate_diagnosis
from app.core.rag import find_similar_cases

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


# ========== 기존 엔드포인트 (유지) ==========
@router.post("/diagnose", response_model=DiagnosisResponse)
async def diagnose(request: DiagnosisRequest):
    try:
        diagnosis = await generate_diagnosis(
            equipment_id=request.equipment_id,
            prediction=request.prediction,
            confidence=request.confidence,
            sensors=request.sensors.model_dump(),
            thermal_max_temp=request.thermal_max_temp
        )
        similar_cases = find_similar_cases(
            sensors=request.sensors.model_dump(),
            prediction=request.prediction
        )
        return DiagnosisResponse(
            equipment_id=request.equipment_id,
            severity=request.prediction,
            anomalies=diagnosis["anomalies"],
            probable_cause=diagnosis["probable_cause"],
            recommended_action=diagnosis["recommended_action"],
            similar_cases=similar_cases
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ========== 새 엔드포인트 (LangGraph) ==========
@router.post("/diagnose/graph")
async def diagnose_graph(request: DiagnosisRequest):
    try:
        from app.agents.diagnosis_graph import run_diagnosis
        
        result = await run_diagnosis(
            equipment_id=request.equipment_id,
            prediction_result={"label": request.prediction, "confidence": request.confidence},
            sensor_data=request.sensors.model_dump()
        )
        return {
            "equipment_id": request.equipment_id,
            "final_report": result.get("final_report", ""),
            "status": result.get("status", "unknown")
        }
    except ImportError:
        raise HTTPException(status_code=501, detail="LangGraph not installed. pip install langgraph")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ========== 새 엔드포인트 (고급 RAG) ==========
@router.post("/diagnose/rag")
async def diagnose_rag(request: DiagnosisRequest):
    try:
        from app.rag.case_retriever import retriever
        
        query = f"NTC={request.sensors.ntc} CT={request.sensors.ct1} {request.prediction}"
        cases = retriever.search(query, k=3)
        
        return {
            "equipment_id": request.equipment_id,
            "prediction": request.prediction,
            "similar_cases": cases
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ========== 새 엔드포인트 (메트릭) ==========
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
