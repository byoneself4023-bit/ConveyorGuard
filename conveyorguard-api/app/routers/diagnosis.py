"""진단 이력 API"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Optional, List
from app.db import get_supabase

router = APIRouter(prefix="/diagnosis", tags=["Diagnosis"])


class DiagnosisCreate(BaseModel):
    equipment_id: str
    severity: str
    prediction: Optional[int] = None
    confidence: Optional[float] = None
    probable_cause: Optional[str] = None
    recommended_action: Optional[str] = None
    anomalies: Optional[List[str]] = None


@router.get("/{equipment_id}")
async def get_diagnosis_history(
    equipment_id: str,
    limit: int = Query(default=20, le=100),
):
    """장비별 진단 이력"""
    sb = get_supabase()
    result = (
        sb.table("diagnosis_history")
        .select("*")
        .eq("equipment_id", equipment_id)
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
    )
    return {"success": True, "data": result.data}


@router.get("/recent/all")
async def get_recent_diagnosis(limit: int = Query(default=10, le=50)):
    """최근 전체 진단 이력"""
    sb = get_supabase()
    result = (
        sb.table("diagnosis_history")
        .select("*")
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
    )
    return {"success": True, "data": result.data}


@router.post("", status_code=201)
async def create_diagnosis(body: DiagnosisCreate):
    """진단 결과 저장"""
    sb = get_supabase()
    result = sb.table("diagnosis_history").insert(body.model_dump()).execute()
    return {"success": True, "data": result.data[0]}
