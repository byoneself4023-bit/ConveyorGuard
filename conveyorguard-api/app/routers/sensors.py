"""센서 데이터 API"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Optional
from app.db import get_supabase

router = APIRouter(prefix="/sensors", tags=["Sensors"])


class SensorDataCreate(BaseModel):
    equipment_id: str
    ntc: float
    pm1_0: float
    pm2_5: float
    pm10: float
    ct1: float
    ct2: float
    ct3: float
    ct4: float


@router.get("/{equipment_id}/latest")
async def get_latest_sensor(equipment_id: str):
    """장비의 최신 센서값"""
    sb = get_supabase()
    result = (
        sb.table("sensor_data")
        .select("*")
        .eq("equipment_id", equipment_id)
        .order("recorded_at", desc=True)
        .limit(1)
        .execute()
    )
    if not result.data:
        raise HTTPException(status_code=404, detail="No sensor data found")
    return {"success": True, "data": result.data[0]}


@router.get("/{equipment_id}/history")
async def get_sensor_history(
    equipment_id: str,
    limit: int = Query(default=60, le=500),
):
    """센서 이력 (트렌드 차트용)"""
    sb = get_supabase()
    result = (
        sb.table("sensor_data")
        .select("ntc, pm1_0, pm2_5, pm10, ct1, ct2, ct3, ct4, recorded_at")
        .eq("equipment_id", equipment_id)
        .order("recorded_at", desc=True)
        .limit(limit)
        .execute()
    )
    # 시간순 정렬 (오래된것 먼저)
    data = list(reversed(result.data))
    return {"success": True, "data": data, "count": len(data)}


@router.post("", status_code=201)
async def create_sensor_data(body: SensorDataCreate):
    """센서 데이터 저장"""
    sb = get_supabase()
    result = sb.table("sensor_data").insert(body.model_dump()).execute()
    return {"success": True, "data": result.data[0]}
