"""장비 관리 API"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from app.db import get_supabase

router = APIRouter(prefix="/equipment", tags=["Equipment"])


class EquipmentCreate(BaseModel):
    id: str
    name: str
    type: str  # OHT or AGV
    location: Optional[str] = None


class EquipmentUpdate(BaseModel):
    name: Optional[str] = None
    location: Optional[str] = None
    state: Optional[int] = None
    confidence: Optional[float] = None


@router.get("")
async def list_equipment():
    """전체 장비 목록 + 최신 센서값"""
    sb = get_supabase()
    result = sb.table("equipment").select("*").order("id").execute()
    equipment_list = result.data

    # 각 장비의 최신 센서값 조회
    for eq in equipment_list:
        sensor = (
            sb.table("sensor_data")
            .select("ntc, pm1_0, pm2_5, pm10, ct1, ct2, ct3, ct4, recorded_at")
            .eq("equipment_id", eq["id"])
            .order("recorded_at", desc=True)
            .limit(1)
            .execute()
        )
        eq["latest_sensor"] = sensor.data[0] if sensor.data else None

    return {"success": True, "data": equipment_list}


@router.get("/{equipment_id}")
async def get_equipment(equipment_id: str):
    """장비 상세"""
    sb = get_supabase()
    result = sb.table("equipment").select("*").eq("id", equipment_id).execute()
    if not result.data:
        raise HTTPException(status_code=404, detail="Equipment not found")

    eq = result.data[0]

    # 최신 센서값
    sensor = (
        sb.table("sensor_data")
        .select("*")
        .eq("equipment_id", equipment_id)
        .order("recorded_at", desc=True)
        .limit(1)
        .execute()
    )
    eq["latest_sensor"] = sensor.data[0] if sensor.data else None

    return {"success": True, "data": eq}


@router.post("", status_code=201)
async def create_equipment(body: EquipmentCreate):
    """장비 등록"""
    sb = get_supabase()
    result = sb.table("equipment").insert(body.model_dump()).execute()
    return {"success": True, "data": result.data[0]}


@router.put("/{equipment_id}")
async def update_equipment(equipment_id: str, body: EquipmentUpdate):
    """장비 정보 수정"""
    sb = get_supabase()
    update_data = {k: v for k, v in body.model_dump().items() if v is not None}
    if not update_data:
        raise HTTPException(status_code=400, detail="No fields to update")
    result = sb.table("equipment").update(update_data).eq("id", equipment_id).execute()
    if not result.data:
        raise HTTPException(status_code=404, detail="Equipment not found")
    return {"success": True, "data": result.data[0]}


@router.delete("/{equipment_id}")
async def delete_equipment(equipment_id: str):
    """장비 삭제"""
    sb = get_supabase()
    sb.table("equipment").delete().eq("id", equipment_id).execute()
    return {"success": True, "message": "Deleted"}
