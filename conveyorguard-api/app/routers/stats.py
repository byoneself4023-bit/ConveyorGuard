"""대시보드 통계 API"""

from fastapi import APIRouter
from app.db import get_supabase

router = APIRouter(prefix="/stats", tags=["Stats"])


@router.get("/overview")
async def get_overview():
    """장비 상태 요약"""
    sb = get_supabase()
    result = sb.table("equipment").select("state").execute()
    states = [e["state"] for e in result.data]
    return {
        "success": True,
        "data": {
            "total": len(states),
            "normal": states.count(0),
            "minor": states.count(1),
            "moderate": states.count(2),
            "severe": states.count(3),
        },
    }


@router.get("/sensors/avg")
async def get_sensor_averages():
    """전체 장비 최신 센서 평균값"""
    sb = get_supabase()
    equipment = sb.table("equipment").select("id").execute()

    totals = {"ntc": 0, "pm2_5": 0, "ct1": 0, "count": 0}
    for eq in equipment.data:
        sensor = (
            sb.table("sensor_data")
            .select("ntc, pm2_5, ct1")
            .eq("equipment_id", eq["id"])
            .order("recorded_at", desc=True)
            .limit(1)
            .execute()
        )
        if sensor.data:
            s = sensor.data[0]
            totals["ntc"] += s["ntc"] or 0
            totals["pm2_5"] += s["pm2_5"] or 0
            totals["ct1"] += s["ct1"] or 0
            totals["count"] += 1

    n = totals["count"] or 1
    return {
        "success": True,
        "data": {
            "avg_temperature": round(totals["ntc"] / n, 1),
            "avg_pm25": round(totals["pm2_5"] / n, 1),
            "avg_current": round(totals["ct1"] / n, 1),
        },
    }
