"""알림 API"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from app.db import get_supabase

router = APIRouter(prefix="/alerts", tags=["Alerts"])


class AlertCreate(BaseModel):
    equipment_id: str
    message: str
    level: str  # minor, moderate, severe


@router.get("")
async def list_alerts(
    unread_only: bool = Query(default=False),
    limit: int = Query(default=20, le=100),
):
    """알림 목록"""
    sb = get_supabase()
    query = sb.table("alerts").select("*").order("created_at", desc=True).limit(limit)
    if unread_only:
        query = query.eq("is_read", False)
    result = query.execute()
    return {"success": True, "data": result.data}


@router.get("/count")
async def unread_count():
    """읽지 않은 알림 수"""
    sb = get_supabase()
    result = sb.table("alerts").select("id", count="exact").eq("is_read", False).execute()
    return {"success": True, "count": result.count}


@router.put("/{alert_id}/read")
async def mark_as_read(alert_id: int):
    """알림 읽음 처리"""
    sb = get_supabase()
    result = sb.table("alerts").update({"is_read": True}).eq("id", alert_id).execute()
    if not result.data:
        raise HTTPException(status_code=404, detail="Alert not found")
    return {"success": True}


@router.post("", status_code=201)
async def create_alert(body: AlertCreate):
    """알림 생성"""
    sb = get_supabase()
    result = sb.table("alerts").insert(body.model_dump()).execute()
    return {"success": True, "data": result.data[0]}
