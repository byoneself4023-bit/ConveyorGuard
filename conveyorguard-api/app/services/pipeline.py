"""자동 AI 파이프라인 — 센서 → ML 예측 → LLM 진단 → 알림"""

import os
import logging
import httpx
import torch
from app.db import get_supabase
from app.core.loader import model_loader

logger = logging.getLogger(__name__)

STATE_LABELS = {0: "정상", 1: "경미", 2: "중간", 3: "심각"}
LLM_API_BASE = os.getenv("LLM_API_URL", "http://localhost:8001")


async def run_pipeline(equipment_id: str, latest_sensors: dict) -> dict | None:
    """
    센서 데이터가 들어올 때마다 호출.
    1. 최근 30개 센서 수집 → ML predict
    2. equipment.state 업데이트
    3. state >= 2 → LLM 진단 → diagnosis_history + alerts

    Returns: pipeline result dict or None if skipped
    """
    sb = get_supabase()

    # 1. 최근 30개 센서 데이터 수집
    history = (
        sb.table("sensor_data")
        .select("ntc, pm1_0, pm2_5, pm10, ct1, ct2, ct3, ct4")
        .eq("equipment_id", equipment_id)
        .order("recorded_at", desc=True)
        .limit(30)
        .execute()
    )

    if len(history.data) < 5:
        return None  # 데이터 부족

    # 30개에 못 미치면 마지막 값으로 패딩
    rows = list(reversed(history.data))
    while len(rows) < 30:
        rows.insert(0, rows[0])

    # 2. ML 예측
    prediction = None
    confidence = 0.0
    probabilities = [0.25, 0.25, 0.25, 0.25]

    if model_loader.is_loaded():
        try:
            sensor_seq = [[r["ntc"], r["pm1_0"], r["pm2_5"], r["pm10"],
                           r["ct1"], r["ct2"], r["ct3"], r["ct4"]] for r in rows]

            # 열화상/외부 데이터는 합성 (센서 기반)
            thermal = [[[r["ntc"] * 0.8] * 80 for _ in range(60)] for r in rows]
            external = [[r["ntc"], 50.0, 500.0] for r in rows]

            device = model_loader.device
            s_t = torch.tensor([sensor_seq], dtype=torch.float32).to(device)
            i_t = torch.tensor([thermal], dtype=torch.float32).to(device)
            e_t = torch.tensor([external], dtype=torch.float32).to(device)

            result = model_loader.model.predict(s_t, i_t, e_t)
            prediction = result["prediction"].item()
            confidence = result["confidence"].item()
            probabilities = result["probabilities"].squeeze().tolist()
        except Exception as e:
            logger.warning(f"ML predict failed for {equipment_id}: {e}")
            # 모델 실패 시 센서 기반 간이 판정
            prediction = _simple_predict(latest_sensors)
            confidence = 0.7
    else:
        # 모델 미로드 시 간이 판정
        prediction = _simple_predict(latest_sensors)
        confidence = 0.7

    label = STATE_LABELS.get(prediction, "정상")

    # 3. equipment.state 업데이트
    try:
        sb.table("equipment").update({
            "state": prediction,
            "confidence": round(confidence, 3),
        }).eq("id", equipment_id).execute()
    except Exception as e:
        logger.warning(f"State update failed: {e}")

    result = {
        "equipment_id": equipment_id,
        "prediction": prediction,
        "label": label,
        "confidence": confidence,
        "probabilities": probabilities,
    }

    # 4. state >= 2 → LLM 진단
    if prediction >= 2:
        diagnosis = await _run_llm_diagnosis(equipment_id, label, confidence, latest_sensors)
        if diagnosis:
            result["diagnosis"] = diagnosis

            # diagnosis_history에 저장
            try:
                sb.table("diagnosis_history").insert({
                    "equipment_id": equipment_id,
                    "severity": label,
                    "prediction": prediction,
                    "confidence": round(confidence, 3),
                    "probable_cause": diagnosis.get("probable_cause", ""),
                    "recommended_action": diagnosis.get("recommended_action", ""),
                    "anomalies": diagnosis.get("anomalies", []),
                }).execute()
            except Exception as e:
                logger.warning(f"Diagnosis save failed: {e}")

            # 알림 생성
            alert_level = "moderate" if prediction == 2 else "severe"
            alert_msg = f"{label} 감지: {diagnosis.get('probable_cause', '이상 감지')}"
            try:
                sb.table("alerts").insert({
                    "equipment_id": equipment_id,
                    "message": alert_msg[:200],
                    "level": alert_level,
                }).execute()
                result["alert"] = {"message": alert_msg, "level": alert_level}
            except Exception as e:
                logger.warning(f"Alert create failed: {e}")

    return result


def _simple_predict(sensors: dict) -> int:
    """센서값 기반 간이 상태 판정 (ML 모델 없을 때)"""
    ntc = sensors.get("ntc", 0)
    pm2_5 = sensors.get("pm2_5", 0)
    ct1 = sensors.get("ct1", 0)

    score = 0
    if ntc > 65:
        score += 2
    elif ntc > 45:
        score += 1
    if pm2_5 > 70:
        score += 2
    elif pm2_5 > 35:
        score += 1
    if ct1 > 5:
        score += 2
    elif ct1 > 3.5:
        score += 1

    if score >= 5:
        return 3
    elif score >= 3:
        return 2
    elif score >= 1:
        return 1
    return 0


async def _run_llm_diagnosis(equipment_id: str, label: str, confidence: float, sensors: dict) -> dict | None:
    """LLM 진단 호출 (실패 시 None)"""
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            res = await client.post(f"{LLM_API_BASE}/api/v1/diagnose", json={
                "equipment_id": equipment_id,
                "prediction": label,
                "confidence": confidence,
                "sensors": sensors,
            })
            if res.status_code == 200:
                return res.json()
    except Exception as e:
        logger.warning(f"LLM diagnosis failed for {equipment_id}: {e}")

    # LLM 실패 시 기본 진단
    return {
        "severity": label,
        "anomalies": [f"센서 이상 감지 ({label})"],
        "probable_cause": f"자동 감지된 {label} 상태 — 센서 데이터 기반 판정",
        "recommended_action": "점검 필요" if label == "중간" else "즉시 점검 필요",
    }
