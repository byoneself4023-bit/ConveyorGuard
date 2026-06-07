"""단일 /diagnose 엔드포인트 계약 테스트.

run_diagnosis(그래프)를 모킹 — Gemini 키 없이 응답 매핑·계약만 검증.
계약: 응답이 conveyorguard-api persist가 읽는 구조화 키를 모두 포함하는지.
"""

from fastapi.testclient import TestClient

from app.main import app
from app.api import router as router_mod

client = TestClient(app)

# conveyorguard-api _persist_diagnosis_and_alert가 소비하는 키
CONSUMER_KEYS = {"anomalies", "probable_cause", "recommended_action"}

REQUEST = {
    "equipment_id": "OHT-007",
    "prediction": "심각",
    "confidence": 0.94,
    "sensors": {"ntc": 78.5, "pm1_0": 120, "pm2_5": 180, "pm10": 250,
                "ct1": 142, "ct2": 85, "ct3": 78, "ct4": 92},
}


async def _fake_run_diagnosis(equipment_id, prediction_result, sensor_data):
    return {
        "similar_cases": [{
            "case_id": "CASE-2025-001", "equipment_id": "OHT-001", "date": "2025-03-15",
            "severity": "심각", "issue": "벨트 과열", "action": "베어링 교체", "similarity": 0.9,
        }],
        "structured": {
            "anomalies": ["NTC 78.5°C 초과", "CT1 142A 급등"],
            "probable_cause": "베어링 소착",
            "recommended_action": "베어링 교체 및 냉각 점검",
        },
        "final_report": "# 진단 리포트 ...",
        "status": "approved",
    }


def test_diagnose_returns_structured_contract(monkeypatch):
    monkeypatch.setattr(router_mod, "run_diagnosis", _fake_run_diagnosis)

    res = client.post("/api/v1/diagnose", json=REQUEST)
    assert res.status_code == 200
    body = res.json()

    # 계약: 소비자가 읽는 구조화 키 모두 존재
    assert CONSUMER_KEYS <= set(body.keys())
    assert body["probable_cause"] == "베어링 소착"
    assert body["anomalies"] == ["NTC 78.5°C 초과", "CT1 142A 급등"]
    # 그래프의 통합 RAG 사례가 응답에 실린다
    assert body["similar_cases"][0]["issue"] == "벨트 과열"
    assert body["severity"] == "심각"


def test_removed_endpoints_are_gone():
    """구 엔드포인트(/diagnose/graph, /diagnose/rag)는 제거됐다."""
    assert client.post("/api/v1/diagnose/graph", json=REQUEST).status_code == 404
    assert client.post("/api/v1/diagnose/rag", json=REQUEST).status_code == 404
