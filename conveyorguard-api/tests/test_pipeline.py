"""run_pipeline 동작 회귀 테스트.

②SRP 분리·④원자성 수정이 동작을 깨지 않는지 지키는 안전망.
외부 의존(Supabase, ML 모델, LLM)은 모두 모킹한다.
"""

import pytest

from app.services import pipeline


# --- Supabase 페이크 (fluent 빌더 + 호출 기록) ---

class _Resp:
    def __init__(self, data):
        self.data = data


class _Query:
    def __init__(self, sb, table):
        self.sb = sb
        self.table = table
        self.op = "select"

    # 읽기 체인
    def select(self, *a, **k):
        self.op = "select"
        return self

    def eq(self, *a, **k):
        return self

    def order(self, *a, **k):
        return self

    def limit(self, *a, **k):
        return self

    # 쓰기 체인
    def update(self, payload):
        self.op = "update"
        return self

    def insert(self, payload):
        self.op = "insert"
        return self

    def delete(self):
        self.op = "delete"
        return self

    def execute(self):
        self.sb.calls.append((self.table, self.op))
        err = self.sb.errors.get((self.table, self.op))
        if err is not None:
            raise err
        if self.op == "select":
            return _Resp(self.sb.select_returns.get(self.table, []))
        if self.op == "insert":
            return _Resp(self.sb.insert_returns.get(self.table, [{"id": f"{self.table}-id"}]))
        return _Resp([])


class FakeSupabase:
    def __init__(self):
        self.calls = []
        self.select_returns = {}
        self.insert_returns = {}
        self.errors = {}  # (table, op) -> Exception

    def table(self, name):
        return _Query(self, name)

    def tables_inserted(self):
        return [t for (t, op) in self.calls if op == "insert"]


def _sensor_rows(n=6, ntc=20.0):
    """수집 가드(>=5)를 통과할 더미 센서 이력."""
    return [
        {"ntc": ntc, "pm1_0": 1.0, "pm2_5": 1.0, "pm10": 1.0,
         "ct1": 1.0, "ct2": 1.0, "ct3": 1.0, "ct4": 1.0}
        for _ in range(n)
    ]


LOW_SENSORS = {"ntc": 20.0, "pm2_5": 5.0, "ct1": 1.0}    # _simple_predict -> 0 (게이트 미만)
HIGH_SENSORS = {"ntc": 80.0, "pm2_5": 90.0, "ct1": 7.0}  # _simple_predict -> 3 (게이트 이상)


@pytest.fixture
def fake_sb(monkeypatch):
    sb = FakeSupabase()
    sb.select_returns["sensor_data"] = _sensor_rows()
    monkeypatch.setattr(pipeline, "get_supabase", lambda: sb)
    # ML 모델 미로드 경로(_simple_predict) 강제 — torch 모델 회피
    monkeypatch.setattr(pipeline.model_loader, "is_loaded", lambda: False)
    return sb


async def _fake_diag(*a, **k):
    return {
        "severity": "심각",
        "anomalies": ["test"],
        "probable_cause": "테스트 원인",
        "recommended_action": "점검",
    }


async def test_below_gate_no_diagnosis_or_alert(fake_sb, monkeypatch):
    """게이트 미만이면 진단/알림을 저장하지 않는다."""
    monkeypatch.setattr(pipeline, "_run_llm_diagnosis", _fake_diag)
    result = await pipeline.run_pipeline("eq-1", LOW_SENSORS)

    assert result["prediction"] < 2
    inserted = fake_sb.tables_inserted()
    assert "diagnosis_history" not in inserted
    assert "alerts" not in inserted


async def test_at_gate_persists_diagnosis_and_alert(fake_sb, monkeypatch):
    """게이트 이상이면 진단과 알림을 모두 저장한다."""
    monkeypatch.setattr(pipeline, "_run_llm_diagnosis", _fake_diag)
    result = await pipeline.run_pipeline("eq-1", HIGH_SENSORS)

    assert result["prediction"] >= 2
    inserted = fake_sb.tables_inserted()
    assert "diagnosis_history" in inserted
    assert "alerts" in inserted
