"""4-에이전트 그래프의 통합 RAG 참조 테스트.

LLM(get_llm)·retriever를 모킹 — Gemini 키·임베딩 없이 그래프 배선만 검증.
"""

from types import SimpleNamespace

from app.agents import diagnosis_graph
from app.agents.diagnosis_graph import run_diagnosis

MOCK_CASES = [
    {"case_id": "CASE-2025-007", "issue": "다축 전류 동시 상승",
     "action": "롤러 점검", "similarity": 0.91},
]


class _FakeLLM:
    def __init__(self, prompts):
        self._prompts = prompts

    def invoke(self, prompt):
        self._prompts.append(prompt)
        # reviewer가 1회 만에 종료되도록 APPROVE 포함
        return SimpleNamespace(content="검토 완료. APPROVE")


async def test_graph_injects_rag_cases(monkeypatch):
    prompts = []
    monkeypatch.setattr(diagnosis_graph, "get_llm", lambda: _FakeLLM(prompts))
    monkeypatch.setattr(diagnosis_graph.retriever, "search", lambda q, k=3: MOCK_CASES)

    result = await run_diagnosis(
        equipment_id="OHT-004",
        prediction_result={"label": "심각", "confidence": 0.94},
        sensor_data={"ntc": 72.8, "ct1": 138, "pm2_5": 280},
    )

    # 1) retrieval 노드가 사례를 state에 실었다
    assert result["similar_cases"] == MOCK_CASES
    # 2) diagnostician 프롬프트에 사례 근거가 주입됐다
    assert any("CASE-2025-007" in p for p in prompts)
    # 3) 최종 리포트·상태 산출
    assert result["status"] == "approved"
    assert "진단 리포트" in result["final_report"]
    # 4) finalize는 LLM을 부르지 않는다 — analyzer+diagnostician+reviewer 3콜뿐
    assert len(prompts) == 3


async def test_graph_handles_no_cases(monkeypatch):
    """사례 0건이어도 그래프가 정상 종료(빈 컨텍스트)."""
    prompts = []
    monkeypatch.setattr(diagnosis_graph, "get_llm", lambda: _FakeLLM(prompts))
    monkeypatch.setattr(diagnosis_graph.retriever, "search", lambda q, k=3: [])

    result = await run_diagnosis(
        equipment_id="AGV-001",
        prediction_result={"label": "중간", "confidence": 0.7},
        sensor_data={"ntc": 55, "ct1": 85, "pm2_5": 120},
    )

    assert result["similar_cases"] == []
    assert any("유사 과거 사례 없음" in p for p in prompts)
    assert result["status"] == "approved"


class _RevisingLLM:
    """reviewer가 끝까지 REVISE만 내는 LLM — review_count 한도 강제 종료 경로 검증."""

    def __init__(self, prompts):
        self._prompts = prompts

    def invoke(self, prompt):
        self._prompts.append(prompt)
        return SimpleNamespace(content="보완 필요. REVISE")


async def test_graph_force_finalize_marks_status(monkeypatch):
    """reviewer 미승인인데 review_count>=2면 강제 종료 → status=force_finalized(§F)."""
    prompts = []
    monkeypatch.setattr(diagnosis_graph, "get_llm", lambda: _RevisingLLM(prompts))
    monkeypatch.setattr(diagnosis_graph.retriever, "search", lambda q, k=3: MOCK_CASES)

    result = await run_diagnosis(
        equipment_id="OHT-009",
        prediction_result={"label": "심각", "confidence": 0.9},
        sensor_data={"ntc": 80, "ct1": 150, "pm2_5": 300},
    )

    # 검토 미통과가 approved로 가려지지 않는다
    assert result["status"] == "force_finalized"
    # analyzer1 + diagnostician2 + reviewer2 = 5콜 (finalize는 LLM 없음)
    assert len(prompts) == 5
