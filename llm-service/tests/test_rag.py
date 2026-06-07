"""RAG 단일화(FAISS) 단위 테스트.

임베딩/벡터스토어는 모킹 — 실제 모델 다운로드·네트워크 없이 검색기 로직만 검증.
"""

from types import SimpleNamespace

from app.rag import case_retriever
from app.rag.case_retriever import CaseRetriever, load_cases
from app.core.rag import find_similar_cases

UNIFIED_KEYS = {"case_id", "equipment_id", "date", "severity", "issue", "action", "similarity"}

SAMPLE_CASES = [
    {"case_id": "CASE-1", "equipment_id": "OHT-1", "date": "2025-01-01",
     "severity": "심각", "issue": "과열", "action": "교체"},
    {"case_id": "CASE-2", "equipment_id": "OHT-2", "date": "2025-02-01",
     "severity": "중간", "issue": "필터", "action": "청소"},
    {"case_id": "CASE-3", "equipment_id": "AGV-1", "date": "2025-03-01",
     "severity": "경미", "issue": "진동", "action": "점검"},
]


def test_load_cases_single_source():
    """사례 저장소 단일 출처(cases.json) 로드."""
    cases = load_cases()
    assert len(cases) == 10
    assert cases[0]["case_id"].startswith("CASE-")


def test_search_fallback_logs_and_shapes(caplog):
    """FAISS 미사용 폴백: 로깅 + 통일된 모양 top-k."""
    r = CaseRetriever(cases=SAMPLE_CASES)
    r._index_attempted = True          # 인덱스 구축 건너뜀
    r.vectorstore = None               # 폴백 경로 강제

    import logging
    with caplog.at_level(logging.INFO):
        results = r.search("과열 NTC", k=2)

    assert len(results) == 2
    assert set(results[0].keys()) == UNIFIED_KEYS
    assert any("폴백" in rec.message for rec in caplog.records)  # 실패가 조용히 묻히지 않음


def test_search_with_mocked_vectorstore():
    """FAISS 경로: 모킹된 벡터스토어 결과를 통일 모양으로 변환."""
    r = CaseRetriever(cases=SAMPLE_CASES)
    r._index_attempted = True

    class _FakeVS:
        def similarity_search_with_score(self, query, k):
            # (Document-like, distance) — 작을수록 유사
            return [
                (SimpleNamespace(metadata={"case_id": "CASE-2"}), 0.1),
                (SimpleNamespace(metadata={"case_id": "CASE-1"}), 0.4),
            ][:k]

    r.vectorstore = _FakeVS()
    results = r.search("필터 막힘", k=2)

    assert [c["case_id"] for c in results] == ["CASE-2", "CASE-1"]
    assert set(results[0].keys()) == UNIFIED_KEYS
    assert results[0]["similarity"] == 0.9  # 1 - 0.1


def test_find_similar_cases_adapter(monkeypatch):
    """core.rag 어댑터가 단일 검색기에 위임한다."""
    captured = {}

    def fake_search(query, k=3):
        captured["query"] = query
        captured["k"] = k
        return [{"case_id": "CASE-1"}]

    monkeypatch.setattr(case_retriever.retriever, "search", fake_search)
    out = find_similar_cases({"ntc": 78, "pm2_5": 120, "ct1": 5.5}, "심각", top_k=2)

    assert out == [{"case_id": "CASE-1"}]
    assert captured["k"] == 2
    assert "심각" in captured["query"] and "78" in captured["query"]
