"""
사례 검색 어댑터.

기존 코사인 유사도 RAG를 제거하고 단일 FAISS 검색기(app.rag.case_retriever)로 흡수한다.
`find_similar_cases`는 /diagnose의 기존 호출 계약(sensors+prediction)을 유지하기 위한
얇은 어댑터로, 센서·예측을 텍스트 질의로 만들어 단일 검색기에 위임한다(①SSoT).
"""
from typing import Dict, List

from app.rag.case_retriever import retriever


def _build_query(sensors: Dict, prediction: str) -> str:
    """센서 + 예측 → 의미 검색용 텍스트 질의."""
    return (
        f"{prediction} "
        f"NTC={sensors.get('ntc', '')} "
        f"PM2.5={sensors.get('pm2_5', '')} "
        f"CT1={sensors.get('ct1', '')}"
    )


def find_similar_cases(sensors: Dict, prediction: str, top_k: int = 3) -> List[Dict]:
    """유사 과거 사례 검색 (단일 FAISS 검색기에 위임)."""
    return retriever.search(_build_query(sensors, prediction), k=top_k)
