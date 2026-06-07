"""
RAG 사례 검색 (FAISS 단일화).

사례 저장소는 app/data/cases.json 한 곳(SSoT). 그 위에 FAISS 인덱스를 구축한다.
임베딩/FAISS 미설치 또는 구축 실패 시 로깅 후 휴리스틱 폴백(말없는 fallback 금지).
인덱스는 첫 검색 때 lazy 하게 만든다(import 시 모델 다운로드 회피).
"""
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

CASES_PATH = Path(__file__).parent.parent / "data" / "cases.json"

try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain.schema import Document
    FAISS_OK = True
except ImportError:
    FAISS_OK = False


def load_cases() -> List[Dict]:
    """사례 저장소 로드 (단일 출처: cases.json)."""
    try:
        with open(CASES_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.warning("사례 로딩 실패 (%s): %s", CASES_PATH, e)
        return []


def _shape(case: Dict, similarity: float) -> Dict:
    """검색 결과의 통일된 모양."""
    return {
        "case_id": case.get("case_id", ""),
        "equipment_id": case.get("equipment_id", ""),
        "date": case.get("date", ""),
        "severity": case.get("severity", ""),
        "issue": case.get("issue", ""),
        "action": case.get("action", ""),
        "similarity": round(similarity, 3),
    }


class CaseRetriever:
    """cases.json 기반 단일 사례 검색기."""

    def __init__(self, cases: Optional[List[Dict]] = None):
        self.cases = cases if cases is not None else load_cases()
        self._by_id = {c.get("case_id"): c for c in self.cases}
        self.vectorstore = None
        self._index_attempted = False

    def _ensure_index(self) -> None:
        """FAISS 인덱스 lazy 구축 (1회). 실패는 로깅 후 폴백."""
        if self._index_attempted:
            return
        self._index_attempted = True
        if not (FAISS_OK and self.cases):
            return
        try:
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            docs = [
                Document(
                    page_content=f"{c.get('issue', '')} {c.get('action', '')}",
                    metadata={"case_id": c.get("case_id", "")},
                )
                for c in self.cases
            ]
            self.vectorstore = FAISS.from_documents(docs, embeddings)
        except Exception as e:
            logger.warning("FAISS 인덱스 구축 실패, 폴백 사용: %s", e)

    def search(self, query: str, k: int = 3) -> List[Dict]:
        """질의와 유사한 과거 사례 top-k. 통일된 모양으로 반환."""
        if not self.cases:
            return []

        self._ensure_index()
        if not self.vectorstore:
            # 폴백: 실패를 빈 결과로 숨기지 않고 로깅 후 앞에서 k건
            logger.info("FAISS 미사용 — 휴리스틱 폴백으로 상위 %d건 반환", k)
            return [_shape(c, 0.0) for c in self.cases[:k]]

        results = self.vectorstore.similarity_search_with_score(query, k=k)
        out = []
        for doc, score in results:
            case = self._by_id.get(doc.metadata.get("case_id"), {})
            # FAISS score는 거리(작을수록 유사) → 유사도로 변환
            out.append(_shape(case, max(0.0, 1.0 - float(score))))
        return out


retriever = CaseRetriever()
