"""
RAG 기반 유사 사례 검색 (FAISS)
"""
from typing import List, Dict
from pathlib import Path

try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain.schema import Document
    FAISS_OK = True
except ImportError:
    FAISS_OK = False


CASES = [
    {"id": "CASE-001", "type": "OHT", "symptom": "모터 과열 NTC 75°C+", "cause": "베어링 마모", "solution": "베어링 교체"},
    {"id": "CASE-002", "type": "OHT", "symptom": "전류 급증 CT 6A+", "cause": "벨트 장력 과다", "solution": "장력 조정"},
    {"id": "CASE-003", "type": "AGV", "symptom": "PM2.5 급증", "cause": "필터 막힘", "solution": "필터 교체"},
    {"id": "CASE-004", "type": "OHT", "symptom": "열화상 핫스팟 80°C+", "cause": "접촉 불량", "solution": "커넥터 점검"},
    {"id": "CASE-005", "type": "OHT", "symptom": "온도+전류 동시 상승", "cause": "기어박스 오일 부족", "solution": "오일 보충"},
]


class CaseRetriever:
    def __init__(self):
        self.vectorstore = None
        if FAISS_OK:
            self._init()
    
    def _init(self):
        self.embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        docs = [Document(page_content=f"{c['symptom']} {c['cause']}", metadata=c) for c in CASES]
        self.vectorstore = FAISS.from_documents(docs, self.embeddings)
    
    def search(self, query: str, k: int = 3) -> List[Dict]:
        if not self.vectorstore:
            return [{"case_id": c["id"], "cause": c["cause"], "solution": c["solution"], "similarity": 0.8} for c in CASES[:k]]
        
        results = self.vectorstore.similarity_search_with_score(query, k=k)
        return [{"case_id": d.metadata["id"], "cause": d.metadata["cause"], "solution": d.metadata["solution"], "similarity": round(1-s, 2)} for d, s in results]


retriever = CaseRetriever()
