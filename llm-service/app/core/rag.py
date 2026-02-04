import json
import numpy as np
from pathlib import Path
from typing import Dict, List

# 과거 사례 로드
CASES_PATH = Path(__file__).parent.parent / "data" / "cases.json"

def load_cases() -> List[Dict]:
    """과거 고장 사례 로드"""
    try:
        with open(CASES_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        return []

def calculate_similarity(sensors1: Dict, sensors2: Dict) -> float:
    """센서 데이터 유사도 계산 (코사인 유사도)"""
    keys = ['ntc', 'pm1_0', 'pm2_5', 'pm10', 'ct1', 'ct2', 'ct3', 'ct4']
    
    vec1 = np.array([sensors1.get(k, 0) for k in keys])
    vec2 = np.array([sensors2.get(k, 0) for k in keys])
    
    # 정규화
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    similarity = np.dot(vec1, vec2) / (norm1 * norm2)
    return float(similarity)

def find_similar_cases(
    sensors: Dict,
    prediction: str,
    top_k: int = 3
) -> List[Dict]:
    """유사한 과거 사례 검색"""
    cases = load_cases()
    
    if not cases:
        return []
    
    # 유사도 계산
    scored_cases = []
    for case in cases:
        similarity = calculate_similarity(sensors, case.get("sensors", {}))
        
        # 같은 severity면 가산점
        if case.get("severity") == prediction:
            similarity = min(similarity * 1.2, 1.0)
        
        scored_cases.append({
            "date": case.get("date", ""),
            "equipment_id": case.get("equipment_id", ""),
            "issue": case.get("issue", ""),
            "action": case.get("action", ""),
            "similarity": round(similarity * 100, 1)
        })
    
    # 유사도 순 정렬
    scored_cases.sort(key=lambda x: x["similarity"], reverse=True)
    
    return scored_cases[:top_k]
