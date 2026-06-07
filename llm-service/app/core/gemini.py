"""
진단 JSON 파싱 유틸.

단일 Gemini 호출 진단 경로(generate_diagnosis)는 4-에이전트 그래프로 대체됐다.
구조화 출력 파싱 로직만 공유 유틸로 남겨 그래프 finalize에서 재사용한다.
"""
import json
import logging

logger = logging.getLogger(__name__)

# 진단 구조화 필드 (conveyorguard-api persist 계약)
_FALLBACK_KEYS = ("anomalies", "probable_cause", "recommended_action")


def parse_diagnosis_json(text: str) -> dict:
    """LLM 응답 텍스트에서 진단 JSON을 추출. 실패 시 로깅 후 폴백 구조 반환."""
    text = (text or "").strip()

    # ```json ... ``` 코드블록 처리
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1])

    try:
        result = json.loads(text)
        # 필수 키 보강
        return {
            "anomalies": result.get("anomalies", []),
            "probable_cause": result.get("probable_cause", ""),
            "recommended_action": result.get("recommended_action", ""),
        }
    except (json.JSONDecodeError, AttributeError) as e:
        logger.warning("진단 JSON 파싱 실패, 폴백 구조 반환: %s", text[:200])
        return {
            "anomalies": [text[:200]] if text else [],
            "probable_cause": "구조화 파싱 실패 — 원문 참조",
            "recommended_action": "수동 점검 필요",
        }
