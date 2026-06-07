"""
진단 JSON 파싱 유틸.

단일 Gemini 호출 진단 경로(generate_diagnosis)는 4-에이전트 그래프로 대체됐다.
구조화 출력 파싱 로직만 공유 유틸로 남겨 그래프 finalize에서 재사용한다.
"""
import json
import logging
import re

logger = logging.getLogger(__name__)

# 진단 구조화 필드 (conveyorguard-api persist 계약)
_FALLBACK_KEYS = ("anomalies", "probable_cause", "recommended_action")

# 서술 끝/중간에 박힌 ```json ... ``` (또는 펜스 없는) JSON 객체 추출용
_JSON_BLOCK = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_BARE_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


def strip_json_block(text: str) -> str:
    """서술에서 구조화 JSON 코드블록을 제거해 사람이 읽을 서술 본문만 남긴다."""
    return _JSON_BLOCK.sub("", text or "").strip()


def _extract_json_candidate(text: str) -> str:
    """서술 어디에 있든 JSON 객체 후보를 뽑아낸다(펜스 블록 우선, 없으면 bare 객체)."""
    fenced = _JSON_BLOCK.search(text)
    if fenced:
        return fenced.group(1)
    if text.startswith("```"):  # 전체가 코드블록인 경우(기존 동작)
        return "\n".join(text.split("\n")[1:-1])
    bare = _BARE_OBJECT.search(text)
    if bare:
        return bare.group(0)
    return text


def parse_diagnosis_json(text: str) -> dict:
    """LLM 응답 텍스트에서 진단 JSON을 추출. 실패 시 로깅 후 폴백 구조 반환.

    diagnostician이 "서술 + ```json 블록"을 함께 내므로, 서술 끝에 박힌 블록도 뽑는다.
    """
    text = (text or "").strip()
    candidate = _extract_json_candidate(text)

    try:
        result = json.loads(candidate)
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
