import os
import json
import asyncio
import logging
from dotenv import load_dotenv

load_dotenv()

import google.generativeai as genai

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel("gemini-2.5-flash")

logger = logging.getLogger(__name__)


async def generate_diagnosis(
    equipment_id: str,
    prediction: str,
    confidence: float,
    sensors: dict,
    thermal_max_temp: float = None
) -> dict:
    """
    ML 예측 결과와 센서 데이터를 받아 구조화된 진단 결과 반환
    """
    thermal_info = f"\n- 열화상 최고온도: {thermal_max_temp}°C" if thermal_max_temp else ""

    prompt = f"""당신은 제조설비 예지보전 전문가입니다.

## 장비 정보
- 장비 ID: {equipment_id}

## 예측 결과
- 열화 상태: {prediction}
- 신뢰도: {confidence:.1%}

## 센서 데이터
- NTC (온도): {sensors.get('ntc', 'N/A')}°C
- PM1.0: {sensors.get('pm1_0', 'N/A')} μg/m³
- PM2.5: {sensors.get('pm2_5', 'N/A')} μg/m³
- PM10: {sensors.get('pm10', 'N/A')} μg/m³
- CT1: {sensors.get('ct1', 'N/A')}A
- CT2: {sensors.get('ct2', 'N/A')}A
- CT3: {sensors.get('ct3', 'N/A')}A
- CT4: {sensors.get('ct4', 'N/A')}A{thermal_info}

## 임계값 기준
- NTC: 50°C 이상 주의, 70°C 이상 위험
- PM2.5: 35 μg/m³ 이상 주의
- CT (전류): 5A 이상 주의

## 작업
아래 JSON 형식으로만 응답하세요. 다른 텍스트 없이 JSON만 출력하세요.

{{
  "anomalies": ["이상 징후 1", "이상 징후 2"],
  "probable_cause": "예상 원인 설명",
  "recommended_action": "권장 조치 사항"
}}

전문적이고 간결하게 작성하세요.
"""

    response = await asyncio.to_thread(model.generate_content, prompt)
    text = response.text.strip()

    # JSON 블록 추출 (```json ... ``` 형태 처리)
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1])

    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        logger.warning("Gemini 응답 JSON 파싱 실패, 기본 구조로 반환: %s", text[:200])
        result = {
            "anomalies": [text[:200]],
            "probable_cause": "Gemini 응답 파싱 실패 - 원문 참조",
            "recommended_action": "수동 점검 필요"
        }

    return result
