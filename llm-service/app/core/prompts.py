from typing import Dict, List, Optional

def get_diagnosis_prompt(
    equipment_id: str,
    prediction: str,
    confidence: float,
    sensors: Dict,
    anomalies: List[str],
    thermal_max_temp: Optional[float] = None
) -> str:
    """진단 프롬프트 생성"""
    
    anomaly_text = "\n".join(f"- {a}" for a in anomalies) if anomalies else "- 없음"
    
    prompt = f"""당신은 제조 설비 예지보전 전문가입니다. 아래 데이터를 분석하고 JSON 형식으로 진단 결과를 제공하세요.

## 장비 정보
- 장비 ID: {equipment_id}
- AI 예측: {prediction} 열화 (신뢰도: {confidence*100:.1f}%)

## 센서 데이터
- NTC (온도): {sensors.get('ntc', 0)}°C
- PM1.0: {sensors.get('pm1_0', 0)} μg/m³
- PM2.5: {sensors.get('pm2_5', 0)} μg/m³
- PM10: {sensors.get('pm10', 0)} μg/m³
- CT1 (전류1): {sensors.get('ct1', 0)} A
- CT2 (전류2): {sensors.get('ct2', 0)} A
- CT3 (전류3): {sensors.get('ct3', 0)} A
- CT4 (전류4): {sensors.get('ct4', 0)} A
- 열화상 최고온도: {thermal_max_temp if thermal_max_temp else 'N/A'}°C

## 감지된 이상
{anomaly_text}

## 응답 형식 (JSON만 출력)
{{
    "probable_cause": "추정되는 고장 원인 (한 문장)",
    "recommended_action": "권장 조치사항 (한 문장)"
}}
"""
    return prompt
