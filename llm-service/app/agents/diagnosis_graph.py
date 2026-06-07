"""
LangGraph 기반 멀티 에이전트 진단 시스템.

플로우: retrieval → analyzer → diagnostician → reviewer →(APPROVE/2회)→ finalize
통합 RAG(app.rag.case_retriever)를 retrieval 노드에서 호출해 사례 근거를
diagnostician·reviewer에 주입한다(환각 방지 / 컨텍스트 엔지니어링).
"""
import asyncio
from typing import TypedDict, List, Dict
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

from app.rag.case_retriever import retriever
from app.core.gemini import parse_diagnosis_json

load_dotenv()


class DiagnosisState(TypedDict):
    equipment_id: str
    prediction_result: dict
    sensor_data: dict
    similar_cases: List[Dict]   # retrieval 노드가 write, diagnostician·reviewer가 read
    analysis: str
    diagnosis: str
    review: str
    review_count: int
    final_report: str
    structured: dict            # finalize가 write — conveyorguard-api persist 계약 필드
    status: str


def get_llm():
    # 지연 import: 모듈 import 시 외부 의존성 강제하지 않음
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)


def _format_cases(cases: List[Dict]) -> str:
    """프롬프트 주입용 사례 컨텍스트 포맷."""
    if not cases:
        return "(유사 과거 사례 없음)"
    return "\n".join(
        f"- [{c.get('case_id', '')}] {c.get('issue', '')} → 조치: {c.get('action', '')} "
        f"(유사도 {c.get('similarity', 0)})"
        for c in cases
    )


def retrieval_node(state: DiagnosisState) -> dict:
    """통합 RAG: 센서·예측으로 유사 사례를 검색해 state에 싣는다."""
    sensor = state["sensor_data"]
    pred = state["prediction_result"]
    query = (
        f"{pred.get('label', '')} "
        f"NTC={sensor.get('ntc')} CT1={sensor.get('ct1')} PM2.5={sensor.get('pm2_5')}"
    )
    return {"similar_cases": retriever.search(query, k=3)}


def analyzer_node(state: DiagnosisState) -> dict:
    llm = get_llm()
    sensor = state["sensor_data"]
    pred = state["prediction_result"]

    prompt = f"""제조설비 센서 분석 전문가입니다.
장비: {state['equipment_id']}
예측: {pred.get('label')} (신뢰도: {pred.get('confidence', 0):.1%})
센서: NTC={sensor.get('ntc')}°C, CT1={sensor.get('ct1')}A, PM2.5={sensor.get('pm2_5')}

임계값 초과 항목과 이상 징후를 분석하세요."""

    return {"analysis": llm.invoke(prompt).content}


def diagnostician_node(state: DiagnosisState) -> dict:
    llm = get_llm()
    review = state.get("review", "")
    cases = _format_cases(state.get("similar_cases", []))

    prompt = f"""제조설비 진단 전문가입니다.
분석: {state['analysis']}

## 유사 과거 사례 (근거로 활용)
{cases}
{"피드백: " + review if "REVISE" in review else ""}

위 사례를 근거로 원인, 긴급도(즉시/24시간/1주일/정기), 조치방안을 제시하세요."""

    return {"diagnosis": llm.invoke(prompt).content}


def reviewer_node(state: DiagnosisState) -> dict:
    llm = get_llm()
    cases = _format_cases(state.get("similar_cases", []))

    prompt = f"""품질관리 책임자입니다.
분석: {state['analysis']}
진단: {state['diagnosis']}

## 참고 사례
{cases}

진단이 위 사례 근거를 반영했는지 점검하세요. 검토 후 마지막에 APPROVE 또는 REVISE 작성."""

    return {"review": llm.invoke(prompt).content, "review_count": state.get("review_count", 0) + 1}


def finalize_node(state: DiagnosisState) -> dict:
    """승인된 진단을 구조화 필드로 변환(계약 보존) + 리포트 조립."""
    llm = get_llm()
    structuring_prompt = f"""다음 진단을 JSON으로만 변환하세요. 다른 텍스트 없이 JSON만 출력.
진단 내용: {state['diagnosis']}

형식:
{{"anomalies": ["이상 징후 1", "이상 징후 2"], "probable_cause": "예상 원인", "recommended_action": "권장 조치"}}"""
    structured = parse_diagnosis_json(llm.invoke(structuring_prompt).content)

    cases = _format_cases(state.get("similar_cases", []))
    report = f"""# 진단 리포트
## 장비: {state['equipment_id']}
### 분석
{state['analysis']}
### 진단
{state['diagnosis']}
### 검토
{state['review']}
### 참고 사례
{cases}"""
    return {"final_report": report, "structured": structured, "status": "approved"}


def should_continue(state: DiagnosisState) -> str:
    if state.get("review_count", 0) >= 2 or "APPROVE" in state.get("review", "").upper():
        return "finalize"
    return "revise"


def create_graph():
    wf = StateGraph(DiagnosisState)
    wf.add_node("retrieval", retrieval_node)
    wf.add_node("analyzer", analyzer_node)
    wf.add_node("diagnostician", diagnostician_node)
    wf.add_node("reviewer", reviewer_node)
    wf.add_node("finalize", finalize_node)

    wf.set_entry_point("retrieval")
    wf.add_edge("retrieval", "analyzer")
    wf.add_edge("analyzer", "diagnostician")
    wf.add_edge("diagnostician", "reviewer")
    wf.add_conditional_edges("reviewer", should_continue, {"revise": "diagnostician", "finalize": "finalize"})
    wf.add_edge("finalize", END)

    return wf.compile()


async def run_diagnosis(equipment_id: str, prediction_result: dict, sensor_data: dict) -> dict:
    """그래프 실행 후 전체 최종 state 반환(similar_cases·리포트 포함)."""
    graph = create_graph()
    state = {
        "equipment_id": equipment_id,
        "prediction_result": prediction_result,
        "sensor_data": sensor_data,
        "similar_cases": [],
        "analysis": "", "diagnosis": "", "review": "",
        "review_count": 0, "final_report": "", "structured": {}, "status": "",
    }
    return await asyncio.to_thread(graph.invoke, state)
