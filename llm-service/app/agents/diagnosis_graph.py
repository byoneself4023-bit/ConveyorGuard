"""
LangGraph 기반 멀티 에이전트 진단 시스템
"""
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()


class DiagnosisState(TypedDict):
    equipment_id: str
    prediction_result: dict
    sensor_data: dict
    analysis: str
    diagnosis: str
    review: str
    review_count: int
    final_report: str
    status: str


def get_llm():
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)


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
    
    prompt = f"""제조설비 진단 전문가입니다.
분석: {state['analysis']}
{"피드백: " + review if "REVISE" in review else ""}

원인, 긴급도(즉시/24시간/1주일/정기), 조치방안을 제시하세요."""

    return {"diagnosis": llm.invoke(prompt).content}


def reviewer_node(state: DiagnosisState) -> dict:
    llm = get_llm()
    
    prompt = f"""품질관리 책임자입니다.
분석: {state['analysis']}
진단: {state['diagnosis']}

검토 후 마지막에 APPROVE 또는 REVISE 작성."""

    return {"review": llm.invoke(prompt).content, "review_count": state.get("review_count", 0) + 1}


def finalize_node(state: DiagnosisState) -> dict:
    report = f"""# 진단 리포트
## 장비: {state['equipment_id']}
### 분석
{state['analysis']}
### 진단
{state['diagnosis']}
### 검토
{state['review']}"""
    return {"final_report": report, "status": "approved"}


def should_continue(state: DiagnosisState) -> str:
    if state.get("review_count", 0) >= 2 or "APPROVE" in state.get("review", "").upper():
        return "finalize"
    return "revise"


def create_graph():
    wf = StateGraph(DiagnosisState)
    wf.add_node("analyzer", analyzer_node)
    wf.add_node("diagnostician", diagnostician_node)
    wf.add_node("reviewer", reviewer_node)
    wf.add_node("finalize", finalize_node)
    
    wf.set_entry_point("analyzer")
    wf.add_edge("analyzer", "diagnostician")
    wf.add_edge("diagnostician", "reviewer")
    wf.add_conditional_edges("reviewer", should_continue, {"revise": "diagnostician", "finalize": "finalize"})
    wf.add_edge("finalize", END)
    
    return wf.compile()


async def run_diagnosis(equipment_id: str, prediction_result: dict, sensor_data: dict) -> dict:
    graph = create_graph()
    state = {
        "equipment_id": equipment_id,
        "prediction_result": prediction_result,
        "sensor_data": sensor_data,
        "analysis": "", "diagnosis": "", "review": "",
        "review_count": 0, "final_report": "", "status": ""
    }
    
    result = None
    for output in graph.stream(state):
        result = output
    
    return result[list(result.keys())[0]] if result else state
