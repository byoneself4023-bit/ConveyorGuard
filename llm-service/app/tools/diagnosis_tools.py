# llm-service/app/tools/diagnosis_tools.py
"""
LangChain Tools for ConveyorGuard
- 장비 정보 조회 (DB)
- ML API 호출
- 유사 사례 검색
- 정비 이력 조회
"""

from typing import Optional, Type
from langchain.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun
from pydantic import BaseModel, Field
import httpx
import json


# ============================================
# 1. Tool Input Schemas
# ============================================
class EquipmentQueryInput(BaseModel):
    """장비 정보 조회 입력"""
    equipment_id: str = Field(description="장비 ID (예: OHT-003)")


class SensorDataInput(BaseModel):
    """센서 데이터 조회 입력"""
    equipment_id: str = Field(description="장비 ID")
    time_range: str = Field(default="1h", description="조회 기간 (1h, 6h, 24h, 7d)")


class MaintenanceHistoryInput(BaseModel):
    """정비 이력 조회 입력"""
    equipment_id: str = Field(description="장비 ID")
    limit: int = Field(default=5, description="조회할 이력 수")


class SimilarCaseInput(BaseModel):
    """유사 사례 검색 입력"""
    symptom: str = Field(description="현재 증상 설명")
    equipment_type: str = Field(default=None, description="장비 유형 (OHT/AGV)")


class MLPredictionInput(BaseModel):
    """ML 예측 요청 입력"""
    equipment_id: str = Field(description="장비 ID")


# ============================================
# 2. Mock Data (실제로는 DB에서 조회)
# ============================================
EQUIPMENT_DB = {
    "OHT-001": {
        "id": "OHT-001",
        "type": "OHT",
        "location": "FAB1-Zone A",
        "install_date": "2022-03-15",
        "last_maintenance": "2025-01-10",
        "status": "정상"
    },
    "OHT-002": {
        "id": "OHT-002",
        "type": "OHT",
        "location": "FAB1-Zone B",
        "install_date": "2022-03-15",
        "last_maintenance": "2025-01-08",
        "status": "주의"
    },
    "OHT-003": {
        "id": "OHT-003",
        "type": "OHT",
        "location": "FAB2-Zone A",
        "install_date": "2021-11-20",
        "last_maintenance": "2024-12-20",
        "status": "심각"
    },
    "AGV-001": {
        "id": "AGV-001",
        "type": "AGV",
        "location": "Warehouse A",
        "install_date": "2023-06-01",
        "last_maintenance": "2025-01-12",
        "status": "정상"
    }
}

MAINTENANCE_HISTORY = {
    "OHT-003": [
        {"date": "2024-12-20", "type": "정기점검", "description": "벨트 장력 조정"},
        {"date": "2024-10-15", "type": "수리", "description": "베어링 교체"},
        {"date": "2024-08-01", "type": "정기점검", "description": "윤활유 보충"},
        {"date": "2024-05-20", "type": "수리", "description": "센서 교체"},
        {"date": "2024-03-10", "type": "정기점검", "description": "전체 점검"}
    ]
}


# ============================================
# 3. Tools 정의
# ============================================
class EquipmentInfoTool(BaseTool):
    """장비 정보 조회 Tool"""
    name: str = "get_equipment_info"
    description: str = "장비의 기본 정보를 조회합니다. 장비 ID, 유형, 위치, 설치일, 마지막 정비일 등을 반환합니다."
    args_schema: Type[BaseModel] = EquipmentQueryInput
    
    def _run(
        self,
        equipment_id: str,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """장비 정보 조회 실행"""
        equipment = EQUIPMENT_DB.get(equipment_id)
        
        if not equipment:
            return f"장비 '{equipment_id}'를 찾을 수 없습니다."
        
        return json.dumps(equipment, ensure_ascii=False, indent=2)


class MaintenanceHistoryTool(BaseTool):
    """정비 이력 조회 Tool"""
    name: str = "get_maintenance_history"
    description: str = "장비의 정비 이력을 조회합니다. 과거 정비 기록, 수리 내역 등을 반환합니다."
    args_schema: Type[BaseModel] = MaintenanceHistoryInput
    
    def _run(
        self,
        equipment_id: str,
        limit: int = 5,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """정비 이력 조회 실행"""
        history = MAINTENANCE_HISTORY.get(equipment_id, [])
        
        if not history:
            return f"장비 '{equipment_id}'의 정비 이력이 없습니다."
        
        return json.dumps(history[:limit], ensure_ascii=False, indent=2)


class SensorDataTool(BaseTool):
    """센서 데이터 조회 Tool"""
    name: str = "get_sensor_data"
    description: str = "장비의 현재 센서 데이터를 조회합니다. 온도, 전류, 먼지 농도 등을 반환합니다."
    args_schema: Type[BaseModel] = SensorDataInput
    
    def _run(
        self,
        equipment_id: str,
        time_range: str = "1h",
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """센서 데이터 조회 (Mock)"""
        # 실제로는 DB 또는 API에서 조회
        mock_data = {
            "equipment_id": equipment_id,
            "timestamp": "2025-01-15T11:00:00",
            "sensors": {
                "NTC": 78.5,
                "PM1_0": 45,
                "PM2_5": 102,
                "PM10": 128,
                "CT1": 5.8,
                "CT2": 6.2,
                "CT3": 5.5,
                "CT4": 6.8
            },
            "thermal_max": 85.2
        }
        
        return json.dumps(mock_data, ensure_ascii=False, indent=2)


class MLPredictionTool(BaseTool):
    """ML 예측 요청 Tool"""
    name: str = "get_ml_prediction"
    description: str = "ML 모델을 호출하여 장비의 열화 상태를 예측합니다."
    args_schema: Type[BaseModel] = MLPredictionInput
    
    ml_api_url: str = "http://localhost:8000"
    
    def _run(
        self,
        equipment_id: str,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """ML 예측 실행"""
        try:
            # 실제 API 호출 (비동기 필요시 httpx.AsyncClient 사용)
            with httpx.Client(timeout=30) as client:
                response = client.get(
                    f"{self.ml_api_url}/predict/test",
                    params={"equipment_id": equipment_id}
                )
                
                if response.status_code == 200:
                    return response.text
                else:
                    return f"ML API 오류: {response.status_code}"
                    
        except Exception as e:
            # 연결 실패시 Mock 데이터 반환
            mock_result = {
                "equipment_id": equipment_id,
                "prediction": {
                    "class": 3,
                    "label": "심각",
                    "confidence": 0.92,
                    "probabilities": [0.02, 0.03, 0.03, 0.92]
                }
            }
            return json.dumps(mock_result, ensure_ascii=False, indent=2)


class SimilarCaseTool(BaseTool):
    """유사 사례 검색 Tool"""
    name: str = "search_similar_cases"
    description: str = "현재 증상과 유사한 과거 장애 사례를 검색합니다."
    args_schema: Type[BaseModel] = SimilarCaseInput
    
    def _run(
        self,
        symptom: str,
        equipment_type: str = None,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """유사 사례 검색 (RAG 연동)"""
        try:
            from app.rag.case_retriever import CaseRetriever
            
            retriever = CaseRetriever()
            cases = retriever.search_similar_cases(
                query=symptom,
                equipment_type=equipment_type,
                k=3
            )
            
            return json.dumps(cases, ensure_ascii=False, indent=2)
            
        except Exception as e:
            # RAG 실패시 Mock 반환
            mock_cases = [
                {
                    "case_id": "CASE-001",
                    "cause": "베어링 마모",
                    "similarity": 0.85
                }
            ]
            return json.dumps(mock_cases, ensure_ascii=False, indent=2)


# ============================================
# 4. Tool 목록
# ============================================
def get_diagnosis_tools():
    """진단에 사용할 Tool 목록 반환"""
    return [
        EquipmentInfoTool(),
        MaintenanceHistoryTool(),
        SensorDataTool(),
        MLPredictionTool(),
        SimilarCaseTool()
    ]


# ============================================
# 5. Agent with Tools
# ============================================
def create_diagnosis_agent():
    """Tool을 사용하는 진단 Agent 생성"""
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain.agents import create_tool_calling_agent, AgentExecutor
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    
    # LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3
    )
    
    # Tools
    tools = get_diagnosis_tools()
    
    # Prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", """당신은 제조설비 진단 전문가입니다.
사용 가능한 도구를 활용하여 장비 상태를 진단하세요.

진단 절차:
1. 장비 정보 조회
2. 정비 이력 확인
3. 센서 데이터 조회
4. ML 예측 실행
5. 유사 사례 검색
6. 종합 진단 결과 작성

필요한 도구를 적절히 사용하세요."""),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])
    
    # Agent
    agent = create_tool_calling_agent(llm, tools, prompt)
    
    # Executor
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=10
    )
    
    return executor


# ============================================
# 6. 테스트
# ============================================
if __name__ == "__main__":
    # 개별 Tool 테스트
    print("=== Tool 테스트 ===\n")
    
    # 장비 정보
    tool = EquipmentInfoTool()
    print("장비 정보:")
    print(tool._run("OHT-003"))
    
    print("\n" + "=" * 40 + "\n")
    
    # 정비 이력
    tool = MaintenanceHistoryTool()
    print("정비 이력:")
    print(tool._run("OHT-003", limit=3))
    
    print("\n" + "=" * 40 + "\n")
    
    # 센서 데이터
    tool = SensorDataTool()
    print("센서 데이터:")
    print(tool._run("OHT-003"))
