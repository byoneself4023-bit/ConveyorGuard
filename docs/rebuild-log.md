# Rebuild Log — 단일 진단 플로우 통합 (rebuild/unified-diagnosis)

> 기준: `docs/Multi-Agent_Harness.md`(§2 A~H), `docs/6_Principle.md`
> 범위: 3분리 진단 경로 → 단일 `/diagnose` 플로우(구조 재설계). 6원칙 작업(`fix/six-principles`) 후속.
> before baseline: 이 환경엔 LLM/벡터 의존성·Gemini 키가 없어 **라이브 캡처 불가** → 아래 before는 **코드 기준** 기록(라이브 응답 아님). 검증은 LLM/임베딩/retriever 모킹 단위 테스트로.

## 무엇을 통합했나 (단계별)

| 단계 | 통합 내용 | before (3경로) | after (1플로우) | 사유 |
|---|---|---|---|---|
| 1 | RAG 일원화 | 코사인(`core/rag.py`, cases.json 10) + FAISS(`case_retriever.py`, 인라인 CASES 5) 2종 | 단일 FAISS 검색기 1종, 저장소 `cases.json` 정본(①SSoT). lazy 인덱스 + 로깅 폴백 | RAG 2구현·사례 저장소 2곳 중복 → 변경 시 양쪽 손봐야. 단일 출처로. |
| 2 | 그래프 RAG 참조 | 4-에이전트 그래프가 **RAG 미사용**(가장 똑똑한 경로가 근거 없이 추론) | retrieval 노드 추가 → `similar_cases` state → diagnostician·reviewer 프롬프트 주입(§D 환각방지/§H 컨텍스트) | 사례 근거 없는 진단은 환각 위험. 근거 주입 + reviewer 검증. |
| 3 | 단일 엔드포인트 | `/diagnose`(단일 Gemini), `/diagnose/graph`(그래프), `/diagnose/rag`(FAISS) 3진입점 | `/diagnose` 하나 → 그래프 실행, 구조화 출력. 구 2엔드포인트·데드코드 제거 | 같은 목적 3진입점(③일관성·SSoT 붕괴). 진입점 하나로. |
| 4 | 원자성 경계 확인 | (변경 없음 — 계약 불변) | 저장+알림 원자성 그대로 유효, 그래프는 트랜잭션 밖 | 통합 후에도 부분실패 방지가 유지되는지 확인. |

## 응답 계약 (불변)
`/diagnose`의 유일한 코드 소비자는 conveyorguard-api `pipeline.py:_persist_diagnosis_and_alert`로, `{anomalies, probable_cause, recommended_action}`를 저장한다. 통합 후에도 그래프 `finalize`가 이 구조화 필드를 산출(`core/gemini.py:parse_diagnosis_json` 재사용)해 **응답 스키마를 유지** → conveyorguard-api 무변경. 계약 테스트(`tests/test_endpoint.py`)로 응답 키 == 소비자 키 검증.

## 원자성 경계 (명시)
- **트랜잭션 안 (원자):** conveyorguard-api `_persist_diagnosis_and_alert` — 진단 저장 + 알림을 앱레벨 보상(알림 실패 시 진단 회수)으로 묶음. 이전 6원칙 작업에서 구현, 계약 불변이라 **그대로 유효**(회귀 `conveyorguard-api/tests/test_pipeline.py` 4 passed).
- **트랜잭션 밖:** 4-에이전트 그래프(외부 Gemini 다중 호출)는 HTTP 너머 llm-service에서 실행 → DB 트랜잭션에 넣지 않는다. 외부 호출을 트랜잭션에 묶으면 커넥션 장기 점유·롤백 무의미. 트랜잭션은 "그래프 종료 후 저장+알림"만 감싼다.

## Multi-Agent_Harness §2 점검 결과

| 항목 | 판정 | 근거 |
|---|---|---|
| **A 멀티에이전트 필요성 / 과설계** | 유지(정당) | analyzer·diagnostician·reviewer는 역할이 진짜 다르고 순차 의존. reviewer = 독립 검증층(D). retrieval·finalize는 LLM 역할이 아닌 유틸 노드(RAG 호출 / 포맷+구조화). 즉 LLM 에이전트 3 + 유틸 2 — 과분할 아님. (향후 단순화 여지: analyzer를 diagnostician에 흡수 가능하나, 검증 루프 가치를 위해 현행 유지.) |
| **B 순차/병렬** | 순차(정당) | 진단은 분석 출력 의존, 검토는 진단 출력 의존 → 순차 필연. 병렬화 대상 없음. |
| **C State** | 명시 | `similar_cases`(write=retrieval, read=diagnostician·reviewer), `structured`(write=finalize, read=엔드포인트 매핑). 안 쓰는 필드 없음. |
| **D 검증층** | 적용 | reviewer가 진단의 사례 근거 반영을 점검(APPROVE/REVISE), 재시도 루프 가드 `review_count>=2`(무한루프 방지). |
| **E 비용 게이트** | 적용 | 게이트(예측>=2)는 conveyorguard-api에 단일 유지(SSoT). 비싼 4-에이전트(약 4~5 LLM콜)를 게이트가 거름. `/diagnose`는 게이트 통과 입력에만 도달. |
| **F 실패 가시성** | 적용 | 그래프 노드 LLM 실패는 예외 전파 → `/diagnose` 500(빈 결과 둔갑 X). conveyorguard-api `_run_llm_diagnosis`는 HTTP 실패 시 로깅+폴백 진단. RAG 실패는 retriever에서 로깅+폴백. 트레이싱: `/metrics`(langsmith). |
| **G 하네스 병행** | 해당없음 | 단일 하네스(LangGraph). |
| **H 컨텍스트 엔지니어링** | 적용 | RAG 사례를 `_format_cases`로 구조화해 프롬프트에 주입. |

## 과적용 제외 / 의식적 한계
- 게이트를 `/diagnose` 안에 **중복 설치하지 않음** — 게이트는 conveyorguard-api 한 곳(SSoT). 가이드 다이어그램의 "게이트→그래프"는 두 서비스에 걸친 전체 플로우를 뜻하며, 진입점 정리(구 엔드포인트 제거)로 "단일 진입점"을 달성.
- `create_graph()`를 호출마다 컴파일(소소한 비용) — 캐싱은 향후 최적화로 보류.

## 환경 제약 (검증 한계)
- `sentence-transformers`가 torch<2.4로 실행 불가 → 실제 임베딩(FAISS) 경로는 이 환경에서 미실행. retriever가 로깅+폴백으로 graceful degrade하므로 서비스는 죽지 않음. **실제 임베딩·Gemini end-to-end는 사용자 환경(키 보유)에서 검증 권장.**
- 단위 테스트는 LLM(`get_llm`)·retriever·`run_diagnosis`를 모킹 — 키 없이 그래프 배선·RAG 주입·구조화 출력·엔드포인트 계약을 검증.

## 검증 결과
- `llm-service`: `pytest -q` → **7 passed** (RAG 3 / 그래프 2 / 엔드포인트 계약·404 2).
- `conveyorguard-api`: `pytest -q` → **4 passed** (계약 불변, 원자성 보상 유지).
- 구조 점검: `/diagnose/graph`·`/diagnose/rag`·`core/rag.py`·`diagnosis_tools.py` 제거 후 코드 내 잔존 참조 없음(docstring·404 테스트 제외).

## 커밋 매핑 (`rebuild/unified-diagnosis`, 한 단계=한 결정)
1. `refactor(RAG):` FAISS 단일화 + cases.json 정본
2. `feat(graph):` 4-에이전트 그래프가 통합 RAG 참조
3. `refactor(api):` /diagnose 단일 플로우 + 구 엔드포인트·데드코드 제거
4. `docs:` 원자성 경계 + Harness 점검 + rebuild-log (본 문서)
