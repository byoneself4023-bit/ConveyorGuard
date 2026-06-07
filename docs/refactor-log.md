# Refactor Log — 6원칙 점검·수정 (fix/six-principles)

> 기준: `docs/6_Principle.md` (SSoT·SRP·일관성·원자성·멱등성·말없는 fallback 금지)
> 범위: 코드 한 줄 품질(6원칙). 게이트·4에이전트·FAISS 합치는 3분리 통합 재구축은 별도(`ConveyorGuard_통합재구축_가이드.md`).
> 베이스라인: 기존 테스트 스위트 없음 → 동작 변경 카드(②·④) 검증용 최소 pytest를 먼저 도입(`conveyorguard-api/tests/`). 모든 수정 후 4 passed 유지.

## 수정 표

| 원칙 | 위치 | before | after | 사유(비용·장애 언어) |
|---|---|---|---|---|
| ⑥ 말없는 fallback | `llm-service/app/core/rag.py:14` | `except: return []` | `except (FileNotFoundError, JSONDecodeError)` → `logging.warning` 후 `[]` | 사례 저장소가 깨져도 "0건"과 구분 없이 조용히 무진단으로 넘어가 RAG 품질 저하가 로그 없이 묻혔다. 실패를 로그로 가시화. |
| ② SRP | `conveyorguard-api/app/services/pipeline.py` `run_pipeline()` | 수집·예측·state갱신·진단·저장·알림 6책임을 한 함수에 | `_collect_recent_sensors` / `_predict` / `_update_equipment_state` / `_persist_diagnosis_and_alert` 로 분리, `run_pipeline`은 흐름 조립만 | 변경 이유가 6개라 한 곳을 고치면 다른 책임이 깨질 위험. 책임별 분리로 변경 영향 격리. (동작 보존 — 회귀 테스트 green) |
| ④ 원자성 | `pipeline.py` `_persist_diagnosis_and_alert` | 진단 insert·알림 insert가 각각 독립 try/except → 절반 상태 가능 | 앱레벨 보상: 진단 실패 시 알림 생략, 알림 실패 시 진단 `delete`로 롤백 (전부-아니면-전무) | 진단 없는 알림 / 알림 없는 진단으로 운영자가 잘못된 상태를 보던 위험 제거. Supabase는 멀티테이블 트랜잭션 미지원이라 보상 방식 채택. |
| ① SSoT | `pipeline.py:98,118` 게이트 `>=2`/`==2` + 라벨 3곳 중복(`pipeline`·`api/schemas`·`core/preprocessing`) | 임계값 하드코딩, 라벨 중복 + pipeline 표기 불일치("경미/중간/심각") | `app/config.py` 신설(`SEVERITY_GATE`, `STATE_LABELS`), 세 곳은 config에서 re-export. `label=="중간"` 매직스트링 → `STATE_LABELS[SEVERITY_GATE]` | 게이트·라벨이 바뀌면 4곳을 고쳐야 했고 표기 불일치가 잠재 버그(`label=="중간"`). 이제 config 한 곳만 고친다. |

## 커밋 (한 커밋 = 한 결정)

1. `chore:` run_pipeline 회귀 테스트 하네스 추가 (pytest) — 베이스라인 안전망
2. `fix(⑥):` rag.py 사례 로딩 실패 좁히고 로깅
3. `refactor(②SRP):` run_pipeline 6책임을 단계별 헬퍼로 분리 (동작 보존)
4. `fix(④원자성):` 진단+알림 저장을 앱레벨 보상으로 전부-아니면-전무
5. `refactor(①SSoT):` 게이트 임계값·상태 라벨을 app.config 단일 정의로

## 보류 (이번 범위에서 수정 안 함)

- **⑤ 멱등성/재현성 — 합성 열화상** (`pipeline.py` `_predict`, `r["ntc"] * 0.8`): 재현이 아니라 실데이터 대체. **데이터 의존 → 실데이터 확보 시** 다룬다. (전처리 `GroupShuffleSplit(seed=42)`는 재현성 강함 — 유지.)
- **① SSoT — 사례 저장소 일원화**: `llm-service/app/data/cases.json`(10건, 코사인 유사도) vs `llm-service/app/rag/case_retriever.py` `CASES`(5건, FAISS) + 검색 경로 2개. **구조 의존 → 통합 재구축(FAISS 단일화)에서** 통합. 이번엔 임계값 config화까지만.

## 과적용 제외 / 의식적 한계

- **②SRP**: `run_pipeline`을 4개 헬퍼까지만 분리. 의미 단위가 하나인 블록은 더 쪼개지 않음(§3 SRP↔단순성 — 파일·간접호출 폭증 방지).
- **③일관성(부분 보류)**: 라벨 정의는 SSoT로 단일화했으나 이름이 API는 `STATE_LABELS`, core는 `LABEL_MAP`로 갈린 **네이밍 불일치**는 남겨둠. 통일하려면 `core/__init__` export·import처 전체를 함께 바꿔야 해(§3 일관성↔개선) 블래스트 반경이 커서 호환 별칭(`STATE_LABELS as LABEL_MAP`)으로 정의만 통합. 향후 일관성 카드로 분리 처리.

## 참고 (범위 밖, 통합 재구축 때 확인)

- `llm-service/app/tools/diagnosis_tools.py:233`이 `retriever.search_similar_cases(...)`를 호출하나 `case_retriever.py`의 `CaseRetriever`엔 `search()`만 존재 → 잠재 버그. 사례 저장소 통합 시 함께 정리.

## 검증

- `cd conveyorguard-api && pytest -q` → **4 passed** (게이트 미만/이상, 알림 실패 보상 롤백, 진단 실패 시 알림 생략).
- 라벨 단일화: `pipeline.STATE_LABELS is schemas.STATE_LABELS is preprocessing.LABEL_MAP is config.STATE_LABELS` 확인.
- rag.py: 정상 10건 로드 / 파일 없음 경로에서 `WARNING` 로깅 + `[]` 반환 확인.
