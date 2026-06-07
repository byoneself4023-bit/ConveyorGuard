"""테스트 공통 설정.

목적: run_pipeline 같은 서비스 로직을 무거운 외부 SDK 없이 import/검증한다.
- supabase SDK는 설치 없이 동작하도록 stub 모듈을 주입한다(실제 DB는 테스트에서 모킹).
- import 시점에 필요한 더미 env를 채워 app.db import가 깨지지 않게 한다.
"""

import os
import sys
import types

# app.db.get_supabase가 import 시 요구하는 env (값은 테스트에서 쓰이지 않음)
os.environ.setdefault("SUPABASE_URL", "http://localhost")
os.environ.setdefault("SUPABASE_KEY", "test-key")

# supabase SDK 미설치 환경에서도 import 체인이 살아있도록 최소 stub 주입
if "supabase" not in sys.modules:
    _stub = types.ModuleType("supabase")
    _stub.create_client = lambda *a, **k: None  # 실제 호출은 테스트에서 모킹
    _stub.Client = object
    sys.modules["supabase"] = _stub
