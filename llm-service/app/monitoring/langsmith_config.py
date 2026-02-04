"""
LangSmith 모니터링 + 로깅
"""
import os
from datetime import datetime
from functools import wraps
import time
import json


def setup_langsmith(project: str = "conveyorguard"):
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = project
    return bool(os.environ.get("LANGCHAIN_API_KEY"))


class Logger:
    def __init__(self, log_dir: str = "./logs"):
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir
        self.logs = []
    
    def log(self, request_id: str, equipment_id: str, latency_ms: float, success: bool, error: str = None):
        entry = {
            "ts": datetime.now().isoformat(),
            "request_id": request_id,
            "equipment_id": equipment_id,
            "latency_ms": latency_ms,
            "success": success,
            "error": error
        }
        self.logs.append(entry)
        
        with open(f"{self.log_dir}/diagnosis_{datetime.now():%Y-%m-%d}.jsonl", "a") as f:
            f.write(json.dumps(entry) + "\n")
    
    def metrics(self, n: int = 100) -> dict:
        recent = self.logs[-n:]
        if not recent:
            return {"total": 0}
        ok = [l for l in recent if l["success"]]
        return {
            "total": len(recent),
            "success_rate": len(ok) / len(recent) * 100,
            "avg_latency_ms": sum(l["latency_ms"] for l in ok) / len(ok) if ok else 0
        }


logger = Logger()


def track(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        import uuid
        rid = str(uuid.uuid4())[:8]
        start = time.time()
        try:
            result = await func(*args, **kwargs)
            logger.log(rid, kwargs.get("equipment_id", "?"), (time.time()-start)*1000, True)
            return result
        except Exception as e:
            logger.log(rid, kwargs.get("equipment_id", "?"), (time.time()-start)*1000, False, str(e))
            raise
    return wrapper
