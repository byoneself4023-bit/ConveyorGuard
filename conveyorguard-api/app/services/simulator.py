"""센서 데이터 시뮬레이터 — 장비별 현실적 센서값 생성 + WebSocket broadcast"""

import asyncio
import json
import random
import math
from datetime import datetime, timezone
from typing import Set
from fastapi import WebSocket

from app.db import get_supabase


class SensorSimulator:
    def __init__(self):
        self.running = False
        self.clients: Set[WebSocket] = set()
        self._task: asyncio.Task | None = None
        self.tick = 0

        # 장비별 기준값 (state가 높을수록 센서 이상)
        self._profiles = {
            "OHT-001": {"state": 0, "ntc_base": 32, "pm_base": 15, "ct_base": 2.0},
            "OHT-002": {"state": 1, "ntc_base": 48, "pm_base": 30, "ct_base": 3.2},
            "OHT-003": {"state": 3, "ntc_base": 75, "pm_base": 90, "ct_base": 5.8},
            "OHT-004": {"state": 2, "ntc_base": 56, "pm_base": 50, "ct_base": 4.2},
            "OHT-005": {"state": 0, "ntc_base": 30, "pm_base": 12, "ct_base": 1.9},
        }

    def _generate_sensor(self, profile: dict) -> dict:
        """프로파일 기반 현실적 센서값 생성 (시간에 따라 서서히 변동)"""
        t = self.tick
        state = profile["state"]

        # 노이즈 크기: state가 높을수록 불안정
        noise_scale = 1 + state * 0.5

        ntc = profile["ntc_base"] + math.sin(t * 0.1) * 2 * noise_scale + random.gauss(0, 1 * noise_scale)
        pm2_5 = max(0, profile["pm_base"] + math.sin(t * 0.08) * 3 * noise_scale + random.gauss(0, 2 * noise_scale))
        pm1_0 = max(0, pm2_5 * 0.6 + random.gauss(0, 1))
        pm10 = pm2_5 * 1.3 + random.gauss(0, 2)

        ct_base = profile["ct_base"]
        ct1 = max(0, ct_base + math.sin(t * 0.12) * 0.3 * noise_scale + random.gauss(0, 0.1 * noise_scale))
        ct2 = max(0, ct_base - 0.2 + random.gauss(0, 0.1 * noise_scale))
        ct3 = max(0, ct_base + 0.1 + random.gauss(0, 0.1 * noise_scale))
        ct4 = max(0, ct_base + random.gauss(0, 0.15 * noise_scale))

        return {
            "ntc": round(ntc, 1),
            "pm1_0": round(pm1_0, 1),
            "pm2_5": round(pm2_5, 1),
            "pm10": round(pm10, 1),
            "ct1": round(ct1, 2),
            "ct2": round(ct2, 2),
            "ct3": round(ct3, 2),
            "ct4": round(ct4, 2),
        }

    async def _broadcast(self, message: dict):
        """모든 클라이언트에 메시지 전송"""
        dead = set()
        data = json.dumps(message, ensure_ascii=False)
        for ws in self.clients:
            try:
                await ws.send_text(data)
            except Exception:
                dead.add(ws)
        self.clients -= dead

    async def _loop(self):
        """5초 간격으로 센서 데이터 생성 → DB 저장 → AI 파이프라인 → broadcast"""
        from app.services.pipeline import run_pipeline

        sb = get_supabase()
        pipeline_counter = 0  # 매 6틱(30초)마다 파이프라인 실행

        while self.running:
            self.tick += 1
            pipeline_counter += 1
            batch = []

            for eq_id, profile in self._profiles.items():
                sensors = self._generate_sensor(profile)
                record = {"equipment_id": eq_id, **sensors}
                batch.append(record)

            # DB에 벌크 insert
            try:
                sb.table("sensor_data").insert(batch).execute()
            except Exception:
                pass  # DB 실패해도 broadcast는 계속

            # 각 장비별로 broadcast + 주기적 파이프라인
            for item in batch:
                sensors_only = {k: v for k, v in item.items() if k != "equipment_id"}
                msg = {
                    "type": "sensor_update",
                    "equipment_id": item["equipment_id"],
                    "sensors": sensors_only,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }

                # 30초마다 AI 파이프라인 실행
                if pipeline_counter >= 6:
                    try:
                        pipeline_result = await run_pipeline(item["equipment_id"], sensors_only)
                        if pipeline_result:
                            msg["prediction"] = pipeline_result.get("prediction")
                            msg["confidence"] = pipeline_result.get("confidence")
                            msg["label"] = pipeline_result.get("label")
                            if "diagnosis" in pipeline_result:
                                msg["diagnosis"] = pipeline_result["diagnosis"]
                            if "alert" in pipeline_result:
                                msg["alert"] = pipeline_result["alert"]
                    except Exception:
                        pass  # 파이프라인 실패해도 센서 데이터는 전송

                await self._broadcast(msg)

            if pipeline_counter >= 6:
                pipeline_counter = 0

            await asyncio.sleep(5)

    def start(self):
        if self.running:
            return
        self.running = True
        self.tick = 0
        self._task = asyncio.create_task(self._loop())

    def stop(self):
        self.running = False
        if self._task:
            self._task.cancel()
            self._task = None

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.clients.add(ws)

    def disconnect(self, ws: WebSocket):
        self.clients.discard(ws)


# 싱글톤
simulator = SensorSimulator()
