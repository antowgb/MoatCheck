"""Lightweight in-process ASGI response benchmark for the performance gate."""

import asyncio
import math
import time

import httpx

from app.main import app


def test_health_endpoint_p95_ttfb_under_concurrent_load():
    async def measure() -> list[float]:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://benchmark.test") as client:
            async def sample() -> float:
                started = time.perf_counter()
                response = await client.get("/health")
                elapsed_ms = (time.perf_counter() - started) * 1000
                assert response.status_code == 200
                return elapsed_ms

            await asyncio.gather(*(sample() for _ in range(10)))  # warm caches/import paths
            batches = await asyncio.gather(*(asyncio.gather(*(sample() for _ in range(20))) for _ in range(5)))
            return [latency for batch in batches for latency in batch]

    latencies = sorted(asyncio.run(measure()))
    p95_ms = latencies[math.ceil(len(latencies) * 0.95) - 1]
    print(f"health endpoint in-process ASGI p95 TTFB: {p95_ms:.2f} ms across {len(latencies)} requests")
    assert p95_ms < 150, f"p95 TTFB was {p95_ms:.2f} ms; budget is 150 ms"
