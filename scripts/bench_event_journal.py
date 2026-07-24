from __future__ import annotations

import argparse
import asyncio
import statistics
import tempfile
import time
from pathlib import Path

from kama_claude.core.bus.events import LlmTokenEvent, RunFinishedEvent
from kama_claude.core.events.journal import EventJournalCoordinator


# 解析 benchmark 参数，不设置性能通过阈值
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure v2 event journal behavior")
    parser.add_argument("--events", type=int, default=10_000)
    return parser.parse_args()


# 返回排序样本的最近秩百分位毫秒值
def _percentile(samples_ns: list[int], percentile: float) -> float:
    ordered = sorted(samples_ns)
    index = max(0, min(len(ordered) - 1, int(len(ordered) * percentile) - 1))
    return ordered[index] / 1_000_000


# 在临时 run/session journals 上收集 enqueue、flush 与 terminal latency 证据
async def _benchmark(event_count: int) -> dict[str, float | int]:
    if event_count <= 0:
        raise ValueError("--events must be positive")
    with tempfile.TemporaryDirectory(prefix="kama-journal-bench-") as temp_dir:
        root = Path(temp_dir)
        coordinator = EventJournalCoordinator()
        session_path = root / "sessions" / "sess-bench"
        run_path = session_path / "runs" / "run-bench"
        await coordinator.register_session("sess-bench", session_path)
        await coordinator.register_run(
            "run-bench",
            run_path,
            session_id="sess-bench",
        )
        enqueue_ns: list[int] = []
        started = time.perf_counter_ns()
        for index in range(event_count):
            before = time.perf_counter_ns()
            await coordinator.handle(
                LlmTokenEvent(
                    run_id="run-bench",
                    token=f"token-{index}",
                    ts="2026-07-21T00:00:00Z",
                )
            )
            enqueue_ns.append(time.perf_counter_ns() - before)
            if (index + 1) % 2048 == 0:
                await coordinator.flush_all()
        await coordinator.flush_all()
        flush_elapsed_s = (time.perf_counter_ns() - started) / 1_000_000_000

        terminal_started = time.perf_counter_ns()
        await coordinator.handle(
            RunFinishedEvent(
                run_id="run-bench",
                status="success",
                steps=1,
                ts="2026-07-21T00:00:01Z",
            )
        )
        terminal_ms = (time.perf_counter_ns() - terminal_started) / 1_000_000
        run_metrics = coordinator.stream_metrics("run:run-bench")
        session_metrics = coordinator.stream_metrics("session:sess-bench")
        await coordinator.close()

        durable_records = event_count * 2 + 2
        return {
            "events": event_count,
            "durable_records": durable_records,
            "enqueue_mean_ms": statistics.fmean(enqueue_ns) / 1_000_000,
            "enqueue_p50_ms": _percentile(enqueue_ns, 0.50),
            "enqueue_p95_ms": _percentile(enqueue_ns, 0.95),
            "flush_records_per_s": durable_records / flush_elapsed_s,
            "max_queue_bytes": max(
                run_metrics["peak_bytes"],
                session_metrics["peak_bytes"],
            ),
            "terminal_flush_ms": terminal_ms,
        }


# 输出当前机器证据，结果仅用于观察而不作为跨平台 SLA
def main() -> None:
    args = _parse_args()
    result = asyncio.run(_benchmark(args.events))
    print("event journal benchmark (evidence only; no pass/fail threshold)")
    for key, value in result.items():
        if isinstance(value, float):
            print(f"{key}={value:.3f}")
        else:
            print(f"{key}={value}")


if __name__ == "__main__":
    main()
