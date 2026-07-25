from __future__ import annotations

import argparse
import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from kama_claude.core.config import KamaConfig, get_config
from kama_claude.core.events.bus import EventBus
from kama_claude.core.events.journal import EventJournalCoordinator
from kama_claude.core.runner import AgentRunner, RunOutcome
from kama_claude.core.trace.writer import TraceWriter
from kama_claude.eval.models import WorkerRequest, WorkerResult


class _RunnerLike(Protocol):
    # 描述 worker 唯一需要的现有 AgentRunner one-shot 接口
    async def run_and_capture(self, goal: str, *, run_id: str) -> RunOutcome: ...


RunnerFactory = Callable[..., _RunnerLike]
ConfigLoader = Callable[[], KamaConfig]


# 执行单个公开 worker request，并用现有 AgentRunner wiring 生成 runtime 证据
async def execute_request(
    request: WorkerRequest,
    *,
    runner_factory: RunnerFactory = AgentRunner,
    config_loader: ConfigLoader = get_config,
) -> WorkerResult:
    bus = EventBus()
    journal = EventJournalCoordinator()
    bus.subscribe(journal.handle)
    trace = TraceWriter(Path(request.trace_path))
    await trace.start()
    try:
        runner = runner_factory(
            config_loader(),
            workspace_root=Path(request.workspace),
            bus=bus,
            runs_dir=Path(request.runs_dir),
            trace=trace,
            journal=journal,
        )
        outcome = await runner.run_and_capture(request.goal, run_id=request.run_id)
        return WorkerResult(
            run_id=request.run_id,
            runtime_status="success" if outcome.status == "success" else "failed",
            result=outcome.result,
            reason=outcome.reason,
        )
    except asyncio.CancelledError:
        raise
    except Exception:
        return WorkerResult(
            run_id=request.run_id,
            runtime_status="infra_error",
            infra_error="worker execution failed",
        )
    finally:
        try:
            await journal.flush_all()
        finally:
            try:
                await journal.close()
            finally:
                await trace.stop()


# 原子写入 worker result，避免 parent 读取到部分 JSON
def _write_result(path: Path, result: WorkerResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(result.model_dump_json(), encoding="utf-8")
    temporary.replace(path)


# 解析 worker 子进程的内部 request/result 路径参数
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="kama-eval-worker")
    parser.add_argument("--request", required=True)
    parser.add_argument("--result", required=True)
    return parser.parse_args()


# 读取严格 request、运行一次 AgentRunner 并只输出净化后的结构化结果
def main() -> None:
    args = _parse_args()
    request_path = Path(args.request)
    result_path = Path(args.result)
    try:
        request = WorkerRequest.model_validate_json(request_path.read_text(encoding="utf-8"))
        result = asyncio.run(execute_request(request))
    except Exception:
        result = WorkerResult(
            run_id="unknown",
            runtime_status="infra_error",
            infra_error="invalid worker request",
        )
    _write_result(result_path, result)


if __name__ == "__main__":
    main()
