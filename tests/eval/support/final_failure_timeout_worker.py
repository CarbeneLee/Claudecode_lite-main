from __future__ import annotations

import argparse
import asyncio
from datetime import UTC, datetime
from pathlib import Path

from kama_claude.core.bus.events import LlmModelSelectedEvent, RunStartedEvent
from kama_claude.core.config import KamaConfig
from kama_claude.core.context import ExecutionContext
from kama_claude.core.events.bus import EventBus
from kama_claude.core.events.journal import EventJournalCoordinator
from kama_claude.core.llm.types import LlmResponse, ToolCallBlock
from kama_claude.core.loop import AgentLoop
from kama_claude.core.runner import RunOutcome
from kama_claude.core.tools.base import BaseTool, ToolResult
from kama_claude.core.tools.registry import ToolRegistry
from kama_claude.core.trace.provider import TracingProvider
from kama_claude.core.trace.writer import TraceWriter
from kama_claude.eval.models import WorkerRequest
from kama_claude.eval.worker import execute_request


# 返回测试 producer 事件使用的当前 UTC 时间
def _now() -> str:
    return datetime.now(UTC).isoformat()


class _FinalFailureTool(BaseTool):
    name = "final_failure"
    description = "Return a deterministic non-retryable failure."
    input_schema: dict[str, object] = {"type": "object", "properties": {}}

    # 通过真实 invocation path 返回非 retryable final failure
    async def invoke(self, params: dict[str, object]) -> ToolResult:
        return ToolResult(
            content="expected failure",
            is_error=True,
            error_type="command_failed",
        )


class _FailureThenBlockProvider:
    # 初始化模型身份、第二步 ready marker 与调用计数
    def __init__(self, model: str, ready_marker: Path) -> None:
        self._model = model
        self._ready_marker = ready_marker
        self._calls = 0
        self._never_release = asyncio.Event()

    # 发布真实 model event，第一步请求工具并在第二步永久阻塞
    async def chat(
        self,
        messages: list[dict[str, object]],
        tool_schemas: list[dict[str, object]],
        bus: EventBus,
        run_id: str,
        *,
        step: int = 0,
        system: str | None = None,
    ) -> LlmResponse:
        self._calls += 1
        await bus.publish(
            LlmModelSelectedEvent(
                run_id=run_id,
                model=self._model,
                strategy="static",
                ts=_now(),
            )
        )
        if self._calls == 1:
            return LlmResponse(
                stop_reason="tool_use",
                tool_calls=[
                    ToolCallBlock(
                        id="tool-final-failure",
                        name="final_failure",
                        input={},
                    )
                ],
            )
        self._ready_marker.write_text("ready", encoding="utf-8")
        await self._never_release.wait()
        return LlmResponse(stop_reason="end_turn", text="unreachable")


class _TimeoutRunner:
    # 保存 execute_request 注入的真实 observer资源与测试配置
    def __init__(
        self,
        config: KamaConfig,
        *,
        workspace_root: Path,
        bus: EventBus,
        runs_dir: Path,
        trace: TraceWriter,
        journal: EventJournalCoordinator,
        **_: object,
    ) -> None:
        self._config = config
        self._workspace_root = workspace_root
        self._bus = bus
        self._runs_dir = runs_dir
        self._trace = trace
        self._journal = journal

    # 运行真实 AgentLoop 并在第二步阻塞直到 parent timeout 终止进程
    async def run_and_capture(self, goal: str, *, run_id: str) -> RunOutcome:
        run_path = self._runs_dir / run_id
        run_path.mkdir(parents=True, exist_ok=True)
        await self._journal.register_run(run_id, run_path, session_id=None)
        await self._bus.publish(
            RunStartedEvent(
                run_id=run_id,
                goal=goal,
                ts=_now(),
            )
        )
        registry = ToolRegistry()
        registry.register(_FinalFailureTool())
        provider = TracingProvider(
            _FailureThenBlockProvider(
                self._config.llm.default_model,
                self._workspace_root / "step-2-ready",
            ),
            self._trace,
            include_payload=True,
        )
        context = ExecutionContext(
            run_id=run_id,
            goal=goal,
            max_steps=self._config.agent.max_steps,
        )
        await AgentLoop(provider, registry, self._bus).run(context)
        return RunOutcome(
            status=context.status,
            result=context.result,
            reason=context.reason,
        )


# 解析 production worker兼容的 request/result参数
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    parser.add_argument("--result", required=True)
    return parser.parse_args()


# 使用 test-only runner执行真实 worker observer路径并等待 parent timeout
def main() -> None:
    args = _parse_args()
    request = WorkerRequest.model_validate_json(
        Path(args.request).read_text(encoding="utf-8")
    )
    asyncio.run(execute_request(request, runner_factory=_TimeoutRunner))


if __name__ == "__main__":
    main()
