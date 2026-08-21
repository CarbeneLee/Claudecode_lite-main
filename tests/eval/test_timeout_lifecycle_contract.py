from __future__ import annotations

import asyncio
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

import kama_claude.eval.runner as eval_runner
from kama_claude.benchmark.experiment import EvidenceMode, collect_observed_identity
from kama_claude.core.bus.events import RunStartedEvent
from kama_claude.core.context import ExecutionContext
from kama_claude.core.events.bus import EventBus
from kama_claude.core.events.journal import EventJournalCoordinator
from kama_claude.core.llm.types import LlmResponse, ToolCallBlock
from kama_claude.core.loop import AgentLoop
from kama_claude.core.tools.base import BaseTool, ToolResult
from kama_claude.core.tools.invocation import DirectToolInvoker
from kama_claude.core.tools.registry import ToolRegistry
from kama_claude.eval.evaluator import evaluate_task
from kama_claude.eval.failure import FailureCategory
from kama_claude.eval.graders import grade_timeout_trace_prefix


class _FinalFailureTool(BaseTool):
    name = "final_failure"
    description = "Return a deterministic non-retryable failure."
    input_schema: dict[str, object] = {"type": "object", "properties": {}}

    # 通过真实 invocation path 返回稳定 final failure
    async def invoke(self, params: dict[str, object]) -> ToolResult:
        return ToolResult(
            content="expected failure",
            is_error=True,
            error_type="command_failed",
        )


class _FailureThenBlockProvider:
    # 初始化调用计数和第二步到达屏障
    def __init__(self) -> None:
        self.calls = 0
        self.second_call_started = asyncio.Event()
        self.release_second_call = asyncio.Event()

    # 第一步请求失败工具，第二步阻塞以形成 timeout-style open prefix
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
        self.calls += 1
        if self.calls == 1:
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
        self.second_call_started.set()
        await self.release_second_call.wait()
        return LlmResponse(stop_reason="end_turn", text="unreachable")


# 返回测试事件使用的当前 UTC 时间
def _now() -> str:
    return datetime.now(UTC).isoformat()


# 创建使用极短 timeout 且不含可通过 private grader 的可信任务
def _timeout_task_dir(tmp_path: Path) -> Path:
    task_dir = tmp_path / "timeout-final-failure"
    workspace = task_dir / "public" / "workspace"
    private = task_dir / "private"
    workspace.mkdir(parents=True)
    private.mkdir()
    (workspace / "input.txt").write_text("trusted fixture", encoding="utf-8")
    (task_dir / "public" / "task.json").write_text(
        json.dumps(
            {
                "id": "timeout-final-failure",
                "goal": "Exercise the timeout observer contract.",
                "workspace_fixture": "public/workspace",
                "timeout_s": 1.0,
            }
        ),
        encoding="utf-8",
    )
    (private / "grader.json").write_text(
        json.dumps(
            {
                "criteria": [
                    {
                        "id": "must-not-run",
                        "kind": "file_exists",
                        "path": "never-created.txt",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return task_dir


# 功能：验证真实 producer 的 final failure barrier 与后续 open step 是合法 timeout prefix
# 设计：贯通 AgentLoop、invoke_tool、EventBus和真实journal writer，在第二次provider调用处取消形成无terminal前缀
@pytest.mark.asyncio
async def test_real_producer_final_failure_prefix_passes_timeout_grader(
    tmp_path: Path,
) -> None:
    run_id = "producer-final-failure"
    run_path = tmp_path / run_id
    bus = EventBus()
    journal = EventJournalCoordinator()
    bus.subscribe(journal.handle)
    await journal.register_run(run_id, run_path, session_id=None)
    await bus.publish(
        RunStartedEvent(
            run_id=run_id,
            goal="Exercise final failure lifecycle.",
            ts=_now(),
        )
    )
    provider = _FailureThenBlockProvider()
    registry = ToolRegistry()
    registry.register(_FinalFailureTool())
    context = ExecutionContext(
        run_id=run_id,
        goal="Exercise final failure lifecycle.",
        max_steps=5,
    )
    loop = AgentLoop(provider, DirectToolInvoker(registry, bus, run_id), bus)
    loop_task = asyncio.create_task(loop.run(context))

    await asyncio.wait_for(provider.second_call_started.wait(), timeout=2.0)
    loop_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await loop_task
    await journal.flush_all()
    await journal.close()

    grade = grade_timeout_trace_prefix(
        run_path / "events.v2.jsonl",
        expected_run_id=run_id,
    )

    assert provider.calls == 2
    assert grade.passed is True
    assert grade.errors == []


# 功能：验证 parent timeout保存的真实 final-failure prefix可继续通过 benchmark identity observer
# 设计：启动test-only worker子进程贯通cleanup、partial preservation和collect_observed_identity，不注入网络provider
@pytest.mark.asyncio
async def test_parent_timeout_preserves_valid_final_failure_identity_prefix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    helper = (
        Path(__file__).parent
        / "support"
        / "final_failure_timeout_worker.py"
    ).resolve()
    monkeypatch.setattr(
        eval_runner,
        "_worker_argv",
        lambda: [sys.executable, str(helper)],
    )
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    output = tmp_path / "evaluation-output"

    report = await evaluate_task(_timeout_task_dir(tmp_path), output)
    attempt_root = (
        output
        / "attempts"
        / report.task_id
        / report.attempt_id
    )
    observed = collect_observed_identity(
        attempt_root,
        evidence_mode=EvidenceMode.TIMEOUT_PARTIAL,
    )

    assert report.failure_category is FailureCategory.TIMEOUT
    assert report.runtime_success is False
    assert report.task_success is False
    assert report.trace_sanity_passed is False
    assert report.criteria == []
    assert report.metrics.step_count == 0
    assert report.metrics.tool_count == 0
    assert report.metrics.retry_count == 0
    assert report.metrics.token_usage.input_tokens == 0
    assert report.metrics.token_usage.output_tokens == 0
    assert report.metrics.token_usage.cache_tokens == 0
    assert (attempt_root / "runtime" / "events.v2.jsonl").is_file()
    assert (attempt_root / "runtime" / "trace.jsonl").is_file()
    assert (attempt_root / "_work" / "workspace" / "step-2-ready").is_file()
    assert not (attempt_root / "_work" / "worker-result.json").exists()
    assert not (attempt_root / "runtime" / "initial-workspace.json").exists()
    assert not (attempt_root / "runtime" / "final-workspace.json").exists()
    assert not (attempt_root / "runtime" / "workspace.diff").exists()
    assert not (attempt_root / "private" / "grades.json").exists()
    assert not (attempt_root / "private" / "command-results.json").exists()
    assert observed.api_call_count == 2
    assert len(observed.model_event_ids) == 2
