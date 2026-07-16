from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel

from kama_claude.core.bus.events import (
    SubagentFinishedEvent,
    SubagentStartedEvent,
    ToolCallFailedEvent,
    ToolCallFinishedEvent,
    ToolCallStartedEvent,
)
from kama_claude.core.context import ExecutionContext
from kama_claude.core.events.bus import EventBus
from kama_claude.core.llm.types import ToolCallBlock
from kama_claude.core.loop import AgentLoop
from kama_claude.core.subagent.registry import BackgroundTaskRegistry
from kama_claude.core.subagent.tool import AgentResultTool, SpawnAgentTool
from kama_claude.core.tools.base import BaseTool, ToolResult
from kama_claude.core.tools.invocation import invoke_tool
from kama_claude.core.tools.registry import ToolRegistry


# 构造绑定真实 EventBus、ExecutionContext 路径和后台 registry 的 spawn 工具
def _make_tool(
    tmp_path: Path,
    *,
    provider: Any | None = None,
    depth: int = 0,
) -> tuple[SpawnAgentTool, BackgroundTaskRegistry, EventBus, Any]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    resolved_provider = provider or MagicMock()
    if not hasattr(resolved_provider, "chat"):
        resolved_provider.chat = AsyncMock()
    bus = EventBus()
    registry = BackgroundTaskRegistry()
    tool = SpawnAgentTool(
        provider=resolved_provider,
        workspace_root=tmp_path,
        parent_bus=bus,
        parent_run_id="parent-run",
        permission_manager=None,
        max_steps=5,
        task_registry=registry,
        runs_dir=tmp_path,
        session_id="session-1",
        depth=depth,
    )
    return tool, registry, bus, resolved_provider


# 订阅真实 EventBus 并返回按发布时间排序的事件列表
def _collect(bus: EventBus) -> list[BaseModel]:
    events: list[BaseModel] = []

    # 收集同一对象引用以保留精确的事件类型与字段
    async def _append(event: BaseModel) -> None:
        events.append(event)

    bus.subscribe(_append)
    return events


# 通过真实 ToolRegistry 和 invoke_tool 执行 producer 并返回结果与事件
async def _invoke(
    tool: BaseTool,
    params: dict[str, object],
    bus: EventBus,
    *,
    tool_use_id: str,
) -> tuple[ToolResult, list[BaseModel]]:
    registry = ToolRegistry()
    registry.register(tool)
    events = _collect(bus)
    result = await invoke_tool(
        registry,
        ToolCallBlock(id=tool_use_id, name=tool.name, input=params),
        bus,
        run_id="parent-run",
    )
    return result, events


# 从 spawn 的稳定返回文案中提取后台 run_id
def _run_id(result: ToolResult) -> str:
    return result.content.split("run_id=")[1].split(".")[0]


# 功能：验证 nesting limit 使用 invalid_input，且 direct/central 都不会调用 provider
# 设计：同一真实 SpawnAgentTool 先 direct 再 invoke_tool，锁定永久错误 attempt=1 和零子事件
async def test_nesting_limit_is_non_retryable_invalid_input(tmp_path: Path) -> None:
    tool, _, bus, provider = _make_tool(tmp_path, depth=2)
    direct = await tool.invoke({"description": "nested", "prompt": "work"})

    result, events = await _invoke(
        tool,
        {"description": "nested", "prompt": "work"},
        bus,
        tool_use_id="nesting",
    )

    assert direct.error_type == "invalid_input"
    assert result.error_type == "invalid_input"
    provider.chat.assert_not_called()
    assert len([event for event in events if isinstance(event, ToolCallStartedEvent)]) == 1
    failed = [event for event in events if isinstance(event, ToolCallFailedEvent)]
    assert [event.attempt for event in failed] == [1]
    assert not any(isinstance(event, ToolCallFinishedEvent) for event in events)
    assert not any(
        isinstance(event, (SubagentStartedEvent, SubagentFinishedEvent))
        for event in events
    )


# 功能：验证 unknown background run_id 使用 not_found 且不重试
# 设计：空真实 registry 同时走 direct 与 invoke_tool，断言 attempt=1 且消息不含内部路径
async def test_unknown_run_id_is_non_retryable_not_found(tmp_path: Path) -> None:
    registry = BackgroundTaskRegistry()
    tool = AgentResultTool(registry)
    direct = await tool.invoke({"run_id": "missing-run"})
    bus = EventBus()

    result, events = await _invoke(
        tool,
        {"run_id": "missing-run"},
        bus,
        tool_use_id="missing-result",
    )

    assert direct.error_type == "not_found"
    assert result.error_type == "not_found"
    assert "missing-run" in result.content
    assert "/" not in result.content
    failed = [event for event in events if isinstance(event, ToolCallFailedEvent)]
    assert [event.attempt for event in failed] == [1]
    assert not any(isinstance(event, ToolCallFinishedEvent) for event in events)


@pytest.mark.parametrize(
    ("child_result", "expected_content"),
    [
        ("partial delegated result", "partial delegated result"),
        ("", "Subagent failed to complete the delegated task."),
    ],
)
# 功能：验证 foreground normal failed 映射 command_failed 且不泄漏 context.reason
# 设计：只替换 AgentLoop.run 的业务结局，保留真实 SpawnAgentTool、EventWriter、bus 与 context
async def test_foreground_normal_failure_is_safe_command_failed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    child_result: str,
    expected_content: str,
) -> None:
    secret = "/private/workspace/.env token=reason-secret"

    # 模拟子 loop 正常返回 failed context，而不是抛 Python 异常
    async def _normal_failure(
        self: AgentLoop,
        context: ExecutionContext,
    ) -> None:
        context.result = child_result
        context.mark_failed(secret)

    monkeypatch.setattr(AgentLoop, "run", _normal_failure)
    tool, _, bus, _ = _make_tool(tmp_path)
    events = _collect(bus)

    result = await tool.invoke({"description": "child", "prompt": "work"})

    assert result == ToolResult(
        content=expected_content,
        is_error=True,
        error_type="command_failed",
    )
    assert secret not in result.content
    assert len([event for event in events if isinstance(event, SubagentStartedEvent)]) == 1
    finished = [event for event in events if isinstance(event, SubagentFinishedEvent)]
    assert [event.status for event in finished] == ["failed"]


# 功能：验证 foreground normal failed 经 invoke_tool 只执行一次且发布单一 failed attempt
# 设计：用 loop 调用计数配合真实 SpawnAgentTool 与 EventBus，排除 command_failed 被中央重试
async def test_foreground_normal_failure_runs_once_through_invocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    # 记录 child loop 实际执行次数并返回正常 failed context
    async def _normal_failure(
        self: AgentLoop,
        context: ExecutionContext,
    ) -> None:
        nonlocal calls
        calls += 1
        context.mark_failed("private diagnostic reason")

    monkeypatch.setattr(AgentLoop, "run", _normal_failure)
    tool, _, bus, _ = _make_tool(tmp_path)

    result, events = await _invoke(
        tool,
        {"description": "child", "prompt": "work"},
        bus,
        tool_use_id="normal-failure",
    )

    assert calls == 1
    assert result == ToolResult(
        content="Subagent failed to complete the delegated task.",
        is_error=True,
        error_type="command_failed",
    )
    failed = [event for event in events if isinstance(event, ToolCallFailedEvent)]
    assert [event.attempt for event in failed] == [1]
    finished = [event for event in events if isinstance(event, SubagentFinishedEvent)]
    assert [event.status for event in finished] == ["failed"]


# 功能：验证 foreground RuntimeError 配对 finished 后原样传播并由中央安全分类一次
# 设计：两个固定异常分别覆盖 direct/invoke_tool，联合检查身份、secret 净化和 attempt=1
async def test_foreground_unknown_exception_pairs_and_propagates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "/private/workspace/.env token=foreground-secret"
    direct_error = RuntimeError(secret)
    invoked_error = RuntimeError(secret)
    errors = iter([direct_error, invoked_error])

    # 每次 loop 执行抛出对应的固定异常对象
    async def _raise_unknown(
        self: AgentLoop,
        context: ExecutionContext,
    ) -> None:
        raise next(errors)

    monkeypatch.setattr(AgentLoop, "run", _raise_unknown)
    direct_tool, _, direct_bus, _ = _make_tool(tmp_path / "direct")
    direct_events = _collect(direct_bus)
    with pytest.raises(RuntimeError) as exc_info:
        await direct_tool.invoke({"description": "child", "prompt": "work"})

    assert exc_info.value is direct_error
    direct_finished = [
        event for event in direct_events if isinstance(event, SubagentFinishedEvent)
    ]
    assert [event.status for event in direct_finished] == ["failed"]

    invoked_tool, _, invoked_bus, _ = _make_tool(tmp_path / "invoked")
    result, events = await _invoke(
        invoked_tool,
        {"description": "child", "prompt": "work"},
        invoked_bus,
        tool_use_id="foreground-error",
    )

    assert result == ToolResult(
        content="tool execution failed",
        is_error=True,
        error_type="execution_error",
    )
    assert secret not in result.content
    finished = [event for event in events if isinstance(event, SubagentFinishedEvent)]
    assert [event.status for event in finished] == ["failed"]
    failed = [event for event in events if isinstance(event, ToolCallFailedEvent)]
    assert [event.attempt for event in failed] == [1]
    assert all(secret not in event.error_message for event in failed)
    assert not any(isinstance(event, ToolCallFinishedEvent) for event in events)


# 功能：验证 foreground CancelledError 配对 failed finished 后原样传播
# 设计：通过真实 invoke_tool 捕获取消对象，断言无 ToolResult、无 tool failed 且 finished exactly once
async def test_foreground_cancellation_pairs_and_propagates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cancellation = asyncio.CancelledError()

    # 在 child loop await 边界抛出固定 cancellation
    async def _cancel(
        self: AgentLoop,
        context: ExecutionContext,
    ) -> None:
        raise cancellation

    monkeypatch.setattr(AgentLoop, "run", _cancel)
    tool, _, bus, _ = _make_tool(tmp_path)
    registry = ToolRegistry()
    registry.register(tool)
    events = _collect(bus)

    with pytest.raises(asyncio.CancelledError) as exc_info:
        await invoke_tool(
            registry,
            ToolCallBlock(
                id="foreground-cancel",
                name="spawn_agent",
                input={"description": "child", "prompt": "work"},
            ),
            bus,
            run_id="parent-run",
        )

    assert exc_info.value is cancellation
    assert len([event for event in events if isinstance(event, ToolCallStartedEvent)]) == 1
    assert not any(isinstance(event, ToolCallFailedEvent) for event in events)
    assert not any(isinstance(event, ToolCallFinishedEvent) for event in events)
    finished = [event for event in events if isinstance(event, SubagentFinishedEvent)]
    assert [event.status for event in finished] == ["failed"]


# 功能：验证 background 从 pending 到 success，并且重复查询不重复 finished
# 设计：asyncio.Event 驱动真实 task 状态转换，完成前后查询并锁定稳定结果与单一事件配对
async def test_background_pending_success_and_repeated_result_are_stable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entered = asyncio.Event()
    release = asyncio.Event()

    # 阻塞真实 background task，释放后写入成功结果
    async def _success(
        self: AgentLoop,
        context: ExecutionContext,
    ) -> None:
        entered.set()
        await release.wait()
        context.result = "background answer"
        context.mark_success()

    monkeypatch.setattr(AgentLoop, "run", _success)
    tool, registry, bus, _ = _make_tool(tmp_path)
    events = _collect(bus)
    spawn_result = await tool.invoke(
        {"description": "child", "prompt": "work", "run_in_background": True}
    )
    run_id = _run_id(spawn_result)
    entry = registry.get(run_id)
    assert entry is not None
    task, context = entry
    await entered.wait()

    pending = await AgentResultTool(registry).invoke({"run_id": run_id})
    assert pending == ToolResult(content="still running")

    release.set()
    await task
    first = await AgentResultTool(registry).invoke({"run_id": run_id})
    second = await AgentResultTool(registry).invoke({"run_id": run_id})

    assert task.done() and not task.cancelled()
    assert context.status == "success"
    assert first == second == ToolResult(content="background answer")
    assert len([event for event in events if isinstance(event, SubagentStartedEvent)]) == 1
    finished = [event for event in events if isinstance(event, SubagentFinishedEvent)]
    assert [event.status for event in finished] == ["success"]


# 功能：验证 background normal failed 不再被 agent_result 误报为成功
# 设计：Event 释放真实 task 后检查 failed context、command_failed、安全 result 和单一 finished
async def test_background_normal_failure_is_safe_command_failed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entered = asyncio.Event()
    release = asyncio.Event()
    secret = "/private/workspace/.env token=background-reason"

    # 让真实 background task 正常结束但把 context 标记为失败
    async def _normal_failure(
        self: AgentLoop,
        context: ExecutionContext,
    ) -> None:
        entered.set()
        await release.wait()
        context.result = "partial background result"
        context.mark_failed(secret)

    monkeypatch.setattr(AgentLoop, "run", _normal_failure)
    tool, registry, bus, _ = _make_tool(tmp_path)
    events = _collect(bus)
    spawn_result = await tool.invoke(
        {"description": "child", "prompt": "work", "run_in_background": True}
    )
    run_id = _run_id(spawn_result)
    entry = registry.get(run_id)
    assert entry is not None
    task, context = entry
    await entered.wait()
    release.set()
    await task

    result_tool = AgentResultTool(registry)
    first = await result_tool.invoke({"run_id": run_id})
    second = await result_tool.invoke({"run_id": run_id})

    assert context.status == "failed"
    assert first == second == ToolResult(
        content="partial background result",
        is_error=True,
        error_type="command_failed",
    )
    assert secret not in first.content
    finished = [event for event in events if isinstance(event, SubagentFinishedEvent)]
    assert [event.status for event in finished] == ["failed"]

    invoked, result_events = await _invoke(
        result_tool,
        {"run_id": run_id},
        EventBus(),
        tool_use_id="failed-result",
    )
    assert invoked == first
    failed = [event for event in result_events if isinstance(event, ToolCallFailedEvent)]
    assert [event.attempt for event in failed] == [1]


# 功能：验证 background task 真实取消后配对 finished 并安全映射 command_failed
# 设计：Event 确认 task 正在运行后调用 cancel，等待 cancelled 状态，再经 invoke_tool 锁定 attempt=1
async def test_background_cancellation_is_safe_command_failed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entered = asyncio.Event()
    never_release = asyncio.Event()

    # 保持 child loop 挂起，直到测试取消注册表中的真实 task
    async def _wait_forever(
        self: AgentLoop,
        context: ExecutionContext,
    ) -> None:
        entered.set()
        await never_release.wait()

    monkeypatch.setattr(AgentLoop, "run", _wait_forever)
    tool, registry, bus, _ = _make_tool(tmp_path)
    events = _collect(bus)
    spawn_result = await tool.invoke(
        {"description": "child", "prompt": "work", "run_in_background": True}
    )
    run_id = _run_id(spawn_result)
    entry = registry.get(run_id)
    assert entry is not None
    task, context = entry
    await entered.wait()

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert task.cancelled()
    assert context.status == "failed"
    assert context.reason == "cancelled"
    finished = [event for event in events if isinstance(event, SubagentFinishedEvent)]
    assert [event.status for event in finished] == ["failed"]

    result, result_events = await _invoke(
        AgentResultTool(registry),
        {"run_id": run_id},
        EventBus(),
        tool_use_id="cancelled-result",
    )
    assert result == ToolResult(
        content="Subagent was cancelled.",
        is_error=True,
        error_type="command_failed",
    )
    failed = [event for event in result_events if isinstance(event, ToolCallFailedEvent)]
    assert [event.attempt for event in failed] == [1]


# 功能：验证 background RuntimeError 保留 task exception 并由 agent_result 安全映射
# 设计：Event 驱动真实 task 抛固定异常，检查 traceback、finished 配对、secret 净化和中央 attempt=1
async def test_background_unknown_exception_is_safe_execution_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    entered = asyncio.Event()
    release = asyncio.Event()
    secret = "/private/workspace/.env token=background-exception"
    error = RuntimeError(secret)

    # 在 release 后让真实 background task 保存固定 RuntimeError
    async def _raise_unknown(
        self: AgentLoop,
        context: ExecutionContext,
    ) -> None:
        entered.set()
        await release.wait()
        raise error

    monkeypatch.setattr(AgentLoop, "run", _raise_unknown)
    tool, registry, bus, _ = _make_tool(tmp_path)
    events = _collect(bus)
    spawn_result = await tool.invoke(
        {"description": "child", "prompt": "work", "run_in_background": True}
    )
    run_id = _run_id(spawn_result)
    entry = registry.get(run_id)
    assert entry is not None
    task, context = entry
    await entered.wait()

    with caplog.at_level(logging.ERROR, logger="kama_claude.core.subagent.tool"):
        release.set()
        with pytest.raises(RuntimeError) as exc_info:
            await task

    assert exc_info.value is error
    assert task.exception() is error
    assert context.status == "failed"
    assert context.reason == "subagent_error"
    assert "Traceback (most recent call last)" in caplog.text
    finished = [event for event in events if isinstance(event, SubagentFinishedEvent)]
    assert [event.status for event in finished] == ["failed"]

    direct = await AgentResultTool(registry).invoke({"run_id": run_id})
    assert direct == ToolResult(
        content="Subagent execution failed.",
        is_error=True,
        error_type="execution_error",
    )
    assert secret not in direct.content

    result, result_events = await _invoke(
        AgentResultTool(registry),
        {"run_id": run_id},
        EventBus(),
        tool_use_id="exception-result",
    )
    assert result.error_type == "execution_error"
    assert result.content == "tool execution failed"
    assert secret not in result.content
    failed = [event for event in result_events if isinstance(event, ToolCallFailedEvent)]
    assert [event.attempt for event in failed] == [1]
    assert all(secret not in event.error_message for event in failed)


# 功能：验证 agent_result 查询本身被取消时原样传播且不伪造 failed event
# 设计：真实 AgentResultTool 的 registry lookup 注入固定取消对象，再经 invoke_tool 检查身份与唯一 started
async def test_agent_result_query_cancellation_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cancellation = asyncio.CancelledError()
    registry = BackgroundTaskRegistry()

    # 在同步 registry 边界抛出查询任务收到的固定取消信号
    def _cancel_lookup(run_id: str) -> None:
        raise cancellation

    monkeypatch.setattr(registry, "get", _cancel_lookup)
    tool = AgentResultTool(registry)
    tool_registry = ToolRegistry()
    tool_registry.register(tool)
    bus = EventBus()
    events = _collect(bus)

    with pytest.raises(asyncio.CancelledError) as exc_info:
        await invoke_tool(
            tool_registry,
            ToolCallBlock(
                id="query-cancel",
                name="agent_result",
                input={"run_id": "run-1"},
            ),
            bus,
            run_id="parent-run",
        )

    assert exc_info.value is cancellation
    assert [type(event) for event in events] == [ToolCallStartedEvent]
