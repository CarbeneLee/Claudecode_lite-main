from __future__ import annotations

import asyncio
import logging
from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel

from kama_claude.core.bus.events import (
    ToolCallFailedEvent,
    ToolCallFinishedEvent,
    ToolCallStartedEvent,
)
from kama_claude.core.events.bus import EventBus
from kama_claude.core.llm.types import ToolCallBlock
from kama_claude.core.mcp.client import (
    McpClient,
    McpServerUnavailableError,
    McpToolDef,
    McpToolError,
)
from kama_claude.core.mcp.tool import McpTool
from kama_claude.core.tools.base import ToolResult
from kama_claude.core.tools.invocation import invoke_tool
from kama_claude.core.tools.registry import ToolRegistry


# 构造保留真实 MCP wrapper 元数据、仅替换远端 I/O 的确定性工具
def _make_tool(side_effect: BaseException | None = None) -> tuple[McpTool, AsyncMock]:
    client = AsyncMock(spec=McpClient)
    client.call_tool = AsyncMock(side_effect=side_effect)
    tool = McpTool(
        client,
        "remote",
        McpToolDef(
            name="mutate_record",
            description="Mutate a remote record",
            input_schema={
                "type": "object",
                "properties": {"record_id": {"type": "integer"}},
                "required": ["record_id"],
            },
        ),
    )
    return tool, client


# 通过真实 invoke_tool 收集 MCP producer 的结果和生命周期事件
async def _invoke(tool: McpTool) -> tuple[ToolResult, list[BaseModel]]:
    registry = ToolRegistry()
    registry.register(tool)
    bus = EventBus()
    events: list[BaseModel] = []

    # 按发布顺序保留真实事件对象
    async def _collect(event: BaseModel) -> None:
        events.append(event)

    bus.subscribe(_collect)
    result = await invoke_tool(
        registry,
        ToolCallBlock(
            id="mcp-call",
            name="remote__mutate_record",
            input={"record_id": 7},
        ),
        bus,
        run_id="run-1",
    )
    return result, events


# 功能：验证 McpToolError direct/central 都映射固定安全 command_failed 且各调用一次
# 设计：两个独立真实 wrapper 隔离调用计数，异常携带 secret 以同时验证结果和 failed event 净化
async def test_mcp_tool_error_is_safe_non_retryable_command_failed() -> None:
    secret = "/private/mcp payload token=tool-secret"
    direct_tool, direct_client = _make_tool(McpToolError(secret))

    direct = await direct_tool.invoke({"record_id": 7})

    assert direct == ToolResult(
        content="MCP tool reported an error.",
        is_error=True,
        error_type="command_failed",
    )
    assert secret not in direct.content
    direct_client.call_tool.assert_awaited_once_with("mutate_record", {"record_id": 7})

    invoked_tool, invoked_client = _make_tool(McpToolError(secret))
    result, events = await _invoke(invoked_tool)

    assert result == direct
    invoked_client.call_tool.assert_awaited_once_with("mutate_record", {"record_id": 7})
    assert len([event for event in events if isinstance(event, ToolCallStartedEvent)]) == 1
    failed = [event for event in events if isinstance(event, ToolCallFailedEvent)]
    assert [event.error_class for event in failed] == ["command_failed"]
    assert [event.attempt for event in failed] == [1]
    assert all(secret not in event.error_message for event in failed)
    assert not any(isinstance(event, ToolCallFinishedEvent) for event in events)


# 功能：验证 unavailable 使用安全 execution_error 且绝不因不确定副作用自动重放
# 设计：direct 与 central 使用独立 client，精确断言 call_count=1、attempt=1 和 secret 不泄漏
async def test_mcp_unavailable_is_safe_non_retryable_execution_error() -> None:
    secret = "/private/socket token=unavailable-secret"
    direct_tool, direct_client = _make_tool(McpServerUnavailableError(secret))

    direct = await direct_tool.invoke({"record_id": 7})

    assert direct == ToolResult(
        content="MCP server is unavailable.",
        is_error=True,
        error_type="execution_error",
    )
    assert secret not in direct.content
    direct_client.call_tool.assert_awaited_once_with("mutate_record", {"record_id": 7})

    invoked_tool, invoked_client = _make_tool(McpServerUnavailableError(secret))
    result, events = await _invoke(invoked_tool)

    assert result == ToolResult(
        content="tool execution failed",
        is_error=True,
        error_type="execution_error",
    )
    invoked_client.call_tool.assert_awaited_once_with("mutate_record", {"record_id": 7})
    failed = [event for event in events if isinstance(event, ToolCallFailedEvent)]
    assert [event.error_class for event in failed] == ["execution_error"]
    assert [event.attempt for event in failed] == [1]
    assert all(secret not in event.error_message for event in failed)
    assert not any(isinstance(event, ToolCallFinishedEvent) for event in events)


# 功能：验证 generic RuntimeError direct 原对象上抛、central 安全分类且远端只调用一次
# 设计：分别保存两个异常对象，中央路径用 caplog 证明 traceback 只进入受控 classifier 日志
async def test_mcp_unknown_exception_propagates_to_central_classifier(
    caplog: pytest.LogCaptureFixture,
) -> None:
    secret = "/private/mcp token=runtime-secret"
    direct_error = RuntimeError(secret)
    direct_tool, direct_client = _make_tool(direct_error)

    with pytest.raises(RuntimeError) as exc_info:
        await direct_tool.invoke({"record_id": 7})

    assert exc_info.value is direct_error
    direct_client.call_tool.assert_awaited_once_with("mutate_record", {"record_id": 7})

    invoked_error = RuntimeError(secret)
    invoked_tool, invoked_client = _make_tool(invoked_error)
    with caplog.at_level(logging.ERROR, logger="kama_claude.core.tools.errors"):
        result, events = await _invoke(invoked_tool)

    assert result == ToolResult(
        content="tool execution failed",
        is_error=True,
        error_type="execution_error",
    )
    invoked_client.call_tool.assert_awaited_once_with("mutate_record", {"record_id": 7})
    failed = [event for event in events if isinstance(event, ToolCallFailedEvent)]
    assert [event.attempt for event in failed] == [1]
    assert all(secret not in event.error_message for event in failed)
    assert not any(isinstance(event, ToolCallFinishedEvent) for event in events)
    assert "Traceback (most recent call last)" in caplog.text
    assert secret in caplog.text


# 功能：验证 MCP cancellation 在 direct 和 invoke_tool 都原样传播且不伪造失败结果
# 设计：两个固定 CancelledError 锁定对象身份，中央事件只能包含 started，client 各调用一次
async def test_mcp_cancellation_propagates_without_failed_event() -> None:
    direct_cancel = asyncio.CancelledError()
    direct_tool, direct_client = _make_tool(direct_cancel)

    with pytest.raises(asyncio.CancelledError) as direct_info:
        await direct_tool.invoke({"record_id": 7})

    assert direct_info.value is direct_cancel
    direct_client.call_tool.assert_awaited_once_with("mutate_record", {"record_id": 7})

    invoked_cancel = asyncio.CancelledError()
    invoked_tool, invoked_client = _make_tool(invoked_cancel)
    registry = ToolRegistry()
    registry.register(invoked_tool)
    bus = EventBus()
    events: list[BaseModel] = []

    # 收集 cancellation 发生前唯一允许发布的 started 事件
    async def _collect(event: BaseModel) -> None:
        events.append(event)

    bus.subscribe(_collect)
    with pytest.raises(asyncio.CancelledError) as invoked_info:
        await invoke_tool(
            registry,
            ToolCallBlock(
                id="mcp-cancel",
                name="remote__mutate_record",
                input={"record_id": 7},
            ),
            bus,
            run_id="run-1",
        )

    assert invoked_info.value is invoked_cancel
    invoked_client.call_tool.assert_awaited_once_with("mutate_record", {"record_id": 7})
    assert [type(event) for event in events] == [ToolCallStartedEvent]

