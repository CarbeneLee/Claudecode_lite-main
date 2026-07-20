from __future__ import annotations

import os
import threading
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, cast

import pytest

import kama_claude.core.tools.builtin.search_code as search_code_module
from kama_claude.core.bus.events import (
    PermissionRequestedEvent,
    ToolCallFailedEvent,
    ToolCallFinishedEvent,
    ToolCallStartedEvent,
)
from kama_claude.core.events.bus import EventBus
from kama_claude.core.permissions.manager import PermissionManager
from kama_claude.core.permissions.policy import PermissionDecision, param_preview
from kama_claude.core.tools.base import ToolResult
from kama_claude.core.tools.builtin.search_code import SearchCodeParams
from kama_claude.core.tools.invocation import invoke_tool
from kama_claude.core.tools.registry import ToolRegistry
from tests.unit._search_code_test_support import _call, _collect_events, _tool


class _FailIfPermissionChecked:
    # 初始化权限边界调用记录
    def __init__(self) -> None:
        self.checked = False

    # 如果 schema 无效输入越过验证边界则立即失败
    async def check_and_wait(
        self,
        tool_use_id: str,
        tool_name: str,
        params: dict[str, Any],
        session_id: str,
        event_emitter: Callable[[dict[str, Any]], Awaitable[None]],
    ) -> tuple[bool, str]:
        self.checked = True
        raise AssertionError("schema-invalid params reached permission preview")


# 功能：验证 started event 保留调用者提交的原始参数值
# 设计：经真实 invoke_tool 成功路径比较 dict 值而非对象 identity，锁定 trace/replay 事实
async def test_started_event_preserves_original_input_values(tmp_path: Path) -> None:
    (tmp_path / "sample.py").write_text("needle\n", encoding="utf-8")
    original_input: dict[str, object] = {
        "query": "needle",
        "path": ".",
        "include_glob": "*.py",
        "case_sensitive": True,
        "max_results": 7,
    }
    registry = ToolRegistry()
    registry.register(_tool(tmp_path))
    bus = EventBus()
    events = _collect_events(bus)

    result = await invoke_tool(registry, _call(original_input), bus, run_id="run-1")

    assert not result.is_error
    started = cast(ToolCallStartedEvent, events[0])
    assert isinstance(started, ToolCallStartedEvent)
    assert started.params == original_input
    assert [type(event) for event in events] == [
        ToolCallStartedEvent,
        ToolCallFinishedEvent,
    ]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("query", "needle\x1bsecret"),
        ("query", "needle\u2028secret"),
        ("query", "needle\u2029secret"),
        ("path", "src\x1bsecret"),
        ("include_glob", "*.py\u2028secret"),
    ],
)
# 功能：验证含冻结控制字符的参数只进入原始 started 事件后就以 schema_error 终止
# 设计：参数化 ESC 和 Unicode 行分隔符，用会自动失败的权限 stub 证明 preview 与 invoke 都不可达
async def test_invalid_controls_stop_before_permission_and_invoke(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: str,
) -> None:
    original_input: dict[str, object] = {"query": "needle", "path": "."}
    original_input[field] = value
    tool = _tool(tmp_path)
    invoked = False

    # 记录同步 worker 是否被错误启动
    def _unexpected_search(
        params: SearchCodeParams,
        stop_event: threading.Event,
    ) -> ToolResult:
        nonlocal invoked
        invoked = True
        raise AssertionError("schema-invalid params reached search worker")

    monkeypatch.setattr(tool, "_search_sync", _unexpected_search)
    registry = ToolRegistry()
    registry.register(tool)
    manager = _FailIfPermissionChecked()
    bus = EventBus()
    events = _collect_events(bus)

    result = await invoke_tool(
        registry,
        _call(original_input),
        bus,
        run_id="run-invalid",
        permission_manager=cast(Any, manager),
        session_id="session-1",
    )

    assert result.is_error
    assert result.error_type == "schema_error"
    assert manager.checked is False
    assert invoked is False
    assert [type(event) for event in events] == [
        ToolCallStartedEvent,
        ToolCallFailedEvent,
    ]
    started = cast(ToolCallStartedEvent, events[0])
    assert started.params == original_input
    failed = cast(ToolCallFailedEvent, events[1])
    assert failed.error_class == "schema_error"
    assert failed.attempt == 1
    assert not any(isinstance(event, PermissionRequestedEvent) for event in events)


# 功能：验证 search_code 使用默认 ALLOW 权限并以 query 生成 preview
# 设计：真实 PermissionManager 走 auto_allow 成功路径，同时锁定 preview 不暴露搜索结果
async def test_default_permission_allows_search_and_preview_uses_query(
    tmp_path: Path,
) -> None:
    (tmp_path / "sample.py").write_text("needle\n", encoding="utf-8")
    params: dict[str, object] = {"query": "needle", "path": "src"}
    manager = PermissionManager()

    assert manager.evaluate("search_code", params) == PermissionDecision.ALLOW
    assert param_preview("search_code", params) == "query='needle'"

    registry = ToolRegistry()
    registry.register(_tool(tmp_path))
    bus = EventBus()
    events = _collect_events(bus)
    result = await invoke_tool(
        registry,
        _call({"query": "needle", "path": "."}),
        bus,
        run_id="run-permission",
        permission_manager=manager,
        session_id="session-1",
    )

    assert not result.is_error
    assert [type(event) for event in events] == [
        ToolCallStartedEvent,
        ToolCallFinishedEvent,
    ]


# 功能：验证 worker 未知异常经中央边界安全映射且不重试
# 设计：同步 seam 抛出含 secret 的 RuntimeError，联合检查 ToolResult、failed event 和 attempt
async def test_worker_exception_maps_to_safe_single_attempt_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool = _tool(tmp_path)

    # 模拟文件系统 worker 中的未知异常
    def _raise_secret(
        params: SearchCodeParams,
        stop_event: threading.Event,
    ) -> ToolResult:
        raise RuntimeError("/private/workspace/.env token=search-secret")

    monkeypatch.setattr(tool, "_search_sync", _raise_secret)
    registry = ToolRegistry()
    registry.register(tool)
    bus = EventBus()
    events = _collect_events(bus)

    result = await invoke_tool(
        registry,
        _call({"query": "needle"}),
        bus,
        run_id="run-error",
    )

    assert result == ToolResult(
        content="tool execution failed",
        is_error=True,
        error_type="execution_error",
    )
    assert "search-secret" not in result.content
    assert [type(event) for event in events] == [
        ToolCallStartedEvent,
        ToolCallFailedEvent,
    ]
    failed = cast(ToolCallFailedEvent, events[1])
    assert failed.error_class == "execution_error"
    assert failed.error_message == "tool execution failed"
    assert failed.attempt == 1
    assert "search-secret" not in failed.error_message


@pytest.mark.parametrize("root_kind", ["file", "directory"])
# 功能：验证显式不可读 root 经 invoke_tool 映射 permission_error 且只尝试一次
# 设计：真实 registry/EventBus 配合 fd seam 注入 PermissionError，联合锁定 result 与 failed event
async def test_explicit_unreadable_root_maps_permission_error_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    root_kind: str,
) -> None:
    target = tmp_path / "target"
    if root_kind == "directory":
        target.mkdir()
    else:
        target.write_text("needle", encoding="utf-8")
    tool = _tool(tmp_path)
    real_open = tool._open_workspace_fd

    # 只让显式 root 的安全 fd open 抛含敏感详情的权限错误
    def deny_root(path: Path, *, directory: bool) -> int:
        if path == target.resolve(strict=True):
            raise PermissionError("private-root-detail")
        return real_open(path, directory=directory)

    monkeypatch.setattr(tool, "_open_workspace_fd", deny_root)
    registry = ToolRegistry()
    registry.register(tool)
    bus = EventBus()
    events = _collect_events(bus)

    result = await invoke_tool(
        registry,
        _call({"query": "needle", "path": "target"}),
        bus,
        run_id="run-unreadable-root",
    )

    assert result == ToolResult(
        content="tool lacks permission for this operation",
        is_error=True,
        error_type="permission_error",
    )
    assert [type(event) for event in events] == [
        ToolCallStartedEvent,
        ToolCallFailedEvent,
    ]
    failed = cast(ToolCallFailedEvent, events[1])
    assert failed.error_class == "permission_error"
    assert failed.attempt == 1
    assert "private-root-detail" not in failed.error_message


# 功能：验证显式 FIFO root 经 invoke_tool 返回 invalid_input 且不尝试 open
# 设计：真实 FIFO 加 fail-fast fd helper，证明中央边界只发布一次 failed 而不会阻塞或 finished
async def test_explicit_fifo_root_maps_invalid_input_without_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fifo = tmp_path / "pipe"
    os.mkfifo(fifo)
    tool = _tool(tmp_path)
    opened = False

    # FIFO 不得进入任何 fd open 路径
    def unexpected_open(_path: Path, *, directory: bool) -> int:
        nonlocal opened
        opened = True
        raise AssertionError(f"FIFO reached fd open: {directory}")

    monkeypatch.setattr(tool, "_open_workspace_fd", unexpected_open)
    registry = ToolRegistry()
    registry.register(tool)
    bus = EventBus()
    events = _collect_events(bus)

    result = await invoke_tool(
        registry,
        _call({"query": "needle", "path": "pipe"}),
        bus,
        run_id="run-fifo",
    )

    assert opened is False
    assert result == ToolResult(
        content="search path must be a regular file or directory",
        is_error=True,
        error_type="invalid_input",
    )
    assert [type(event) for event in events] == [
        ToolCallStartedEvent,
        ToolCallFailedEvent,
    ]
    assert cast(ToolCallFailedEvent, events[1]).attempt == 1


# 功能：验证缺失 POSIX capability 经 invoke_tool 安全映射 execution_error 且不重试
# 设计：移除 os.open dir_fd capability，断言固定 direct detail 不进入 result/failed event
async def test_missing_posix_capability_maps_safe_execution_error_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "hit.txt").write_text("needle", encoding="utf-8")
    monkeypatch.setattr(search_code_module.os, "supports_dir_fd", frozenset())
    registry = ToolRegistry()
    registry.register(_tool(tmp_path))
    bus = EventBus()
    events = _collect_events(bus)

    result = await invoke_tool(
        registry,
        _call({"query": "needle"}),
        bus,
        run_id="run-platform",
    )

    assert result == ToolResult(
        content="tool execution failed",
        is_error=True,
        error_type="execution_error",
    )
    assert [type(event) for event in events] == [
        ToolCallStartedEvent,
        ToolCallFailedEvent,
    ]
    failed = cast(ToolCallFailedEvent, events[1])
    assert failed.error_class == "execution_error"
    assert failed.attempt == 1
    assert "POSIX" not in result.content
    assert "POSIX" not in failed.error_message
