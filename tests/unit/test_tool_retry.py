from __future__ import annotations

import pytest
from pydantic import BaseModel

import kama_claude.core.tools.invocation as inv_mod
from kama_claude.core.bus.events import (
    ToolCallFailedEvent,
    ToolCallFinishedEvent,
    ToolCallStartedEvent,
)
from kama_claude.core.events.bus import EventBus
from kama_claude.core.llm.types import ToolCallBlock
from kama_claude.core.sandbox.errors import ContainerNotReadyError
from kama_claude.core.tools.base import BaseTool, ToolResult
from kama_claude.core.tools.errors import (
    RETRYABLE_ERROR_TYPES,
    RateLimitedError,
    TransientToolError,
)
from kama_claude.core.tools.invocation import invoke_tool
from kama_claude.core.tools.registry import ToolRegistry
from kama_claude.core.workspace.errors import (
    InvalidWorkspacePathError,
    SensitivePathError,
    WorkspaceEscapeError,
)


# 功能：验证只有显式瞬态错误类型属于可重试安全策略
# 设计：直接锁定 allowlist 契约，防止新增或删除错误类型静默改变重试边界
def test_only_explicit_transient_errors_are_retryable() -> None:
    assert RETRYABLE_ERROR_TYPES == frozenset(
        {
            "transient_error",
            "rate_limited",
            "container_not_ready",
            "git_unavailable",
            "git_lock",
            "checkpoint_failed",
            "commit_failed",
            "semantic_index_unavailable",
        }
    )


class _ResultErrorTool(BaseTool):
    name = "result_error"
    description = "Returns a configured ToolResult error"
    input_schema: dict[str, object] = {"type": "object", "properties": {}, "required": []}

    # 保存错误类型并记录真实 invoke 次数
    def __init__(
        self,
        error_type: str | None,
        content: str = "original tool payload",
    ) -> None:
        self._error_type = error_type
        self._content = content
        self.calls = 0

    # 返回配置的错误结果以测试归一化和重试判定
    async def invoke(self, params: dict[str, object]) -> ToolResult:
        self.calls += 1
        return ToolResult(
            content=self._content,
            is_error=True,
            error_type=self._error_type,
        )


class _TransientResultNTimes(BaseTool):
    name = "transient_n"
    description = "Returns a transient error n times then succeeds"
    input_schema: dict[str, object] = {"type": "object", "properties": {}, "required": []}

    # 保存失败次数并初始化调用计数
    def __init__(self, failures: int) -> None:
        self._failures = failures
        self.calls = 0

    # 在指定次数内返回显式瞬态错误，随后成功
    async def invoke(self, params: dict[str, object]) -> ToolResult:
        self.calls += 1
        if self.calls <= self._failures:
            return ToolResult(
                content="provider transient payload",
                is_error=True,
                error_type="transient_error",
            )
        return ToolResult(content="ok")


class _RateLimitedNTimes(BaseTool):
    name = "rate_n"
    description = "Raises a rate-limit error n times then succeeds"
    input_schema: dict[str, object] = {"type": "object", "properties": {}, "required": []}

    # 保存失败次数并初始化调用计数
    def __init__(self, failures: int) -> None:
        self._failures = failures
        self.calls = 0

    # 在指定次数内抛出限流异常，随后成功
    async def invoke(self, params: dict[str, object]) -> ToolResult:
        self.calls += 1
        if self.calls <= self._failures:
            raise RateLimitedError("429 vendor payload")
        return ToolResult(content="ok")


class _ExceptionTool(BaseTool):
    name = "exception"
    description = "Raises a configured exception"
    input_schema: dict[str, object] = {"type": "object", "properties": {}, "required": []}

    # 保存待抛异常并初始化调用计数
    def __init__(self, exc: Exception) -> None:
        self._exc = exc
        self.calls = 0

    # 记录调用后抛出配置的真实异常
    async def invoke(self, params: dict[str, object]) -> ToolResult:
        self.calls += 1
        raise self._exc


class _TransientExceptionNTimes(BaseTool):
    name = "transient_exception_n"
    description = "Raises TransientToolError n times then succeeds"
    input_schema: dict[str, object] = {"type": "object", "properties": {}, "required": []}

    # 保存瞬态失败次数并初始化调用计数
    def __init__(self, failures: int) -> None:
        self._failures = failures
        self.calls = 0

    # 在指定次数内抛出显式瞬态异常，随后成功
    async def invoke(self, params: dict[str, object]) -> ToolResult:
        self.calls += 1
        if self.calls <= self._failures:
            raise TransientToolError("provider transient secret")
        return ToolResult(content="ok")


class _ContainerNotReadyNTimes(BaseTool):
    name = "not_ready_n"
    description = "Raises ContainerNotReadyError n times then succeeds"
    input_schema: dict[str, object] = {"type": "object", "properties": {}, "required": []}

    # 保存容器未就绪失败次数并初始化调用计数
    def __init__(self, failures: int) -> None:
        self._failures = failures
        self.calls = 0

    # 在指定次数内抛出容器未就绪异常，随后成功
    async def invoke(self, params: dict[str, object]) -> ToolResult:
        self.calls += 1
        if self.calls <= self._failures:
            raise ContainerNotReadyError("sandbox container not ready")
        return ToolResult(content="ok")


# 构造固定标识的空参数工具调用
def _call(name: str) -> ToolCallBlock:
    return ToolCallBlock(id="t1", name=name, input={})


# 执行一次 logical invocation 并收集其完整事件序列
async def _run(
    tool: BaseTool,
    *,
    monkeypatch: pytest.MonkeyPatch,
    retry_base_s: float = 0.0,
) -> tuple[ToolResult, list[BaseModel]]:
    monkeypatch.setattr(inv_mod, "_RETRY_BASE_S", retry_base_s)
    registry = ToolRegistry()
    registry.register(tool)
    bus = EventBus()
    events: list[BaseModel] = []

    # 收集事件对象以断言 schema 与顺序
    async def _collect(event: BaseModel) -> None:
        events.append(event)

    bus.subscribe(_collect)
    result = await invoke_tool(registry, _call(tool.name), bus, run_id="r")
    return result, events


# 从事件列表筛选失败 attempt
def _failed(events: list[BaseModel]) -> list[ToolCallFailedEvent]:
    return [event for event in events if isinstance(event, ToolCallFailedEvent)]


@pytest.mark.parametrize(
    ("error_type", "expected_error_type"),
    [
        ("not_found", "not_found"),
        ("invalid_path", "invalid_path"),
        ("sensitive_path", "sensitive_path"),
        ("permission_error", "permission_error"),
        ("is_directory", "is_directory"),
        ("not_directory", "not_directory"),
        ("execution_error", "execution_error"),
        ("timeout", "timeout"),
        ("schema_error", "schema_error"),
        ("permission_denied", "permission_denied"),
        ("command_failed", "command_failed"),
        ("runtime_error", "execution_error"),
        (None, "execution_error"),
        ("vendor_specific_error", "execution_error"),
    ],
)
# 功能：验证永久、legacy、缺失与未知 ToolResult error_type 均只执行一次
# 设计：同一计数工具参数化覆盖所有非重试类型，并同时锁定 legacy/未知归一化和单个 failed attempt
async def test_non_retryable_tool_results_run_once(
    error_type: str | None,
    expected_error_type: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool = _ResultErrorTool(error_type)

    result, events = await _run(tool, monkeypatch=monkeypatch)

    assert tool.calls == 1
    assert result.is_error
    assert result.error_type == expected_error_type
    assert len([event for event in events if isinstance(event, ToolCallStartedEvent)]) == 1
    assert [event.attempt for event in _failed(events)] == [1]
    assert not any(isinstance(event, ToolCallFinishedEvent) for event in events)
    if expected_error_type == "execution_error":
        assert result.content == "tool execution failed"
        assert "original tool payload" not in result.content
    else:
        assert result.content == "original tool payload"


# 功能：验证显式 execution_error ToolResult 经 invoke_tool 后强制净化且不重试
# 设计：工具返回含路径和 token 的 content，断言最终结果与 failed event 都只保留固定摘要并记录一次调用
async def test_execution_error_tool_result_is_sanitized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "/private/.env token=secret"
    tool = _ResultErrorTool("execution_error", content=secret)

    result, events = await _run(tool, monkeypatch=monkeypatch)

    failed = _failed(events)
    assert tool.calls == 1
    assert result.is_error
    assert result.error_type == "execution_error"
    assert result.content == "tool execution failed"
    assert secret not in result.content
    assert [event.error_message for event in failed] == ["tool execution failed"]
    assert all(secret not in event.error_message for event in failed)
    assert [event.attempt for event in failed] == [1]


@pytest.mark.parametrize(
    ("exc", "expected_error_type"),
    [
        (FileNotFoundError("missing secret"), "not_found"),
        (InvalidWorkspacePathError("invalid secret"), "invalid_path"),
        (WorkspaceEscapeError("escape secret"), "invalid_path"),
        (SensitivePathError("sensitive secret"), "sensitive_path"),
        (PermissionError("permission secret"), "permission_error"),
        (IsADirectoryError("directory secret"), "is_directory"),
        (NotADirectoryError("not-directory secret"), "not_directory"),
    ],
)
# 功能：验证永久异常经 invoke_tool 分类后只执行一次并发布单个失败 attempt
# 设计：参数化覆盖 filesystem/workspace 父子类，联合断言调用次数、最终类型和 started/failed/finished 事件
async def test_permanent_exceptions_run_once_through_invocation_pipeline(
    exc: Exception,
    expected_error_type: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool = _ExceptionTool(exc)

    result, events = await _run(tool, monkeypatch=monkeypatch)

    failed = _failed(events)
    assert tool.calls == 1
    assert result.is_error
    assert result.error_type == expected_error_type
    assert len([event for event in events if isinstance(event, ToolCallStartedEvent)]) == 1
    assert len(failed) == 1
    assert [event.error_class for event in failed] == [expected_error_type]
    assert [event.attempt for event in failed] == [1]
    assert not any(isinstance(event, ToolCallFinishedEvent) for event in events)


# 功能：验证 TransientToolError 异常失败两次后第三次成功
# 设计：通过真实 exception classifier 进入 retry loop，锁定调用次数、attempt、固定摘要和最终 finished 事件
async def test_transient_exception_retries_and_succeeds_on_third_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool = _TransientExceptionNTimes(2)

    result, events = await _run(tool, monkeypatch=monkeypatch)

    failed = _failed(events)
    assert tool.calls == 3
    assert not result.is_error
    assert result.content == "ok"
    assert [event.attempt for event in failed] == [1, 2]
    assert [event.error_class for event in failed] == [
        "transient_error",
        "transient_error",
    ]
    assert [event.error_message for event in failed] == [
        "temporary tool failure",
        "temporary tool failure",
    ]
    assert all("provider transient secret" not in event.error_message for event in failed)
    assert [type(event) for event in events] == [
        ToolCallStartedEvent,
        ToolCallFailedEvent,
        ToolCallFailedEvent,
        ToolCallFinishedEvent,
    ]


# 功能：验证重试等待按 1x、2x 指数退避而不是固定、递减或偏移倍率
# 设计：替换 sleep 为只记录 delay 的异步函数，保留真实 retry loop 并观察两次等待的外部时序参数
async def test_retry_backoff_doubles_between_attempts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    delays: list[float] = []

    # 记录退避参数而不产生真实等待
    async def _record_sleep(delay: float) -> None:
        delays.append(delay)

    monkeypatch.setattr(inv_mod.asyncio, "sleep", _record_sleep)
    tool = _TransientResultNTimes(2)

    result, events = await _run(
        tool,
        monkeypatch=monkeypatch,
        retry_base_s=0.25,
    )

    assert not result.is_error
    assert tool.calls == 3
    assert delays == [0.25, 0.5]
    assert [event.attempt for event in _failed(events)] == [1, 2]


@pytest.mark.parametrize(
    ("tool_type", "expected_error_type", "expected_message"),
    [
        (_TransientResultNTimes, "transient_error", "temporary tool failure"),
        (_RateLimitedNTimes, "rate_limited", "tool rate limit exceeded"),
    ],
)
# 功能：验证两类显式瞬态错误失败两次后第三次成功
# 设计：每个参数 case 内新建 ToolResult 或异常工具，保证重复 runner 隔离并断言调用次数、attempt 与消息净化
async def test_retryable_error_succeeds_on_third_attempt(
    tool_type: type[_TransientResultNTimes] | type[_RateLimitedNTimes],
    expected_error_type: str,
    expected_message: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool = tool_type(2)
    result, events = await _run(tool, monkeypatch=monkeypatch)

    failed = _failed(events)
    assert tool.calls == 3
    assert not result.is_error
    assert result.content == "ok"
    assert [event.attempt for event in failed] == [1, 2]
    assert [event.error_class for event in failed] == [expected_error_type] * 2
    assert [event.error_message for event in failed] == [expected_message] * 2
    assert [type(event) for event in events] == [
        ToolCallStartedEvent,
        ToolCallFailedEvent,
        ToolCallFailedEvent,
        ToolCallFinishedEvent,
    ]


# 功能：验证 container_not_ready 属于可重试类型，失败两次后第三次成功
# 设计：真实 exception classifier 进入 retry loop，锁定调用次数、attempt 与最终 finished 事件
async def test_container_not_ready_retries_and_succeeds_on_third_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool = _ContainerNotReadyNTimes(2)

    result, events = await _run(tool, monkeypatch=monkeypatch)

    failed = _failed(events)
    assert tool.calls == 3
    assert not result.is_error
    assert result.content == "ok"
    assert [event.attempt for event in failed] == [1, 2]
    assert [event.error_class for event in failed] == [
        "container_not_ready",
        "container_not_ready",
    ]
    assert [type(event) for event in events] == [
        ToolCallStartedEvent,
        ToolCallFailedEvent,
        ToolCallFailedEvent,
        ToolCallFinishedEvent,
    ]


@pytest.mark.parametrize(
    ("tool_type", "expected_error_type"),
    [
        (_TransientResultNTimes, "transient_error"),
        (_RateLimitedNTimes, "rate_limited"),
    ],
)
# 功能：验证两类显式瞬态错误耗尽三个 attempt 后返回最终错误
# 设计：每个参数 case 内新建持续失败工具，保证重复 runner 隔离并断言恰好三次 attempt 且绝不发布 finished
async def test_retryable_error_exhausts_three_attempts(
    tool_type: type[_TransientResultNTimes] | type[_RateLimitedNTimes],
    expected_error_type: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool = tool_type(10)
    result, events = await _run(tool, monkeypatch=monkeypatch)

    failed = _failed(events)
    assert tool.calls == 3
    assert result.is_error
    assert result.error_type == expected_error_type
    assert [event.attempt for event in failed] == [1, 2, 3]
    assert [event.error_class for event in failed] == [expected_error_type] * 3
    assert len([event for event in events if isinstance(event, ToolCallStartedEvent)]) == 1
    assert not any(isinstance(event, ToolCallFinishedEvent) for event in events)
