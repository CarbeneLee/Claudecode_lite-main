from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pytest
from pydantic import BaseModel, ValidationError

from kama_claude.core.bus.events import (
    ToolCallFailedEvent,
    ToolCallFinishedEvent,
    ToolCallStartedEvent,
)
from kama_claude.core.context import ExecutionContext
from kama_claude.core.events.bus import EventBus
from kama_claude.core.llm.types import LlmResponse, ToolCallBlock
from kama_claude.core.loop import AgentLoop
from kama_claude.core.task.manager import TaskManager
from kama_claude.core.tools.base import BaseTool, ToolResult
from kama_claude.core.tools.builtin.task_create import TaskCreateTool
from kama_claude.core.tools.builtin.task_get import TaskGetTool
from kama_claude.core.tools.builtin.task_update import TaskUpdateTool
from kama_claude.core.tools.invocation import DirectToolInvoker, invoke_tool
from kama_claude.core.tools.registry import ToolRegistry


# 功能：通过真实 invoke_tool 执行 task 工具并收集生命周期事件
# 设计：统一注册、事件订阅和 ToolCallBlock 构造，所有分类仍经过生产 invocation 边界
async def _invoke(
    tool: BaseTool,
    params: dict[str, object],
) -> tuple[ToolResult, list[BaseModel]]:
    registry = ToolRegistry()
    registry.register(tool)
    bus = EventBus()
    events: list[BaseModel] = []

    # 收集一个 logical invocation 的完整事件序列
    async def _collect(event: BaseModel) -> None:
        events.append(event)

    bus.subscribe(_collect)
    result = await invoke_tool(
        registry,
        ToolCallBlock(id="task-call-1", name=tool.name, input=params),
        bus,
        run_id="run-1",
    )
    return result, events


# 功能：按测试 case 构造四类永久 task producer 错误
# 设计：四类错误全部走真实 manager domain path，不使用异常 side_effect 模拟业务规则
def _permanent_case(
    tmp_path: Path,
    case: str,
) -> tuple[BaseTool, dict[str, object], str]:
    manager = TaskManager(tmp_path / case)
    if case == "create_invalid":
        return (
            TaskCreateTool(manager),
            {"subject": "blocked", "blocked_by": [999]},
            "invalid_input",
        )
    if case == "get_missing":
        return TaskGetTool(manager), {"task_id": 999}, "not_found"
    if case == "update_missing":
        return TaskUpdateTool(manager), {"task_id": 999}, "not_found"

    manager.create("work")
    return (
        TaskUpdateTool(manager),
        {"task_id": 1, "add_blocked_by": [999]},
        "invalid_input",
    )


class _TaskLoopProvider:
    # 初始化调用次数和每轮收到的消息快照
    def __init__(self) -> None:
        self.calls = 0
        self.seen_messages: list[list[dict[str, object]]] = []

    # 第一轮请求缺失 task，第二轮观察错误后结束
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
        self.seen_messages.append([dict(message) for message in messages])
        if self.calls == 1:
            return LlmResponse(
                stop_reason="tool_use",
                tool_calls=[
                    ToolCallBlock(
                        id="task-call-1",
                        name="task_get",
                        input={"task_id": 999},
                    )
                ],
            )
        return LlmResponse(stop_reason="end_turn", text="handled missing task")


# 功能：验证 task_create 将缺失依赖精确映射为 invalid_input
# 设计：真实 manager 抛 TaskValidationError，direct invoke 只捕获该 domain 类型
async def test_task_create_missing_dependency_is_invalid_input(tmp_path: Path) -> None:
    result = await TaskCreateTool(TaskManager(tmp_path)).invoke(
        {"subject": "blocked", "blocked_by": [999]}
    )

    assert result.is_error
    assert result.error_type == "invalid_input"


# 功能：验证 task_get 将缺失 task ID 精确映射为 not_found
# 设计：空任务目录触发 TaskNotFoundError，断言不再使用 legacy runtime_error
async def test_task_get_missing_id_is_not_found(tmp_path: Path) -> None:
    result = await TaskGetTool(TaskManager(tmp_path)).invoke({"task_id": 999})

    assert result.is_error
    assert result.error_type == "not_found"


# 功能：验证 task_update 将缺失 task ID 精确映射为 not_found
# 设计：空任务目录走真实 update→_load 路径，冻结资源缺失与业务校验的差异
async def test_task_update_missing_id_is_not_found(tmp_path: Path) -> None:
    result = await TaskUpdateTool(TaskManager(tmp_path)).invoke({"task_id": 999})

    assert result.is_error
    assert result.error_type == "not_found"


# 功能：验证 task_update 通过真实缺失依赖业务路径返回 invalid_input
# 设计：真实 manager 校验 add_blocked_by=[999]，不使用 Mock 制造 domain exception
async def test_task_update_missing_dependency_is_invalid_input(tmp_path: Path) -> None:
    manager = TaskManager(tmp_path)
    manager.create("work")

    result = await TaskUpdateTool(manager).invoke(
        {"task_id": 1, "add_blocked_by": [999]}
    )

    assert result.error_type == "invalid_input"
    assert manager.get(1).blocked_by == []


# 功能：验证三个 task 工具的 direct invoke 与中央参数校验保持一致
# 设计：为每个工具提供类型错误输入，统一断言 Pydantic ValidationError 在副作用前抛出
async def test_task_tools_direct_invoke_validate_params(tmp_path: Path) -> None:
    cases: list[tuple[BaseTool, dict[str, object]]] = [
        (TaskCreateTool(TaskManager(tmp_path / "create")), {"subject": 42}),
        (TaskGetTool(TaskManager(tmp_path / "get")), {"task_id": "bad"}),
        (
            TaskUpdateTool(TaskManager(tmp_path / "update")),
            {"task_id": 1, "add_blocked_by": ["bad"]},
        ),
    ]

    for tool, params in cases:
        with pytest.raises(ValidationError):
            await tool.invoke(params)


# 功能：验证 task 参数类型错误由 params_model 在中央映射为 schema_error
# 设计：真实 invoke_tool 调用 task_get，断言 manager 未执行且仅一个 failed attempt
async def test_task_params_error_is_schema_error_without_retry(tmp_path: Path) -> None:
    manager = TaskManager(tmp_path)
    get_mock = Mock(wraps=manager.get)
    manager_get = get_mock
    setattr(manager, "get", manager_get)

    result, events = await _invoke(
        TaskGetTool(manager),
        {"task_id": "not-an-id"},
    )

    assert result.error_type == "schema_error"
    assert get_mock.call_count == 0
    assert len([e for e in events if isinstance(e, ToolCallStartedEvent)]) == 1
    assert [e.attempt for e in events if isinstance(e, ToolCallFailedEvent)] == [1]
    assert not any(isinstance(e, ToolCallFinishedEvent) for e in events)


@pytest.mark.parametrize(
    "case",
    ["create_invalid", "get_missing", "update_missing", "update_invalid"],
)
# 功能：验证四类 task 永久错误经 invocation 都只执行一个 attempt
# 设计：参数化 producer 映射，联合锁定最终类型、一个 started/failed 和零 finished
async def test_task_permanent_errors_emit_one_failed_attempt(
    tmp_path: Path,
    case: str,
) -> None:
    tool, params, expected_type = _permanent_case(tmp_path, case)

    result, events = await _invoke(tool, params)

    assert result.error_type == expected_type
    assert len([e for e in events if isinstance(e, ToolCallStartedEvent)]) == 1
    assert [e.attempt for e in events if isinstance(e, ToolCallFailedEvent)] == [1]
    assert not any(isinstance(e, ToolCallFinishedEvent) for e in events)


# 功能：验证非 domain ValueError 不被 task_get 捕获或泄露
# 设计：direct invoke 必须上抛，真实 invocation 则安全化为 execution_error 且隐藏 token
async def test_task_unknown_value_error_bubbles_to_safe_central_classifier(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "/private/tasks.json token=task-secret"
    manager = TaskManager(tmp_path)
    monkeypatch.setattr(manager, "get", Mock(side_effect=ValueError(secret)))
    tool = TaskGetTool(manager)

    with pytest.raises(ValueError, match="task-secret"):
        await tool.invoke({"task_id": 1})

    result, events = await _invoke(tool, {"task_id": 1})

    assert result.error_type == "execution_error"
    assert result.content == "tool execution failed"
    assert secret not in result.content
    assert [e.attempt for e in events if isinstance(e, ToolCallFailedEvent)] == [1]


# 功能：验证 task 永久错误结果继续进入现有 AgentLoop context
# 设计：两轮 provider 先调用真实 task_get 再观察 tool_result，断言错误标记和内容可见
async def test_task_error_result_enters_agent_loop_context(tmp_path: Path) -> None:
    provider = _TaskLoopProvider()
    registry = ToolRegistry()
    registry.register(TaskGetTool(TaskManager(tmp_path)))
    context = ExecutionContext(run_id="run-1", goal="inspect task", max_steps=3)

    bus = EventBus()
    await AgentLoop(provider, DirectToolInvoker(registry, bus, context.run_id), bus).run(context)

    tool_result_message = provider.seen_messages[1][2]
    content = tool_result_message["content"]
    assert isinstance(content, list)
    block = content[0]
    assert isinstance(block, dict)
    assert block["is_error"] is True
    assert "task 999 not found" in str(block["content"])
    assert context.status == "success"
