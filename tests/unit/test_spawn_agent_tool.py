from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from kama_claude.core.agents.loader import AgentProfile
from kama_claude.core.bus.events import (
    SubagentFinishedEvent,
    SubagentStartedEvent,
    ToolCallFailedEvent,
    ToolCallFinishedEvent,
    ToolCallStartedEvent,
)
from kama_claude.core.context import ExecutionContext
from kama_claude.core.events.bus import EventBus
from kama_claude.core.llm.types import LlmResponse, ToolCallBlock, UsageStats
from kama_claude.core.loop import AgentLoop
from kama_claude.core.sandbox.config import SandboxConfig
from kama_claude.core.sandbox.executors import ContainerExecutor, HostExecutor
from kama_claude.core.sandbox.manager import SandboxManager
from kama_claude.core.subagent import tool as subagent_tool_module
from kama_claude.core.subagent.registry import BackgroundTaskRegistry
from kama_claude.core.subagent.tool import AgentResultTool, SpawnAgentTool
from kama_claude.core.tools.base import ToolResult
from kama_claude.core.tools.builtin.bash import BashTool
from kama_claude.core.tools.builtin.list_dir import ListDirTool
from kama_claude.core.tools.builtin.read_file import ReadFileTool
from kama_claude.core.tools.builtin.search_code import SearchCodeTool
from kama_claude.core.tools.builtin.write_file import WriteFileTool
from kama_claude.core.tools.invocation import invoke_tool
from kama_claude.core.tools.registry import ToolRegistry

_REQUIREMENT_CONTRACT = (
    "Before changing the workspace, create a concise requirement contract from every "
    "explicit acceptance criterion. For each item, record the required observable "
    "behavior, relevant failure or invalid-input behavior, any side-effect or state "
    "invariant, and the evidence you plan to use for verification. Keep this checklist "
    "visible in the conversation as you work, and update each item as implemented, "
    "verified, or unchecked. Before finishing, review every item. Do not assume unchecked "
    "items are complete: verify them when possible, otherwise clearly report the "
    "limitation. Keep the contract brief and auditable; do not expose private "
    "chain-of-thought or force any particular tool."
)
_STATE_TRANSITION_PROTOCOL = (
    "When a task changes persistent or shared state through multiple operations, briefly "
    "map the pre-state, each mutation point, every later operation that can fail, and the "
    "required post-state after success or failure. Before finishing, exercise at least "
    "one failure after an earlier mutation succeeds, and verify that rollback or "
    "compensation preserves the stated invariant. Do not apply this protocol to tasks "
    "without multi-step side effects."
)


def _make_provider(result_text: str = "child done") -> Any:
    provider = AsyncMock()
    provider.chat = AsyncMock(
        return_value=LlmResponse(
            stop_reason="end_turn",
            tool_calls=[],
            text=result_text,
            usage=UsageStats(
                input_tokens=10,
                output_tokens=5,
                cache_read_input_tokens=0,
                cache_creation_input_tokens=0,
                context_pct=0.01,
            ),
        )
    )
    return provider


def _make_tool(
    tmp_path: Path,
    provider: Any = None,
    depth: int = 0,
    journal: Any = None,
    sandbox_manager: SandboxManager | None = None,
) -> tuple[SpawnAgentTool, BackgroundTaskRegistry, EventBus]:
    bus = EventBus()
    registry = BackgroundTaskRegistry()
    tool = SpawnAgentTool(
        provider=provider or _make_provider(),
        workspace_root=tmp_path.resolve(),
        parent_bus=bus,
        parent_run_id="parent-run-01",
        permission_manager=None,
        max_steps=5,
        task_registry=registry,
        runs_dir=tmp_path,
        session_id="sess-test",
        depth=depth,
        journal=journal,
        sandbox_manager=sandbox_manager,
    )
    return tool, registry, bus


# 功能：验证 child run stream 注册严格早于 SubagentStartedEvent 发布
# 设计：fake journal 与真实 parent bus 共享顺序列表，执行最短前台 child lifecycle 比较边界顺序
async def test_child_stream_registers_before_subagent_started(tmp_path: Path) -> None:
    order: list[str] = []

    class RecordingJournal:
        # 记录 child run owner 与 parent session mapping 注册
        async def register_run(
            self,
            run_id: str,
            run_path: Path,
            *,
            session_id: str | None,
        ) -> object:
            order.append(f"register:{run_id}:{session_id}")
            return object()

    tool, _registry, bus = _make_tool(tmp_path, journal=RecordingJournal())

    async def collect(event: object) -> None:
        if isinstance(event, SubagentStartedEvent):
            order.append(f"event:{event.run_id}")

    bus.subscribe(collect)

    await tool.invoke({"description": "child", "prompt": "finish"})

    register_index = next(
        index for index, item in enumerate(order) if item.startswith("register:")
    )
    event_index = next(
        index for index, item in enumerate(order) if item.startswith("event:")
    )
    assert register_index < event_index


# 功能：验证 SpawnAgentTool 必须显式接收 workspace_root
# 设计：省略该 keyword 构造工具并断言 TypeError，防止引入 cwd fallback
def test_spawn_agent_requires_workspace_root(tmp_path: Path) -> None:
    with pytest.raises(TypeError):
        SpawnAgentTool(
            provider=_make_provider(),
            parent_bus=EventBus(),
            parent_run_id="parent",
            permission_manager=None,
            max_steps=5,
            task_registry=BackgroundTaskRegistry(),
            runs_dir=tmp_path,
            session_id="session",
        )


# 功能：验证 SpawnAgentTool 保存 canonical workspace root
# 设计：通过目录 symlink 构造工具并检查内部路径已 strict resolve
def test_spawn_agent_canonicalizes_workspace_root(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    alias = tmp_path / "workspace-link"
    alias.symlink_to(workspace, target_is_directory=True)

    tool, _, _ = _make_tool(alias)

    assert tool._workspace_root == workspace.resolve(strict=True)


# 功能：验证 nested SpawnAgentTool 继承 parent 的同一 canonical workspace
# 设计：直接构建 child registry 并检查其中 spawn_agent 的 workspace 状态
def test_nested_spawn_agent_inherits_workspace(tmp_path: Path) -> None:
    tool, _, _ = _make_tool(tmp_path)

    registry = tool._build_child_registry(EventBus(), "child-run", None)
    nested = registry.get("spawn_agent")

    assert isinstance(nested, SpawnAgentTool)
    assert nested._workspace_root == tmp_path.resolve(strict=True)


# 功能：验证注入 sandbox_manager 后 child registry 的 bash 使用容器执行器
# 设计：沿 child registry 组装路径检查 bash executor 类型，不执行命令
def test_child_registry_bash_uses_container_executor_when_sandbox_injected(
    tmp_path: Path,
) -> None:
    manager = SandboxManager(
        config=SandboxConfig(image="python:3.12-slim"),
        workspace_root=tmp_path.resolve(),
    )
    tool, _, _ = _make_tool(tmp_path, sandbox_manager=manager)

    registry = tool._build_child_registry(EventBus(), "child-run", None)
    bash_tool = registry.get("bash")

    assert isinstance(bash_tool, BashTool)
    assert isinstance(bash_tool._executor, ContainerExecutor)


# 功能：验证未注入 sandbox_manager 时 child registry 的 bash 保持宿主执行器
# 设计：默认路径断言 HostExecutor，防止子 agent 沙箱决策与顶层漂移
def test_child_registry_bash_uses_host_executor_without_sandbox(
    tmp_path: Path,
) -> None:
    tool, _, _ = _make_tool(tmp_path)

    registry = tool._build_child_registry(EventBus(), "child-run", None)
    bash_tool = registry.get("bash")

    assert isinstance(bash_tool, BashTool)
    assert isinstance(bash_tool._executor, HostExecutor)


# 功能：验证 child registry 的 read/list 工具绑定 parent workspace
# 设计：直接检查 child registry 工具 resolver root，隔离 LLM 与文件系统执行
def test_child_registry_injects_workspace_into_read_and_list_tools(
    tmp_path: Path,
) -> None:
    tool, _, _ = _make_tool(tmp_path)

    registry = tool._build_child_registry(EventBus(), "child-run", None)
    read_tool = registry.get("read_file")
    list_tool = registry.get("list_dir")

    assert isinstance(read_tool, ReadFileTool)
    assert isinstance(list_tool, ListDirTool)
    assert read_tool._resolver.root == tmp_path.resolve(strict=True)
    assert list_tool._resolver.root == tmp_path.resolve(strict=True)


# 功能：验证 Subagent child registry 注册绑定 parent workspace 的 search_code
# 设计：不运行 child loop，直接检查真实工具类型和 resolver canonical root
def test_child_registry_injects_workspace_into_search_code(tmp_path: Path) -> None:
    tool, _, _ = _make_tool(tmp_path)

    registry = tool._build_child_registry(EventBus(), "child-run", None)
    search_tool = registry.get("search_code")

    assert isinstance(search_tool, SearchCodeTool)
    assert search_tool._resolver.root == tmp_path.resolve(strict=True)


# 功能：验证 Subagent profile allowlist 可单独允许或排除 search_code
# 设计：用两个最小 AgentProfile 构建 registry，交叉断言 search/read 工具存在性
def test_child_registry_filters_search_code_by_profile_allowlist(
    tmp_path: Path,
) -> None:
    tool, _, _ = _make_tool(tmp_path)
    search_only = AgentProfile(
        name="searcher",
        description="",
        system_prompt="",
        allowed_tools=["search_code"],
    )
    read_only = AgentProfile(
        name="reader",
        description="",
        system_prompt="",
        allowed_tools=["read_file"],
    )

    search_registry = tool._build_child_registry(EventBus(), "search-run", search_only)
    read_registry = tool._build_child_registry(EventBus(), "read-run", read_only)

    assert isinstance(search_registry.get("search_code"), SearchCodeTool)
    assert search_registry.get("read_file") is None
    assert read_registry.get("search_code") is None
    assert isinstance(read_registry.get("read_file"), ReadFileTool)


# 功能：验证 child registry 的 write 工具与 Bash 继承 parent workspace
# 设计：检查 child 工具内部 root，不执行有副作用操作
def test_child_registry_injects_workspace_into_write_and_bash_tools(
    tmp_path: Path,
) -> None:
    tool, _, _ = _make_tool(tmp_path)

    registry = tool._build_child_registry(EventBus(), "child-run", None)
    write_tool = registry.get("write_file")
    bash_tool = registry.get("bash")

    assert isinstance(write_tool, WriteFileTool)
    assert isinstance(bash_tool, BashTool)
    assert write_tool._resolver.root == tmp_path.resolve(strict=True)
    assert bash_tool._workspace_root == tmp_path.resolve(strict=True)


# 功能：验证 nested subagent 的 filesystem 工具继续继承同一 workspace
# 设计：从 parent child registry 取 nested tool，再构建下一层 registry 检查四个工具 root
def test_nested_child_registry_keeps_workspace_bound_tools(tmp_path: Path) -> None:
    tool, _, _ = _make_tool(tmp_path)
    child_registry = tool._build_child_registry(EventBus(), "child-run", None)
    nested = child_registry.get("spawn_agent")
    assert isinstance(nested, SpawnAgentTool)

    nested_registry = nested._build_child_registry(EventBus(), "nested-run", None)
    read_tool = nested_registry.get("read_file")
    write_tool = nested_registry.get("write_file")
    list_tool = nested_registry.get("list_dir")
    bash_tool = nested_registry.get("bash")

    assert isinstance(read_tool, ReadFileTool)
    assert isinstance(write_tool, WriteFileTool)
    assert isinstance(list_tool, ListDirTool)
    assert isinstance(bash_tool, BashTool)
    assert read_tool._resolver.root == tmp_path.resolve(strict=True)
    assert write_tool._resolver.root == tmp_path.resolve(strict=True)
    assert list_tool._resolver.root == tmp_path.resolve(strict=True)
    assert bash_tool._workspace_root == tmp_path.resolve(strict=True)


# 功能：验证不同 workspace 的 subagent 分别加载各自 profile 与 project context
# 设计：A/B 使用同名 profile 和不同 context，捕获 provider system 参数并交叉排除污染
async def test_subagents_isolate_profile_and_context_by_workspace(tmp_path: Path) -> None:
    systems: list[str] = []
    for name in ("a", "b"):
        workspace = tmp_path / f"workspace-{name}"
        agents = workspace / ".kama" / "agents"
        agents.mkdir(parents=True)
        (workspace / ".kama" / "context.md").write_text(
            f"context-{name}",
            encoding="utf-8",
        )
        (agents / "planner.toml").write_text(
            '[agent]\ndescription = "local"\n'
            f'system_prompt = "profile-{name}"\nallowed_tools = ["read_file"]\n',
            encoding="utf-8",
        )
        provider = _make_provider()
        tool, _, _ = _make_tool(workspace, provider)

        await tool.invoke(
            {
                "description": "inspect",
                "prompt": "inspect workspace",
                "subagent_type": "planner",
            }
        )

        system = provider.chat.await_args.kwargs["system"]
        assert isinstance(system, str)
        systems.append(system)

    assert "profile-a" in systems[0]
    assert "context-a" in systems[0]
    assert "profile-b" not in systems[0]
    assert "context-b" not in systems[0]
    assert "profile-b" in systems[1]
    assert "context-b" in systems[1]
    assert "profile-a" not in systems[1]
    assert "context-a" not in systems[1]
    assert _REQUIREMENT_CONTRACT not in systems[0]
    assert _REQUIREMENT_CONTRACT not in systems[1]
    assert _STATE_TRANSITION_PROTOCOL not in systems[0]
    assert _STATE_TRANSITION_PROTOCOL not in systems[1]


# 功能：验证未指定 profile 的 subagent 各继承一次 repaired default v1 与 v2
# 设计：执行真实前台 child loop 并捕获 provider system，区别于 profile override 的完全替换路径
async def test_unprofiled_subagent_inherits_v1_and_v2(tmp_path: Path) -> None:
    provider = _make_provider()
    tool, _, _ = _make_tool(tmp_path, provider)

    result = await tool.invoke(
        {
            "description": "inspect requirements",
            "prompt": "Implement behavior A and preserve invariant B.",
        }
    )

    assert result.is_error is False
    system = provider.chat.await_args.kwargs["system"]
    assert isinstance(system, str)
    assert system.count(_REQUIREMENT_CONTRACT) == 1
    assert system.count(_STATE_TRANSITION_PROTOCOL) == 1


# 功能：验证 subagent 模块不保留绑定项目目录的全局 profile loader
# 设计：直接检查模块命名空间，锁定跨 workspace 共享实例被移除
def test_subagent_has_no_module_profile_loader() -> None:
    assert not hasattr(subagent_tool_module, "_profile_loader")


# 功能：前台模式下 spawn_agent 应阻塞直到子 agent 完成并返回其结果
# 设计：使用返回 end_turn 的 mock provider，验证 tool_result.content 包含 provider 返回的文字
@pytest.mark.asyncio
async def test_foreground_returns_result(tmp_path: Path) -> None:
    tool, _, _ = _make_tool(tmp_path, _make_provider("analysis complete"))
    result = await tool.invoke({
        "description": "分析代码",
        "prompt": "分析 src/ 目录",
    })
    assert not result.is_error
    assert "analysis complete" in result.content


# 功能：后台模式应立即返回含 run_id 的消息，不阻塞等待子 agent
# 设计：run_in_background=true 后验证返回消息含 "run_id=" 并且任务注册表已有对应条目
@pytest.mark.asyncio
async def test_background_returns_run_id(tmp_path: Path) -> None:
    tool, registry, _ = _make_tool(tmp_path)
    result = await tool.invoke({
        "description": "后台任务",
        "prompt": "做点事",
        "run_in_background": True,
    })
    assert not result.is_error
    assert "run_id=" in result.content
    # extract run_id from message
    run_id = result.content.split("run_id=")[1].split(".")[0]
    assert registry.get(run_id) is not None


# 功能：验证后台 run_id 返回后立即取消仍进入生命周期边界并配对 finished
# 设计：不等待 child entered 信号而直接取消真实 registry task，稳定复现旧实现的首次调度竞态
async def test_background_immediate_cancellation_after_run_id_is_paired(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    never_release = asyncio.Event()

    # 保持 child loop 挂起，确保立即取消由 lifecycle owner 处理
    async def _wait_forever(
        self: AgentLoop,
        context: ExecutionContext,
    ) -> None:
        await never_release.wait()

    monkeypatch.setattr(AgentLoop, "run", _wait_forever)
    tool, registry, bus = _make_tool(tmp_path)
    events: list[Any] = []

    # 收集真实父 bus 上的公开 lifecycle 事件
    async def _collect(event: Any) -> None:
        events.append(event)

    bus.subscribe(_collect)
    spawn_result = await tool.invoke(
        {"description": "child", "prompt": "work", "run_in_background": True}
    )
    run_id = spawn_result.content.split("run_id=")[1].split(".")[0]
    entry = registry.get(run_id)
    assert entry is not None
    task, context = entry

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert task.cancelled()
    assert context.status == "failed"
    assert context.reason == "cancelled"
    lifecycle_events = [
        event
        for event in events
        if isinstance(event, (SubagentStartedEvent, SubagentFinishedEvent))
    ]
    assert [type(event) for event in lifecycle_events] == [
        SubagentStartedEvent,
        SubagentFinishedEvent,
    ]
    assert lifecycle_events[1].status == "failed"
    assert await AgentResultTool(registry).invoke({"run_id": run_id}) == ToolResult(
        content="Subagent was cancelled.",
        is_error=True,
        error_type="command_failed",
    )


# 功能：验证 spawn 在等待后台生命周期握手时取消会清理已注册 task 并保持取消身份
# 设计：受控 Event.wait 在真实 invoke_tool 任务内触发取消，检查公开 registry 与事件而非 task 私有字段
async def test_background_handshake_wait_cancellation_cleans_registered_task(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    child_blocker = asyncio.Event()
    handshake_blocker = asyncio.Event()
    observed_cancellations: list[asyncio.CancelledError] = []

    class _CancellingHandshake:
        # 保持受控握手未完成，使 invoke 停留在 wait 边界
        def set(self) -> None:
            return None

        # 在当前 invoke task 内注入取消并记录接收到的异常对象
        async def wait(self) -> None:
            current = asyncio.current_task()
            assert current is not None
            asyncio.get_running_loop().call_soon(current.cancel)
            try:
                await handshake_blocker.wait()
            except asyncio.CancelledError as exc:
                observed_cancellations.append(exc)
                raise

    # 保持 background task 活跃，供 invoke cancellation 路径负责清理
    async def _wait_forever(
        self: AgentLoop,
        context: ExecutionContext,
    ) -> None:
        await child_blocker.wait()

    monkeypatch.setattr(AgentLoop, "run", _wait_forever)
    monkeypatch.setattr(subagent_tool_module.asyncio, "Event", _CancellingHandshake)
    tool, registry, bus = _make_tool(tmp_path)
    tool_registry = ToolRegistry()
    tool_registry.register(tool)
    events: list[Any] = []

    # 收集 tool 与 subagent 公开事件以排除伪造失败结果
    async def _collect(event: Any) -> None:
        events.append(event)

    bus.subscribe(_collect)
    with pytest.raises(asyncio.CancelledError) as exc_info:
        await invoke_tool(
            tool_registry,
            ToolCallBlock(
                id="handshake-cancel",
                name="spawn_agent",
                input={
                    "description": "child",
                    "prompt": "work",
                    "run_in_background": True,
                },
            ),
            bus,
            run_id="parent-run-01",
        )

    assert observed_cancellations == [exc_info.value]
    entries = registry.all()
    assert len(entries) == 1
    task, context = entries[0]
    assert task.cancelled()
    assert context.status == "failed"
    assert context.reason == "cancelled"
    assert [type(event) for event in events] == [
        ToolCallStartedEvent,
        SubagentStartedEvent,
        SubagentFinishedEvent,
    ]
    assert not any(isinstance(event, ToolCallFailedEvent) for event in events)
    assert not any(isinstance(event, ToolCallFinishedEvent) for event in events)


# 功能：后台任务未完成时 agent_result 应返回 "still running"
# 设计：用 Event 阻塞 provider.chat，在未等待任务完成时查询 agent_result
@pytest.mark.asyncio
async def test_agent_result_pending(tmp_path: Path) -> None:
    event = asyncio.Event()

    async def slow_chat(*args: Any, **kwargs: Any) -> LlmResponse:
        await event.wait()
        return LlmResponse(
            stop_reason="end_turn",
            tool_calls=[],
            text="done",
            usage=UsageStats(0, 0, 0, 0, 0.0),
        )

    provider = MagicMock()
    provider.chat = slow_chat

    tool, registry, _ = _make_tool(tmp_path, provider)
    spawn_result = await tool.invoke({
        "description": "slow task",
        "prompt": "do something slow",
        "run_in_background": True,
    })
    run_id = spawn_result.content.split("run_id=")[1].split(".")[0]

    result_tool = AgentResultTool(registry)
    result = await result_tool.invoke({"run_id": run_id})
    assert result.content == "still running"
    assert not result.is_error

    event.set()
    entry = registry.get(run_id)
    assert entry is not None
    task, _ = entry
    await task


# 功能：后台任务完成后 agent_result 应返回子 agent 的最终文本
# 设计：等待后台任务 task 完成后调用 agent_result，断言返回内容与 provider 结果一致
@pytest.mark.asyncio
async def test_agent_result_done(tmp_path: Path) -> None:
    tool, registry, _ = _make_tool(tmp_path, _make_provider("final answer"))
    spawn_result = await tool.invoke({
        "description": "bg task",
        "prompt": "do it",
        "run_in_background": True,
    })
    run_id = spawn_result.content.split("run_id=")[1].split(".")[0]

    entry = registry.get(run_id)
    assert entry is not None
    task, _ = entry
    await asyncio.wait_for(task, timeout=5.0)

    result_tool = AgentResultTool(registry)
    result = await result_tool.invoke({"run_id": run_id})
    assert not result.is_error
    assert "final answer" in result.content


# 功能：depth=2 时调用 spawn_agent 应返回 is_error=True（嵌套限制）
# 设计：构造 depth=2 的工具，断言 invoke 直接返回错误而不调用 provider
@pytest.mark.asyncio
async def test_nesting_limit(tmp_path: Path) -> None:
    provider = _make_provider()
    tool, _, _ = _make_tool(tmp_path, provider, depth=2)
    result = await tool.invoke({
        "description": "nested",
        "prompt": "do nested work",
    })
    assert result.is_error
    assert result.error_type == "invalid_input"
    assert "nesting limit" in result.content
    provider.chat.assert_not_called()


# 功能：agent_result 查询不存在的 run_id 应返回 is_error=True
# 设计：空 registry 中查询随机 run_id，验证错误消息含 "Unknown"
@pytest.mark.asyncio
async def test_agent_result_unknown_run_id(tmp_path: Path) -> None:
    registry = BackgroundTaskRegistry()
    tool = AgentResultTool(registry)
    result = await tool.invoke({"run_id": "nonexistent-id"})
    assert result.is_error
    assert result.error_type == "not_found"
    assert "Unknown" in result.content


# 功能：SubagentStartedEvent 应在前台 spawn 时发布到父 bus
# 设计：订阅父 bus 收集所有事件，断言 subagent.started 出现，且 parent_run_id 和 description 正确
@pytest.mark.asyncio
async def test_foreground_publishes_started_event(tmp_path: Path) -> None:
    from kama_claude.core.bus.events import SubagentStartedEvent

    tool, _, bus = _make_tool(tmp_path)
    events: list[Any] = []

    async def _collect(e: Any) -> None:
        events.append(e)

    bus.subscribe(_collect)

    await tool.invoke({
        "description": "test task",
        "prompt": "test prompt",
    })
    started = [e for e in events if isinstance(e, SubagentStartedEvent)]
    assert len(started) == 1
    assert started[0].parent_run_id == "parent-run-01"
    assert started[0].description == "test task"
