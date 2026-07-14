from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from kama_claude.core.events.bus import EventBus
from kama_claude.core.llm.types import LlmResponse, UsageStats
from kama_claude.core.subagent import tool as subagent_tool_module
from kama_claude.core.subagent.registry import BackgroundTaskRegistry
from kama_claude.core.subagent.tool import AgentResultTool, SpawnAgentTool


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
    )
    return tool, registry, bus


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
    await asyncio.sleep(0.05)


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
