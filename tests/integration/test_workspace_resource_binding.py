from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from kama_claude.core.config import KamaConfig
from kama_claude.core.events.bus import EventBus
from kama_claude.core.llm.types import LlmResponse, ToolCallBlock
from kama_claude.core.runner import AgentRunner
from kama_claude.core.session.manager import SessionManager
from kama_claude.core.session.store import SessionStore


class _CapturingProvider:
    # 初始化 system prompt 捕获列表
    def __init__(self) -> None:
        self.systems: list[str] = []

    # 捕获每轮 system prompt 并立即结束 run
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
        self.systems.append(system or "")
        return LlmResponse(stop_reason="end_turn", text="done")


class _SpawnProvider:
    # 初始化父子调用状态和 child system 捕获值
    def __init__(self) -> None:
        self.parent_calls = 0
        self.child_system = ""

    # 父 agent 首轮请求 subagent，child 完成后父 agent 结束
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
        if system and "local-planner" in system:
            self.child_system = system
            return LlmResponse(stop_reason="end_turn", text="child done")
        self.parent_calls += 1
        if self.parent_calls == 1:
            return LlmResponse(
                stop_reason="tool_use",
                tool_calls=[
                    ToolCallBlock(
                        id="spawn-1",
                        name="spawn_agent",
                        input={
                            "description": "inspect project",
                            "prompt": "inspect project resources",
                            "subagent_type": "planner",
                        },
                    )
                ],
            )
        return LlmResponse(stop_reason="end_turn", text="parent done")


# 创建使用显式 workspace 与 fake provider 的真实 AgentRunner
def _runner(
    workspace_root: Path,
    provider: object,
    runs_dir: Path,
) -> AgentRunner:
    return AgentRunner(
        KamaConfig(),
        workspace_root=workspace_root,
        provider=provider,  # type: ignore[arg-type]
        runs_dir=runs_dir,
    )


# 写入指定 workspace 的项目 context
def _write_context(workspace: Path, content: str) -> None:
    kama_dir = workspace / ".kama"
    kama_dir.mkdir(parents=True)
    (kama_dir / "context.md").write_text(content, encoding="utf-8")


# 功能：验证 daemon cwd=A 时 Session workspace=B 的真实 Runner 只加载 B context
# 设计：通过 SessionManager 到 AgentRunner 的完整内存链路捕获 LLM system prompt
async def test_session_runner_uses_workspace_context_not_daemon_cwd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon_cwd = tmp_path / "daemon"
    workspace = tmp_path / "workspace"
    _write_context(daemon_cwd, "daemon-context")
    _write_context(workspace, "workspace-context")
    monkeypatch.chdir(daemon_cwd)
    provider = _CapturingProvider()
    store = SessionStore(tmp_path / "sessions")
    manager = SessionManager(
        store,
        lambda root: _runner(root, provider, tmp_path / "runs"),
        EventBus(),
    )
    session = await manager.create("chat", workspace_root=workspace.resolve())

    await manager.send_message(session.id, "inspect")
    await asyncio.gather(*manager.active_run_tasks())

    assert "workspace-context" in provider.systems[0]
    assert "daemon-context" not in provider.systems[0]


# 功能：验证两个 Session 通过真实 runner factory 分别获得 workspace A/B
# 设计：记录 factory root 并用不同 context 的 system prompt 验证构造参数与资源读取一致
async def test_two_sessions_build_runners_with_distinct_workspaces(tmp_path: Path) -> None:
    workspace_a = tmp_path / "workspace-a"
    workspace_b = tmp_path / "workspace-b"
    _write_context(workspace_a, "context-a")
    _write_context(workspace_b, "context-b")
    roots: list[Path] = []
    provider = _CapturingProvider()

    # 记录 manager 传入的 root 并创建真实 runner
    def runner_factory(root: Path) -> AgentRunner:
        roots.append(root)
        return _runner(root, provider, tmp_path / "runs")

    manager = SessionManager(
        SessionStore(tmp_path / "sessions"),
        runner_factory,
        EventBus(),
    )
    session_a = await manager.create("chat", workspace_root=workspace_a.resolve())
    session_b = await manager.create("chat", workspace_root=workspace_b.resolve())

    await manager.send_message(session_a.id, "from a")
    await asyncio.gather(*manager.active_run_tasks())
    await manager.send_message(session_b.id, "from b")
    await asyncio.gather(*manager.active_run_tasks())

    assert roots == [workspace_a.resolve(), workspace_b.resolve()]
    assert "context-a" in provider.systems[0]
    assert "context-b" in provider.systems[1]


# 功能：验证 chat 多轮始终使用 Session 创建时的 workspace
# 设计：两轮之间改变 process cwd，断言 factory root 和 system context 均保持不变
async def test_chat_multiround_keeps_session_workspace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    daemon_a = tmp_path / "daemon-a"
    daemon_b = tmp_path / "daemon-b"
    _write_context(workspace, "stable-context")
    daemon_a.mkdir()
    daemon_b.mkdir()
    roots: list[Path] = []
    provider = _CapturingProvider()

    # 记录每轮 runner root，验证会话多轮传播稳定
    def runner_factory(root: Path) -> AgentRunner:
        roots.append(root)
        return _runner(root, provider, tmp_path / "runs")

    manager = SessionManager(
        SessionStore(tmp_path / "sessions"),
        runner_factory,
        EventBus(),
    )
    session = await manager.create("chat", workspace_root=workspace.resolve())
    monkeypatch.chdir(daemon_a)
    await manager.send_message(session.id, "first")
    await asyncio.gather(*manager.active_run_tasks())
    monkeypatch.chdir(daemon_b)
    await manager.send_message(session.id, "second")
    await asyncio.gather(*manager.active_run_tasks())

    assert roots == [workspace.resolve(), workspace.resolve()]
    assert len(provider.systems) == 2
    assert all("stable-context" in system for system in provider.systems)


# 功能：验证 parent Runner 派生的 subagent 继承 workspace profile 与 context
# 设计：fake provider 发起真实 spawn_agent 调用并捕获 child system prompt
async def test_runner_subagent_inherits_project_resources(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    agents = workspace / ".kama" / "agents"
    agents.mkdir(parents=True)
    (workspace / ".kama" / "context.md").write_text(
        "project-context",
        encoding="utf-8",
    )
    (agents / "planner.toml").write_text(
        '[agent]\ndescription = "local"\nsystem_prompt = "local-planner"\n'
        'allowed_tools = ["read_file"]\n',
        encoding="utf-8",
    )
    provider = _SpawnProvider()
    runner = _runner(workspace.resolve(), provider, tmp_path / "runs")

    outcome = await runner.run_and_capture("delegate")

    assert outcome.status == "success"
    assert "local-planner" in provider.child_system
    assert "project-context" in provider.child_system
