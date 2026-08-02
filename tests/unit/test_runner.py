from __future__ import annotations

import asyncio
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from kama_claude.core.bus.events import (
    GitRunDiffEvent,
    LlmModelSelectedEvent,
    StepFinishedEvent,
    ToolCallStartedEvent,
)
from kama_claude.core.config import KamaConfig
from kama_claude.core.events.bus import EventBus
from kama_claude.core.events.journal import EventJournalCoordinator
from kama_claude.core.events.writer import EventWriter
from kama_claude.core.git.config import GitConfig
from kama_claude.core.git.errors import DirtyWorkspaceError, GitUnavailableError
from kama_claude.core.git.manager import GitDiff
from kama_claude.core.llm.types import LlmResponse, ToolCallBlock
from kama_claude.core.loop import AgentLoop
from kama_claude.core.runner import AgentRunner
from kama_claude.core.sandbox.config import SandboxConfig
from kama_claude.core.sandbox.executors import (
    ContainerExecutor,
    ExecResult,
    HostExecutor,
)
from kama_claude.core.sandbox.manager import SandboxManager
from kama_claude.core.sandbox.runtime import ContainerRuntime
from kama_claude.core.subagent.tool import SpawnAgentTool
from kama_claude.core.task.manager import TaskManager
from kama_claude.core.tools.builtin.bash import BashTool
from kama_claude.core.tools.builtin.list_dir import ListDirTool
from kama_claude.core.tools.builtin.read_file import ReadFileTool
from kama_claude.core.tools.builtin.search_code import SearchCodeTool
from kama_claude.core.tools.builtin.write_file import WriteFileTool
from kama_claude.eval.graders import grade_trace

# --- mock provider -----------------------------------------------------------


class _EndTurnProvider:
    """Immediately returns end_turn; no API calls made."""

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
        return LlmResponse(stop_reason="end_turn", text="done")


class _LoopingProvider:
    """Always returns tool_use with an unknown tool to exhaust max_steps."""

    def __init__(self) -> None:
        self._call = 0

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
        self._call += 1
        tc = ToolCallBlock(id=f"t{self._call}", name="unknown_tool", input={})
        return LlmResponse(stop_reason="tool_use", tool_calls=[tc])


class _CapturingProvider:
    # 初始化捕获型 provider，保存固定响应
    def __init__(self, response: LlmResponse) -> None:
        self.response = response
        self.messages: list[dict[str, object]] = []
        self.system: str | None = None

    # 捕获本次 LLM 调用的 messages 和 system prompt
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
        self.messages = [dict(m) for m in messages]
        self.system = system
        return self.response


class _ModelThenErrorProvider:
    # 发布稳定model identity后抛出无网络provider异常
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
        await bus.publish(
            LlmModelSelectedEvent(
                run_id=run_id,
                model="local-test-model",
                strategy="static",
                ts="2026-07-29T00:00:00+00:00",
            )
        )
        raise RuntimeError("local provider failure")


# --- helpers -----------------------------------------------------------------


def _config(max_steps: int = 5) -> KamaConfig:
    cfg = KamaConfig()
    cfg.agent.max_steps = max_steps
    return cfg


async def _run(
    goal: str = "test goal",
    *,
    provider: object | None = None,
    config: KamaConfig | None = None,
    tmp_path: Path,
) -> list[BaseModel]:
    collected: list[BaseModel] = []

    async def _collect(e: BaseModel) -> None:
        collected.append(e)

    cfg = config or _config()
    runner = AgentRunner(
        cfg,
        workspace_root=tmp_path.resolve(),
        provider=provider or _EndTurnProvider(),  # type: ignore[arg-type]
        extra_handlers=[_collect],
        runs_dir=tmp_path,
    )
    await runner.run(goal)
    return collected


# --- tests -------------------------------------------------------------------


# 功能：验证 AgentRunner 构造必须显式提供 workspace_root
# 设计：直接调用最小构造器并期待 TypeError，防止未来加入 Path.cwd 默认值
def test_agent_runner_requires_workspace_root() -> None:
    with pytest.raises(TypeError):
        AgentRunner(_config())


# 功能：验证 AgentRunner 内部保存 canonical workspace Path
# 设计：通过指向真实 workspace 的 symlink 构造 runner，断言内部值已 strict resolve
def test_agent_runner_canonicalizes_workspace_root(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    alias = tmp_path / "workspace-link"
    alias.symlink_to(workspace, target_is_directory=True)

    runner = AgentRunner(_config(), workspace_root=alias)

    assert runner._workspace_root == workspace.resolve(strict=True)


# 功能：验证 AgentRunner 构造顶层 SpawnAgentTool 时传递自身 workspace
# 设计：调用 registry 组装路径并检查注册工具的 canonical root，不执行 LLM
def test_runner_passes_workspace_to_spawn_agent(tmp_path: Path) -> None:
    runner = AgentRunner(_config(), workspace_root=tmp_path.resolve())

    registry = runner._build_registry(
        TaskManager(tmp_path / "tasks"),
        run_id="run-1",
        provider=_EndTurnProvider(),
        bus=EventBus(),
    )
    spawn_tool = registry.get("spawn_agent")

    assert spawn_tool is not None
    assert spawn_tool._workspace_root == tmp_path.resolve(strict=True)  # type: ignore[attr-defined]


class _RecordingRuntime(ContainerRuntime):
    # 进程内 fake runtime：记录 exec 调用，验证 bash 经容器路径执行
    def __init__(self) -> None:
        self.ensure_calls = 0
        self.exec_calls = 0
        self.close_calls = 0
        self.calls: list[tuple[str, str, float]] = []

    async def ensure_running(self) -> None:
        self.ensure_calls += 1

    async def exec(self, command: str, *, cwd: str, timeout: float) -> ExecResult:
        self.exec_calls += 1
        self.calls.append((command, cwd, timeout))
        return ExecResult(output=b"fake-out", returncode=0, timed_out=False)

    async def close(self) -> None:
        self.close_calls += 1


def _sandbox_manager(workspace: Path, runtime: ContainerRuntime) -> SandboxManager:
    return SandboxManager(
        config=SandboxConfig(image="python:3.12-slim"),
        workspace_root=workspace,
        runtime=runtime,
    )


# 功能：验证注入 sandbox_manager 后 registry 的 bash 工具改用容器执行器
# 设计：检查 bash 工具持有的 executor 类型与绑定的 workspace，不执行命令
def test_runner_bash_uses_container_executor_when_sandbox_injected(
    tmp_path: Path,
) -> None:
    runtime = _RecordingRuntime()
    manager = _sandbox_manager(tmp_path.resolve(), runtime)
    runner = AgentRunner(
        _config(),
        workspace_root=tmp_path.resolve(),
        sandbox_manager=manager,
    )

    registry = runner._build_registry(TaskManager(tmp_path / "tasks"))
    bash_tool = registry.get("bash")

    assert isinstance(bash_tool, BashTool)
    assert isinstance(bash_tool._executor, ContainerExecutor)
    assert runtime.ensure_calls == 0  # 懒创建：装配不触发容器启动


# 功能：验证未注入 sandbox_manager 时 bash 工具保持宿主执行器
# 设计：默认装配路径断言 HostExecutor，防止沙箱关闭场景误入容器路径
def test_runner_bash_uses_host_executor_without_sandbox(tmp_path: Path) -> None:
    runner = AgentRunner(_config(), workspace_root=tmp_path.resolve())

    registry = runner._build_registry(TaskManager(tmp_path / "tasks"))
    bash_tool = registry.get("bash")

    assert isinstance(bash_tool, BashTool)
    assert isinstance(bash_tool._executor, HostExecutor)


# 功能：验证注入沙箱后 bash 调用经容器路径转发（映射 cwd + 透传命令）
# 设计：真实 invoke 走 runner 装配链，断言 fake runtime 收到容器内路径与命令
async def test_runner_bash_invokes_through_container(tmp_path: Path) -> None:
    runtime = _RecordingRuntime()
    manager = _sandbox_manager(tmp_path.resolve(), runtime)
    runner = AgentRunner(
        _config(),
        workspace_root=tmp_path.resolve(),
        sandbox_manager=manager,
    )
    registry = runner._build_registry(TaskManager(tmp_path / "tasks"))

    result = await registry.get("bash").invoke({"command": "echo hi"})  # type: ignore[union-attr]

    assert not result.is_error
    assert result.content == "fake-out"
    assert runtime.ensure_calls == 1
    assert runtime.exec_calls == 1
    assert runtime.calls == [("echo hi", "/workspace", 60)]


# 功能：验证 AgentRunner 把 sandbox_manager 传递给顶层 SpawnAgentTool
# 设计：沿 registry 组装路径检查 spawn 工具持有同一 manager 实例
def test_runner_passes_sandbox_manager_to_spawn_agent(tmp_path: Path) -> None:
    runtime = _RecordingRuntime()
    manager = _sandbox_manager(tmp_path.resolve(), runtime)
    runner = AgentRunner(
        _config(),
        workspace_root=tmp_path.resolve(),
        sandbox_manager=manager,
    )

    registry = runner._build_registry(
        TaskManager(tmp_path / "tasks"),
        run_id="run-1",
        provider=_EndTurnProvider(),
        bus=EventBus(),
    )
    spawn_tool = registry.get("spawn_agent")

    assert spawn_tool is not None
    assert spawn_tool._sandbox_manager is manager  # type: ignore[attr-defined]


# 功能：验证 Runner registry 的 read/list 工具绑定 Runner workspace
# 设计：检查真实 registry 中两个工具持有的 resolver root，不执行文件操作
def test_runner_injects_workspace_into_read_and_list_tools(tmp_path: Path) -> None:
    runner = AgentRunner(_config(), workspace_root=tmp_path.resolve())

    registry = runner._build_registry(TaskManager(tmp_path / "tasks"))
    read_tool = registry.get("read_file")
    list_tool = registry.get("list_dir")

    assert isinstance(read_tool, ReadFileTool)
    assert isinstance(list_tool, ListDirTool)
    assert read_tool._resolver.root == tmp_path.resolve(strict=True)
    assert list_tool._resolver.root == tmp_path.resolve(strict=True)


# 功能：验证 Runner 注册 workspace-bound search_code 并尊重 tool whitelist
# 设计：分别构建仅允许 search 和仅允许 read 的 registry，同时检查 resolver root
def test_runner_registers_workspace_bound_search_code_with_whitelist(
    tmp_path: Path,
) -> None:
    runner = AgentRunner(_config(), workspace_root=tmp_path.resolve())

    search_registry = runner._build_registry(
        TaskManager(tmp_path / "search-tasks"),
        tool_whitelist=["search_code"],
    )
    read_registry = runner._build_registry(
        TaskManager(tmp_path / "read-tasks"),
        tool_whitelist=["read_file"],
    )
    search_tool = search_registry.get("search_code")

    assert isinstance(search_tool, SearchCodeTool)
    assert search_tool._resolver.root == tmp_path.resolve(strict=True)
    assert read_registry.get("search_code") is None


# 功能：验证 Runner registry 的 write 工具与 Bash 启动目录绑定 Runner workspace
# 设计：检查真实 registry 中 resolver root 和 Bash workspace，不执行命令或写入
def test_runner_injects_workspace_into_write_and_bash_tools(tmp_path: Path) -> None:
    runner = AgentRunner(_config(), workspace_root=tmp_path.resolve())

    registry = runner._build_registry(TaskManager(tmp_path / "tasks"))
    write_tool = registry.get("write_file")
    bash_tool = registry.get("bash")

    assert isinstance(write_tool, WriteFileTool)
    assert isinstance(bash_tool, BashTool)
    assert write_tool._resolver.root == tmp_path.resolve(strict=True)
    assert bash_tool._workspace_root == tmp_path.resolve(strict=True)


# 功能：验证 workspace A/B 的 Runner filesystem 工具实例不共享 root
# 设计：构造两个 registry 并比较 read/write/Bash 的内部 root 与对象身份
def test_runner_tool_instances_are_workspace_isolated(tmp_path: Path) -> None:
    workspace_a = tmp_path / "workspace-a"
    workspace_b = tmp_path / "workspace-b"
    workspace_a.mkdir()
    workspace_b.mkdir()
    registry_a = AgentRunner(
        _config(), workspace_root=workspace_a
    )._build_registry(TaskManager(tmp_path / "tasks-a"))
    registry_b = AgentRunner(
        _config(), workspace_root=workspace_b
    )._build_registry(TaskManager(tmp_path / "tasks-b"))

    read_a = registry_a.get("read_file")
    read_b = registry_b.get("read_file")
    write_a = registry_a.get("write_file")
    write_b = registry_b.get("write_file")
    bash_a = registry_a.get("bash")
    bash_b = registry_b.get("bash")

    assert isinstance(read_a, ReadFileTool)
    assert isinstance(read_b, ReadFileTool)
    assert isinstance(write_a, WriteFileTool)
    assert isinstance(write_b, WriteFileTool)
    assert isinstance(bash_a, BashTool)
    assert isinstance(bash_b, BashTool)
    assert read_a._resolver.root == workspace_a.resolve(strict=True)
    assert read_b._resolver.root == workspace_b.resolve(strict=True)
    assert write_a._resolver is not write_b._resolver
    assert bash_a._workspace_root == workspace_a.resolve(strict=True)
    assert bash_b._workspace_root == workspace_b.resolve(strict=True)


# 功能：验证四个 workspace-bound builtin 工具拒绝零参数构造
# 设计：逐个省略依赖并断言 TypeError，防止 production 重新引入 cwd fallback
def test_workspace_bound_tools_require_constructor_dependencies() -> None:
    for tool_type in (ReadFileTool, WriteFileTool, ListDirTool, BashTool):
        with pytest.raises(TypeError):
            tool_type()


# 功能：验证 run 开始时发布携带正确 goal 的 run.started 事件
# 设计：用 extra_handlers 收集事件，而非从 events.jsonl 读取，避免文件 I/O 耦合；聚焦 runner 层的事件发布职责
async def test_run_started_event_published(tmp_path: Path) -> None:
    events = await _run(goal="my goal", tmp_path=tmp_path)
    types = [e.type for e in events]  # type: ignore[attr-defined]
    assert "run.started" in types
    started = next(e for e in events if e.type == "run.started")  # type: ignore[attr-defined]
    assert started.goal == "my goal"  # type: ignore[attr-defined]


# 功能：验证成功完成时发布 status=success 的 run.finished 事件
# 设计：EndTurnProvider 触发最短成功路径，聚焦 runner 层对任何终止路径都能保证发布 finished 事件
async def test_run_finished_event_published_on_success(tmp_path: Path) -> None:
    events = await _run(tmp_path=tmp_path)
    finished = next(
        (e for e in events if e.type == "run.finished"), None  # type: ignore[attr-defined]
    )
    assert finished is not None
    assert finished.status == "success"  # type: ignore[attr-defined]


# 功能：验证步数耗尽时 run.finished 携带 failed 状态和正确的失败原因
# 设计：LoopingProvider + max_steps=2 触发失败路径，确认 runner 在失败终止路径同样发布 finished 事件
async def test_run_finished_event_published_on_max_steps(tmp_path: Path) -> None:
    events = await _run(
        provider=_LoopingProvider(),
        config=_config(max_steps=2),
        tmp_path=tmp_path,
    )
    finished = next(e for e in events if e.type == "run.finished")  # type: ignore[attr-defined]
    assert finished.status == "failed"  # type: ignore[attr-defined]
    assert finished.reason == "exceeded_max_steps"  # type: ignore[attr-defined]


# 功能：验证 events.v2.jsonl 第一条 domain event 为 run.started、最后一条为 run.finished
# 设计：解析真实 v2 wrapper 的 event 字段，证明 runner 已脱离 legacy writer 且 terminal barrier 完成
async def test_events_jsonl_created_with_started_and_finished(tmp_path: Path) -> None:
    await _run(tmp_path=tmp_path)
    jsonl_files = list(tmp_path.rglob("events.v2.jsonl"))
    assert len(jsonl_files) == 1
    lines = [json.loads(ln) for ln in jsonl_files[0].read_text().splitlines() if ln]
    event_types = [row["event"]["type"] for row in lines]
    assert event_types[0] == "run.started"
    assert event_types[-1] == "run.finished"


# 功能：验证 runner 在 runs_dir 下创建 run 子目录并写入唯一 events.v2.jsonl
# 设计：检查目录结构与 v2 文件名，防止生产路径继续双写 legacy events.jsonl
async def test_run_creates_run_subdirectory(tmp_path: Path) -> None:
    await _run(tmp_path=tmp_path)
    subdirs = [p for p in tmp_path.iterdir() if p.is_dir()]
    assert len(subdirs) == 1
    assert (subdirs[0] / "events.v2.jsonl").exists()
    assert not (subdirs[0] / "events.jsonl").exists()


# 功能：验证通过 extra_handlers 注入的回调能收到所有事件
# 设计：注入第二个收集器，确认 extra_handlers 机制有效；这是测试代码注入 mock 观察器、生产代码接入 StdoutPrinter 的同一扩展点
async def test_extra_handlers_receive_events(tmp_path: Path) -> None:
    secondary: list[BaseModel] = []

    async def _second(e: BaseModel) -> None:
        secondary.append(e)

    cfg = _config()
    runner = AgentRunner(
        cfg,
        workspace_root=tmp_path.resolve(),
        provider=_EndTurnProvider(),  # type: ignore[arg-type]
        extra_handlers=[_second],
        runs_dir=tmp_path,
    )
    await runner.run("goal")
    assert len(secondary) > 0


# 功能：验证 config.agent.max_steps 被正确传递给 AgentLoop，控制 LLM 调用次数上限
# 设计：用 LoopingProvider 的调用次数反推 max_steps 是否生效，不依赖内部状态检查，从行为角度验证配置传递
async def test_config_max_steps_passed_to_loop(tmp_path: Path) -> None:
    provider = _LoopingProvider()
    await _run(provider=provider, config=_config(max_steps=3), tmp_path=tmp_path)
    assert provider._call == 3


# 功能：验证 run.started 和 run.finished 事件使用相同且非空的 run_id
# 设计：同时检查两个事件的 run_id 字段，确认 runner 在整个 run 生命周期使用同一个 run_id
async def test_run_id_embedded_in_started_event(tmp_path: Path) -> None:
    events = await _run(tmp_path=tmp_path)
    started = next(e for e in events if e.type == "run.started")  # type: ignore[attr-defined]
    finished = next(e for e in events if e.type == "run.finished")  # type: ignore[attr-defined]
    assert started.run_id == finished.run_id  # type: ignore[attr-defined]
    assert len(started.run_id) > 0  # type: ignore[attr-defined]


# 功能：验证注入外部 EventBus 时，runner 使用该 bus 而不自建，外部订阅者能收到所有事件
# 设计：显式传入 EventBus 实例并订阅收集器，确认 runner 不再内部新建 bus（否则外部订阅者收不到事件）；
#       这是 CoreApp 注入全局 bus 的核心行为，单元测试级别验证可避免集成测试的守护进程依赖
async def test_injected_bus_receives_events(tmp_path: Path) -> None:
    from kama_claude.core.events.bus import EventBus

    external_bus = EventBus()
    collected: list[object] = []

    async def collect(e: object) -> None:
        collected.append(e)

    external_bus.subscribe(collect)

    runner = AgentRunner(
        _config(),
        workspace_root=tmp_path.resolve(),
        bus=external_bus,
        provider=_EndTurnProvider(),  # type: ignore[arg-type]
        runs_dir=tmp_path,
    )
    await runner.run("goal")

    types = [e.type for e in collected]  # type: ignore[attr-defined]
    assert "run.started" in types
    assert "run.finished" in types


# 功能：验证 session run 会从 thread.jsonl 预填 messages，并把 notes 注入 system prompt
# 设计：用 CapturingProvider 截获 LLM 入参，不触发真实 API；同时断言 run 目录写到 session/runs 下
async def test_session_history_and_notes_injected(tmp_path: Path) -> None:
    from kama_claude.core.session.model import Session
    from kama_claude.core.session.store import SessionStore

    store = SessionStore(tmp_path / "sessions")
    session = Session(
        id="sess-1",
        mode="chat",
        status="active",
        title="",
        created_at="t",
        updated_at="t",
        workspace_root=tmp_path.resolve(),
    )
    store.write_meta(session)
    store.append_message("sess-1", "user", "remember python")
    store.append_note("sess-1", "Python 3.12", "run-old")

    provider = _CapturingProvider(LlmResponse(stop_reason="end_turn", text="done"))
    runner = AgentRunner(
        _config(),
        workspace_root=session.workspace_root,
        provider=provider,
        runs_dir=tmp_path / "runs",
    )

    await runner.run_and_capture("remember python", run_id="run-new", session=session, store=store)

    assert provider.messages == [{"role": "user", "content": "remember python"}]
    assert provider.system is not None
    assert "Python 3.12" in provider.system
    assert (store.runs_dir("sess-1") / "run-new" / "events.v2.jsonl").exists()
    assert not (tmp_path / "runs" / "run-new").exists()


# 功能：验证 session run 中注册了 note_save，工具调用会写入 notes.md
# 设计：mock provider 第一步请求 note_save、第二步 end_turn，覆盖 runner→registry→tool invocation 的完整路径
async def test_session_registers_note_save_tool(tmp_path: Path) -> None:
    from kama_claude.core.session.model import Session
    from kama_claude.core.session.store import SessionStore

    class _NoteProvider:
        # 初始化调用计数器，用于返回两步响应
        def __init__(self) -> None:
            self.calls = 0

        # 第一步请求 note_save，第二步返回 end_turn
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
                            id="note-1",
                            name="note_save",
                            input={"content": "Use Python 3.12"},
                        )
                    ],
                )
            return LlmResponse(stop_reason="end_turn", text="noted")

    store = SessionStore(tmp_path / "sessions")
    session = Session(
        id="sess-1",
        mode="chat",
        status="active",
        title="",
        created_at="t",
        updated_at="t",
        workspace_root=tmp_path.resolve(),
    )
    store.append_message("sess-1", "user", "remember")

    runner = AgentRunner(
        _config(max_steps=3),
        workspace_root=session.workspace_root,
        provider=_NoteProvider(),
        runs_dir=tmp_path,
    )
    await runner.run_and_capture("remember", run_id="run-1", session=session, store=store)

    assert "Use Python 3.12" in store.read_notes("sess-1")


# 功能：验证 Runner 只加载显式 workspace 下的项目 context 而不读取 daemon cwd
# 设计：daemon A 与 workspace B 写入冲突内容，捕获 system prompt 并断言仅包含 B
async def test_project_context_uses_runner_workspace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon_cwd = tmp_path / "daemon"
    workspace = tmp_path / "workspace"
    (daemon_cwd / ".kama").mkdir(parents=True)
    (workspace / ".kama").mkdir(parents=True)
    (daemon_cwd / ".kama" / "context.md").write_text("daemon-context", encoding="utf-8")
    (workspace / ".kama" / "context.md").write_text("workspace-context", encoding="utf-8")
    monkeypatch.chdir(daemon_cwd)
    provider = _CapturingProvider(LlmResponse(stop_reason="end_turn", text="done"))
    runner = AgentRunner(
        _config(),
        workspace_root=workspace.resolve(),
        provider=provider,
        runs_dir=tmp_path / "runs",
    )

    await runner.run("inspect context")

    assert provider.system is not None
    assert "workspace-context" in provider.system
    assert "daemon-context" not in provider.system


# 功能：验证 workspace A/B 的 Runner 分别加载各自项目 context
# 设计：两个 capturing provider 独立运行，直接比较各自收到的 system prompt
async def test_project_context_isolated_between_runners(tmp_path: Path) -> None:
    workspace_a = tmp_path / "workspace-a"
    workspace_b = tmp_path / "workspace-b"
    for workspace, content in ((workspace_a, "context-a"), (workspace_b, "context-b")):
        kama_dir = workspace / ".kama"
        kama_dir.mkdir(parents=True)
        (kama_dir / "context.md").write_text(content, encoding="utf-8")
    provider_a = _CapturingProvider(LlmResponse(stop_reason="end_turn", text="done"))
    provider_b = _CapturingProvider(LlmResponse(stop_reason="end_turn", text="done"))

    for workspace, provider in ((workspace_a, provider_a), (workspace_b, provider_b)):
        runner = AgentRunner(
            _config(),
            workspace_root=workspace.resolve(),
            provider=provider,
            runs_dir=tmp_path / "runs",
        )
        await runner.run("inspect context")

    assert provider_a.system is not None
    assert provider_b.system is not None
    assert "context-a" in provider_a.system
    assert "context-b" not in provider_a.system
    assert "context-b" in provider_b.system
    assert "context-a" not in provider_b.system


class _CancellationProvider:
    # 初始化 provider entered gate 与捕获的 cancellation 引用
    def __init__(self) -> None:
        self.entered = asyncio.Event()
        self.cancelled: asyncio.CancelledError | None = None

    # 阻塞 LLM 调用直到 runner task 被取消并保存同一异常对象
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
        self.entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError as exc:
            self.cancelled = exc
            raise
        raise AssertionError("unreachable")


# 功能：验证 terminal journal failure 不覆盖 runner 原始 cancellation identity
# 设计：先 flush run.started，再让仅含 run.finished 的 batch 失败，比较 provider 与调用方捕获对象 is 相同
async def test_cancellation_identity_survives_terminal_journal_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _CancellationProvider()
    runner = AgentRunner(
        _config(),
        workspace_root=tmp_path.resolve(),
        provider=provider,
        runs_dir=tmp_path / "runs",
    )
    task = asyncio.create_task(runner.run("cancel me", run_id="run-cancel"))
    await provider.entered.wait()
    await runner._journal.flush_all()
    original = EventWriter.append_and_flush

    # 只让 terminal batch 失败，保留 run.started 的既有 durable prefix
    def fail_terminal(writer: EventWriter, rows: Iterable[bytes]) -> None:
        materialized = tuple(rows)
        if any(b'"type":"run.finished"' in row for row in materialized):
            raise OSError("terminal-disk-secret")
        original(writer, materialized)

    monkeypatch.setattr(EventWriter, "append_and_flush", fail_terminal)
    task.cancel("primary-run-cancel")

    try:
        await task
    except asyncio.CancelledError as caught:
        observed = caught
    else:
        raise AssertionError("runner cancellation did not propagate")

    assert provider.cancelled is observed


# 读取 v2 wrappers 中的 domain event 字段供 lifecycle 顺序断言
def _read_v2_events(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)["event"]
        for line in path.read_text(encoding="utf-8").splitlines()
    ]


# 功能：验证provider异常run在真实Runner与journal中先闭合step再发布failed terminal
# 设计：provider发布model event后抛错，读取真实v2 journal并交给complete grader验证精确序列
async def test_provider_error_persists_complete_failed_lifecycle(
    tmp_path: Path,
) -> None:
    run_id = "run-provider-error"
    runs_dir = tmp_path / "runs"
    runner = AgentRunner(
        _config(),
        workspace_root=tmp_path.resolve(),
        provider=_ModelThenErrorProvider(),  # type: ignore[arg-type]
        runs_dir=runs_dir,
    )

    outcome = await runner.run_and_capture("fail locally", run_id=run_id)
    journal_path = runs_dir / run_id / "events.v2.jsonl"
    events = _read_v2_events(journal_path)
    grade = grade_trace(journal_path, expected_run_id=run_id)

    assert [event["type"] for event in events] == [
        "run.started",
        "step.started",
        "llm.model_selected",
        "step.finished",
        "run.finished",
    ]
    assert events[-1]["status"] == "failed"
    assert events[-1]["reason"] == "llm_error"
    assert events[-1]["steps"] == 1
    assert outcome.status == "failed"
    assert outcome.reason == "llm_error"
    assert grade.passed is True
    assert grade.errors == []


# 功能：验证无primary的step delivery failure阻止Runner提交success并保留fail-closed journal
# 设计：end_turn先形成pending success，再由journal前subscriber破坏finish，检查最终outcome与真实grader
async def test_step_delivery_failure_prevents_runner_success(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    run_id = "run-step-delivery-failure"
    runs_dir = tmp_path / "runs"
    bus = EventBus()
    attempts = 0
    finalizer_sentinel = "FINALIZER_ONLY_SECRET_SENTINEL_DO_NOT_LOG"

    # 只阻断step terminal到达journal，允许Runner terminal继续持久化
    async def fail_before_journal(event: BaseModel) -> None:
        nonlocal attempts
        if not isinstance(event, StepFinishedEvent):
            return
        attempts += 1
        raise RuntimeError(finalizer_sentinel)

    bus.subscribe(fail_before_journal)
    runner = AgentRunner(
        _config(),
        workspace_root=tmp_path.resolve(),
        bus=bus,
        provider=_EndTurnProvider(),  # type: ignore[arg-type]
        runs_dir=runs_dir,
    )

    outcome = await runner.run_and_capture("finish locally", run_id=run_id)
    journal_path = runs_dir / run_id / "events.v2.jsonl"
    events = _read_v2_events(journal_path)
    grade = grade_trace(journal_path, expected_run_id=run_id)
    runner_records = [
        record for record in caplog.records if record.name == "kama_claude.core.runner"
    ]

    assert attempts == 1
    assert outcome.status == "failed"
    assert outcome.result == ""
    assert outcome.reason == "llm_error"
    assert [event["type"] for event in events] == [
        "run.started",
        "step.started",
        "run.finished",
    ]
    assert events[-1]["status"] == "failed"
    assert events[-1]["reason"] == "llm_error"
    assert grade.passed is False
    assert grade.errors == ["run finished with open step"]
    assert len(runner_records) == 1
    assert runner_records[0].getMessage() == (
        "agent run failed run_id=run-step-delivery-failure step=1 "
        "failure_role=primary failure_category=propagated_exception"
    )
    assert runner_records[0].exc_info is None
    assert runner_records[0].exc_text in (None, "")
    assert finalizer_sentinel not in caplog.text
    assert "Traceback" not in caplog.text
    assert "RuntimeError" not in caplog.text
    assert str(Path(__file__).resolve()) not in caplog.text


# 功能：验证provider primary遇到step delivery secondary时保持llm_error且grader对缺帧fail closed
# 设计：把失败subscriber注册在Runner自有journal之前，组合验证primary优先级、单次发布和真实持久化证据
async def test_provider_primary_with_step_delivery_failure_fails_trace_closed(
    tmp_path: Path,
) -> None:
    run_id = "run-provider-primary-step-secondary"
    runs_dir = tmp_path / "runs"
    bus = EventBus()
    attempts = 0

    # 只在step terminal阻断后续journal subscriber，保留其他事件与run terminal持久化
    async def fail_before_journal(event: BaseModel) -> None:
        nonlocal attempts
        if not isinstance(event, StepFinishedEvent):
            return
        attempts += 1
        raise RuntimeError("secondary step delivery failure")

    bus.subscribe(fail_before_journal)
    runner = AgentRunner(
        _config(),
        workspace_root=tmp_path.resolve(),
        bus=bus,
        provider=_ModelThenErrorProvider(),  # type: ignore[arg-type]
        runs_dir=runs_dir,
    )

    outcome = await runner.run_and_capture("fail locally", run_id=run_id)
    journal_path = runs_dir / run_id / "events.v2.jsonl"
    events = _read_v2_events(journal_path)
    grade = grade_trace(journal_path, expected_run_id=run_id)

    assert attempts == 1
    assert outcome.status == "failed"
    assert outcome.reason == "llm_error"
    assert [event["type"] for event in events] == [
        "run.started",
        "step.started",
        "llm.model_selected",
        "run.finished",
    ]
    assert events[-1]["status"] == "failed"
    assert events[-1]["reason"] == "llm_error"
    assert grade.passed is False
    assert grade.errors == ["run finished with open step"]


# 功能：验证tool-path普通异常与step delivery失败组合时Runner仍映射llm_error且grader fail closed
# 设计：在真实ToolCallStartedEvent与StepFinishedEvent上依次抛primary/secondary，读取真实journal
async def test_tool_path_primary_with_step_delivery_failure_fails_trace_closed(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    run_id = "run-tool-primary-step-secondary"
    runs_dir = tmp_path / "runs"
    bus = EventBus()
    primary = RuntimeError("TOOL_PRIMARY_SENTINEL_DO_NOT_LOG")
    secondary = RuntimeError("FINALIZER_SECRET_SENTINEL_DO_NOT_LOG")
    finish_attempts = 0

    # 先在真实tool invocation事件抛primary，再在唯一step terminal抛secondary
    async def fail_tool_and_step(event: BaseModel) -> None:
        nonlocal finish_attempts
        if isinstance(event, ToolCallStartedEvent):
            raise primary
        if isinstance(event, StepFinishedEvent):
            finish_attempts += 1
            raise secondary

    bus.subscribe(fail_tool_and_step)
    runner = AgentRunner(
        _config(),
        workspace_root=tmp_path.resolve(),
        bus=bus,
        provider=_LoopingProvider(),  # type: ignore[arg-type]
        runs_dir=runs_dir,
    )

    outcome = await runner.run_and_capture("fail in tool path", run_id=run_id)
    journal_path = runs_dir / run_id / "events.v2.jsonl"
    events = _read_v2_events(journal_path)
    grade = grade_trace(journal_path, expected_run_id=run_id)
    await asyncio.sleep(0)
    orphan_publications = [
        task
        for task in asyncio.all_tasks()
        if task is not asyncio.current_task()
        and not task.done()
        and task.get_coro().__qualname__ == "EventBus.publish"
    ]
    runner_records = [
        record for record in caplog.records if record.name == "kama_claude.core.runner"
    ]

    assert finish_attempts == 1
    assert outcome.status == "failed"
    assert outcome.result == ""
    assert outcome.reason == "llm_error"
    assert [event["type"] for event in events] == [
        "run.started",
        "step.started",
        "run.finished",
    ]
    assert events[-1]["status"] == "failed"
    assert events[-1]["reason"] == "llm_error"
    assert grade.passed is False
    assert grade.errors == ["run finished with open step"]
    assert orphan_publications == []
    assert len(runner_records) == 1
    assert runner_records[0].getMessage() == (
        "agent run failed run_id=run-tool-primary-step-secondary step=1 "
        "failure_role=primary failure_category=propagated_exception"
    )
    assert runner_records[0].exc_info is None
    assert runner_records[0].exc_text in (None, "")
    assert "TOOL_PRIMARY_SENTINEL_DO_NOT_LOG" not in caplog.text
    assert "FINALIZER_SECRET_SENTINEL_DO_NOT_LOG" not in caplog.text
    assert "Traceback" not in caplog.text
    assert "RuntimeError" not in caplog.text
    assert str(Path(__file__).resolve()) not in caplog.text


# 功能：验证finalizer期间取消会完成同一次journal publication再发布cancelled run terminal
# 设计：把阻塞subscriber注册在journal之前，取消Runner后释放，确保shield而非subscriber顺序保护生命周期
async def test_finalizer_cancellation_finishes_step_before_run_terminal(
    tmp_path: Path,
) -> None:
    run_id = "run-finalizer-cancel"
    runs_dir = tmp_path / "runs"
    bus = EventBus()
    journal = EventJournalCoordinator()
    entered = asyncio.Event()
    release = asyncio.Event()
    completed = asyncio.Event()
    attempts = 0

    # 在journal之前阻塞step terminal，制造publication尚未到达durable subscriber的窗口
    async def block_before_journal(event: BaseModel) -> None:
        nonlocal attempts
        if not isinstance(event, StepFinishedEvent):
            return
        attempts += 1
        entered.set()
        await release.wait()
        completed.set()

    bus.subscribe(block_before_journal)
    bus.subscribe(journal.handle)
    runner = AgentRunner(
        _config(),
        workspace_root=tmp_path.resolve(),
        bus=bus,
        provider=_EndTurnProvider(),  # type: ignore[arg-type]
        runs_dir=runs_dir,
        journal=journal,
    )
    observed_in_runner: asyncio.CancelledError | None = None

    # 保存Runner边界传播出的取消对象供外层做identity比较
    async def observe_runner_cancellation() -> None:
        nonlocal observed_in_runner
        try:
            await runner.run_and_capture("cancel finalizer", run_id=run_id)
        except asyncio.CancelledError as exc:
            observed_in_runner = exc
            raise

    task = asyncio.create_task(observe_runner_cancellation())
    await entered.wait()
    task.cancel("cancel-finalizer-runner")
    await asyncio.sleep(0)
    release.set()

    try:
        await task
    except asyncio.CancelledError as caught:
        observed_by_caller = caught
    else:
        raise AssertionError("runner cancellation did not propagate")
    await journal.flush_all()
    await journal.close()

    journal_path = runs_dir / run_id / "events.v2.jsonl"
    events = _read_v2_events(journal_path)
    grade = grade_trace(journal_path, expected_run_id=run_id)

    assert attempts == 1
    assert completed.is_set()
    assert observed_in_runner is observed_by_caller
    assert observed_by_caller.args == ("cancel-finalizer-runner",)
    assert [event["type"] for event in events] == [
        "run.started",
        "step.started",
        "step.finished",
        "run.finished",
    ]
    assert events[-1]["status"] == "failed"
    assert events[-1]["reason"] == "cancelled"
    assert events[-1]["steps"] == 1
    assert grade.passed is True
    assert grade.errors == []


# 功能：验证finalizer阻塞期间重复取消仍复用一次publication并传播provider捕获的首个对象
# 设计：真实Runner/journal链路执行三次cancel，在释放subscriber后检查identity、顺序与无orphan
async def test_repeated_cancellation_during_step_finalization_reuses_one_publication(
    tmp_path: Path,
) -> None:
    run_id = "run-repeated-finalizer-cancel"
    runs_dir = tmp_path / "runs"
    bus = EventBus()
    journal = EventJournalCoordinator()
    provider = _CancellationProvider()
    finalizer_entered = asyncio.Event()
    release = asyncio.Event()
    completed = asyncio.Event()
    finish_attempts = 0
    loop_errors: list[dict[str, Any]] = []
    running_loop = asyncio.get_running_loop()
    previous_handler = running_loop.get_exception_handler()

    # 捕获未retrieve task异常，确保重复取消不会遗留finalizer warning
    def capture_loop_error(
        event_loop: asyncio.AbstractEventLoop,
        context: dict[str, Any],
    ) -> None:
        loop_errors.append(context)

    # 在journal之前阻塞唯一step terminal以建立两次secondary cancellation窗口
    async def block_before_journal(event: BaseModel) -> None:
        nonlocal finish_attempts
        if not isinstance(event, StepFinishedEvent):
            return
        finish_attempts += 1
        finalizer_entered.set()
        await release.wait()
        completed.set()

    bus.subscribe(block_before_journal)
    bus.subscribe(journal.handle)
    runner = AgentRunner(
        _config(),
        workspace_root=tmp_path.resolve(),
        bus=bus,
        provider=provider,
        runs_dir=runs_dir,
        journal=journal,
    )
    observed_in_runner: asyncio.CancelledError | None = None

    # 保存Runner边界传播对象，验证它仍是provider第一次捕获的X
    async def observe_runner_cancellation() -> None:
        nonlocal observed_in_runner
        try:
            await runner.run_and_capture("cancel repeatedly", run_id=run_id)
        except asyncio.CancelledError as exc:
            observed_in_runner = exc
            raise

    running_loop.set_exception_handler(capture_loop_error)
    try:
        task = asyncio.create_task(observe_runner_cancellation())
        await provider.entered.wait()
        assert task.cancel("primary-provider-cancel") is True
        await finalizer_entered.wait()
        assert task.cancel("secondary-finalizer-cancel-1") is True
        await asyncio.sleep(0)
        assert task.cancel("secondary-finalizer-cancel-2") is True
        cancellation_count = task.cancelling()
        await asyncio.sleep(0)
        release.set()

        try:
            await task
        except asyncio.CancelledError as caught:
            observed_by_caller = caught
        else:
            raise AssertionError("repeated cancellation did not propagate")
        await journal.flush_all()
        await journal.close()
        await asyncio.sleep(0)
    finally:
        running_loop.set_exception_handler(previous_handler)

    journal_path = runs_dir / run_id / "events.v2.jsonl"
    events = _read_v2_events(journal_path)
    grade = grade_trace(journal_path, expected_run_id=run_id)
    orphan_publications = [
        pending
        for pending in asyncio.all_tasks()
        if pending is not asyncio.current_task()
        and not pending.done()
        and pending.get_coro().__qualname__ == "EventBus.publish"
    ]

    assert cancellation_count >= 3
    assert finish_attempts == 1
    assert completed.is_set()
    assert provider.cancelled is observed_in_runner
    assert observed_in_runner is observed_by_caller
    assert observed_by_caller.args == ("primary-provider-cancel",)
    assert [event["type"] for event in events] == [
        "run.started",
        "step.started",
        "step.finished",
        "run.finished",
    ]
    assert events[-1]["status"] == "failed"
    assert events[-1]["reason"] == "cancelled"
    assert events[-1]["steps"] == 1
    assert grade.passed is True
    assert grade.errors == []
    assert orphan_publications == []
    assert not any(
        context.get("message") == "Task exception was never retrieved"
        for context in loop_errors
    )


# 功能：验证 parent run terminal 前会 cancel/join 所有 background child 并先持久化 child finished
# 设计：让 parent loop 启动真实 background SpawnAgentTool 后立即成功、child 永久等待，比较 task 终态和 parent journal 顺序
async def test_parent_run_joins_background_child_before_terminal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    child_entered = asyncio.Event()
    never_release = asyncio.Event()

    # parent 真实调用 spawn_agent；child 只在可取消 gate 上等待
    async def controlled_run(loop: AgentLoop, context: Any) -> None:
        if context.run_id == "run-parent":
            tool = loop._registry.get("spawn_agent")
            assert isinstance(tool, SpawnAgentTool)
            await tool.invoke(
                {
                    "description": "background child",
                    "prompt": "wait for parent cleanup",
                    "run_in_background": True,
                }
            )
            context.mark_success()
            return
        child_entered.set()
        await never_release.wait()

    monkeypatch.setattr(AgentLoop, "run", controlled_run)
    runner = AgentRunner(
        _config(),
        workspace_root=tmp_path.resolve(),
        provider=_EndTurnProvider(),
        runs_dir=tmp_path / "runs",
    )
    await runner.run("parent", run_id="run-parent")
    await child_entered.wait()
    entries = runner._task_registry.all()
    assert len(entries) == 1
    child_task, child_context = entries[0]
    try:
        assert child_task.done()
        assert child_context.reason == "cancelled"
        child_rows = _read_v2_events(
            tmp_path / "runs" / child_context.run_id / "events.v2.jsonl"
        )
        parent_rows = _read_v2_events(
            tmp_path / "runs" / "run-parent" / "events.v2.jsonl"
        )
        assert child_rows[-1]["type"] == "subagent.finished"
        parent_types = [event["type"] for event in parent_rows]
        assert parent_types.index("subagent.finished") < parent_types.index("run.finished")
    finally:
        if not child_task.done():
            child_task.cancel()
        await asyncio.gather(child_task, return_exceptions=True)


# --- git lifecycle hooks (P4) -------------------------------------------------


class _FakeGitManager:
    # 记录 git 钩子调用；config 复用真实 GitConfig 控制行为开关
    def __init__(
        self,
        *,
        dirty: bool = False,
        checkpoint_mode: str = "per_run",
        auto_rollback_on_fail: bool = False,
        ensure_fails: bool = False,
    ) -> None:
        self.config = GitConfig(
            checkpoint_mode=checkpoint_mode,
            auto_rollback_on_fail=auto_rollback_on_fail,
        )
        self.calls: list[str] = []
        self.dirty = dirty
        self.ensure_fails = ensure_fails

    async def ensure_ready(self) -> None:
        self.calls.append("ensure_ready")
        if self.ensure_fails:
            raise GitUnavailableError("no git")

    async def status(self) -> object:
        self.calls.append("status")
        return type("Status", (), {"dirty": self.dirty, "entries": ()})()

    async def snapshot_pre_run(self, run_id: str, label: str = "pre-run") -> None:
        self.calls.append(f"snapshot_pre_run:{run_id}")
        return None

    async def ensure_task_branch(self, task_id: str) -> None:
        self.calls.append(f"ensure_task_branch:{task_id}")

    async def create_checkpoint(
        self, run_id: str, step: int, label: str, *, force: bool = False
    ) -> object:
        self.calls.append(f"create_checkpoint:{run_id}:{step}:{label}")
        return type("Cp", (), {"step": step})()

    async def get_checkpoint(self, run_id: str, step: int) -> object:
        self.calls.append(f"get_checkpoint:{run_id}:{step}")
        if step == 0:
            return type("Cp", (), {"step": 0})()
        return None

    async def restore(self, cp: object) -> None:
        self.calls.append(f"restore:{getattr(cp, 'step', '?')}")

    async def diff(self, ref: str | None = None) -> GitDiff:
        self.calls.append("diff")
        return GitDiff(stat="f.txt | 1 +\n", truncated=False)

    async def close(self) -> None:
        self.calls.append("close")


class _FakePermissionManager:
    # 记录审批请求并返回预设结果，替代真实用户应答
    def __init__(self, allowed: bool) -> None:
        self.allowed = allowed
        self.asked: list[tuple[str, dict[str, Any]]] = []

    async def check_and_wait(
        self,
        tool_use_id: str,
        tool_name: str,
        params: dict[str, Any],
        session_id: str,
        event_emitter: Any,
    ) -> tuple[bool, str]:
        self.asked.append((tool_name, params))
        return self.allowed, ("allow" if self.allowed else "deny")


class _OnceToolUseProvider:
    # 第一步返回未知工具调用（触发失败），第二步结束
    def __init__(self) -> None:
        self._call = 0

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
        self._call += 1
        if self._call == 1:
            tc = ToolCallBlock(id="t1", name="unknown_tool", input={})
            return LlmResponse(stop_reason="tool_use", tool_calls=[tc])
        return LlmResponse(stop_reason="end_turn", text="done")


class _CancelProvider:
    # 模拟取消：首次 LLM 调用直接抛 CancelledError
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
        raise asyncio.CancelledError()


def _git_runner(
    tmp_path: Path,
    git_manager: _FakeGitManager | None,
    *,
    provider: object | None = None,
    permission_manager: object | None = None,
) -> tuple[AgentRunner, list[BaseModel]]:
    collected: list[BaseModel] = []

    async def _collect(e: BaseModel) -> None:
        collected.append(e)

    runner = AgentRunner(
        _config(),
        workspace_root=tmp_path.resolve(),
        provider=provider or _EndTurnProvider(),  # type: ignore[arg-type]
        extra_handlers=[_collect],
        runs_dir=tmp_path / "runs",
        permission_manager=permission_manager,  # type: ignore[arg-type]
        git_manager=git_manager,  # type: ignore[arg-type]
    )
    return runner, collected


# 功能：验证 run start 自动建立 baseline（ensure_ready → status → task 分支 → baseline checkpoint）
# 设计：干净工作树 + fake manager，断言调用顺序与 run end diff 事件发布
async def test_run_start_creates_baseline_in_order(tmp_path: Path) -> None:
    git = _FakeGitManager()
    runner, collected = _git_runner(tmp_path, git)
    outcome = await runner.run_and_capture("goal", run_id="r1")
    assert outcome.status == "success"
    assert git.calls == [
        "ensure_ready",
        "status",
        "ensure_task_branch:r1",
        "create_checkpoint:r1:0:baseline",
        "diff",
    ]
    assert any(isinstance(e, GitRunDiffEvent) for e in collected)


# 功能：验证 dirty 工作树且用户批准时调用 snapshot_pre_run（含用户修改的 baseline）
# 设计：dirty=True + 批准 → ASK 记录与 snapshot 调用，且仍建立 task 分支与 baseline
async def test_dirty_run_approves_snapshot(tmp_path: Path) -> None:
    git = _FakeGitManager(dirty=True)
    perm = _FakePermissionManager(allowed=True)
    runner, _collected = _git_runner(tmp_path, git, permission_manager=perm)
    await runner.run_and_capture("goal", run_id="r1")
    assert perm.asked and perm.asked[0][0] == "git_pre_run_snapshot"
    assert "snapshot_pre_run:r1" in git.calls
    assert "ensure_task_branch:r1" in git.calls
    assert "create_checkpoint:r1:0:baseline" in git.calls


# 功能：验证 dirty 工作树且用户拒绝时抛 DirtyWorkspaceError 且不触碰工作树
# 设计：dirty=True + 拒绝 → run 抛错，git 调用止于 status（无分支/checkpoint）
async def test_dirty_run_declined_raises(tmp_path: Path) -> None:
    git = _FakeGitManager(dirty=True)
    perm = _FakePermissionManager(allowed=False)
    runner, _collected = _git_runner(tmp_path, git, permission_manager=perm)
    with pytest.raises(DirtyWorkspaceError):
        await runner.run_and_capture("goal", run_id="r1")
    assert "ensure_task_branch:r1" not in git.calls
    assert "create_checkpoint" not in "".join(git.calls)


# 功能：验证 run 失败且 auto_rollback_on_fail 时恢复到 baseline
# 设计：provider 抛错使 run 标记 failed → restore(baseline) 被调用
async def test_run_failure_auto_rollback(tmp_path: Path) -> None:
    git = _FakeGitManager(auto_rollback_on_fail=True)
    runner, _collected = _git_runner(tmp_path, git, provider=_ModelThenErrorProvider())
    outcome = await runner.run_and_capture("goal", run_id="r1")
    assert outcome.status == "failed"
    assert "restore:0" in git.calls


# 功能：验证取消时不做自动 rollback（用户可能正在查看，refs 已持久化）
# 设计：auto_rollback 开启 + CancelledError → run 抛 CancelledError 且 restore 未被调用
async def test_cancelled_run_skips_auto_rollback(tmp_path: Path) -> None:
    git = _FakeGitManager(auto_rollback_on_fail=True)
    runner, _collected = _git_runner(tmp_path, git, provider=_CancelProvider())
    with pytest.raises(asyncio.CancelledError):
        await runner.run_and_capture("goal", run_id="r1")
    assert "restore" not in "".join(git.calls)


# 功能：验证非 git 仓库 fail-open——run 继续但无 git 能力
# 设计：ensure_ready 抛 GitUnavailableError → run 成功且无分支/checkpoint/diff
async def test_non_repo_run_fails_open(tmp_path: Path) -> None:
    git = _FakeGitManager(ensure_fails=True)
    runner, collected = _git_runner(tmp_path, git)
    outcome = await runner.run_and_capture("goal", run_id="r1")
    assert outcome.status == "success"
    assert git.calls == ["ensure_ready"]
    assert not any(isinstance(e, GitRunDiffEvent) for e in collected)


# 功能：验证 per_step 模式下每个 step 结束自动 checkpoint
# 设计：两步 run（tool_use + end_turn）→ auto-step-1 与 auto-step-2 各一次
async def test_per_step_checkpoint_on_step_finished(tmp_path: Path) -> None:
    git = _FakeGitManager(checkpoint_mode="per_step")
    runner, _collected = _git_runner(tmp_path, git, provider=_OnceToolUseProvider())
    await runner.run_and_capture("goal", run_id="r1")
    assert "create_checkpoint:r1:1:auto-step-1" in git.calls
    assert "create_checkpoint:r1:2:auto-step-2" in git.calls


# 功能：验证 git_manager 存在时注册 5 个 git 工具，缺失时不注册
# 设计：_build_registry 带 run_id 断言工具齐全；无 manager 断言 git_commit 缺失
def test_registry_includes_git_tools_when_manager_present(tmp_path: Path) -> None:
    runner, _collected = _git_runner(tmp_path, _FakeGitManager())
    registry = runner._build_registry(TaskManager(tmp_path / "t" / ".tasks"), run_id="r1")
    for name in (
        "git_status",
        "git_diff",
        "git_checkpoint",
        "git_commit",
        "git_rollback",
    ):
        assert registry.get(name) is not None

    plain, _plain_collected = _git_runner(tmp_path, None)
    plain_registry = plain._build_registry(
        TaskManager(tmp_path / "t2" / ".tasks"), run_id="r1"
    )
    assert plain_registry.get("git_commit") is None
