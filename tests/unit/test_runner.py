from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import BaseModel

from kama_claude.core.config import KamaConfig
from kama_claude.core.events.bus import EventBus
from kama_claude.core.llm.types import LlmResponse, ToolCallBlock
from kama_claude.core.runner import AgentRunner
from kama_claude.core.task.manager import TaskManager
from kama_claude.core.tools.builtin.bash import BashTool
from kama_claude.core.tools.builtin.list_dir import ListDirTool
from kama_claude.core.tools.builtin.read_file import ReadFileTool
from kama_claude.core.tools.builtin.search_code import SearchCodeTool
from kama_claude.core.tools.builtin.write_file import WriteFileTool

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


# 功能：验证 events.jsonl 第一行为 run.started、最后一行为 run.finished
# 设计：从 tmp_path 递归查找 events.jsonl 并按行解析，因为 events.jsonl 是 S1 的核心产物，首尾事件是完整性的最低要求
async def test_events_jsonl_created_with_started_and_finished(tmp_path: Path) -> None:
    await _run(tmp_path=tmp_path)
    jsonl_files = list(tmp_path.rglob("events.jsonl"))
    assert len(jsonl_files) == 1
    lines = [json.loads(ln) for ln in jsonl_files[0].read_text().splitlines() if ln]
    event_types = [e["type"] for e in lines]
    assert event_types[0] == "run.started"
    assert event_types[-1] == "run.finished"


# 功能：验证 runner 在 runs_dir 下创建以 run_id 命名的子目录并写入 events.jsonl
# 设计：检查 tmp_path 下只有一个子目录且该目录包含 events.jsonl，确认目录结构约定（runs/<run_id>/events.jsonl）
async def test_run_creates_run_subdirectory(tmp_path: Path) -> None:
    await _run(tmp_path=tmp_path)
    subdirs = [p for p in tmp_path.iterdir() if p.is_dir()]
    assert len(subdirs) == 1
    assert (subdirs[0] / "events.jsonl").exists()


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
    assert (store.runs_dir("sess-1") / "run-new" / "events.jsonl").exists()
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
