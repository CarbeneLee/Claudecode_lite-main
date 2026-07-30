from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest

from kama_claude.core.bus.envelope import HandlerError
from kama_claude.core.config import KamaConfig
from kama_claude.core.events.bus import EventBus
from kama_claude.core.llm.types import LlmResponse
from kama_claude.core.runner import AgentRunner, RunOutcome
from kama_claude.core.session.manager import SESSION_CLOSED, SESSION_NOT_FOUND, SessionManager
from kama_claude.core.session.model import Session
from kama_claude.core.session.store import SessionStore

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


class _Runner:
    # 初始化 prompt override 与 goal 观测记录
    def __init__(self) -> None:
        self.seen_goals: list[str] = []
        self.seen_system_prompt_overrides: list[str | None] = []

    # 模拟 AgentRunner，将 run 新消息写入 thread 后返回成功
    async def run_and_capture(
        self,
        goal: str,
        *,
        run_id: str | None = None,
        session: Session | None = None,
        store: SessionStore | None = None,
        system_prompt_override: str | None = None,
        tool_whitelist: list[str] | None = None,
    ) -> RunOutcome:
        assert run_id is not None
        assert session is not None
        assert store is not None
        self.seen_goals.append(goal)
        self.seen_system_prompt_overrides.append(system_prompt_override)
        store.append_messages(
            session.id,
            [{"role": "assistant", "content": [{"type": "text", "text": f"done {goal}"}]}],
            run_id,
        )
        return RunOutcome(status="success", result="done", reason=None)


class _RecordingJournal:
    # 初始化 lifecycle 调用顺序记录
    def __init__(self, order: list[str]) -> None:
        self.order = order

    # 记录 session stream owner 在首事件前注册
    async def register_session(self, session_id: str, session_path: Path) -> object:
        self.order.append(f"register:session:{session_id}")
        return object()

    # 记录 run stream 与 parent session mapping 在首 run 事件前注册
    async def register_run(
        self,
        run_id: str,
        run_path: Path,
        *,
        session_id: str | None,
    ) -> object:
        self.order.append(f"register:run:{run_id}:{session_id}")
        return object()


class _SessionPromptProvider:
    # 初始化正常 session run 的真实 LLM 输入观测
    def __init__(self) -> None:
        self.messages: list[dict[str, object]] = []
        self.system: str | None = None

    # 捕获 AgentRunner 传入的真实 session messages 与 system 后本地结束
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
        self.messages = [dict(message) for message in messages]
        self.system = system
        return LlmResponse(stop_reason="end_turn", text="done")


# 功能：验证 create 会创建 active session、写入 meta 并发布 session.created 事件
# 设计：用真实 SessionStore + EventBus 收集事件，覆盖 manager 与 store/bus 的协作边界
async def test_create_session_writes_meta_and_event(tmp_path: Path) -> None:
    events: list[object] = []
    bus = EventBus()

    async def collect(event: object) -> None:
        events.append(event)

    bus.subscribe(collect)
    store = SessionStore(tmp_path)
    manager = SessionManager(
        store,
        lambda _workspace_root: _Runner(),
        bus,
    )  # type: ignore[arg-type]

    workspace_root = tmp_path.resolve()
    session = await manager.create("chat", "title", workspace_root=workspace_root)

    assert session.status == "active"
    assert session.workspace_root == workspace_root
    assert store.read_meta(session.id).title == "title"
    assert store.read_meta(session.id).workspace_root == workspace_root
    assert [e.type for e in events] == ["session.created"]  # type: ignore[attr-defined]


# 功能：验证 session stream owner 注册严格早于 session.created 发布与 meta 后续使用
# 设计：journal 与 bus 共享顺序列表，比较注册和真实 event handler 的先后而非 mock 调用次数
async def test_session_stream_registers_before_session_created_event(tmp_path: Path) -> None:
    order: list[str] = []
    bus = EventBus()

    # 把首个 session event 写入共享顺序记录
    async def collect(event: Any) -> None:
        order.append(f"event:{event.type}")

    bus.subscribe(collect)
    manager = SessionManager(
        SessionStore(tmp_path / "sessions"),
        cast(Any, lambda _workspace_root: _Runner()),
        bus,
        journal=cast(Any, _RecordingJournal(order)),
    )

    session = await manager.create("chat", workspace_root=tmp_path.resolve())

    assert order.index(f"register:session:{session.id}") < order.index(
        "event:session.created"
    )


# 功能：验证 slash skill 的 run stream/mapping 注册严格早于 skill.invoked
# 设计：使用真实 skill loader 触发第一个 run-bound event，并在共享顺序列表中比较生命周期边界
async def test_run_stream_registers_before_skill_invoked_event(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    skill_dir = workspace / ".kama" / "skills"
    skill_dir.mkdir(parents=True)
    (skill_dir / "demo.md").write_text(
        "---\nname: demo\ndescription: demo\n---\nDo $ARGUMENTS\n",
        encoding="utf-8",
    )
    order: list[str] = []
    bus = EventBus()

    # 把首个 run-bound skill event 写入共享顺序记录
    async def collect(event: Any) -> None:
        order.append(f"event:{event.type}")

    bus.subscribe(collect)
    store = SessionStore(tmp_path / "sessions")
    manager = SessionManager(
        store,
        cast(Any, lambda _workspace_root: _Runner()),
        bus,
        journal=cast(Any, _RecordingJournal(order)),
    )
    session = await manager.create("chat", workspace_root=workspace.resolve())

    run_id = await manager.send_message(session.id, "/demo now")

    assert order.index(f"register:run:{run_id}:{session.id}") < order.index(
        "event:skill.invoked"
    )


# 功能：验证正常 session 通过真实 Runner/Loop 继承一次 v1 且不包含 v2
# 设计：仅在 provider seam 使用本地 fake，保留 SessionManager、AgentRunner 与 AgentLoop 的真实组合路径
async def test_normal_session_inherits_v1_and_excludes_v2(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    provider = _SessionPromptProvider()
    config = KamaConfig()
    config.agent.max_steps = 2
    store = SessionStore(tmp_path / "sessions")

    # 为 SessionManager 构造真实 AgentRunner，仅替换外部 provider
    def runner_factory(workspace_root: Path) -> AgentRunner:
        return AgentRunner(
            config,
            workspace_root=workspace_root,
            provider=provider,  # type: ignore[arg-type]
            runs_dir=tmp_path / "runs",
        )

    manager = SessionManager(store, runner_factory, EventBus())
    session = await manager.create("chat", workspace_root=workspace.resolve())

    await manager.send_message(session.id, "Implement behavior A.")

    assert provider.messages == [{"role": "user", "content": "Implement behavior A."}]
    assert provider.system is not None
    assert provider.system.count(_REQUIREMENT_CONTRACT) == 1
    assert provider.system.count(_STATE_TRANSITION_PROTOCOL) == 0


# 功能：验证 slash skill 传入的 system prompt override 不包含 default v1/v2
# 设计：让 SessionManager 解析真实项目 skill，并由 recording runner 检查 override 的完整字节
async def test_slash_skill_override_excludes_default_v1_and_v2(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    skill_dir = workspace / ".kama" / "skills"
    skill_dir.mkdir(parents=True)
    (skill_dir / "custom.md").write_text(
        "---\nname: custom\ndescription: custom\n---\nCustom override $ARGUMENTS\n",
        encoding="utf-8",
    )
    runner = _Runner()
    manager = SessionManager(
        SessionStore(tmp_path / "sessions"),
        lambda _workspace_root: runner,  # type: ignore[arg-type]
        EventBus(),
    )
    session = await manager.create("chat", workspace_root=workspace.resolve())

    await manager.send_message(session.id, "/custom alpha")

    assert runner.seen_goals == ["Custom override alpha"]
    assert runner.seen_system_prompt_overrides == ["Custom override $ARGUMENTS"]
    override = runner.seen_system_prompt_overrides[0]
    assert override is not None
    assert _REQUIREMENT_CONTRACT not in override
    assert _STATE_TRANSITION_PROTOCOL not in override


# 功能：验证 chat session 处理一条消息后进入 waiting_for_input，并保留 user/assistant thread
# 设计：mock runner 主动追加 assistant 消息，确认 send_message 负责 user 消息、状态流转和 run_id 记录
async def test_send_message_chat_enters_waiting_and_writes_thread(tmp_path: Path) -> None:
    store = SessionStore(tmp_path)
    manager = SessionManager(
        store,
        lambda _workspace_root: _Runner(),
        EventBus(),
    )  # type: ignore[arg-type]
    session = await manager.create("chat", workspace_root=tmp_path.resolve())

    run_id = await manager.send_message(session.id, "hello")

    loaded = store.read_meta(session.id)
    assert loaded.status == "waiting_for_input"
    assert loaded.run_ids == [run_id]
    messages = store.read_messages(session.id)
    assert messages[0] == {"role": "user", "content": "hello"}
    assert messages[1]["role"] == "assistant"


# 功能：验证 one_shot session 在单次消息完成后自动 closed
# 设计：复用 mock runner 的成功路径，聚焦 mode 对最终状态的影响，保证 kama run 的统一路径正确
async def test_one_shot_auto_closes(tmp_path: Path) -> None:
    store = SessionStore(tmp_path)
    manager = SessionManager(
        store,
        lambda _workspace_root: _Runner(),
        EventBus(),
    )  # type: ignore[arg-type]
    session = await manager.create("one_shot", workspace_root=tmp_path.resolve())

    await manager.send_message(session.id, "hello")

    assert store.read_meta(session.id).status == "closed"


# 功能：验证不存在的 session_id 返回 session_not_found 错误码
# 设计：直接调用 get_history 的查找路径，断言 HandlerError code，覆盖 IPC handler 可结构化返回错误
async def test_missing_session_raises_handler_error(tmp_path: Path) -> None:
    manager = SessionManager(
        SessionStore(tmp_path),
        lambda _workspace_root: _Runner(),
        EventBus(),
    )  # type: ignore[arg-type]
    with pytest.raises(HandlerError) as exc:
        await manager.get_history("missing")
    assert exc.value.code == SESSION_NOT_FOUND


# 功能：验证 closed session 不能继续 send_message
# 设计：先显式 close，再发送消息，断言 session_closed 错误码，覆盖状态机拒绝路径
async def test_closed_session_rejects_message(tmp_path: Path) -> None:
    store = SessionStore(tmp_path)
    manager = SessionManager(
        store,
        lambda _workspace_root: _Runner(),
        EventBus(),
    )  # type: ignore[arg-type]
    session = await manager.create("chat", workspace_root=tmp_path.resolve())
    await manager.close(session.id)

    with pytest.raises(HandlerError) as exc:
        await manager.send_message(session.id, "again")
    assert exc.value.code == SESSION_CLOSED


# 功能：验证 SessionManager 用 Session.workspace_root 构造每次 AgentRunner
# 设计：runner_factory 只记录实际收到的 Path，不读 process cwd，直接锁定 session→runner 边界
async def test_runner_factory_receives_session_workspace(tmp_path: Path) -> None:
    workspace = (tmp_path / "workspace").resolve()
    workspace.mkdir()
    received: list[Path] = []

    # 记录 runner factory 收到的 workspace 并返回测试 runner
    def runner_factory(workspace_root: Path) -> _Runner:
        received.append(workspace_root)
        return _Runner()

    manager = SessionManager(
        SessionStore(tmp_path / "sessions"),
        runner_factory,
        EventBus(),
    )  # type: ignore[arg-type]
    session = await manager.create("chat", workspace_root=workspace)

    await manager.send_message(session.id, "hello")

    assert received == [workspace]


# 功能：验证两个 Session 的 runner 分别绑定 workspace A/B 且不受 daemon cwd 影响
# 设计：将 process cwd 切到第三个目录，依次发送 A/B 消息并断言 factory 的调用序列
async def test_runner_factory_isolated_between_session_workspaces(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon_cwd = tmp_path / "daemon"
    workspace_a = (tmp_path / "workspace-a").resolve()
    workspace_b = (tmp_path / "workspace-b").resolve()
    daemon_cwd.mkdir()
    workspace_a.mkdir()
    workspace_b.mkdir()
    monkeypatch.chdir(daemon_cwd)
    received: list[Path] = []

    # 记录多 Session 构造 runner 时的 workspace 顺序
    def runner_factory(workspace_root: Path) -> _Runner:
        received.append(workspace_root)
        return _Runner()

    manager = SessionManager(
        SessionStore(tmp_path / "sessions"),
        runner_factory,
        EventBus(),
    )  # type: ignore[arg-type]
    session_a = await manager.create("chat", workspace_root=workspace_a)
    session_b = await manager.create("chat", workspace_root=workspace_b)

    await manager.send_message(session_a.id, "from a")
    await manager.send_message(session_b.id, "from b")

    assert received == [workspace_a, workspace_b]


# 功能：验证 SessionManager 按各 Session workspace 解析同名 slash skill
# 设计：A/B 写入不同模板并复用真实 manager，通过各自 thread 中 runner 回显的 goal 判断解析来源
async def test_slash_skills_are_isolated_by_session_workspace(tmp_path: Path) -> None:
    workspace_a = tmp_path / "workspace-a"
    workspace_b = tmp_path / "workspace-b"
    for workspace, prompt in ((workspace_a, "prompt-a"), (workspace_b, "prompt-b")):
        skills = workspace / ".kama" / "skills"
        skills.mkdir(parents=True)
        (skills / "local.md").write_text(
            f"---\nname: local\ndescription: local\n---\n{prompt} $ARGUMENTS\n",
            encoding="utf-8",
        )
    store = SessionStore(tmp_path / "sessions")
    manager = SessionManager(
        store,
        lambda _workspace_root: _Runner(),
        EventBus(),
    )  # type: ignore[arg-type]
    session_a = await manager.create("chat", workspace_root=workspace_a.resolve())
    session_b = await manager.create("chat", workspace_root=workspace_b.resolve())

    await manager.send_message(session_a.id, "/local alpha")
    await manager.send_message(session_b.id, "/local beta")

    message_a = store.read_messages(session_a.id)[1]["content"][0]["text"]
    message_b = store.read_messages(session_b.id)[1]["content"][0]["text"]
    assert message_a == "done prompt-a alpha"
    assert message_b == "done prompt-b beta"
