from __future__ import annotations

import asyncio
import dataclasses
import subprocess
from collections.abc import Callable
from pathlib import Path

import pytest

import kama_claude.core.app as app_module
from kama_claude.core.app import CoreApp
from kama_claude.core.bus.envelope import HandlerError
from kama_claude.core.config import KamaConfig
from kama_claude.core.git.config import GitConfig
from kama_claude.core.git.manager import GitManager
from kama_claude.core.sandbox.config import SandboxConfig
from kama_claude.core.sandbox.executors import ExecResult
from kama_claude.core.sandbox.manager import SandboxManager
from kama_claude.core.semantic.service import SemanticRetrievalService
from kama_claude.core.session.model import AgentModeSnapshot, Session, SessionMode
from kama_claude.core.workspace.context import WorkspaceContext
from kama_claude.core.workspace.errors import INVALID_WORKSPACE


def _git(cwd: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


class _Sessions:
    # 初始化用于记录 CoreApp 调用参数的会话替身
    def __init__(self) -> None:
        self.created: list[tuple[SessionMode, str, Path]] = []
        self.messages: list[tuple[str, str, str | None]] = []

    # 记录创建参数并返回最小 active Session
    async def create(
        self,
        mode: SessionMode,
        title: str = "",
        *,
        workspace_root: Path,
        agent_mode: str = "direct",
    ) -> Session:
        self.created.append((mode, title, workspace_root))
        return Session(
            id="sess-test",
            mode=mode,
            agent_mode=agent_mode,  # type: ignore[arg-type]
            status="active",
            title=title,
            created_at="t",
            updated_at="t",
            workspace_root=workspace_root,
        )

    # 记录 one-shot handler 创建的后台消息任务
    async def send_message(
        self,
        sid: str,
        content: str,
        *,
        run_id: str | None = None,
    ) -> str:
        self.messages.append((sid, content, run_id))
        return run_id or "run-test"

    # 返回固定 mode authority，覆盖 CoreApp 到 wire result 的 revision 映射
    async def get_agent_mode(self, sid: str) -> AgentModeSnapshot:
        assert sid == "sess-test"
        return AgentModeSnapshot("plan", 4)

    # 返回固定 committed mode authority，覆盖 set handler 的 revision 映射
    async def set_agent_mode(self, sid: str, agent_mode: str) -> AgentModeSnapshot:
        assert sid == "sess-test"
        assert agent_mode == "direct"
        return AgentModeSnapshot("direct", 5)


# 功能：验证 agent.run 校验 workspace 后用 canonical Path 创建 one_shot Session
# 设计：通过 symlink 传入非 canonical 绝对路径，直接观察 CoreApp 与 SessionManager 边界
async def test_agent_run_handler_creates_session_with_canonical_workspace(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    alias = tmp_path / "workspace-link"
    alias.symlink_to(workspace, target_is_directory=True)
    sessions = _Sessions()
    app = CoreApp()
    app._sessions = sessions  # type: ignore[assignment]

    result = await app._agent_run_handler(
        {"goal": "inspect", "workspace_root": str(alias)}
    )
    await asyncio.gather(*app._running_runs)

    assert result.run_id
    assert sessions.created == [("one_shot", "inspect", workspace.resolve(strict=True))]
    assert sessions.messages == [("sess-test", "inspect", result.run_id)]


# 功能：验证 CoreApp mode handlers 将 SessionManager snapshot 映射为带 revision 的 wire result
# 设计：使用轻量 session manager stub，隔离 socket 层并直接检查 daemon handler 的公共协议边界
async def test_mode_handlers_return_authoritative_revision() -> None:
    app = CoreApp()
    app._sessions = _Sessions()  # type: ignore[assignment]

    get_result = await app._session_get_agent_mode_handler({"session_id": "sess-test"})
    set_result = await app._session_set_agent_mode_handler(
        {"session_id": "sess-test", "agent_mode": "direct"}
    )

    assert get_result.agent_mode == "plan"
    assert get_result.revision == 4
    assert set_result.agent_mode == "direct"
    assert set_result.revision == 5


# 功能：验证 agent.run 对相对 workspace 返回与 session.create 相同的 domain error
# 设计：直接覆盖 one-shot handler 的错误分支，避免只由 chat handler 间接证明映射一致性
async def test_agent_run_handler_rejects_relative_workspace() -> None:
    app = CoreApp()
    app._sessions = _Sessions()  # type: ignore[assignment]

    with pytest.raises(HandlerError) as exc:
        await app._agent_run_handler(
            {"goal": "inspect", "workspace_root": "project"}
        )

    assert exc.value.code == INVALID_WORKSPACE
    assert str(exc.value) == "invalid workspace_root"
    assert exc.value.data == {"reason": "not_absolute"}


# 功能：验证 session.create 校验 workspace 后传入 canonical Path
# 设计：使用真实目录和记录型会话替身，隔离 runner 与持久化以聚焦 handler 传播
async def test_session_create_handler_uses_canonical_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    sessions = _Sessions()
    app = CoreApp()
    app._sessions = sessions  # type: ignore[assignment]

    result = await app._session_create_handler(
        {
            "mode": "chat",
            "title": "project",
            "workspace_root": str(workspace),
        }
    )

    assert result.session_id == "sess-test"
    assert sessions.created == [("chat", "project", workspace.resolve(strict=True))]


# 功能：验证 CoreApp 将相对 workspace 转换为稳定 INVALID_WORKSPACE domain error
# 设计：直接调用 session.create handler，同时断言 code、message 和 reason 三层契约
async def test_session_create_handler_rejects_relative_workspace() -> None:
    app = CoreApp()
    app._sessions = _Sessions()  # type: ignore[assignment]

    with pytest.raises(HandlerError) as exc:
        await app._session_create_handler(
            {"mode": "chat", "workspace_root": "project"}
        )

    assert exc.value.code == INVALID_WORKSPACE
    assert str(exc.value) == "invalid workspace_root"
    assert exc.value.data == {"reason": "not_absolute"}


# 功能：验证 CoreApp 将不存在 workspace 转换为 not_found domain error
# 设计：使用 tmp_path 下未创建的绝对路径，确保错误不被归类为相对路径
async def test_session_create_handler_rejects_missing_workspace(tmp_path: Path) -> None:
    app = CoreApp()
    app._sessions = _Sessions()  # type: ignore[assignment]

    with pytest.raises(HandlerError) as exc:
        await app._session_create_handler(
            {"mode": "chat", "workspace_root": str(tmp_path / "missing")}
        )

    assert exc.value.code == INVALID_WORKSPACE
    assert str(exc.value) == "invalid workspace_root"
    assert exc.value.data == {"reason": "not_found"}


# 功能：验证 CoreApp 将普通文件 workspace 转换为 not_directory domain error
# 设计：创建已存在文件排除 not_found，确认 handler 保留验证器的稳定 reason
async def test_session_create_handler_rejects_file_workspace(tmp_path: Path) -> None:
    workspace_file = tmp_path / "workspace.txt"
    workspace_file.write_text("file", encoding="utf-8")
    app = CoreApp()
    app._sessions = _Sessions()  # type: ignore[assignment]

    with pytest.raises(HandlerError) as exc:
        await app._session_create_handler(
            {"mode": "chat", "workspace_root": str(workspace_file)}
        )

    assert exc.value.code == INVALID_WORKSPACE
    assert str(exc.value) == "invalid workspace_root"
    assert exc.value.data == {"reason": "not_directory"}


# 功能：验证 CoreApp 的 runner_factory 将收到的 workspace 传给 AgentRunner
# 设计：在 SessionManager 组装点调用真实 factory，用哨兵异常在 socket 启动前停止 app.run
async def test_core_app_runner_factory_passes_workspace_to_agent_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path.resolve()
    captured: list[Path] = []
    config = KamaConfig()
    config.trace.enabled = False

    class _StopAfterWiring(Exception):
        pass

    class _SessionManager:
        # 调用 CoreApp 注入的 runner factory 后立即停止启动流程
        def __init__(
            self,
            store: object,
            runner_factory: Callable[[Path], object],
            bus: object,
            provider: object,
        ) -> None:
            runner_factory(workspace)
            raise _StopAfterWiring

    # 捕获 CoreApp 构造 AgentRunner 时的显式 workspace 参数
    def fake_agent_runner(
        config_arg: KamaConfig,
        *,
        workspace_root: Path,
        **kwargs: object,
    ) -> object:
        captured.append(workspace_root)
        return object()

    monkeypatch.setattr(app_module, "get_config", lambda: config)
    monkeypatch.setattr(app_module, "setup_logging", lambda config_arg: None)
    monkeypatch.setattr(app_module, "SessionStore", lambda root: object())
    monkeypatch.setattr(app_module, "AnthropicProvider", lambda model: object())
    monkeypatch.setattr(app_module, "SessionManager", _SessionManager)
    monkeypatch.setattr(app_module, "AgentRunner", fake_agent_runner)

    with pytest.raises(_StopAfterWiring):
        await CoreApp().run()

    assert captured == [workspace]


class _FakeRuntime:
    # 记录 close 调用次数的最小容器运行时替身
    def __init__(self) -> None:
        self.close_calls = 0

    # 记录并忽略 ensure_running；shutdown 测试不触发创建路径
    async def ensure_running(self) -> None:
        pass

    # shutdown 测试不应执行命令
    async def exec(self, command: str, *, cwd: str, timeout: float) -> ExecResult:
        raise AssertionError("shutdown 测试不应执行命令")

    # 记录 close 调用，供 _shutdown 遍历断言
    async def close(self) -> None:
        self.close_calls += 1


# 功能：验证 sandbox 启用时 runner_factory 注入 per-workspace SandboxManager
# 设计：捕获 runner_factory 对同一 workspace 的两次调用，断言注入 SandboxManager 且复用同一实例（注册表）
async def test_core_app_runner_factory_injects_sandbox_manager(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path.resolve()
    captured: list[object] = []
    config = KamaConfig()
    config.trace.enabled = False

    class _StopAfterWiring(Exception):
        pass

    class _SessionManager:
        # 同一 workspace 调用两次 runner_factory，验证注册表复用
        def __init__(
            self,
            store: object,
            runner_factory: Callable[[Path], object],
            bus: object,
            provider: object,
        ) -> None:
            runner_factory(workspace)
            runner_factory(workspace)
            raise _StopAfterWiring

    # 捕获 CoreApp 构造 AgentRunner 时传入的 sandbox_manager 参数
    def fake_agent_runner(
        config_arg: KamaConfig,
        *,
        workspace_root: Path,
        **kwargs: object,
    ) -> object:
        captured.append(kwargs.get("sandbox_manager"))
        return object()

    monkeypatch.setattr(app_module, "get_config", lambda: config)
    monkeypatch.setattr(app_module, "setup_logging", lambda config_arg: None)
    monkeypatch.setattr(app_module, "SessionStore", lambda root: object())
    monkeypatch.setattr(app_module, "AnthropicProvider", lambda model: object())
    monkeypatch.setattr(app_module, "SessionManager", _SessionManager)
    monkeypatch.setattr(app_module, "AgentRunner", fake_agent_runner)

    with pytest.raises(_StopAfterWiring):
        await CoreApp().run()

    assert len(captured) == 2
    manager = captured[0]
    assert isinstance(manager, SandboxManager)
    assert manager._config is config.sandbox  # 同一配置对象
    assert captured[1] is manager  # 注册表按 workspace 复用同一实例


# 功能：验证 sandbox 关闭时 runner_factory 注入 None（宿主执行路径）
# 设计：config.sandbox.enabled=False，断言 sandbox_manager 参数为 None 且不触发管理器创建
async def test_core_app_runner_factory_no_sandbox_when_disabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path.resolve()
    captured: list[object] = []
    config = KamaConfig()
    config.trace.enabled = False
    config.sandbox = SandboxConfig(enabled=False)

    class _StopAfterWiring(Exception):
        pass

    class _SessionManager:
        def __init__(
            self,
            store: object,
            runner_factory: Callable[[Path], object],
            bus: object,
            provider: object,
        ) -> None:
            runner_factory(workspace)
            raise _StopAfterWiring

    def fake_agent_runner(
        config_arg: KamaConfig,
        *,
        workspace_root: Path,
        **kwargs: object,
    ) -> object:
        captured.append(kwargs.get("sandbox_manager"))
        return object()

    monkeypatch.setattr(app_module, "get_config", lambda: config)
    monkeypatch.setattr(app_module, "setup_logging", lambda config_arg: None)
    monkeypatch.setattr(app_module, "SessionStore", lambda root: object())
    monkeypatch.setattr(app_module, "AnthropicProvider", lambda model: object())
    monkeypatch.setattr(app_module, "SessionManager", _SessionManager)
    monkeypatch.setattr(app_module, "AgentRunner", fake_agent_runner)

    with pytest.raises(_StopAfterWiring):
        await CoreApp().run()

    assert captured == [None]


# 功能：验证 _shutdown 遍历关闭所有已注册 workspace context 的 sandbox manager
# 设计：预置两个 WorkspaceContext（真实 SandboxManager + 记录型 runtime），断言 close 各一次
async def test_core_app_shutdown_closes_sandbox_managers(tmp_path: Path) -> None:
    class _Server:
        # 记录型 socket server 替身，只提供 stop
        async def stop(self) -> None:
            pass

    first_runtime = _FakeRuntime()
    second_runtime = _FakeRuntime()
    workspace_a = tmp_path / "a"
    workspace_b = tmp_path / "b"
    app = CoreApp()
    app._contexts = {
        workspace_a: WorkspaceContext(
            root=workspace_a,
            sandbox=SandboxManager(
                config=SandboxConfig(),
                workspace_root=workspace_a,
                runtime=first_runtime,  # type: ignore[arg-type]
            ),
            git=None,
        ),
        workspace_b: WorkspaceContext(
            root=workspace_b,
            sandbox=SandboxManager(
                config=SandboxConfig(),
                workspace_root=workspace_b,
                runtime=second_runtime,  # type: ignore[arg-type]
            ),
            git=None,
        ),
    }

    await app._shutdown(_Server())

    assert first_runtime.close_calls == 1
    assert second_runtime.close_calls == 1


# 功能：验证 git 启用时 runner_factory 注入 per-workspace GitManager
# 设计：捕获 runner_factory 对同一 workspace 的两次调用，断言注入 GitManager 且复用同一实例（注册表）
async def test_core_app_runner_factory_injects_git_manager(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path.resolve()
    captured: list[object] = []
    config = KamaConfig()
    config.trace.enabled = False

    class _StopAfterWiring(Exception):
        pass

    class _SessionManager:
        # 同一 workspace 调用两次 runner_factory，验证注册表复用
        def __init__(
            self,
            store: object,
            runner_factory: Callable[[Path], object],
            bus: object,
            provider: object,
        ) -> None:
            runner_factory(workspace)
            runner_factory(workspace)
            raise _StopAfterWiring

    # 捕获 CoreApp 构造 AgentRunner 时传入的 git_manager 参数
    def fake_agent_runner(
        config_arg: KamaConfig,
        *,
        workspace_root: Path,
        **kwargs: object,
    ) -> object:
        captured.append(kwargs.get("git_manager"))
        return object()

    monkeypatch.setattr(app_module, "get_config", lambda: config)
    monkeypatch.setattr(app_module, "setup_logging", lambda config_arg: None)
    monkeypatch.setattr(app_module, "SessionStore", lambda root: object())
    monkeypatch.setattr(app_module, "AnthropicProvider", lambda model: object())
    monkeypatch.setattr(app_module, "SessionManager", _SessionManager)
    monkeypatch.setattr(app_module, "AgentRunner", fake_agent_runner)

    with pytest.raises(_StopAfterWiring):
        await CoreApp().run()

    assert len(captured) == 2
    manager = captured[0]
    assert isinstance(manager, GitManager)
    assert manager.config is config.git  # 同一配置对象
    assert captured[1] is manager  # 注册表按 workspace 复用同一实例


# 功能：验证 git 关闭时 runner_factory 注入 None（无 git 能力路径）
# 设计：config.git.enabled=False，断言 git_manager 参数为 None 且不触发管理器创建
async def test_core_app_runner_factory_no_git_when_disabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path.resolve()
    captured: list[object] = []
    config = KamaConfig()
    config.trace.enabled = False
    config.git = GitConfig(enabled=False)

    class _StopAfterWiring(Exception):
        pass

    class _SessionManager:
        def __init__(
            self,
            store: object,
            runner_factory: Callable[[Path], object],
            bus: object,
            provider: object,
        ) -> None:
            runner_factory(workspace)
            raise _StopAfterWiring

    def fake_agent_runner(
        config_arg: KamaConfig,
        *,
        workspace_root: Path,
        **kwargs: object,
    ) -> object:
        captured.append(kwargs.get("git_manager"))
        return object()

    monkeypatch.setattr(app_module, "get_config", lambda: config)
    monkeypatch.setattr(app_module, "setup_logging", lambda config_arg: None)
    monkeypatch.setattr(app_module, "SessionStore", lambda root: object())
    monkeypatch.setattr(app_module, "AnthropicProvider", lambda model: object())
    monkeypatch.setattr(app_module, "SessionManager", _SessionManager)
    monkeypatch.setattr(app_module, "AgentRunner", fake_agent_runner)

    with pytest.raises(_StopAfterWiring):
        await CoreApp().run()

    assert captured == [None]


# 功能：验证 _shutdown 级联关闭所有已注册 workspace context（sandbox + git 各一次）
# 设计：预置两个 WorkspaceContext（真实 SandboxManager + 真实 GitManager），断言运行时
#       close 各一次、git manager 进入 CLOSED 终态，且重复 close 幂等
async def test_core_app_shutdown_closes_contexts(tmp_path: Path) -> None:
    class _Server:
        # 记录型 socket server 替身，只提供 stop
        async def stop(self) -> None:
            pass

    first_runtime = _FakeRuntime()
    second_runtime = _FakeRuntime()
    workspace_a = tmp_path / "a"
    workspace_b = tmp_path / "b"
    app = CoreApp()
    app._contexts = {
        workspace_a: WorkspaceContext(
            root=workspace_a,
            sandbox=SandboxManager(
                config=SandboxConfig(),
                workspace_root=workspace_a,
                runtime=first_runtime,  # type: ignore[arg-type]
            ),
            git=GitManager(config=GitConfig(), workspace_root=workspace_a),
        ),
        workspace_b: WorkspaceContext(
            root=workspace_b,
            sandbox=SandboxManager(
                config=SandboxConfig(),
                workspace_root=workspace_b,
                runtime=second_runtime,  # type: ignore[arg-type]
            ),
            git=GitManager(config=GitConfig(), workspace_root=workspace_b),
        ),
    }

    await app._shutdown(_Server())

    assert first_runtime.close_calls == 1
    assert second_runtime.close_calls == 1
    assert app._contexts[workspace_a].git is not None
    assert app._contexts[workspace_a].git.state == "closed"
    assert app._contexts[workspace_b].git.state == "closed"

    # 幂等：shutdown 后重复 close 不报错、不重复 close 底层
    await app._contexts[workspace_a].close()
    assert first_runtime.close_calls == 1


# 功能：验证 runner_factory 注入 SemanticRetrievalService（配置启用时）且注册表复用
# 设计：捕获两次 runner_factory 调用的 semantic_service 参数，断言类型、配置对象与实例复用
async def test_core_app_runner_factory_injects_semantic_service(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path.resolve()
    captured: list[object] = []
    config = KamaConfig()
    config.trace.enabled = False
    config.semantic = dataclasses.replace(
        config.semantic, index_dir=str(tmp_path / "idx")
    )

    class _StopAfterWiring(Exception):
        pass

    class _SessionManager:
        # 同一 workspace 调用两次 runner_factory，验证注册表复用
        def __init__(
            self,
            store: object,
            runner_factory: Callable[[Path], object],
            bus: object,
            provider: object,
        ) -> None:
            runner_factory(workspace)
            runner_factory(workspace)
            raise _StopAfterWiring

    # 捕获 CoreApp 构造 AgentRunner 时传入的 semantic_service 参数
    def fake_agent_runner(
        config_arg: KamaConfig,
        *,
        workspace_root: Path,
        **kwargs: object,
    ) -> object:
        captured.append(kwargs.get("semantic_service"))
        return object()

    monkeypatch.setattr(app_module, "get_config", lambda: config)
    monkeypatch.setattr(app_module, "setup_logging", lambda config_arg: None)
    monkeypatch.setattr(app_module, "SessionStore", lambda root: object())
    monkeypatch.setattr(app_module, "AnthropicProvider", lambda model: object())
    monkeypatch.setattr(app_module, "SessionManager", _SessionManager)
    monkeypatch.setattr(app_module, "AgentRunner", fake_agent_runner)

    with pytest.raises(_StopAfterWiring):
        await CoreApp().run()

    assert len(captured) == 2
    service = captured[0]
    assert isinstance(service, SemanticRetrievalService)
    assert service._config is config.semantic  # type: ignore[attr-defined]  # 同一配置对象
    assert service._git_head_provider is not None  # type: ignore[attr-defined]  # 注入分支检测 provider
    assert captured[1] is service  # 注册表按 workspace 复用同一实例


# 功能：验证 semantic 关闭时 runner_factory 注入 None（无检索能力路径）
# 设计：config.semantic.enabled=False，断言 semantic_service 参数为 None 且不创建服务
async def test_core_app_runner_factory_no_semantic_when_disabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path.resolve()
    captured: list[object] = []
    config = KamaConfig()
    config.trace.enabled = False
    config.semantic = dataclasses.replace(config.semantic, enabled=False)

    class _StopAfterWiring(Exception):
        pass

    class _SessionManager:
        def __init__(
            self,
            store: object,
            runner_factory: Callable[[Path], object],
            bus: object,
            provider: object,
        ) -> None:
            runner_factory(workspace)
            raise _StopAfterWiring

    def fake_agent_runner(
        config_arg: KamaConfig,
        *,
        workspace_root: Path,
        **kwargs: object,
    ) -> object:
        captured.append(kwargs.get("semantic_service"))
        return object()

    monkeypatch.setattr(app_module, "get_config", lambda: config)
    monkeypatch.setattr(app_module, "setup_logging", lambda config_arg: None)
    monkeypatch.setattr(app_module, "SessionStore", lambda root: object())
    monkeypatch.setattr(app_module, "AnthropicProvider", lambda model: object())
    monkeypatch.setattr(app_module, "SessionManager", _SessionManager)
    monkeypatch.setattr(app_module, "AgentRunner", fake_agent_runner)

    with pytest.raises(_StopAfterWiring):
        await CoreApp().run()

    assert captured == [None]


# 功能：验证 git_head_provider 对真实 git 仓库返回 HEAD sha
# 设计：git init + 空提交后 provider 输出与 rev-parse HEAD 一致（分支切换检测的输入）
def test_git_head_provider_returns_sha_for_repo(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(
        repo,
        "-c",
        "user.email=t@example.com",
        "-c",
        "user.name=t",
        "commit",
        "--allow-empty",
        "-m",
        "init",
    )
    head = _git(repo, "rev-parse", "HEAD").strip()

    assert app_module.git_head_provider(repo) == head


# 功能：验证 git_head_provider 对非 git 目录返回 None
# 设计：普通目录无 .git，provider 返回 None（语义索引跳过 git_head 检查）
def test_git_head_provider_returns_none_for_non_repo(tmp_path: Path) -> None:
    assert app_module.git_head_provider(tmp_path.resolve()) is None
