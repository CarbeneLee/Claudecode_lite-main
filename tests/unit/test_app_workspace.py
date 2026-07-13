from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from kama_claude.core.app import CoreApp
from kama_claude.core.bus.envelope import HandlerError
from kama_claude.core.session.model import Session, SessionMode
from kama_claude.core.workspace.errors import INVALID_WORKSPACE


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
    ) -> Session:
        self.created.append((mode, title, workspace_root))
        return Session(
            id="sess-test",
            mode=mode,
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
