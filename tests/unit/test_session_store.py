from __future__ import annotations

import json
from pathlib import Path

import pytest

from kama_claude.core.session.model import Session
from kama_claude.core.session.store import SessionStore


# 功能：验证 SessionStore 初始化时自动创建 sessions 根目录
# 设计：传入 tmp_path 下不存在的目录，断言目录被创建，覆盖首次启动 daemon 的冷路径
def test_store_creates_root(tmp_path: Path) -> None:
    root = tmp_path / "sessions"
    SessionStore(root)
    assert root.exists()


# 功能：验证 session meta 写入后能完整读回
# 设计：构造含 run_ids 的 Session，经过 JSON 文件往返后断言字段保持，覆盖 meta.json 的持久化契约
def test_session_meta_roundtrip_preserves_workspace_root(tmp_path: Path) -> None:
    store = SessionStore(tmp_path)
    workspace_root = (tmp_path / "workspace").resolve()
    workspace_root.mkdir()
    session = Session(
        id="sess-1",
        mode="chat",
        status="waiting_for_input",
        title="hello",
        created_at="t1",
        updated_at="t2",
        workspace_root=workspace_root,
        run_ids=["run-1"],
    )
    store.write_meta(session)
    loaded = store.read_meta("sess-1")
    assert loaded == session


# 功能：验证 Session.to_dict 将 workspace_root Path 序列化为字符串
# 设计：直接检查 domain model 输出，隔离 JSON writer 以锁定 Path 转换边界
def test_session_to_dict_serializes_workspace_root(tmp_path: Path) -> None:
    session = Session(
        id="sess-1",
        mode="chat",
        status="active",
        title="",
        created_at="t1",
        updated_at="t1",
        workspace_root=tmp_path.resolve(),
    )

    assert session.to_dict()["workspace_root"] == str(tmp_path.resolve())


# 功能：验证 Session.from_dict 将 workspace_root 字符串恢复为 Path
# 设计：传入完整 meta dict 并断言类型与值，避免只靠 dataclass 等值间接覆盖
def test_session_from_dict_restores_workspace_root(tmp_path: Path) -> None:
    workspace_root = tmp_path.resolve()
    session = Session.from_dict(
        {
            "id": "sess-1",
            "mode": "chat",
            "status": "active",
            "title": "",
            "created_at": "t1",
            "updated_at": "t1",
            "workspace_root": str(workspace_root),
            "run_ids": [],
        }
    )

    assert isinstance(session.workspace_root, Path)
    assert session.workspace_root == workspace_root


# 功能：验证缺少 workspace_root 的旧 session meta 读取时明确失败
# 设计：直接写入 legacy meta.json 再经 SessionStore 读取，确认不会 fallback 到 cwd
def test_legacy_session_meta_without_workspace_root_raises(tmp_path: Path) -> None:
    store = SessionStore(tmp_path)
    session_dir = store.session_dir("sess-legacy")
    session_dir.mkdir()
    (session_dir / "meta.json").write_text(
        json.dumps(
            {
                "id": "sess-legacy",
                "mode": "chat",
                "status": "active",
                "title": "legacy",
                "created_at": "t1",
                "updated_at": "t1",
                "run_ids": [],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(KeyError, match="workspace_root"):
        store.read_meta("sess-legacy")


# 功能：验证含 tool_use/tool_result block 的 thread 消息能按 Anthropic 格式读回
# 设计：追加 assistant tool_use 和 user tool_result，读取时应剥离 ts/run_id，只保留 API messages 所需字段
def test_thread_message_roundtrip_with_tool_blocks(tmp_path: Path) -> None:
    store = SessionStore(tmp_path)
    store.append_message("sess-1", "user", "read file")
    store.append_message(
        "sess-1",
        "assistant",
        [{"type": "tool_use", "id": "t1", "name": "read_file", "input": {"path": "x"}}],
        run_id="run-1",
    )
    store.append_message(
        "sess-1",
        "user",
        [{"type": "tool_result", "tool_use_id": "t1", "content": "ok"}],
        run_id="run-1",
    )

    messages = store.read_messages("sess-1")
    assert messages == [
        {"role": "user", "content": "read file"},
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "t1", "name": "read_file", "input": {"path": "x"}}
            ],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "ok"}],
        },
    ]


# 功能：验证 thread 尾部孤儿 tool_use 会被裁掉
# 设计：构造一条未配对 tool_result 的 assistant tool_use，读取时只返回最后一次配平之前的消息，避免 API 报 messages.invalid
def test_read_messages_trims_orphan_tool_use_tail(tmp_path: Path) -> None:
    store = SessionStore(tmp_path)
    store.append_message("sess-1", "user", "hello")
    store.append_message(
        "sess-1",
        "assistant",
        [{"type": "tool_use", "id": "orphan", "name": "read_file", "input": {}}],
        run_id="run-1",
    )
    assert store.read_messages("sess-1") == [{"role": "user", "content": "hello"}]


# 功能：验证 notes.md 不存在时读为空，追加笔记后能读到内容和 run_id
# 设计：先读空状态再追加，覆盖 chat 第一轮前和 note_save 调用后的两个关键状态
def test_notes_read_and_append(tmp_path: Path) -> None:
    store = SessionStore(tmp_path)
    assert store.read_notes("sess-1") == ""
    store.append_note("sess-1", "Python 3.12", "run-1")
    notes = store.read_notes("sess-1")
    assert "Python 3.12" in notes
    assert "run-1" in notes
