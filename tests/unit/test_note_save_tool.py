from __future__ import annotations

from pathlib import Path

import pytest

from kama_claude.core.session.store import SessionStore
from kama_claude.core.tools.builtin.note_save import NoteSaveTool


# 功能：验证 note_save 正常调用会把 content 写入 notes.md
# 设计：使用真实 SessionStore 和 tmp_path，断言工具返回与文件内容，覆盖工具到文件层的完整路径
async def test_note_save_appends_note(tmp_path: Path) -> None:
    store = SessionStore(tmp_path)
    tool = NoteSaveTool(store, "sess-1", "run-1")

    result = await tool.invoke({"content": "Python 3.12"})

    assert result.content == "saved"
    assert not result.is_error
    assert "Python 3.12" in store.read_notes("sess-1")


# 功能：验证空字符串和纯空白 content 都是永久输入错误
# 设计：参数化 stripped 后为空的代表输入，断言 invalid_input 且不产生 notes 副作用
@pytest.mark.parametrize("content", ["", "   ", "\n\t"])
async def test_note_save_rejects_empty_content(
    tmp_path: Path,
    content: str,
) -> None:
    store = SessionStore(tmp_path)
    tool = NoteSaveTool(store, "sess-1", "run-1")

    result = await tool.invoke({"content": content})

    assert result.is_error
    assert result.error_type == "invalid_input"
    assert store.read_notes("sess-1") == ""
