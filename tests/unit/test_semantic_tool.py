from __future__ import annotations

import re

import pytest
from pydantic import ValidationError

from kama_claude.core.semantic.tools import SearchSemanticParams
from tests.unit._semantic_test_support import _footer, _records, _tool

AUTH_PY = (
    '"""auth module"""\n'
    "import os\n"
    "\n"
    "def reset_password(user):\n"
    "    token = user.token\n"
    "    return token\n"
    "\n"
    "class Session:\n"
    "    def refresh(self):\n"
    "        return None\n"
)

# 层级展示格式：path:start-end [score=x.xx] parent → child()
_RECORD_RE = re.compile(
    r"^auth\.py:(\d+)-(\d+) \[score=(\d\.\d\d)\] "
    r"(?:(?P<parent>\w+) → )?(?P<symbol>\w+)\(\)$"
)


def _workspace(tmp_path) -> None:
    (tmp_path / "ws").mkdir(parents=True, exist_ok=True)
    (tmp_path / "ws" / "auth.py").write_text(AUTH_PY, encoding="utf-8")


# 功能：验证工具返回层级展示格式（parent → child 与行号区间 + 分数）
# 设计：类成员函数查询命中 Session → refresh()；顶层函数命中 reset_password()
async def test_invoke_returns_hierarchical_results(tmp_path) -> None:
    _workspace(tmp_path)
    tool = _tool(tmp_path / "ws", tmp_path / "idx")

    result = await tool.invoke({"query": "refresh"})

    assert result.is_error is False
    lines = _records(result.content)
    assert len(lines) == 1
    match = _RECORD_RE.fullmatch(lines[0])
    assert match is not None, lines[0]
    assert match.group("parent") == "Session"
    assert match.group("symbol") == "refresh"
    assert _footer(result.content) == {
        "results": 1,
        "degraded": "none",
        "truncated": "none",
    }

    top_level = await tool.invoke({"query": "reset_password"})
    lines = _records(top_level.content)
    match = _RECORD_RE.fullmatch(lines[0])
    assert match is not None, lines[0]
    assert match.group("parent") is None
    assert match.group("symbol") == "reset_password"


# 功能：验证 top_k 限制结果数量并写入 footer
# 设计：多符号文件查询共享 token，top_k=2 只返回 2 条
async def test_top_k_limits_results(tmp_path) -> None:
    multi = "\n\n".join(f"def fn{i}():\n    token = {i}\n" for i in range(1, 5))
    (tmp_path / "ws").mkdir(parents=True)
    (tmp_path / "ws" / "multi.py").write_text(multi, encoding="utf-8")
    tool = _tool(tmp_path / "ws", tmp_path / "idx")

    result = await tool.invoke({"query": "token", "top_k": 2})

    assert len(_records(result.content)) == 2
    assert _footer(result.content)["results"] == 2


# 功能：验证无匹配返回空结果（仅 footer）
# 设计：工作区无共享 gram 的查询 → results=0
async def test_no_match_returns_empty_with_footer(tmp_path) -> None:
    _workspace(tmp_path)
    tool = _tool(tmp_path / "ws", tmp_path / "idx")

    result = await tool.invoke({"query": "zzzzzzzz"})

    assert result.is_error is False
    assert _records(result.content) == []
    assert _footer(result.content)["results"] == 0


# 功能：验证退化查询（过短/无 gram）返回空并在 footer 标记 degraded=query
# 设计：参数化两类退化查询，均 results=0 且 degraded=query（不自动降级字面量）
@pytest.mark.parametrize("query", ["ab", "!!!"])
async def test_degraded_query_marks_footer(tmp_path, query: str) -> None:
    _workspace(tmp_path)
    tool = _tool(tmp_path / "ws", tmp_path / "idx")

    result = await tool.invoke({"query": query})

    assert _records(result.content) == []
    footer = _footer(result.content)
    assert footer["results"] == 0
    assert footer["degraded"] == "query"


# 功能：验证查询参数校验——空/纯空白/超长/非法控制字符均拒绝
# 设计：参数化 4 类非法查询，model_validate 抛 ValidationError（与 search_code 一致）
@pytest.mark.parametrize(
    "query",
    ["", "   ", "x" * 257, "has\ttab"],
)
def test_query_validation_rejects_invalid(query: str) -> None:
    with pytest.raises(ValidationError):
        SearchSemanticParams.model_validate({"query": query})


# 功能：验证 top_k 边界——0 与超上限拒绝，合法值通过
# 设计：参数化非法 top_k 抛 ValidationError；合法 top_k 原样保留
@pytest.mark.parametrize("top_k", [0, 51])
def test_top_k_validation_rejects_out_of_range(top_k: int) -> None:
    with pytest.raises(ValidationError):
        SearchSemanticParams.model_validate({"query": "needle", "top_k": top_k})


def test_top_k_validation_accepts_boundary() -> None:
    assert SearchSemanticParams.model_validate({"query": "needle", "top_k": 50}).top_k == 50
    assert SearchSemanticParams.model_validate({"query": "needle"}).top_k is None


# 功能：验证输出超字节上限时截断记录但保留完整 footer（truncated 标记）
# 设计：top_k=50 全命中 + 250 字符级超长相对路径 → 单条记录 ~800B，合计超 32KiB，
#       截断后 footer 完整且 truncated=output_limit
async def test_output_cap_keeps_footer(tmp_path) -> None:
    ws = tmp_path / "ws"
    component = "d" * 250
    for i in range(50):
        path = ws / component / component / component / f"f{i:02d}.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("def helper_token():\n    pass\n", encoding="utf-8")
    tool = _tool(ws, tmp_path / "idx")

    result = await tool.invoke({"query": "helper_token", "top_k": 50})

    assert result.is_error is False
    footer = _footer(result.content)
    assert footer["truncated"] == "output_limit"
    assert len(result.content.encode("utf-8")) <= 32 * 1024
