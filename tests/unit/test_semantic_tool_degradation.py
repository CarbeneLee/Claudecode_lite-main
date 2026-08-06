from __future__ import annotations

from pathlib import Path

import pytest

from kama_claude.core.semantic.errors import IndexUnavailableError
from kama_claude.core.semantic.tools import SearchSemanticTool
from kama_claude.core.tools.base import ToolResult
from kama_claude.core.tools.builtin.search_code import SearchCodeTool
from kama_claude.core.tools.errors import (
    _STABLE_ERROR_TYPES,
    RETRYABLE_ERROR_TYPES,
    classify_tool_exception,
)
from kama_claude.core.workspace.policy import WorkspaceAccessPolicy
from kama_claude.core.workspace.resolver import WorkspacePathResolver


class _UnavailableService:
    """search 必抛 IndexUnavailableError 的桩服务（模拟索引构建失败）"""

    def __init__(self) -> None:
        self.calls = 0

    async def search(self, query: str, *, top_k: int | None = None):
        self.calls += 1
        raise IndexUnavailableError("index build failed: disk full")

    def degraded_query(self, query: str) -> bool:
        return False


def _fallback(workspace: Path) -> SearchCodeTool:
    return SearchCodeTool(
        WorkspacePathResolver(workspace),
        WorkspaceAccessPolicy(workspace),
    )


def _tool(workspace: Path, *, degradation: str = "literal_fallback") -> SearchSemanticTool:
    return SearchSemanticTool(
        _UnavailableService(), fallback=_fallback(workspace), degradation=degradation
    )


# 功能：验证 literal_fallback——索引不可用时降级调 search_code，结果附加 degraded 标记
# 设计：工作区含匹配字面量，降级结果含 search_code 记录与末行标记，且非错误
async def test_literal_fallback_calls_search_code(tmp_path) -> None:
    (tmp_path / "ws").mkdir(parents=True)
    (tmp_path / "ws" / "auth.py").write_text(
        "def reset_password(user):\n    return user.token\n", encoding="utf-8"
    )
    tool = _tool(tmp_path / "ws")

    result = await tool.invoke({"query": "reset_password"})

    assert result.is_error is False
    lines = result.content.splitlines()
    assert any("reset_password" in line for line in lines)
    assert lines[-1] == "[search_semantic] degraded=literal_fallback"


# 功能：验证 literal_fallback 无匹配时仍附加标记
# 设计：字面量无命中，search_code 空结果 + 标记行
async def test_literal_fallback_no_match_still_marked(tmp_path) -> None:
    (tmp_path / "ws").mkdir(parents=True)
    (tmp_path / "ws" / "auth.py").write_text("def f():\n    pass\n", encoding="utf-8")
    tool = _tool(tmp_path / "ws")

    result = await tool.invoke({"query": "needle_zzz"})

    assert result.is_error is False
    assert result.content.splitlines()[-1] == "[search_semantic] degraded=literal_fallback"


# 功能：验证 fail_closed——索引不可用时返回 is_error + 稳定错误码，不降级
# 设计：fail_closed 配置下结果含语义化错误且 error_type=semantic_index_unavailable
async def test_fail_closed_returns_error_with_stable_code(tmp_path) -> None:
    (tmp_path / "ws").mkdir(parents=True)
    (tmp_path / "ws" / "auth.py").write_text("def f():\n    pass\n", encoding="utf-8")
    tool = _tool(tmp_path / "ws", degradation="fail_closed")

    result = await tool.invoke({"query": "reset_password"})

    assert result.is_error is True
    assert result.error_type == "semantic_index_unavailable"
    assert "index build failed" in result.content


# 功能：验证 fail_closed 不触发 fallback（杜绝隐藏降级）
# 设计：fallback 被替换为必炸桩，fail_closed 下不会触达
async def test_fail_closed_does_not_touch_fallback(tmp_path, monkeypatch) -> None:
    (tmp_path / "ws").mkdir(parents=True)
    service = _UnavailableService()
    fallback = _fallback(tmp_path / "ws")

    def explode(params):
        raise AssertionError("fallback must not be invoked under fail_closed")

    monkeypatch.setattr(fallback, "invoke", explode)
    tool = SearchSemanticTool(service, fallback=fallback, degradation="fail_closed")

    result = await tool.invoke({"query": "needle"})

    assert result.is_error is True


# 功能：验证 fallback 自身错误原样透传（不加 degraded 标记、不改 error_type）
# 设计：fallback 返回 is_error 时工具直接返回该结果
async def test_fallback_error_passes_through(tmp_path, monkeypatch) -> None:
    (tmp_path / "ws").mkdir(parents=True)
    service = _UnavailableService()
    fallback = _fallback(tmp_path / "ws")

    async def failing_invoke(params: dict[str, object]):
        return await _error_result()

    monkeypatch.setattr(fallback, "invoke", failing_invoke)
    tool = SearchSemanticTool(service, fallback=fallback, degradation="literal_fallback")

    result = await tool.invoke({"query": "needle"})

    assert result.is_error is True
    assert result.content.splitlines()[-1] != "[search_semantic] degraded=literal_fallback"


# 功能：验证非法 degradation 配置在构造时即拒绝
# 设计：未知模式抛 ValueError
def test_unknown_degradation_rejected(tmp_path) -> None:
    (tmp_path / "ws").mkdir(parents=True)
    with pytest.raises(ValueError, match="degradation"):
        SearchSemanticTool(
            _UnavailableService(),
            fallback=_fallback(tmp_path / "ws"),
            degradation="bogus",
        )


# 功能：验证错误分类——IndexUnavailableError → 稳定码 semantic_index_unavailable
# 设计：classify 映射正确，且稳定码进 _STABLE_ERROR_TYPES 与 RETRYABLE_ERROR_TYPES
def test_classify_index_unavailable_error() -> None:
    error_type, message = classify_tool_exception(
        IndexUnavailableError("index build failed")
    )

    assert error_type == "semantic_index_unavailable"
    assert "index" in message
    assert "semantic_index_unavailable" in _STABLE_ERROR_TYPES
    assert "semantic_index_unavailable" in RETRYABLE_ERROR_TYPES


async def _error_result() -> ToolResult:
    return ToolResult(content="search failed", is_error=True, error_type="transient_error")
