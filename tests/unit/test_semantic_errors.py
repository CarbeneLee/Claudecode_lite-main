from __future__ import annotations

import pytest

from kama_claude.core.semantic.errors import (
    EmbeddingStrategyUnavailableError,
    IndexCorruptedError,
    IndexUnavailableError,
    SemanticError,
)

_ALL_ERROR_TYPES = [
    IndexUnavailableError,
    EmbeddingStrategyUnavailableError,
    IndexCorruptedError,
]


# 功能：验证三个稳定检索异常类型都继承 SemanticError 基类并可携带 detail
# 设计：参数化遍历全部异常类型，断言基类归属与 detail 透传，确保分类与调用方契约稳定
@pytest.mark.parametrize("exc_type", _ALL_ERROR_TYPES)
def test_semantic_errors_share_base_and_detail(exc_type: type[SemanticError]) -> None:
    exc = exc_type("boom", detail="reason")

    assert isinstance(exc, SemanticError)
    assert exc.detail == "reason"
    assert str(exc) == "boom"


# 功能：验证三个异常互不构成父子关系，可被调用方分别精确捕获
# 设计：两两断言 isinstance 反向失败，防止误继承导致的分类泄漏
def test_semantic_error_types_are_orthogonal() -> None:
    instances = [
        IndexUnavailableError("a"),
        EmbeddingStrategyUnavailableError("b"),
        IndexCorruptedError("c"),
    ]

    for i, inst in enumerate(instances):
        for j, other in enumerate(instances):
            if i != j:
                assert not isinstance(inst, type(other))
