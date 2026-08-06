from __future__ import annotations

import pytest

from kama_claude.core.semantic.config import SemanticConfig


# 功能：验证 SemanticConfig 的默认值契约：默认启用、lexical 策略、资源上限与降级策略
# 设计：参数化 (字段, 期望值) 对，用 getattr 统一断言，避免为纯数据类写重复测试
@pytest.mark.parametrize(
    ("field", "expected"),
    [
        ("enabled", True),
        ("strategy", "lexical"),
        ("index_dir", "~/.kama/semantic"),
        ("chunk_size", 200),
        ("min_chunk_lines", 5),
        ("ngram_n", 3),
        ("default_top_k", 10),
        ("similarity_threshold", 0.10),
        ("max_index_files", 5000),
        ("max_file_bytes", 1024 * 1024),
        ("total_index_bytes", 32 * 1024 * 1024),
        ("degradation", "literal_fallback"),
        ("max_query_chars", 256),
    ],
)
def test_semantic_config_defaults(field: str, expected: object) -> None:
    assert getattr(SemanticConfig(), field) == expected


# 功能：验证 SemanticConfig 支持全字段显式构造覆盖
# 设计：显式传代表四类取值（bool/str 枚举/int/float）的字段断言保留，验证构造路径与默认路径一致
def test_semantic_config_explicit_override() -> None:
    cfg = SemanticConfig(
        enabled=False,
        strategy="onnx",
        degradation="fail_closed",
        default_top_k=5,
        similarity_threshold=0.25,
    )

    assert (cfg.enabled, cfg.strategy, cfg.degradation, cfg.default_top_k) == (
        False,
        "onnx",
        "fail_closed",
        5,
    )
    assert cfg.similarity_threshold == 0.25


# 功能：验证枚举与数值约束在构造时即拒绝非法值，避免配置错误扩散到运行时
# 设计：参数化 (字段, 非法值, 错误消息片段)，覆盖枚举白名单与数值边界两类校验路径
@pytest.mark.parametrize(
    ("field", "bad_value", "fragment"),
    [
        ("strategy", "fastembed", "strategy must be one of"),
        ("degradation", "fail_open", "degradation must be one of"),
        ("ngram_n", 0, "ngram_n must be between"),
        ("ngram_n", 7, "ngram_n must be between"),
        ("chunk_size", 0, "chunk_size must be a positive integer"),
        ("min_chunk_lines", -1, "min_chunk_lines must be a positive integer"),
        ("default_top_k", 0, "default_top_k must be a positive integer"),
        ("similarity_threshold", 0.0, "similarity_threshold must be in"),
        ("similarity_threshold", 1.5, "similarity_threshold must be in"),
        ("max_index_files", 0, "max_index_files must be a positive integer"),
        ("max_file_bytes", 0, "max_file_bytes must be a positive integer"),
        ("total_index_bytes", 0, "total_index_bytes must be a positive integer"),
        ("max_query_chars", 0, "max_query_chars must be a positive integer"),
    ],
)
def test_semantic_config_invalid(field: str, bad_value: object, fragment: str) -> None:
    with pytest.raises(ValueError) as exc:
        SemanticConfig(**{field: bad_value})  # type: ignore[arg-type]

    assert fragment in str(exc.value)
