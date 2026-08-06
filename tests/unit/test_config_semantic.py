from __future__ import annotations

from pathlib import Path

import pytest

from kama_claude.core.config import KamaConfig, get_config

_SEMANTIC_ENV_VARS = (
    "KAMA_SEMANTIC_ENABLED",
    "KAMA_SEMANTIC_STRATEGY",
    "KAMA_SEMANTIC_DEGRADATION",
    "KAMA_SEMANTIC_INDEX_DIR",
    "KAMA_SEMANTIC_DEFAULT_TOP_K",
    "KAMA_SEMANTIC_SIMILARITY_THRESHOLD",
)


# 用临时 TOML 加载配置；可选环境变量覆盖，其余 semantic 变量清除排除宿主干扰
def _load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    toml_body: str = "",
    env: dict[str, str] | None = None,
) -> KamaConfig:
    toml_path = tmp_path / "kama.toml"
    toml_path.write_text(toml_body, encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("KAMA_CONFIG", str(toml_path))
    for name in _SEMANTIC_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    for name, value in (env or {}).items():
        monkeypatch.setenv(name, value)
    return get_config()


# 功能：验证无 [semantic] 配置时 cfg.semantic 保持内建默认值
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
def test_semantic_defaults(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    expected: object,
) -> None:
    cfg = _load(tmp_path, monkeypatch)

    assert getattr(cfg.semantic, field) == expected


# 功能：验证 [semantic] 组显式值被解析到 cfg.semantic 对应字段
# 设计：参数化单字段覆盖，覆盖 bool/str/int/float 四类取值，隔离验证解析路径
@pytest.mark.parametrize(
    ("key", "toml_value", "field", "expected"),
    [
        ("enabled", "false", "enabled", False),
        ("strategy", '"onnx"', "strategy", "onnx"),
        ("index_dir", '"/data/index"', "index_dir", "/data/index"),
        ("chunk_size", "400", "chunk_size", 400),
        ("ngram_n", "2", "ngram_n", 2),
        ("default_top_k", "20", "default_top_k", 20),
        ("similarity_threshold", "0.25", "similarity_threshold", 0.25),
        ("degradation", '"fail_closed"', "degradation", "fail_closed"),
        ("max_query_chars", "128", "max_query_chars", 128),
    ],
)
def test_semantic_toml_explicit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    key: str,
    toml_value: str,
    field: str,
    expected: object,
) -> None:
    cfg = _load(tmp_path, monkeypatch, toml_body=f"[semantic]\n{key} = {toml_value}\n")

    assert getattr(cfg.semantic, field) == expected


# 功能：验证 [semantic] 未知 key 触发硬退出（与其他配置组一致的严格校验）
# 设计：写 semantic.foo = 1，断言 SystemExit 消息含未知 key 名
def test_semantic_toml_unknown_key_exits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(SystemExit) as exc:
        _load(tmp_path, monkeypatch, toml_body="[semantic]\nfoo = 1\n")

    assert "Unknown [semantic] keys: foo" in str(exc.value)


# 功能：验证 [semantic] 非 table 值触发硬退出
# 设计：顶层写 semantic = "x" 标量，断言表类型错误消息
def test_semantic_toml_not_a_table_exits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(SystemExit) as exc:
        _load(tmp_path, monkeypatch, toml_body='semantic = "x"\n')

    assert "Config error: [semantic] must be a table" in str(exc.value)


# 功能：验证 [semantic] 字段类型与取值约束校验硬退出
# 设计：参数化 (坏值, 错误消息片段)，覆盖 bool/str/int/float 类型与枚举取值约束
@pytest.mark.parametrize(
    ("key", "bad_value", "fragment"),
    [
        ("enabled", "'yes'", "semantic.enabled must be a boolean"),
        ("strategy", "123", "semantic.strategy must be a string"),
        ("index_dir", "123", "semantic.index_dir must be a string"),
        ("chunk_size", "'big'", "semantic.chunk_size must be an integer"),
        ("similarity_threshold", "'high'", "semantic.similarity_threshold must be a number"),
        ("strategy", '"fastembed"', "strategy must be one of"),
        ("degradation", '"fail_open"', "degradation must be one of"),
        ("ngram_n", "0", "ngram_n must be between"),
    ],
)
def test_semantic_toml_invalid_exits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    key: str,
    bad_value: str,
    fragment: str,
) -> None:
    with pytest.raises(SystemExit) as exc:
        _load(tmp_path, monkeypatch, toml_body=f"[semantic]\n{key} = {bad_value}\n")

    assert fragment in str(exc.value)


# 功能：验证 KAMA_SEMANTIC_* 环境变量覆盖对应字段
# 设计：参数化 (env 名, 值, 字段, 期望) 对，隔离验证每条解析路径
@pytest.mark.parametrize(
    ("env_name", "env_value", "field", "expected"),
    [
        ("KAMA_SEMANTIC_ENABLED", "false", "enabled", False),
        ("KAMA_SEMANTIC_STRATEGY", "onnx", "strategy", "onnx"),
        ("KAMA_SEMANTIC_DEGRADATION", "fail_closed", "degradation", "fail_closed"),
        ("KAMA_SEMANTIC_INDEX_DIR", "/data/index", "index_dir", "/data/index"),
        ("KAMA_SEMANTIC_DEFAULT_TOP_K", "20", "default_top_k", 20),
        ("KAMA_SEMANTIC_SIMILARITY_THRESHOLD", "0.25", "similarity_threshold", 0.25),
    ],
)
def test_semantic_env_overrides(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    env_name: str,
    env_value: str,
    field: str,
    expected: object,
) -> None:
    cfg = _load(tmp_path, monkeypatch, env={env_name: env_value})

    assert getattr(cfg.semantic, field) == expected


# 功能：验证系统环境变量优先级高于 [semantic] TOML 值
# 设计：TOML 写 enabled = true 同时 env 写 false，断言最终为 env 值（四级优先链约束）
def test_semantic_env_priority_over_toml(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _load(
        tmp_path,
        monkeypatch,
        toml_body="[semantic]\nenabled = true\n",
        env={"KAMA_SEMANTIC_ENABLED": "false"},
    )

    assert cfg.semantic.enabled is False


# 功能：验证 KAMA_SEMANTIC_* 非法取值时硬退出（含枚举与数值解析两条分支）
# 设计：参数化 (env 名, 坏值, 错误消息片段)，覆盖枚举白名单、整数与浮点解析失败
@pytest.mark.parametrize(
    ("env_name", "bad_value", "fragment"),
    [
        ("KAMA_SEMANTIC_STRATEGY", "fastembed", "strategy must be one of"),
        ("KAMA_SEMANTIC_DEGRADATION", "fail_open", "degradation must be one of"),
        ("KAMA_SEMANTIC_DEFAULT_TOP_K", "abc", "KAMA_SEMANTIC_DEFAULT_TOP_K must be an integer"),
        ("KAMA_SEMANTIC_DEFAULT_TOP_K", "0", "positive integer"),
        ("KAMA_SEMANTIC_SIMILARITY_THRESHOLD", "abc", "must be a number"),
        ("KAMA_SEMANTIC_SIMILARITY_THRESHOLD", "2.0", "similarity_threshold must be in"),
    ],
)
def test_semantic_env_invalid_exits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    env_name: str,
    bad_value: str,
    fragment: str,
) -> None:
    with pytest.raises(SystemExit) as exc:
        _load(tmp_path, monkeypatch, env={env_name: bad_value})

    assert fragment in str(exc.value)
