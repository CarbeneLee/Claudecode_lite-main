from __future__ import annotations

from pathlib import Path

import pytest

from kama_claude.core.config import KamaConfig, get_config

_SANDBOX_ENV_VARS = (
    "KAMA_SANDBOX_ENABLED",
    "KAMA_SANDBOX_IMAGE",
    "KAMA_SANDBOX_NETWORK",
    "KAMA_SANDBOX_EXEC_TIMEOUT_S",
)


# 用临时 TOML 加载配置；可选环境变量覆盖，其余 sandbox 变量清除排除宿主干扰
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
    for name in _SANDBOX_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    for name, value in (env or {}).items():
        monkeypatch.setenv(name, value)
    return get_config()


# 功能：验证无 [sandbox] 配置时 cfg.sandbox 保持内建默认值
# 设计：参数化 (字段, 期望值) 对，用 getattr 统一断言，避免为纯数据类写重复测试
@pytest.mark.parametrize(
    ("field", "expected"),
    [
        ("enabled", True),
        ("image", "python:3.12-slim"),
        ("network", True),
        ("exec_timeout_s", 120),
    ],
)
def test_sandbox_defaults(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    expected: object,
) -> None:
    cfg = _load(tmp_path, monkeypatch)

    assert getattr(cfg.sandbox, field) == expected


# 功能：验证 [sandbox] 组显式值被解析到 cfg.sandbox 对应字段
# 设计：参数化单字段覆盖，每次只写一个 key，隔离验证解析路径
@pytest.mark.parametrize(
    ("key", "toml_value", "field", "expected"),
    [
        ("enabled", "false", "enabled", False),
        ("image", '"alpine:3.20"', "image", "alpine:3.20"),
        ("network", "false", "network", False),
        ("exec_timeout_s", "60", "exec_timeout_s", 60),
    ],
)
def test_sandbox_toml_explicit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    key: str,
    toml_value: str,
    field: str,
    expected: object,
) -> None:
    cfg = _load(tmp_path, monkeypatch, toml_body=f"[sandbox]\n{key} = {toml_value}\n")

    assert getattr(cfg.sandbox, field) == expected


# 功能：验证 [sandbox] 未知 key 触发硬退出（与其他配置组一致的严格校验）
# 设计：写 sandbox.foo = 1，断言 SystemExit 消息含未知 key 名
def test_sandbox_toml_unknown_key_exits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(SystemExit) as exc:
        _load(tmp_path, monkeypatch, toml_body="[sandbox]\nfoo = 1\n")

    assert "Unknown [sandbox] keys: foo" in str(exc.value)


# 功能：验证 [sandbox] 非 table 值触发硬退出
# 设计：顶层写 sandbox = "x" 标量，断言表类型错误消息
def test_sandbox_toml_not_a_table_exits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(SystemExit) as exc:
        _load(tmp_path, monkeypatch, toml_body='sandbox = "x"\n')

    assert "Config error: [sandbox] must be a table" in str(exc.value)


# 功能：验证 [sandbox] 字段类型与取值约束校验硬退出
# 设计：参数化 (坏值, 错误消息片段)，覆盖 bool/str/int 类型与正数约束
@pytest.mark.parametrize(
    ("key", "bad_value", "fragment"),
    [
        ("enabled", "'yes'", "sandbox.enabled must be a boolean"),
        ("image", "123", "sandbox.image must be a string"),
        ("network", "'yes'", "sandbox.network must be a boolean"),
        ("exec_timeout_s", "'60'", "sandbox.exec_timeout_s must be a positive integer"),
        ("exec_timeout_s", "0", "sandbox.exec_timeout_s must be a positive integer"),
    ],
)
def test_sandbox_toml_invalid_exits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    key: str,
    bad_value: str,
    fragment: str,
) -> None:
    with pytest.raises(SystemExit) as exc:
        _load(tmp_path, monkeypatch, toml_body=f"[sandbox]\n{key} = {bad_value}\n")

    assert fragment in str(exc.value)


# 功能：验证 KAMA_SANDBOX_* 环境变量覆盖 cfg.sandbox 对应字段
# 设计：参数化 (变量名, 值, 字段, 期望值)，每次只设一个变量隔离验证
@pytest.mark.parametrize(
    ("env_name", "env_value", "field", "expected"),
    [
        ("KAMA_SANDBOX_ENABLED", "false", "enabled", False),
        ("KAMA_SANDBOX_IMAGE", "alpine:3.20", "image", "alpine:3.20"),
        ("KAMA_SANDBOX_NETWORK", "false", "network", False),
        ("KAMA_SANDBOX_EXEC_TIMEOUT_S", "30", "exec_timeout_s", 30),
    ],
)
def test_sandbox_env_overrides(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    env_name: str,
    env_value: str,
    field: str,
    expected: object,
) -> None:
    cfg = _load(tmp_path, monkeypatch, env={env_name: env_value})

    assert getattr(cfg.sandbox, field) == expected


# 功能：验证系统环境变量优先级高于 [sandbox] TOML 值
# 设计：TOML 写 enabled = true 同时 env 写 false，断言最终为 env 值（四级优先链约束）
def test_sandbox_env_priority_over_toml(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _load(
        tmp_path,
        monkeypatch,
        toml_body="[sandbox]\nenabled = true\n",
        env={"KAMA_SANDBOX_ENABLED": "false"},
    )

    assert cfg.sandbox.enabled is False


# 功能：验证 KAMA_SANDBOX_EXEC_TIMEOUT_S 非整数或非正数时硬退出
# 设计：参数化 (坏值, 错误消息片段)，覆盖非整数与正数约束两条分支
@pytest.mark.parametrize(
    ("bad_value", "fragment"),
    [
        ("abc", "must be an integer"),
        ("0", "must be a positive integer"),
        ("-5", "must be a positive integer"),
    ],
)
def test_sandbox_env_invalid_timeout_exits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    bad_value: str,
    fragment: str,
) -> None:
    with pytest.raises(SystemExit) as exc:
        _load(
            tmp_path,
            monkeypatch,
            env={"KAMA_SANDBOX_EXEC_TIMEOUT_S": bad_value},
        )

    assert "KAMA_SANDBOX_EXEC_TIMEOUT_S" in str(exc.value)
    assert fragment in str(exc.value)
