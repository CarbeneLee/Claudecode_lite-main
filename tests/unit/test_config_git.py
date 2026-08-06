from __future__ import annotations

from pathlib import Path

import pytest

from kama_claude.core.config import KamaConfig, get_config

_GIT_ENV_VARS = (
    "KAMA_GIT_ENABLED",
    "KAMA_GIT_CHECKPOINT_MODE",
    "KAMA_GIT_BRANCH_PREFIX",
    "KAMA_GIT_MODE",
    "KAMA_GIT_AUTO_ROLLBACK_ON_FAIL",
)


# 用临时 TOML 加载配置；可选环境变量覆盖，其余 git 变量清除排除宿主干扰
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
    for name in _GIT_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    for name, value in (env or {}).items():
        monkeypatch.setenv(name, value)
    return get_config()


# 功能：验证无 [git] 配置时 cfg.git 保持内建默认值
# 设计：参数化 (字段, 期望值) 对，用 getattr 统一断言，避免为纯数据类写重复测试
@pytest.mark.parametrize(
    ("field", "expected"),
    [
        ("enabled", True),
        ("checkpoint_mode", "per_run"),
        ("branch_prefix", "agent"),
        ("mode", "branch"),
        ("auto_rollback_on_fail", False),
        ("author", "KamaClaude Agent <agent@kama.local>"),
        ("checkpoint_namespace", "refs/kama"),
        ("squash_on_finalize", True),
        ("keep_checkpoint_refs", True),
        ("rollback_strategy", "reset"),
    ],
)
def test_git_defaults(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    expected: object,
) -> None:
    cfg = _load(tmp_path, monkeypatch)

    assert getattr(cfg.git, field) == expected


# 功能：验证 [git] 组显式值被解析到 cfg.git 对应字段
# 设计：参数化单字段覆盖，每次只写一个 key，隔离验证解析路径
@pytest.mark.parametrize(
    ("key", "toml_value", "field", "expected"),
    [
        ("enabled", "false", "enabled", False),
        ("checkpoint_mode", '"per_step"', "checkpoint_mode", "per_step"),
        ("branch_prefix", '"task"', "branch_prefix", "task"),
        ("mode", '"none"', "mode", "none"),
        ("auto_rollback_on_fail", "true", "auto_rollback_on_fail", True),
        ("rollback_strategy", '"revert"', "rollback_strategy", "revert"),
    ],
)
def test_git_toml_explicit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    key: str,
    toml_value: str,
    field: str,
    expected: object,
) -> None:
    cfg = _load(tmp_path, monkeypatch, toml_body=f"[git]\n{key} = {toml_value}\n")

    assert getattr(cfg.git, field) == expected


# 功能：验证 [git] 未知 key 触发硬退出（与其他配置组一致的严格校验）
# 设计：写 git.foo = 1，断言 SystemExit 消息含未知 key 名
def test_git_toml_unknown_key_exits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(SystemExit) as exc:
        _load(tmp_path, monkeypatch, toml_body="[git]\nfoo = 1\n")

    assert "Unknown [git] keys: foo" in str(exc.value)


# 功能：验证 [git] 非 table 值触发硬退出
# 设计：顶层写 git = "x" 标量，断言表类型错误消息
def test_git_toml_not_a_table_exits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(SystemExit) as exc:
        _load(tmp_path, monkeypatch, toml_body='git = "x"\n')

    assert "Config error: [git] must be a table" in str(exc.value)


# 功能：验证 [git] 字段类型与取值约束校验硬退出
# 设计：参数化 (坏值, 错误消息片段)，覆盖 bool/str 类型与枚举取值约束
@pytest.mark.parametrize(
    ("key", "bad_value", "fragment"),
    [
        ("enabled", "'yes'", "git.enabled must be a boolean"),
        ("checkpoint_mode", "123", "git.checkpoint_mode must be a string"),
        ("branch_prefix", "123", "git.branch_prefix must be a string"),
        ("mode", "123", "git.mode must be a string"),
        ("auto_rollback_on_fail", "'yes'", "git.auto_rollback_on_fail must be a boolean"),
        ("checkpoint_mode", '"weekly"', "checkpoint_mode must be one of"),
        ("mode", '"bare"', "mode must be one of"),
        ("rollback_strategy", '"squash"', "rollback_strategy must be one of"),
    ],
)
def test_git_toml_invalid_exits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    key: str,
    bad_value: str,
    fragment: str,
) -> None:
    with pytest.raises(SystemExit) as exc:
        _load(tmp_path, monkeypatch, toml_body=f"[git]\n{key} = {bad_value}\n")

    assert fragment in str(exc.value)


# 功能：验证 KAMA_GIT_* 环境变量覆盖对应字段
# 设计：参数化 (env 名, 值, 字段, 期望) 对，隔离验证每条解析路径
@pytest.mark.parametrize(
    ("env_name", "env_value", "field", "expected"),
    [
        ("KAMA_GIT_ENABLED", "false", "enabled", False),
        ("KAMA_GIT_CHECKPOINT_MODE", "per_step", "checkpoint_mode", "per_step"),
        ("KAMA_GIT_BRANCH_PREFIX", "task", "branch_prefix", "task"),
        ("KAMA_GIT_MODE", "none", "mode", "none"),
        ("KAMA_GIT_AUTO_ROLLBACK_ON_FAIL", "true", "auto_rollback_on_fail", True),
    ],
)
def test_git_env_overrides(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    env_name: str,
    env_value: str,
    field: str,
    expected: object,
) -> None:
    cfg = _load(tmp_path, monkeypatch, env={env_name: env_value})

    assert getattr(cfg.git, field) == expected


# 功能：验证系统环境变量优先级高于 [git] TOML 值
# 设计：TOML 写 enabled = true 同时 env 写 false，断言最终为 env 值（四级优先链约束）
def test_git_env_priority_over_toml(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _load(
        tmp_path,
        monkeypatch,
        toml_body="[git]\nenabled = true\n",
        env={"KAMA_GIT_ENABLED": "false"},
    )

    assert cfg.git.enabled is False


# 功能：验证 KAMA_GIT_CHECKPOINT_MODE 非法枚举值时硬退出
# 设计：参数化 (坏值, 错误消息片段)，覆盖非法枚举与错误变量名两条分支
@pytest.mark.parametrize(
    ("bad_value", "fragment"),
    [
        ("weekly", "checkpoint_mode must be one of"),
        ("", "checkpoint_mode must be one of"),
    ],
)
def test_git_env_invalid_mode_exits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    bad_value: str,
    fragment: str,
) -> None:
    with pytest.raises(SystemExit) as exc:
        _load(
            tmp_path,
            monkeypatch,
            env={"KAMA_GIT_CHECKPOINT_MODE": bad_value},
        )

    assert "KAMA_GIT_CHECKPOINT_MODE" in str(exc.value)
    assert fragment in str(exc.value)
