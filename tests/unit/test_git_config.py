from __future__ import annotations

import pytest

from kama_claude.core.git.config import GitConfig


# 功能：验证 GitConfig 的默认值契约：默认启用、per_run 模式、默认分支前缀、reset 回滚
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
def test_git_config_defaults(field: str, expected: object) -> None:
    assert getattr(GitConfig(), field) == expected


# 功能：验证 GitConfig 支持全字段显式构造覆盖
# 设计：显式传全部字段断言值被保留，验证构造路径与默认路径一致
def test_git_config_explicit_override() -> None:
    cfg = GitConfig(
        enabled=False,
        checkpoint_mode="per_step",
        branch_prefix="worker",
        mode="worktree",
        auto_rollback_on_fail=True,
        author="CI Bot <ci@kama.local>",
        checkpoint_namespace="refs/ci-checkpoints",
        squash_on_finalize=False,
        keep_checkpoint_refs=False,
        rollback_strategy="revert",
    )
    assert cfg.enabled is False
    assert cfg.checkpoint_mode == "per_step"
    assert cfg.branch_prefix == "worker"
    assert cfg.mode == "worktree"
    assert cfg.auto_rollback_on_fail is True
    assert cfg.author == "CI Bot <ci@kama.local>"
    assert cfg.checkpoint_namespace == "refs/ci-checkpoints"
    assert cfg.squash_on_finalize is False
    assert cfg.keep_checkpoint_refs is False
    assert cfg.rollback_strategy == "revert"


# 功能：验证枚举类字段的非法取值在构造时即被拒绝
# 设计：参数化 (字段, 非法值) 对，覆盖 checkpoint_mode/mode/rollback_strategy 三组取值约束
@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("checkpoint_mode", "always"),
        ("checkpoint_mode", ""),
        ("mode", "merge"),
        ("mode", ""),
        ("rollback_strategy", "delete"),
        ("rollback_strategy", ""),
    ],
)
def test_git_config_invalid_enum_rejected(field: str, bad_value: str) -> None:
    with pytest.raises(ValueError):
        GitConfig(**{field: bad_value})
