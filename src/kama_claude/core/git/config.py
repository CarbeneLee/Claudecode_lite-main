from __future__ import annotations

from dataclasses import dataclass

_DEFAULT_AUTHOR = "KamaClaude Agent <agent@kama.local>"
_DEFAULT_NAMESPACE = "refs/kama"
_CHECKPOINT_MODES = ("none", "per_run", "per_step")
_MODES = ("branch", "worktree", "none")
_ROLLBACK_STRATEGIES = ("reset", "revert")


@dataclass(frozen=True)
class GitConfig:
    # 是否启用 git 工作流；false 时行为与无 git 能力完全一致
    enabled: bool = True
    # checkpoint 粒度："none" | "per_run"（run 边界）| "per_step"（每 step）
    checkpoint_mode: str = "per_run"
    # task 分支前缀：agent/<task_id>
    branch_prefix: str = "agent"
    # 隔离方式："branch"（默认）| "worktree" | "none"
    mode: str = "branch"
    # run 失败时自动回滚到 baseline
    auto_rollback_on_fail: bool = False
    # agent 产生的 commit 作者身份
    author: str = _DEFAULT_AUTHOR
    # 内部 checkpoint 引用命名空间（不进用户分支 log，本地 refs 不 push）
    checkpoint_namespace: str = _DEFAULT_NAMESPACE
    # git_commit 时把内部 checkpoint 压缩为单个可见 commit
    squash_on_finalize: bool = True
    # close 时是否保留 checkpoint refs（crash recovery / 审计）
    keep_checkpoint_refs: bool = True
    # 回滚策略："reset"（仅 agent 分支）| "revert"（保留历史）
    rollback_strategy: str = "reset"

    def __post_init__(self) -> None:
        # 枚举类字段在构造时即拒绝非法取值，避免配置错误扩散到运行时
        if self.checkpoint_mode not in _CHECKPOINT_MODES:
            raise ValueError(
                f"checkpoint_mode must be one of {_CHECKPOINT_MODES}, "
                f"got {self.checkpoint_mode!r}"
            )
        if self.mode not in _MODES:
            raise ValueError(f"mode must be one of {_MODES}, got {self.mode!r}")
        if self.rollback_strategy not in _ROLLBACK_STRATEGIES:
            raise ValueError(
                f"rollback_strategy must be one of {_ROLLBACK_STRATEGIES}, "
                f"got {self.rollback_strategy!r}"
            )
