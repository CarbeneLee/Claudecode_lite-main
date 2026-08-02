from __future__ import annotations


class GitError(Exception):
    """git 错误基类：携带可选详情，供错误分类与日志使用"""

    def __init__(self, message: str, *, detail: str = "") -> None:
        self.detail = detail
        super().__init__(message)


class GitUnavailableError(GitError):
    """git CLI 不存在或 git 子进程执行超时"""


class RepositoryNotFoundError(GitError):
    """workspace 不是 git 仓库"""


class DirtyWorkspaceError(GitError):
    """用户拒绝快照式 baseline 时的工作树非干净状态"""


class GitLockError(GitError):
    """.git 下存在 index.lock 等锁文件，另一 git 进程正在操作"""


class CheckpointFailedError(GitError):
    """checkpoint 对象库写入失败（磁盘满/索引损坏/只读），与 CLI 不可用区分"""


class CommitFailedError(GitError):
    """commit 失败（hook 拒绝/签名失败/权限/空 commit）"""


class RollbackFailedError(GitError):
    """rollback 失败（reset/revert 失败、文件被锁）"""


class MergeConflictError(GitError):
    """merge/revert 冲突，或 finalize 前置校验失败（用户中途 commit）"""


# 将 git CLI stderr 关键词分类为稳定 git 异常；未知内容归入 commit_failed
def classify_cli_error(stderr: str) -> GitError:
    lowered = stderr.lower()
    if "not a git repository" in lowered:
        return RepositoryNotFoundError("workspace is not a git repository", detail=stderr)
    if ".lock" in lowered:
        return GitLockError("git index or ref locked by another process", detail=stderr)
    if "command not found" in lowered:
        return GitUnavailableError("git CLI unavailable", detail=stderr)
    if "conflict" in lowered:
        return MergeConflictError("merge or revert conflict", detail=stderr)
    if (
        "failed to sign" in lowered
        or "empty ident" in lowered
        or "pre-commit hook" in lowered
        or "cannot open .git/commit_editmsg" in lowered
    ):
        return CommitFailedError("git commit failed", detail=stderr)
    if (
        "index file corrupt" in lowered
        or "object database" in lowered
        or "disk quota" in lowered
        or "read-only file system" in lowered
        or "unable to write new index file" in lowered
    ):
        return CheckpointFailedError("checkpoint object write failed", detail=stderr)
    if (
        "could not reset" in lowered
        or "failed to unpack" in lowered
        or "untracked working tree files would be overwritten" in lowered
    ):
        return RollbackFailedError("git rollback failed", detail=stderr)
    return CommitFailedError("git command failed", detail=stderr)
