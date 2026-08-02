from __future__ import annotations

import pytest

from kama_claude.core.git.errors import (
    CheckpointFailedError,
    CommitFailedError,
    DirtyWorkspaceError,
    GitError,
    GitLockError,
    GitUnavailableError,
    MergeConflictError,
    RepositoryNotFoundError,
    RollbackFailedError,
    classify_cli_error,
)

_ALL_ERROR_TYPES = [
    GitUnavailableError,
    RepositoryNotFoundError,
    DirtyWorkspaceError,
    GitLockError,
    CheckpointFailedError,
    CommitFailedError,
    RollbackFailedError,
    MergeConflictError,
]


# 功能：验证八个稳定 git 异常类型都继承 GitError 基类并可携带 detail
# 设计：参数化遍历全部异常类型，断言基类归属与 detail 透传，确保分类与调用方契约稳定
@pytest.mark.parametrize("exc_type", _ALL_ERROR_TYPES)
def test_git_errors_share_base_and_detail(exc_type: type[GitError]) -> None:
    exc = exc_type("boom", detail="stderr content")
    assert isinstance(exc, GitError)
    assert exc.detail == "stderr content"
    assert str(exc) == "boom"


# 功能：验证非 git 仓库的 stderr 被分类为 RepositoryNotFoundError
# 设计：参数化不同措辞的 git stderr，覆盖大小写与路径前缀差异，确保分类不依赖精确匹配
@pytest.mark.parametrize(
    "stderr",
    [
        "fatal: not a git repository (or any of the parent directories): .git",
        "fatal: not a git repository: '/home/user/ws'",
        "Not a git repository: .git",
    ],
)
def test_classify_repository_not_found(stderr: str) -> None:
    assert isinstance(classify_cli_error(stderr), RepositoryNotFoundError)


# 功能：验证 index 锁冲突的 stderr 被分类为 GitLockError
# 设计：参数化不同锁路径措辞，覆盖 index.lock 语义识别
@pytest.mark.parametrize(
    "stderr",
    [
        "fatal: Unable to create '/ws/.git/index.lock': File exists.",
        "error: cannot lock ref 'refs/heads/agent/t1': Unable to create '.git/refs/heads/agent/t1.lock'",
        "fatal: Unable to create '.git/ORIG_HEAD.lock': File exists.",
    ],
)
def test_classify_git_lock(stderr: str) -> None:
    assert isinstance(classify_cli_error(stderr), GitLockError)


# 功能：验证 git CLI 缺失的 stderr 被分类为 GitUnavailableError
# 设计：command not found 语义识别，与仓库/锁类错误区分
@pytest.mark.parametrize(
    "stderr",
    [
        "sh: git: command not found",
        "git: command not found",
    ],
)
def test_classify_git_unavailable(stderr: str) -> None:
    assert isinstance(classify_cli_error(stderr), GitUnavailableError)


# 功能：验证冲突类 stderr 被分类为 MergeConflictError
# 设计：参数化 revert/merge 冲突标记，识别冲突语义
@pytest.mark.parametrize(
    "stderr",
    [
        "CONFLICT (content): Merge conflict in src/app.py",
        "error: could not revert 1a2b3c4... Merge conflict in tests/test_x.py",
    ],
)
def test_classify_merge_conflict(stderr: str) -> None:
    assert isinstance(classify_cli_error(stderr), MergeConflictError)


# 功能：验证提交类失败 stderr 被分类为 CommitFailedError
# 设计：参数化签名失败/空 ident/hook 拒绝三类提交路径故障
@pytest.mark.parametrize(
    "stderr",
    [
        "error: gpg failed to sign the data",
        "fatal: empty ident name (for <agent@kama.local>) not allowed",
        "error: failed to run pre-commit hook",
    ],
)
def test_classify_commit_failed(stderr: str) -> None:
    assert isinstance(classify_cli_error(stderr), CommitFailedError)


# 功能：验证对象库写入类故障 stderr 被分类为 CheckpointFailedError
# 设计：参数化索引损坏/只读对象库/磁盘配额三类写入故障，与 CLI 缺失（unavailable）区分
@pytest.mark.parametrize(
    "stderr",
    [
        "fatal: index file corrupt",
        "fatal: object database .git/objects is read-only",
        "error: unable to write new index file",
        "fatal: disk quota exceeded",
    ],
)
def test_classify_checkpoint_failed(stderr: str) -> None:
    assert isinstance(classify_cli_error(stderr), CheckpointFailedError)


# 功能：验证回滚类失败 stderr 被分类为 RollbackFailedError
# 设计：参数化 reset/revert 解包失败与本地修改保护拒绝两类恢复路径故障
@pytest.mark.parametrize(
    "stderr",
    [
        "fatal: failed to unpack trees",
        "fatal: Could not reset index file to revision 'abc123'.",
        "error: The following untracked working tree files would be overwritten by checkout",
        "error: Your local changes to the following files would be overwritten by merge",
        "fatal: revert failed",
    ],
)
def test_classify_rollback_failed(stderr: str) -> None:
    assert isinstance(classify_cli_error(stderr), RollbackFailedError)


# 功能：验证未知 stderr 内容归入 CommitFailedError 兜底
# 设计：用与任何已知关键词无关的文本覆盖兜底路径，防止未分类错误泄漏为其他类型
def test_classify_fallback_to_commit_failed() -> None:
    assert isinstance(classify_cli_error(""), CommitFailedError)
    assert isinstance(classify_cli_error("git: something unexpected happened"), CommitFailedError)
