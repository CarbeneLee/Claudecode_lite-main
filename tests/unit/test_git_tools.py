from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from kama_claude.core.git.config import GitConfig
from kama_claude.core.git.errors import (
    CommitFailedError,
    MergeConflictError,
    RollbackFailedError,
)
from kama_claude.core.git.manager import GitManager
from kama_claude.core.git.runtime import GitCliRuntime
from kama_claude.core.git.tools import (
    GitCheckpointTool,
    GitCommitTool,
    GitDiffTool,
    GitRollbackTool,
    GitStatusTool,
)
from kama_claude.core.permissions.policy import (
    DEFAULT_POLICIES,
    PermissionDecision,
    evaluate,
)

_IDENT_ENV = {
    **os.environ,
    "GIT_AUTHOR_NAME": "user",
    "GIT_AUTHOR_EMAIL": "u@u",
    "GIT_COMMITTER_NAME": "user",
    "GIT_COMMITTER_EMAIL": "u@u",
}


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(["git", *args], cwd=repo, capture_output=True, check=True)


# 真实 git 仓库夹具：main 分支 + 一个 init commit
@pytest.fixture
def repo(tmp_path: Path) -> Path:
    r = tmp_path / "repo"
    r.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main", str(r)], check=True)
    (r / "f.txt").write_text("v1\n", encoding="utf-8")
    _git(r, "add", "f.txt")
    subprocess.run(["git", "commit", "-qm", "init"], cwd=r, check=True, env=_IDENT_ENV)
    return r


def _manager(repo: Path, **overrides: object) -> GitManager:
    return GitManager(config=GitConfig(**overrides), workspace_root=repo)


# 功能：验证五个 git 工具的元数据契约（名称/描述/schema 形状）
# 设计：参数化工具类，断言 registry 面向 LLM 暴露的字段齐全且 schema 为 object
@pytest.mark.parametrize(
    ("tool_cls", "expected_name"),
    [
        (GitStatusTool, "git_status"),
        (GitDiffTool, "git_diff"),
        (GitCheckpointTool, "git_checkpoint"),
        (GitCommitTool, "git_commit"),
        (GitRollbackTool, "git_rollback"),
    ],
)
def test_tool_metadata(tool_cls: type, expected_name: str) -> None:
    assert tool_cls.name == expected_name
    assert tool_cls.description
    assert tool_cls.input_schema["type"] == "object"


# 功能：验证权限策略默认值——只读工具 auto_allow，写操作 ASK
# 设计：直接 evaluate 五个工具的空参数，断言 ALLOW/ASK 归属与设计 §7.1 一致
@pytest.mark.parametrize(
    ("tool_name", "expected"),
    [
        ("git_status", PermissionDecision.ALLOW),
        ("git_diff", PermissionDecision.ALLOW),
        ("git_checkpoint", PermissionDecision.ASK),
        ("git_commit", PermissionDecision.ASK),
        ("git_rollback", PermissionDecision.ASK),
    ],
)
def test_tool_policy_defaults(tool_name: str, expected: PermissionDecision) -> None:
    assert tool_name in DEFAULT_POLICIES
    assert evaluate(tool_name, {}) == expected


# 功能：验证 git_status 输出干净工作树（分支名 + clean 标记）
# 设计：干净仓库上 invoke，断言内容含分支与 clean 且非错误
async def test_git_status_clean(repo: Path) -> None:
    tool = GitStatusTool(_manager(repo))
    result = await tool.invoke({})
    assert not result.is_error
    assert "main" in result.content
    assert "clean" in result.content


# 功能：验证 git_status 输出脏条目（修改文件进入输出）
# 设计：写文件后 invoke，断言条目行含 f.txt
async def test_git_status_dirty_entry(repo: Path) -> None:
    (repo / "f.txt").write_text("v2\n", encoding="utf-8")
    tool = GitStatusTool(_manager(repo))
    result = await tool.invoke({})
    assert not result.is_error
    assert "f.txt" in result.content
    assert "clean" not in result.content


# 功能：验证 git_diff 输出 stat 摘要（改动文件出现在 diff）
# 设计：写文件后 invoke，断言内容含文件名
async def test_git_diff_after_edit(repo: Path) -> None:
    (repo / "f.txt").write_text("v2\n", encoding="utf-8")
    tool = GitDiffTool(_manager(repo))
    result = await tool.invoke({})
    assert not result.is_error
    assert "f.txt" in result.content


# 功能：验证 git_checkpoint 自动取下一步号（baseline step 0 之后为 step 1）
# 设计：force baseline 后 invoke，断言 ref r1/1 存在且 label 透传
async def test_git_checkpoint_auto_step(repo: Path) -> None:
    manager = _manager(repo)
    await manager.ensure_task_branch("t1")
    await manager.create_checkpoint("r1", 0, "baseline", force=True)
    (repo / "f.txt").write_text("v2\n", encoding="utf-8")
    tool = GitCheckpointTool(manager, "r1")
    result = await tool.invoke({"label": "work-in-progress"})
    assert not result.is_error
    assert "step 1" in result.content
    refs = _git(repo, "for-each-ref", "refs/kama/checkpoints/r1").stdout.decode()
    assert "refs/kama/checkpoints/r1/1" in refs
    cp = await manager.get_checkpoint("r1", 1)
    assert cp is not None
    assert cp.label == "work-in-progress"


# 功能：验证干净工作树上 git_checkpoint 返回说明性结果而非错误
# 设计：无改动时 invoke，断言非错误且内容含 clean
async def test_git_checkpoint_clean_noop(repo: Path) -> None:
    manager = _manager(repo)
    await manager.ensure_task_branch("t1")
    tool = GitCheckpointTool(manager, "r1")
    result = await tool.invoke({"label": "noop"})
    assert not result.is_error
    assert "clean" in result.content


# 功能：验证 git_commit 触发 finalize——squash 为单个可见 commit 且用户分支零污染
# 设计：baseline + checkpoint 后 finalize，断言 agent 分支 log 恰好 +1、main 不变
async def test_git_commit_finalize_squash(repo: Path) -> None:
    manager = _manager(repo)
    await manager.ensure_task_branch("t1")
    await manager.create_checkpoint("r1", 0, "baseline", force=True)
    (repo / "f.txt").write_text("v2\n", encoding="utf-8")
    await manager.create_checkpoint("r1", 1, "step-1")
    tool = GitCommitTool(manager, "r1")
    result = await tool.invoke({"summary": "finish the task"})
    assert not result.is_error
    assert "finish the task" in result.content
    agent_log = _git(repo, "log", "--format=%s", "agent/t1").stdout.decode()
    assert agent_log.strip().splitlines() == ["kama: finish the task", "init"]
    main_log = _git(repo, "log", "--format=%s", "main").stdout.decode()
    assert "kama" not in main_log  # 用户分支零污染
    refs = _git(repo, "for-each-ref", "refs/kama/checkpoints/r1").stdout.decode()
    assert "refs/kama/checkpoints/r1/1" in refs  # 内部 commit 对象经 ref 保留


# 功能：验证 finalize 在非 agent 分支被拒绝（reset --soft 会移动用户分支指针）
# 设计：main 分支上 git_commit 抛 RollbackFailedError
async def test_git_commit_refused_on_user_branch(repo: Path) -> None:
    manager = _manager(repo)
    await manager.ensure_ready()
    tool = GitCommitTool(manager, "r1")
    with pytest.raises(RollbackFailedError, match="refused"):
        await tool.invoke({"summary": "should not commit"})


# 功能：验证用户中途 commit 时 finalize 拒绝（防 squash 吞用户提交）
# 设计：checkpoint 后用户直接在 agent 分支 commit，git_commit 抛 MergeConflictError
async def test_git_commit_refused_on_interleaved_user_commit(repo: Path) -> None:
    manager = _manager(repo)
    await manager.ensure_task_branch("t1")
    await manager.create_checkpoint("r1", 0, "baseline", force=True)
    (repo / "f.txt").write_text("v2\n", encoding="utf-8")
    await manager.create_checkpoint("r1", 1, "step-1")
    (repo / "f.txt").write_text("v3\n", encoding="utf-8")
    subprocess.run(
        ["git", "commit", "-am", "user hotfix"], cwd=repo, check=True, env=_IDENT_ENV
    )
    tool = GitCommitTool(manager, "r1")
    with pytest.raises(MergeConflictError, match="refused"):
        await tool.invoke({"summary": "cannot finalize"})


# 功能：验证 git_rollback 按 step 恢复工作树
# 设计：checkpoint step 1 后继续修改，rollback step 1 断言文件回到 v2
async def test_git_rollback_restores_step(repo: Path) -> None:
    manager = _manager(repo)
    await manager.ensure_task_branch("t1")
    await manager.create_checkpoint("r1", 0, "baseline", force=True)
    (repo / "f.txt").write_text("v2\n", encoding="utf-8")
    await manager.create_checkpoint("r1", 1, "step-1")
    (repo / "f.txt").write_text("v3\n", encoding="utf-8")
    tool = GitRollbackTool(manager, "r1")
    result = await tool.invoke({"step": 1})
    assert not result.is_error
    assert "step 1" in result.content
    assert (repo / "f.txt").read_text(encoding="utf-8") == "v2\n"


# 功能：验证未知 step 的 rollback 返回 not_found 错误而非抛异常
# 设计：不存在 step 99 时 invoke 返回 is_error 且 error_type 为 not_found
async def test_git_rollback_unknown_step(repo: Path) -> None:
    manager = _manager(repo)
    await manager.ensure_task_branch("t1")
    tool = GitRollbackTool(manager, "r1")
    result = await tool.invoke({"step": 99})
    assert result.is_error
    assert result.error_type == "not_found"
    assert "99" in result.content


_FAKE_GIT_SCRIPT = """\
#!/bin/sh
# 记录每个 argv 一行到 FAKE_GIT_LOG；按子命令注入可控输出
printf '%s\\n' "$@" >> "$FAKE_GIT_LOG"
if [ "$1" = "-C" ]; then shift 2; fi  # runtime 前缀 -C <workspace>
cmd="$1"
case "$cmd" in
  status)
    [ -n "$FAKE_GIT_BRANCH" ] && printf '## %s\\n' "$FAKE_GIT_BRANCH"
    if [ "$FAKE_GIT_DIRTY" = "1" ]; then echo " M f.txt"; fi ;;
  rev-parse)
    if [ "$2" = "HEAD" ]; then printf '%s\\n' "$FAKE_GIT_HEAD"; fi ;;
  diff)
    printf '%s\\n' "$FAKE_GIT_DIFF" ;;
esac
exit 0
"""


# 功能：验证 finalize 的 git 调用顺序（分支检查 → 范围校验 → reset --soft → commit）
# 设计：fake git 记录 argv，断言 log/reset/commit 相对顺序与关键参数
async def test_finalize_call_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = tmp_path / "git"
    script.write_text(_FAKE_GIT_SCRIPT, encoding="utf-8")
    script.chmod(0o755)
    log = tmp_path / "log.txt"
    monkeypatch.setenv("FAKE_GIT_LOG", str(log))
    monkeypatch.setenv("FAKE_GIT_BRANCH", "agent/t1")
    monkeypatch.setenv("FAKE_GIT_DIRTY", "1")
    monkeypatch.setenv("FAKE_GIT_HEAD", "a" * 40)
    workspace = tmp_path / "ws"
    workspace.mkdir()
    manager = GitManager(
        config=GitConfig(),
        workspace_root=workspace,
        runtime=GitCliRuntime(git_executable=str(script), workspace_root=workspace),
    )
    tool = GitCommitTool(manager, "r1")
    await tool.invoke({"summary": "ship it"})

    lines = log.read_text(encoding="utf-8").splitlines()
    i_log = lines.index("log")
    i_reset = lines.index("reset")
    i_commit = lines.index("commit")
    assert i_log < i_reset < i_commit  # 范围校验在前，reset --soft 先于 commit
    assert lines[i_reset + 1] == "--soft"
    assert lines[i_commit + 1] == "-m"
    assert lines[i_commit + 2] == "kama: ship it"


# 功能：验证 finalize 提交前扫描 staged diff，命中 secret 时拒绝提交且不泄漏值
# 设计：fake git 的 diff --cached 输出含 AWS key；断言 CommitFailedError 的
#       detail 含 file+line 与规则名、不含 secret 值，且 git commit 未被执行
async def test_finalize_rejects_secret_in_staged_diff(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = tmp_path / "git"
    script.write_text(_FAKE_GIT_SCRIPT, encoding="utf-8")
    script.chmod(0o755)
    log = tmp_path / "log.txt"
    monkeypatch.setenv("FAKE_GIT_LOG", str(log))
    monkeypatch.setenv("FAKE_GIT_BRANCH", "agent/t1")
    monkeypatch.setenv("FAKE_GIT_DIRTY", "1")
    monkeypatch.setenv("FAKE_GIT_HEAD", "a" * 40)
    monkeypatch.setenv(
        "FAKE_GIT_DIFF",
        "diff --git a/creds.py b/creds.py\n"
        "--- a/creds.py\n"
        "+++ b/creds.py\n"
        "@@ -1 +1,2 @@\n"
        " ok\n"
        "+AKIAIOSFODNN7EXAMPLE\n",
    )
    workspace = tmp_path / "ws"
    workspace.mkdir()
    manager = GitManager(
        config=GitConfig(),
        workspace_root=workspace,
        runtime=GitCliRuntime(git_executable=str(script), workspace_root=workspace),
    )
    tool = GitCommitTool(manager, "r1")

    with pytest.raises(CommitFailedError) as exc:
        await tool.invoke({"summary": "ship it"})

    assert "creds.py" in exc.value.detail
    assert ":2" in exc.value.detail  # 命中行号
    assert "AKIAIOSFODNN7EXAMPLE" not in exc.value.detail  # 不泄漏 secret 值
    lines = log.read_text(encoding="utf-8").splitlines()
    assert "commit" not in lines  # 拒绝提交：git commit 未执行
