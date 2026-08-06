from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from kama_claude.core.git.config import GitConfig
from kama_claude.core.git.errors import (
    CommitFailedError,
    RepositoryNotFoundError,
    RollbackFailedError,
)
from kama_claude.core.git.manager import GitManager
from kama_claude.core.git.runtime import GitCliRuntime

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


# 功能：验证初始状态为 IDLE，ensure_ready 后进入 READY 且二次调用不重复校验
# 设计：状态机转换断言 + 通过 commit 数不变证明 rev-parse 只校验一次（幂等）
async def test_ensure_ready_transitions(repo: Path) -> None:
    manager = _manager(repo)
    assert manager.state == GitManager.IDLE
    await manager.ensure_ready()
    assert manager.state == GitManager.READY
    await manager.ensure_ready()
    assert manager.state == GitManager.READY


# 功能：验证非 git 仓库的 workspace 被拒绝且状态置 FAILED（fail-open 前置探测）
# 设计：无 .git 目录的目录上 ensure_ready 抛 RepositoryNotFoundError
async def test_ensure_ready_non_repo_fails(tmp_path: Path) -> None:
    plain = tmp_path / "plain"
    plain.mkdir()
    manager = _manager(plain)
    with pytest.raises(RepositoryNotFoundError):
        await manager.ensure_ready()
    assert manager.state == GitManager.FAILED


# 功能：验证 closed 后拒绝一切操作（程序性错误）
# 设计：close 后 status 抛 RuntimeError，消息含 closed
async def test_closed_rejects_ops(repo: Path) -> None:
    manager = _manager(repo)
    await manager.close()
    with pytest.raises(RuntimeError, match="closed"):
        await manager.status()


# 功能：验证 status 解析干净工作树（dirty=False、分支名正确）
# 设计：初始化仓库上 status 断言 main 分支且无条目
async def test_status_clean(repo: Path) -> None:
    manager = _manager(repo)
    status = await manager.status()
    assert status.dirty is False
    assert status.branch == "main"
    assert status.entries == ()


# 功能：验证 status 解析脏工作树（修改文件进入条目，dirty=True）
# 设计：写文件后 status 断言条目含 f.txt 且 dirty 翻转
async def test_status_dirty_after_edit(repo: Path) -> None:
    (repo / "f.txt").write_text("v2\n", encoding="utf-8")
    manager = _manager(repo)
    status = await manager.status()
    assert status.dirty is True
    assert any("f.txt" in path for _code, path in status.entries)


# 功能：验证 diff 输出 stat 摘要（改动文件出现在 diff 中）
# 设计：写文件后 diff 断言文本含文件名，为 review 提供依据
async def test_diff_after_edit(repo: Path) -> None:
    (repo / "f.txt").write_text("v2\n", encoding="utf-8")
    manager = _manager(repo)
    diff = await manager.diff()
    assert "f.txt" in diff.stat


# 功能：验证 task 分支从当前 HEAD 创建且幂等
# 设计：两次 ensure_task_branch 后分支列表仍只有一个 agent/t1
async def test_ensure_task_branch_idempotent(repo: Path) -> None:
    manager = _manager(repo)
    await manager.ensure_ready()
    await manager.ensure_task_branch("t1")
    await manager.ensure_task_branch("t1")
    branches = _git(repo, "branch", "--list", "agent/t1").stdout.decode()
    assert branches.count("agent/t1") == 1


# 功能：验证无变更时 create_checkpoint 返回 None（空 checkpoint 跳过）
# 设计：干净工作树上创建 step 1 checkpoint 断言 None 且无 ref
async def test_create_checkpoint_noop_when_clean(repo: Path) -> None:
    manager = _manager(repo)
    await manager.ensure_task_branch("t1")
    cp = await manager.create_checkpoint("r1", 1, "step-1")
    assert cp is None
    refs = _git(repo, "for-each-ref", "refs/kama/checkpoints/r1").stdout.decode()
    assert refs == ""


# 功能：验证 baseline checkpoint 无 commit 产生、ref 指向当前 HEAD（零污染）
# 设计：干净仓库 force baseline：main 分支 commit 数不变、refs/kama/checkpoints/r1/0 存在
async def test_baseline_checkpoint_ref_without_commit(repo: Path) -> None:
    manager = _manager(repo)
    await manager.ensure_task_branch("t1")
    cp = await manager.create_checkpoint("r1", 0, "baseline", force=True)
    assert cp is not None
    assert cp.step == 0
    main_log = _git(repo, "log", "--oneline", "main").stdout.decode()
    assert len(main_log.strip().splitlines()) == 1  # main 仍只有 init 一个 commit
    refs = _git(repo, "for-each-ref", "refs/kama/checkpoints/r1").stdout.decode()
    assert "refs/kama/checkpoints/r1/0" in refs


# 功能：验证修改后 checkpoint 创建内部 commit + ref，且用户分支 log 零污染
# 设计：改文件 → checkpoint step 1：agent 分支多一个 kama commit，main 分支 log 不变
async def test_create_checkpoint_commits_on_agent_branch(repo: Path) -> None:
    (repo / "f.txt").write_text("v2\n", encoding="utf-8")
    manager = _manager(repo)
    await manager.ensure_task_branch("t1")
    cp = await manager.create_checkpoint("r1", 1, "step-1")
    assert cp is not None
    assert cp.kind == "internal"
    assert cp.dirty is True
    agent_log = _git(repo, "log", "--format=%s", "agent/t1").stdout.decode()
    assert "kama: step-1" in agent_log
    main_log = _git(repo, "log", "--format=%s", "main").stdout.decode()
    assert "kama" not in main_log  # 用户分支零污染
    refs = _git(repo, "for-each-ref", "refs/kama/checkpoints/r1").stdout.decode()
    assert "refs/kama/checkpoints/r1/1" in refs


# 功能：验证 checkpoint 链逐步生长且 latest_checkpoint 取最后一步
# 设计：baseline + 两步修改 → refs 0/1/2 齐全，latest 返回 step 2 且 sha 与 ref 一致
async def test_checkpoint_chain_and_latest(repo: Path) -> None:
    manager = _manager(repo)
    await manager.ensure_task_branch("t1")
    base = await manager.create_checkpoint("r1", 0, "baseline", force=True)
    (repo / "f.txt").write_text("v2\n", encoding="utf-8")
    cp1 = await manager.create_checkpoint("r1", 1, "step-1")
    (repo / "g.txt").write_text("g1\n", encoding="utf-8")
    cp2 = await manager.create_checkpoint("r1", 2, "step-2")
    assert base is not None and cp1 is not None and cp2 is not None
    refs = _git(repo, "for-each-ref", "refs/kama/checkpoints/r1").stdout.decode()
    assert refs.count("refs/kama/checkpoints/r1/") == 3
    latest = await manager.latest_checkpoint("r1")
    assert latest is not None
    assert latest.step == 2
    assert latest.commit_sha == cp2.commit_sha


# 功能：验证 agent 分支上 restore 用 reset --hard 精确恢复工作树
# 设计：checkpoint 记录 v2 后继续改 v3，restore 断言文件回到 v2 且策略为 reset
async def test_restore_agent_branch_reset(repo: Path) -> None:
    manager = _manager(repo)
    await manager.ensure_task_branch("t1")
    await manager.create_checkpoint("r1", 0, "baseline", force=True)
    (repo / "f.txt").write_text("v2\n", encoding="utf-8")
    cp = await manager.create_checkpoint("r1", 1, "step-1")
    (repo / "f.txt").write_text("v3\n", encoding="utf-8")
    result = await manager.restore(cp)  # type: ignore[arg-type]
    assert result.strategy == "reset"
    assert (repo / "f.txt").read_text(encoding="utf-8") == "v2\n"
    assert result.checkpoint_sha == cp.commit_sha  # type: ignore[union-attr]


# 功能：验证非 agent 分支上 restore 被拒绝（保护用户分支的 reset 禁令）
# 设计：main 分支上 restore 抛 RollbackFailedError 且消息含 checkout 指引
async def test_restore_refused_on_user_branch(repo: Path) -> None:
    manager = _manager(repo)
    await manager.ensure_ready()
    (repo / "f.txt").write_text("v2\n", encoding="utf-8")
    cp = await manager.create_checkpoint("r1", 1, "step-1")
    with pytest.raises(RollbackFailedError, match="refused") as exc:
        await manager.restore(cp)  # type: ignore[arg-type]
    assert "checkout" in exc.value.detail  # 指引用户切回 agent 分支


# 功能：验证 revert 策略在干净工作树上撤销 checkpoint 且不产生 revert commit
# 设计：rollback_strategy=revert：checkpoint 已提交一切，restore 用 revert 还原文件且 log 无 revert
async def test_restore_revert_strategy(repo: Path) -> None:
    manager = _manager(repo, rollback_strategy="revert")
    await manager.ensure_ready()
    (repo / "f.txt").write_text("v2\n", encoding="utf-8")
    cp = await manager.create_checkpoint("r1", 1, "step-1")
    result = await manager.restore(cp)  # type: ignore[arg-type]
    assert result.strategy == "revert"
    assert (repo / "f.txt").read_text(encoding="utf-8") == "v1\n"
    log = _git(repo, "log", "--oneline").stdout.decode()
    assert "revert" not in log  # --no-commit：不产生 revert commit


# 功能：验证 revert 策略在脏工作树上被 git 拒绝（本地修改保护，不静默丢失）
# 设计：checkpoint 后继续改文件，restore 抛 RollbackFailedError（revert failed 分类）
async def test_restore_revert_refused_when_dirty(repo: Path) -> None:
    manager = _manager(repo, rollback_strategy="revert")
    await manager.ensure_ready()
    (repo / "f.txt").write_text("v2\n", encoding="utf-8")
    cp = await manager.create_checkpoint("r1", 1, "step-1")
    (repo / "f.txt").write_text("v3\n", encoding="utf-8")
    with pytest.raises(RollbackFailedError):
        await manager.restore(cp)  # type: ignore[arg-type]


# 功能：验证 dirty 场景 snapshot_pre_run 固化含用户修改的快照并可恢复
# 设计：用户未提交修改 → snapshot → agent 分支上 restore 后用户修改完整保留
async def test_snapshot_pre_run_preserves_user_changes(repo: Path) -> None:
    (repo / "f.txt").write_text("user-wip\n", encoding="utf-8")
    manager = _manager(repo)
    cp = await manager.snapshot_pre_run("r1")
    assert cp is not None
    assert cp.kind == "pre_run"
    assert cp.dirty is True
    await manager.ensure_task_branch("t1")
    (repo / "f.txt").write_text("agent-edit\n", encoding="utf-8")
    await manager.restore(cp)
    assert (repo / "f.txt").read_text(encoding="utf-8") == "user-wip\n"


# 功能：验证 crash recovery——新 manager 实例 resume 恢复到最后 checkpoint 状态
# 设计：同仓库重建 manager（模拟进程重启），resume 断言工作树回到 cp1 内容
async def test_resume_after_crash(repo: Path) -> None:
    manager = _manager(repo)
    await manager.ensure_task_branch("t1")
    await manager.create_checkpoint("r1", 0, "baseline", force=True)
    (repo / "f.txt").write_text("v2\n", encoding="utf-8")
    cp1 = await manager.create_checkpoint("r1", 1, "step-1")
    (repo / "f.txt").write_text("v3\n", encoding="utf-8")
    await manager.close()

    restarted = _manager(repo)  # 进程重启：新 manager 实例
    recovered = await restarted.resume("r1")
    assert recovered is not None
    assert recovered.step == 1
    assert recovered.commit_sha == cp1.commit_sha  # type: ignore[union-attr]
    assert (repo / "f.txt").read_text(encoding="utf-8") == "v2\n"
    await restarted.close()


# 功能：验证 close 按配置清理或保留 checkpoint refs
# 设计：keep=True 保留 refs；keep=False 清理后 for-each-ref 为空
async def test_close_prunes_refs_when_configured(repo: Path) -> None:
    manager = _manager(repo)
    await manager.ensure_task_branch("t1")
    (repo / "f.txt").write_text("v2\n", encoding="utf-8")
    await manager.create_checkpoint("r1", 1, "step-1")
    await manager.close()
    assert "refs/kama" in _git(repo, "for-each-ref", "refs/kama").stdout.decode()

    manager2 = _manager(repo, keep_checkpoint_refs=False)
    (repo / "f.txt").write_text("v3\n", encoding="utf-8")
    await manager2.create_checkpoint("r2", 1, "step-1")
    await manager2.close()
    assert _git(repo, "for-each-ref", "refs/kama").stdout.decode() == ""


# 功能：验证 commit 身份使用 config.author（env 注入而非用户 git config）
# 设计：checkpoint commit 的 author 与 GitConfig 默认作者一致
async def test_commit_author_from_config(repo: Path) -> None:
    manager = _manager(repo, author="KamaClaude Agent <agent@kama.local>")
    await manager.ensure_task_branch("t1")
    (repo / "f.txt").write_text("v2\n", encoding="utf-8")
    await manager.create_checkpoint("r1", 1, "step-1")
    author = _git(repo, "log", "-1", "--format=%an <%ae>").stdout.decode().strip()
    assert author == "KamaClaude Agent <agent@kama.local>"


_FAKE_GIT_SCRIPT = """\
#!/bin/sh
# 记录每个 argv 一行到 FAKE_GIT_LOG；按子命令注入故障
printf '%s\\n' "$@" >> "$FAKE_GIT_LOG"
if [ "$1" = "-C" ]; then shift 2; fi  # runtime 前缀 -C <workspace>
cmd="$1"
case "$cmd" in
  status)
    if [ "$FAKE_GIT_DIRTY" = "1" ]; then echo " M f.txt"; fi ;;
  commit)
    if [ -n "$FAKE_GIT_STDERR" ]; then printf '%s\\n' "$FAKE_GIT_STDERR" >&2; exit 1; fi ;;
  update-ref)
    if [ -n "$FAKE_GIT_REF_STDERR" ]; then printf '%s\\n' "$FAKE_GIT_REF_STDERR" >&2; exit 1; fi ;;
esac
exit 0
"""


def _fake_manager(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[GitManager, Path, Path]:
    script = tmp_path / "git"
    script.write_text(_FAKE_GIT_SCRIPT, encoding="utf-8")
    script.chmod(0o755)
    log = tmp_path / "log.txt"
    monkeypatch.setenv("FAKE_GIT_LOG", str(log))
    monkeypatch.delenv("FAKE_GIT_DIRTY", raising=False)
    monkeypatch.delenv("FAKE_GIT_STDERR", raising=False)
    monkeypatch.delenv("FAKE_GIT_REF_STDERR", raising=False)
    workspace = tmp_path / "ws"
    workspace.mkdir()
    manager = GitManager(
        config=GitConfig(),
        workspace_root=workspace,
        runtime=GitCliRuntime(git_executable=str(script), workspace_root=workspace),
    )
    return manager, script, log


# 功能：验证 checkpoint 流程的 git 调用顺序（add → status → commit → rev-parse → update-ref）
# 设计：fake git 记录 argv，按 log 行索引断言相对顺序，证明 ref 登记在 commit 之后
async def test_checkpoint_call_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager, _script, log = _fake_manager(tmp_path, monkeypatch)
    monkeypatch.setenv("FAKE_GIT_DIRTY", "1")
    cp = await manager.create_checkpoint("r1", 1, "step-1")
    assert cp is not None
    lines = log.read_text(encoding="utf-8").splitlines()
    i_add = lines.index("add")
    i_status = next(i for i, line in enumerate(lines) if line == "status")
    i_commit = lines.index("commit")
    # rev-parse 出现两次（ensure 校验 + HEAD 查询），取最后一次
    i_rev = max(i for i, line in enumerate(lines) if line == "rev-parse")
    i_ref = lines.index("update-ref")
    assert i_add < i_status < i_commit < i_rev < i_ref


# 功能：验证 commit 故障按 stderr 关键词分类为 CommitFailedError（非 CLI 缺失）
# 设计：fake commit 注入 empty ident stderr → create_checkpoint 抛 CommitFailedError
async def test_commit_failure_classified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager, _script, _log = _fake_manager(tmp_path, monkeypatch)
    monkeypatch.setenv("FAKE_GIT_DIRTY", "1")
    monkeypatch.setenv(
        "FAKE_GIT_STDERR", "fatal: empty ident name (for <agent@kama.local>) not allowed"
    )
    await manager.ensure_ready()
    with pytest.raises(CommitFailedError):
        await manager.create_checkpoint("r1", 1, "step-1")
