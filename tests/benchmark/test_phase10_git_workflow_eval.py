from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

from kama_claude.core.git.config import GitConfig
from kama_claude.core.git.manager import GitManager

_IDENT_ENV = {
    **os.environ,
    "GIT_AUTHOR_NAME": "user",
    "GIT_AUTHOR_EMAIL": "u@u",
    "GIT_COMMITTER_NAME": "user",
    "GIT_COMMITTER_EMAIL": "u@u",
}

# P7 指标表：5 个 evaluation case 各自记录指标，preflight 统一校验（镜像 phase9d 冻结产物风格）
_METRICS: dict[str, float] = {}
_METRIC_THRESHOLDS: dict[str, tuple[float, str]] = {
    "rollback.tree_hash_consistency": (1.0, "=="),   # 回滚后 HEAD == checkpoint 哈希，100%
    "diff.file_coverage": (1.0, "=="),               # 3 文件修改 diff 全覆盖，3/3
    "recovery.test_pass_after_rollback": (1.0, "=="),  # 失败回滚后测试重新通过
    "isolation.file_leakage": (0.0, "<="),           # 并发 task 文件零泄漏
    "crash.tree_hash_consistency": (1.0, "=="),      # 崩溃恢复后工作树 == checkpoint，100%
    "crash.resume_seconds": (2.0, "<"),              # resume 耗时上限
}


def _record(key: str, value: float) -> None:
    _METRICS[key] = value


def _within(value: float, threshold: float, op: str) -> bool:
    if op == "==":
        return value == threshold
    if op == "<=":
        return value <= threshold
    return value < threshold


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(["git", *args], cwd=repo, capture_output=True, check=True)


def _manager(repo: Path) -> GitManager:
    return GitManager(config=GitConfig(), workspace_root=repo)


# 真实 git 仓库夹具：main 分支 + 一个空 init commit（各 case 自行写入初始内容）
@pytest.fixture
def repo(tmp_path: Path) -> Path:
    r = tmp_path / "repo"
    r.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main", str(r)], check=True)
    subprocess.run(
        ["git", "commit", "-qm", "init", "--allow-empty"],
        cwd=r, check=True, env=_IDENT_ENV,
    )
    return r


# ── Case 1: rollback 哈希一致性 ────────────────────────────────────────────────

# 功能：验证 checkpoint 后继续修改，rollback 恢复后 HEAD 与 checkpoint 提交哈希一致
# 设计：修改→checkpoint(step1)→再修改→restore(step1)，断言 HEAD==cp.sha 且文件内容回滚
async def test_case1_rollback_hash_consistency(repo: Path) -> None:
    (repo / "f.txt").write_text("v1\n", encoding="utf-8")
    _git(repo, "add", "f.txt")
    _git(repo, "commit", "-qm", "seed")
    manager = _manager(repo)
    await manager.ensure_task_branch("t1")
    await manager.create_checkpoint("r1", 0, "baseline", force=True)
    (repo / "f.txt").write_text("v2\n", encoding="utf-8")
    cp = await manager.create_checkpoint("r1", 1, "step-1")
    assert cp is not None
    (repo / "f.txt").write_text("v3\n", encoding="utf-8")

    await manager.restore(cp)

    head = _git(repo, "rev-parse", "HEAD").stdout.decode().strip()
    _record("rollback.tree_hash_consistency", 1.0 if head == cp.commit_sha else 0.0)
    assert head == cp.commit_sha
    assert (repo / "f.txt").read_text(encoding="utf-8") == "v2\n"


# ── Case 2: 多文件 diff 完整性 ──────────────────────────────────────────────────

# 功能：验证单次 checkpoint 捕获 3 个文件的修改，diff stat 覆盖 3/3
# 设计：3 文件各写不同内容 → checkpoint → 用 checkpoint diff_stat 与原始 git 双路核对
async def test_case2_multi_file_diff_coverage(repo: Path) -> None:
    (repo / "a.txt").write_text("a1\n", encoding="utf-8")
    (repo / "b.txt").write_text("b1\n", encoding="utf-8")
    (repo / "c.txt").write_text("c1\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "seed")
    manager = _manager(repo)
    await manager.ensure_task_branch("t2")
    await manager.create_checkpoint("r2", 0, "baseline", force=True)
    for name in ("a.txt", "b.txt", "c.txt"):
        (repo / name).write_text(f"{name}-v2\n", encoding="utf-8")
    cp = await manager.create_checkpoint("r2", 1, "multi-file")
    assert cp is not None

    names = {name for name in ("a.txt", "b.txt", "c.txt") if name in cp.diff_stat}
    stat = _git(repo, "show", "--stat", "--format=", cp.commit_sha).stdout.decode()
    for name in ("a.txt", "b.txt", "c.txt"):
        assert name in stat

    _record("diff.file_coverage", len(names) / 3)
    assert len(names) == 3  # diff stat 覆盖 3/3


# ── Case 3: test failure 自动恢复 ──────────────────────────────────────────────

# 功能：验证 run 失败触发 auto_rollback（restore baseline）后，测试恢复通过
# 设计：测试文件随初始 commit 入库（通过态）→ baseline → 引入失败（cp1）→
#       run failed 路径 restore(baseline) → 重新执行测试断言 rc==0
async def test_case3_test_failure_auto_recovery(repo: Path) -> None:
    (repo / "test_math.py").write_text(
        "def test_sum():\n    assert 1 + 1 == 2\n", encoding="utf-8"
    )
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "seed with passing test")
    manager = _manager(repo)
    await manager.ensure_task_branch("t3")
    baseline = await manager.create_checkpoint("r3", 0, "baseline", force=True)
    assert baseline is not None
    assert _run_pytest(repo) == 0  # 基线态测试通过

    (repo / "test_math.py").write_text(
        "def test_sum():\n    assert 1 + 1 == 3\n", encoding="utf-8"
    )
    await manager.create_checkpoint("r3", 1, "broken-state")
    assert _run_pytest(repo) != 0  # agent 引入失败

    await manager.restore(baseline)  # run failed → auto_rollback 路径

    head = _git(repo, "rev-parse", "HEAD").stdout.decode().strip()
    assert head == baseline.commit_sha
    _record("recovery.test_pass_after_rollback", 1.0 if _run_pytest(repo) == 0 else 0.0)
    assert _run_pytest(repo) == 0  # 恢复后测试通过


# ── Case 4: worktree 隔离（并发） ───────────────────────────────────────────────

# 功能：验证两个 task 各自 worktree 并发执行时文件零泄漏
# 设计：主仓库 + git worktree add 第二 worktree；两 manager 各管一个 worktree，
#       asyncio.gather 并发跑 task-a / task-b，断言分支树互不相交且各只含己方文件
async def test_case4_worktree_isolation(repo: Path) -> None:
    (repo / "seed.txt").write_text("seed\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "seed")
    worktree_b = repo.parent / "wt-b"
    subprocess.run(
        ["git", "worktree", "add", "-q", "-b", "agent/b", str(worktree_b)],
        cwd=repo, check=True,
    )

    manager_a = _manager(repo)
    manager_b = _manager(worktree_b)

    async def task_a() -> None:
        await manager_a.ensure_task_branch("a")
        (repo / "a.txt").write_text("from-a\n", encoding="utf-8")
        await manager_a.create_checkpoint("ra", 1, "task-a")

    async def task_b() -> None:
        await manager_b.ensure_task_branch("b")
        (worktree_b / "b.txt").write_text("from-b\n", encoding="utf-8")
        await manager_b.create_checkpoint("rb", 1, "task-b")

    await asyncio.gather(task_a(), task_b())

    tree_a = {
        line for line in _git(repo, "ls-tree", "-r", "--name-only", "agent/a")
        .stdout.decode().splitlines() if line
    }
    tree_b = {
        line for line in _git(repo, "ls-tree", "-r", "--name-only", "agent/b")
        .stdout.decode().splitlines() if line
    }
    assert tree_a == {"seed.txt", "a.txt"}
    assert tree_b == {"seed.txt", "b.txt"}
    # 泄漏 = 己方专属文件出现在对方树中（共享的 seed.txt 不算）
    leaked = int("b.txt" in tree_a) + int("a.txt" in tree_b)
    _record("isolation.file_leakage", float(leaked))
    assert leaked == 0  # 文件隔离 0 泄漏


# ── Case 5: crash recovery ─────────────────────────────────────────────────────

# 功能：验证进程崩溃后新 manager 实例从 refs 恢复到最后 checkpoint
# 设计：checkpoint 后丢弃原实例（模拟崩溃，状态只存在于 refs）→ 新实例 resume(run_id)
#       → 断言工作树==checkpoint 且恢复耗时低于 2s
async def test_case5_crash_recovery(repo: Path) -> None:
    (repo / "f.txt").write_text("v1\n", encoding="utf-8")
    _git(repo, "add", "f.txt")
    _git(repo, "commit", "-qm", "seed")
    manager = _manager(repo)
    await manager.ensure_task_branch("t5")
    await manager.create_checkpoint("r5", 0, "baseline", force=True)
    (repo / "f.txt").write_text("v2\n", encoding="utf-8")
    cp = await manager.create_checkpoint("r5", 1, "step-1")
    assert cp is not None
    del manager  # 崩溃：丢弃全部内存状态，仅 refs 存活

    recovered = _manager(repo)
    started = time.monotonic()
    restored = await recovered.resume("r5")
    elapsed = time.monotonic() - started

    head = _git(repo, "rev-parse", "HEAD").stdout.decode().strip()
    assert restored is not None
    assert head == cp.commit_sha
    assert (repo / "f.txt").read_text(encoding="utf-8") == "v2\n"
    _record("crash.tree_hash_consistency", 1.0 if head == cp.commit_sha else 0.0)
    _record("crash.resume_seconds", elapsed)
    assert elapsed < 2.0  # resume < 2s


# ── 指标 preflight：完整性与阈值 ────────────────────────────────────────────────

# 功能：验证 5 个 case 的指标全部记录且满足阈值表（镜像 phase9d 冻结产物校验风格）
# 设计：缺失任一 metric 即失败；逐项比较值与阈值
def test_metrics_report_complete_and_within_thresholds() -> None:
    missing = [key for key in _METRIC_THRESHOLDS if key not in _METRICS]
    assert missing == [], f"metrics missing for cases: {missing}"

    for key, (threshold, op) in _METRIC_THRESHOLDS.items():
        value = _METRICS[key]
        assert _within(value, threshold, op), (
            f"metric {key}={value} violates threshold {op} {threshold}"
        )


def _run_pytest(repo: Path) -> int:
    # 清掉 pyc/pytest 缓存：源文件变更后残留的旧字节码会掩盖真实失败（沙箱 mtime 粒度下更易触发）
    shutil.rmtree(repo / "__pycache__", ignore_errors=True)
    shutil.rmtree(repo / ".pytest_cache", ignore_errors=True)
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--no-header", "-p", "no:cacheprovider", "test_math.py"],
        cwd=repo, capture_output=True, env=env,
    )
    return proc.returncode
