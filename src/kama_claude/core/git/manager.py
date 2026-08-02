from __future__ import annotations

import asyncio
import logging
import os
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from kama_claude.core.git.checkpoint import (
    KIND_INTERNAL,
    KIND_PRE_RUN,
    GitCheckpoint,
    checkpoint_ref,
    pre_run_ref,
)
from kama_claude.core.git.config import GitConfig
from kama_claude.core.git.errors import (
    CommitFailedError,
    GitError,
    MergeConflictError,
    RollbackFailedError,
)
from kama_claude.core.git.runtime import GitCliRuntime

_LOGGER = logging.getLogger(__name__)

_EMAIL_RE = re.compile(r"^(.*) <([^<>]+)>$")
_MAX_DIFF_LINES = 200


@dataclass(frozen=True)
class GitStatus:
    # 工作树状态解析结果：dirty 标志 + 分支/上下游 + 原始 porcelain 条目
    dirty: bool
    branch: str
    ahead: int
    behind: int
    entries: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class GitDiff:
    # diff stat 摘要（截断保护），供 review 与事件发布
    stat: str
    truncated: bool


@dataclass(frozen=True)
class GitRestoreResult:
    # 回滚结果：目标 checkpoint sha + 实际策略（reset / revert）
    checkpoint_sha: str
    strategy: str


@dataclass(frozen=True)
class GitCommitResult:
    # finalize 产物：唯一用户可见 commit 的元数据
    commit_sha: str
    short_sha: str
    summary: str
    diff_stat: str


# 解析 git status --porcelain=v1 -b 输出为结构化状态
def _parse_status(porcelain: str) -> GitStatus:
    entries: list[tuple[str, str]] = []
    branch = ""
    ahead = behind = 0
    for line in porcelain.splitlines():
        if line.startswith("##"):
            header = line[2:].strip()
            branch = header.split("...")[0]
            match = re.search(r"\[ahead (\d+)(?:, behind (\d+))?\]", header)
            if match:
                ahead = int(match.group(1))
                behind = int(match.group(2) or 0)
        elif line:
            entries.append((line[:2], line[3:]))
    return GitStatus(
        dirty=bool(entries),
        branch=branch,
        ahead=ahead,
        behind=behind,
        entries=tuple(entries),
    )


class GitManager:
    # git 工作流状态机：idle→ready→closed；failed 非终态，下一次调用重试
    IDLE = "idle"
    READY = "ready"
    FAILED = "failed"
    CLOSED = "closed"

    def __init__(
        self,
        *,
        config: GitConfig,
        workspace_root: Path,
        runtime: GitCliRuntime | None = None,
    ) -> None:
        self._config = config
        self._state = self.IDLE
        self._lock = asyncio.Lock()
        self._runtime = runtime or GitCliRuntime(workspace_root=workspace_root)

    @property
    def state(self) -> str:
        # 只读状态，供日志与测试断言状态机
        return self._state

    @property
    def config(self) -> GitConfig:
        return self._config

    # 幂等关闭：只执行一次；按配置清理 checkpoint refs，失败仅记日志
    async def close(self) -> None:
        if self._state == self.CLOSED:
            return
        self._state = self.CLOSED
        if self._config.keep_checkpoint_refs:
            return
        try:
            result = await self._runtime.run(
                ["for-each-ref", "--format=%(refname)", self._config.checkpoint_namespace]
            )
            for refname in result.output.decode("utf-8", errors="replace").splitlines():
                if refname:
                    await self._runtime.run(["update-ref", "-d", refname])
        except (Exception, asyncio.CancelledError):
            _LOGGER.exception("git checkpoint ref pruning failed")

    # 公共 preflight：校验 workspace 是 git 仓库；非仓库抛 RepositoryNotFoundError
    async def ensure_ready(self) -> None:
        async with self._lock:
            await self._ensure_ready_locked()

    # lock 内校验仓库；失败置 FAILED 并上抛（fail-open 由调用方按错误类型降级）
    async def _ensure_ready_locked(self) -> None:
        if self._state == self.CLOSED:
            raise RuntimeError("git manager closed")
        if self._state == self.READY:
            return
        try:
            await self._runtime.run_check(["rev-parse", "--is-inside-work-tree"])
        except asyncio.CancelledError:
            self._state = self.FAILED
            raise
        except GitError:
            self._state = self.FAILED
            raise
        self._state = self.READY

    # 解析当前工作树状态（porcelain v1 + 分支头）
    async def status(self) -> GitStatus:
        async with self._lock:
            await self._ensure_ready_locked()
            return await self._status_locked()

    # lock 内直接解析状态（供其他加锁操作复用，避免重入死锁）
    async def _status_locked(self) -> GitStatus:
        result = await self._runtime.run_check(["status", "--porcelain=v1", "-b"])
        return _parse_status(result.output.decode("utf-8", errors="replace"))

    # 生成 diff stat 摘要；ref 为空时对比工作树与 HEAD
    async def diff(self, ref: str | None = None) -> GitDiff:
        async with self._lock:
            await self._ensure_ready_locked()
            args = ["diff", "--stat"] + ([ref] if ref else ["HEAD"])
            result = await self._runtime.run_check(args)
            lines = result.output.decode("utf-8", errors="replace").splitlines()
            truncated = len(lines) > _MAX_DIFF_LINES
            stat = "\n".join(lines[:_MAX_DIFF_LINES])
            if truncated:
                stat += f"\n... ({len(lines)} lines total)"
            return GitDiff(stat=stat, truncated=truncated)

    # 从当前 HEAD 创建 task 分支（幂等）；已存在时直接复用
    async def ensure_task_branch(self, task_id: str) -> None:
        async with self._lock:
            await self._ensure_ready_locked()
            branch = f"{self._config.branch_prefix}/{task_id}"
            result = await self._runtime.run(
                ["rev-parse", "--verify", "--quiet", f"refs/heads/{branch}"]
            )
            if result.returncode == 0:
                return
            await self._runtime.run_check(["switch", "-c", branch])

    # 固化含用户未提交修改的工作树为 pre-run 快照（dirty 场景的 baseline 基底）
    async def snapshot_pre_run(self, run_id: str, label: str = "pre-run") -> GitCheckpoint | None:
        ref = pre_run_ref(self._config.checkpoint_namespace, run_id)
        return await self._capture(ref, run_id, 0, label, KIND_PRE_RUN, force=True)

    # 创建内部 checkpoint：staged 全部改动 → internal commit → ref 登记；
    # 无变更且非 force 时返回 None（空 checkpoint 跳过）；force 时 ref 指向 HEAD 不产生 commit
    async def create_checkpoint(
        self,
        run_id: str,
        step: int,
        label: str,
        *,
        force: bool = False,
    ) -> GitCheckpoint | None:
        ref = checkpoint_ref(self._config.checkpoint_namespace, run_id, step)
        return await self._capture(ref, run_id, step, label, KIND_INTERNAL, force=force)

    # 捕获一次 checkpoint：add → status → stat → commit → rev-parse → update-ref
    async def _capture(
        self,
        ref: str,
        run_id: str,
        step: int,
        label: str,
        kind: str,
        *,
        force: bool,
    ) -> GitCheckpoint | None:
        async with self._lock:
            await self._ensure_ready_locked()
            await self._runtime.run_check(["add", "-A"])
            dirty = (await self._status_locked()).dirty
            if not dirty and not force:
                return None
            stat_result = await self._runtime.run_check(["diff", "--cached", "--stat"])
            diff_stat = stat_result.output.decode("utf-8", errors="replace").strip()
            if dirty:
                await self._runtime.run_check(
                    ["commit", "--no-verify", "-m", f"kama: {label}"],
                    env=self._author_env(),
                )
            sha = await self._head_sha()
            await self._runtime.run_check(["update-ref", ref, sha])
            return GitCheckpoint(
                run_id=run_id,
                step=step,
                label=label,
                kind=kind,
                commit_sha=sha,
                ref=ref,
                short_sha=sha[:7],
                diff_stat=diff_stat,
                ts=datetime.now(UTC).isoformat(),
                dirty=dirty,
            )

    # 回滚到 checkpoint：reset 策略仅限 agent 分支（保护用户分支）；revert 策略任意分支
    async def restore(self, checkpoint: GitCheckpoint) -> GitRestoreResult:
        async with self._lock:
            await self._ensure_ready_locked()
            if self._config.rollback_strategy == "revert":
                await self._runtime.run_check(
                    ["revert", "--no-commit", checkpoint.commit_sha]
                )
                return GitRestoreResult(
                    checkpoint_sha=checkpoint.commit_sha, strategy="revert"
                )
            branch = (await self._status_locked()).branch
            prefix = self._config.branch_prefix + "/"
            if not branch.startswith(prefix):
                raise RollbackFailedError(
                    "restore refused on non-agent branch",
                    detail=f"current branch {branch!r}; checkout {prefix}<task> first",
                )
            await self._runtime.run_check(["reset", "--hard", checkpoint.commit_sha])
            return GitRestoreResult(
                checkpoint_sha=checkpoint.commit_sha, strategy="reset"
            )

    # finalize：squash 内部 checkpoint 为唯一用户可见 commit（git_commit 的实质）；
    # 前置校验（baseline..tip 全为 kama commit）失败抛 MergeConflictError
    async def finalize(self, run_id: str, summary: str) -> GitCommitResult:
        async with self._lock:
            await self._ensure_ready_locked()
            branch = (await self._status_locked()).branch
            prefix = self._config.branch_prefix + "/"
            if not branch.startswith(prefix):
                raise RollbackFailedError(
                    "finalize refused on non-agent branch",
                    detail=f"current branch {branch!r}; checkout {prefix}<task> first",
                )
            baseline_sha = await self._baseline_sha_locked(run_id)
            # 用户中途 commit 会出现在范围内且非 kama 前缀 → 拒绝 squash，防吞用户提交
            log_result = await self._runtime.run_check(
                ["log", "--format=%s", f"{baseline_sha}..HEAD"]
            )
            foreign = [
                s
                for s in log_result.output.decode("utf-8", errors="replace").splitlines()
                if not s.startswith("kama: ")
            ]
            if foreign:
                raise MergeConflictError(
                    "finalize refused: user commits interleaved in run range",
                    detail="\n".join(foreign[:10]),
                )
            await self._runtime.run_check(["reset", "--soft", baseline_sha])
            await self._runtime.run_check(
                ["commit", "-m", f"kama: {summary}"], env=self._author_env()
            )
            sha = await self._head_sha()
            stat_result = await self._runtime.run_check(["show", "--stat", "--format=", sha])
            return GitCommitResult(
                commit_sha=sha,
                short_sha=sha[:7],
                summary=summary,
                diff_stat=stat_result.output.decode("utf-8", errors="replace").strip(),
            )

    # 解析 finalize 的 squash 基底：step-0 baseline 优先，pre-run 快照兜底
    async def _baseline_sha_locked(self, run_id: str) -> str:
        refs = [
            checkpoint_ref(self._config.checkpoint_namespace, run_id, 0),
            pre_run_ref(self._config.checkpoint_namespace, run_id),
        ]
        for ref in refs:
            result = await self._runtime.run(
                ["rev-parse", "--verify", "--quiet", ref]
            )
            if result.returncode == 0:
                return result.output.decode("utf-8", errors="replace").strip()
        raise CommitFailedError("finalize refused: no run baseline checkpoint")

    # 取 run 的最后一步 checkpoint（crash recovery 的恢复点）；无则返回 None
    async def latest_checkpoint(self, run_id: str) -> GitCheckpoint | None:
        async with self._lock:
            await self._ensure_ready_locked()
            prefix = f"{self._config.checkpoint_namespace}/checkpoints/{run_id}"
            result = await self._runtime.run(
                ["for-each-ref", "--format=%(objectname) %(refname)", f"{prefix}/*"]
            )
            pairs: list[tuple[int, str, str]] = []
            for line in result.output.decode("utf-8", errors="replace").splitlines():
                parts = line.split()
                if len(parts) != 2:
                    continue
                try:
                    step = int(parts[1].rsplit("/", 1)[1])
                except ValueError:
                    continue
                pairs.append((step, parts[0], parts[1]))
            if not pairs:
                return None
            _step, sha, refname = max(pairs, key=lambda p: p[0])
            subject = await self._run_log_format("%s", sha)
            ts = await self._run_log_format("%cI", sha)
            return GitCheckpoint(
                run_id=run_id,
                step=_step,
                label=subject.removeprefix("kama: "),
                kind=KIND_INTERNAL,
                commit_sha=sha,
                ref=refname,
                short_sha=sha[:7],
                diff_stat="",
                ts=ts,
                dirty=False,
            )

    # 按 step 解析指定 checkpoint（git_rollback 的目标）；不存在返回 None
    async def get_checkpoint(self, run_id: str, step: int) -> GitCheckpoint | None:
        async with self._lock:
            await self._ensure_ready_locked()
            ref = checkpoint_ref(self._config.checkpoint_namespace, run_id, step)
            result = await self._runtime.run(["rev-parse", "--verify", "--quiet", ref])
            if result.returncode != 0:
                return None
            sha = result.output.decode("utf-8", errors="replace").strip()
            subject = await self._run_log_format("%s", sha)
            ts = await self._run_log_format("%cI", sha)
            return GitCheckpoint(
                run_id=run_id,
                step=step,
                label=subject.removeprefix("kama: "),
                kind=KIND_INTERNAL,
                commit_sha=sha,
                ref=ref,
                short_sha=sha[:7],
                diff_stat="",
                ts=ts,
                dirty=False,
            )

    # 崩溃恢复入口：恢复到最后 checkpoint 状态并返回该 checkpoint；无则返回 None
    async def resume(self, run_id: str) -> GitCheckpoint | None:
        checkpoint = await self.latest_checkpoint(run_id)
        if checkpoint is None:
            return None
        await self.restore(checkpoint)
        return checkpoint

    # 读取 commit 的单一元数据字段（%s 主题 / %cI 提交时间）
    async def _run_log_format(self, fmt: str, sha: str) -> str:
        result = await self._runtime.run_check(["log", "-1", f"--format={fmt}", sha])
        return result.output.decode("utf-8", errors="replace").strip()

    # 当前 HEAD sha；调用方保证仓库存在且至少一个 commit
    async def _head_sha(self) -> str:
        result = await self._runtime.run_check(["rev-parse", "HEAD"])
        return result.output.decode("utf-8", errors="replace").strip()

    # 从 config.author 解析 name/email 覆盖身份环境（保留其余环境变量）
    def _author_env(self) -> dict[str, str]:
        match = _EMAIL_RE.match(self._config.author)
        if match:
            name, email = match.group(1), match.group(2)
        else:
            name, email = self._config.author, "agent@kama.local"
        return {
            **os.environ,
            "GIT_AUTHOR_NAME": name,
            "GIT_AUTHOR_EMAIL": email,
            "GIT_COMMITTER_NAME": name,
            "GIT_COMMITTER_EMAIL": email,
        }
