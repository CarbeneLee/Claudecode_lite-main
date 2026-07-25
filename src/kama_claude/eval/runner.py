from __future__ import annotations

import asyncio
import json
import os
import shutil
import signal
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from kama_claude.eval.failure import FailureCategory
from kama_claude.eval.models import WorkerRequest, WorkerResult
from kama_claude.eval.task import LoadedTask


@dataclass(frozen=True)
class PreparedAttempt:
    attempt_id: str
    attempt_dir: Path
    workspace: Path
    home: Path
    tmp: Path
    runs_dir: Path
    trace_path: Path
    request_path: Path
    result_path: Path
    request: WorkerRequest


@dataclass(frozen=True)
class AttemptExecution:
    prepared: PreparedAttempt
    worker_result: WorkerResult | None
    failure_category: FailureCategory
    wall_latency_ms: float


# 返回 production worker 的固定模块入口，测试可替换该内部 seam
def _worker_argv() -> list[str]:
    return [sys.executable, "-m", "kama_claude.eval.worker"]


# 将公开任务 fixture 复制到唯一 attempt，并生成不含 private 数据的 worker request
def prepare_attempt(task: LoadedTask, output_root: Path | str) -> PreparedAttempt:
    attempt_id = f"attempt-{uuid.uuid4().hex}"
    attempt_dir = (
        Path(output_root).resolve(strict=False)
        / "attempts"
        / task.public.id
        / attempt_id
    )
    work_dir = attempt_dir / "_work"
    workspace = work_dir / "workspace"
    home = work_dir / "home"
    tmp = work_dir / "tmp"
    runs_dir = work_dir / "runs"
    runtime_dir = attempt_dir / "runtime"
    public_dir = attempt_dir / "public"
    private_dir = attempt_dir / "private"
    for directory in (home, tmp, runs_dir, runtime_dir, public_dir, private_dir):
        directory.mkdir(parents=True, exist_ok=False if directory == home else True)
    shutil.copytree(task.workspace_fixture, workspace, symlinks=True)
    run_id = f"eval-{uuid.uuid4().hex}"
    trace_path = runtime_dir / "trace.jsonl"
    request = WorkerRequest(
        task_id=task.public.id,
        run_id=run_id,
        goal=task.public.goal,
        workspace=str(workspace.resolve()),
        runs_dir=str(runs_dir.resolve()),
        trace_path=str(trace_path.resolve()),
    )
    request_path = work_dir / "request.json"
    result_path = work_dir / "worker-result.json"
    request_path.write_text(request.model_dump_json(), encoding="utf-8")
    (public_dir / "task-input.json").write_text(
        json.dumps(task.public.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return PreparedAttempt(
        attempt_id=attempt_id,
        attempt_dir=attempt_dir,
        workspace=workspace,
        home=home,
        tmp=tmp,
        runs_dir=runs_dir,
        trace_path=trace_path,
        request_path=request_path,
        result_path=result_path,
        request=request,
    )


# 向 worker 进程组发送信号，即使 leader 已退出也清理仍存活的 descendants
def _signal_process_group(process: asyncio.subprocess.Process, sig: signal.Signals) -> None:
    try:
        os.killpg(process.pid, sig)
    except (ProcessLookupError, PermissionError):
        return


# 先温和终止再强制杀死 worker 进程组，并始终 reap 主进程
async def _terminate_process_group(process: asyncio.subprocess.Process) -> None:
    _signal_process_group(process, signal.SIGTERM)
    await asyncio.sleep(0.05)
    _signal_process_group(process, signal.SIGKILL)
    await process.wait()


# 完成 cleanup 后重新抛出期间到达的 cancellation，避免控制流被清理屏障吞掉
async def _cleanup_process_group(process: asyncio.subprocess.Process) -> None:
    cleanup = asyncio.create_task(_terminate_process_group(process))
    cancelled = False
    while not cleanup.done():
        try:
            await asyncio.shield(cleanup)
        except asyncio.CancelledError:
            cancelled = True
            continue
    await cleanup
    if cancelled:
        raise asyncio.CancelledError


# 读取并严格校验 worker result，任何缺失或身份冲突都视为 infrastructure failure
def _read_worker_result(prepared: PreparedAttempt) -> WorkerResult | None:
    try:
        result = WorkerResult.model_validate_json(
            prepared.result_path.read_text(encoding="utf-8")
        )
    except (OSError, ValidationError):
        return None
    return result if result.run_id == prepared.request.run_id else None


# 运行一个隔离 attempt，由 parent 负责 timeout、取消传播与子进程组清理
async def run_attempt(task: LoadedTask, output_root: Path | str) -> AttemptExecution:
    prepared = prepare_attempt(task, output_root)
    environment = os.environ.copy()
    environment["HOME"] = str(prepared.home)
    environment["TMPDIR"] = str(prepared.tmp)
    argv = [
        *_worker_argv(),
        "--request",
        str(prepared.request_path),
        "--result",
        str(prepared.result_path),
    ]
    started = time.monotonic()
    process = await asyncio.create_subprocess_exec(
        *argv,
        cwd=prepared.workspace,
        env=environment,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
        start_new_session=True,
    )
    try:
        try:
            async with asyncio.timeout(task.public.timeout_s):
                return_code = await process.wait()
        except TimeoutError:
            await _cleanup_process_group(process)
            return AttemptExecution(
                prepared=prepared,
                worker_result=None,
                failure_category=FailureCategory.TIMEOUT,
                wall_latency_ms=(time.monotonic() - started) * 1000,
            )
    except asyncio.CancelledError:
        await _cleanup_process_group(process)
        raise
    await _cleanup_process_group(process)
    result = _read_worker_result(prepared) if return_code == 0 else None
    category = (
        FailureCategory.NONE
        if result is not None and result.runtime_status == "success"
        else (
            FailureCategory.RUNTIME_FAILED
            if result is not None and result.runtime_status == "failed"
            else FailureCategory.INFRA_ERROR
        )
    )
    return AttemptExecution(
        prepared=prepared,
        worker_result=result,
        failure_category=category,
        wall_latency_ms=(time.monotonic() - started) * 1000,
    )
