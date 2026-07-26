from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

import kama_claude.eval.runner as eval_runner
from kama_claude.core.config import KamaConfig
from kama_claude.core.runner import RunOutcome
from kama_claude.eval.failure import FailureCategory
from kama_claude.eval.runner import prepare_attempt, run_attempt
from kama_claude.eval.task import load_task
from kama_claude.eval.worker import WorkerRequest, execute_request


# 创建只含公开输入与空 private criterion 的最小任务目录
def _task_dir(tmp_path: Path, *, timeout_s: float = 2.0) -> Path:
    task_dir = tmp_path / "task-a"
    workspace = task_dir / "public" / "workspace"
    private = task_dir / "private"
    workspace.mkdir(parents=True)
    private.mkdir()
    (workspace / "input.txt").write_text("input", encoding="utf-8")
    (task_dir / "public" / "task.json").write_text(
        json.dumps(
            {
                "id": "task-a",
                "goal": "Create result.txt.",
                "workspace_fixture": "public/workspace",
                "timeout_s": timeout_s,
            }
        ),
        encoding="utf-8",
    )
    (private / "grader.json").write_text(
        json.dumps(
            {
                "criteria": [
                    {"id": "result", "kind": "file_exists", "path": "result.txt"}
                ]
            }
        ),
        encoding="utf-8",
    )
    return task_dir


# 构造绝对且互相分离的最小 worker request
def _request(tmp_path: Path) -> WorkerRequest:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return WorkerRequest(
        task_id="task-a",
        run_id="run-a",
        goal="Create result.txt.",
        workspace=str(workspace.resolve()),
        runs_dir=str((tmp_path / "runs").resolve()),
        trace_path=str((tmp_path / "trace.jsonl").resolve()),
    )


# 功能：验证 worker protocol 只携带公开任务与运行路径并拒绝所有 eval/runtime override
# 设计：向合法序列化结果逐一注入 private 与行为字段，锁定跨进程数据流而非 CLI 解析
def test_worker_request_contains_no_private_or_runtime_configuration(tmp_path: Path) -> None:
    request = _request(tmp_path)

    assert set(request.model_dump(mode="json")) == {
        "task_id",
        "run_id",
        "goal",
        "workspace",
        "runs_dir",
        "trace_path",
    }
    for forbidden in (
        "grader",
        "criteria",
        "provider",
        "model",
        "tools",
        "tool_whitelist",
        "permission",
        "system_prompt",
        "max_steps",
    ):
        with pytest.raises(ValidationError):
            WorkerRequest.model_validate(
                {**request.model_dump(mode="json"), forbidden: "forbidden"}
            )


# 功能：验证两个 attempt 拥有独立 HOME、workspace、tmp、runs、trace 和 artifact 路径
# 设计：从同一 LoadedTask 连续 prepare 两次，断言所有可写边界不相交且只复制公开 fixture
def test_prepare_attempt_creates_unique_public_only_isolation(tmp_path: Path) -> None:
    loaded = load_task(_task_dir(tmp_path))

    first = prepare_attempt(loaded, tmp_path / "output")
    second = prepare_attempt(loaded, tmp_path / "output")

    first_paths = {
        first.workspace,
        first.home,
        first.tmp,
        first.runs_dir,
        first.trace_path,
        first.attempt_dir,
    }
    second_paths = {
        second.workspace,
        second.home,
        second.tmp,
        second.runs_dir,
        second.trace_path,
        second.attempt_dir,
    }
    assert first_paths.isdisjoint(second_paths)
    assert (first.workspace / "input.txt").read_text(encoding="utf-8") == "input"
    assert not (first.workspace / "grader.json").exists()
    assert not (first.workspace / "hidden_tests").exists()


class _FakeRunner:
    seen_init: dict[str, object] = {}
    seen_call: dict[str, object] = {}

    # 记录 AgentRunner 兼容构造参数，验证 worker 未传 provider 或行为 override
    def __init__(self, config: KamaConfig, **kwargs: object) -> None:
        type(self).seen_init = {"config": config, **kwargs}

    # 返回稳定成功结果并记录 one-shot 调用参数
    async def run_and_capture(self, goal: str, *, run_id: str) -> RunOutcome:
        type(self).seen_call = {"goal": goal, "run_id": run_id}
        return RunOutcome(status="success", result="done", reason=None)


# 功能：验证 production worker wiring 只调用现有 KamaConfig 与 AgentRunner one-shot 接口
# 设计：用构造 seam 替代真实网络 provider，检查初始化和调用形状而不新增序列化 provider 配置
@pytest.mark.asyncio
async def test_worker_uses_existing_runner_without_behavior_overrides(tmp_path: Path) -> None:
    result = await execute_request(
        _request(tmp_path),
        runner_factory=_FakeRunner,
        config_loader=KamaConfig,
    )

    assert result.runtime_status == "success"
    assert _FakeRunner.seen_call == {"goal": "Create result.txt.", "run_id": "run-a"}
    assert "provider" not in _FakeRunner.seen_init
    assert "permission_manager" not in _FakeRunner.seen_init


# 功能：验证 worker 在 AgentRunner 前记录脱敏 provider/runtime identity 且不改变 runner 调用
# 设计：注入真实 KamaConfig 与 fake runner，解析落盘 trace 而不调用模型或断言 mock 内部行为
@pytest.mark.asyncio
async def test_worker_records_sanitized_runtime_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = KamaConfig()
    config.agent.max_steps = 20
    config.llm.default_model = "deepseek-v4-pro"
    config.llm.router = "static"
    config.trace.enabled = True
    config.trace.include_llm_payload = True
    config.compaction.auto_threshold = 0.0
    config.compaction.tool_result_limit = 8000
    config.compaction.tool_result_keep = 4000
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://api.deepseek.com/anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "must-not-appear-in-trace")
    request = _request(tmp_path)

    result = await execute_request(
        request,
        runner_factory=_FakeRunner,
        config_loader=lambda: config,
    )

    records = [
        json.loads(line)
        for line in Path(request.trace_path).read_text(encoding="utf-8").splitlines()
    ]
    identities = [record for record in records if record["kind"] == "runtime_identity"]
    assert result.runtime_status == "success"
    assert len(identities) == 1
    data = identities[0]["data"]
    assert data["provider"] == {
        "service_provider": "deepseek",
        "wire_protocol": "anthropic_messages",
        "endpoint_id": "deepseek-anthropic-compatible",
        "endpoint": "https://api.deepseek.com/anthropic",
        "model_id": "deepseek-v4-pro",
        "sdk_distribution": "anthropic",
        "sdk_version": "0.111.0",
    }
    assert data["runtime"] == {
        "max_steps": 20,
        "router": "static",
        "compaction_threshold": 0.0,
        "tool_result_limit": 8000,
        "tool_result_keep": 4000,
        "mcp_enabled": False,
        "trace_enabled": True,
        "include_llm_payload": True,
    }
    assert "must-not-appear-in-trace" not in json.dumps(records)


# 写入一个能按 worker CLI 协议返回成功结果的测试脚本
def _write_success_worker(path: Path) -> None:
    path.write_text(
        """
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--request", required=True)
parser.add_argument("--result", required=True)
args = parser.parse_args()
request = json.loads(Path(args.request).read_text())
Path(args.result).write_text(json.dumps({
    "run_id": request["run_id"],
    "runtime_status": "success",
    "result": "done",
    "reason": None,
    "infra_error": None
}))
""".strip(),
        encoding="utf-8",
    )


# 功能：验证 parent runner 能启动测试 worker 并读取严格成功结果
# 设计：替换内部 argv 构造函数而不改变 public run_attempt 或 worker request schema
@pytest.mark.asyncio
async def test_run_attempt_accepts_test_only_worker_launcher(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = tmp_path / "success_worker.py"
    _write_success_worker(script)
    monkeypatch.setattr(eval_runner, "_worker_argv", lambda: [sys.executable, str(script)])

    execution = await run_attempt(load_task(_task_dir(tmp_path)), tmp_path / "output")

    assert execution.failure_category is FailureCategory.NONE
    assert execution.worker_result is not None
    assert execution.worker_result.runtime_status == "success"


# 写入一个派生延迟子进程并永久等待的测试 worker
def _write_hanging_worker(path: Path, marker: Path, started: Path) -> None:
    child_code = (
        "import signal,time; from pathlib import Path; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        f"time.sleep(0.5); Path({str(marker)!r}).write_text('leaked')"
    )
    path.write_text(
        f"""
import subprocess
import sys
import time
from pathlib import Path

Path({str(started)!r}).write_text("started")
subprocess.Popen([sys.executable, "-c", {child_code!r}])
time.sleep(30)
""".strip(),
        encoding="utf-8",
    )


# 功能：验证 attempt timeout 会终止 worker 及其派生进程组并返回中央 timeout 类别
# 设计：子进程延迟写 marker；runner 返回后等待越过延迟窗口，marker 不存在即证明 child 被清理
@pytest.mark.asyncio
async def test_run_attempt_timeout_reaps_worker_process_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    marker = tmp_path / "leaked.txt"
    started = tmp_path / "started.txt"
    script = tmp_path / "hanging_worker.py"
    _write_hanging_worker(script, marker, started)
    monkeypatch.setattr(eval_runner, "_worker_argv", lambda: [sys.executable, str(script)])

    execution = await run_attempt(
        load_task(_task_dir(tmp_path, timeout_s=0.1)),
        tmp_path / "output",
    )
    await asyncio.sleep(0.6)

    assert started.exists()
    assert execution.failure_category is FailureCategory.TIMEOUT
    assert not marker.exists()


# 写入一个正常返回 result 但遗留后台子进程的测试 worker
def _write_exiting_worker_with_child(path: Path, marker: Path) -> None:
    child_code = (
        "import signal,time; from pathlib import Path; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        f"time.sleep(0.5); Path({str(marker)!r}).write_text('leaked')"
    )
    path.write_text(
        f"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--request", required=True)
parser.add_argument("--result", required=True)
args = parser.parse_args()
request = json.loads(Path(args.request).read_text())
subprocess.Popen([sys.executable, "-c", {child_code!r}])
Path(args.result).write_text(json.dumps({{
    "run_id": request["run_id"],
    "runtime_status": "success",
    "result": "done",
    "reason": None,
    "infra_error": None,
}}))
""".strip(),
        encoding="utf-8",
    )


# 功能：验证 worker leader 正常退出后 runner 仍清理同组后台子进程
# 设计：worker 先写合法 result 再退出，忽略 SIGTERM 的 child 延迟写 marker，防止正常路径漏掉 group cleanup
@pytest.mark.asyncio
async def test_run_attempt_normal_exit_reaps_worker_process_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    marker = tmp_path / "normal-exit-leaked.txt"
    script = tmp_path / "exiting_worker.py"
    _write_exiting_worker_with_child(script, marker)
    monkeypatch.setattr(eval_runner, "_worker_argv", lambda: [sys.executable, str(script)])

    execution = await run_attempt(load_task(_task_dir(tmp_path)), tmp_path / "output")
    await asyncio.sleep(0.6)

    assert execution.failure_category is FailureCategory.NONE
    assert not marker.exists()


# 功能：验证 caller cancellation 保持 CancelledError identity 并清理正在运行的 worker
# 设计：等待 worker 写出 started 信号后取消 task，再越过 child 延迟窗口检查无泄漏 marker
@pytest.mark.asyncio
async def test_run_attempt_preserves_caller_cancellation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    marker = tmp_path / "cancel-leaked.txt"
    started = tmp_path / "cancel-started.txt"
    script = tmp_path / "cancel_worker.py"
    _write_hanging_worker(script, marker, started)
    monkeypatch.setattr(eval_runner, "_worker_argv", lambda: [sys.executable, str(script)])
    running = asyncio.create_task(
        run_attempt(load_task(_task_dir(tmp_path, timeout_s=5.0)), tmp_path / "output")
    )
    async with asyncio.timeout(2):
        while not started.exists():
            await asyncio.sleep(0.01)

    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running
    await asyncio.sleep(0.6)

    assert not marker.exists()


# 功能：验证 cancellation 即使在 normal-exit cleanup 窗口到达也不会被吞掉
# 设计：用事件控制内部 terminate seam，精确在 cleanup 已开始后取消 parent task，再释放 cleanup
@pytest.mark.asyncio
async def test_run_attempt_preserves_cancellation_during_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = tmp_path / "success_worker.py"
    _write_success_worker(script)
    monkeypatch.setattr(eval_runner, "_worker_argv", lambda: [sys.executable, str(script)])
    cleanup_started = asyncio.Event()
    cleanup_release = asyncio.Event()
    real_terminate = eval_runner._terminate_process_group

    # 延迟内部 cleanup，使测试能把 cancellation 精确注入 leader 已退出后的窗口
    async def delayed_terminate(process: asyncio.subprocess.Process) -> None:
        cleanup_started.set()
        await cleanup_release.wait()
        await real_terminate(process)

    monkeypatch.setattr(eval_runner, "_terminate_process_group", delayed_terminate)
    running = asyncio.create_task(
        run_attempt(load_task(_task_dir(tmp_path)), tmp_path / "output")
    )
    await asyncio.wait_for(cleanup_started.wait(), timeout=2.0)

    running.cancel()
    cleanup_release.set()

    with pytest.raises(asyncio.CancelledError):
        await running
