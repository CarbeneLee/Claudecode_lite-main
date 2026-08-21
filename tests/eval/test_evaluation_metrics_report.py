from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

import kama_claude.eval.runner as eval_runner
from kama_claude.eval.cli import _parse_args
from kama_claude.eval.evaluator import evaluate_task
from kama_claude.eval.failure import FailureCategory
from kama_claude.eval.metrics import compute_basic_metrics
from kama_claude.eval.models import WorkerResult
from kama_claude.eval.runner import AttemptExecution, prepare_attempt
from kama_claude.eval.task import load_task


# 创建 private expected text 不出现在 public goal 的最小任务
def _task_dir(tmp_path: Path) -> Path:
    task_dir = tmp_path / "task-a"
    workspace = task_dir / "public" / "workspace"
    private = task_dir / "private"
    workspace.mkdir(parents=True)
    private.mkdir()
    (workspace / "input.txt").write_text("input\n", encoding="utf-8")
    (task_dir / "public" / "task.json").write_text(
        json.dumps(
            {
                "id": "task-a",
                "goal": "Create the requested result file.",
                "workspace_fixture": "public/workspace",
                "timeout_s": 30.0,
            }
        ),
        encoding="utf-8",
    )
    (private / "grader.json").write_text(
        json.dumps(
            {
                "criteria": [
                    {
                        "id": "private-content",
                        "kind": "file_contains",
                        "path": "result.txt",
                        "text": "SECRET_EXPECTED_VALUE",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return task_dir


# 构建包含 usage、step 与 retry 的事件列表
def _metric_events() -> list[dict[str, object]]:
    return [
        {"type": "step.started", "run_id": "run-a", "step": 1},
        {"type": "step.started", "run_id": "run-a", "step": 2},
        {
            "type": "tool.call_started",
            "run_id": "run-a",
            "tool_use_id": "tool-a",
        },
        {
            "type": "tool.call_failed",
            "run_id": "run-a",
            "tool_use_id": "tool-a",
            "attempt": 1,
        },
        {
            "type": "tool.call_finished",
            "run_id": "run-a",
            "tool_use_id": "tool-a",
        },
        {
            "type": "tool.call_started",
            "run_id": "run-a",
            "tool_use_id": "tool-b",
        },
        {
            "type": "tool.call_failed",
            "run_id": "run-a",
            "tool_use_id": "tool-b",
            "attempt": 1,
        },
        {
            "type": "tool.call_failed",
            "run_id": "run-a",
            "tool_use_id": "tool-b",
            "attempt": 2,
        },
        {
            "type": "llm.usage",
            "run_id": "run-a",
            "input_tokens": 10,
            "output_tokens": 4,
            "cache_read_input_tokens": 3,
            "cache_creation_input_tokens": 2,
        },
    ]


# 功能：验证 basic metrics 只机械统计 success、step、tool、retry、latency 和 token
# 设计：组合成功重试与最终失败重试，锁定 retry 定义而不评价调用是否必要
def test_compute_basic_metrics_counts_only_frozen_signals() -> None:
    metrics = compute_basic_metrics(
        _metric_events(),
        task_success=False,
        runtime_success=True,
        wall_latency_ms=25.5,
        failure_category=FailureCategory.TASK_FAILED,
    )

    assert metrics.task_success is False
    assert metrics.runtime_success is True
    assert metrics.step_count == 2
    assert metrics.tool_count == 2
    assert metrics.retry_count == 2
    assert metrics.wall_latency_ms == 25.5
    assert metrics.token_usage.model_dump() == {
        "input_tokens": 10,
        "output_tokens": 4,
        "cache_tokens": 5,
    }
    assert metrics.failure_category is FailureCategory.TASK_FAILED
    assert "cost" not in metrics.model_dump(mode="json")
    assert "planning_efficiency" not in metrics.model_dump(mode="json")


# 写入一个合法 canonical journal 并返回完成 execution
def _execution(tmp_path: Path, *, task_passes: bool) -> AttemptExecution:
    loaded = load_task(_task_dir(tmp_path))
    prepared = prepare_attempt(loaded, tmp_path / "output")
    if task_passes:
        (prepared.workspace / "result.txt").write_text(
            "SECRET_EXPECTED_VALUE\n", encoding="utf-8"
        )
    run_dir = prepared.runs_dir / prepared.request.run_id
    run_dir.mkdir()
    events = [
        {
            "type": "run.started",
            "run_id": prepared.request.run_id,
            "goal": loaded.public.goal,
            "ts": "t1",
        },
        {
            "type": "step.started",
            "run_id": prepared.request.run_id,
            "step": 1,
            "ts": "t2",
        },
        {
            "type": "llm.usage",
            "run_id": prepared.request.run_id,
            "input_tokens": 5,
            "output_tokens": 2,
            "cache_read_input_tokens": 1,
            "cache_creation_input_tokens": 0,
            "context_pct": 0.0,
            "ts": "t3",
        },
        {
            "type": "step.finished",
            "run_id": prepared.request.run_id,
            "step": 1,
            "ts": "t4",
        },
        {
            "type": "run.finished",
            "run_id": prepared.request.run_id,
            "status": "success",
            "reason": None,
            "steps": 1,
            "ts": "t5",
        },
    ]
    rows = [
        {
            "schema_version": 2,
            "event_id": f"evt-{index}",
            "stream_id": f"run:{prepared.request.run_id}",
            "seq": index,
            "event": event,
        }
        for index, event in enumerate(events, 1)
    ]
    (run_dir / "events.v2.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    prepared.trace_path.write_text("{}\n", encoding="utf-8")
    return AttemptExecution(
        prepared=prepared,
        worker_result=WorkerResult(
            run_id=prepared.request.run_id,
            runtime_status="success",
            result="done",
        ),
        failure_category=FailureCategory.NONE,
        wall_latency_ms=15.0,
    )


# 构造保留 journal prefix 与 trace 的 timeout attempt
def _timeout_execution(tmp_path: Path) -> AttemptExecution:
    loaded = load_task(_task_dir(tmp_path))
    prepared = prepare_attempt(loaded, tmp_path / "output")
    run_dir = prepared.runs_dir / prepared.request.run_id
    run_dir.mkdir()
    row = {
        "schema_version": 2,
        "event_id": "evt-timeout-1",
        "stream_id": f"run:{prepared.request.run_id}",
        "seq": 1,
        "event": {
            "type": "run.started",
            "run_id": prepared.request.run_id,
            "goal": loaded.public.goal,
            "ts": "t1",
        },
    }
    (run_dir / "events.v2.jsonl").write_text(
        json.dumps(row) + "\n",
        encoding="utf-8",
    )
    prepared.trace_path.write_text(
        json.dumps({"kind": "runtime_identity", "run_id": prepared.request.run_id})
        + "\n",
        encoding="utf-8",
    )
    return AttemptExecution(
        prepared=prepared,
        worker_result=None,
        failure_category=FailureCategory.TIMEOUT,
        wall_latency_ms=30_000.0,
    )


# 功能：验证 evaluator 生成一致 JSON/Markdown 且不泄露 private expected value
# 设计：fake attempt 产出满足 hidden criterion 的真实 artifact，随后搜索全部 public report 文本
@pytest.mark.asyncio
async def test_evaluator_writes_public_success_report_without_private_values(
    tmp_path: Path,
) -> None:
    execution = _execution(tmp_path, task_passes=True)

    async def fake_attempt(_task: object, _output: object) -> AttemptExecution:
        return execution

    report = await evaluate_task(
        _task_dir(tmp_path / "second"),
        tmp_path / "output",
        attempt_runner=fake_attempt,
    )
    report_json = (tmp_path / "output" / "report.json").read_text(encoding="utf-8")
    report_md = (tmp_path / "output" / "report.md").read_text(encoding="utf-8")

    assert report.task_success is True
    assert report.failure_category is FailureCategory.NONE
    assert json.loads(report_json)["metrics"]["step_count"] == 1
    assert "private-content" in report_json
    assert "private-content" in report_md
    assert "SECRET_EXPECTED_VALUE" not in report_json
    assert "SECRET_EXPECTED_VALUE" not in report_md
    assert (tmp_path / "output" / "manifest.json").is_file()


# 功能：验证 runtime success 不能覆盖 required rule failure
# 设计：使用合法成功 lifecycle 但不创建目标文件，断言中央分类选择 task_failed
@pytest.mark.asyncio
async def test_evaluator_separates_runtime_success_from_task_success(tmp_path: Path) -> None:
    execution = _execution(tmp_path, task_passes=False)

    async def fake_attempt(_task: object, _output: object) -> AttemptExecution:
        return execution

    report = await evaluate_task(
        _task_dir(tmp_path / "second"),
        tmp_path / "output",
        attempt_runner=fake_attempt,
    )

    assert report.runtime_success is True
    assert report.task_success is False
    assert report.failure_category is FailureCategory.TASK_FAILED


# 功能：验证 timeout attempt 仅保留 identity 所需 runtime prefix，不进入 grader 或正式轨迹指标
# 设计：提供真实 attempt 目录中的部分 journal/trace，断言 canonical copy 与 timeout 语义同时成立
@pytest.mark.asyncio
async def test_evaluator_preserves_timeout_partial_identity_evidence(
    tmp_path: Path,
) -> None:
    execution = _timeout_execution(tmp_path)

    async def fake_attempt(_task: object, _output: object) -> AttemptExecution:
        return execution

    report = await evaluate_task(
        _task_dir(tmp_path / "second"),
        tmp_path / "output",
        attempt_runner=fake_attempt,
    )
    attempt = execution.prepared.attempt_dir

    assert report.failure_category is FailureCategory.TIMEOUT
    assert report.runtime_success is False
    assert report.task_success is False
    assert report.trace_sanity_passed is False
    assert report.criteria == []
    assert report.metrics.step_count == 0
    assert report.metrics.tool_count == 0
    assert report.metrics.retry_count == 0
    assert report.metrics.token_usage.input_tokens == 0
    assert (attempt / "runtime" / "events.v2.jsonl").is_file()
    assert (attempt / "runtime" / "trace.jsonl").is_file()
    assert not (attempt / "runtime" / "initial-workspace.json").exists()
    assert not (attempt / "runtime" / "final-workspace.json").exists()
    assert not (attempt / "runtime" / "workspace.diff").exists()
    assert not (attempt / "private" / "grades.json").exists()
    assert not (attempt / "private" / "command-results.json").exists()


# 功能：验证无法启动 private grader command 时仍生成集中分类的公开报告
# 设计：保留合法 runtime/journal，仅将 private argv 设为不存在的程序，锁定 grader_error 优先级与报告路径
@pytest.mark.asyncio
async def test_evaluator_classifies_missing_grader_command(tmp_path: Path) -> None:
    execution = _execution(tmp_path, task_passes=True)
    evaluation_task = _task_dir(tmp_path / "second")
    (evaluation_task / "private" / "grader.json").write_text(
        json.dumps(
            {
                "criteria": [
                    {
                        "id": "missing-command",
                        "kind": "command_exit",
                        "argv": ["kama-eval-executable-that-does-not-exist"],
                        "expected_exit_code": 0,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    async def fake_attempt(_task: object, _output: object) -> AttemptExecution:
        return execution

    report = await evaluate_task(
        evaluation_task,
        tmp_path / "output",
        attempt_runner=fake_attempt,
    )

    assert report.task_success is False
    assert report.failure_category is FailureCategory.GRADER_ERROR
    assert (tmp_path / "output" / "report.json").is_file()


# 功能：验证 CLI 只接受单任务单 attempt 的 run 命令
# 设计：检查 run namespace 没有 provider/repeats，再让 compare、baseline 和 provider 参数解析失败
def test_cli_exposes_only_evaluation_run_command() -> None:
    args = _parse_args(["run", "--task", "task-a", "--output", "artifacts"])

    assert args.command == "run"
    assert vars(args) == {"command": "run", "task": "task-a", "output": "artifacts"}
    for forbidden in (
        ["compare"],
        ["run", "--task", "task-a", "--output", "out", "--provider", "scripted"],
        ["run", "--task", "task-a", "--output", "out", "--repeats", "3"],
        ["run", "--baseline", "a", "--candidate", "b"],
    ):
        with pytest.raises(SystemExit):
            _parse_args(forbidden)


# 写入遵循 internal worker protocol 的 deterministic vertical-slice worker
def _write_vertical_slice_worker(path: Path) -> None:
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
assert "SECRET_EXPECTED_VALUE" not in json.dumps(request)
workspace = Path(request["workspace"])
(workspace / "result.txt").write_text("SECRET_EXPECTED_VALUE\\n")
run_dir = Path(request["runs_dir"]) / request["run_id"]
run_dir.mkdir(parents=True)
events = [
    {"type": "run.started", "run_id": request["run_id"], "goal": request["goal"], "ts": "t1"},
    {"type": "step.started", "run_id": request["run_id"], "step": 1, "ts": "t2"},
    {"type": "step.finished", "run_id": request["run_id"], "step": 1, "ts": "t3"},
    {
        "type": "run.finished",
        "run_id": request["run_id"],
        "status": "success",
        "reason": None,
        "steps": 1,
        "ts": "t4",
    },
]
rows = [
    {
        "schema_version": 2,
        "event_id": f"evt-{index}",
        "stream_id": f"run:{request['run_id']}",
        "seq": index,
        "event": event,
    }
    for index, event in enumerate(events, 1)
]
(run_dir / "events.v2.jsonl").write_text(
    "".join(json.dumps(row) + "\\n" for row in rows)
)
Path(request["trace_path"]).write_text("{}\\n")
Path(args.result).write_text(json.dumps({
    "run_id": request["run_id"],
    "runtime_status": "success",
    "result": "done",
    "reason": None,
    "infra_error": None,
}))
""".strip(),
        encoding="utf-8",
    )


# 功能：验证真实 parent subprocess 边界可贯通 collector、grader、metrics 和 report
# 设计：test-only worker 从 public request 产出 canonical artifacts，并主动断言 request 无 private value
@pytest.mark.asyncio
async def test_evaluation_vertical_slice_uses_public_only_worker_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = tmp_path / "vertical_worker.py"
    _write_vertical_slice_worker(script)
    monkeypatch.setattr(eval_runner, "_worker_argv", lambda: [sys.executable, str(script)])

    report = await evaluate_task(_task_dir(tmp_path), tmp_path / "artifacts")

    assert report.task_success is True
    assert report.runtime_success is True
    assert report.trace_sanity_passed is True
    assert report.metrics.step_count == 1
    assert (tmp_path / "artifacts" / "report.json").is_file()
