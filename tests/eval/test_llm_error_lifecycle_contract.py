from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

import kama_claude.eval.runner as eval_runner
from kama_claude.benchmark.experiment import collect_observed_identity
from kama_claude.eval.evaluator import evaluate_task
from kama_claude.eval.failure import FailureCategory


# 创建grader可通过但runtime必失败的可信公开task
def _provider_error_task_dir(tmp_path: Path) -> Path:
    task_dir = tmp_path / "provider-error-task"
    workspace = task_dir / "public" / "workspace"
    private = task_dir / "private"
    workspace.mkdir(parents=True)
    private.mkdir()
    (workspace / "input.txt").write_text("trusted fixture", encoding="utf-8")
    (task_dir / "public" / "task.json").write_text(
        json.dumps(
            {
                "id": "provider-error-task",
                "goal": "Exercise the provider failure lifecycle contract.",
                "workspace_fixture": "public/workspace",
                "timeout_s": 10.0,
            }
        ),
        encoding="utf-8",
    )
    (private / "grader.json").write_text(
        json.dumps(
            {
                "criteria": [
                    {
                        "id": "fixture-preserved",
                        "kind": "file_exists",
                        "path": "input.txt",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return task_dir


# 读取canonical v2 journal中的domain events
def _read_events(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)["event"]
        for line in path.read_text(encoding="utf-8").splitlines()
    ]


# 功能：验证provider异常贯通真实worker/evaluator后仍是完整failed lifecycle与可验证identity
# 设计：替换唯一worker argv为tests-only fake provider，保留Runner、Journal、grader、collector全部production路径
@pytest.mark.asyncio
async def test_provider_error_is_runtime_failure_with_valid_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    helper = (
        Path(__file__).parent / "support" / "provider_error_worker.py"
    ).resolve()
    monkeypatch.setattr(
        eval_runner,
        "_worker_argv",
        lambda: [sys.executable, str(helper)],
    )
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv(
        "ANTHROPIC_BASE_URL",
        "https://api.deepseek.com/anthropic",
    )
    monkeypatch.setenv("KAMA_LLM_DEFAULT_MODEL", "deepseek-v4-pro")
    monkeypatch.setenv("KAMA_MAX_STEPS", "20")
    monkeypatch.setenv("KAMA_TRACE_ENABLED", "true")
    monkeypatch.setenv("KAMA_TRACE_INCLUDE_LLM_PAYLOAD", "true")
    output = tmp_path / "evaluation-output"

    report = await evaluate_task(_provider_error_task_dir(tmp_path), output)
    attempt_root = (
        output
        / "attempts"
        / report.task_id
        / report.attempt_id
    )
    events = _read_events(attempt_root / "runtime" / "events.v2.jsonl")
    observed = collect_observed_identity(attempt_root)

    assert [event["type"] for event in events] == [
        "run.started",
        "step.started",
        "llm.model_selected",
        "step.finished",
        "run.finished",
    ]
    assert events[-1]["status"] == "failed"
    assert events[-1]["reason"] == "llm_error"
    assert events[-1]["steps"] == 1
    assert report.runtime_success is False
    assert report.task_success is False
    assert report.trace_sanity_passed is True
    assert report.failure_category is FailureCategory.RUNTIME_FAILED
    assert len(report.criteria) == 1
    assert report.criteria[0].passed is True
    assert observed.provider.model_id == "deepseek-v4-pro"
    assert observed.api_call_count == 1
    assert observed.model_event_ids == ["deepseek-v4-pro"]
