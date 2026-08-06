from __future__ import annotations

import asyncio
import importlib
import json
from pathlib import Path
from types import ModuleType

import pytest

from kama_claude.benchmark.schema import load_suite
from kama_claude.eval.failure import FailureCategory
from kama_claude.eval.metrics import BasicMetrics, TokenUsage
from kama_claude.eval.report import EvaluationReport


# 加载尚待实现的 orchestrator 模块，并把缺失模块转换为清晰的 RED 断言
def _orchestrator_module() -> ModuleType:
    try:
        return importlib.import_module("kama_claude.benchmark.orchestrator")
    except ModuleNotFoundError:
        pytest.fail("benchmark orchestrator module is missing")


# 创建两个 criterion group 对应的最小 Phase 8A task
def _write_task(tasks_root: Path, task_id: str, category: str) -> None:
    task_dir = tasks_root / task_id
    workspace = task_dir / "public" / "workspace"
    private = task_dir / "private"
    workspace.mkdir(parents=True)
    private.mkdir()
    (workspace / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    (task_dir / "public" / "task.json").write_text(
        json.dumps(
            {
                "id": task_id,
                "goal": "Complete the requested repository task.",
                "workspace_fixture": "public/workspace",
                "timeout_s": 30.0,
            }
        ),
        encoding="utf-8",
    )
    groups = {
        "target_behavior": ["target-tests"],
        "regression": ["regression-tests"],
    }
    (private / "grader.json").write_text(
        json.dumps(
            {
                "criteria": [
                    {
                        "id": criterion_id,
                        "kind": "file_exists",
                        "path": f"{criterion_id}.txt",
                    }
                    for criterion_id in ("target-tests", "regression-tests")
                ]
            }
        ),
        encoding="utf-8",
    )
    (task_dir / "benchmark.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "task_id": task_id,
                "task_version": 1,
                "category": category,
                "criterion_groups": groups,
            }
        ),
        encoding="utf-8",
    )


# 创建一个包含 bug 与 feature task 的合法 suite
def _loaded_suite(tmp_path: Path) -> object:
    tasks_root = tmp_path / "tasks"
    _write_task(tasks_root, "bugfix-001", "bug_fixing")
    _write_task(tasks_root, "feature-001", "feature_implementation")
    suite_path = tmp_path / "suite.json"
    suite_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "suite_id": "kama-internal-mvp",
                "suite_version": 1,
                "task_ids": ["bugfix-001", "feature-001"],
            }
        ),
        encoding="utf-8",
    )
    return load_suite(suite_path, tasks_root)


# 构造只含公开 Phase 8A 字段的 deterministic attempt report
def _report(task_id: str, attempt_id: str) -> EvaluationReport:
    return EvaluationReport(
        task_id=task_id,
        attempt_id=attempt_id,
        task_success=True,
        runtime_success=True,
        trace_sanity_passed=True,
        failure_category=FailureCategory.NONE,
        criteria=[],
        metrics=BasicMetrics(
            task_success=True,
            runtime_success=True,
            step_count=1,
            tool_count=1,
            retry_count=0,
            wall_latency_ms=10.0,
            token_usage=TokenUsage(
                input_tokens=5,
                output_tokens=2,
                cache_tokens=0,
            ),
            failure_category=FailureCategory.NONE,
        ),
    )


# 功能：验证 orchestrator 按 suite 顺序和 repeat 顺序逐次调用 Phase 8A evaluator
# 设计：fake evaluator 主动 yield 并记录最大并发数，既锁定调用参数也能杀死 gather 并发实现
@pytest.mark.asyncio
async def test_orchestrator_runs_phase8a_attempts_sequentially(tmp_path: Path) -> None:
    orchestrator = _orchestrator_module()
    suite = _loaded_suite(tmp_path)
    calls: list[tuple[str, Path]] = []
    active = 0
    max_active = 0

    # 记录 evaluator 的真实 task/output 边界，并在事件循环中让并发实现暴露重叠
    async def fake_evaluate(task_dir: Path | str, output: Path | str) -> EvaluationReport:
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        await asyncio.sleep(0)
        task_id = Path(task_dir).name
        calls.append((task_id, Path(output)))
        active -= 1
        return _report(task_id, f"attempt-{len(calls)}")

    run = await orchestrator.run_suite(
        suite,
        tmp_path / "artifacts",
        repeats=2,
        evaluator=fake_evaluate,
    )

    assert max_active == 1
    assert [(attempt.task_id, attempt.repeat) for attempt in run.attempts] == [
        ("bugfix-001", 1),
        ("bugfix-001", 2),
        ("feature-001", 1),
        ("feature-001", 2),
    ]
    assert [task_id for task_id, _ in calls] == [
        "bugfix-001",
        "bugfix-001",
        "feature-001",
        "feature-001",
    ]
    assert [path.relative_to(tmp_path / "artifacts").as_posix() for _, path in calls] == [
        "tasks/bugfix-001/repeat-01/evaluation",
        "tasks/bugfix-001/repeat-02/evaluation",
        "tasks/feature-001/repeat-01/evaluation",
        "tasks/feature-001/repeat-02/evaluation",
    ]


# 功能：验证 orchestrator 默认依赖就是 Phase 8A evaluate_task 而非第二套 evaluator
# 设计：替换模块级 evaluate_task seam 后不传 evaluator，断言唯一调用收到原 task directory
@pytest.mark.asyncio
async def test_orchestrator_defaults_to_phase8a_evaluate_task(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orchestrator = _orchestrator_module()
    suite = _loaded_suite(tmp_path)
    seen: list[Path] = []

    # 代替网络 Agent run，但保持与 Phase 8A evaluate_task 相同的调用合同
    async def fake_phase8a(
        task_dir: Path | str,
        output: Path | str,
    ) -> EvaluationReport:
        seen.append(Path(task_dir))
        return _report(Path(task_dir).name, f"attempt-{len(seen)}")

    monkeypatch.setattr(orchestrator, "evaluate_task", fake_phase8a)

    run = await orchestrator.run_suite(suite, tmp_path / "artifacts", repeats=1)

    assert seen == [task.task_dir for task in suite.tasks]
    assert len(run.attempts) == 2


# 功能：验证 caller cancellation 保留异常对象且不会启动后续 benchmark attempt
# 设计：首 attempt 用 Event 停住并捕获 cancellation identity，释放后断言调用列表仍只有首项
@pytest.mark.asyncio
async def test_orchestrator_preserves_cancellation_and_stops_future_attempts(
    tmp_path: Path,
) -> None:
    orchestrator = _orchestrator_module()
    suite = _loaded_suite(tmp_path)
    entered = asyncio.Event()
    calls: list[str] = []
    captured: list[asyncio.CancelledError] = []

    # 首个 evaluator 等待 caller cancellation，不自行转换为普通 benchmark failure
    async def blocking_evaluate(
        task_dir: Path | str,
        output: Path | str,
    ) -> EvaluationReport:
        calls.append(Path(task_dir).name)
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError as exc:
            captured.append(exc)
            raise
        raise AssertionError("unreachable")

    task = asyncio.create_task(
        orchestrator.run_suite(
            suite,
            tmp_path / "artifacts",
            repeats=2,
            evaluator=blocking_evaluate,
        )
    )
    await entered.wait()
    task.cancel("benchmark-cancel")

    with pytest.raises(asyncio.CancelledError) as raised:
        await task

    assert raised.value is captured[0]
    assert calls == ["bugfix-001"]


# 功能：验证 repeats 边界和 evaluator task identity mismatch 都会 fail closed
# 设计：先用零与四拒绝超范围实验，再返回错误 task ID 防止报告被归到错误任务
@pytest.mark.asyncio
async def test_orchestrator_rejects_invalid_repeats_and_report_identity(
    tmp_path: Path,
) -> None:
    orchestrator = _orchestrator_module()
    suite = _loaded_suite(tmp_path)

    for repeats in (0, 4):
        with pytest.raises(ValueError, match="repeats"):
            await orchestrator.run_suite(
                suite,
                tmp_path / f"invalid-{repeats}",
                repeats=repeats,
                evaluator=lambda _task, _output: asyncio.sleep(0),
            )

    # 返回另一 task 的 report，验证 orchestrator 不接受错配 evidence
    async def mismatched_evaluate(
        task_dir: Path | str,
        output: Path | str,
    ) -> EvaluationReport:
        return _report("wrong-task", "attempt-wrong")

    with pytest.raises(ValueError, match="report task identity mismatch"):
        await orchestrator.run_suite(
            suite,
            tmp_path / "mismatch",
            repeats=1,
            evaluator=mismatched_evaluate,
        )
