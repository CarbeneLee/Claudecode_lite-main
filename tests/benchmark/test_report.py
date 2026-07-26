from __future__ import annotations

import importlib
import json
from pathlib import Path
from types import ModuleType

import pytest

from kama_claude.benchmark.analyzers import (
    AttemptAnalysis,
    aggregate_attempts,
)
from kama_claude.benchmark.schema import load_suite
from kama_claude.eval.failure import FailureCategory
from kama_claude.eval.metrics import BasicMetrics, TokenUsage
from kama_claude.eval.report import EvaluationReport, PublicCriterionResult


# 加载尚待实现的 report 模块，并把缺失模块转换为清晰的 RED 断言
def _report_module() -> ModuleType:
    try:
        return importlib.import_module("kama_claude.benchmark.report")
    except ModuleNotFoundError:
        pytest.fail("benchmark report module is missing")


# 创建用于 identity hash 的最小合法 bug benchmark task
def _loaded_suite(tmp_path: Path) -> object:
    tasks_root = tmp_path / "tasks"
    task_dir = tasks_root / "bugfix-001"
    workspace = task_dir / "public" / "workspace"
    private = task_dir / "private"
    workspace.mkdir(parents=True)
    private.mkdir()
    (workspace / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    (task_dir / "public" / "task.json").write_text(
        json.dumps(
            {
                "id": "bugfix-001",
                "goal": "Fix the public bug.",
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
                        "id": "target-tests",
                        "kind": "file_contains",
                        "path": "module.py",
                        "text": "PRIVATE_EXPECTED_VALUE",
                    },
                    {
                        "id": "regression-tests",
                        "kind": "file_exists",
                        "path": "module.py",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    (task_dir / "benchmark.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "task_id": "bugfix-001",
                "task_version": 1,
                "category": "bug_fixing",
                "criterion_groups": {
                    "target_behavior": ["target-tests"],
                    "regression": ["regression-tests"],
                },
            }
        ),
        encoding="utf-8",
    )
    suite_path = tmp_path / "suite.json"
    suite_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "suite_id": "kama-internal-mvp",
                "suite_version": 1,
                "task_ids": ["bugfix-001"],
            }
        ),
        encoding="utf-8",
    )
    return load_suite(suite_path, tasks_root)


# 构造一个成功的机械 attempt analysis
def _attempt() -> AttemptAnalysis:
    return AttemptAnalysis(
        task_id="bugfix-001",
        category="bug_fixing",
        repeat=1,
        task_success=True,
        runtime_success=True,
        trace_sanity_passed=True,
        failure_category=FailureCategory.NONE,
        step_count=3,
        tool_count=2,
        retry_count=0,
        wall_latency_ms=25.0,
        token_usage=TokenUsage(
            input_tokens=20,
            output_tokens=5,
            cache_tokens=4,
        ),
        group_results={"target_behavior": True, "regression": True},
        regression_introduced=False,
        changed_files=1,
        source_changed_files=1,
        test_changed_files=0,
        diff_additions=1,
        diff_deletions=1,
        tests_passed=2,
        tests_failed=0,
        coverage_delta=None,
        retry_sequences=0,
        recovered_retries=0,
        retry_recovery_rate=None,
    )


# 功能：验证 experiment identity 对 suite/task 内容、commit、model 和环境做可比较指纹
# 设计：使用注入的 repository state 避免依赖当前 Git，再修改 fixture 证明 task hash 会变化
def test_capture_experiment_identity_hashes_fixed_inputs(tmp_path: Path) -> None:
    report_module = _report_module()
    suite = _loaded_suite(tmp_path)
    repository = report_module.RepositoryState(
        commit="a" * 40,
        dirty=False,
    )

    first = report_module.capture_experiment_identity(
        suite,
        repeats=3,
        model_id="claude-fixed-model",
        repository=repository,
    )
    fixture = suite.tasks[0].evaluation_task.workspace_fixture / "module.py"
    fixture.write_text("VALUE = 2\n", encoding="utf-8")
    second = report_module.capture_experiment_identity(
        suite,
        repeats=3,
        model_id="claude-fixed-model",
        repository=repository,
    )

    assert first.suite_id == "kama-internal-mvp"
    assert first.suite_version == 1
    assert first.repeats == 3
    assert first.model_id == "claude-fixed-model"
    assert first.git_commit == "a" * 40
    assert first.git_dirty is False
    assert first.task_hashes["bugfix-001"] != second.task_hashes["bugfix-001"]
    assert len(first.suite_hash) == 64
    assert first.python_version
    assert first.platform


# 功能：验证 repository probe 只返回 full commit 与 dirty 布尔值而不暴露变更路径
# 设计：对当前真实 Git checkout 执行只读 probe，断言公开模型字段严格且 commit 可复现
def test_probe_repository_records_redacted_git_identity() -> None:
    report_module = _report_module()
    repository_root = Path(__file__).resolve().parents[2]

    state = report_module.probe_repository(repository_root)

    assert len(state.commit) == 40
    assert set(state.commit) <= set("0123456789abcdef")
    assert isinstance(state.dirty, bool)
    assert set(state.model_dump()) == {"commit", "dirty"}


# 功能：验证 baseline JSON 与 Markdown 来自同一模型并强制披露四项 claim boundary
# 设计：写入固定 identity/analysis 后解析 JSON、检查 Markdown 和 private secret 全文泄漏
def test_write_baseline_report_is_canonical_and_discloses_limits(
    tmp_path: Path,
) -> None:
    report_module = _report_module()
    suite = _loaded_suite(tmp_path)
    identity = report_module.capture_experiment_identity(
        suite,
        repeats=1,
        model_id="claude-fixed-model",
        repository=report_module.RepositoryState(
            commit="b" * 40,
            dirty=False,
        ),
    )
    attempts = [_attempt()]
    report = report_module.build_baseline_report(
        identity,
        attempts,
        aggregate_attempts(attempts),
    )
    output = tmp_path / "baseline-output"

    report_module.write_baseline_report(output, report)

    json_text = (output / "baseline.json").read_text(encoding="utf-8")
    markdown = (output / "baseline.md").read_text(encoding="utf-8")
    payload = json.loads(json_text)
    assert payload == report.model_dump(mode="json")
    assert payload["scope"] == "fixed_task_internal_benchmark"
    assert payload["security_boundary"] == "process_isolation_not_security_sandbox"
    assert payload["statistical_claim"] == "descriptive_not_statistically_significant"
    assert payload["external_benchmark_claim"] == "not_swe_bench_or_general_coding_ability"
    assert "Fixed-task internal benchmark" in markdown
    assert "not a security sandbox" in markdown
    assert "not statistically significant" in markdown
    assert "not SWE-bench" in markdown
    assert "bugfix-001" in markdown
    assert "PRIVATE_EXPECTED_VALUE" not in json_text
    assert "PRIVATE_EXPECTED_VALUE" not in markdown


# 功能：验证公开 baseline report 不接受空 attempts 或 identity/attempt 数量错配
# 设计：分别传入空列表和 repeats 声明三次但只给一次的结果，防止漂亮但不完整的报告
def test_build_baseline_report_rejects_incomplete_attempt_matrix(
    tmp_path: Path,
) -> None:
    report_module = _report_module()
    suite = _loaded_suite(tmp_path)
    identity = report_module.capture_experiment_identity(
        suite,
        repeats=3,
        model_id="claude-fixed-model",
        repository=report_module.RepositoryState(
            commit="c" * 40,
            dirty=True,
        ),
    )

    with pytest.raises(ValueError, match="attempt matrix"):
        report_module.build_baseline_report(
            identity,
            [],
            aggregate_attempts([_attempt()]),
        )
    with pytest.raises(ValueError, match="attempt matrix"):
        report_module.build_baseline_report(
            identity,
            [_attempt()],
            aggregate_attempts([_attempt()]),
        )


# 功能：验证 kama-bench CLI 只暴露固定 suite run 参数而没有 runtime 或 comparison 配置
# 设计：解析合法 run 后逐一拒绝 provider、model、prompt、compare 和 dashboard 参数
def test_benchmark_cli_exposes_only_fixed_suite_run() -> None:
    try:
        cli = importlib.import_module("kama_claude.benchmark.cli")
    except ModuleNotFoundError:
        pytest.fail("benchmark CLI module is missing")

    args = cli._parse_args(
        [
            "run",
            "--suite",
            "suite.json",
            "--tasks-root",
            "tasks",
            "--output",
            "artifacts",
            "--repeats",
            "3",
        ]
    )

    assert vars(args) == {
        "command": "run",
        "suite": "suite.json",
        "tasks_root": "tasks",
        "output": "artifacts",
        "repeats": 3,
    }
    for forbidden in (
        ["compare"],
        [
            "run",
            "--suite",
            "suite.json",
            "--tasks-root",
            "tasks",
            "--output",
            "out",
            "--repeats",
            "1",
            "--provider",
            "scripted",
        ],
        [
            "run",
            "--suite",
            "suite.json",
            "--tasks-root",
            "tasks",
            "--output",
            "out",
            "--repeats",
            "1",
            "--model",
            "other",
        ],
        ["dashboard"],
    ):
        with pytest.raises(SystemExit):
            cli._parse_args(forbidden)


# 写入一次 fake Phase 8A evaluator 的 canonical artifact 与公开 report
def _write_fake_evaluation(
    task_dir: Path | str,
    output: Path | str,
) -> EvaluationReport:
    task_id = Path(task_dir).name
    attempt_id = "attempt-cli"
    root = Path(output) / "attempts" / task_id / attempt_id
    runtime = root / "runtime"
    private = root / "private"
    runtime.mkdir(parents=True)
    private.mkdir()
    manifest = {
        "files": [
            {"path": "module.py", "size": 10, "sha256": "1" * 64},
        ],
        "total_bytes": 10,
    }
    for name in ("initial-workspace.json", "final-workspace.json"):
        (runtime / name).write_text(json.dumps(manifest), encoding="utf-8")
    (runtime / "workspace.diff").write_text("", encoding="utf-8")
    (runtime / "events.v2.jsonl").write_text(
        json.dumps(
            {
                "event_id": "event-1",
                "stream_id": "run:test",
                "seq": 1,
                "event": {"type": "run.started"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (private / "command-results.json").write_text("[]\n", encoding="utf-8")
    criteria = [
        PublicCriterionResult(id="target-tests", kind="file_contains", passed=True),
        PublicCriterionResult(id="regression-tests", kind="file_exists", passed=True),
    ]
    return EvaluationReport(
        task_id=task_id,
        attempt_id=attempt_id,
        task_success=True,
        runtime_success=True,
        trace_sanity_passed=True,
        failure_category=FailureCategory.NONE,
        criteria=criteria,
        metrics=BasicMetrics(
            task_success=True,
            runtime_success=True,
            step_count=1,
            tool_count=0,
            retry_count=0,
            wall_latency_ms=5.0,
            token_usage=TokenUsage(
                input_tokens=3,
                output_tokens=1,
                cache_tokens=0,
            ),
            failure_category=FailureCategory.NONE,
        ),
    )


# 功能：验证 CLI execution 贯通真实 suite、orchestrator、analyzer 与 baseline writer
# 设计：只替换整个 Phase 8A evaluator 边界并写 canonical artifacts，避免 mock 内部聚合函数
@pytest.mark.asyncio
async def test_execute_benchmark_wires_real_observer_pipeline(tmp_path: Path) -> None:
    try:
        cli = importlib.import_module("kama_claude.benchmark.cli")
    except ModuleNotFoundError:
        pytest.fail("benchmark CLI module is missing")
    _loaded_suite(tmp_path)
    output = tmp_path / "benchmark-output"
    args = cli._parse_args(
        [
            "run",
            "--suite",
            str(tmp_path / "suite.json"),
            "--tasks-root",
            str(tmp_path / "tasks"),
            "--output",
            str(output),
            "--repeats",
            "1",
        ]
    )

    # 保持 async evaluator 合同，同时复用 helper 写真实 artifact layout
    async def fake_evaluator(
        task_dir: Path | str,
        evaluation_output: Path | str,
    ) -> EvaluationReport:
        return _write_fake_evaluation(task_dir, evaluation_output)

    exit_code = await cli.execute_benchmark(
        args,
        evaluator=fake_evaluator,
        repository=cli.RepositoryState(commit="d" * 40, dirty=False),
        model_id="claude-fixed-model",
    )

    assert exit_code == 0
    payload = json.loads((output / "baseline.json").read_text(encoding="utf-8"))
    assert payload["metrics"]["overall"]["successful_attempts"] == 1
    assert payload["attempts"][0]["task_id"] == "bugfix-001"
    assert (output / "baseline.md").is_file()
