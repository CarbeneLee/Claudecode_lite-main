from __future__ import annotations

import hashlib
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


# 构造 report-facing verified experiment identity，复用真实 suite/task hash 但不调用模型
def _verified_identity(
    report_module: ModuleType,
    suite: object,
    *,
    repeats: int,
) -> object:
    legacy = report_module.capture_experiment_identity(
        suite,
        repeats=repeats,
        model_id="deepseek-v4-pro",
        repository=report_module.RepositoryState(commit="b" * 40, dirty=False),
    )
    provider = {
        "service_provider": "deepseek",
        "wire_protocol": "anthropic_messages",
        "endpoint_id": "deepseek-anthropic-compatible",
        "endpoint": "https://api.deepseek.com/anthropic",
        "model_id": "deepseek-v4-pro",
        "sdk_distribution": "anthropic",
        "sdk_version": "0.111.0",
    }
    runtime = {
        "max_steps": 20,
        "router": "static",
        "compaction_threshold": 0.0,
        "tool_result_limit": 8000,
        "tool_result_keep": 4000,
        "mcp_enabled": False,
        "trace_enabled": True,
        "include_llm_payload": True,
    }
    experiment = importlib.import_module("kama_claude.benchmark.experiment")
    declared = experiment.DeclaredExperimentIdentity(
        profile_id="baseline-profile",
        profile_hash="1" * 64,
        git={"commit": "b" * 40, "dirty": False},
        suite={
            "suite_id": legacy.suite_id,
            "suite_version": legacy.suite_version,
            "suite_hash": legacy.suite_hash,
            "task_hashes": legacy.task_hashes,
            "grader_hashes": {task_id: "2" * 64 for task_id in legacy.task_hashes},
        },
        provider=provider,
        prompt_hash="3" * 64,
        tool_schema_hash="4" * 64,
        runtime=runtime,
        runtime_config_hash=experiment.canonical_hash(runtime),
        dependency={
            "pyproject_hash": "5" * 64,
            "uv_lock_hash": "6" * 64,
            "dependency_hash": "7" * 64,
        },
        host={
            "python_version": "3.12.13",
            "os": "Darwin",
            "os_release": "test",
            "architecture": "arm64",
        },
        schedule={
            "repeats": repeats,
            "execution_order": "suite_task_then_repeat_ascending",
        },
        artifacts={
            "output_root_must_be_new": True,
            "output_root_must_be_outside_repository": True,
            "retain_all_attempts": True,
            "raw_trace_visibility": "private",
        },
    )
    return experiment.VerifiedExperimentIdentity(
        declared=declared,
        observed={
            "provider": provider,
            "prompt_hash": "3" * 64,
            "tool_schema_hash": "4" * 64,
            "runtime": runtime,
            "runtime_config_hash": experiment.canonical_hash(runtime),
            "attempts": repeats,
            "api_calls": repeats,
            "model_event_ids": ["deepseek-v4-pro"],
        },
        verification={
            "status": "match",
            "verified_attempts": repeats,
            "mismatches": [],
        },
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
    identity = _verified_identity(report_module, suite, repeats=1)
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
    assert payload["artifact_version"] == 2
    assert payload["experiment"]["status"] == "valid"
    assert payload["experiment"]["declared"]["provider"]["service_provider"] == "deepseek"
    assert payload["experiment"]["declared"]["provider"]["wire_protocol"] == (
        "anthropic_messages"
    )
    assert payload["experiment"]["declared"]["provider"]["endpoint_id"] == (
        "deepseek-anthropic-compatible"
    )
    assert payload["experiment"]["declared"]["provider"]["model_id"] == "deepseek-v4-pro"
    assert payload["experiment"]["declared"]["prompt_hash"] == "3" * 64
    assert payload["experiment"]["declared"]["tool_schema_hash"] == "4" * 64
    assert payload["experiment"]["declared"]["runtime"]["max_steps"] == 20
    assert payload["experiment"]["verification"]["status"] == "match"
    assert payload["scope"] == "fixed_task_internal_benchmark"
    assert payload["security_boundary"] == "process_isolation_not_security_sandbox"
    assert payload["statistical_claim"] == "descriptive_not_statistically_significant"
    assert payload["external_benchmark_claim"] == "not_swe_bench_or_general_coding_ability"
    assert "Fixed-task internal benchmark" in markdown
    assert "not a security sandbox" in markdown
    assert "not statistically significant" in markdown
    assert "not SWE-bench" in markdown
    assert "deepseek" in markdown
    assert "anthropic_messages" in markdown
    assert "deepseek-v4-pro" in markdown
    assert "Prompt hash" in markdown
    assert "Tool schema hash" in markdown
    assert "Max steps" in markdown
    assert "Runtime config hash" in markdown
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
    identity = _verified_identity(report_module, suite, repeats=3)

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
# 设计：解析 profile-driven run 后逐一拒绝 ad-hoc suite、repeat、provider、model 和 dashboard 参数
def test_benchmark_cli_exposes_only_fixed_suite_run() -> None:
    try:
        cli = importlib.import_module("kama_claude.benchmark.cli")
    except ModuleNotFoundError:
        pytest.fail("benchmark CLI module is missing")

    args = cli._parse_args(
        [
            "run",
            "--experiment",
            "experiment.json",
            "--output",
            "artifacts",
        ]
    )

    assert vars(args) == {
        "command": "run",
        "experiment": "experiment.json",
        "output": "artifacts",
    }
    for forbidden in (
        ["compare"],
        [
            "run",
            "--experiment",
            "experiment.json",
            "--output",
            "out",
            "--provider",
            "scripted",
        ],
        [
            "run",
            "--experiment",
            "experiment.json",
            "--output",
            "out",
            "--model",
            "other",
        ],
        [
            "run",
            "--experiment",
            "experiment.json",
            "--output",
            "out",
            "--repeats",
            "1",
        ],
        [
            "run",
            "--suite",
            "suite.json",
            "--tasks-root",
            "tasks",
            "--output",
            "out",
        ],
        ["dashboard"],
    ):
        with pytest.raises(SystemExit):
            cli._parse_args(forbidden)


# 写入一次带 matching runtime identity evidence 的 fake Phase 8A artifact 与公开 report
def _write_fake_evaluation(
    task_dir: Path | str,
    output: Path | str,
    *,
    prompt: str,
    tools: list[dict[str, object]],
    max_steps: int = 20,
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
    runtime_identity = {
        "provider": {
            "service_provider": "deepseek",
            "wire_protocol": "anthropic_messages",
            "endpoint_id": "deepseek-anthropic-compatible",
            "endpoint": "https://api.deepseek.com/anthropic",
            "model_id": "deepseek-v4-pro",
            "sdk_distribution": "anthropic",
            "sdk_version": "0.111.0",
        },
        "runtime": {
            "max_steps": max_steps,
            "router": "static",
            "compaction_threshold": 0.0,
            "tool_result_limit": 8000,
            "tool_result_keep": 4000,
            "mcp_enabled": False,
            "trace_enabled": True,
            "include_llm_payload": True,
        },
    }
    trace_records = [
        {
            "ts": "2026-07-26T00:00:00+00:00",
            "direction": "CORE",
            "layer": "event",
            "kind": "runtime_identity",
            "run_id": "run-cli",
            "data": runtime_identity,
        },
        {
            "ts": "2026-07-26T00:00:01+00:00",
            "direction": "CORE→LLM",
            "layer": "llm",
            "kind": "api_call",
            "run_id": "run-cli",
            "step": 1,
            "data": {
                "messages": [{"role": "user", "content": "private"}],
                "tool_schemas": tools,
                "system": prompt,
            },
        },
    ]
    (runtime / "trace.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in trace_records),
        encoding="utf-8",
    )
    (runtime / "events.v2.jsonl").write_text(
        json.dumps(
            {
                "event_id": "event-1",
                "stream_id": "run:run-cli",
                "seq": 1,
                "event": {
                    "type": "llm.model_selected",
                    "run_id": "run-cli",
                    "model": "deepseek-v4-pro",
                    "strategy": "static",
                },
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


# 为 CLI vertical test 写入 strict profile、freeze manifest 与 dependency inputs
def _write_cli_experiment_profile(
    tmp_path: Path,
    report_module: ModuleType,
    suite: object,
    *,
    prompt: str,
    tools: list[dict[str, object]],
) -> Path:
    experiment = importlib.import_module("kama_claude.benchmark.experiment")
    experiments = tmp_path / "experiments"
    experiments.mkdir()
    legacy = report_module.capture_experiment_identity(
        suite,
        repeats=1,
        model_id="deepseek-v4-pro",
        repository=report_module.RepositoryState(commit="d" * 40, dirty=False),
    )
    task = suite.tasks[0]
    grader_path = task.task_dir / "private" / "grader.json"
    grader_digest = hashlib.sha256()
    grader_digest.update(
        grader_path.relative_to(task.task_dir).as_posix().encode("utf-8")
    )
    grader_digest.update(b"\0")
    grader_digest.update(grader_path.read_bytes())
    grader_digest.update(b"\0")
    freeze = {
        "schema_version": 1,
        "suite_id": legacy.suite_id,
        "suite_version": legacy.suite_version,
        "suite_hash": legacy.suite_hash,
        "tasks": [
            {
                "task_id": task.metadata.task_id,
                "task_hash": legacy.task_hashes[task.metadata.task_id],
                "grader_hash": grader_digest.hexdigest(),
                "reference_hash": "9" * 64,
            }
        ],
    }
    (tmp_path / "freeze.json").write_text(json.dumps(freeze), encoding="utf-8")
    profile = {
        "schema_version": 1,
        "profile_id": "cli-baseline-profile",
        "suite": {
            "manifest": "../suite.json",
            "freeze_manifest": "../freeze.json",
            "tasks_root": "../tasks",
            "expected_suite_hash": legacy.suite_hash,
        },
        "provider": {
            "service_provider": "deepseek",
            "wire_protocol": "anthropic_messages",
            "endpoint_id": "deepseek-anthropic-compatible",
            "endpoint": "https://api.deepseek.com/anthropic",
            "model_id": "deepseek-v4-pro",
            "sdk_distribution": "anthropic",
            "sdk_version": "0.111.0",
            "credential_env": "ANTHROPIC_API_KEY",
        },
        "runtime": {
            "max_steps": 20,
            "router": "static",
            "compaction_threshold": 0.0,
            "tool_result_limit": 8000,
            "tool_result_keep": 4000,
            "mcp_enabled": False,
            "trace_enabled": True,
            "include_llm_payload": True,
        },
        "schedule": {
            "repeats": 1,
            "execution_order": "suite_task_then_repeat_ascending",
        },
        "artifacts": {
            "output_root_must_be_new": True,
            "output_root_must_be_outside_repository": True,
            "retain_all_attempts": True,
            "raw_trace_visibility": "private",
        },
        "expected_identity": {
            "prompt_hash": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "tool_schema_hash": experiment.canonical_hash(tools),
        },
    }
    profile_path = experiments / "profile.json"
    profile_path.write_text(json.dumps(profile), encoding="utf-8")
    repository = tmp_path / "repository"
    repository.mkdir()
    (repository / "pyproject.toml").write_text(
        "[project]\nname='test'\n",
        encoding="utf-8",
    )
    (repository / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    return profile_path


# 功能：验证 CLI execution 贯通真实 suite、orchestrator、analyzer 与 baseline writer
# 设计：只替换 Phase 8A evaluator，并提供 matching trace/journal，验证 declaration 到 report 全链路
@pytest.mark.asyncio
async def test_execute_benchmark_wires_real_observer_pipeline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    try:
        cli = importlib.import_module("kama_claude.benchmark.cli")
    except ModuleNotFoundError:
        pytest.fail("benchmark CLI module is missing")
    suite = _loaded_suite(tmp_path)
    prompt = "effective prompt"
    tools = [{"name": "read_file", "input_schema": {"type": "object"}}]
    profile_path = _write_cli_experiment_profile(
        tmp_path,
        _report_module(),
        suite,
        prompt=prompt,
        tools=tools,
    )
    output = tmp_path / "benchmark-output"
    args = cli._parse_args(
        [
            "run",
            "--experiment",
            str(profile_path),
            "--output",
            str(output),
        ]
    )
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-credential-not-an-api-call")

    # 保持 async evaluator 合同，同时复用 helper 写真实 artifact layout
    async def fake_evaluator(
        task_dir: Path | str,
        evaluation_output: Path | str,
    ) -> EvaluationReport:
        return _write_fake_evaluation(
            task_dir,
            evaluation_output,
            prompt=prompt,
            tools=tools,
        )

    exit_code = await cli.execute_experiment(
        args,
        evaluator=fake_evaluator,
        repository=cli.RepositoryState(commit="d" * 40, dirty=False),
        repository_root=tmp_path / "repository",
        installed_sdk_version="0.111.0",
    )

    assert exit_code == 0
    declared = json.loads(
        (output / "declared-experiment.json").read_text(encoding="utf-8")
    )
    payload = json.loads((output / "baseline.json").read_text(encoding="utf-8"))
    assert declared["provider"]["model_id"] == "deepseek-v4-pro"
    assert payload["artifact_version"] == 2
    assert payload["experiment"]["status"] == "valid"
    assert payload["experiment"]["verification"]["status"] == "match"
    assert payload["metrics"]["overall"]["successful_attempts"] == 1
    assert payload["attempts"][0]["task_id"] == "bugfix-001"
    assert (output / "baseline.md").is_file()


# 功能：验证 behavior identity mismatch 写 invalid receipt、退出 2 且绝不生成 baseline score
# 设计：只让 observed max_steps 偏离 declaration，保留完整 artifact shape 以隔离 identity 分支
@pytest.mark.asyncio
async def test_execute_experiment_identity_mismatch_is_invalid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cli = importlib.import_module("kama_claude.benchmark.cli")
    suite = _loaded_suite(tmp_path)
    prompt = "effective prompt"
    tools = [{"name": "read_file", "input_schema": {"type": "object"}}]
    profile_path = _write_cli_experiment_profile(
        tmp_path,
        _report_module(),
        suite,
        prompt=prompt,
        tools=tools,
    )
    output = tmp_path / "invalid-output"
    args = cli._parse_args(
        [
            "run",
            "--experiment",
            str(profile_path),
            "--output",
            str(output),
        ]
    )
    monkeypatch.setenv("ANTHROPIC_API_KEY", "must-not-be-serialized")

    # 生成合法 Phase 8A report，但 runtime identity 故意声明错误 max_steps
    async def mismatched_evaluator(
        task_dir: Path | str,
        evaluation_output: Path | str,
    ) -> EvaluationReport:
        return _write_fake_evaluation(
            task_dir,
            evaluation_output,
            prompt=prompt,
            tools=tools,
            max_steps=99,
        )

    exit_code = await cli.execute_experiment(
        args,
        evaluator=mismatched_evaluator,
        repository=cli.RepositoryState(commit="d" * 40, dirty=False),
        repository_root=tmp_path / "repository",
        installed_sdk_version="0.111.0",
    )

    invalid_text = (output / "experiment-invalid.json").read_text(encoding="utf-8")
    assert exit_code == 2
    assert (output / "declared-experiment.json").is_file()
    assert json.loads(invalid_text)["mismatches"] == [
        "runtime",
        "runtime_config_hash",
    ]
    assert not (output / "baseline.json").exists()
    assert not (output / "baseline.md").exists()
    assert "must-not-be-serialized" not in invalid_text
