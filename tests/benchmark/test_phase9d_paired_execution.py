from __future__ import annotations

import ast
import hashlib
import importlib
import importlib.util
import json
import os
import signal
import subprocess
import sys
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import ModuleType

import pytest
from pydantic import ValidationError

from kama_claude.benchmark.analyzers import AttemptAnalysis, aggregate_attempts
from kama_claude.benchmark.experiment import (
    DeclaredExperimentIdentity,
    VerifiedExperimentIdentity,
)
from kama_claude.benchmark.report import (
    BaselineReport,
    write_baseline_report,
)
from kama_claude.eval.failure import FailureCategory
from kama_claude.eval.metrics import TokenUsage
from kama_claude.eval.report import EvaluationReport, write_report

_TASKS = (
    ("bugfix-subtract", "bug_fixing"),
    ("feature-low-stock", "feature_implementation"),
    ("testgen-normalize-username", "test_generation"),
    ("bugfix-config-precedence", "bug_fixing"),
    ("feature-atomic-bulk-import", "feature_implementation"),
    ("testgen-quoted-query-parser", "test_generation"),
    ("bugfix-retry-state-idempotency", "bug_fixing"),
    ("feature-inventory-reservation-lifecycle", "feature_implementation"),
    ("testgen-dependency-planner", "test_generation"),
)


# 加载待实现的paired observer模块
def _paired() -> ModuleType:
    return importlib.import_module("kama_claude.benchmark.paired")


# 从允许新增的scripts文件加载side-effect边界
def _script() -> ModuleType:
    path = Path(__file__).resolve().parents[2] / "scripts" / "phase9d_paired.py"
    spec = importlib.util.spec_from_file_location("phase9d_paired_script", path)
    if spec is None or spec.loader is None:
        pytest.fail("paired script cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# 返回真实冻结receipt的唯一测试路径
def _receipt_path() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "benchmarks"
        / "receipts"
        / "phase9d-repaired-v1-v2-paired-experiment.json"
    )


# 返回当前KamaClaude仓库根目录
def _repository() -> Path:
    return Path(__file__).resolve().parents[2]


# 从真实Git历史读取冻结receipt所在commit
def _receipt_commit() -> str:
    result = subprocess.run(
        [
            "git",
            "log",
            "-1",
            "--format=%H",
            "--",
            str(_receipt_path().relative_to(_repository())),
        ],
        cwd=_repository(),
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


# 计算真实冻结receipt的原始byte identity
def _receipt_sha256() -> str:
    return hashlib.sha256(_receipt_path().read_bytes()).hexdigest()


# 从production Git observer获得冻结receipt reference
def _receipt_reference() -> object:
    return _paired().observe_receipt_reference(
        _repository(),
        _receipt_commit(),
        _receipt_path(),
    )


# 执行本地临时Git命令并返回stdout
def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


# 构造两个commit包含相同receipt blob的临时Git仓库
def _same_receipt_bytes_git_repo(tmp_path: Path) -> tuple[Path, str, str, Path]:
    repo = tmp_path / "receipt-git"
    receipt = repo / "benchmarks" / "receipts" / _receipt_path().name
    receipt.parent.mkdir(parents=True)
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    _git(repo, "config", "user.email", "phase9d@example.invalid")
    _git(repo, "config", "user.name", "Phase 9D")
    receipt.write_bytes(_receipt_path().read_bytes())
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "add receipt")
    first = _git(repo, "rev-parse", "HEAD")
    (repo / "README.md").write_text("same receipt bytes\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "unrelated change")
    second = _git(repo, "rev-parse", "HEAD")
    return repo, first, second, receipt


# 加载真实冻结receipt
def _receipt() -> object:
    return _paired().load_paired_receipt(_receipt_path())


# 构造一个指定任务结果的canonical attempt row
def _attempt(
    task_id: str,
    category: str,
    repeat: int,
    *,
    success: bool,
    timeout: bool = False,
    latency: float = 100.0,
    tokens: int = 1000,
) -> AttemptAnalysis:
    failure = (
        FailureCategory.TIMEOUT
        if timeout
        else FailureCategory.NONE
        if success
        else FailureCategory.TASK_FAILED
    )
    return AttemptAnalysis(
        task_id=task_id,
        category=category,
        repeat=repeat,
        task_success=success,
        runtime_success=not timeout,
        trace_sanity_passed=not timeout,
        failure_category=failure,
        step_count=0 if timeout else 3,
        tool_count=0 if timeout else 2,
        retry_count=0,
        wall_latency_ms=0.0 if timeout else latency,
        token_usage=TokenUsage(
            input_tokens=0 if timeout else tokens - 100,
            output_tokens=0 if timeout else 100,
            cache_tokens=0,
        ),
        group_results={"target_behavior": success, "regression": True},
        regression_introduced=False,
        changed_files=0 if timeout else 1,
        source_changed_files=0 if timeout else 1,
        test_changed_files=0,
        diff_additions=0 if timeout else 1,
        diff_deletions=0,
        tests_passed=None,
        tests_failed=None,
        coverage_delta=None,
        retry_sequences=0,
        recovered_retries=0,
        retry_recovery_rate=None,
    )


# 根据成功task集合构造完整27-row attempt matrix
def _attempts(
    successful_tasks: set[str],
    *,
    timeout_keys: set[tuple[str, int]] | None = None,
    latency: float = 100.0,
    tokens: int = 1000,
) -> list[AttemptAnalysis]:
    timeout_keys = timeout_keys or set()
    return [
        _attempt(
            task_id,
            category,
            repeat,
            success=task_id in successful_tasks and (task_id, repeat) not in timeout_keys,
            timeout=(task_id, repeat) in timeout_keys,
            latency=latency,
            tokens=tokens,
        )
        for task_id, category in _TASKS
        for repeat in range(1, 4)
    ]


# 为指定arm构造真实BaselineReport identity模型
def _identity(receipt: object, arm: str) -> VerifiedExperimentIdentity:
    arm_receipt = getattr(receipt.arms, arm)
    runtime = {
        "max_steps": receipt.shared_identity.max_steps,
        "router": "static",
        "compaction_threshold": 0.0,
        "tool_result_limit": 8000,
        "tool_result_keep": 4000,
        "mcp_enabled": False,
        "trace_enabled": True,
        "include_llm_payload": True,
    }
    provider = {
        "service_provider": receipt.shared_identity.provider,
        "wire_protocol": receipt.shared_identity.protocol,
        "endpoint_id": receipt.shared_identity.endpoint_id,
        "endpoint": "https://api.deepseek.com/anthropic",
        "model_id": receipt.shared_identity.model,
        "sdk_distribution": "anthropic",
        "sdk_version": "0.111.0",
    }
    task_hashes = {task_id: "1" * 64 for task_id, _category in _TASKS}
    declared = DeclaredExperimentIdentity(
        profile_id=arm_receipt.profile_id,
        profile_hash=arm_receipt.profile_canonical_sha256,
        git={"commit": arm_receipt.commit, "dirty": False},
        suite={
            "suite_id": "kama-coding-mvp",
            "suite_version": 1,
            "suite_hash": receipt.shared_identity.suite_sha256,
            "task_hashes": task_hashes,
            "grader_hashes": {task_id: "2" * 64 for task_id in task_hashes},
        },
        provider=provider,
        prompt_hash=arm_receipt.prompt_sha256,
        tool_schema_hash=receipt.shared_identity.tool_schema_sha256,
        runtime=runtime,
        runtime_config_hash=receipt.shared_identity.runtime_config_sha256,
        dependency={
            "pyproject_hash": "3" * 64,
            "uv_lock_hash": "4" * 64,
            "dependency_hash": receipt.shared_identity.dependency_sha256,
        },
        host={
            "python_version": receipt.host_policy.python,
            "os": receipt.host_policy.os,
            "os_release": "test",
            "architecture": receipt.host_policy.architecture,
        },
        schedule={
            "repeats": receipt.shared_identity.repeats,
            "execution_order": receipt.shared_identity.execution_order,
        },
        artifacts={
            "output_root_must_be_new": True,
            "output_root_must_be_outside_repository": True,
            "retain_all_attempts": True,
            "raw_trace_visibility": "private",
        },
    )
    return VerifiedExperimentIdentity(
        declared=declared,
        observed={
            "provider": provider,
            "prompt_hash": arm_receipt.prompt_sha256,
            "tool_schema_hash": receipt.shared_identity.tool_schema_sha256,
            "runtime": runtime,
            "runtime_config_hash": receipt.shared_identity.runtime_config_sha256,
            "attempts": receipt.arm_validity.identity_verified,
            "api_calls": 81,
            "model_event_ids": [receipt.shared_identity.model],
        },
        verification={
            "status": "match",
            "verified_attempts": receipt.arm_validity.identity_verified,
            "mismatches": [],
        },
    )


# 写入真实single-arm canonical JSON/Markdown与declared identity证据
def _write_arm_output(
    root: Path,
    receipt: object,
    arm: str,
    attempts: list[AttemptAnalysis],
) -> BaselineReport:
    identity = _identity(receipt, arm)
    report = BaselineReport(
        experiment=identity,
        metrics=aggregate_attempts(attempts),
        attempts=attempts,
    )
    write_baseline_report(root, report)
    (root / "declared-experiment.json").write_text(
        json.dumps(
            identity.declared.model_dump(mode="json"),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    for row in attempts:
        attempt_id = f"{row.task_id}-repeat-{row.repeat:02d}"
        evaluation_root = (
            root
            / "run"
            / "tasks"
            / row.task_id
            / f"repeat-{row.repeat:02d}"
            / "evaluation"
        )
        evaluation_report = EvaluationReport(
            task_id=row.task_id,
            attempt_id=attempt_id,
            task_success=row.task_success,
            runtime_success=row.runtime_success,
            trace_sanity_passed=row.trace_sanity_passed,
            failure_category=row.failure_category,
            criteria=[],
            metrics={
                "task_success": row.task_success,
                "runtime_success": row.runtime_success,
                "step_count": row.step_count,
                "tool_count": row.tool_count,
                "retry_count": row.retry_count,
                "wall_latency_ms": row.wall_latency_ms,
                "token_usage": row.token_usage,
                "failure_category": row.failure_category,
            },
        )
        write_report(evaluation_root, evaluation_report)
        attempt_root = (
            evaluation_root / "attempts" / row.task_id / attempt_id
        )
        public = attempt_root / "public"
        runtime = attempt_root / "runtime"
        private = attempt_root / "private"
        for directory in (public, runtime, private):
            directory.mkdir(parents=True, exist_ok=True)
        for path in (public / "outcome.json", public / "metrics.json"):
            path.write_text("{}\n", encoding="utf-8")
        for path in (runtime / "events.v2.jsonl", runtime / "trace.jsonl"):
            path.write_text("{}\n", encoding="utf-8")
        if row.failure_category is not FailureCategory.TIMEOUT:
            for path in (
                runtime / "initial-workspace.json",
                runtime / "final-workspace.json",
                runtime / "workspace.diff",
                private / "grades.json",
                private / "command-results.json",
            ):
                path.write_text("{}\n", encoding="utf-8")
    return report


# 构造与receipt绑定的expected arm identity
def _expected_arm(receipt: object, arm: str) -> object:
    arm_receipt = getattr(receipt.arms, arm)
    return _paired().ExpectedArmIdentity(
        arm=arm,
        commit=arm_receipt.commit,
        profile_id=arm_receipt.profile_id,
        profile_hash=arm_receipt.profile_canonical_sha256,
        prompt_sha256=arm_receipt.prompt_sha256,
        suite_sha256=receipt.shared_identity.suite_sha256,
        tool_schema_sha256=receipt.shared_identity.tool_schema_sha256,
        runtime_config_sha256=receipt.shared_identity.runtime_config_sha256,
        dependency_sha256=receipt.shared_identity.dependency_sha256,
        provider=receipt.shared_identity.provider,
        model=receipt.shared_identity.model,
        protocol=receipt.shared_identity.protocol,
        sdk="anthropic==0.111.0",
        max_steps=receipt.shared_identity.max_steps,
        repeats=receipt.shared_identity.repeats,
        task_ids=[task_id for task_id, _category in _TASKS],
    )


# 构造create-once authorization-use record
def _use_record() -> object:
    return _paired().AuthorizationUseRecord(
        schema_version=1,
        reservation_id="phase9d-use-test",
        status="RESERVED_FOR_ONE_PAIRED_EXECUTION",
        created_at_utc="2026-08-01T00:00:00Z",
        authorization_sha256="1" * 64,
        paired_receipt_sha256="2" * 64,
        output_parent_sha256="3" * 64,
        absolute_path_persisted=False,
        credential_value_persisted=False,
    )


# 构造与runtime source root绑定且不含绝对路径的import evidence
def _source_import_evidence(source_root: Path, marker: str = "a") -> object:
    return _paired().SourceImportEvidence(
        source_root_sha256=hashlib.sha256(
            str(source_root.resolve()).encode("utf-8")
        ).hexdigest(),
        imported_module_path_sha256=marker * 64,
        imported_module_file_sha256=marker * 64,
        module_within_source_root=True,
        absolute_path_persisted=False,
    )


# 构造手工推导的完整成功transition history，避免复用production reducer计算期望
def _successful_transition_history() -> list[object]:
    paired = _paired()
    rows = [
        ("NOT_STARTED", "AUTHORIZATION_RESERVED", "AUTHORIZATION_RESERVED"),
        ("AUTHORIZATION_RESERVED", "CONTROL_STARTED", "CONTROL_RUNNING"),
        ("CONTROL_RUNNING", "CONTROL_VALID", "CONTROL_VALID"),
        ("CONTROL_VALID", "TREATMENT_STARTED", "TREATMENT_RUNNING"),
        ("TREATMENT_RUNNING", "TREATMENT_VALID", "BOTH_VALID"),
        ("BOTH_VALID", "FINALIZED", "TERMINAL"),
    ]
    return [
        paired.PairTransition(from_state=start, event=event, to_state=end)
        for start, event, end in rows
    ]


# 构造不含process id或raw output的成功child terminal evidence
def _successful_child_evidence() -> object:
    return _paired().ChildTerminationEvidence(
        spawned=True,
        exit_code=0,
        signal_number=None,
        cancelled=False,
        failure_category=None,
    )


# 构造不依赖single-arm writer的完整VALID arm audit供artifact语义测试复用
def _valid_arm_audit(arm: str) -> object:
    marker = "a" if arm == "control" else "b"
    return _paired().ArmAudit(
        arm=arm,
        status="VALID",
        reasons=[],
        exit_code=0,
        signal_number=None,
        planned=27,
        started=27,
        completed=27,
        identity_verified=27,
        runtime_failures=0,
        infrastructure_failures=0,
        trace_failures=0,
        grader_failures=0,
        timeouts=0,
        provider_calls=81,
        required_artifact_evidence=True,
        overall_successes=20 if arm == "control" else 21,
        feature_successes=6 if arm == "control" else 7,
        bug_fixing_successes=9,
        inventory_successes=0 if arm == "control" else 1,
        complete_median_latency_ms=100.0,
        complete_median_input_output_tokens=1000.0,
        baseline_json_sha256=marker * 64,
        baseline_markdown_sha256=marker * 64,
    )


# 构造由production classifier与builder生成的最小合法capability result
def _valid_capability_result_for_reference(receipt_reference: object) -> object:
    paired = _paired()
    receipt = _receipt()
    control = _valid_arm_audit("control")
    treatment = _valid_arm_audit("treatment")
    return paired.build_paired_result(
        result_id="phase9d-artifact-semantics",
        created_at_utc="2026-08-01T00:00:00Z",
        receipt_reference=receipt_reference,
        preflight_commit="a" * 40,
        preflight_sha256="2" * 64,
        authorization_commit="b" * 40,
        authorization_sha256="3" * 64,
        authorization_use_sha256="4" * 64,
        receipt=receipt,
        control=control,
        treatment=treatment,
        control_child=_successful_child_evidence(),
        treatment_child=_successful_child_evidence(),
        transitions=_successful_transition_history(),
    )


# 构造由production classifier与builder生成的最小合法capability result
def _valid_capability_result() -> object:
    return _valid_capability_result_for_reference(_receipt_reference())


# 以production canonical serializer重签被mutation的paired result bundle
def _rewrite_paired_bundle(
    root: Path,
    payload: dict[str, object],
    markdown: str,
) -> None:
    paired = _paired()
    json_payload = (paired.canonical_json(payload) + "\n").encode("utf-8")
    markdown_payload = markdown.encode("utf-8")
    (root / "paired-result.json").write_bytes(json_payload)
    (root / "paired-result.md").write_bytes(markdown_payload)
    manifest = {
        "schema_version": 1,
        "json_sha256": hashlib.sha256(json_payload).hexdigest(),
        "markdown_sha256": hashlib.sha256(markdown_payload).hexdigest(),
    }
    (root / "manifest.json").write_text(
        paired.canonical_json(manifest) + "\n",
        encoding="utf-8",
    )


# 用production renderer与canonical serializer重签一个绕过model validation的mutation
def _write_resigned_result_mutation(
    root: Path,
    result: object,
    mutated: object,
) -> None:
    paired = _paired()
    _write_paired_result_bundle(root, result)
    _rewrite_paired_bundle(
        root,
        mutated.model_dump(mode="json"),
        paired.render_paired_markdown(mutated),
    )


# 写入使用authoritative receipt reference绑定的paired result bundle
def _write_paired_result_bundle(root: Path, result: object) -> object:
    return _paired().write_paired_result(
        root,
        result,
        repository=_repository(),
        expected_receipt=_receipt_reference(),
    )


# 读取使用authoritative receipt reference绑定的paired result bundle
def _read_paired_result_bundle(root: Path) -> object:
    return _paired().read_paired_result_bundle(
        root,
        repository=_repository(),
        expected_receipt=_receipt_reference(),
    )


# 读取使用authoritative receipt reference绑定的paired result JSON
def _read_paired_result_json(path: Path) -> object:
    return _paired().read_paired_result_json(
        path,
        repository=_repository(),
        expected_receipt=_receipt_reference(),
    )


# 验证paired terminal XOR时统一使用authoritative receipt reference
def _validate_pair_terminal_exclusivity(
    paths: object,
    *,
    authorization_consumed: bool,
    execution_complete: bool,
) -> object:
    return _paired().validate_pair_terminal_exclusivity(
        paths,
        authorization_consumed=authorization_consumed,
        execution_complete=execution_complete,
        repository=_repository(),
        expected_receipt=_receipt_reference(),
    )


# 按手工推导的reducer路径构造指定failure event的合法terminal history
def _failure_transition_history(event: str) -> list[object]:
    rows = [("NOT_STARTED", "AUTHORIZATION_RESERVED", "AUTHORIZATION_RESERVED")]
    if event in {"PARENT_INTERRUPTED", "PARENT_SYSTEM_EXIT", "PRIVATE_EVIDENCE_FAILED"}:
        rows.append(("AUTHORIZATION_RESERVED", event, "TERMINAL"))
    else:
        rows.append(("AUTHORIZATION_RESERVED", "CONTROL_STARTED", "CONTROL_RUNNING"))
        if event in {"CONTROL_SPAWN_FAILED", "CONTROL_INVALID"}:
            rows.append(("CONTROL_RUNNING", event, "TERMINAL"))
        else:
            rows.append(("CONTROL_RUNNING", "CONTROL_VALID", "CONTROL_VALID"))
            if event == "BETWEEN_ARM_INVALID":
                rows.append(("CONTROL_VALID", event, "TERMINAL"))
            else:
                rows.append(("CONTROL_VALID", "TREATMENT_STARTED", "TREATMENT_RUNNING"))
                if event in {"TREATMENT_SPAWN_FAILED", "TREATMENT_INVALID"}:
                    rows.append(("TREATMENT_RUNNING", event, "TERMINAL"))
                else:
                    rows.append(("TREATMENT_RUNNING", "TREATMENT_VALID", "BOTH_VALID"))
                    rows.append(("BOTH_VALID", event, "TERMINAL"))
    return [
        _paired().PairTransition(from_state=start, event=transition, to_state=end)
        for start, transition, end in rows
    ]


# 为各failure phase提供与执行进度兼容的arm audit presence
def _terminal_audits_for_phase(phase: str) -> tuple[object | None, object | None]:
    control = _valid_arm_audit("control")
    treatment = _valid_arm_audit("treatment")
    if phase == "control_spawn" or phase == "parent_interrupt":
        return None, None
    if phase == "control_audit":
        return control.model_copy(update={"status": "INVALID", "reasons": ["invalid"]}), None
    if phase in {"between_arms", "treatment_spawn"}:
        return control, None
    if phase == "treatment_audit":
        return control, treatment.model_copy(
            update={"status": "INVALID", "reasons": ["invalid"]}
        )
    return control, treatment


# 返回不读取真实credential或用户environment的最小child环境
def _isolated_child_env() -> dict[str, str]:
    return {"PATH": os.defpath, "PYTHONUNBUFFERED": "1"}


# 功能：验证exit 1加完整canonical baseline仍通过arm validity
# 设计：真实构造27-row BaselineReport，防止observer把能力失败误当identity invalid
def test_audit_arm_accepts_exit_one_with_complete_valid_baseline(tmp_path: Path) -> None:
    paired = _paired()
    receipt = _receipt()
    successes = {task_id for task_id, _category in _TASKS}
    output = tmp_path / "control"
    _write_arm_output(output, receipt, "control", _attempts(successes))

    audit = paired.audit_arm_result(
        expected=_expected_arm(receipt, "control"),
        exit_code=1,
        signal_number=None,
        output_root=output,
        receipt=receipt,
    )

    assert audit.status == "VALID"
    assert audit.planned == audit.started == audit.completed == 27
    assert audit.identity_verified == 27


# 功能：验证Git receipt observer从冻结commit/path/blob生成批准receipt reference
# 设计：读取真实tracked receipt的Git对象而非当前文件替代，断言commit/path/bytes/SHA四元组完整匹配
def test_observe_receipt_reference_accepts_frozen_receipt_git_identity() -> None:
    reference = _receipt_reference()

    assert reference.commit == "5af1ec2e1d235ab110314afe98b92e6702093657"
    assert reference.path == (
        "benchmarks/receipts/phase9d-repaired-v1-v2-paired-experiment.json"
    )
    assert reference.bytes == 9710
    assert (
        reference.sha256
        == "58aaf8309d9e8eee1f64dc469453407e8c43c2eef884e7dfe00ced91ffd35958"
    )
    assert reference.authorization_remains_false_zero is True


# 功能：验证exit 0不能掩盖缺失、重复或不完整attempt matrix
# 设计：从真实report删除最后row并重算metrics，确保observer独立于producer builder复审
def test_audit_arm_rejects_exit_zero_with_incomplete_baseline(tmp_path: Path) -> None:
    paired = _paired()
    receipt = _receipt()
    successes = {task_id for task_id, _category in _TASKS}
    output = tmp_path / "control"
    _write_arm_output(output, receipt, "control", _attempts(successes)[:-1])

    audit = paired.audit_arm_result(
        expected=_expected_arm(receipt, "control"),
        exit_code=0,
        signal_number=None,
        output_root=output,
        receipt=receipt,
    )

    assert audit.status == "INVALID"
    assert "attempt_schedule" in audit.reasons


# 功能：验证exit 2、signal和缺baseline均确定性INVALID
# 设计：参数化child终态并保留空output root，覆盖exit-code不能替代artifact evidence
@pytest.mark.parametrize(
    ("exit_code", "signal_number"),
    [(2, None), (None, signal.SIGTERM), (0, None)],
)
def test_audit_arm_rejects_invalid_child_or_missing_evidence(
    tmp_path: Path,
    exit_code: int | None,
    signal_number: int | None,
) -> None:
    receipt = _receipt()
    output = tmp_path / f"missing-{exit_code}-{signal_number}"
    output.mkdir()

    audit = _paired().audit_arm_result(
        expected=_expected_arm(receipt, "control"),
        exit_code=exit_code,
        signal_number=signal_number,
        output_root=output,
        receipt=receipt,
    )

    assert audit.status == "INVALID"
    assert audit.required_artifact_evidence is False


# 功能：验证timeout能力失败在证据完整时不自动使arm INVALID
# 设计：仅把一个attempt设为timeout，保留27-row identity和artifact完整性
def test_audit_arm_keeps_verified_timeout_as_valid_experiment(tmp_path: Path) -> None:
    receipt = _receipt()
    successes = {task_id for task_id, _category in _TASKS}
    output = tmp_path / "control"
    _write_arm_output(
        output,
        receipt,
        "control",
        _attempts(successes, timeout_keys={("testgen-dependency-planner", 3)}),
    )

    audit = _paired().audit_arm_result(
        expected=_expected_arm(receipt, "control"),
        exit_code=1,
        signal_number=None,
        output_root=output,
        receipt=receipt,
    )

    assert audit.status == "VALID"
    assert audit.timeouts == 1
    assert audit.runtime_failures == 0


# 功能：验证baseline Markdown存在时必须由同一JSON model精确渲染
# 设计：单点追加错误verdict文本，确保observer不接受分叉report逻辑
def test_audit_arm_rejects_markdown_json_divergence(tmp_path: Path) -> None:
    receipt = _receipt()
    successes = {task_id for task_id, _category in _TASKS}
    output = tmp_path / "control"
    _write_arm_output(output, receipt, "control", _attempts(successes))
    (output / "baseline.md").write_text("diverged\n", encoding="utf-8")

    audit = _paired().audit_arm_result(
        expected=_expected_arm(receipt, "control"),
        exit_code=0,
        signal_number=None,
        output_root=output,
        receipt=receipt,
    )

    assert audit.status == "INVALID"
    assert "baseline_markdown" in audit.reasons


# 功能：验证arm audit机械重算baseline metrics而不信任可篡改聚合值
# 设计：保持27个attempt和全部identity不变，仅改overall success计数，要求observer判INVALID
def test_audit_arm_rejects_aggregate_metrics_drift(tmp_path: Path) -> None:
    paired = _paired()
    receipt = _receipt()
    root = tmp_path / "control"
    attempts = _attempts({task_id for task_id, _category in _TASKS})
    report = _write_arm_output(root, receipt, "control", attempts)
    overall = report.metrics.overall.model_copy(update={"successful_attempts": 0})
    metrics = report.metrics.model_copy(update={"overall": overall})
    write_baseline_report(root, report.model_copy(update={"metrics": metrics}))

    audit = paired.audit_arm_result(
        expected=_expected_arm(receipt, "control"),
        exit_code=0,
        signal_number=None,
        output_root=root,
        receipt=receipt,
    )

    assert audit.status == "INVALID"
    assert "baseline_metrics" in audit.reasons


# 功能：验证任一canonical attempt runtime evidence缺失都会使arm fail closed
# 设计：删除单个trace文件但保留baseline/identity，隔离required artifact evidence门禁
def test_audit_arm_rejects_missing_attempt_artifact(tmp_path: Path) -> None:
    paired = _paired()
    receipt = _receipt()
    root = tmp_path / "control"
    attempts = _attempts({task_id for task_id, _category in _TASKS})
    _write_arm_output(root, receipt, "control", attempts)
    missing = next((root / "run").rglob("trace.jsonl"))
    missing.unlink()

    audit = paired.audit_arm_result(
        expected=_expected_arm(receipt, "control"),
        exit_code=0,
        signal_number=None,
        output_root=root,
        receipt=receipt,
    )

    assert audit.status == "INVALID"
    assert "required_artifact_evidence" in audit.reasons


# 功能：验证arm audit是纯只读observer且不会修复或重写single-arm artifacts
# 设计：审计前后对完整artifact tree逐文件hash，精确证明bytes集合保持不变
def test_audit_arm_never_modifies_single_arm_artifacts(tmp_path: Path) -> None:
    paired = _paired()
    receipt = _receipt()
    root = tmp_path / "control"
    _write_arm_output(
        root,
        receipt,
        "control",
        _attempts({task_id for task_id, _category in _TASKS}),
    )

    # 对只读审计边界生成相对路径到bytes hash的稳定映射
    def snapshot() -> dict[str, str]:
        return {
            path.relative_to(root).as_posix(): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in sorted(root.rglob("*"))
            if path.is_file()
        }

    before = snapshot()
    audit = paired.audit_arm_result(
        expected=_expected_arm(receipt, "control"),
        exit_code=0,
        signal_number=None,
        output_root=root,
        receipt=receipt,
    )

    assert audit.status == "VALID"
    assert snapshot() == before


# 功能：验证profile/prompt/tool/runtime/suite/dependency任一identity漂移均INVALID
# 设计：参数化expected identity单字段mutation，复用同一真实baseline隔离observer比较节点
@pytest.mark.parametrize(
    "field",
    [
        "commit",
        "profile_hash",
        "prompt_sha256",
        "suite_sha256",
        "tool_schema_sha256",
        "runtime_config_sha256",
        "dependency_sha256",
        "model",
    ],
)
def test_audit_arm_rejects_identity_drift(tmp_path: Path, field: str) -> None:
    receipt = _receipt()
    successes = {task_id for task_id, _category in _TASKS}
    output = tmp_path / "control"
    _write_arm_output(output, receipt, "control", _attempts(successes))
    expected = _expected_arm(receipt, "control")
    replacement = "f" * (40 if field == "commit" else 64)
    if field == "model":
        replacement = "wrong-model"
    expected = expected.model_copy(update={field: replacement})

    audit = _paired().audit_arm_result(
        expected=expected,
        exit_code=0,
        signal_number=None,
        output_root=output,
        receipt=receipt,
    )

    assert audit.status == "INVALID"
    assert "identity" in audit.reasons


# 功能：验证authorization-use首次成功、第二次冲突且record保持canonical私有模式
# 设计：真实O_EXCL文件系统边界证明不可覆盖，不用mock os.open
def test_authorization_use_reservation_is_create_once(tmp_path: Path) -> None:
    paired = _paired()
    path = tmp_path / "pair" / "authorization-use.json"
    path.parent.mkdir()
    record = _use_record()

    paired.reserve_authorization_use(path, record)

    assert paired.read_strict_artifact(path, paired.AuthorizationUseRecord) == record
    assert path.stat().st_mode & 0o777 == 0o600
    with pytest.raises(ValueError, match="already reserved"):
        paired.reserve_authorization_use(path, record)


# 功能：验证authorization-use helper只在调用方已验证并创建的private parent内写入
# 设计：传不存在parent，要求fail closed且不得由底层helper暗中扩大已验证目录边界
def test_authorization_use_reservation_requires_existing_parent(tmp_path: Path) -> None:
    path = tmp_path / "missing-private" / "authorization-use.json"

    with pytest.raises(ValueError, match="parent"):
        _paired().reserve_authorization_use(path, _use_record())

    assert not path.parent.exists()


# 功能：验证并发reservation竞争只有一个成功且失败方不能覆盖
# 设计：两个线程同时调用真实O_EXCL，结果必须恰为一次success一次conflict
def test_authorization_use_reservation_has_single_concurrent_winner(
    tmp_path: Path,
) -> None:
    paired = _paired()
    path = tmp_path / "authorization-use.json"
    barrier = threading.Barrier(2)

    # 在线程中等待同一barrier后尝试真实reservation
    def reserve() -> str:
        barrier.wait()
        try:
            paired.reserve_authorization_use(path, _use_record())
        except ValueError:
            return "conflict"
        return "success"

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(reserve) for _ in range(2)]
        results = sorted(future.result() for future in futures)

    assert results == ["conflict", "success"]


# 功能：验证symlink/非普通文件冲突和权限错误均拒绝reservation
# 设计：分别创建悬空symlink与monkeypatch os.open权限异常，覆盖不安全目标形态
def test_authorization_use_reservation_rejects_conflicts_and_permission_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paired = _paired()
    symlink = tmp_path / "authorization-use.json"
    symlink.symlink_to(tmp_path / "missing")
    with pytest.raises(ValueError, match="already reserved"):
        paired.reserve_authorization_use(symlink, _use_record())

    target = tmp_path / "permission" / "authorization-use.json"
    target.parent.mkdir()
    original_open = os.open

    # 仅对目标reservation注入权限错误，保留其他文件系统行为
    def deny(path: object, flags: int, mode: int = 0o777) -> int:
        candidate = Path(path)
        if candidate.parent == target.parent and candidate.name.startswith(
            f".{target.name}."
        ):
            raise PermissionError("denied")
        return original_open(path, flags, mode)

    monkeypatch.setattr(os, "open", deny)
    with pytest.raises(ValueError, match="cannot be created"):
        paired.reserve_authorization_use(target, _use_record())


# 功能：验证authorization临时写成功但publish失败时目标不存在且授权尚未消费
# 设计：只让atomic link失败，要求typed error明确consumed=false并清理temporary文件
def test_authorization_use_publish_failure_is_unconsumed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paired = _paired()
    target = tmp_path / "authorization-use.json"
    monkeypatch.setattr(
        os,
        "link",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(PermissionError("denied")),
    )

    with pytest.raises(paired.AuthorizationReservationError) as exc_info:
        paired.reserve_authorization_use(target, _use_record())

    assert exc_info.value.consumed is False
    assert not os.path.lexists(target)
    assert list(tmp_path.iterdir()) == []


# 功能：验证authorization publish后parent fsync失败仍永久计为已消费且record完整
# 设计：只破坏directory durability acknowledgement，检查typed error consumed=true和canonical target
def test_authorization_use_parent_fsync_failure_remains_consumed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paired = _paired()
    target = tmp_path / "authorization-use.json"

    # 在link成功后注入目录fsync失败，模拟无法确认目录项durability
    def fail_fsync(_path: Path) -> None:
        raise ValueError("fsync failed")

    monkeypatch.setattr(paired, "_fsync_directory", fail_fsync)
    with pytest.raises(paired.AuthorizationReservationError) as exc_info:
        paired.reserve_authorization_use(target, _use_record())

    assert exc_info.value.consumed is True
    assert paired.read_strict_artifact(
        target,
        paired.AuthorizationUseRecord,
    ) == _use_record()


# 功能：验证authorization临时文件发生短写时不得发布不完整use record
# 设计：用保留真实fd/fsync边界的短写stream返回len-1，要求typed error保持unconsumed并清理临时文件
def test_authorization_use_partial_write_is_unconsumed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paired = _paired()
    target = tmp_path / "authorization-use.json"

    class PartialWriteStream:
        # 保存真实descriptor供write、fsync与close使用
        def __init__(self, descriptor: int) -> None:
            self.descriptor = descriptor

        # 返回当前stream供with语句使用
        def __enter__(self) -> PartialWriteStream:
            return self

        # 关闭真实descriptor且不抑制异常
        def __exit__(self, *_args: object) -> None:
            os.close(self.descriptor)

        # 只写入payload前缀并显式报告短写
        def write(self, data: bytes) -> int:
            return os.write(self.descriptor, data[:-1])

        # 保持与真实二进制stream相同的flush调用面
        def flush(self) -> None:
            return None

        # 返回真实descriptor供fsync使用
        def fileno(self) -> int:
            return self.descriptor

    # 将fd包装为可观测短写stream而不改变open/link语义
    def partial_fdopen(
        descriptor: int,
        _mode: str,
        *,
        closefd: bool,
    ) -> PartialWriteStream:
        assert closefd is True
        return PartialWriteStream(descriptor)

    monkeypatch.setattr(os, "fdopen", partial_fdopen)

    with pytest.raises(paired.AuthorizationReservationError) as exc_info:
        paired.reserve_authorization_use(target, _use_record())

    assert exc_info.value.consumed is False
    assert not os.path.lexists(target)
    assert list(tmp_path.iterdir()) == []


# 功能：验证use record前失败保持NOT_STARTED而reservation后spawn失败变INVALID
# 设计：同一state transition函数只改变reserved边界，锁定授权不可复用语义
def test_state_machine_distinguishes_pre_reservation_and_post_reservation_failure() -> None:
    paired = _paired()
    receipt = _receipt()

    before = paired.transition_pair_state(
        paired.PairState.NOT_STARTED,
        paired.PairEvent.PREFLIGHT_FAILED,
        receipt=receipt,
        authorization_use_reserved=False,
    )
    reserved = paired.transition_pair_state(
        paired.PairState.NOT_STARTED,
        paired.PairEvent.AUTHORIZATION_RESERVED,
        receipt=receipt,
        authorization_use_reserved=True,
    )
    running = paired.transition_pair_state(
        reserved,
        paired.PairEvent.CONTROL_STARTED,
        receipt=receipt,
        authorization_use_reserved=True,
    )
    after = paired.transition_pair_state(
        running,
        paired.PairEvent.CONTROL_SPAWN_FAILED,
        receipt=receipt,
        authorization_use_reserved=True,
    )

    assert before is paired.PairState.NOT_STARTED
    assert after is paired.PairState.TERMINAL


# 功能：验证between-arm invalid只能由CONTROL_VALID经专用event进入TERMINAL
# 设计：先走完整reservation/control-valid reducer路径，再断言非法跳转和合法终结语义
def test_state_machine_owns_between_arm_invalid_transition() -> None:
    paired = _paired()
    receipt = _receipt()
    state = paired.PairState.NOT_STARTED
    for event in (
        paired.PairEvent.AUTHORIZATION_RESERVED,
        paired.PairEvent.CONTROL_STARTED,
        paired.PairEvent.CONTROL_VALID,
    ):
        state = paired.transition_pair_state(
            state,
            event,
            receipt=receipt,
            authorization_use_reserved=True,
        )

    assert paired.transition_pair_state(
        state,
        paired.PairEvent.BETWEEN_ARM_INVALID,
        receipt=receipt,
        authorization_use_reserved=True,
    ) is paired.PairState.TERMINAL
    with pytest.raises(ValueError, match="invalid pair transition"):
        paired.transition_pair_state(
            paired.PairState.NOT_STARTED,
            paired.PairEvent.BETWEEN_ARM_INVALID,
            receipt=receipt,
            authorization_use_reserved=True,
        )


# 功能：验证C1 VALID无论能力分数或exit 1都进入treatment路径
# 设计：state machine只接收validity事件，不接收score，防止result-aware stop
def test_state_machine_progresses_from_valid_control_without_score_input() -> None:
    paired = _paired()
    receipt = _receipt()

    reserved = paired.transition_pair_state(
        paired.PairState.NOT_STARTED,
        paired.PairEvent.AUTHORIZATION_RESERVED,
        receipt=receipt,
        authorization_use_reserved=True,
    )
    running = paired.transition_pair_state(
        reserved,
        paired.PairEvent.CONTROL_STARTED,
        receipt=receipt,
        authorization_use_reserved=True,
    )
    valid = paired.transition_pair_state(
        running,
        paired.PairEvent.CONTROL_VALID,
        receipt=receipt,
        authorization_use_reserved=True,
    )
    treatment = paired.transition_pair_state(
        valid,
        paired.PairEvent.TREATMENT_STARTED,
        receipt=receipt,
        authorization_use_reserved=True,
    )

    assert treatment is paired.PairState.TREATMENT_RUNNING


# 功能：验证orchestrator transition实际读取receipt execution_state_machine而非独立硬编码
# 设计：仅在内存中把control-valid的run_treatment改为false，合法状态对必须被receipt gate拒绝
def test_state_machine_progression_is_receipt_driven() -> None:
    paired = _paired()
    receipt = _receipt()
    blocked_control = receipt.execution_state_machine.control_valid_and_complete.model_copy(
        update={"run_treatment": False}
    )
    blocked_machine = receipt.execution_state_machine.model_copy(
        update={"control_valid_and_complete": blocked_control}
    )
    blocked_receipt = receipt.model_copy(
        update={"execution_state_machine": blocked_machine}
    )

    with pytest.raises(ValueError, match="receipt forbids treatment progression"):
        paired.transition_pair_state(
            paired.PairState.CONTROL_VALID,
            paired.PairEvent.TREATMENT_STARTED,
            receipt=blocked_receipt,
            authorization_use_reserved=True,
        )


# 功能：验证C1 INVALID后任何treatment start都属于impossible transition
# 设计：先合法进入CONTROL_INVALID再尝试C2，确保状态机而非caller约定阻断
def test_state_machine_rejects_treatment_after_invalid_control() -> None:
    paired = _paired()
    receipt = _receipt()
    invalid = paired.transition_pair_state(
        paired.PairState.CONTROL_RUNNING,
        paired.PairEvent.CONTROL_INVALID,
        receipt=receipt,
        authorization_use_reserved=True,
    )

    with pytest.raises(ValueError, match="invalid pair transition"):
        paired.transition_pair_state(
            invalid,
            paired.PairEvent.TREATMENT_STARTED,
            receipt=receipt,
            authorization_use_reserved=True,
        )


# 功能：验证state/event笛卡尔积中除冻结合同外的每个跳转都被canonical reducer拒绝
# 设计：穷举全部enum组合并与独立手写合法集合比较，任何新增旁路或dead-state跳转都会被测试杀死
def test_state_machine_rejects_every_unregistered_transition() -> None:
    paired = _paired()
    receipt = _receipt()
    allowed = {
        (paired.PairState.NOT_STARTED, paired.PairEvent.PREFLIGHT_FAILED): (
            paired.PairState.NOT_STARTED
        ),
        (paired.PairState.NOT_STARTED, paired.PairEvent.AUTHORIZATION_RESERVED): (
            paired.PairState.AUTHORIZATION_RESERVED
        ),
        (
            paired.PairState.AUTHORIZATION_RESERVED,
            paired.PairEvent.PRIVATE_EVIDENCE_FAILED,
        ): paired.PairState.TERMINAL,
        (
            paired.PairState.AUTHORIZATION_RESERVED,
            paired.PairEvent.CONTROL_STARTED,
        ): paired.PairState.CONTROL_RUNNING,
        (
            paired.PairState.CONTROL_RUNNING,
            paired.PairEvent.CONTROL_SPAWN_FAILED,
        ): paired.PairState.TERMINAL,
        (paired.PairState.CONTROL_RUNNING, paired.PairEvent.CONTROL_VALID): (
            paired.PairState.CONTROL_VALID
        ),
        (paired.PairState.CONTROL_RUNNING, paired.PairEvent.CONTROL_INVALID): (
            paired.PairState.TERMINAL
        ),
        (paired.PairState.CONTROL_VALID, paired.PairEvent.BETWEEN_ARM_INVALID): (
            paired.PairState.TERMINAL
        ),
        (paired.PairState.CONTROL_VALID, paired.PairEvent.TREATMENT_STARTED): (
            paired.PairState.TREATMENT_RUNNING
        ),
        (
            paired.PairState.TREATMENT_RUNNING,
            paired.PairEvent.TREATMENT_SPAWN_FAILED,
        ): paired.PairState.TERMINAL,
        (
            paired.PairState.TREATMENT_RUNNING,
            paired.PairEvent.TREATMENT_VALID,
        ): paired.PairState.BOTH_VALID,
        (
            paired.PairState.TREATMENT_RUNNING,
            paired.PairEvent.TREATMENT_INVALID,
        ): paired.PairState.TERMINAL,
        (paired.PairState.BOTH_VALID, paired.PairEvent.CLASSIFICATION_FAILED): (
            paired.PairState.TERMINAL
        ),
        (paired.PairState.BOTH_VALID, paired.PairEvent.RESULT_WRITE_FAILED): (
            paired.PairState.TERMINAL
        ),
        (paired.PairState.BOTH_VALID, paired.PairEvent.FINALIZED): (
            paired.PairState.TERMINAL
        ),
    }
    for state in paired.PairState:
        if state in {paired.PairState.NOT_STARTED, paired.PairState.TERMINAL}:
            continue
        allowed[(state, paired.PairEvent.PARENT_INTERRUPTED)] = (
            paired.PairState.TERMINAL
        )
        allowed[(state, paired.PairEvent.PARENT_SYSTEM_EXIT)] = (
            paired.PairState.TERMINAL
        )

    for state in paired.PairState:
        for event in paired.PairEvent:
            expected = allowed.get((state, event))
            reserved = event is not paired.PairEvent.PREFLIGHT_FAILED
            if expected is None:
                with pytest.raises(ValueError, match="invalid pair transition"):
                    paired.transition_pair_state(
                        state,
                        event,
                        receipt=receipt,
                        authorization_use_reserved=reserved,
                    )
            else:
                assert paired.transition_pair_state(
                    state,
                    event,
                    receipt=receipt,
                    authorization_use_reserved=reserved,
                ) is expected


# 构造完整between-arm revalidation snapshot
def _between_arm() -> object:
    return _paired().BetweenArmEvidence(
        receipt_commit="5" * 40,
        receipt_sha256="1" * 64,
        preflight_commit="a" * 40,
        preflight_sha256="2" * 64,
        authorization_commit="b" * 40,
        authorization_sha256="3" * 64,
        git_artifact_identity_sha256="4" * 64,
        control_commit_exists=True,
        treatment_commit_exists=True,
        main_ref_sha256="4" * 64,
        treatment_worktree_sha256="5" * 64,
        treatment_profile_sha256="6" * 64,
        treatment_prompt_sha256="7" * 64,
        source_binding_sha256="8" * 64,
        treatment_source_import=_paired().SourceImportEvidence(
            source_root_sha256="8" * 64,
            imported_module_path_sha256="7" * 64,
            imported_module_file_sha256="6" * 64,
            module_within_source_root=True,
            absolute_path_persisted=False,
        ),
        environment_sha256="9" * 64,
        pyproject_sha256="a" * 64,
        uv_lock_sha256="b" * 64,
        dependency_sha256="c" * 64,
        suite_sha256="d" * 64,
        task_bundle_sha256="e" * 64,
        grader_bundle_sha256="f" * 64,
        tool_schema_sha256="0" * 64,
        runtime_config_sha256="1" * 64,
        output_parent_path_sha256="2" * 64,
        output_parent_object_sha256="3" * 64,
        treatment_root_absent=True,
        credential_present=True,
        authorization_use_sha256="4" * 64,
        experiment_unchanged=True,
    )


# 功能：验证between-arm任一identity/gate漂移都阻断treatment
# 设计：对每个字段做单点mutation并调用同一revalidator，覆盖无现场修复边界
@pytest.mark.parametrize(
    "field",
    list(_paired().BetweenArmEvidence.model_fields),
)
def test_revalidate_before_treatment_rejects_every_gate_drift(field: str) -> None:
    paired = _paired()
    expected = _between_arm()
    current = getattr(expected, field)
    replacement = (
        False
        if current is True
        else ("8" * 64 if current == "9" * 64 else "9" * 64)
    )
    observed = expected.model_copy(update={field: replacement})

    with pytest.raises(ValueError, match="between-arm identity drift"):
        paired.revalidate_before_treatment(expected, observed)


# 功能：验证非默认但包含全部唯一verdict的receipt order通过语义校验
# 设计：只改真实receipt的order并经production loader读取，直接暴露source literal重复冻结
def test_receipt_accepts_alternate_complete_classification_order(
    tmp_path: Path,
) -> None:
    paired = _paired()
    payload = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "benchmarks"
            / "receipts"
            / "phase9d-repaired-v1-v2-paired-experiment.json"
        ).read_text(encoding="utf-8")
    )
    alternate = ["ACCEPT", "INVALID", "MIXED", "REJECT"]
    payload["decision_contract"]["classification_order"] = alternate
    path = tmp_path / "alternate-receipt.json"
    path.write_text(paired.canonical_json(payload) + "\n", encoding="utf-8")

    observed = paired.load_paired_receipt(path)

    assert observed.decision_contract.classification_order == alternate


# 功能：验证production precedence selector按receipt实际顺序选择多个true候选中的首项
# 设计：受控predicate map让ACCEPT与INVALID同时为true，避免classifier天然互斥掩盖order读取
def test_classification_precedence_uses_receipt_order() -> None:
    paired = _paired()
    order = ["ACCEPT", "INVALID", "MIXED", "REJECT"]
    matches = {
        "INVALID": True,
        "REJECT": False,
        "ACCEPT": True,
        "MIXED": False,
    }

    assert paired.select_classification_verdict(order, matches) == "ACCEPT"


# 功能：验证缺失、重复、未知、错误长度和非字符串classification order全部fail closed
# 设计：逐项重签真实receipt并调用production loader，避免测试复制structural validator
@pytest.mark.parametrize(
    "order",
    [
        ["INVALID", "REJECT", "ACCEPT"],
        ["INVALID", "REJECT", "ACCEPT", "ACCEPT"],
        ["INVALID", "REJECT", "ACCEPT", "UNKNOWN"],
        [],
        ["INVALID", "REJECT", "ACCEPT", "MIXED", "INVALID"],
        ["INVALID", "REJECT", "ACCEPT", 1],
    ],
)
def test_receipt_rejects_structurally_invalid_classification_order(
    tmp_path: Path,
    order: list[object],
) -> None:
    paired = _paired()
    payload = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "benchmarks"
            / "receipts"
            / "phase9d-repaired-v1-v2-paired-experiment.json"
        ).read_text(encoding="utf-8")
    )
    payload["decision_contract"]["classification_order"] = order
    path = tmp_path / "invalid-order-receipt.json"
    path.write_text(paired.canonical_json(payload) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="invalid paired receipt"):
        paired.load_paired_receipt(path)


# 功能：验证receipt-driven classifier在当前阈值下产生唯一ACCEPT并响应阈值mutation
# 设计：control 20成功/treatment 21成功且inventory delta1，单改receipt阈值后必须变REJECT
def test_classifier_reads_thresholds_from_receipt_instead_of_constants(
    tmp_path: Path,
) -> None:
    paired = _paired()
    receipt = _receipt()
    all_tasks = {task_id for task_id, _category in _TASKS}
    control_tasks = all_tasks - {"feature-inventory-reservation-lifecycle"}
    control_root = tmp_path / "control"
    treatment_root = tmp_path / "treatment"
    _write_arm_output(control_root, receipt, "control", _attempts(control_tasks))
    _write_arm_output(treatment_root, receipt, "treatment", _attempts(all_tasks))
    control = paired.audit_arm_result(
        expected=_expected_arm(receipt, "control"),
        exit_code=1,
        signal_number=None,
        output_root=control_root,
        receipt=receipt,
    )
    treatment = paired.audit_arm_result(
        expected=_expected_arm(receipt, "treatment"),
        exit_code=0,
        signal_number=None,
        output_root=treatment_root,
        receipt=receipt,
    )
    outcome = paired.derive_pair_outcome(control, treatment)

    assert paired.classify_pair(receipt, outcome).verdict == "ACCEPT"
    mutated_inventory = receipt.primary_comparison.inventory_lifecycle.model_copy(
        update={"treatment_minimum_successes": 4}
    )
    mutated_primary = receipt.primary_comparison.model_copy(
        update={"inventory_lifecycle": mutated_inventory}
    )
    mutated_receipt = receipt.model_copy(update={"primary_comparison": mutated_primary})
    assert paired.classify_pair(mutated_receipt, outcome).verdict == "REJECT"


# 功能：验证效率超阈值但无reject条件时分类唯一MIXED
# 设计：两臂能力满足primary，只把treatment latency/token提高到receipt上限之外
def test_classifier_returns_mixed_for_efficiency_only_failure(tmp_path: Path) -> None:
    paired = _paired()
    receipt = _receipt()
    all_tasks = {task_id for task_id, _category in _TASKS}
    control_tasks = all_tasks - {"feature-inventory-reservation-lifecycle"}
    control_root = tmp_path / "control"
    treatment_root = tmp_path / "treatment"
    _write_arm_output(
        control_root,
        receipt,
        "control",
        _attempts(control_tasks, latency=100.0, tokens=1000),
    )
    _write_arm_output(
        treatment_root,
        receipt,
        "treatment",
        _attempts(all_tasks, latency=120.0, tokens=1200),
    )
    control = paired.audit_arm_result(
        expected=_expected_arm(receipt, "control"),
        exit_code=1,
        signal_number=None,
        output_root=control_root,
        receipt=receipt,
    )
    treatment = paired.audit_arm_result(
        expected=_expected_arm(receipt, "treatment"),
        exit_code=0,
        signal_number=None,
        output_root=treatment_root,
        receipt=receipt,
    )

    evidence = paired.classify_pair(
        receipt,
        paired.derive_pair_outcome(control, treatment),
    )

    assert evidence.verdict == "MIXED"
    assert sum(evidence.matches.values()) == 1


# 功能：验证PairedResult JSON与Markdown来自同一模型且不包含raw payload
# 设计：构造最小result后精确回读JSON，并扫描Markdown/JSON claim与canary边界
def test_paired_result_json_markdown_are_same_model_and_redacted(tmp_path: Path) -> None:
    paired = _paired()
    receipt = _receipt()
    all_tasks = {task_id for task_id, _category in _TASKS}
    control_tasks = all_tasks - {"feature-inventory-reservation-lifecycle"}
    control_root = tmp_path / "control"
    treatment_root = tmp_path / "treatment"
    _write_arm_output(control_root, receipt, "control", _attempts(control_tasks))
    _write_arm_output(treatment_root, receipt, "treatment", _attempts(all_tasks))
    control = paired.audit_arm_result(
        expected=_expected_arm(receipt, "control"),
        exit_code=1,
        signal_number=None,
        output_root=control_root,
        receipt=receipt,
    )
    treatment = paired.audit_arm_result(
        expected=_expected_arm(receipt, "treatment"),
        exit_code=0,
        signal_number=None,
        output_root=treatment_root,
        receipt=receipt,
    )
    result = paired.build_paired_result(
        result_id="phase9d-pair-test",
        created_at_utc="2026-08-01T00:00:00Z",
        receipt_reference=_receipt_reference(),
        preflight_commit="a" * 40,
        preflight_sha256="2" * 64,
        authorization_commit="b" * 40,
        authorization_sha256="3" * 64,
        authorization_use_sha256="4" * 64,
        receipt=receipt,
        control=control,
        treatment=treatment,
        control_child=_successful_child_evidence(),
        treatment_child=_successful_child_evidence(),
        transitions=_successful_transition_history(),
    )
    output = tmp_path / "pair-result"

    _write_paired_result_bundle(output, result)

    json_text = (output / "paired-result.json").read_text(encoding="utf-8")
    markdown = (output / "paired-result.md").read_text(encoding="utf-8")
    assert json.loads(json_text) == result.model_dump(mode="json")
    assert f"Verdict: `{result.verdict}`" in markdown
    assert result.provider_call_count == 162
    for forbidden in ("system_prompt", "messages", "tool_schemas", "SECRET_CANARY"):
        assert forbidden not in json_text
        assert forbidden not in markdown
    assert "not SWE-bench" in markdown
    assert "not statistically significant" in markdown
    with pytest.raises(ValidationError):
        paired.PairedResult.model_validate(
            {
                **result.model_dump(mode="json"),
                "child_stdout": "CHILD_CREDENTIAL_SENTINEL",
            }
        )


# 功能：验证capability result拒绝由PARENT_INTERRUPTED进入TERMINAL的合法reducer history
# 设计：只替换最终canonical triple，保持history连续和终态合法以隔离artifact-kind语义缺口
def test_paired_result_rejects_failure_terminal_event() -> None:
    result = _valid_capability_result()
    payload = result.model_dump(mode="json")
    payload["transitions"][-1] = {
        "from_state": "BOTH_VALID",
        "event": "PARENT_INTERRUPTED",
        "to_state": "TERMINAL",
    }

    with pytest.raises(ValidationError, match="terminal evidence"):
        _paired().PairedResult.model_validate(payload)


# 功能：验证capability result不能使用仅属于failure terminal的INVALID verdict
# 设计：同步修改classifier与顶层verdict保持旧一致性校验自洽，隔离artifact-kind verdict合同
def test_paired_result_rejects_invalid_verdict() -> None:
    result = _valid_capability_result()
    payload = result.model_dump(mode="json")
    payload["verdict"] = "INVALID"
    payload["classifier"]["verdict"] = "INVALID"
    payload["classifier"]["matches"] = {
        "INVALID": True,
        "REJECT": False,
        "ACCEPT": False,
        "MIXED": False,
    }

    with pytest.raises(ValidationError):
        _paired().PairedResult.model_validate(payload)


# 功能：验证capability result拒绝INVALID或计数不完整的arm audit
# 设计：分别单改control status与completed count，保持final success triple以隔离双臂完整性合同
@pytest.mark.parametrize(
    "control_update",
    [
        {"status": "INVALID", "reasons": ["invalid"]},
        {"completed": 26},
    ],
)
def test_paired_result_requires_complete_valid_arms(
    control_update: dict[str, object],
) -> None:
    result = _valid_capability_result()
    payload = result.model_dump(mode="json")
    payload["control"].update(control_update)

    with pytest.raises(ValidationError, match="terminal evidence"):
        _paired().PairedResult.model_validate(payload)


# 功能：验证success reader拒绝重签hash与Markdown后仍以failure event结束的bundle
# 设计：从production writer创建bundle并重算canonical JSON/manifest，确保只由semantic validator杀死
def test_paired_result_reader_rejects_resigned_failure_event_bundle(
    tmp_path: Path,
) -> None:
    result = _valid_capability_result()
    root = tmp_path / "resigned-failure-event-result"
    _write_paired_result_bundle(root, result)
    payload = result.model_dump(mode="json")
    payload["transitions"][-1] = {
        "from_state": "BOTH_VALID",
        "event": "PARENT_INTERRUPTED",
        "to_state": "TERMINAL",
    }
    markdown = (root / "paired-result.md").read_text(encoding="utf-8")
    _rewrite_paired_bundle(root, payload, markdown)

    with pytest.raises(ValueError, match="paired result bundle"):
        _read_paired_result_bundle(root)


# 功能：验证success reader拒绝JSON、Markdown与manifest全部重签后的INVALID capability verdict
# 设计：同步修改classifier、公开Markdown和两个manifest hash，排除projection/checksum旧门禁的间接拒绝
def test_paired_result_reader_rejects_resigned_invalid_verdict_bundle(
    tmp_path: Path,
) -> None:
    result = _valid_capability_result()
    root = tmp_path / "resigned-invalid-verdict-result"
    _write_paired_result_bundle(root, result)
    payload = result.model_dump(mode="json")
    payload["verdict"] = "INVALID"
    payload["classifier"]["verdict"] = "INVALID"
    payload["classifier"]["matches"] = {
        "INVALID": True,
        "REJECT": False,
        "ACCEPT": False,
        "MIXED": False,
    }
    markdown = (root / "paired-result.md").read_text(encoding="utf-8").replace(
        "Verdict: `ACCEPT`",
        "Verdict: `INVALID`",
    )
    _rewrite_paired_bundle(root, payload, markdown)

    with pytest.raises(ValueError, match="paired result bundle"):
        _read_paired_result_bundle(root)


# 功能：验证reader拒绝top-level VALID但nested outcome control为INVALID的完整重签bundle
# 设计：用model_copy绕过constructor validation并用production renderer重签，隔离cross-field semantic gate
def test_paired_result_reader_rejects_resigned_nested_arm_contradiction(
    tmp_path: Path,
) -> None:
    result = _valid_capability_result()
    nested_control = result.outcome.control.model_copy(
        update={"status": "INVALID", "reasons": ["mutated"]}
    )
    mutated_outcome = result.outcome.model_copy(update={"control": nested_control})
    mutated = result.model_copy(update={"outcome": mutated_outcome})
    root = tmp_path / "resigned-nested-arm-contradiction"
    _write_resigned_result_mutation(root, result, mutated)

    with pytest.raises(ValueError, match="paired result bundle"):
        _read_paired_result_bundle(root)


# 功能：验证reader拒绝verdict为ACCEPT但matches声称INVALID命中的完整重签bundle
# 设计：只替换production classifier evidence的matches并重签所有投影，确保不是hash或Markdown门禁拒绝
def test_paired_result_reader_rejects_resigned_match_verdict_contradiction(
    tmp_path: Path,
) -> None:
    result = _valid_capability_result()
    matches = {
        "INVALID": True,
        "REJECT": False,
        "ACCEPT": False,
        "MIXED": False,
    }
    classifier = result.classifier.model_copy(update={"matches": matches})
    mutated = result.model_copy(update={"classifier": classifier})
    root = tmp_path / "resigned-match-verdict-contradiction"
    _write_resigned_result_mutation(root, result, mutated)

    with pytest.raises(ValueError, match="paired result bundle"):
        _read_paired_result_bundle(root)


# 功能：验证reader拒绝已重签但receipt commit漂移的result bundle和单JSON
# 设计：只改receipt_commit并重签JSON/Markdown/manifest，复现v11发现的receipt reference绑定缺口
def test_paired_result_readers_reject_resigned_receipt_commit_mutation(
    tmp_path: Path,
) -> None:
    result = _valid_capability_result()
    mutated = result.model_copy(update={"receipt_commit": "f" * 40})
    root = tmp_path / "resigned-receipt-commit"
    _write_resigned_result_mutation(root, result, mutated)

    with pytest.raises(ValueError, match="paired result bundle"):
        _read_paired_result_bundle(root)
    with pytest.raises(ValueError):
        _read_paired_result_json(root / "paired-result.json")


# 功能：验证reader拒绝已重签但receipt path漂移的result bundle和单JSON
# 设计：只改result自报path并重签所有投影，证明reader绑定外部ReceiptReference而非result字段
def test_paired_result_readers_reject_resigned_receipt_path_mutation(
    tmp_path: Path,
) -> None:
    result = _valid_capability_result()
    mutated = result.model_copy(
        update={"receipt_path": "benchmarks/receipts/other.json"}
    )
    root = tmp_path / "resigned-receipt-path"
    _write_resigned_result_mutation(root, result, mutated)

    with pytest.raises(ValueError, match="paired result bundle"):
        _read_paired_result_bundle(root)
    with pytest.raises(ValueError):
        _read_paired_result_json(root / "paired-result.json")


# 功能：验证reader拒绝已重签但receipt bytes漂移的result bundle和单JSON
# 设计：只改result自报bytes并重签所有投影，覆盖commit/path/hash正确但size claim错误
def test_paired_result_readers_reject_resigned_receipt_bytes_mutation(
    tmp_path: Path,
) -> None:
    result = _valid_capability_result()
    mutated = result.model_copy(update={"receipt_bytes": result.receipt_bytes + 1})
    root = tmp_path / "resigned-receipt-bytes"
    _write_resigned_result_mutation(root, result, mutated)

    with pytest.raises(ValueError, match="paired result bundle"):
        _read_paired_result_bundle(root)
    with pytest.raises(ValueError):
        _read_paired_result_json(root / "paired-result.json")


# 功能：验证reader拒绝已重签但receipt SHA漂移的result bundle和单JSON
# 设计：只改result自报sha并重签所有投影，覆盖旧SHA-only reader无法发现的identity contradiction
def test_paired_result_readers_reject_resigned_receipt_sha_mutation(
    tmp_path: Path,
) -> None:
    result = _valid_capability_result()
    mutated = result.model_copy(update={"receipt_sha256": "f" * 64})
    root = tmp_path / "resigned-receipt-sha"
    _write_resigned_result_mutation(root, result, mutated)

    with pytest.raises(ValueError, match="paired result bundle"):
        _read_paired_result_bundle(root)
    with pytest.raises(ValueError):
        _read_paired_result_json(root / "paired-result.json")


# 功能：验证reader拒绝错误expected receipt reference即使result本身未变
# 设计：保持合法bundle不变，仅替换外部trusted commit，证明expected不是从result反向构造
def test_paired_result_readers_reject_wrong_expected_receipt_reference(
    tmp_path: Path,
) -> None:
    paired = _paired()
    result = _valid_capability_result()
    root = tmp_path / "wrong-expected-reference"
    _write_paired_result_bundle(root, result)
    wrong_reference = _receipt_reference().model_copy(update={"commit": "f" * 40})

    with pytest.raises(ValueError, match="paired result bundle"):
        paired.read_paired_result_bundle(
            root,
            repository=_repository(),
            expected_receipt=wrong_reference,
        )
    with pytest.raises(ValueError):
        paired.read_paired_result_json(
            root / "paired-result.json",
            repository=_repository(),
            expected_receipt=wrong_reference,
        )


# 功能：验证相同receipt bytes位于不同commit时不能互相作为approved reference
# 设计：真实临时Git创建两个commit且receipt blob相同，result声明commit B但reader用commit A必须拒绝
def test_paired_result_readers_reject_same_bytes_different_commit_alias(
    tmp_path: Path,
) -> None:
    paired = _paired()
    repo, first, second, receipt_path = _same_receipt_bytes_git_repo(tmp_path)
    first_reference = paired.observe_receipt_reference(repo, first, receipt_path)
    second_reference = paired.observe_receipt_reference(repo, second, receipt_path)
    assert first_reference.sha256 == second_reference.sha256
    assert first_reference.bytes == second_reference.bytes
    assert first_reference.commit != second_reference.commit
    result = _valid_capability_result_for_reference(second_reference)
    root = tmp_path / "same-bytes-different-commit"
    paired.write_paired_result(
        root,
        result,
        repository=repo,
        expected_receipt=second_reference,
    )

    with pytest.raises(ValueError, match="paired result bundle"):
        paired.read_paired_result_bundle(
            root,
            repository=repo,
            expected_receipt=first_reference,
        )
    with pytest.raises(ValueError):
        paired.read_paired_result_json(
            root / "paired-result.json",
            repository=repo,
            expected_receipt=first_reference,
        )


# 功能：验证Git receipt observer拒绝commit/path/blob关系不成立
# 设计：真实临时Git仓库中请求缺失path，并用错误bytes reference读取合法bundle覆盖observer和reader两层
def test_receipt_reference_rejects_commit_path_blob_mismatch(
    tmp_path: Path,
) -> None:
    paired = _paired()
    repo, _first, second, receipt_path = _same_receipt_bytes_git_repo(tmp_path)
    reference = paired.observe_receipt_reference(repo, second, receipt_path)
    result = _valid_capability_result_for_reference(reference)
    root = tmp_path / "blob-mismatch"
    paired.write_paired_result(
        root,
        result,
        repository=repo,
        expected_receipt=reference,
    )

    with pytest.raises(ValueError, match="Git reference"):
        paired.observe_receipt_reference(
            repo,
            second,
            "benchmarks/receipts/missing.json",
        )
    with pytest.raises(ValueError, match="paired result bundle"):
        paired.read_paired_result_bundle(
            root,
            repository=repo,
            expected_receipt=reference.model_copy(
                update={"bytes": reference.bytes + 1}
            ),
        )


# 功能：验证reader逐字段拒绝所有非canonical derived outcome metric mutation
# 设计：参数只提供字段和值，期望由真实arm fixture手工确定，bundle用production renderer与manifest重签
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("required_artifact_evidence", False),
        ("inventory_control_successes", 1),
        ("inventory_treatment_successes", 2),
        ("feature_control_successes", 7),
        ("feature_treatment_successes", 8),
        ("overall_control_successes", 21),
        ("overall_treatment_successes", 22),
        ("latency_ratio", 2.0),
        ("token_ratio", 2.0),
    ],
)
def test_paired_result_reader_rejects_each_resigned_derived_metric_mutation(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    result = _valid_capability_result()
    mutated_outcome = result.outcome.model_copy(update={field: value})
    mutated = result.model_copy(update={"outcome": mutated_outcome})
    root = tmp_path / f"resigned-derived-{field}"
    _write_resigned_result_mutation(root, result, mutated)

    with pytest.raises(ValueError, match="paired result bundle"):
        _read_paired_result_bundle(root)


# 功能：验证reader逐类拒绝无法由receipt与canonical outcome重算出的predicate evidence
# 设计：翻转每个predicate map的一个真实键并完整重签，保留verdict/matches以隔离receipt-bound重算
@pytest.mark.parametrize(
    "field",
    [
        "invalid_predicates",
        "reject_predicates",
        "accept_predicates",
        "mixed_predicates",
    ],
)
def test_paired_result_reader_rejects_each_resigned_predicate_mutation(
    tmp_path: Path,
    field: str,
) -> None:
    result = _valid_capability_result()
    predicates = dict(getattr(result.classifier, field))
    key = next(iter(predicates))
    predicates[key] = not predicates[key]
    classifier = result.classifier.model_copy(update={field: predicates})
    mutated = result.model_copy(update={"classifier": classifier})
    root = tmp_path / f"resigned-predicate-{field}"
    _write_resigned_result_mutation(root, result, mutated)

    with pytest.raises(ValueError, match="paired result bundle"):
        _read_paired_result_bundle(root)


# 功能：验证单文件paired-result JSON reader也拒绝已重签但语义矛盾的predicate evidence
# 设计：复用完整重签helper后绕过manifest与Markdown入口，直接命中JSON reader的receipt-bound重算
def test_paired_result_json_reader_rejects_resigned_predicate_mutation(
    tmp_path: Path,
) -> None:
    result = _valid_capability_result()
    predicates = dict(result.classifier.accept_predicates)
    key = next(iter(predicates))
    predicates[key] = not predicates[key]
    classifier = result.classifier.model_copy(
        update={"accept_predicates": predicates}
    )
    mutated = result.model_copy(update={"classifier": classifier})
    root = tmp_path / "resigned-json-predicate"
    _write_resigned_result_mutation(root, result, mutated)

    with pytest.raises(ValueError, match="receipt-bound evidence"):
        _read_paired_result_json(root / "paired-result.json")


# 功能：验证production builder生成的canonical bundle仍能被同一production reader接受
# 设计：不手写expected verdict，只断言完整result经writer、manifest和reader逐字节回读一致
def test_paired_result_reader_accepts_production_canonical_bundle(
    tmp_path: Path,
) -> None:
    result = _valid_capability_result()
    root = tmp_path / "canonical-paired-result"

    _write_paired_result_bundle(root, result)

    assert _read_paired_result_bundle(root) == result


# 功能：验证ClassificationEvidence拒绝缺键、多键、非bool、零命中和多命中的matches
# 设计：从production classifier逐类单点mutation并直接触发strict model，覆盖每个局部shape不变量
@pytest.mark.parametrize(
    "matches",
    [
        {"INVALID": False, "REJECT": False, "ACCEPT": True},
        {
            "INVALID": False,
            "REJECT": False,
            "ACCEPT": True,
            "MIXED": False,
            "UNKNOWN": False,
        },
        {"INVALID": False, "REJECT": False, "ACCEPT": 1, "MIXED": False},
        {"INVALID": False, "REJECT": False, "ACCEPT": False, "MIXED": False},
        {"INVALID": True, "REJECT": False, "ACCEPT": True, "MIXED": False},
    ],
)
def test_classification_evidence_rejects_invalid_matches(
    matches: dict[str, object],
) -> None:
    paired = _paired()
    classifier = _valid_capability_result().classifier
    payload = classifier.model_dump(mode="json")
    payload["matches"] = matches

    with pytest.raises(ValidationError):
        paired.ClassificationEvidence.model_validate(payload)


# 功能：验证success reader拒绝外部expected receipt reference的SHA漂移
# 设计：先写合法bundle，再只改trusted reference的SHA，证明reader不信任result自报identity
def test_paired_result_reader_rejects_receipt_identity_mismatch(
    tmp_path: Path,
) -> None:
    paired = _paired()
    result = _valid_capability_result()
    root = tmp_path / "receipt-bound-result"
    _write_paired_result_bundle(root, result)
    changed_reference = _receipt_reference().model_copy(
        update={"sha256": "f" * 64}
    )

    with pytest.raises(ValueError, match="paired result bundle"):
        paired.read_paired_result_bundle(
            root,
            repository=_repository(),
            expected_receipt=changed_reference,
        )


# 功能：验证authorization validator机械绑定receipt/preflight/attempt/provider/root identities
# 设计：先验证完整fixture，再单改总attempt和preflight hash确保不能进入execute
def test_validate_execution_authorization_binds_all_frozen_references() -> None:
    paired = _paired()
    receipt = _receipt()
    preflight = paired.FinalPreflightArtifact.model_validate(
        importlib.import_module(
            "tests.benchmark.test_phase9d_final_preflight"
        )._preflight_payload()
    )
    payload = importlib.import_module(
        "tests.benchmark.test_phase9d_final_preflight"
    )._authorization_payload()
    payload["paired_receipt"] = {
        "commit": preflight.paired_receipt.commit,
        "sha256": "1" * 64,
    }
    payload["final_preflight"] = {"commit": "a" * 40, "sha256": "2" * 64}
    payload["control_commit"] = receipt.arms.control.commit
    payload["treatment_commit"] = receipt.arms.treatment.commit
    payload["attempts"] = {
        "control": receipt.execution_plan.attempts_per_arm,
        "treatment": receipt.execution_plan.attempts_per_arm,
        "total": receipt.execution_plan.total_attempts,
    }
    payload["maximum_authorized_attempts"] = receipt.execution_plan.total_attempts
    payload["output_parent_sha256"] = preflight.external_parent.canonical_path_sha256
    payload["logical_basenames"] = {
        "control": receipt.execution_plan.control_output_logical_root,
        "treatment": receipt.execution_plan.treatment_output_logical_root,
    }
    authorization = paired.ExecutionAuthorizationArtifact.model_validate(payload)

    paired.validate_execution_authorization(
        authorization,
        preflight=preflight,
        receipt=receipt,
        receipt_sha256="1" * 64,
        preflight_commit="a" * 40,
        preflight_sha256="2" * 64,
    )

    invalid_attempts = authorization.attempts.model_copy(update={"total": 53})
    invalid = authorization.model_copy(update={"attempts": invalid_attempts})
    with pytest.raises(ValueError, match="authorization identity mismatch"):
        paired.validate_execution_authorization(
            invalid,
            preflight=preflight,
            receipt=receipt,
            receipt_sha256="1" * 64,
            preflight_commit="a" * 40,
            preflight_sha256="2" * 64,
        )


# 功能：验证private child分别保留exit 0/1/2且不把stdout/stderr嵌入result
# 设计：真实启动离线Python子进程并写private log，防止shell/mocked returncode掩盖边界
@pytest.mark.parametrize("exit_code", [0, 1, 2])
def test_private_child_captures_exit_status_without_serializing_output(
    tmp_path: Path,
    exit_code: int,
) -> None:
    script = _script()
    stdout = tmp_path / f"stdout-{exit_code}.log"
    stderr = tmp_path / f"stderr-{exit_code}.log"
    canary = "PRIVATE_CHILD_CANARY"

    result = script.run_private_child(
        [sys.executable, "-c", f"import sys; print('{canary}'); sys.exit({exit_code})"],
        cwd=tmp_path,
        env=_isolated_child_env(),
        stdout_path=stdout,
        stderr_path=stderr,
    )

    assert result.exit_code == exit_code
    assert result.signal_number is None
    assert canary in stdout.read_text(encoding="utf-8")
    assert canary not in repr(result)
    assert stdout.stat().st_mode & 0o777 == 0o600
    assert stderr.stat().st_mode & 0o777 == 0o600


# 功能：验证parent cancellation终止整个child process group并完成reap
# 设计：child再spawn grandchild，event触发取消后用killpg探测整个group已不存在
def test_private_child_cancellation_reaps_child_process_group(tmp_path: Path) -> None:
    script = _script()
    cancel = threading.Event()
    source = (
        "import subprocess,sys,time; "
        "subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); "
        "print('ready', flush=True); time.sleep(60)"
    )
    timer = threading.Timer(0.2, cancel.set)
    timer.start()
    try:
        result = script.run_private_child(
            [sys.executable, "-c", source],
            cwd=tmp_path,
            env=_isolated_child_env(),
            stdout_path=tmp_path / "stdout.log",
            stderr_path=tmp_path / "stderr.log",
            cancel_event=cancel,
        )
    finally:
        timer.cancel()

    assert result.cancelled is True
    assert result.process_group_id is not None
    assert result.signal_number == signal.SIGTERM
    assert result.cleanup_term_sent is True
    assert result.cleanup_kill_sent is False
    with pytest.raises(ProcessLookupError):
        os.killpg(result.process_group_id, 0)
    assert "ready" in (tmp_path / "stdout.log").read_text(encoding="utf-8")


# 功能：验证direct child正常退出后同group grandchild也必须被清理且保留原exit code
# 设计：真实child派生长睡眠grandchild后立即exit，最终用killpg探测并在RED失败时安全清理
@pytest.mark.parametrize("exit_code", [0, 1])
def test_private_child_reaps_descendants_after_direct_exit(
    tmp_path: Path,
    exit_code: int,
) -> None:
    script = _script()
    source = (
        "import subprocess,sys; from pathlib import Path; "
        "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); "
        "Path('grandchild.pid').write_text(str(child.pid), encoding='utf-8'); "
        f"sys.exit({exit_code})"
    )
    result = script.run_private_child(
        [sys.executable, "-c", source],
        cwd=tmp_path,
        env=_isolated_child_env(),
        stdout_path=tmp_path / "stdout.log",
        stderr_path=tmp_path / "stderr.log",
    )
    assert result.process_group_id is not None
    try:
        assert result.exit_code == exit_code
        with pytest.raises(ProcessLookupError):
            os.killpg(result.process_group_id, 0)
    finally:
        try:
            os.killpg(result.process_group_id, signal.SIGKILL)
        except ProcessLookupError:
            pass


# 功能：验证忽略SIGTERM的descendant触发SIGKILL升级且direct exit 0不被覆盖
# 设计：grandchild先安装TERM ignore再由direct child退出，断言production cleanup evidence和group消失
def test_private_child_escalates_descendant_cleanup_to_sigkill(tmp_path: Path) -> None:
    script = _script()
    grandchild = (
        "import signal,time; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "time.sleep(60)"
    )
    source = (
        "import subprocess,sys,time; "
        f"subprocess.Popen([sys.executable,'-c',{grandchild!r}]); "
        "time.sleep(0.2); sys.exit(0)"
    )

    result = script.run_private_child(
        [sys.executable, "-c", source],
        cwd=tmp_path,
        env=_isolated_child_env(),
        stdout_path=tmp_path / "stdout.log",
        stderr_path=tmp_path / "stderr.log",
    )
    assert result.process_group_id is not None
    try:
        assert result.exit_code == 0
        assert result.cleanup_term_sent is True
        assert result.cleanup_kill_sent is True
        assert result.process_group_gone is True
        with pytest.raises(ProcessLookupError):
            os.killpg(result.process_group_id, 0)
    finally:
        try:
            os.killpg(result.process_group_id, signal.SIGKILL)
        except ProcessLookupError:
            pass


# 功能：验证无descendant的direct exit不会虚构cleanup signal evidence
# 设计：运行立即exit 0的真实child，断言group gone但TERM/KILL均未发送
def test_private_child_without_descendants_sends_no_cleanup_signal(
    tmp_path: Path,
) -> None:
    result = _script().run_private_child(
        [sys.executable, "-c", "raise SystemExit(0)"],
        cwd=tmp_path,
        env=_isolated_child_env(),
        stdout_path=tmp_path / "stdout.log",
        stderr_path=tmp_path / "stderr.log",
    )

    assert result.exit_code == 0
    assert result.cleanup_term_sent is False
    assert result.cleanup_kill_sent is False
    assert result.process_group_gone is True


# 功能：验证cleanup完成后若observer仍不能证明process group消失则必须fail closed
# 设计：先执行真实TERM/KILL cleanup，再只把最终proof改为gone=false，检查run_private_child拒绝返回成功证据
def test_private_child_rejects_unverifiable_process_group_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = _script()
    original_cleanup = script._terminate_process_group

    # 保留真实signals/reap，仅模拟最终liveness proof不可确认
    def unverifiable_cleanup(process: object, process_group_id: int) -> object:
        observed = original_cleanup(process, process_group_id)
        return script._ProcessGroupCleanup(
            term_sent=observed.term_sent,
            kill_sent=observed.kill_sent,
            gone=False,
        )

    monkeypatch.setattr(script, "_terminate_process_group", unverifiable_cleanup)

    with pytest.raises(RuntimeError, match="process group cleanup"):
        script.run_private_child(
            [sys.executable, "-c", "raise SystemExit(0)"],
            cwd=tmp_path,
            env=_isolated_child_env(),
            stdout_path=tmp_path / "stdout.log",
            stderr_path=tmp_path / "stderr.log",
        )


# 功能：验证parent cancellation在TERM无效时记录direct child真实SIGKILL而非固定SIGTERM
# 设计：真实child忽略TERM并由event取消，断言最终reaped signal和cleanup升级证据
def test_private_child_cancellation_records_actual_sigkill(tmp_path: Path) -> None:
    cancel = threading.Event()
    timer = threading.Timer(0.2, cancel.set)
    timer.start()
    try:
        result = _script().run_private_child(
            [
                sys.executable,
                "-c",
                "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)",
            ],
            cwd=tmp_path,
            env=_isolated_child_env(),
            stdout_path=tmp_path / "stdout.log",
            stderr_path=tmp_path / "stderr.log",
            cancel_event=cancel,
        )
    finally:
        timer.cancel()

    assert result.cancelled is True
    assert result.exit_code is None
    assert result.signal_number == signal.SIGKILL
    assert result.cleanup_term_sent is True
    assert result.cleanup_kill_sent is True
    assert result.process_group_gone is True


# 功能：验证spawn failure返回脱敏终态且不会伪造exit code
# 设计：使用不存在executable触发真实Popen错误，结果中不保存argv/env/path
def test_private_child_spawn_failure_is_redacted(tmp_path: Path) -> None:
    result = _script().run_private_child(
        [str(tmp_path / "missing-python")],
        cwd=tmp_path,
        env={"ANTHROPIC_API_KEY": "SPAWN_SECRET_CANARY"},
        stdout_path=tmp_path / "stdout.log",
        stderr_path=tmp_path / "stderr.log",
    )

    assert result.spawned is False
    assert result.exit_code is None
    assert "SPAWN_SECRET_CANARY" not in repr(result)


# 功能：验证第二个private log打开失败时已打开的第一个fd一定被关闭
# 设计：真实打开stdout fd后只对stderr注入PermissionError，再用fstat探测descriptor生命周期
def test_private_child_closes_stdout_fd_when_stderr_open_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = _script()
    stdout = tmp_path / "stdout.log"
    stderr = tmp_path / "stderr.log"
    original_open = os.open
    opened_fd: int | None = None

    # 保留stdout真实open，只在stderr边界制造失败
    def fail_second(path: object, flags: int, mode: int = 0o777) -> int:
        nonlocal opened_fd
        if Path(path) == stderr:
            raise PermissionError("private log denied")
        descriptor = original_open(path, flags, mode)
        if Path(path) == stdout:
            opened_fd = descriptor
        return descriptor

    monkeypatch.setattr(os, "open", fail_second)
    with pytest.raises(PermissionError):
        script.run_private_child(
            [sys.executable, "-c", "pass"],
            cwd=tmp_path,
            env=_isolated_child_env(),
            stdout_path=stdout,
            stderr_path=stderr,
        )

    assert opened_fd is not None
    with pytest.raises(OSError):
        os.fstat(opened_fd)


# 功能：验证private child把自发SIGTERM/SIGINT与普通exit code严格区分
# 设计：参数化两个真实信号，避免父取消路径掩盖负returncode映射
@pytest.mark.parametrize("termination_signal", [signal.SIGTERM, signal.SIGINT])
def test_private_child_captures_signal_termination(
    tmp_path: Path,
    termination_signal: signal.Signals,
) -> None:
    result = _script().run_private_child(
        [
            sys.executable,
            "-c",
            f"import os,signal; os.kill(os.getpid(), {int(termination_signal)})",
        ],
        cwd=tmp_path,
        env=_isolated_child_env(),
        stdout_path=tmp_path / "stdout.log",
        stderr_path=tmp_path / "stderr.log",
    )

    assert result.exit_code is None
    assert result.signal_number == termination_signal


# 功能：验证argv中的shell metacharacter只作为普通参数传递
# 设计：把分号和touch文本放入sys.argv并检查marker未创建，杀死shell=True或字符串拼接
def test_private_child_does_not_interpret_shell_metacharacters(tmp_path: Path) -> None:
    marker = tmp_path / "MUST_NOT_EXIST"
    result = _script().run_private_child(
        [sys.executable, "-c", "import sys; assert len(sys.argv) == 2", f";touch {marker}"],
        cwd=tmp_path,
        env=_isolated_child_env(),
        stdout_path=tmp_path / "stdout.log",
        stderr_path=tmp_path / "stderr.log",
    )

    assert result.exit_code == 0
    assert not marker.exists()


# 功能：验证脚本CLI只暴露preflight/execute/inspect且无provider/model覆盖
# 设计：解析三条合法命令并拒绝execute --model，确保profile仍是唯一behavior source
def test_paired_script_cli_has_only_observer_subcommands() -> None:
    script = _script()

    assert script._parse_args(["inspect", "--artifact", "a.json"]).command == "inspect"
    terminal = script._parse_args(
        ["inspect", "--artifact", "terminal.json", "--kind", "terminal"]
    )
    assert terminal.kind == "terminal"
    with pytest.raises(SystemExit):
        script._parse_args(
            [
                "execute",
                "--preflight",
                "p.json",
                "--authorization",
                "a.json",
                "--model",
                "other",
            ]
        )

    valid_execute = [
        "execute",
        "--receipt",
        "r.json",
        "--preflight",
        "p.json",
        "--authorization",
        "a.json",
        "--repository",
        "repo",
        "--control-worktree",
        "c1",
        "--treatment-worktree",
        "c2",
    ]
    assert script._parse_args(valid_execute).command == "execute"
    with pytest.raises(SystemExit):
        script._parse_args(
            [*valid_execute, "--authorization-use", "/tmp/arbitrary-use.json"]
        )
    with pytest.raises(SystemExit):
        script._parse_args([*valid_execute, "--pair-output", "/tmp/arbitrary-result"])


# 功能：验证CLI observer错误不会回显credential canary或主机private path
# 设计：inspect不存在且带canary的路径，断言固定stderr和exit2而非Python traceback
def test_paired_cli_sanitizes_observer_exceptions(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    secret_path = tmp_path / "SECRET_CREDENTIAL_CANARY.json"

    exit_code = _script().main(["inspect", "--artifact", str(secret_path)])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.err == "phase9d paired observer failed\n"
    assert "SECRET_CREDENTIAL_CANARY" not in captured.err
    assert str(tmp_path) not in captured.err


# 功能：验证execution main必须精确绑定authorization artifact commit且与remote一致
# 设计：用受控Git adapter先证明合法identity，再只改变HEAD杀死“任意clean新commit仍可执行”漏洞
def test_execution_main_identity_is_bound_to_authorization_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = _script()
    preflight = _paired().FinalPreflightArtifact.model_validate(
        importlib.import_module(
            "tests.benchmark.test_phase9d_final_preflight"
        )._preflight_payload()
    )
    authorized_commit = "b" * 40
    values = {
        ("rev-parse", preflight.generator.remote_ref): authorized_commit,
        ("rev-parse", "HEAD"): authorized_commit,
        ("branch", "--show-current"): preflight.generator.branch,
        ("status", "--porcelain", "--untracked-files=all"): "",
    }
    monkeypatch.setattr(script, "_git", lambda _repo, *args: values[args])

    assert script._validate_execution_main_identity(
        tmp_path,
        preflight,
        authorized_commit,
    ) == authorized_commit
    values[("rev-parse", "HEAD")] = "c" * 40
    with pytest.raises(ValueError, match="execution main checkout identity drift"):
        script._validate_execution_main_identity(
            tmp_path,
            preflight,
            authorized_commit,
        )


# 功能：验证每个arm的CLI物化固定profile、cwd、PYTHONPATH和private logs
# 设计：分别构造C1/C2 launch并检查全部behavior-bearing tokens，禁止profile串臂或路径override
def test_materialize_arm_launch_binds_profile_source_and_private_output(
    tmp_path: Path,
) -> None:
    paired = _paired()
    script = _script()
    receipt = _receipt()
    private_parent = tmp_path / "private-parent"
    private_parent.mkdir()
    private_paths = paired.derive_private_evidence_paths(
        private_parent.resolve(),
        receipt.receipt_id,
    )
    source_environment = {
        "ANTHROPIC_API_KEY": "FAKE_MATERIALIZATION_SENTINEL",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
    }
    launches = []
    for arm in ("control", "treatment"):
        worktree = tmp_path / f"{arm}-worktree"
        (worktree / "src" / "kama_claude").mkdir(parents=True)
        output_root = tmp_path / f"{arm}-output"
        launch = script._materialize_arm_launch(
            arm=arm,
            receipt=receipt,
            declared=_identity(receipt, arm).declared,
            worktree=worktree,
            output_root=output_root,
            private_paths=private_paths,
            interpreter=sys.executable,
            source_environment=source_environment,
            credential_env="ANTHROPIC_API_KEY",
        )
        launches.append(launch)
        receipt_arm = getattr(receipt.arms, arm)
        assert launch.cwd == worktree.resolve()
        assert launch.argv[-4:] == (
            "--experiment",
            receipt_arm.profile_path,
            "--output",
            str(output_root),
        )
        assert launch.env["PYTHONPATH"] == str(worktree.resolve() / "src")
        assert "FAKE_MATERIALIZATION_SENTINEL" not in repr(launch.argv)
        assert launch.stdout_path.is_relative_to(private_paths.root)
        assert launch.stderr_path.is_relative_to(private_paths.root)
    assert launches[0].argv != launches[1].argv


# 功能：验证worktree validator拒绝branch-attached、dirty、wrong-head和wrong-source binding
# 设计：对纯observation逐字段mutation，避免创建正式Phase 9D worktree
@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("registered", False),
        ("detached", False),
        ("clean", False),
        ("observed_head", "f" * 40),
        ("canonical_path_sha256", "f" * 64),
        ("profile_exists", False),
    ],
)
def test_worktree_binding_rejects_every_observer_drift(
    tmp_path: Path,
    field: str,
    replacement: object,
) -> None:
    paired = _paired()
    repository = tmp_path / "repo"
    output_parent = tmp_path / "output"
    worktree = tmp_path / "worktree"
    for path in (repository, output_parent, worktree / "src"):
        path.mkdir(parents=True)
    observation = paired.WorktreeObservation(
        label="C1_WORKTREE",
        path=worktree,
        canonical_path_sha256=hashlib.sha256(
            str(worktree.resolve()).encode("utf-8")
        ).hexdigest(),
        source_root=worktree / "src",
        registered=True,
        detached=True,
        clean=True,
        observed_head="c" * 40,
        profile_exists=True,
        source_import=_source_import_evidence(worktree / "src"),
    ).model_copy(update={field: replacement})

    with pytest.raises(ValueError, match="worktree binding"):
        paired.validate_worktree_binding(
            observation,
            expected_commit="c" * 40,
            repository=repository,
            output_parent=output_parent,
        )


# 功能：验证source import root若指向main checkout即使布尔标志为true也会被拒绝
# 设计：显式携带canonical source path而非信任source_import_matches，覆盖错误PYTHONPATH绑定
def test_worktree_binding_rejects_source_root_from_main_checkout(tmp_path: Path) -> None:
    paired = _paired()
    repository = tmp_path / "repo"
    output_parent = tmp_path / "output"
    worktree = tmp_path / "worktree"
    for path in (repository / "src", output_parent, worktree / "src"):
        path.mkdir(parents=True)
    observation = paired.WorktreeObservation(
        label="C1_WORKTREE",
        path=worktree,
        canonical_path_sha256=hashlib.sha256(
            str(worktree.resolve()).encode("utf-8")
        ).hexdigest(),
        source_root=repository / "src",
        registered=True,
        detached=True,
        clean=True,
        observed_head="c" * 40,
        profile_exists=True,
        source_import=_source_import_evidence(repository / "src"),
    )

    with pytest.raises(ValueError, match="worktree binding"):
        paired.validate_worktree_binding(
            observation,
            expected_commit="c" * 40,
            repository=repository,
            output_parent=output_parent,
        )


# 构造与冻结receipt一致的preflight与单次authorization
def _execution_artifacts(receipt: object) -> tuple[object, object]:
    paired = _paired()
    preflight_payload = importlib.import_module(
        "tests.benchmark.test_phase9d_final_preflight"
    )._preflight_payload()
    preflight_payload["paired_receipt"] = _receipt_reference().model_dump(
        mode="json"
    )
    preflight_payload["external_parent"]["canonical_path_sha256"] = "3" * 64
    preflight = paired.FinalPreflightArtifact.model_validate(preflight_payload)
    authorization_payload = importlib.import_module(
        "tests.benchmark.test_phase9d_final_preflight"
    )._authorization_payload()
    authorization_payload["paired_receipt"] = {
        "commit": preflight.paired_receipt.commit,
        "sha256": _receipt_sha256(),
    }
    authorization_payload["final_preflight"] = {
        "commit": "a" * 40,
        "sha256": "2" * 64,
    }
    authorization_payload["control_commit"] = receipt.arms.control.commit
    authorization_payload["treatment_commit"] = receipt.arms.treatment.commit
    authorization_payload["attempts"] = {
        "control": receipt.execution_plan.attempts_per_arm,
        "treatment": receipt.execution_plan.attempts_per_arm,
        "total": receipt.execution_plan.total_attempts,
    }
    authorization_payload["maximum_authorized_attempts"] = (
        receipt.execution_plan.total_attempts
    )
    authorization_payload["logical_basenames"] = {
        "control": receipt.execution_plan.control_output_logical_root,
        "treatment": receipt.execution_plan.treatment_output_logical_root,
    }
    authorization = paired.ExecutionAuthorizationArtifact.model_validate(
        authorization_payload
    )
    return preflight, authorization


# 使用统一离线launch fixture执行一次pair，便于覆盖reservation后的失败边界
def _execute_with_child(
    tmp_path: Path,
    child: object,
    *,
    observe_between: object | None = None,
    mutate_authorization: Callable[[object], object] | None = None,
) -> tuple[object, Path]:
    script = _script()
    receipt = _receipt()
    preflight, authorization = _execution_artifacts(receipt)
    if mutate_authorization is not None:
        authorization = mutate_authorization(authorization)
    private_paths = _paired().derive_private_evidence_paths(
        tmp_path.resolve(),
        receipt.receipt_id,
    )
    use_path = private_paths.authorization_use
    summary = script.execute_control_first(
        repository=_repository(),
        receipt=receipt,
        preflight=preflight,
        authorization=authorization,
        receipt_sha256=_receipt_sha256(),
        preflight_commit="a" * 40,
        preflight_sha256="2" * 64,
        authorization_commit="b" * 40,
        authorization_sha256="4" * 64,
        authorization_use_record=_use_record().model_copy(
            update={
                "authorization_sha256": "4" * 64,
                "paired_receipt_sha256": _receipt_sha256(),
            }
        ),
        receipt_path=_receipt_path(),
        private_paths=private_paths,
        control=script.ArmLaunch(
            expected=_expected_arm(receipt, "control"),
            argv=("fake",),
            cwd=tmp_path,
            env={},
            output_root=tmp_path / "control",
            stdout_path=private_paths.control_stdout,
            stderr_path=private_paths.control_stderr,
        ),
        treatment=script.ArmLaunch(
            expected=_expected_arm(receipt, "treatment"),
            argv=("fake",),
            cwd=tmp_path,
            env={},
            output_root=tmp_path / "treatment",
            stdout_path=private_paths.treatment_stdout,
            stderr_path=private_paths.treatment_stderr,
        ),
        between_expected=_between_arm(),
        observe_between=(
            (lambda: _between_arm())
            if observe_between is None
            else observe_between
        ),
        result_id="phase9d-pair-test",
        created_at_utc="2026-08-01T00:00:00Z",
        child_runner=child,
    )
    return summary, use_path


# 功能：验证authorization gate失败发生在use reservation和C1 spawn之前
# 设计：只破坏total attempts，断言child调用为0且private parent中不存在use record
def test_pre_reservation_authorization_failure_starts_no_child(
    tmp_path: Path,
) -> None:
    calls = 0

    # 若pre-use gate失效，此child会留下可观察调用计数
    def child(_launch: object, _cancel: threading.Event | None) -> object:
        nonlocal calls
        calls += 1
        return _script().ChildResult(True, 0, None, False, 123, None)

    # 只改变冻结attempt总数，保持schema对象其余身份不变
    def invalidate(authorization: object) -> object:
        attempts = authorization.attempts.model_copy(update={"total": 53})
        return authorization.model_copy(update={"attempts": attempts})

    with pytest.raises(ValueError, match="authorization identity mismatch"):
        _execute_with_child(
            tmp_path,
            child,
            mutate_authorization=invalidate,
        )

    assert calls == 0
    assert not (tmp_path / "private" / "authorization-use.json").exists()


# 功能：验证低能力分但artifact完整的VALID C1仍必须继续执行C2
# 设计：C1的27次全部task_failed并exit1，C2全部成功；只允许validity而非score控制顺序
def test_execute_control_first_runs_treatment_after_low_score_valid_control(
    tmp_path: Path,
) -> None:
    receipt = _receipt()
    calls: list[str] = []
    all_successes = {task_id for task_id, _category in _TASKS}

    # 为C1写完整低分证据，为C2写完整成功证据
    def child(launch: object, _cancel: threading.Event | None) -> object:
        calls.append(launch.expected.arm)
        successes = set() if launch.expected.arm == "control" else all_successes
        _write_arm_output(
            launch.output_root,
            receipt,
            launch.expected.arm,
            _attempts(successes),
        )
        exit_code = 1 if launch.expected.arm == "control" else 0
        return _script().ChildResult(True, exit_code, None, False, 123, None)

    summary, _use_path = _execute_with_child(tmp_path, child)

    assert calls == ["control", "treatment"]
    assert summary.state is _paired().PairState.TERMINAL
    assert summary.control is not None and summary.control.status == "VALID"


# 功能：验证between-arm任一漂移在真实orchestrator边界阻止C2 spawn
# 设计：C1写完整VALID证据后只改credential_present，调用计数必须保持1且use不可回收
def test_execute_control_first_blocks_treatment_on_between_arm_drift(
    tmp_path: Path,
) -> None:
    receipt = _receipt()
    calls = 0

    # 仅允许C1运行，若C2被调用则测试立即失败
    def child(launch: object, _cancel: threading.Event | None) -> object:
        nonlocal calls
        calls += 1
        if calls > 1:
            pytest.fail("treatment must not start after between-arm drift")
        _write_arm_output(
            launch.output_root,
            receipt,
            "control",
            _attempts({task_id for task_id, _category in _TASKS}),
        )
        return _script().ChildResult(True, 0, None, False, 123, None)

    observed = _between_arm().model_copy(update={"credential_present": False})
    summary, use_path = _execute_with_child(
        tmp_path,
        child,
        observe_between=lambda: observed,
    )

    assert calls == 1
    assert summary.state is _paired().PairState.TERMINAL
    assert summary.treatment is None
    assert use_path.is_file()
    assert not (tmp_path / "treatment").exists()


# 功能：验证reservation后的cancelled C1保留partial root并使pair永久INVALID
# 设计：fake child写partial marker再返回signal/cancel终态，断言不删除、不resume、不启动C2
def test_execute_control_first_preserves_cancelled_control_partial_artifacts(
    tmp_path: Path,
) -> None:
    calls = 0

    # 模拟C1取消前已产生partial artifact目录
    def child(launch: object, _cancel: threading.Event | None) -> object:
        nonlocal calls
        calls += 1
        launch.output_root.mkdir()
        (launch.output_root / "partial.marker").write_text("partial\n", encoding="utf-8")
        return _script().ChildResult(
            True,
            None,
            signal.SIGTERM,
            True,
            123,
            "cancelled",
        )

    summary, use_path = _execute_with_child(tmp_path, child)

    assert calls == 1
    assert summary.state is _paired().PairState.TERMINAL
    assert (tmp_path / "control" / "partial.marker").is_file()
    assert use_path.is_file()
    assert not (tmp_path / "treatment").exists()
    paths = _paired().derive_private_evidence_paths(
        tmp_path.resolve(),
        _receipt().receipt_id,
    )
    terminal = _paired().read_strict_artifact(
        paths.terminal_record,
        _paired().PairTerminalRecord,
    )
    assert terminal.phase == "control_audit"
    assert terminal.control_child is not None
    assert terminal.control_child.cancelled is True
    assert terminal.capability_delta_published is False


# 功能：验证thin orchestrator仅在C1 VALID后运行C2并写入唯一result
# 设计：fake child在调用时生成真实BaselineReport，锁定reservation与审计顺序
def test_execute_control_first_runs_both_valid_arms_in_order(tmp_path: Path) -> None:
    paired = _paired()
    script = _script()
    receipt = _receipt()
    preflight, authorization = _execution_artifacts(receipt)
    order: list[str] = []
    successes = {task_id for task_id, _category in _TASKS}
    control_root = tmp_path / "control"
    treatment_root = tmp_path / "treatment"
    private_paths = paired.derive_private_evidence_paths(
        tmp_path.resolve(),
        receipt.receipt_id,
    )

    # 为当前arm生成完整单臂evidence并返回真实exit语义
    def child(launch: object, _cancel: threading.Event | None) -> object:
        order.append(launch.expected.arm)
        _write_arm_output(
            launch.output_root,
            receipt,
            launch.expected.arm,
            _attempts(successes),
        )
        return script.ChildResult(True, 0, None, False, 123, None)

    summary = script.execute_control_first(
        repository=_repository(),
        receipt=receipt,
        preflight=preflight,
        authorization=authorization,
        receipt_sha256=_receipt_sha256(),
        preflight_commit="a" * 40,
        preflight_sha256="2" * 64,
        authorization_commit="b" * 40,
        authorization_sha256="4" * 64,
        authorization_use_record=_use_record().model_copy(
            update={
                "authorization_sha256": "4" * 64,
                "paired_receipt_sha256": _receipt_sha256(),
            }
        ),
        receipt_path=_receipt_path(),
        private_paths=private_paths,
        control=script.ArmLaunch(
            expected=_expected_arm(receipt, "control"),
            argv=("fake",),
            cwd=tmp_path,
            env={},
            output_root=control_root,
            stdout_path=private_paths.control_stdout,
            stderr_path=private_paths.control_stderr,
        ),
        treatment=script.ArmLaunch(
            expected=_expected_arm(receipt, "treatment"),
            argv=("fake",),
            cwd=tmp_path,
            env={},
            output_root=treatment_root,
            stdout_path=private_paths.treatment_stdout,
            stderr_path=private_paths.treatment_stderr,
        ),
        between_expected=_between_arm(),
        observe_between=lambda: _between_arm(),
        result_id="phase9d-pair-test",
        created_at_utc="2026-08-01T00:00:00Z",
        child_runner=child,
    )

    assert order == ["control", "treatment"]
    assert summary.state is paired.PairState.TERMINAL
    assert summary.result is not None
    assert summary.result.verdict in {"ACCEPT", "MIXED", "REJECT"}
    assert private_paths.paired_result.is_dir()
    assert (private_paths.paired_result / "paired-result.json").is_file()
    assert (private_paths.paired_result / "paired-result.md").is_file()
    assert not private_paths.terminal_record.exists()


# 功能：验证C1 artifact INVALID时不运行C2且one-use仍永久占用
# 设计：fake child仅写入缺一row的baseline，同时用调用计数证明无treatment spawn
def test_execute_control_first_stops_after_invalid_control(tmp_path: Path) -> None:
    paired = _paired()
    script = _script()
    receipt = _receipt()
    preflight, authorization = _execution_artifacts(receipt)
    calls = 0
    private_paths = paired.derive_private_evidence_paths(
        tmp_path.resolve(),
        receipt.receipt_id,
    )
    use_path = private_paths.authorization_use
    successes = {task_id for task_id, _category in _TASKS}

    # 只生成不完整C1 matrix，若调用C2则让测试立即失败
    def child(launch: object, _cancel: threading.Event | None) -> object:
        nonlocal calls
        calls += 1
        if calls > 1:
            pytest.fail("treatment must not start after invalid control")
        _write_arm_output(
            launch.output_root,
            receipt,
            "control",
            _attempts(successes)[:-1],
        )
        return script.ChildResult(True, 0, None, False, 123, None)

    summary = script.execute_control_first(
        repository=_repository(),
        receipt=receipt,
        preflight=preflight,
        authorization=authorization,
        receipt_sha256=_receipt_sha256(),
        preflight_commit="a" * 40,
        preflight_sha256="2" * 64,
        authorization_commit="b" * 40,
        authorization_sha256="4" * 64,
        authorization_use_record=_use_record().model_copy(
            update={
                "authorization_sha256": "4" * 64,
                "paired_receipt_sha256": _receipt_sha256(),
            }
        ),
        receipt_path=_receipt_path(),
        private_paths=private_paths,
        control=script.ArmLaunch(
            expected=_expected_arm(receipt, "control"),
            argv=("fake",),
            cwd=tmp_path,
            env={},
            output_root=tmp_path / "control",
            stdout_path=private_paths.control_stdout,
            stderr_path=private_paths.control_stderr,
        ),
        treatment=script.ArmLaunch(
            expected=_expected_arm(receipt, "treatment"),
            argv=("fake",),
            cwd=tmp_path,
            env={},
            output_root=tmp_path / "treatment",
            stdout_path=private_paths.treatment_stdout,
            stderr_path=private_paths.treatment_stderr,
        ),
        between_expected=_between_arm(),
        observe_between=lambda: _between_arm(),
        result_id="phase9d-pair-test",
        created_at_utc="2026-08-01T00:00:00Z",
        child_runner=child,
    )

    assert calls == 1
    assert summary.state is paired.PairState.TERMINAL
    assert summary.result is None
    assert use_path.is_file()
    assert private_paths.terminal_record.is_file()
    assert not private_paths.paired_result.exists()


# 功能：验证authorization-use成功后child spawn failure永久消耗授权并阻止C2
# 设计：让child runner返回真实spawned=false终态，断言pair INVALID且第二次reservation冲突
def test_post_reservation_spawn_failure_is_invalid_and_consumes_use(
    tmp_path: Path,
) -> None:
    calls = 0

    # 返回脱敏spawn failure，若C2被调用则测试立即失败
    def child(_launch: object, _cancel: threading.Event | None) -> object:
        nonlocal calls
        calls += 1
        return _script().ChildResult(False, None, None, False, None, "spawn_failed")

    summary, use_path = _execute_with_child(tmp_path, child)

    assert calls == 1
    assert summary.state is _paired().PairState.TERMINAL
    assert summary.result is None
    assert use_path.is_file()
    with pytest.raises(ValueError, match="already reserved"):
        _paired().reserve_authorization_use(use_path, _use_record())


# 功能：验证一次完整post-reservation失败后再次调用orchestrator不能启动任何child或覆盖terminal
# 设计：第一次spawned=false消费use并记录terminal，第二次走同一入口时断言reservation冲突且bytes保持不变
def test_second_execute_cannot_cross_consumed_authorization_boundary(
    tmp_path: Path,
) -> None:
    calls = 0

    # 第一次返回spawn failure，第二次若越过reservation则立即暴露调用
    def child(_launch: object, _cancel: threading.Event | None) -> object:
        nonlocal calls
        calls += 1
        if calls > 1:
            pytest.fail("consumed authorization must block every later child")
        return _script().ChildResult(False, None, None, False, None, "spawn_failed")

    summary, paths = _execute_with_terminal_contract(tmp_path, child)
    assert summary.terminal is not None
    terminal_bytes = paths.terminal_record.read_bytes()

    with pytest.raises(ValueError, match="already reserved"):
        _execute_with_terminal_contract(tmp_path, child)

    assert calls == 1
    assert paths.authorization_use.is_file()
    assert paths.terminal_record.read_bytes() == terminal_bytes


# 功能：验证reservation后child adapter抛异常仍返回INVALID而不泄漏或允许C2
# 设计：注入带canary的OSError，断言use保留且异常不穿透orchestrator边界
def test_post_reservation_child_exception_is_redacted_invalid(tmp_path: Path) -> None:
    calls = 0

    # 以异常模拟log open或Popen外围失败，禁止caller得到canary详情
    def child(_launch: object, _cancel: threading.Event | None) -> object:
        nonlocal calls
        calls += 1
        raise OSError("SECRET_CHILD_EXCEPTION_CANARY")

    summary, use_path = _execute_with_child(tmp_path, child)

    assert calls == 1
    assert summary.state is _paired().PairState.TERMINAL
    assert summary.result is None
    assert use_path.is_file()
    assert "SECRET_CHILD_EXCEPTION_CANARY" not in repr(summary)


# 功能：验证precheck后出现的C1 root collision不会触发改名、删除或重试
# 设计：fake child模拟existing CLI在spawn后创建root并exit2，检查use持久且无-002/C2
def test_post_reservation_control_root_race_is_invalid_without_retry(
    tmp_path: Path,
) -> None:
    calls = 0

    # 模拟single-arm CLI在竞争窗口发现或创建冻结root后以2退出
    def child(launch: object, _cancel: threading.Event | None) -> object:
        nonlocal calls
        calls += 1
        launch.output_root.mkdir()
        return _script().ChildResult(True, 2, None, False, 123, None)

    summary, use_path = _execute_with_child(tmp_path, child)

    assert calls == 1
    assert summary.state is _paired().PairState.TERMINAL
    assert use_path.is_file()
    assert (tmp_path / "control").is_dir()
    assert not (tmp_path / "control-002").exists()
    assert not (tmp_path / "treatment").exists()


# 功能：验证tracked artifact必须与HEAD blob逐字节一致而非只要ls-files命中
# 设计：在tmp Git repo提交artifact后修改工作树bytes，确保observer拒绝未提交漂移
def test_tracked_artifact_identity_rejects_worktree_blob_drift(tmp_path: Path) -> None:
    script = _script()
    repository = tmp_path / "repo"
    repository.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
    artifact = repository / "artifact.json"
    artifact.write_text('{"version":1}\n', encoding="utf-8")
    subprocess.run(["git", "add", "artifact.json"], cwd=repository, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Phase9D Test",
            "-c",
            "user.email=phase9d@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        cwd=repository,
        check=True,
    )
    commit, _digest = script._tracked_artifact_identity(repository, artifact)
    assert len(commit) == 40
    artifact.write_text('{"version":2}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="tracked artifact"):
        script._tracked_artifact_identity(repository, artifact)


# 功能：验证paired inspect同时校验canonical JSON与Markdown投影
# 设计：先写合法bundle再只篡改Markdown verdict，要求离线bundle reader fail closed
def test_paired_result_bundle_rejects_markdown_json_verdict_divergence(
    tmp_path: Path,
) -> None:
    paired = _paired()
    receipt = _receipt()
    successes = {task_id for task_id, _category in _TASKS}
    control_root = tmp_path / "control-arm"
    treatment_root = tmp_path / "treatment-arm"
    _write_arm_output(control_root, receipt, "control", _attempts(successes))
    _write_arm_output(treatment_root, receipt, "treatment", _attempts(successes))
    control = paired.audit_arm_result(
        expected=_expected_arm(receipt, "control"),
        exit_code=0,
        signal_number=None,
        output_root=control_root,
        receipt=receipt,
    )
    treatment = paired.audit_arm_result(
        expected=_expected_arm(receipt, "treatment"),
        exit_code=0,
        signal_number=None,
        output_root=treatment_root,
        receipt=receipt,
    )
    result = paired.build_paired_result(
        result_id="phase9d-bundle-test",
        created_at_utc="2026-08-01T00:00:00Z",
        receipt_reference=_receipt_reference(),
        preflight_commit="a" * 40,
        preflight_sha256="2" * 64,
        authorization_commit="b" * 40,
        authorization_sha256="3" * 64,
        authorization_use_sha256="4" * 64,
        receipt=receipt,
        control=control,
        treatment=treatment,
        control_child=_successful_child_evidence(),
        treatment_child=_successful_child_evidence(),
        transitions=_successful_transition_history(),
    )
    root = tmp_path / "paired-bundle"
    _write_paired_result_bundle(root, result)
    markdown = root / "paired-result.md"
    markdown.write_text(
        markdown.read_text(encoding="utf-8").replace(
            f"Verdict: `{result.verdict}`",
            "Verdict: `INVALID`",
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="paired result"):
        _read_paired_result_bundle(root)


# 功能：验证reader拒绝缺失manifest或manifest hash漂移的partial paired bundle
# 设计：先经production writer发布完整bundle，再单点破坏commit marker或hash避免手写positive fixture
@pytest.mark.parametrize("mutation", ["missing_manifest", "wrong_hash"])
def test_read_paired_result_bundle_rejects_manifest_mutation(
    tmp_path: Path,
    mutation: str,
) -> None:
    receipt = _receipt()
    summary, paths = _execute_with_terminal_contract(
        tmp_path,
        _complete_fake_child(receipt),
    )
    assert summary.result is not None
    manifest_path = paths.paired_result / "manifest.json"
    if mutation == "missing_manifest":
        manifest_path.unlink()
    else:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload["json_sha256"] = "0" * 64
        manifest_path.write_text(
            _paired().canonical_json(payload) + "\n",
            encoding="utf-8",
        )

    with pytest.raises(ValueError, match="paired result"):
        _read_paired_result_bundle(paths.paired_result)


# 功能：验证rename前失败只留下不可接受staging且不会发布final目录
# 设计：先取得真实PairedResult，再只在atomic rename边界注入OSError检查commit point
def test_write_paired_result_rename_failure_is_not_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    summary, _paths = _execute_with_terminal_contract(
        source,
        _complete_fake_child(_receipt()),
    )
    assert summary.result is not None
    target = tmp_path / "candidate"
    monkeypatch.setattr(
        _paired().os,
        "rename",
        lambda *_args: (_ for _ in ()).throw(OSError("rename failed")),
    )

    with pytest.raises(ValueError, match="atomically published"):
        _write_paired_result_bundle(target, summary.result)

    assert not target.exists()
    assert (tmp_path / ".candidate.staging").is_dir()
    with pytest.raises(ValueError, match="paired result"):
        _read_paired_result_bundle(target)

    with pytest.raises(ValueError, match="paired result"):
        _read_paired_result_bundle(tmp_path / ".candidate.staging")


# 功能：验证existing final或staging均使publisher在写入前fail closed
# 设计：参数化创建一种冲突目录并断言production不会覆盖该create-once边界
@pytest.mark.parametrize("conflict", ["final", "staging"])
def test_write_paired_result_rejects_publication_path_conflict(
    tmp_path: Path,
    conflict: str,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    summary, _paths = _execute_with_terminal_contract(
        source,
        _complete_fake_child(_receipt()),
    )
    assert summary.result is not None
    target = tmp_path / "candidate"
    conflict_path = target if conflict == "final" else tmp_path / ".candidate.staging"
    conflict_path.mkdir()

    with pytest.raises(ValueError, match="already exists"):
        _write_paired_result_bundle(target, summary.result)


# 功能：验证paired publisher每个rename前I/O边界失败都不能形成可接受final或staging success
# 设计：跟踪真实fd对应文件并在open/write/short-write/fsync/validation/rename单点注入，调用production publisher/reader
@pytest.mark.parametrize(
    "boundary",
    [
        "json_open",
        "json_write",
        "json_short_write",
        "json_fsync",
        "markdown_open",
        "markdown_write",
        "markdown_short_write",
        "markdown_fsync",
        "manifest_open",
        "manifest_write",
        "manifest_short_write",
        "manifest_fsync",
        "staging_fsync",
        "pre_rename_validation",
        "rename",
    ],
)
def test_paired_result_precommit_fault_matrix_never_publishes_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    boundary: str,
) -> None:
    paired = _paired()
    source = tmp_path / "source"
    source.mkdir()
    summary, _paths = _execute_with_terminal_contract(
        source,
        _complete_fake_child(_receipt()),
    )
    assert summary.result is not None
    target = tmp_path / "candidate"
    staging = tmp_path / ".candidate.staging"
    descriptor_paths: dict[int, Path] = {}
    original_open = os.open
    original_write = os.write
    original_fsync = os.fsync
    original_bundle_reader = paired._read_paired_result_bundle

    # 将publisher fd绑定到logical JSON/Markdown/manifest/staging boundary
    def kind(path: Path) -> str | None:
        if "paired-result.json" in path.name:
            return "json"
        if "paired-result.md" in path.name:
            return "markdown"
        if "manifest.json" in path.name:
            return "manifest"
        if path == staging:
            return "staging"
        return None

    # 在指定文件open前失败，否则保存真实fd到path关联
    def observed_open(path: object, flags: int, mode: int = 0o777) -> int:
        candidate = Path(path)
        file_kind = kind(candidate)
        if boundary == f"{file_kind}_open":
            raise OSError("publication open failure")
        descriptor = original_open(path, flags, mode)
        descriptor_paths[descriptor] = candidate
        return descriptor

    # 在指定文件执行真实error或len-1短写，其余保持production syscall
    def observed_write(descriptor: int, data: bytes) -> int:
        file_kind = kind(descriptor_paths.get(descriptor, Path("unknown")))
        if boundary == f"{file_kind}_write":
            raise OSError("publication write failure")
        if boundary == f"{file_kind}_short_write":
            return original_write(descriptor, data[:-1])
        return original_write(descriptor, data)

    # 在指定file或完整manifest已存在后的staging fsync失败
    def observed_fsync(descriptor: int) -> None:
        file_kind = kind(descriptor_paths.get(descriptor, Path("unknown")))
        staging_final = file_kind == "staging" and (staging / "manifest.json").exists()
        if boundary == f"{file_kind}_fsync" and (
            file_kind != "staging" or staging_final
        ):
            raise OSError("publication fsync failure")
        original_fsync(descriptor)

    monkeypatch.setattr(os, "open", observed_open)
    monkeypatch.setattr(os, "write", observed_write)
    monkeypatch.setattr(os, "fsync", observed_fsync)
    if boundary == "pre_rename_validation":
        monkeypatch.setattr(
            paired,
            "_read_paired_result_bundle",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                ValueError("validation failure")
            ),
        )
    if boundary == "rename":
        monkeypatch.setattr(
            os,
            "rename",
            lambda *_args: (_ for _ in ()).throw(OSError("rename failure")),
        )

    with pytest.raises((OSError, ValueError)):
        _write_paired_result_bundle(target, summary.result)

    monkeypatch.setattr(paired, "_read_paired_result_bundle", original_bundle_reader)

    assert not target.exists()
    with pytest.raises(ValueError, match="paired result"):
        _read_paired_result_bundle(target)
    if staging.exists():
        with pytest.raises(ValueError, match="paired result"):
            _read_paired_result_bundle(staging)


# 功能：验证同一授权消费下valid success与valid failure terminal绝不能共存
# 设计：先完成真实fake pair，再用共享reducer构造合法INVALID terminal触发XOR validator
def test_pair_terminal_exclusivity_rejects_success_failure_coexistence(
    tmp_path: Path,
) -> None:
    paired = _paired()
    receipt = _receipt()
    _summary, paths = _execute_with_terminal_contract(
        tmp_path,
        _complete_fake_child(receipt),
    )
    first = paired.reduce_pair_transition(
        paired.PairState.NOT_STARTED,
        paired.PairEvent.AUTHORIZATION_RESERVED,
        receipt=receipt,
        authorization_use_reserved=True,
    )
    second = paired.reduce_pair_transition(
        paired.PairState.AUTHORIZATION_RESERVED,
        paired.PairEvent.PRIVATE_EVIDENCE_FAILED,
        receipt=receipt,
        authorization_use_reserved=True,
    )
    terminal = paired.build_terminal_record(
        terminal_id="coexisting-terminal",
        created_at_utc="2026-08-01T00:00:00Z",
        phase="control_spawn",
        receipt_sha256="1" * 64,
        preflight_sha256="2" * 64,
        authorization_sha256="4" * 64,
        authorization_use_sha256=hashlib.sha256(
            paths.authorization_use.read_bytes()
        ).hexdigest(),
        transitions=[first, second],
        failure_category="private_evidence_unavailable",
    )
    paired.write_terminal_record(paths.terminal_record, terminal)

    with pytest.raises(ValueError, match="not exclusive"):
        _validate_pair_terminal_exclusivity(
            paths,
            authorization_consumed=True,
            execution_complete=True,
        )


# 功能：验证reservation前或执行未terminal时两种terminal都不存在是合法pending状态
# 设计：使用无任何evidence的mechanical paths调用同一XOR validator，避免把尚未开始误判为损坏
def test_pair_terminal_exclusivity_allows_preterminal_pending(tmp_path: Path) -> None:
    paths = _paired().derive_private_evidence_paths(
        tmp_path.resolve(),
        _receipt().receipt_id,
    )

    assert (
        _paired().validate_pair_terminal_exclusivity(
            paths,
            authorization_consumed=False,
            execution_complete=False,
        )
        == "pending"
    )
    _paired().reserve_authorization_use(paths.authorization_use, _use_record())
    assert (
        _paired().validate_pair_terminal_exclusivity(
            paths,
            authorization_consumed=True,
            execution_complete=False,
        )
        == "pending"
    )
    with pytest.raises(ValueError, match="not exclusive"):
        _paired().validate_pair_terminal_exclusivity(
            paths,
            authorization_consumed=True,
            execution_complete=True,
        )


# 使用review-remediation API执行一次pair并返回直接绑定output parent的private paths
def _execute_with_terminal_contract(
    tmp_path: Path,
    child: object,
    *,
    observe_between: object | None = None,
    between_expected: object | None = None,
    script_module: ModuleType | None = None,
) -> tuple[object, object]:
    paired = _paired()
    script = _script() if script_module is None else script_module
    receipt = _receipt()
    preflight, authorization = _execution_artifacts(receipt)
    paths = paired.derive_private_evidence_paths(
        tmp_path.resolve(),
        receipt.receipt_id,
    )
    summary = script.execute_control_first(
        repository=_repository(),
        receipt=receipt,
        preflight=preflight,
        authorization=authorization,
        receipt_sha256=_receipt_sha256(),
        preflight_commit="a" * 40,
        preflight_sha256="2" * 64,
        authorization_commit="b" * 40,
        authorization_sha256="4" * 64,
        authorization_use_record=_use_record().model_copy(
            update={
                "authorization_sha256": "4" * 64,
                "paired_receipt_sha256": _receipt_sha256(),
            }
        ),
        receipt_path=_receipt_path(),
        private_paths=paths,
        control=script.ArmLaunch(
            expected=_expected_arm(receipt, "control"),
            argv=("fake",),
            cwd=tmp_path,
            env={},
            output_root=tmp_path / "control",
            stdout_path=paths.control_stdout,
            stderr_path=paths.control_stderr,
        ),
        treatment=script.ArmLaunch(
            expected=_expected_arm(receipt, "treatment"),
            argv=("fake",),
            cwd=tmp_path,
            env={},
            output_root=tmp_path / "treatment",
            stdout_path=paths.treatment_stdout,
            stderr_path=paths.treatment_stderr,
        ),
        between_expected=(
            _between_arm() if between_expected is None else between_expected
        ),
        observe_between=(
            (lambda: _between_arm())
            if observe_between is None
            else observe_between
        ),
        result_id="phase9d-pair-test",
        created_at_utc="2026-08-01T00:00:00Z",
        child_runner=child,
    )
    return summary, paths


# 功能：验证完整C1 artifact审计为INVALID后仍写唯一durable terminal evidence
# 设计：fake child写缺一row的真实baseline，断言TERMINAL、固定phase、无C2和create-once record
def test_control_invalid_writes_durable_terminal_record(tmp_path: Path) -> None:
    receipt = _receipt()
    calls = 0

    # 只写不完整C1 evidence以触发control_audit INVALID
    def child(launch: object, _cancel: threading.Event | None) -> object:
        nonlocal calls
        calls += 1
        _write_arm_output(
            launch.output_root,
            receipt,
            "control",
            _attempts({task_id for task_id, _category in _TASKS})[:-1],
        )
        return _script().ChildResult(True, 0, None, False, 123, None)

    summary, paths = _execute_with_terminal_contract(tmp_path, child)

    assert calls == 1
    assert summary.state is _paired().PairState.TERMINAL
    terminal = _paired().read_strict_artifact(
        paths.terminal_record,
        _paired().PairTerminalRecord,
    )
    assert terminal.phase == "control_audit"
    assert terminal.status == "INVALID"
    assert terminal.capability_delta_published is False
    assert terminal.transitions[-1].to_state == "TERMINAL"
    assert not (tmp_path / "treatment").exists()


# 功能：验证between-arm漂移只能经reducer进入TERMINAL且不能声称C2运行
# 设计：C1写完整证据后单改credential flag，检查history事件、terminal phase和treatment child缺失
def test_between_arm_drift_terminalizes_through_state_machine(tmp_path: Path) -> None:
    receipt = _receipt()
    calls = 0

    # 仅允许control执行并写完整canonical single-arm evidence
    def child(launch: object, _cancel: threading.Event | None) -> object:
        nonlocal calls
        calls += 1
        _write_arm_output(
            launch.output_root,
            receipt,
            "control",
            _attempts({task_id for task_id, _category in _TASKS}),
        )
        return _script().ChildResult(True, 0, None, False, 123, None)

    drift = _between_arm().model_copy(update={"credential_present": False})
    summary, paths = _execute_with_terminal_contract(
        tmp_path,
        child,
        observe_between=lambda: drift,
    )

    terminal = _paired().read_strict_artifact(
        paths.terminal_record,
        _paired().PairTerminalRecord,
    )
    assert calls == 1
    assert summary.state is _paired().PairState.TERMINAL
    assert terminal.phase == "between_arms"
    assert terminal.treatment_child is None
    assert terminal.transitions[-1].event == "BETWEEN_ARM_INVALID"


# 功能：验证authorization use先于private root创建，mkdir失败后仍可证明授权已消费
# 设计：预占固定private root模拟mkdir失败，断言parent级use与terminal同时持久且没有child
def test_private_root_collision_after_reservation_is_terminal_invalid(
    tmp_path: Path,
) -> None:
    calls = 0
    receipt = _receipt()
    paths = _paired().derive_private_evidence_paths(
        tmp_path.resolve(),
        receipt.receipt_id,
    )
    paths.root.mkdir()

    # 若private root gate错误地继续，此child会暴露调用
    def child(_launch: object, _cancel: threading.Event | None) -> object:
        nonlocal calls
        calls += 1
        return _script().ChildResult(True, 0, None, False, 123, None)

    summary, observed_paths = _execute_with_terminal_contract(tmp_path, child)

    assert calls == 0
    assert summary.state is _paired().PairState.TERMINAL
    assert observed_paths.authorization_use.is_file()
    assert observed_paths.terminal_record.is_file()
    terminal = _paired().read_strict_artifact(
        observed_paths.terminal_record,
        _paired().PairTerminalRecord,
    )
    assert terminal.phase == "control_spawn"
    assert terminal.failure_category == "private_evidence_unavailable"


# 功能：验证authorization link已发布但首次parent fsync失败时仍消费use并写durable INVALID terminal
# 设计：只让第一次目录fsync失败、后续terminal fsync恢复，锁定consumed=true分支不创建child或第三状态
def test_execute_parent_fsync_failure_terminalizes_consumed_authorization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paired = _paired()
    original_fsync = paired._fsync_directory
    fsync_calls = 0
    child_calls = 0

    # 仅破坏reservation publish后的第一次parent fsync
    def fail_first_fsync(path: Path) -> None:
        nonlocal fsync_calls
        fsync_calls += 1
        if fsync_calls == 1:
            raise ValueError("SECRET_PARENT_FSYNC_CANARY")
        original_fsync(path)

    # 若consumed failure错误继续执行，此child立即暴露调用
    def child(_launch: object, _cancel: threading.Event | None) -> object:
        nonlocal child_calls
        child_calls += 1
        return _script().ChildResult(True, 0, None, False, 123, None)

    monkeypatch.setattr(paired, "_fsync_directory", fail_first_fsync)
    summary, paths = _execute_with_terminal_contract(tmp_path, child)

    assert child_calls == 0
    assert summary.state is paired.PairState.TERMINAL
    assert paths.authorization_use.is_file()
    assert paths.terminal_record.is_file()
    assert not paths.root.exists()
    assert "SECRET_PARENT_FSYNC_CANARY" not in paths.terminal_record.read_text(
        encoding="utf-8"
    )


# 功能：验证C2 cancellation保留partial evidence并以treatment_audit terminal结束
# 设计：C1写完整报告、C2写partial marker并返回cancelled child，断言两个arm分支共享durable取消语义
def test_treatment_cancellation_writes_durable_terminal_record(tmp_path: Path) -> None:
    receipt = _receipt()
    calls = 0

    # 根据arm返回完整control或cancelled treatment
    def child(launch: object, _cancel: threading.Event | None) -> object:
        nonlocal calls
        calls += 1
        if launch.expected.arm == "control":
            _write_arm_output(
                launch.output_root,
                receipt,
                "control",
                _attempts({task_id for task_id, _category in _TASKS}),
            )
            return _script().ChildResult(True, 0, None, False, 123, None)
        launch.output_root.mkdir()
        (launch.output_root / "partial.marker").write_text(
            "partial\n",
            encoding="utf-8",
        )
        return _script().ChildResult(
            True,
            None,
            signal.SIGTERM,
            True,
            456,
            "cancelled",
        )

    summary, paths = _execute_with_terminal_contract(tmp_path, child)

    assert calls == 2
    assert summary.state is _paired().PairState.TERMINAL
    assert summary.terminal is not None
    assert summary.terminal.phase == "treatment_audit"
    assert summary.terminal.treatment_child is not None
    assert summary.terminal.treatment_child.cancelled is True
    assert (tmp_path / "treatment" / "partial.marker").is_file()
    assert paths.terminal_record.is_file()


# 功能：验证child环境只含冻结allowlist并覆盖source/home/tmp，拒绝所有无关secret与KAMA配置
# 设计：注入四类canary和污染PYTHONPATH，逐键断言真实env mapping而非仅扫描command spec
def test_child_environment_is_minimal_allowlist(tmp_path: Path) -> None:
    source = {
        "PATH": "/approved/bin",
        "LANG": "en_US.UTF-8",
        "LC_ALL": "C.UTF-8",
        "ANTHROPIC_API_KEY": "FAKE_APPROVED_CREDENTIAL",
        "AWS_SECRET_ACCESS_KEY": "AWS_SECRET_CANARY",
        "GITHUB_TOKEN": "GITHUB_SECRET_CANARY",
        "DATABASE_PASSWORD": "DATABASE_SECRET_CANARY",
        "KAMA_CONFIG": "KAMA_CONFIG_CANARY",
        "PYTHONPATH": "POLLUTED_SOURCE_CANARY",
    }
    source_root = tmp_path / "arm" / "src"
    home = tmp_path / "private" / "home"
    temp = tmp_path / "private" / "tmp"

    child = _script()._build_child_environment(
        source,
        credential_env="ANTHROPIC_API_KEY",
        source_root=source_root,
        home=home,
        tmpdir=temp,
    )

    assert child == {
        "ANTHROPIC_API_KEY": "FAKE_APPROVED_CREDENTIAL",
        "HOME": str(home),
        "LANG": "en_US.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PATH": "/approved/bin",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": str(source_root),
        "TMPDIR": str(temp),
    }
    serialized = repr(child)
    for canary in (
        "AWS_SECRET_CANARY",
        "GITHUB_SECRET_CANARY",
        "DATABASE_SECRET_CANARY",
        "KAMA_CONFIG_CANARY",
        "POLLUTED_SOURCE_CANARY",
    ):
        assert canary not in serialized


# 功能：验证ReceiptExecutionStateMachine源码字段集合精确且每个字段只声明一次
# 设计：AST读取唯一允许的production文件，防止Python/Pydantic静默覆盖重复annotation
def test_receipt_state_machine_has_unique_exact_source_fields() -> None:
    path = Path(__file__).resolve().parents[2] / "src" / "kama_claude" / "benchmark" / "paired.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "ReceiptExecutionStateMachine"
    )
    fields = [
        node.target.id
        for node in class_node.body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    ]

    assert fields == [
        "control_preflight_failed",
        "control_started_then_invalid_or_incomplete",
        "control_valid_and_complete",
        "treatment_started_then_invalid_or_incomplete",
        "both_arms_valid_and_complete",
    ]
    assert len(fields) == len(set(fields))


# 功能：验证worktree evidence只接受真实import hashes而不保留可伪造boolean attestation
# 设计：直接检查strict model公开字段，删除probe调用或只写source_import_matches都会破坏合同
def test_worktree_models_have_no_boolean_source_import_attestation() -> None:
    paired = _paired()

    assert "source_import_matches" not in paired.WorktreeEvidence.model_fields
    assert "source_import_matches" not in paired.WorktreeObservation.model_fields
    assert "source_import" in paired.WorktreeEvidence.model_fields
    assert "source_import" in paired.WorktreeObservation.model_fields


# 功能：验证child明确报告spawned=false时记录control_spawn终态而非伪装成artifact audit失败
# 设计：不创建任何arm output，检查固定phase/category、一次调用与direct-parent terminal文件
def test_control_spawn_failure_has_durable_terminal_evidence(tmp_path: Path) -> None:
    calls = 0

    # 返回真实child adapter使用的脱敏spawn failure结构
    def child(_launch: object, _cancel: threading.Event | None) -> object:
        nonlocal calls
        calls += 1
        return _script().ChildResult(False, None, None, False, None, "spawn_failed")

    summary, paths = _execute_with_terminal_contract(tmp_path, child)

    assert calls == 1
    assert summary.state is _paired().PairState.TERMINAL
    assert summary.terminal is not None
    assert summary.terminal.phase == "control_spawn"
    assert summary.terminal.failure_category == "control_spawn_failed"
    assert paths.terminal_record.is_file()


# 功能：验证C2 spawn异常在保留VALID C1 audit后写treatment_spawn terminal
# 设计：第一次child写完整C1，第二次抛固定OSError，断言C2未产生audit且history终结
def test_treatment_spawn_failure_has_durable_terminal_evidence(tmp_path: Path) -> None:
    receipt = _receipt()
    calls = 0

    # 第一次返回完整control，第二次模拟spawn边界异常
    def child(launch: object, _cancel: threading.Event | None) -> object:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("SECRET_TREATMENT_SPAWN_CANARY")
        _write_arm_output(
            launch.output_root,
            receipt,
            "control",
            _attempts({task_id for task_id, _category in _TASKS}),
        )
        return _script().ChildResult(True, 0, None, False, 123, None)

    summary, paths = _execute_with_terminal_contract(tmp_path, child)

    assert calls == 2
    assert summary.state is _paired().PairState.TERMINAL
    assert summary.control is not None and summary.control.status == "VALID"
    assert summary.treatment is None
    assert summary.terminal is not None
    assert summary.terminal.phase == "treatment_spawn"
    assert "SECRET_TREATMENT_SPAWN_CANARY" not in paths.terminal_record.read_text(
        encoding="utf-8"
    )


# 功能：验证C2 artifact INVALID后写treatment_audit terminal且不发布capability delta
# 设计：C1写完整matrix、C2缺一row，断言两个child已运行但最终证据为INVALID/private
def test_treatment_invalid_has_durable_terminal_evidence(tmp_path: Path) -> None:
    receipt = _receipt()
    calls = 0

    # 根据arm写完整control与不完整treatment evidence
    def child(launch: object, _cancel: threading.Event | None) -> object:
        nonlocal calls
        calls += 1
        attempts = _attempts({task_id for task_id, _category in _TASKS})
        if launch.expected.arm == "treatment":
            attempts = attempts[:-1]
        _write_arm_output(launch.output_root, receipt, launch.expected.arm, attempts)
        return _script().ChildResult(True, 0, None, False, 123, None)

    summary, _paths = _execute_with_terminal_contract(tmp_path, child)

    assert calls == 2
    assert summary.state is _paired().PairState.TERMINAL
    assert summary.treatment is not None and summary.treatment.status == "INVALID"
    assert summary.terminal is not None
    assert summary.terminal.phase == "treatment_audit"
    assert summary.terminal.capability_delta_published is False


# 功能：验证paired-result create-once写失败后另写direct-parent terminal且不留下能力结论
# 设计：两arm均VALID后仅注入result writer异常，检查RESULT_WRITE_FAILED history与redaction
def test_paired_result_write_failure_has_durable_terminal_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt = _receipt()

    # 为两个arm都写完整canonical evidence
    def child(launch: object, _cancel: threading.Event | None) -> object:
        _write_arm_output(
            launch.output_root,
            receipt,
            launch.expected.arm,
            _attempts({task_id for task_id, _category in _TASKS}),
        )
        return _script().ChildResult(True, 0, None, False, 123, None)

    script = _script()

    # 只破坏最终paired bundle writer，不影响classification
    def fail_result_write(
        _path: Path,
        _result: object,
        **_kwargs: object,
    ) -> None:
        raise OSError("SECRET_RESULT_WRITE_CANARY")

    monkeypatch.setattr(script, "write_paired_result", fail_result_write)
    paired = _paired()
    receipt = _receipt()
    preflight, authorization = _execution_artifacts(receipt)
    paths = paired.derive_private_evidence_paths(tmp_path.resolve(), receipt.receipt_id)
    summary = script.execute_control_first(
        repository=_repository(),
        receipt=receipt,
        preflight=preflight,
        authorization=authorization,
        receipt_sha256=_receipt_sha256(),
        preflight_commit="a" * 40,
        preflight_sha256="2" * 64,
        authorization_commit="b" * 40,
        authorization_sha256="4" * 64,
        authorization_use_record=_use_record().model_copy(
            update={
                "authorization_sha256": "4" * 64,
                "paired_receipt_sha256": _receipt_sha256(),
            }
        ),
        receipt_path=_receipt_path(),
        private_paths=paths,
        control=script.ArmLaunch(
            expected=_expected_arm(receipt, "control"),
            argv=("fake",),
            cwd=tmp_path,
            env={},
            output_root=tmp_path / "control",
            stdout_path=paths.control_stdout,
            stderr_path=paths.control_stderr,
        ),
        treatment=script.ArmLaunch(
            expected=_expected_arm(receipt, "treatment"),
            argv=("fake",),
            cwd=tmp_path,
            env={},
            output_root=tmp_path / "treatment",
            stdout_path=paths.treatment_stdout,
            stderr_path=paths.treatment_stderr,
        ),
        between_expected=_between_arm(),
        observe_between=lambda: _between_arm(),
        result_id="phase9d-pair-test",
        created_at_utc="2026-08-01T00:00:00Z",
        child_runner=child,
    )

    assert summary.state is paired.PairState.TERMINAL
    assert summary.result is None
    assert summary.terminal is not None
    assert summary.terminal.phase == "paired_result_write"
    assert summary.terminal.transitions[-1].event == "RESULT_WRITE_FAILED"
    assert "SECRET_RESULT_WRITE_CANARY" not in paths.terminal_record.read_text(
        encoding="utf-8"
    )


# 功能：验证两arm有效但classifier异常时写paired_classification terminal而不发布capability delta
# 设计：保留真实outcome构建并仅让receipt-driven classifier抛固定错误，隔离分类阶段终态和redaction
def test_paired_classification_failure_has_durable_terminal_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt = _receipt()

    # 为两个arm都写完整canonical evidence
    def child(launch: object, _cancel: threading.Event | None) -> object:
        _write_arm_output(
            launch.output_root,
            receipt,
            launch.expected.arm,
            _attempts({task_id for task_id, _category in _TASKS}),
        )
        return _script().ChildResult(True, 0, None, False, 123, None)

    # 仅破坏分类器以验证BOTH_VALID后的fail-closed terminal路径
    def fail_classification(*_args: object, **_kwargs: object) -> object:
        raise ValueError("SECRET_CLASSIFIER_CANARY")

    script = _script()
    monkeypatch.setattr(script, "recompute_classification", fail_classification)
    summary, paths = _execute_with_terminal_contract(
        tmp_path,
        child,
        script_module=script,
    )

    assert summary.state is _paired().PairState.TERMINAL
    assert summary.result is None
    assert summary.terminal is not None
    assert summary.terminal.phase == "paired_classification"
    assert summary.terminal.transitions[-1].event == "CLASSIFICATION_FAILED"
    assert summary.terminal.capability_delta_published is False
    assert "SECRET_CLASSIFIER_CANARY" not in paths.terminal_record.read_text(
        encoding="utf-8"
    )


# 功能：验证failure terminal拒绝success FINALIZED event即使reducer history本身完全合法
# 设计：使用BOTH_VALID到FINALIZED的canonical history与完整audits，仅制造artifact-kind/event矛盾
def test_terminal_record_rejects_success_finalized_event() -> None:
    paired = _paired()
    payload = {
        "schema_version": 1,
        "terminal_id": "wrong-success-event-terminal",
        "created_at_utc": "2026-08-01T00:00:00Z",
        "status": "INVALID",
        "phase": "paired_result_write",
        "receipt_sha256": "1" * 64,
        "preflight_sha256": "2" * 64,
        "authorization_sha256": "3" * 64,
        "authorization_use_sha256": "4" * 64,
        "transitions": [
            row.model_dump(mode="json") for row in _successful_transition_history()
        ],
        "control": _valid_arm_audit("control").model_dump(mode="json"),
        "treatment": _valid_arm_audit("treatment").model_dump(mode="json"),
        "control_child": None,
        "treatment_child": None,
        "provider_call_count": 162,
        "capability_delta_published": False,
        "private_visibility": True,
        "failure_category": "paired_result_write_failed",
    }

    with pytest.raises(ValidationError, match="failure terminal"):
        paired.PairTerminalRecord.model_validate(payload)


# 功能：验证canonical failure mapping覆盖每个正式phase/category且builder产物通过strict model
# 设计：用手工reducer路径和phase-compatible audits构造全部十类terminal，production helper提供唯一event
@pytest.mark.parametrize(
    ("phase", "category", "event"),
    [
        ("control_spawn", "private_evidence_unavailable", "PRIVATE_EVIDENCE_FAILED"),
        ("control_spawn", "control_spawn_failed", "CONTROL_SPAWN_FAILED"),
        ("control_audit", "control_invalid", "CONTROL_INVALID"),
        ("between_arms", "between_arm_invalid", "BETWEEN_ARM_INVALID"),
        ("treatment_spawn", "treatment_spawn_failed", "TREATMENT_SPAWN_FAILED"),
        ("treatment_audit", "treatment_invalid", "TREATMENT_INVALID"),
        ("paired_classification", "paired_classification_failed", "CLASSIFICATION_FAILED"),
        ("paired_result_write", "paired_result_write_failed", "RESULT_WRITE_FAILED"),
        ("parent_interrupt", "parent_interrupted", "PARENT_INTERRUPTED"),
        ("parent_interrupt", "parent_system_exit", "PARENT_SYSTEM_EXIT"),
    ],
)
def test_canonical_failure_mapping_builds_valid_terminal_records(
    phase: str,
    category: str,
    event: str,
) -> None:
    paired = _paired()
    control, treatment = _terminal_audits_for_phase(phase)

    assert paired.terminal_event_for_failure(phase, category).value == event
    terminal = paired.build_terminal_record(
        terminal_id=f"{category}-terminal",
        created_at_utc="2026-08-01T00:00:00Z",
        phase=phase,
        receipt_sha256="1" * 64,
        preflight_sha256="2" * 64,
        authorization_sha256="3" * 64,
        authorization_use_sha256="4" * 64,
        transitions=_failure_transition_history(event),
        failure_category=category,
        control=control,
        treatment=treatment,
    )

    assert terminal.status == "INVALID"
    assert terminal.transitions[-1].event == event


# 功能：验证phase/category/event任一不兼容时failure terminal fail closed
# 设计：从canonical control-spawn terminal只改category，保持event/history合法以隔离mapping validator
def test_terminal_record_rejects_failure_mapping_mismatch() -> None:
    paired = _paired()
    payload = {
        "schema_version": 1,
        "terminal_id": "failure-mapping-mismatch",
        "created_at_utc": "2026-08-01T00:00:00Z",
        "status": "INVALID",
        "phase": "control_spawn",
        "receipt_sha256": "1" * 64,
        "preflight_sha256": "2" * 64,
        "authorization_sha256": "3" * 64,
        "authorization_use_sha256": "4" * 64,
        "transitions": [
            row.model_dump(mode="json")
            for row in _failure_transition_history("CONTROL_SPAWN_FAILED")
        ],
        "control": None,
        "treatment": None,
        "control_child": None,
        "treatment_child": None,
        "provider_call_count": 0,
        "capability_delta_published": False,
        "private_visibility": True,
        "failure_category": "private_evidence_unavailable",
    }

    with pytest.raises(ValidationError, match="failure terminal"):
        paired.PairTerminalRecord.model_validate(payload)


# 功能：验证failure terminal不能携带capability status、published=true或none category
# 设计：从同一canonical failure payload逐字段mutation，锁定failure artifact的固定schema边界
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("status", "ACCEPT"),
        ("capability_delta_published", True),
        ("failure_category", "none"),
    ],
)
def test_terminal_record_rejects_capability_semantics(
    field: str,
    value: object,
) -> None:
    payload = {
        "schema_version": 1,
        "terminal_id": "failure-schema-boundary",
        "created_at_utc": "2026-08-01T00:00:00Z",
        "status": "INVALID",
        "phase": "control_spawn",
        "receipt_sha256": "1" * 64,
        "preflight_sha256": "2" * 64,
        "authorization_sha256": "3" * 64,
        "authorization_use_sha256": "4" * 64,
        "transitions": [
            row.model_dump(mode="json")
            for row in _failure_transition_history("CONTROL_SPAWN_FAILED")
        ],
        "control": None,
        "treatment": None,
        "control_child": None,
        "treatment_child": None,
        "provider_call_count": 0,
        "capability_delta_published": False,
        "private_visibility": True,
        "failure_category": "control_spawn_failed",
    }
    payload[field] = value

    with pytest.raises(ValidationError):
        _paired().PairTerminalRecord.model_validate(payload)


# 功能：验证failure terminal的audit presence不能超前于声明phase
# 设计：在control_spawn terminal注入完整control audit，event/category保持canonical以隔离phase兼容性
def test_terminal_record_rejects_incompatible_audit_presence() -> None:
    paired = _paired()
    payload = {
        "schema_version": 1,
        "terminal_id": "failure-audit-presence",
        "created_at_utc": "2026-08-01T00:00:00Z",
        "status": "INVALID",
        "phase": "control_spawn",
        "receipt_sha256": "1" * 64,
        "preflight_sha256": "2" * 64,
        "authorization_sha256": "3" * 64,
        "authorization_use_sha256": "4" * 64,
        "transitions": [
            row.model_dump(mode="json")
            for row in _failure_transition_history("CONTROL_SPAWN_FAILED")
        ],
        "control": _valid_arm_audit("control").model_dump(mode="json"),
        "treatment": None,
        "control_child": None,
        "treatment_child": None,
        "provider_call_count": 81,
        "capability_delta_published": False,
        "private_visibility": True,
        "failure_category": "control_spawn_failed",
    }

    with pytest.raises(ValidationError, match="audit presence"):
        paired.PairTerminalRecord.model_validate(payload)


# 功能：验证failure reader拒绝canonical重写后使用FINALIZED的terminal artifact
# 设计：直接写production canonical JSON而不损坏encoding/hash，确保reader因artifact semantics拒绝
def test_terminal_reader_rejects_canonical_success_event_artifact(
    tmp_path: Path,
) -> None:
    paired = _paired()
    payload = {
        "schema_version": 1,
        "terminal_id": "canonical-wrong-terminal",
        "created_at_utc": "2026-08-01T00:00:00Z",
        "status": "INVALID",
        "phase": "paired_result_write",
        "receipt_sha256": "1" * 64,
        "preflight_sha256": "2" * 64,
        "authorization_sha256": "3" * 64,
        "authorization_use_sha256": "4" * 64,
        "transitions": [
            row.model_dump(mode="json") for row in _successful_transition_history()
        ],
        "control": _valid_arm_audit("control").model_dump(mode="json"),
        "treatment": _valid_arm_audit("treatment").model_dump(mode="json"),
        "control_child": None,
        "treatment_child": None,
        "provider_call_count": 162,
        "capability_delta_published": False,
        "private_visibility": True,
        "failure_category": "paired_result_write_failed",
    }
    path = tmp_path / "terminal.json"
    path.write_text(paired.canonical_json(payload) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="invalid canonical artifact"):
        paired.read_terminal_record(path)


# 功能：验证terminal schema拒绝断裂或跳过reducer状态的transition history
# 设计：先取得真实control-invalid terminal，再只篡改第二条from_state并严格重建model
def test_terminal_record_rejects_discontinuous_transition_history(
    tmp_path: Path,
) -> None:
    # 产生最早control spawn failure以获得真实canonical terminal fixture
    def child(_launch: object, _cancel: threading.Event | None) -> object:
        return _script().ChildResult(False, None, None, False, None, "spawn_failed")

    summary, _paths = _execute_with_terminal_contract(tmp_path, child)
    assert summary.terminal is not None
    payload = summary.terminal.model_dump(mode="json")
    payload["transitions"][1]["from_state"] = "BOTH_VALID"

    with pytest.raises(ValidationError, match="transition"):
        _paired().PairTerminalRecord.model_validate(payload)


# 功能：验证terminal history拒绝连续但不属于canonical reducer的伪造跳转
# 设计：保持from/to连续和最终TERMINAL，仅把中间三元组改成不存在的转换以区别于普通断链测试
def test_terminal_record_rejects_contiguous_impossible_transition_history(
    tmp_path: Path,
) -> None:
    # 产生真实control spawn terminal作为strict model mutation基线
    def child(_launch: object, _cancel: threading.Event | None) -> object:
        return _script().ChildResult(False, None, None, False, None, "spawn_failed")

    summary, _paths = _execute_with_terminal_contract(tmp_path, child)
    assert summary.terminal is not None
    payload = summary.terminal.model_dump(mode="json")
    payload["transitions"][1] = {
        "from_state": "AUTHORIZATION_RESERVED",
        "event": "CONTROL_STARTED",
        "to_state": "CONTROL_VALID",
    }
    payload["transitions"][2]["from_state"] = "CONTROL_VALID"

    with pytest.raises(ValidationError, match="transition"):
        _paired().PairTerminalRecord.model_validate(payload)


# 功能：验证orchestrator源码不直接构造CONTROL_INVALID/TREATMENT_INVALID绕过reducer
# 设计：AST扫描PairState attribute使用，允许初始化NOT_STARTED和从reducer结果转换，不做文本grep
def test_orchestrator_does_not_assign_intermediate_invalid_states_directly() -> None:
    path = Path(__file__).resolve().parents[2] / "scripts" / "phase9d_paired.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    invalid_attributes = [
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "PairState"
        and node.attr in {"CONTROL_INVALID", "TREATMENT_INVALID"}
    ]

    assert invalid_attributes == []


# 功能：验证execute CLI对durable INVALID terminal返回2，仅成功paired result返回0
# 设计：在CLI最外层注入两种已验证summary，隔离退出码不能只看TERMINAL state的回归
@pytest.mark.parametrize(("has_result", "expected"), [(False, 2), (True, 0)])
def test_execute_cli_exit_code_distinguishes_invalid_terminal_from_result(
    monkeypatch: pytest.MonkeyPatch,
    has_result: bool,
    expected: int,
) -> None:
    script = _script()
    summary = script.PairExecutionSummary(
        state=_paired().PairState.TERMINAL,
        control=None,
        treatment=None,
        result=object() if has_result else None,
        terminal=None if has_result else object(),
    )
    monkeypatch.setattr(script, "execute_from_artifacts", lambda _args: summary)

    exit_code = script._main(
        [
            "execute",
            "--receipt",
            "receipt.json",
            "--preflight",
            "preflight.json",
            "--authorization",
            "authorization.json",
            "--repository",
            "repo",
            "--control-worktree",
            "control",
            "--treatment-worktree",
            "treatment",
        ]
    )

    assert exit_code == expected


# 构造会写完整arm artifacts的真实fake child，供parent interruption边界复用
def _complete_fake_child(receipt: object) -> Callable[[object, object], object]:
    # 根据launch arm写完整canonical single-arm evidence
    def child(launch: object, _cancel: object) -> object:
        _write_arm_output(
            launch.output_root,
            receipt,
            launch.expected.arm,
            _attempts({task_id for task_id, _category in _TASKS}),
        )
        return _script().ChildResult(True, 0, None, False, 123, None)

    return child


# 功能：验证reservation后任一执行phase的KeyboardInterrupt都先写INVALID terminal再传播
# 设计：只在目标production boundary抛真实BaseException，其他phase走完整arm artifacts和reducer
@pytest.mark.parametrize(
    "phase",
    [
        "authorization_publication",
        "private_evidence",
        "control_started_transition",
        "control_root_check",
        "control_spawn",
        "control_audit",
        "control_valid_transition",
        "between_arms",
        "treatment_started_transition",
        "treatment_spawn",
        "treatment_audit",
        "treatment_valid_transition",
        "paired_classification",
        "paired_result_build",
        "paired_result_publication",
    ],
)
def test_parent_keyboard_interrupt_terminalizes_every_reserved_phase(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
) -> None:
    script = _script()
    paired = _paired()
    receipt = _receipt()
    complete_child = _complete_fake_child(receipt)
    audit_calls = 0
    child_calls = 0
    original_audit = script.audit_arm_result
    original_reduce = script.reduce_pair_transition
    original_mkdir = Path.mkdir
    original_lexists = os.path.lexists
    original_fsync_directory = paired._fsync_directory
    paths = paired.derive_private_evidence_paths(tmp_path.resolve(), receipt.receipt_id)
    interrupt_injected = False

    # 在private root首次创建前注入parent interrupt且允许terminal writer继续
    def mkdir(
        path: Path,
        mode: int = 0o777,
        parents: bool = False,
        exist_ok: bool = False,
    ) -> None:
        nonlocal interrupt_injected
        if phase == "private_evidence" and path == paths.root and not interrupt_injected:
            interrupt_injected = True
            raise KeyboardInterrupt
        original_mkdir(path, mode=mode, parents=parents, exist_ok=exist_ok)

    transition_events = {
        "control_started_transition": paired.PairEvent.CONTROL_STARTED,
        "control_valid_transition": paired.PairEvent.CONTROL_VALID,
        "treatment_started_transition": paired.PairEvent.TREATMENT_STARTED,
        "treatment_valid_transition": paired.PairEvent.TREATMENT_VALID,
    }

    # 在目标共享reducer transition首次调用时注入parent interrupt
    def reduce(state: object, event: object, **kwargs: object) -> object:
        nonlocal interrupt_injected
        if phase in transition_events and event is transition_events[phase] and not interrupt_injected:
            interrupt_injected = True
            raise KeyboardInterrupt
        return original_reduce(state, event, **kwargs)

    # 在authorization已消费后的control root race检查注入parent interrupt
    def lexists(path: object) -> bool:
        nonlocal interrupt_injected
        if (
            phase == "control_root_check"
            and Path(path) == tmp_path / "control"
            and paths.authorization_use.is_file()
            and not interrupt_injected
        ):
            interrupt_injected = True
            raise KeyboardInterrupt
        return original_lexists(path)

    # 在use record已link但reserve尚未返回时注入父进程中断
    def fsync_directory(path: Path) -> None:
        nonlocal interrupt_injected
        if (
            phase == "authorization_publication"
            and path == paths.authorization_use.parent
            and paths.authorization_use.is_file()
            and not interrupt_injected
        ):
            interrupt_injected = True
            raise KeyboardInterrupt
        original_fsync_directory(path)

    # 在目标arm phase抛KeyboardInterrupt，其余child仍写完整evidence
    def child(launch: object, cancel: object) -> object:
        nonlocal child_calls
        child_calls += 1
        if phase == "control_spawn" and launch.expected.arm == "control":
            raise KeyboardInterrupt
        if phase == "treatment_spawn" and launch.expected.arm == "treatment":
            raise KeyboardInterrupt
        return complete_child(launch, cancel)

    # 在指定audit调用处抛KeyboardInterrupt并保留其他真实audit
    def audit(**kwargs: object) -> object:
        nonlocal audit_calls
        audit_calls += 1
        if phase == "control_audit" and audit_calls == 1:
            raise KeyboardInterrupt
        if phase == "treatment_audit" and audit_calls == 2:
            raise KeyboardInterrupt
        return original_audit(**kwargs)

    monkeypatch.setattr(script, "audit_arm_result", audit)
    monkeypatch.setattr(Path, "mkdir", mkdir)
    monkeypatch.setattr(script, "reduce_pair_transition", reduce)
    monkeypatch.setattr(os.path, "lexists", lexists)
    monkeypatch.setattr(paired, "_fsync_directory", fsync_directory)
    if phase == "paired_classification":
        monkeypatch.setattr(
            script,
            "recompute_classification",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(KeyboardInterrupt()),
        )
    if phase == "paired_result_publication":
        monkeypatch.setattr(
            script,
            "write_paired_result",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(KeyboardInterrupt()),
        )
    if phase == "paired_result_build":
        monkeypatch.setattr(
            script,
            "build_paired_result",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(KeyboardInterrupt()),
        )

    # 在between-arm barrier注入真实parent interruption
    def observe_between() -> object:
        if phase == "between_arms":
            raise KeyboardInterrupt
        return _between_arm()

    with pytest.raises(KeyboardInterrupt):
        _execute_with_terminal_contract(
            tmp_path,
            child,
            observe_between=observe_between,
            script_module=script,
        )

    terminal = paired.read_strict_artifact(
        paths.terminal_record,
        paired.PairTerminalRecord,
    )
    assert paths.authorization_use.is_file()
    assert terminal.status == "INVALID"
    assert terminal.phase == "parent_interrupt"
    assert terminal.failure_category == "parent_interrupted"
    assert terminal.transitions[-1].event == "PARENT_INTERRUPTED"
    assert terminal.capability_delta_published is False
    if phase in {
        "authorization_publication",
        "private_evidence",
        "control_started_transition",
        "control_root_check",
    }:
        expected_children = 0
    elif phase in {
        "control_spawn",
        "control_audit",
        "control_valid_transition",
        "between_arms",
        "treatment_started_transition",
    }:
        expected_children = 1
    else:
        expected_children = 2
    assert child_calls == expected_children
    with pytest.raises(ValueError, match="already reserved"):
        paired.reserve_authorization_use(paths.authorization_use, _use_record())


# 功能：验证reservation后SystemExit先写固定脱敏terminal再保留原整数exit code传播
# 设计：在真实control child boundary抛SystemExit(23)，拒绝任意object repr进入artifact
def test_parent_system_exit_terminalizes_before_propagation(tmp_path: Path) -> None:
    paired = _paired()
    receipt = _receipt()

    # 直接从post-reservation child boundary触发父级SystemExit
    def child(_launch: object, _cancel: object) -> object:
        raise SystemExit(23)

    paths = paired.derive_private_evidence_paths(tmp_path.resolve(), receipt.receipt_id)
    with pytest.raises(SystemExit) as exc_info:
        _execute_with_terminal_contract(tmp_path, child)

    assert exc_info.value.code == 23
    terminal = paired.read_strict_artifact(
        paths.terminal_record,
        paired.PairTerminalRecord,
    )
    assert terminal.status == "INVALID"
    assert terminal.phase == "parent_interrupt"
    assert terminal.failure_category == "parent_system_exit"
    assert terminal.transitions[-1].event == "PARENT_SYSTEM_EXIT"
    assert "23" not in paths.terminal_record.read_text(encoding="utf-8")


# 功能：验证success bundle提交后的parent fsync失败不会再生成failure terminal
# 设计：仅当production reader已接受bundle时注入目录fsync失败，精确覆盖publication commit point后边界
def test_post_commit_parent_fsync_warning_preserves_single_success_terminal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paired = _paired()
    receipt = _receipt()
    paths = paired.derive_private_evidence_paths(tmp_path.resolve(), receipt.receipt_id)
    original_fsync = paired._fsync_directory
    injected = False

    # 在success reader首次可接受bundle后才让目录fsync失败
    def fail_after_commit(path: Path) -> None:
        nonlocal injected
        try:
            _read_paired_result_bundle(paths.paired_result)
        except ValueError:
            original_fsync(path)
            return
        if not injected:
            injected = True
            raise ValueError("POST_COMMIT_DURABILITY_WARNING")
        original_fsync(path)

    monkeypatch.setattr(paired, "_fsync_directory", fail_after_commit)
    summary, observed_paths = _execute_with_terminal_contract(
        tmp_path,
        _complete_fake_child(receipt),
    )

    assert summary.result is not None
    assert summary.terminal is None
    assert summary.result_durability_warning is True
    assert paired.canonical_json(
        _read_paired_result_bundle(observed_paths.paired_result)
    ) == paired.canonical_json(summary.result)
    assert not observed_paths.terminal_record.exists()
    assert (
        _validate_pair_terminal_exclusivity(
            observed_paths,
            authorization_consumed=True,
            execution_complete=True,
        )
        == "success"
    )


# 功能：验证publisher每个rename前关键边界的KeyboardInterrupt都先形成唯一INVALID terminal再传播
# 设计：保留真实publisher，仅在JSON/Markdown/manifest/staging/validation/rename单点抛BaseException并检查reservation/XOR
@pytest.mark.parametrize(
    "boundary",
    ["json", "markdown", "manifest", "staging", "validation", "rename"],
)
def test_paired_result_precommit_keyboard_interrupt_terminalizes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    boundary: str,
) -> None:
    paired = _paired()
    script = _script()
    receipt = _receipt()
    paths = paired.derive_private_evidence_paths(tmp_path.resolve(), receipt.receipt_id)
    original_write = paired._write_private_bytes
    original_fsync = paired._fsync_directory
    original_reader = paired._read_paired_result_bundle
    original_rename = os.rename
    injected = False

    # 在目标bundle file首次write边界抛真实KeyboardInterrupt
    def interrupt_write(path: Path, payload: bytes) -> None:
        nonlocal injected
        target_kind = {
            "paired-result.json": "json",
            "paired-result.md": "markdown",
            "manifest.json": "manifest",
        }.get(path.name)
        if boundary == target_kind and not injected:
            injected = True
            raise KeyboardInterrupt
        original_write(path, payload)

    # 只在manifest齐全后的staging directory fsync注入一次
    def interrupt_staging(path: Path) -> None:
        nonlocal injected
        if (
            boundary == "staging"
            and path.name.endswith(".staging")
            and (path / "manifest.json").exists()
            and not injected
        ):
            injected = True
            raise KeyboardInterrupt
        original_fsync(path)

    # pre-rename strict reread首次调用时中断，后续final absence probe恢复
    def interrupt_validation(path: Path, **kwargs: object) -> object:
        nonlocal injected
        if boundary == "validation" and not injected:
            injected = True
            raise KeyboardInterrupt
        return original_reader(path, **kwargs)

    # atomic rename调用点中断一次且不创建final目录
    def interrupt_rename(source: object, target: object) -> None:
        nonlocal injected
        if boundary == "rename" and not injected:
            injected = True
            raise KeyboardInterrupt
        original_rename(source, target)

    monkeypatch.setattr(paired, "_write_private_bytes", interrupt_write)
    monkeypatch.setattr(paired, "_fsync_directory", interrupt_staging)
    monkeypatch.setattr(paired, "_read_paired_result_bundle", interrupt_validation)
    monkeypatch.setattr(os, "rename", interrupt_rename)

    with pytest.raises(KeyboardInterrupt):
        _execute_with_terminal_contract(
            tmp_path,
            _complete_fake_child(receipt),
            script_module=script,
        )

    terminal = paired.read_strict_artifact(
        paths.terminal_record,
        paired.PairTerminalRecord,
    )
    assert injected is True
    assert paths.authorization_use.is_file()
    assert terminal.status == "INVALID"
    assert terminal.failure_category == "parent_interrupted"
    assert terminal.transitions[-1].event == "PARENT_INTERRUPTED"
    assert not paths.paired_result.exists()
    assert (
        _validate_pair_terminal_exclusivity(
            paths,
            authorization_consumed=True,
            execution_complete=True,
        )
        == "failure"
    )


# 功能：验证atomic rename后的KeyboardInterrupt保留已提交success且绝不补写failure terminal
# 设计：reader首次接受final bundle后让parent fsync抛BaseException，检查传播后XOR仍为success
def test_post_commit_keyboard_interrupt_preserves_success_terminal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paired = _paired()
    receipt = _receipt()
    paths = paired.derive_private_evidence_paths(tmp_path.resolve(), receipt.receipt_id)
    original_fsync = paired._fsync_directory
    injected = False

    # 只在rename已提交且public reader可接受后中断parent durability acknowledgement
    def interrupt_after_commit(path: Path) -> None:
        nonlocal injected
        try:
            _read_paired_result_bundle(paths.paired_result)
        except ValueError:
            original_fsync(path)
            return
        if not injected:
            injected = True
            raise KeyboardInterrupt
        original_fsync(path)

    monkeypatch.setattr(paired, "_fsync_directory", interrupt_after_commit)

    with pytest.raises(KeyboardInterrupt):
        _execute_with_terminal_contract(
            tmp_path,
            _complete_fake_child(receipt),
        )

    assert injected is True
    assert _read_paired_result_bundle(paths.paired_result).capability_delta_published
    assert not paths.terminal_record.exists()
    assert (
        _validate_pair_terminal_exclusivity(
            paths,
            authorization_consumed=True,
            execution_complete=True,
        )
        == "success"
    )


# 功能：验证failure terminal writer自身失败时authorization仍消费且绝不发布success
# 设计：control spawn failure进入正式terminalize后只破坏writer，断言固定异常传播和one-use不可复用
def test_failure_terminal_write_failure_remains_consumed_and_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paired = _paired()
    script = _script()
    receipt = _receipt()
    paths = paired.derive_private_evidence_paths(tmp_path.resolve(), receipt.receipt_id)
    monkeypatch.setattr(
        script,
        "write_terminal_record",
        lambda *_args: (_ for _ in ()).throw(OSError("terminal write failed")),
    )

    # 返回真实spawned=false使production进入control_spawn terminal路径
    def child(_launch: object, _cancel: object) -> object:
        return script.ChildResult(False, None, None, False, None, "spawn_failed")

    with pytest.raises(OSError, match="terminal write failed"):
        _execute_with_terminal_contract(
            tmp_path,
            child,
            script_module=script,
        )

    assert paths.authorization_use.is_file()
    assert not paths.terminal_record.exists()
    assert not paths.paired_result.exists()
    with pytest.raises(ValueError, match="already reserved"):
        paired.reserve_authorization_use(paths.authorization_use, _use_record())


# 执行不继承secret的临时Git命令并返回stdout
def _temporary_git(repository: Path, *args: str) -> str:
    environment = {
        "GIT_AUTHOR_EMAIL": "phase9d@example.invalid",
        "GIT_AUTHOR_NAME": "Phase9D Test",
        "GIT_COMMITTER_EMAIL": "phase9d@example.invalid",
        "GIT_COMMITTER_NAME": "Phase9D Test",
        "PATH": os.environ.get("PATH", ""),
    }
    result = subprocess.run(
        ["git", *args],
        cwd=repository,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


# 创建三artifact分三commit的临时Git graph并冻结C1→C2 parent关系
def _fresh_git_artifact_fixture(tmp_path: Path) -> dict[str, object]:
    repository = tmp_path / "repository"
    repository.mkdir()
    _temporary_git(repository, "init", "-b", "approved")
    paths = {
        "receipt": repository / "receipt.json",
        "preflight": repository / "preflight.json",
        "authorization": repository / "authorization.json",
    }
    commits: dict[str, str] = {}
    hashes: dict[str, str] = {}
    for name, path in paths.items():
        payload = json.dumps(
            {"artifact": name, "schema_version": 1},
            sort_keys=True,
            separators=(",", ":"),
        ) + "\n"
        path.write_text(payload, encoding="utf-8")
        _temporary_git(repository, "add", path.name)
        _temporary_git(repository, "commit", "-m", name)
        commits[name] = _temporary_git(repository, "rev-parse", "HEAD")
        hashes[name] = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    approved_head = commits["authorization"]
    _temporary_git(
        repository,
        "update-ref",
        "refs/remotes/origin/approved",
        approved_head,
    )
    return {
        "repository": repository,
        "remote_ref": "refs/remotes/origin/approved",
        "approved_head": approved_head,
        "approved_branch": "approved",
        "paths": paths,
        "commits": commits,
        "hashes": hashes,
        "control_commit": commits["receipt"],
        "treatment_commit": commits["preflight"],
    }


# 调用production fresh Git/artifact capture并允许单点覆盖expected参数
def _capture_fresh_git_artifacts(
    fixture: dict[str, object],
    **overrides: object,
) -> object:
    values = {
        "repository": fixture["repository"],
        "remote_ref": fixture["remote_ref"],
        "approved_head": fixture["approved_head"],
        "approved_branch": fixture["approved_branch"],
        "receipt_path": fixture["paths"]["receipt"],
        "receipt_commit": fixture["commits"]["receipt"],
        "receipt_sha256": fixture["hashes"]["receipt"],
        "preflight_path": fixture["paths"]["preflight"],
        "preflight_commit": fixture["commits"]["preflight"],
        "preflight_sha256": fixture["hashes"]["preflight"],
        "authorization_path": fixture["paths"]["authorization"],
        "authorization_commit": fixture["commits"]["authorization"],
        "authorization_sha256": fixture["hashes"]["authorization"],
        "control_commit": fixture["control_commit"],
        "treatment_commit": fixture["treatment_commit"],
    }
    values.update(overrides)
    return _script().capture_between_arm_git_and_artifact_state(**values)


# 功能：验证fresh capture接受HEAD/branch/remote/artifacts/C1→C2全部匹配的真实Git状态
# 设计：三次真实commit和remote ref作为positive control，防止drift tests只验证固定reject
def test_between_arm_fresh_capture_accepts_approved_git_state(tmp_path: Path) -> None:
    fixture = _fresh_git_artifact_fixture(tmp_path)

    evidence = _capture_fresh_git_artifacts(fixture)

    assert evidence.git.commit == fixture["approved_head"]
    assert evidence.git.branch == "approved"
    assert evidence.receipt.sha256 == fixture["hashes"]["receipt"]
    assert evidence.treatment_parent_matches_control is True


# 功能：验证clean但HEAD已切到另一commit的main checkout被fresh between capture拒绝
# 设计：remote保持approved commit，另建clean commit后调用production capture，命中旧逻辑只看remote+dirty的缺口
def test_between_arm_fresh_capture_rejects_clean_head_switch(tmp_path: Path) -> None:
    fixture = _fresh_git_artifact_fixture(tmp_path)
    repository = fixture["repository"]
    (repository / "unrelated.txt").write_text("different head\n", encoding="utf-8")
    _temporary_git(repository, "add", "unrelated.txt")
    _temporary_git(repository, "commit", "-m", "other-head")

    with pytest.raises(ValueError, match="between-arm"):
        _capture_fresh_git_artifacts(fixture)


# 功能：验证相同commit但branch切换也被fresh between capture拒绝
# 设计：只创建并切换branch、不改变HEAD bytes，隔离旧逻辑未检查branch的finding
def test_between_arm_fresh_capture_rejects_branch_switch(tmp_path: Path) -> None:
    fixture = _fresh_git_artifact_fixture(tmp_path)
    _temporary_git(fixture["repository"], "switch", "-c", "other-branch")

    with pytest.raises(ValueError, match="between-arm"):
        _capture_fresh_git_artifacts(fixture)


# 功能：验证receipt/preflight/authorization任一worktree bytes漂移都会被fresh capture拒绝
# 设计：参数化只修改一个真实tracked file且不commit，证明observed hash来自fresh file/blob读取
@pytest.mark.parametrize("artifact", ["receipt", "preflight", "authorization"])
def test_between_arm_fresh_capture_rejects_artifact_worktree_drift(
    tmp_path: Path,
    artifact: str,
) -> None:
    fixture = _fresh_git_artifact_fixture(tmp_path)
    fixture["paths"][artifact].write_text("drift\n", encoding="utf-8")

    with pytest.raises(ValueError, match="between-arm"):
        _capture_fresh_git_artifacts(fixture)


# 功能：验证tracked artifact被同字节新inode替换时fresh identity也会变化
# 设计：先捕获approved evidence，再unlink并写回完全相同bytes，隔离仅比较内容hash的缺口
def test_between_arm_fresh_capture_observes_same_bytes_inode_replacement(
    tmp_path: Path,
) -> None:
    fixture = _fresh_git_artifact_fixture(tmp_path)
    before = _capture_fresh_git_artifacts(fixture)
    receipt_path = fixture["paths"]["receipt"]
    original = receipt_path.read_bytes()
    receipt_path.unlink()
    receipt_path.write_bytes(original)

    after = _capture_fresh_git_artifacts(fixture)

    assert after.receipt.sha256 == before.receipt.sha256
    assert after.receipt.canonical_object_sha256 != (
        before.receipt.canonical_object_sha256
    )


# 功能：验证三个artifact的expected commit/hash错误不能被fresh capture回显为observed
# 设计：分别单点替换commit或hash，要求production从Git blob重新计算后拒绝
@pytest.mark.parametrize(
    "overrides",
    [
        {"receipt_sha256": "0" * 64},
        {"preflight_commit": "0" * 40},
        {"authorization_sha256": "f" * 64},
    ],
)
def test_between_arm_fresh_capture_rejects_artifact_reference_drift(
    tmp_path: Path,
    overrides: dict[str, object],
) -> None:
    fixture = _fresh_git_artifact_fixture(tmp_path)

    with pytest.raises(ValueError, match="between-arm"):
        _capture_fresh_git_artifacts(fixture, **overrides)


# 功能：验证remote ref漂移或ahead/behind非零时fresh capture fail closed
# 设计：把remote回退到C2 commit但保持main HEAD不变，同时触发remote identity和count差异
def test_between_arm_fresh_capture_rejects_remote_and_count_drift(
    tmp_path: Path,
) -> None:
    fixture = _fresh_git_artifact_fixture(tmp_path)
    _temporary_git(
        fixture["repository"],
        "update-ref",
        fixture["remote_ref"],
        fixture["treatment_commit"],
    )

    with pytest.raises(ValueError, match="between-arm"):
        _capture_fresh_git_artifacts(fixture)


# 功能：验证C2非C1直接child的Git graph被fresh capture拒绝
# 设计：把authorization commit伪装为treatment，此时其parent是旧C2而不是control
def test_between_arm_fresh_capture_rejects_wrong_c2_parent(tmp_path: Path) -> None:
    fixture = _fresh_git_artifact_fixture(tmp_path)

    with pytest.raises(ValueError, match="between-arm"):
        _capture_fresh_git_artifacts(
            fixture,
            treatment_commit=fixture["commits"]["authorization"],
        )


# 功能：验证tracked artifact路径被symlink替换时fresh capture拒绝
# 设计：只替换receipt worktree entry为指向同bytes副本的symlink，排除内容hash掩盖路径攻击
def test_between_arm_fresh_capture_rejects_artifact_symlink(tmp_path: Path) -> None:
    fixture = _fresh_git_artifact_fixture(tmp_path)
    receipt = fixture["paths"]["receipt"]
    replacement = tmp_path / "replacement.json"
    replacement.write_bytes(receipt.read_bytes())
    receipt.unlink()
    receipt.symlink_to(replacement)

    with pytest.raises(ValueError, match="between-arm"):
        _capture_fresh_git_artifacts(fixture)


# 功能：验证每类fresh Git/artifact漂移都经正式orchestrator reducer阻断C2并写INVALID terminal
# 设计：真实capture建立expected，单点mutation后observe再次调用production capture，fake child只允许C1完整执行
@pytest.mark.parametrize(
    "mutation",
    [
        "clean_head_switch",
        "detached_head",
        "branch_switch",
        "remote_ref",
        "receipt_bytes",
        "preflight_bytes",
        "authorization_bytes",
        "receipt_reference",
        "preflight_reference",
        "authorization_reference",
        "c2_parent",
        "artifact_symlink",
        "same_bytes_inode",
    ],
)
def test_fresh_between_arm_mutation_blocks_c2_with_invalid_terminal(
    tmp_path: Path,
    mutation: str,
) -> None:
    paired = _paired()
    fixture_root = tmp_path / "git-fixture"
    fixture_root.mkdir()
    fixture = _fresh_git_artifact_fixture(fixture_root)
    initial = _capture_fresh_git_artifacts(fixture)
    expected = _between_arm().model_copy(
        update={"git_artifact_identity_sha256": paired.canonical_sha256(initial)}
    )
    overrides: dict[str, object] = {}
    repository = fixture["repository"]
    paths = fixture["paths"]
    if mutation == "clean_head_switch":
        (repository / "other.txt").write_text("other\n", encoding="utf-8")
        _temporary_git(repository, "add", "other.txt")
        _temporary_git(repository, "commit", "-m", "other-head")
    elif mutation == "detached_head":
        _temporary_git(repository, "checkout", "--detach", "HEAD")
    elif mutation == "branch_switch":
        _temporary_git(repository, "switch", "-c", "other-branch")
    elif mutation == "remote_ref":
        _temporary_git(
            repository,
            "update-ref",
            fixture["remote_ref"],
            fixture["treatment_commit"],
        )
    elif mutation.endswith("_bytes"):
        artifact = mutation.removesuffix("_bytes")
        paths[artifact].write_text("drift\n", encoding="utf-8")
    elif mutation == "receipt_reference":
        overrides["receipt_sha256"] = "0" * 64
    elif mutation == "preflight_reference":
        overrides["preflight_commit"] = "0" * 40
    elif mutation == "authorization_reference":
        overrides["authorization_sha256"] = "f" * 64
    elif mutation == "c2_parent":
        overrides["treatment_commit"] = fixture["commits"]["authorization"]
    elif mutation == "artifact_symlink":
        authorization = paths["authorization"]
        replacement = repository / "authorization-copy.json"
        replacement.write_bytes(authorization.read_bytes())
        authorization.unlink()
        authorization.symlink_to(replacement.name)
    elif mutation == "same_bytes_inode":
        receipt_path = paths["receipt"]
        original = receipt_path.read_bytes()
        receipt_path.unlink()
        receipt_path.write_bytes(original)

    # 重新调用production capture，异常或fresh hash变化都必须由between-arm guard处理
    def observe() -> object:
        current = _capture_fresh_git_artifacts(fixture, **overrides)
        return expected.model_copy(
            update={"git_artifact_identity_sha256": paired.canonical_sha256(current)}
        )

    calls = 0
    receipt = _receipt()

    # 只允许C1写完整evidence，若C2被错误启动则调用计数直接暴露
    def child(launch: object, _cancel: object) -> object:
        nonlocal calls
        calls += 1
        _write_arm_output(
            launch.output_root,
            receipt,
            launch.expected.arm,
            _attempts({task_id for task_id, _category in _TASKS}),
        )
        return _script().ChildResult(True, 0, None, False, 123, None)

    execution_root = tmp_path / "execution"
    execution_root.mkdir()
    summary, private_paths = _execute_with_terminal_contract(
        execution_root,
        child,
        observe_between=observe,
        between_expected=expected,
    )

    terminal = paired.read_strict_artifact(
        private_paths.terminal_record,
        paired.PairTerminalRecord,
    )
    assert calls == 1
    assert summary.result is None
    assert terminal.status == "INVALID"
    assert terminal.phase == "between_arms"
    assert terminal.transitions[-1].event == "BETWEEN_ARM_INVALID"
    assert terminal.treatment_child is None
    assert not (execution_root / "treatment").exists()
