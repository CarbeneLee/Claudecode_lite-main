from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import uuid
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path, PurePosixPath, PureWindowsPath
from statistics import median
from typing import Annotated, Any, Literal, cast, get_args

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    model_validator,
)

from kama_claude.benchmark.analyzers import AttemptAnalysis, aggregate_attempts
from kama_claude.benchmark.report import BaselineReport, render_json, render_markdown
from kama_claude.eval.failure import FailureCategory
from kama_claude.eval.report import (
    EvaluationReport,
)
from kama_claude.eval.report import (
    render_json as render_evaluation_json,
)
from kama_claude.eval.report import (
    render_markdown as render_evaluation_markdown,
)


class _StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        allow_inf_nan=False,
    )


Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
GitCommit = Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]
SafeIdentifier = Annotated[str, Field(min_length=1, pattern=r"^[A-Za-z0-9._-]+$")]


# 验证时间字符串可解析且显式使用Z表示UTC
def _validate_utc_timestamp(value: str) -> str:
    try:
        parsed = datetime.fromisoformat(value.removesuffix("Z") + "+00:00")
    except ValueError as exc:
        raise ValueError("timestamp must be valid RFC3339 UTC") from exc
    if not value.endswith("Z") or parsed.utcoffset() != UTC.utcoffset(parsed):
        raise ValueError("timestamp must be valid RFC3339 UTC")
    return value


# 验证artifact路径为规范POSIX仓库相对路径
def _validate_relative_artifact_path(value: str) -> str:
    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    if (
        not value
        or posix.is_absolute()
        or windows.is_absolute()
        or "\\" in value
        or any(part in {"", ".", ".."} for part in posix.parts)
        or posix.as_posix() != value
    ):
        raise ValueError("artifact path must be normalized and relative")
    return value


# 验证human authorization scope已经折叠首尾和重复空白
def _validate_normalized_scope(value: str) -> str:
    if not value or " ".join(value.split()) != value:
        raise ValueError("authorization scope must be normalized")
    return value


# 验证逻辑root只是一段不可逃逸的basename
def _validate_logical_basename_value(value: str) -> str:
    if (
        not value
        or value in {".", ".."}
        or PurePosixPath(value).is_absolute()
        or PureWindowsPath(value).is_absolute()
        or "/" in value
        or "\\" in value
        or Path(value).name != value
    ):
        raise ValueError("invalid logical basename")
    return value


RelativeArtifactPath = Annotated[
    str,
    Field(min_length=1),
    AfterValidator(_validate_relative_artifact_path),
]
UtcTimestamp = Annotated[
    str,
    Field(pattern=r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]+)?Z$"),
    AfterValidator(_validate_utc_timestamp),
]
NormalizedScope = Annotated[
    str,
    Field(min_length=1),
    AfterValidator(_validate_normalized_scope),
]
LogicalBasename = Annotated[
    str,
    Field(min_length=1),
    AfterValidator(_validate_logical_basename_value),
]


class ReceiptArm(_StrictModel):
    label: Literal["C1", "C2"]
    commit: GitCommit
    direct_parent: GitCommit | None = None
    profile_path: RelativeArtifactPath
    profile_id: SafeIdentifier
    profile_bytes: int = Field(gt=0)
    profile_file_sha256: Sha256
    profile_canonical_sha256: Sha256
    prompt_words: int = Field(gt=0)
    prompt_utf8_bytes: int = Field(gt=0)
    prompt_sha256: Sha256
    addition_words: int | None = Field(default=None, gt=0)
    addition_utf8_bytes: int | None = Field(default=None, gt=0)
    addition_sha256: Sha256 | None = None


class ReceiptArms(_StrictModel):
    control: ReceiptArm
    treatment: ReceiptArm


class SingleVariableContract(_StrictModel):
    treatment_equals_control_plus_two_lf_and_addition: Literal[True]
    normalized_full_module_ast_equal: Literal[True]
    profiles_equal_after_normalizing_id_and_prompt_hash: Literal[True]
    allowed_behavior_difference: Literal[
        "default_prompt_state_transition_addition_only"
    ]


class SharedReceiptIdentity(_StrictModel):
    suite_id: Literal["kama-coding-mvp@1"]
    suite_sha256: Sha256
    task_hashes_verified: int = Field(gt=0)
    grader_hashes_verified: int = Field(gt=0)
    provider: Literal["deepseek"]
    protocol: Literal["anthropic_messages"]
    endpoint_id: Literal["deepseek-anthropic-compatible"]
    model: Literal["deepseek-v4-pro"]
    sdk: Literal["anthropic==0.111.0"]
    tool_schema_sha256: Sha256
    runtime_config_sha256: Sha256
    dependency_sha256: Sha256
    max_steps: int = Field(gt=0)
    repeats: int = Field(gt=0)
    execution_order: Literal["suite_task_then_repeat_ascending"]
    easy_timeout_seconds: int = Field(gt=0)
    medium_challenging_timeout_seconds: int = Field(gt=0)
    mcp_enabled: Literal[False]
    raw_trace_visibility: Literal["private"]


class HostPolicy(_StrictModel):
    same_physical_host: Literal[True]
    same_dependency_environment: Literal[True]
    python: str = Field(min_length=1)
    os: str = Field(min_length=1)
    architecture: str = Field(min_length=1)
    no_environment_update_between_arms: Literal[True]


class ReceiptExecutionPlan(_StrictModel):
    arm_order: list[Literal["control", "treatment"]]
    attempts_per_arm: int = Field(gt=0)
    total_attempts: int = Field(gt=0)
    control_output_logical_root: LogicalBasename
    treatment_output_logical_root: LogicalBasename
    separate_detached_worktrees: Literal[True]
    output_roots_outside_repository: Literal[True]
    output_roots_must_be_new: Literal[True]
    no_paid_smoke_calls: Literal[True]
    no_rerun: Literal[True]
    no_resume: Literal[True]
    no_result_aware_changes_between_arms: Literal[True]
    treatment_runs_only_if_control_is_valid_and_complete: Literal[True]
    treatment_runs_regardless_of_control_capability_scores: Literal[True]


class ControlPreflightFailedState(_StrictModel):
    create_control_output_root: Literal[False]
    maximum_api_calls: Literal[0]
    run_treatment: Literal[False]
    pair_status: Literal["NOT_STARTED"]


class ControlStartedInvalidState(_StrictModel):
    preserve_control_artifacts: Literal[True]
    run_treatment: Literal[False]
    rerun: Literal[False]
    resume: Literal[False]
    pair_status: Literal["INVALID"]
    publish_paired_capability_delta: Literal[False]


class ControlValidState(_StrictModel):
    run_treatment: Literal[True]
    progression_depends_on_control_capability_scores: Literal[False]
    allow_experiment_changes_between_arms: Literal[False]


class TreatmentInvalidState(_StrictModel):
    preserve_all_artifacts: Literal[True]
    rerun: Literal[False]
    resume: Literal[False]
    pair_status: Literal["INVALID"]
    publish_paired_capability_delta: Literal[False]


class BothValidState(_StrictModel):
    evaluate_decision_contract: Literal[True]
    publish_paired_capability_delta: Literal[True]


class ReceiptExecutionStateMachine(_StrictModel):
    control_preflight_failed: ControlPreflightFailedState
    control_started_then_invalid_or_incomplete: ControlStartedInvalidState
    control_valid_and_complete: ControlValidState
    treatment_started_then_invalid_or_incomplete: TreatmentInvalidState
    both_arms_valid_and_complete: BothValidState


class ArmValidityContract(_StrictModel):
    required_status: Literal["VALID"]
    planned: int = Field(gt=0)
    started: int = Field(gt=0)
    completed: int = Field(gt=0)
    identity_verified: int = Field(gt=0)
    maximum_runtime_failures: int = Field(ge=0)
    maximum_infrastructure_failures: int = Field(ge=0)
    maximum_trace_failures: int = Field(ge=0)
    maximum_grader_failures: int = Field(ge=0)


class PrimaryThreshold(_StrictModel):
    treatment_minimum_successes: int = Field(ge=0)
    treatment_minus_control_minimum: int


class PrimaryComparison(_StrictModel):
    inventory_lifecycle: PrimaryThreshold
    feature_implementation: PrimaryThreshold
    overall: PrimaryThreshold


class HardGuardrails(_StrictModel):
    control_bug_fixing_successes_required: int = Field(ge=0)
    treatment_bug_fixing_successes_required: int = Field(ge=0)
    maximum_timeouts_per_arm: int = Field(ge=0)
    treatment_timeouts_must_not_exceed_control: bool


class SecondaryReporting(_StrictModel):
    atomic_plus_inventory_treatment_minimum: int = Field(ge=0)
    atomic_plus_inventory_attempts: int = Field(gt=0)
    atomic_oracle_disposition: str = Field(min_length=1)
    report_task_repeat_win_loss_tie: bool
    analysis_only_stateful_tasks: list[str]
    analysis_only_fields: list[str]


class EfficiencyComparison(_StrictModel):
    maximum_treatment_to_control_complete_median_latency_ratio: float = Field(gt=0)
    maximum_treatment_to_control_complete_median_input_output_token_ratio: float = (
        Field(gt=0)
    )
    median_algorithm: Literal["python.statistics.median"]
    exclude_timeout_zero_token_placeholders: bool
    report_total_experiment_wall_for_each_arm: bool


ClassificationVerdict = Literal["INVALID", "REJECT", "ACCEPT", "MIXED"]
CapabilityVerdict = Literal["REJECT", "ACCEPT", "MIXED"]


class DecisionContract(_StrictModel):
    classification_order: list[ClassificationVerdict]
    invalid_if: dict[str, bool]
    reject_if_both_arms_valid_and_any: dict[str, bool]
    accept_if_both_arms_valid_and_all: dict[str, bool]
    mixed_if_both_arms_valid_and_all: dict[str, bool]
    secondary_reporting_affects_classification: Literal[False]


class ReceiptAuthorization(_StrictModel):
    real_model_experiment_authorized: Literal[False]
    authorized_attempts: Literal[0]
    authorized_data_egress: Literal[False]
    authorized_cost: Literal[False]


class ImmutabilityPolicy(_StrictModel):
    receipt_bytes_must_never_change_after_commit: Literal[True]
    authorization_fields_in_this_receipt_must_remain_false_or_zero: Literal[True]
    future_execution_authorization_requires_separate_tracked_artifact: Literal[True]
    future_authorization_artifact_must_reference_receipt_git_commit: Literal[True]
    future_authorization_artifact_must_reference_receipt_file_sha256: Literal[True]
    future_authorization_must_not_amend_or_replace_this_receipt: Literal[True]


class LogicalRootPolicy(_StrictModel):
    values_are_frozen_identifiers_not_execution_dates: Literal[True]
    must_be_single_basename: Literal[True]
    must_not_be_empty: Literal[True]
    must_not_equal_dot_or_dotdot: Literal[True]
    must_not_contain_posix_separator: Literal[True]
    must_not_contain_windows_separator: Literal[True]
    must_not_be_absolute: Literal[True]
    control_and_treatment_must_differ: Literal[True]
    external_parent_selected_only_at_final_preflight: Literal[True]
    final_preflight_must_preserve_frozen_basenames: Literal[True]
    final_preflight_must_recheck_nonexistence: Literal[True]


class PairedReceipt(_StrictModel):
    schema_version: Literal[1]
    receipt_id: SafeIdentifier
    status: Literal["preregistered_before_any_paired_real_model_execution"]
    shared_repaired_stack_commit: GitCommit
    arms: ReceiptArms
    single_variable_contract: SingleVariableContract
    shared_identity: SharedReceiptIdentity
    host_policy: HostPolicy
    execution_plan: ReceiptExecutionPlan
    execution_state_machine: ReceiptExecutionStateMachine
    arm_validity: ArmValidityContract
    primary_comparison: PrimaryComparison
    hard_guardrails: HardGuardrails
    secondary_reporting: SecondaryReporting
    efficiency_comparison: EfficiencyComparison
    decision_contract: DecisionContract
    authorization: ReceiptAuthorization
    immutability_policy: ImmutabilityPolicy
    logical_root_policy: LogicalRootPolicy


class GitSnapshot(_StrictModel):
    commit: GitCommit
    branch: str = Field(min_length=1)
    remote_ref: str = Field(min_length=1)
    remote_commit: GitCommit
    ahead: Literal[0]
    behind: Literal[0]
    dirty: Literal[False]
    git_operation_in_progress: Literal[False]


class ReceiptReference(_StrictModel):
    commit: GitCommit
    path: RelativeArtifactPath
    bytes: int = Field(gt=0)
    sha256: Sha256
    authorization_remains_false_zero: Literal[True]


class SourceImportEvidence(_StrictModel):
    source_root_sha256: Sha256
    imported_module_path_sha256: Sha256
    imported_module_file_sha256: Sha256
    module_within_source_root: Literal[True]
    absolute_path_persisted: Literal[False]


class WorktreeEvidence(_StrictModel):
    label: SafeIdentifier
    canonical_path_sha256: Sha256
    absolute_path_persisted: Literal[False]
    registered: Literal[True]
    detached: Literal[True]
    clean: Literal[True]
    head_matches: Literal[True]
    profile_exists: Literal[True]
    source_import: SourceImportEvidence
    outside_repository: Literal[True]
    outside_output_parent: Literal[True]


class ArmPreflightEvidence(_StrictModel):
    arm: Literal["control", "treatment"]
    commit: GitCommit
    profile_path: RelativeArtifactPath
    profile_id: SafeIdentifier
    profile_file_sha256: Sha256
    profile_canonical_sha256: Sha256
    prompt_sha256: Sha256
    worktree: WorktreeEvidence


class PreflightArms(_StrictModel):
    control: ArmPreflightEvidence
    treatment: ArmPreflightEvidence


class DistributionSnapshot(_StrictModel):
    sha256: Sha256
    count: int = Field(ge=0)


class EnvironmentSnapshot(_StrictModel):
    python_version: str = Field(min_length=1)
    os: str = Field(min_length=1)
    os_release: str = Field(min_length=1)
    architecture: str = Field(min_length=1)
    sdk_distribution: str = Field(min_length=1)
    sdk_version: str = Field(min_length=1)
    interpreter_path_sha256: Sha256
    interpreter_file_sha256: Sha256
    installed_distributions_sha256: Sha256
    installed_distribution_count: int = Field(ge=0)
    uv_version: str = Field(min_length=1)
    pyproject_sha256: Sha256
    uv_lock_sha256: Sha256
    dependency_sha256: Sha256


class SharedPreflightIdentity(_StrictModel):
    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    protocol: str = Field(min_length=1)
    suite_sha256: Sha256
    tool_schema_sha256: Sha256
    runtime_config_sha256: Sha256
    dependency_sha256: Sha256
    max_steps: int = Field(gt=0)
    repeats: int = Field(gt=0)
    mcp_enabled: Literal[False]


class OutputParentEvidence(_StrictModel):
    env_name: SafeIdentifier = "KAMA_PHASE9D_OUTPUT_PARENT"
    label: SafeIdentifier = "phase9d-external-output-parent"
    canonical_path_sha256: Sha256
    canonical_object_sha256: Sha256
    absolute_path_persisted: Literal[False] = False
    exists: Literal[True] = True
    directory: Literal[True] = True
    writable: Literal[True] = True
    outside_repository: Literal[True] = True
    outside_git_common_dir: Literal[True] = True
    outside_all_worktrees: Literal[True] = True
    canonical_resolution_stable: Literal[True] = True


class LogicalRootsEvidence(_StrictModel):
    control: LogicalBasename
    treatment: LogicalBasename
    control_lexists: Literal[False]
    treatment_lexists: Literal[False]
    roots_created: Literal[0]


class CommandHashes(_StrictModel):
    control_spec_sha256: Sha256
    treatment_spec_sha256: Sha256
    raw_absolute_paths_persisted: Literal[False]
    credential_value_persisted: Literal[False]


class CredentialEvidence(_StrictModel):
    env_name: SafeIdentifier
    present: Literal[True]
    value_persisted: Literal[False] = False
    hash_persisted: Literal[False] = False


class NetworkEvidence(_StrictModel):
    provider_calls: Literal[0]
    paid_smoke_calls: Literal[0]


class FinalPreflightArtifact(_StrictModel):
    schema_version: Literal[1]
    preflight_id: SafeIdentifier
    status: Literal["READY_AWAITING_EXECUTION_AUTHORIZATION"]
    created_at_utc: UtcTimestamp
    generator: GitSnapshot
    paired_receipt: ReceiptReference
    arms: PreflightArms
    environment: EnvironmentSnapshot
    shared_identity: SharedPreflightIdentity
    external_parent: OutputParentEvidence
    logical_roots: LogicalRootsEvidence
    commands: CommandHashes
    credential: CredentialEvidence
    network: NetworkEvidence
    authorization: Literal["AWAITING_SEPARATE_EXECUTION_AUTHORIZATION"]


class ShortArtifactReference(_StrictModel):
    commit: GitCommit
    sha256: Sha256


class AuthorizationAttempts(_StrictModel):
    control: int = Field(gt=0)
    treatment: int = Field(gt=0)
    total: int = Field(gt=0)


class BudgetLimit(_StrictModel):
    amount: float = Field(gt=0)
    currency: Annotated[str, Field(pattern=r"^[A-Z]{3}$")]


class LogicalBasenames(_StrictModel):
    control: LogicalBasename
    treatment: LogicalBasename


class HumanAuthorizationEvidence(_StrictModel):
    recorded_at_utc: UtcTimestamp
    normalized_scope: NormalizedScope
    revoked_at_recording: Literal[False]
    conversation_reference_persisted: Literal[False]
    personal_identity_persisted: Literal[False]
    cryptographic_signature_claimed: Literal[False]


class ExecutionAuthorizationArtifact(_StrictModel):
    schema_version: Literal[1]
    authorization_id: SafeIdentifier
    status: Literal["AUTHORIZED_FOR_ONE_PAIRED_EXECUTION"]
    created_at_utc: UtcTimestamp
    paired_receipt: ShortArtifactReference
    final_preflight: ShortArtifactReference
    control_commit: GitCommit
    treatment_commit: GitCommit
    provider: Literal["deepseek"]
    model: Literal["deepseek-v4-pro"]
    protocol: Literal["anthropic_messages"]
    attempts: AuthorizationAttempts
    maximum_authorized_attempts: int = Field(gt=0)
    data_egress_authorized: Literal[True]
    cost_authorized: Literal[True]
    cost_scope: str = Field(min_length=1)
    maximum_budget: BudgetLimit | None
    maximum_budget_declared_by_user: bool
    no_paid_smoke_calls: Literal[True]
    raw_trace_visibility: Literal["private"]
    output_parent_sha256: Sha256
    logical_basenames: LogicalBasenames
    no_rerun: Literal[True]
    no_resume: Literal[True]
    applies_once: Literal[True]
    authorization_initially_unused: Literal[True]
    expires_on: list[
        Literal[
            "identity_drift",
            "root_existence",
            "environment_drift",
            "code_change",
            "credential_absence",
            "preflight_failure",
            "authorization_use_conflict",
        ]
    ] = Field(min_length=7, max_length=7)
    human_authorization: HumanAuthorizationEvidence
    receipt_remains_immutable: Literal[True]

    @model_validator(mode="after")
    # 要求非空预算只能来自用户显式声明，null预算不得伪装成已声明数值
    def _budget_matches_declaration(self) -> ExecutionAuthorizationArtifact:
        if (self.maximum_budget is None) != (
            not self.maximum_budget_declared_by_user
        ):
            raise ValueError("budget declaration does not match maximum budget")
        expected_expiry = [
            "identity_drift",
            "root_existence",
            "environment_drift",
            "code_change",
            "credential_absence",
            "preflight_failure",
            "authorization_use_conflict",
        ]
        if self.expires_on != expected_expiry:
            raise ValueError("authorization expiry contract does not match")
        return self


class CommandSpec(_StrictModel):
    arm: Literal["control", "treatment"]
    interpreter_label: SafeIdentifier
    interpreter_sha256: Sha256
    worktree_label: SafeIdentifier
    worktree_sha256: Sha256
    working_directory_sha256: Sha256
    argv: list[str]
    profile_path: RelativeArtifactPath
    output_basename: LogicalBasename
    allowed_env_names: list[str]
    source_binding_strategy: Literal["arm_src_via_pythonpath"]
    shell: Literal[False]
    expected_attempts: int = Field(gt=0)
    spec_sha256: Sha256


@dataclass(frozen=True)
class BoundOutputParent:
    path: Path
    control_root: Path
    treatment_root: Path
    evidence: OutputParentEvidence


@dataclass(frozen=True)
class PrivateEvidencePaths:
    root: Path
    authorization_use: Path
    terminal_record: Path
    paired_result: Path
    control_stdout: Path
    control_stderr: Path
    treatment_stdout: Path
    treatment_stderr: Path


# 将JSON-compatible值序列化为排序、紧凑、UTF-8语义的canonical文本
def canonical_json(value: object) -> str:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("value cannot be represented as canonical JSON") from exc


# 对canonical JSON的UTF-8 bytes计算SHA-256
def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


# 对已验证目录执行fsync以持久化create-once目录项
def _fsync_directory(path: Path) -> None:
    descriptor: int | None = None
    try:
        descriptor = os.open(path, os.O_RDONLY)
        os.fsync(descriptor)
    except OSError as exc:
        raise ValueError("artifact directory cannot be synchronized") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


# 从冻结receipt ID机械派生唯一private evidence路径集合
def derive_private_evidence_paths(
    output_parent: Path | str,
    receipt_id: str,
) -> PrivateEvidencePaths:
    parent = Path(output_parent)
    if not parent.is_absolute() or not parent.is_dir() or parent.is_symlink():
        raise ValueError("private evidence parent is invalid")
    if not re.fullmatch(r"[A-Za-z0-9._-]+", receipt_id):
        raise ValueError("private evidence identity is invalid")
    root = parent / f"{receipt_id}.paired-private"
    return PrivateEvidencePaths(
        root=root,
        authorization_use=parent / f"{receipt_id}.authorization-use.json",
        terminal_record=parent / f"{receipt_id}.terminal.json",
        paired_result=root / "paired-result",
        control_stdout=root / "control.stdout.log",
        control_stderr=root / "control.stderr.log",
        treatment_stdout=root / "treatment.stdout.log",
        treatment_stderr=root / "treatment.stderr.log",
    )


# 拒绝任意层级JSON object中的重复键
def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


# 拒绝标准JSON之外的NaN和Infinity常量
def _reject_non_finite(_value: str) -> None:
    raise ValueError("non-finite JSON constant")


# 解析严格JSON object并保留duplicate/nonfinite失败语义
def _strict_json_object(text: str) -> dict[str, Any]:
    value = json.loads(
        text,
        object_pairs_hook=_reject_duplicate_keys,
        parse_constant=_reject_non_finite,
    )
    if not isinstance(value, dict):
        raise ValueError("JSON root must be an object")
    return value


# 为外部observer暴露共享的duplicate/nonfinite严格JSON object解析合同
def parse_strict_json_object(text: str) -> dict[str, Any]:
    return _strict_json_object(text)


# 校验logical root是不可逃逸且不可正规化的单一basename
def _validate_logical_basename(value: str) -> None:
    if (
        not value
        or value in {".", ".."}
        or PurePosixPath(value).is_absolute()
        or PureWindowsPath(value).is_absolute()
        or "/" in value
        or "\\" in value
        or Path(value).name != value
    ):
        raise ValueError("invalid logical basename")


# 验证receipt classification order完整覆盖支持集合但不冻结其业务顺序
def _validated_classification_order(
    order: object,
) -> tuple[ClassificationVerdict, ...]:
    supported = frozenset(get_args(ClassificationVerdict))
    if (
        not isinstance(order, (list, tuple))
        or len(order) != len(supported)
        or any(type(item) is not str or item not in supported for item in order)
        or len(set(order)) != len(order)
        or set(order) != supported
    ):
        raise ValueError("paired receipt classification order is invalid")
    return tuple(cast(ClassificationVerdict, item) for item in order)


# 按receipt提供的顺序选择首个true verdict并拒绝不完整predicate map
def select_classification_verdict(
    order: object,
    matches: Mapping[str, bool],
) -> ClassificationVerdict:
    validated_order = _validated_classification_order(order)
    supported = frozenset(get_args(ClassificationVerdict))
    if (
        set(matches) != supported
        or any(type(value) is not bool for value in matches.values())
    ):
        raise ValueError("paired classification predicates are invalid")
    for verdict in validated_order:
        if matches[verdict]:
            return verdict
    raise ValueError("paired decision contract did not produce one verdict")


# 对真实receipt执行跨字段和值域一致性检查
def _validate_receipt_semantics(receipt: PairedReceipt) -> None:
    plan = receipt.execution_plan
    validity = receipt.arm_validity
    if (
        plan.arm_order != ["control", "treatment"]
        or plan.total_attempts != plan.attempts_per_arm * len(plan.arm_order)
        or validity.planned != plan.attempts_per_arm
        or not (
            validity.planned
            == validity.started
            == validity.completed
            == validity.identity_verified
        )
        or receipt.arms.treatment.direct_parent != receipt.arms.control.commit
    ):
        raise ValueError("paired receipt count or arm identity mismatch")
    _validate_logical_basename(plan.control_output_logical_root)
    _validate_logical_basename(plan.treatment_output_logical_root)
    if plan.control_output_logical_root == plan.treatment_output_logical_root:
        raise ValueError("paired receipt logical roots must differ")
    expected_keys = {
        "invalid_if": {
            "either_arm_status_not_valid",
            "either_arm_attempt_counts_not_exact",
            "either_arm_identity_verified_count_not_exact",
            "either_arm_runtime_failures_above_maximum",
            "either_arm_infrastructure_failures_above_maximum",
            "either_arm_trace_failures_above_maximum",
            "either_arm_grader_failures_above_maximum",
            "required_artifact_evidence_missing",
        },
        "reject_if_both_arms_valid_and_any": {
            "inventory_treatment_below_minimum",
            "inventory_delta_below_minimum",
            "feature_delta_below_minimum",
            "overall_delta_below_minimum",
            "control_bug_fixing_not_required_value",
            "treatment_bug_fixing_not_required_value",
            "either_arm_timeouts_above_maximum",
            "treatment_timeouts_exceed_control",
        },
        "accept_if_both_arms_valid_and_all": {
            "no_reject_condition",
            "inventory_primary_pass",
            "feature_primary_pass",
            "overall_primary_pass",
            "latency_ratio_at_or_below_maximum",
            "token_ratio_at_or_below_maximum",
        },
        "mixed_if_both_arms_valid_and_all": {
            "inventory_primary_pass",
            "no_reject_condition",
            "accept_condition_false",
        },
    }
    contract = receipt.decision_contract
    for field, keys in expected_keys.items():
        if set(getattr(contract, field)) != keys:
            raise ValueError("paired receipt decision contract mismatch")
    _validated_classification_order(contract.classification_order)


# 加载完整严格paired receipt并隐藏底层JSON/Pydantic细节
def load_paired_receipt(path: Path | str) -> PairedReceipt:
    try:
        return _load_paired_receipt_bytes(Path(path).read_bytes())
    except (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        ValidationError,
        ValueError,
    ) as exc:
        raise ValueError("invalid paired receipt") from exc


# 从原始bytes严格加载paired receipt并复用全部语义校验
def _load_paired_receipt_bytes(payload: bytes) -> PairedReceipt:
    receipt = PairedReceipt.model_validate(
        _strict_json_object(payload.decode("utf-8"))
    )
    _validate_receipt_semantics(receipt)
    return receipt


# 执行不携带credential的只读Git命令并返回原始stdout
def _git_stdout(repository: Path, args: Sequence[str]) -> bytes:
    result = subprocess.run(
        ["git", *args],
        cwd=repository,
        check=False,
        capture_output=True,
        env={"PATH": os.environ.get("PATH", "")},
    )
    if result.returncode != 0:
        raise ValueError("paired receipt Git reference is invalid")
    return result.stdout


# 规范化receipt路径为仓库相对POSIX路径并拒绝绝对和逃逸别名
def _normalize_receipt_reference_path(
    repository: Path,
    receipt_path: Path | str,
) -> str:
    root = repository.resolve(strict=True)
    raw = Path(receipt_path)
    if raw.is_absolute():
        if raw.is_symlink():
            raise ValueError("paired receipt path is invalid")
        try:
            relative = raw.resolve(strict=True).relative_to(root)
        except (OSError, ValueError) as exc:
            raise ValueError("paired receipt path is invalid") from exc
        value = relative.as_posix()
    else:
        value = str(receipt_path)
        _validate_relative_artifact_path(value)
        candidate = root / value
        if candidate.exists() and candidate.is_symlink():
            raise ValueError("paired receipt path is invalid")
    return _validate_relative_artifact_path(value)


# 从指定commit和仓库相对路径读取receipt blob并验证commit类型
def _read_receipt_blob_from_git(
    repository: Path | str,
    commit: str,
    receipt_path: str,
) -> bytes:
    root = Path(repository).resolve(strict=True)
    _git_stdout(root, ["cat-file", "-e", f"{commit}^{{commit}}"])
    return _git_stdout(root, ["show", f"{commit}:{receipt_path}"])


# 从Git commit/path/blob机械观察authoritative receipt reference
def observe_receipt_reference(
    repository: Path | str,
    commit: str,
    receipt_path: Path | str,
) -> ReceiptReference:
    root = Path(repository).resolve(strict=True)
    path = _normalize_receipt_reference_path(root, receipt_path)
    blob = _read_receipt_blob_from_git(root, commit, path)
    _load_paired_receipt_bytes(blob)
    return ReceiptReference(
        commit=commit,
        path=path,
        bytes=len(blob),
        sha256=hashlib.sha256(blob).hexdigest(),
        authorization_remains_false_zero=True,
    )


# 用authoritative reference重新读取Git blob并加载receipt
def _load_receipt_from_reference(
    repository: Path | str,
    expected_receipt: ReceiptReference,
) -> PairedReceipt:
    blob = _read_receipt_blob_from_git(
        repository,
        expected_receipt.commit,
        expected_receipt.path,
    )
    if (
        len(blob) != expected_receipt.bytes
        or hashlib.sha256(blob).hexdigest() != expected_receipt.sha256
        or not expected_receipt.authorization_remains_false_zero
    ):
        raise ValueError("paired receipt reference is invalid")
    return _load_paired_receipt_bytes(blob)


# 只返回credential存在性证据，不返回、hash或测量value
def check_credential_presence(
    env: Mapping[str, str],
    env_name: str,
) -> CredentialEvidence:
    if env_name not in env or not env[env_name]:
        raise ValueError("experiment credential is missing")
    return CredentialEvidence(env_name=env_name, present=True)


# 将distribution name正规化为稳定的PEP503风格小写标识
def _normalize_distribution_name(value: str) -> str:
    normalized = re.sub(r"[-_.]+", "-", value.strip()).lower()
    if not normalized:
        raise ValueError("installed distribution identity is invalid")
    return normalized


# 对installed distribution name/version集合生成顺序无关的canonical hash
def snapshot_installed_distributions(
    distributions: Iterable[tuple[str, str]],
) -> DistributionSnapshot:
    versions: dict[str, str] = {}
    for raw_name, raw_version in distributions:
        name = _normalize_distribution_name(raw_name)
        version = raw_version.strip()
        if not version or (name in versions and versions[name] != version):
            raise ValueError("installed distribution identity is ambiguous")
        versions[name] = version
    pairs = sorted(versions.items())
    return DistributionSnapshot(
        sha256=canonical_sha256(pairs),
        count=len(pairs),
    )


# 对文件原始bytes计算SHA-256并净化文件系统错误
def _hash_file(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise ValueError("identity input cannot be hashed") from exc


# 生成不包含路径和自由metadata的deterministic环境快照
def capture_environment_snapshot(
    *,
    interpreter: Path | str,
    distributions: Iterable[tuple[str, str]],
    python_version: str,
    system: str,
    release: str,
    machine: str,
    sdk_distribution: str,
    sdk_version: str,
    uv_version: str,
    pyproject: Path | str,
    uv_lock: Path | str,
) -> EnvironmentSnapshot:
    try:
        interpreter_path = Path(interpreter).resolve(strict=True)
    except OSError as exc:
        raise ValueError("interpreter identity is invalid") from exc
    distribution_snapshot = snapshot_installed_distributions(distributions)
    pyproject_sha256 = _hash_file(Path(pyproject))
    uv_lock_sha256 = _hash_file(Path(uv_lock))
    return EnvironmentSnapshot(
        python_version=python_version,
        os=system,
        os_release=release,
        architecture=machine,
        sdk_distribution=sdk_distribution,
        sdk_version=sdk_version,
        interpreter_path_sha256=hashlib.sha256(
            str(interpreter_path).encode("utf-8", errors="strict")
        ).hexdigest(),
        interpreter_file_sha256=_hash_file(interpreter_path),
        installed_distributions_sha256=distribution_snapshot.sha256,
        installed_distribution_count=distribution_snapshot.count,
        uv_version=uv_version,
        pyproject_sha256=pyproject_sha256,
        uv_lock_sha256=uv_lock_sha256,
        dependency_sha256=canonical_sha256(
            {
                "pyproject.toml": pyproject_sha256,
                "uv.lock": uv_lock_sha256,
            }
        ),
    )


# 判断两个canonical路径是否存在任一方向的containment
def _paths_overlap(first: Path, second: Path) -> bool:
    return (
        first == second
        or first.is_relative_to(second)
        or second.is_relative_to(first)
    )


# 验证external parent并返回仅在内存中携带raw Path的绑定结果
def bind_output_parent(
    raw_parent: Path | str,
    *,
    repository: Path | str,
    git_common_dir: Path | str,
    worktrees: Sequence[Path | str],
    control_basename: str,
    treatment_basename: str,
) -> BoundOutputParent:
    raw = Path(raw_parent)
    if not raw.is_absolute():
        raise ValueError("output parent must be absolute")
    try:
        parent = raw.resolve(strict=True)
        second_resolution = raw.resolve(strict=True)
        repository_path = Path(repository).resolve(strict=True)
        git_path = Path(git_common_dir).resolve(strict=True)
        worktree_paths = [Path(path).resolve(strict=True) for path in worktrees]
    except OSError as exc:
        raise ValueError("output parent or boundary is missing") from exc
    if not parent.is_dir():
        raise ValueError("output parent must be a directory")
    if not os.access(parent, os.W_OK):
        raise ValueError("output parent is not writable")
    if _paths_overlap(parent, repository_path):
        raise ValueError("output parent overlaps repository")
    if _paths_overlap(parent, git_path):
        raise ValueError("output parent overlaps Git common directory")
    if any(_paths_overlap(parent, worktree) for worktree in worktree_paths):
        raise ValueError("output parent overlaps worktree")
    _validate_logical_basename(control_basename)
    _validate_logical_basename(treatment_basename)
    if control_basename == treatment_basename:
        raise ValueError("logical basenames must differ")
    control_root = parent / control_basename
    treatment_root = parent / treatment_basename
    if os.path.lexists(control_root) or os.path.lexists(treatment_root):
        raise ValueError("output root already exists")
    stat = parent.stat()
    if parent != second_resolution:
        raise ValueError("output parent canonical resolution changed")
    evidence = OutputParentEvidence(
        canonical_path_sha256=hashlib.sha256(
            str(parent).encode("utf-8", errors="strict")
        ).hexdigest(),
        canonical_object_sha256=canonical_sha256(
            {"st_dev": stat.st_dev, "st_ino": stat.st_ino}
        ),
        canonical_resolution_stable=True,
    )
    return BoundOutputParent(
        path=parent,
        control_root=control_root,
        treatment_root=treatment_root,
        evidence=evidence,
    )


# 重新绑定external parent并拒绝path或inode身份漂移
def rebind_output_parent(
    expected: OutputParentEvidence,
    raw_parent: Path | str,
    *,
    repository: Path | str,
    git_common_dir: Path | str,
    worktrees: Sequence[Path | str],
    control_basename: str,
    treatment_basename: str,
) -> BoundOutputParent:
    observed = bind_output_parent(
        raw_parent,
        repository=repository,
        git_common_dir=git_common_dir,
        worktrees=worktrees,
        control_basename=control_basename,
        treatment_basename=treatment_basename,
    )
    if (
        observed.evidence.canonical_path_sha256 != expected.canonical_path_sha256
        or observed.evidence.canonical_object_sha256
        != expected.canonical_object_sha256
    ):
        raise ValueError("output parent identity drift")
    return observed


# 构造不含raw绝对路径或credential值的logical child command spec
def build_command_spec(
    *,
    arm: Literal["control", "treatment"],
    interpreter_label: str,
    interpreter_sha256: str,
    worktree_label: str,
    worktree_sha256: str,
    profile_path: str,
    output_basename: str,
    expected_attempts: int,
    working_directory_sha256: str | None = None,
    argv: list[str] | None = None,
    allowed_env_names: list[str] | None = None,
    source_binding_strategy: str = "arm_src_via_pythonpath",
    shell: bool = False,
) -> CommandSpec:
    if Path(profile_path).is_absolute() or ".." in Path(profile_path).parts:
        raise ValueError("command profile path must be relative")
    _validate_logical_basename(output_basename)
    expected_argv = [
        "-m",
        "kama_claude.benchmark.cli",
        "run",
        "--experiment",
        profile_path,
        "--output",
        f"${{KAMA_PHASE9D_OUTPUT_PARENT}}/{output_basename}",
    ]
    expected_env = [
        "ANTHROPIC_API_KEY",
        "HOME",
        "LANG",
        "LC_ALL",
        "PATH",
        "PYTHONPATH",
        "PYTHONDONTWRITEBYTECODE",
        "PYTHONNOUSERSITE",
        "TMPDIR",
    ]
    if argv is not None and argv != expected_argv:
        raise ValueError("command argv does not match logical identity")
    if allowed_env_names is not None and allowed_env_names != expected_env:
        raise ValueError("command environment does not match logical identity")
    if source_binding_strategy != "arm_src_via_pythonpath" or shell is not False:
        raise ValueError("command execution policy is invalid")
    working_hash = working_directory_sha256 or worktree_sha256
    payload = {
        "arm": arm,
        "interpreter_label": interpreter_label,
        "interpreter_sha256": interpreter_sha256,
        "worktree_label": worktree_label,
        "worktree_sha256": worktree_sha256,
        "working_directory_sha256": working_hash,
        "argv": expected_argv,
        "profile_path": profile_path,
        "output_basename": output_basename,
        "allowed_env_names": expected_env,
        "source_binding_strategy": "arm_src_via_pythonpath",
        "shell": False,
        "expected_attempts": expected_attempts,
    }
    return CommandSpec(
        arm=arm,
        interpreter_label=interpreter_label,
        interpreter_sha256=interpreter_sha256,
        worktree_label=worktree_label,
        worktree_sha256=worktree_sha256,
        working_directory_sha256=working_hash,
        argv=expected_argv,
        profile_path=profile_path,
        output_basename=output_basename,
        allowed_env_names=expected_env,
        source_binding_strategy="arm_src_via_pythonpath",
        shell=False,
        expected_attempts=expected_attempts,
        spec_sha256=canonical_sha256(payload),
    )


# 以create-once hard-link发布canonical artifact并在落盘后fsync
def write_canonical_artifact(path: Path | str, artifact: _StrictModel) -> None:
    target = Path(path)
    if target.exists() or os.path.lexists(target):
        raise ValueError("artifact target already exists")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    payload = (canonical_json(artifact.model_dump(mode="json")) + "\n").encode("utf-8")
    descriptor: int | None = None
    try:
        descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            descriptor = None
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, target)
        _fsync_directory(target.parent)
    except FileExistsError as exc:
        raise ValueError("artifact target already exists") from exc
    except OSError as exc:
        raise ValueError("canonical artifact cannot be written") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


# 严格读取canonical artifact并拒绝非canonical或schema损坏内容
def read_strict_artifact[ArtifactModel: _StrictModel](
    path: Path | str,
    model: type[ArtifactModel],
) -> ArtifactModel:
    try:
        text = Path(path).read_text(encoding="utf-8")
        payload = _strict_json_object(text)
        artifact = model.model_validate(payload)
        if text != canonical_json(artifact.model_dump(mode="json")) + "\n":
            raise ValueError("artifact is not canonical")
        return artifact
    except (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        ValidationError,
        ValueError,
    ) as exc:
        raise ValueError("invalid canonical artifact") from exc


class ExpectedArmIdentity(_StrictModel):
    arm: Literal["control", "treatment"]
    commit: GitCommit
    profile_id: SafeIdentifier
    profile_hash: Sha256
    prompt_sha256: Sha256
    suite_sha256: Sha256
    tool_schema_sha256: Sha256
    runtime_config_sha256: Sha256
    dependency_sha256: Sha256
    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    protocol: str = Field(min_length=1)
    sdk: str = Field(min_length=1)
    max_steps: int = Field(gt=0)
    repeats: int = Field(gt=0)
    task_ids: list[SafeIdentifier] = Field(min_length=1)


class ArmAudit(_StrictModel):
    arm: Literal["control", "treatment"]
    status: Literal["VALID", "INVALID"]
    reasons: list[str]
    exit_code: int | None
    signal_number: int | None
    planned: int = Field(ge=0)
    started: int = Field(ge=0)
    completed: int = Field(ge=0)
    identity_verified: int = Field(ge=0)
    runtime_failures: int = Field(ge=0)
    infrastructure_failures: int = Field(ge=0)
    trace_failures: int = Field(ge=0)
    grader_failures: int = Field(ge=0)
    timeouts: int = Field(ge=0)
    provider_calls: int = Field(ge=0)
    required_artifact_evidence: bool
    overall_successes: int = Field(ge=0)
    feature_successes: int = Field(ge=0)
    bug_fixing_successes: int = Field(ge=0)
    inventory_successes: int = Field(ge=0)
    complete_median_latency_ms: float | None = Field(default=None, ge=0)
    complete_median_input_output_tokens: float | None = Field(default=None, ge=0)
    baseline_json_sha256: Sha256 | None = None
    baseline_markdown_sha256: Sha256 | None = None
    attempts: list[AttemptAnalysis] = Field(default_factory=list, exclude=True)


class AuthorizationUseRecord(_StrictModel):
    schema_version: Literal[1]
    reservation_id: SafeIdentifier
    status: Literal["RESERVED_FOR_ONE_PAIRED_EXECUTION"]
    created_at_utc: UtcTimestamp
    authorization_sha256: Sha256
    paired_receipt_sha256: Sha256
    output_parent_sha256: Sha256
    absolute_path_persisted: Literal[False]
    credential_value_persisted: Literal[False]


class AuthorizationReservationError(ValueError):
    # 保存失败时授权是否已通过create-once目标文件永久消费
    def __init__(self, *, consumed: bool) -> None:
        super().__init__("authorization use cannot be created")
        self.consumed = consumed


class WorktreeObservation(_StrictModel):
    label: SafeIdentifier
    path: Path
    canonical_path_sha256: Sha256
    source_root: Path
    registered: bool
    detached: bool
    clean: bool
    observed_head: GitCommit
    profile_exists: bool
    source_import: SourceImportEvidence


class BetweenArmEvidence(_StrictModel):
    receipt_commit: GitCommit
    receipt_sha256: Sha256
    preflight_commit: GitCommit
    preflight_sha256: Sha256
    authorization_commit: GitCommit
    authorization_sha256: Sha256
    git_artifact_identity_sha256: Sha256
    control_commit_exists: bool
    treatment_commit_exists: bool
    main_ref_sha256: Sha256
    treatment_worktree_sha256: Sha256
    treatment_profile_sha256: Sha256
    treatment_prompt_sha256: Sha256
    source_binding_sha256: Sha256
    treatment_source_import: SourceImportEvidence
    environment_sha256: Sha256
    pyproject_sha256: Sha256
    uv_lock_sha256: Sha256
    dependency_sha256: Sha256
    suite_sha256: Sha256
    task_bundle_sha256: Sha256
    grader_bundle_sha256: Sha256
    tool_schema_sha256: Sha256
    runtime_config_sha256: Sha256
    output_parent_path_sha256: Sha256
    output_parent_object_sha256: Sha256
    treatment_root_absent: bool
    credential_present: bool
    authorization_use_sha256: Sha256
    experiment_unchanged: bool


class FreshTrackedArtifactEvidence(_StrictModel):
    commit: GitCommit
    bytes: int = Field(gt=0)
    sha256: Sha256
    canonical_object_sha256: Sha256
    current_matches_blob: Literal[True]
    symlink: Literal[False]


class FreshBetweenArmGitArtifacts(_StrictModel):
    git: GitSnapshot
    receipt: FreshTrackedArtifactEvidence
    preflight: FreshTrackedArtifactEvidence
    authorization: FreshTrackedArtifactEvidence
    control_commit_exists: Literal[True]
    treatment_commit_exists: Literal[True]
    treatment_parent_matches_control: Literal[True]


class PairState(StrEnum):
    NOT_STARTED = "NOT_STARTED"
    AUTHORIZATION_RESERVED = "AUTHORIZATION_RESERVED"
    CONTROL_RUNNING = "CONTROL_RUNNING"
    CONTROL_INVALID = "CONTROL_INVALID"
    CONTROL_VALID = "CONTROL_VALID"
    TREATMENT_RUNNING = "TREATMENT_RUNNING"
    TREATMENT_INVALID = "TREATMENT_INVALID"
    BOTH_VALID = "BOTH_VALID"
    TERMINAL = "TERMINAL"


class PairEvent(StrEnum):
    PREFLIGHT_FAILED = "PREFLIGHT_FAILED"
    AUTHORIZATION_RESERVED = "AUTHORIZATION_RESERVED"
    PRIVATE_EVIDENCE_FAILED = "PRIVATE_EVIDENCE_FAILED"
    CONTROL_STARTED = "CONTROL_STARTED"
    CONTROL_SPAWN_FAILED = "CONTROL_SPAWN_FAILED"
    CONTROL_VALID = "CONTROL_VALID"
    CONTROL_INVALID = "CONTROL_INVALID"
    BETWEEN_ARM_INVALID = "BETWEEN_ARM_INVALID"
    TREATMENT_STARTED = "TREATMENT_STARTED"
    TREATMENT_SPAWN_FAILED = "TREATMENT_SPAWN_FAILED"
    TREATMENT_VALID = "TREATMENT_VALID"
    TREATMENT_INVALID = "TREATMENT_INVALID"
    CLASSIFICATION_FAILED = "CLASSIFICATION_FAILED"
    RESULT_WRITE_FAILED = "RESULT_WRITE_FAILED"
    PARENT_INTERRUPTED = "PARENT_INTERRUPTED"
    PARENT_SYSTEM_EXIT = "PARENT_SYSTEM_EXIT"
    FINALIZED = "FINALIZED"


_POST_RESERVATION_TRANSITIONS: dict[
    tuple[PairState, PairEvent],
    PairState,
] = {
    (PairState.NOT_STARTED, PairEvent.AUTHORIZATION_RESERVED): (
        PairState.AUTHORIZATION_RESERVED
    ),
    (PairState.AUTHORIZATION_RESERVED, PairEvent.PRIVATE_EVIDENCE_FAILED): (
        PairState.TERMINAL
    ),
    (PairState.AUTHORIZATION_RESERVED, PairEvent.CONTROL_STARTED): (
        PairState.CONTROL_RUNNING
    ),
    (PairState.CONTROL_RUNNING, PairEvent.CONTROL_SPAWN_FAILED): PairState.TERMINAL,
    (PairState.CONTROL_RUNNING, PairEvent.CONTROL_VALID): PairState.CONTROL_VALID,
    (PairState.CONTROL_RUNNING, PairEvent.CONTROL_INVALID): PairState.TERMINAL,
    (PairState.CONTROL_VALID, PairEvent.BETWEEN_ARM_INVALID): PairState.TERMINAL,
    (PairState.CONTROL_VALID, PairEvent.TREATMENT_STARTED): (
        PairState.TREATMENT_RUNNING
    ),
    (PairState.TREATMENT_RUNNING, PairEvent.TREATMENT_VALID): PairState.BOTH_VALID,
    (PairState.TREATMENT_RUNNING, PairEvent.TREATMENT_SPAWN_FAILED): (
        PairState.TERMINAL
    ),
    (PairState.TREATMENT_RUNNING, PairEvent.TREATMENT_INVALID): PairState.TERMINAL,
    (PairState.BOTH_VALID, PairEvent.CLASSIFICATION_FAILED): PairState.TERMINAL,
    (PairState.BOTH_VALID, PairEvent.RESULT_WRITE_FAILED): PairState.TERMINAL,
    (PairState.BOTH_VALID, PairEvent.FINALIZED): PairState.TERMINAL,
    **{
        (state, event): PairState.TERMINAL
        for state in PairState
        if state not in {PairState.NOT_STARTED, PairState.TERMINAL}
        for event in {
            PairEvent.PARENT_INTERRUPTED,
            PairEvent.PARENT_SYSTEM_EXIT,
        }
    },
}


class PairTransition(_StrictModel):
    from_state: Literal[
        "NOT_STARTED",
        "AUTHORIZATION_RESERVED",
        "CONTROL_RUNNING",
        "CONTROL_INVALID",
        "CONTROL_VALID",
        "TREATMENT_RUNNING",
        "TREATMENT_INVALID",
        "BOTH_VALID",
        "TERMINAL",
    ]
    event: Literal[
        "PREFLIGHT_FAILED",
        "AUTHORIZATION_RESERVED",
        "PRIVATE_EVIDENCE_FAILED",
        "CONTROL_STARTED",
        "CONTROL_SPAWN_FAILED",
        "CONTROL_VALID",
        "CONTROL_INVALID",
        "BETWEEN_ARM_INVALID",
        "TREATMENT_STARTED",
        "TREATMENT_SPAWN_FAILED",
        "TREATMENT_VALID",
        "TREATMENT_INVALID",
        "CLASSIFICATION_FAILED",
        "RESULT_WRITE_FAILED",
        "PARENT_INTERRUPTED",
        "PARENT_SYSTEM_EXIT",
        "FINALIZED",
    ]
    to_state: Literal[
        "NOT_STARTED",
        "AUTHORIZATION_RESERVED",
        "CONTROL_RUNNING",
        "CONTROL_INVALID",
        "CONTROL_VALID",
        "TREATMENT_RUNNING",
        "TREATMENT_INVALID",
        "BOTH_VALID",
        "TERMINAL",
    ]


# 验证post-reservation history从NOT_STARTED连续推进且只在最后进入TERMINAL
def _validate_terminal_history(transitions: Sequence[PairTransition]) -> None:
    if (
        not transitions
        or transitions[0].from_state != PairState.NOT_STARTED.value
        or transitions[0].event != PairEvent.AUTHORIZATION_RESERVED.value
        or transitions[0].to_state != PairState.AUTHORIZATION_RESERVED.value
        or transitions[-1].to_state != PairState.TERMINAL.value
    ):
        raise ValueError("pair transition history is incomplete")
    for transition in transitions:
        expected = _POST_RESERVATION_TRANSITIONS.get(
            (
                PairState(transition.from_state),
                PairEvent(transition.event),
            )
        )
        if expected is None or expected.value != transition.to_state:
            raise ValueError("pair transition history contains invalid transition")
    for previous, current in zip(transitions, transitions[1:], strict=False):
        if (
            previous.to_state != current.from_state
            or previous.to_state == PairState.TERMINAL.value
        ):
            raise ValueError("pair transition history is discontinuous")


class ChildTerminationEvidence(_StrictModel):
    spawned: bool
    exit_code: int | None
    signal_number: int | None
    cancelled: bool
    failure_category: str | None
    cleanup_term_sent: bool = False
    cleanup_kill_sent: bool = False
    process_group_gone: bool = True


TerminalPhase = Literal[
    "control_spawn",
    "control_audit",
    "between_arms",
    "treatment_spawn",
    "treatment_audit",
    "paired_classification",
    "paired_result_write",
    "parent_interrupt",
]
TerminalFailureCategory = Literal[
    "private_evidence_unavailable",
    "control_spawn_failed",
    "control_invalid",
    "between_arm_invalid",
    "treatment_spawn_failed",
    "treatment_invalid",
    "paired_result_write_failed",
    "paired_classification_failed",
    "parent_interrupted",
    "parent_system_exit",
]


_FAILURE_TERMINAL_EVENTS: dict[tuple[str, str], PairEvent] = {
    ("control_spawn", "private_evidence_unavailable"): (
        PairEvent.PRIVATE_EVIDENCE_FAILED
    ),
    ("control_spawn", "control_spawn_failed"): PairEvent.CONTROL_SPAWN_FAILED,
    ("control_audit", "control_invalid"): PairEvent.CONTROL_INVALID,
    ("between_arms", "between_arm_invalid"): PairEvent.BETWEEN_ARM_INVALID,
    ("treatment_spawn", "treatment_spawn_failed"): (
        PairEvent.TREATMENT_SPAWN_FAILED
    ),
    ("treatment_audit", "treatment_invalid"): PairEvent.TREATMENT_INVALID,
    ("paired_classification", "paired_classification_failed"): (
        PairEvent.CLASSIFICATION_FAILED
    ),
    ("paired_result_write", "paired_result_write_failed"): (
        PairEvent.RESULT_WRITE_FAILED
    ),
    ("parent_interrupt", "parent_interrupted"): PairEvent.PARENT_INTERRUPTED,
    ("parent_interrupt", "parent_system_exit"): PairEvent.PARENT_SYSTEM_EXIT,
}


# 将failure phase/category机械映射为唯一terminal event并拒绝未知组合
def terminal_event_for_failure(
    phase: str,
    failure_category: str,
) -> PairEvent:
    try:
        return _FAILURE_TERMINAL_EVENTS[(phase, failure_category)]
    except KeyError as exc:
        raise ValueError("failure terminal phase/category is invalid") from exc


# 校验failure artifact携带的arm audits与其执行phase相容
def _validate_failure_audit_presence(record: PairTerminalRecord) -> None:
    control = record.control
    treatment = record.treatment
    if treatment is not None and control is None:
        raise ValueError("failure terminal audit presence is invalid")
    if record.phase == "control_spawn" and (control is not None or treatment is not None):
        raise ValueError("failure terminal audit presence is invalid")
    if record.phase == "control_audit" and treatment is not None:
        raise ValueError("failure terminal audit presence is invalid")
    if record.phase in {"between_arms", "treatment_spawn"} and (
        control is None or control.status != "VALID" or treatment is not None
    ):
        raise ValueError("failure terminal audit presence is invalid")
    if record.phase == "treatment_audit" and (
        control is None or control.status != "VALID"
    ):
        raise ValueError("failure terminal audit presence is invalid")
    if record.phase in {"paired_classification", "paired_result_write"} and (
        control is None
        or treatment is None
        or control.status != "VALID"
        or treatment.status != "VALID"
    ):
        raise ValueError("failure terminal audit presence is invalid")


class PairTerminalRecord(_StrictModel):
    schema_version: Literal[1]
    terminal_id: SafeIdentifier
    created_at_utc: UtcTimestamp
    status: Literal["INVALID"]
    phase: TerminalPhase
    receipt_sha256: Sha256
    preflight_sha256: Sha256
    authorization_sha256: Sha256
    authorization_use_sha256: Sha256
    transitions: list[PairTransition] = Field(min_length=1)
    control: ArmAudit | None = None
    treatment: ArmAudit | None = None
    control_child: ChildTerminationEvidence | None = None
    treatment_child: ChildTerminationEvidence | None = None
    provider_call_count: int = Field(ge=0)
    capability_delta_published: Literal[False]
    private_visibility: Literal[True]
    failure_category: TerminalFailureCategory

    @model_validator(mode="after")
    # 校验failure terminal的kind、event、phase/category和audit presence一致
    def _terminal_history_is_complete(self) -> PairTerminalRecord:
        _validate_terminal_history(self.transitions)
        expected_event = terminal_event_for_failure(self.phase, self.failure_category)
        final = self.transitions[-1]
        if (
            final.event != expected_event.value
            or final.event == PairEvent.FINALIZED.value
            or final.to_state != PairState.TERMINAL.value
        ):
            raise ValueError("failure terminal event is invalid")
        _validate_failure_audit_presence(self)
        return self


class PairOutcome(_StrictModel):
    control: ArmAudit
    treatment: ArmAudit
    required_artifact_evidence: bool
    inventory_control_successes: int = Field(ge=0)
    inventory_treatment_successes: int = Field(ge=0)
    feature_control_successes: int = Field(ge=0)
    feature_treatment_successes: int = Field(ge=0)
    overall_control_successes: int = Field(ge=0)
    overall_treatment_successes: int = Field(ge=0)
    latency_ratio: float | None = Field(default=None, ge=0)
    token_ratio: float | None = Field(default=None, ge=0)


# 判断arm audit是否具备capability publication所需的完整VALID evidence
def _arm_audit_is_complete_valid(audit: ArmAudit) -> bool:
    return (
        audit.status == "VALID"
        and audit.planned > 0
        and audit.planned
        == audit.started
        == audit.completed
        == audit.identity_verified
        and audit.runtime_failures == 0
        and audit.infrastructure_failures == 0
        and audit.trace_failures == 0
        and audit.grader_failures == 0
        and audit.required_artifact_evidence
        and audit.baseline_json_sha256 is not None
        and audit.baseline_markdown_sha256 is not None
    )


class ClassificationEvidence(_StrictModel):
    verdict: Literal["INVALID", "REJECT", "ACCEPT", "MIXED"]
    matches: dict[str, bool]
    invalid_predicates: dict[str, bool]
    reject_predicates: dict[str, bool]
    accept_predicates: dict[str, bool]
    mixed_predicates: dict[str, bool]

    @model_validator(mode="after")
    # 校验matches完整互斥且唯一true键与classifier verdict一致
    def _matches_are_exclusive(self) -> ClassificationEvidence:
        supported = frozenset(get_args(ClassificationVerdict))
        if (
            set(self.matches) != supported
            or any(type(value) is not bool for value in self.matches.values())
            or sum(value is True for value in self.matches.values()) != 1
            or not self.matches[self.verdict]
        ):
            raise ValueError("paired classification matches are invalid")
        return self


class PairedResult(_StrictModel):
    schema_version: Literal[1]
    result_id: SafeIdentifier
    created_at_utc: UtcTimestamp
    receipt_commit: GitCommit
    receipt_path: RelativeArtifactPath
    receipt_bytes: int = Field(gt=0)
    receipt_sha256: Sha256
    preflight_commit: GitCommit
    preflight_sha256: Sha256
    authorization_commit: GitCommit
    authorization_sha256: Sha256
    control: ArmAudit
    treatment: ArmAudit
    outcome: PairOutcome
    classifier: ClassificationEvidence
    verdict: CapabilityVerdict
    authorization_use_sha256: Sha256
    phase: Literal["paired_result_write"]
    transitions: list[PairTransition]
    control_child: ChildTerminationEvidence
    treatment_child: ChildTerminationEvidence
    provider_call_count: int = Field(ge=0)
    capability_delta_published: Literal[True]
    raw_provider_payload_persisted: Literal[False]
    raw_trace_visibility: Literal["private"]
    public_visibility: Literal["private_observer_result"]
    limitations: list[str]

    @model_validator(mode="after")
    # 校验capability result只表达双臂完整VALID后的canonical success terminal
    def _paired_result_is_terminal(self) -> PairedResult:
        _validate_terminal_history(self.transitions)
        final = self.transitions[-1]
        canonical_outcome = derive_pair_outcome(self.control, self.treatment)
        if (
            self.verdict != self.classifier.verdict
            or self.outcome != canonical_outcome
            or final.from_state != PairState.BOTH_VALID.value
            or final.event != PairEvent.FINALIZED.value
            or final.to_state != PairState.TERMINAL.value
            or not _arm_audit_is_complete_valid(self.control)
            or not _arm_audit_is_complete_valid(self.treatment)
        ):
            raise ValueError("paired result terminal evidence is invalid")
        return self


class _PairedResultManifest(_StrictModel):
    schema_version: Literal[1]
    json_sha256: Sha256
    markdown_sha256: Sha256


@dataclass(frozen=True)
class ResultPublication:
    durability_warning: bool


# 使用O_EXCL创建一次性授权记录并永久保留冲突证据
def reserve_authorization_use(
    path: Path | str,
    record: AuthorizationUseRecord,
) -> None:
    target = Path(path)
    if (
        not target.parent.is_dir()
        or target.parent.is_symlink()
        or target.parent.resolve(strict=True) != target.parent.absolute()
    ):
        raise ValueError("authorization use parent is not a verified directory")
    if os.path.lexists(target):
        raise ValueError("authorization use already reserved")
    payload = (canonical_json(record) + "\n").encode("utf-8")
    temporary = target.parent / f".{target.name}.{uuid.uuid4().hex}.tmp"
    descriptor: int | None = None
    published = False
    try:
        descriptor = os.open(
            temporary,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            0o600,
        )
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            descriptor = None
            written = stream.write(payload)
            if written != len(payload):
                raise ValueError("authorization use write is incomplete")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, target)
        published = True
        _fsync_directory(target.parent)
    except FileExistsError as exc:
        raise ValueError("authorization use already reserved") from exc
    except (OSError, ValueError) as exc:
        raise AuthorizationReservationError(consumed=published) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


# 验证临时或正式worktree观察值与冻结commit及路径边界一致
def validate_worktree_binding(
    observation: WorktreeObservation,
    *,
    expected_commit: str,
    repository: Path | str,
    output_parent: Path | str,
) -> None:
    try:
        worktree = observation.path.resolve(strict=True)
        source_root = observation.source_root.resolve(strict=True)
        repository_path = Path(repository).resolve(strict=True)
        output_path = Path(output_parent).resolve(strict=True)
    except OSError as exc:
        raise ValueError("worktree binding is invalid") from exc
    valid = (
        observation.registered
        and observation.detached
        and observation.clean
        and observation.observed_head == expected_commit
        and observation.profile_exists
        and observation.canonical_path_sha256
        == hashlib.sha256(str(worktree).encode("utf-8")).hexdigest()
        and source_root == (worktree / "src").resolve(strict=True)
        and source_root.is_relative_to(worktree)
        and observation.source_import.source_root_sha256
        == hashlib.sha256(str(source_root).encode("utf-8")).hexdigest()
        and observation.source_import.module_within_source_root
        and not _paths_overlap(worktree, repository_path)
        and not _paths_overlap(worktree, output_path)
    )
    if not valid:
        raise ValueError("worktree binding is invalid")


# 对control-first状态机执行receipt约束并返回可持久化的唯一transition
def reduce_pair_transition(
    state: PairState,
    event: PairEvent,
    *,
    receipt: PairedReceipt,
    authorization_use_reserved: bool,
) -> PairTransition:
    _validate_receipt_semantics(receipt)
    machine = receipt.execution_state_machine
    if (
        state is PairState.NOT_STARTED
        and event is PairEvent.PREFLIGHT_FAILED
        and not authorization_use_reserved
        and machine.control_preflight_failed.pair_status == "NOT_STARTED"
        and machine.control_preflight_failed.run_treatment is False
    ):
        return PairTransition.model_validate(
            {
                "from_state": state.value,
                "event": event.value,
                "to_state": PairState.NOT_STARTED.value,
            }
        )
    result = _POST_RESERVATION_TRANSITIONS.get((state, event))
    if result is None:
        raise ValueError("invalid pair transition")
    if result is not PairState.NOT_STARTED and not authorization_use_reserved:
        raise ValueError("invalid pair transition before authorization reservation")
    if event in {
        PairEvent.PRIVATE_EVIDENCE_FAILED,
        PairEvent.CONTROL_SPAWN_FAILED,
        PairEvent.CONTROL_INVALID,
    } and (
        machine.control_started_then_invalid_or_incomplete.pair_status != "INVALID"
        or machine.control_started_then_invalid_or_incomplete.run_treatment
    ):
        raise ValueError("receipt forbids control-invalid transition")
    if result is PairState.TREATMENT_RUNNING and (
        not machine.control_valid_and_complete.run_treatment
        or machine.control_valid_and_complete.progression_depends_on_control_capability_scores
    ):
        raise ValueError("receipt forbids treatment progression")
    if event in {
        PairEvent.TREATMENT_SPAWN_FAILED,
        PairEvent.TREATMENT_INVALID,
    } and (
        machine.treatment_started_then_invalid_or_incomplete.pair_status != "INVALID"
        or machine.treatment_started_then_invalid_or_incomplete.rerun
        or machine.treatment_started_then_invalid_or_incomplete.resume
    ):
        raise ValueError("receipt forbids treatment-invalid transition")
    if (
        event
        in {
            PairEvent.CLASSIFICATION_FAILED,
            PairEvent.RESULT_WRITE_FAILED,
            PairEvent.FINALIZED,
        }
        and result is PairState.TERMINAL
        and state is PairState.BOTH_VALID
        and (
        not machine.both_arms_valid_and_complete.evaluate_decision_contract
        or not machine.both_arms_valid_and_complete.publish_paired_capability_delta
        )
    ):
        raise ValueError("receipt forbids paired finalization")
    return PairTransition.model_validate(
        {
            "from_state": state.value,
            "event": event.value,
            "to_state": result.value,
        }
    )


# 返回receipt reducer产生的下一状态，供只关心状态的调用方复用
def transition_pair_state(
    state: PairState,
    event: PairEvent,
    *,
    receipt: PairedReceipt,
    authorization_use_reserved: bool,
) -> PairState:
    transition = reduce_pair_transition(
        state,
        event,
        receipt=receipt,
        authorization_use_reserved=authorization_use_reserved,
    )
    return PairState(transition.to_state)


# 比较between-arm完整证据并在任一漂移时fail closed
def revalidate_before_treatment(
    expected: BetweenArmEvidence,
    observed: BetweenArmEvidence,
) -> None:
    if expected != observed:
        raise ValueError("between-arm identity drift")
    required_flags = (
        observed.control_commit_exists,
        observed.treatment_commit_exists,
        observed.treatment_root_absent,
        observed.credential_present,
        observed.experiment_unchanged,
    )
    if not all(required_flags):
        raise ValueError("between-arm identity drift")


# 将baseline JSON读取为严格schema并验证其规范投影
def _load_baseline(path: Path) -> BaselineReport:
    try:
        text = path.read_text(encoding="utf-8")
        _strict_json_object(text)
        report = BaselineReport.model_validate_json(text)
        if text != render_json(report):
            raise ValueError("baseline JSON is not canonical")
        return report
    except (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        ValidationError,
        ValueError,
    ) as exc:
        raise ValueError("baseline evidence is invalid") from exc


# 严格读取单attempt公开report并验证其同源Markdown与manifest
def _load_evaluation_bundle(path: Path) -> EvaluationReport:
    try:
        report_path = path / "report.json"
        text = report_path.read_text(encoding="utf-8")
        _strict_json_object(text)
        report = EvaluationReport.model_validate_json(text)
        if text != render_evaluation_json(report):
            raise ValueError("evaluation report is not canonical")
        if (path / "report.md").read_text(
            encoding="utf-8"
        ) != render_evaluation_markdown(report):
            raise ValueError("evaluation markdown diverges")
        manifest = _strict_json_object(
            (path / "manifest.json").read_text(encoding="utf-8")
        )
        if manifest != {
            "artifact_version": 1,
            "task_id": report.task_id,
            "attempt_id": report.attempt_id,
        }:
            raise ValueError("evaluation manifest diverges")
        return report
    except (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        ValidationError,
        ValueError,
    ) as exc:
        raise ValueError("attempt artifact evidence is invalid") from exc


# 验证baseline row对应的Phase 8A report与关键runtime/private evidence存在
def _attempt_artifacts_are_complete(root: Path, row: AttemptAnalysis) -> bool:
    evaluation_root = (
        root
        / "run"
        / "tasks"
        / row.task_id
        / f"repeat-{row.repeat:02d}"
        / "evaluation"
    )
    try:
        report = _load_evaluation_bundle(evaluation_root)
    except ValueError:
        return False
    expected_metrics = {
        "task_success": row.task_success,
        "runtime_success": row.runtime_success,
        "step_count": row.step_count,
        "tool_count": row.tool_count,
        "retry_count": row.retry_count,
        "wall_latency_ms": row.wall_latency_ms,
        "token_usage": row.token_usage,
        "failure_category": row.failure_category,
    }
    if (
        report.task_id != row.task_id
        or report.task_success != row.task_success
        or report.runtime_success != row.runtime_success
        or report.trace_sanity_passed != row.trace_sanity_passed
        or report.failure_category != row.failure_category
        or report.metrics.model_dump() != {
            key: (
                value.model_dump()
                if isinstance(value, BaseModel)
                else value
            )
            for key, value in expected_metrics.items()
        }
    ):
        return False
    attempt_root = (
        evaluation_root / "attempts" / row.task_id / report.attempt_id
    )
    required = [
        attempt_root / "public" / "outcome.json",
        attempt_root / "public" / "metrics.json",
        attempt_root / "runtime" / "events.v2.jsonl",
        attempt_root / "runtime" / "trace.jsonl",
    ]
    if row.failure_category is not FailureCategory.TIMEOUT:
        required.extend(
            [
                attempt_root / "runtime" / "initial-workspace.json",
                attempt_root / "runtime" / "final-workspace.json",
                attempt_root / "runtime" / "workspace.diff",
                attempt_root / "private" / "grades.json",
                attempt_root / "private" / "command-results.json",
            ]
        )
    return all(path.is_file() and not path.is_symlink() for path in required)


# 计算不含timeout placeholder的完成attempt中位数
def _complete_medians(
    attempts: Sequence[AttemptAnalysis],
) -> tuple[float | None, float | None]:
    complete = [
        row for row in attempts if row.failure_category is not FailureCategory.TIMEOUT
    ]
    if not complete:
        return None, None
    latency = float(median(row.wall_latency_ms for row in complete))
    tokens = float(
        median(
            row.token_usage.input_tokens + row.token_usage.output_tokens
            for row in complete
        )
    )
    return latency, tokens


# 将canonical single-arm artifacts独立复审为能力无关的VALID或INVALID
def audit_arm_result(
    *,
    expected: ExpectedArmIdentity,
    exit_code: int | None,
    signal_number: int | None,
    output_root: Path | str,
    receipt: PairedReceipt,
) -> ArmAudit:
    root = Path(output_root)
    reasons: list[str] = []
    report: BaselineReport | None = None
    baseline_path = root / "baseline.json"
    declared_path = root / "declared-experiment.json"
    markdown_path = root / "baseline.md"
    if signal_number is not None or exit_code not in {0, 1}:
        reasons.append("child_exit")
    try:
        report = _load_baseline(baseline_path)
    except ValueError:
        reasons.append("required_artifact_evidence")
    if not declared_path.is_file():
        reasons.append("required_artifact_evidence")
    if report is not None and markdown_path.exists():
        try:
            if markdown_path.read_text(encoding="utf-8") != render_markdown(report):
                reasons.append("baseline_markdown")
        except (OSError, UnicodeDecodeError):
            reasons.append("baseline_markdown")
    attempts = [] if report is None else report.attempts
    planned = len(attempts)
    started = len(attempts)
    completed = len(attempts)
    identity_verified = (
        0 if report is None else report.experiment.verification.verified_attempts
    )
    if report is not None:
        if report.metrics != aggregate_attempts(attempts):
            reasons.append("baseline_metrics")
        if not all(_attempt_artifacts_are_complete(root, row) for row in attempts):
            reasons.append("required_artifact_evidence")
        expected_schedule = {
            (task_id, repeat)
            for task_id in expected.task_ids
            for repeat in range(1, expected.repeats + 1)
        }
        observed_schedule = {(row.task_id, row.repeat) for row in attempts}
        if (
            len(observed_schedule) != len(attempts)
            or observed_schedule != expected_schedule
            or planned != receipt.arm_validity.planned
        ):
            reasons.append("attempt_schedule")
        declared = report.experiment.declared
        observed = report.experiment.observed
        sdk = f"{declared.provider.sdk_distribution}=={declared.provider.sdk_version}"
        identity_matches = (
            declared.git.commit == expected.commit
            and declared.git.dirty is False
            and declared.profile_id == expected.profile_id
            and declared.profile_hash == expected.profile_hash
            and declared.prompt_hash == expected.prompt_sha256
            and declared.suite.suite_hash == expected.suite_sha256
            and declared.tool_schema_hash == expected.tool_schema_sha256
            and declared.runtime_config_hash == expected.runtime_config_sha256
            and declared.dependency.dependency_hash == expected.dependency_sha256
            and declared.provider.service_provider == expected.provider
            and declared.provider.model_id == expected.model
            and declared.provider.wire_protocol == expected.protocol
            and sdk == expected.sdk
            and declared.runtime.max_steps == expected.max_steps
            and declared.schedule.repeats == expected.repeats
            and report.experiment.status == "valid"
            and report.experiment.verification.status == "match"
            and not report.experiment.verification.mismatches
            and observed.provider == declared.provider
            and observed.prompt_hash == declared.prompt_hash
            and observed.tool_schema_hash == declared.tool_schema_hash
            and observed.runtime == declared.runtime
            and observed.runtime_config_hash == declared.runtime_config_hash
            and observed.attempts == planned
        )
        try:
            declared_payload = _strict_json_object(
                declared_path.read_text(encoding="utf-8")
            )
            identity_matches = identity_matches and declared_payload == declared.model_dump(
                mode="json"
            )
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
            identity_matches = False
        if not identity_matches:
            reasons.append("identity")
    runtime_failures = sum(
        row.failure_category
        in {FailureCategory.RUNTIME_FAILED, FailureCategory.CANCELLED}
        for row in attempts
    )
    infrastructure_failures = sum(
        row.failure_category is FailureCategory.INFRA_ERROR for row in attempts
    )
    trace_failures = sum(
        row.failure_category is FailureCategory.TRACE_INVALID
        for row in attempts
    )
    grader_failures = sum(
        row.failure_category is FailureCategory.GRADER_ERROR for row in attempts
    )
    timeouts = sum(
        row.failure_category is FailureCategory.TIMEOUT for row in attempts
    )
    validity = receipt.arm_validity
    if (
        runtime_failures > validity.maximum_runtime_failures
        or infrastructure_failures > validity.maximum_infrastructure_failures
        or trace_failures > validity.maximum_trace_failures
        or grader_failures > validity.maximum_grader_failures
    ):
        reasons.append("failure_limits")
    required_artifacts = (
        report is not None
        and declared_path.is_file()
        and "required_artifact_evidence" not in reasons
    )
    latency, tokens = _complete_medians(attempts)
    baseline_hash = _hash_file(baseline_path) if baseline_path.is_file() else None
    markdown_hash = _hash_file(markdown_path) if markdown_path.is_file() else None
    unique_reasons = list(dict.fromkeys(reasons))
    return ArmAudit(
        arm=expected.arm,
        status="VALID" if not unique_reasons else "INVALID",
        reasons=unique_reasons,
        exit_code=exit_code,
        signal_number=signal_number,
        planned=planned,
        started=started,
        completed=completed,
        identity_verified=identity_verified,
        runtime_failures=runtime_failures,
        infrastructure_failures=infrastructure_failures,
        trace_failures=trace_failures,
        grader_failures=grader_failures,
        timeouts=timeouts,
        provider_calls=0 if report is None else report.experiment.observed.api_calls,
        required_artifact_evidence=required_artifacts,
        overall_successes=sum(row.task_success for row in attempts),
        feature_successes=sum(
            row.task_success and row.category == "feature_implementation"
            for row in attempts
        ),
        bug_fixing_successes=sum(
            row.task_success and row.category == "bug_fixing" for row in attempts
        ),
        inventory_successes=sum(
            row.task_success
            and row.task_id == "feature-inventory-reservation-lifecycle"
            for row in attempts
        ),
        complete_median_latency_ms=latency,
        complete_median_input_output_tokens=tokens,
        baseline_json_sha256=baseline_hash,
        baseline_markdown_sha256=markdown_hash,
        attempts=attempts,
    )


# 将两臂审计唯一机械派生为classifier输入而不做阈值判断
def derive_pair_outcome(control: ArmAudit, treatment: ArmAudit) -> PairOutcome:
    latency_ratio = None
    token_ratio = None
    if (
        control.complete_median_latency_ms
        and treatment.complete_median_latency_ms is not None
    ):
        latency_ratio = (
            treatment.complete_median_latency_ms / control.complete_median_latency_ms
        )
    if (
        control.complete_median_input_output_tokens
        and treatment.complete_median_input_output_tokens is not None
    ):
        token_ratio = (
            treatment.complete_median_input_output_tokens
            / control.complete_median_input_output_tokens
        )
    return PairOutcome(
        control=control,
        treatment=treatment,
        required_artifact_evidence=(
            control.required_artifact_evidence
            and treatment.required_artifact_evidence
        ),
        inventory_control_successes=control.inventory_successes,
        inventory_treatment_successes=treatment.inventory_successes,
        feature_control_successes=control.feature_successes,
        feature_treatment_successes=treatment.feature_successes,
        overall_control_successes=control.overall_successes,
        overall_treatment_successes=treatment.overall_successes,
        latency_ratio=latency_ratio,
        token_ratio=token_ratio,
    )


# 根据receipt自身阈值重新计算完整互斥classification evidence
def recompute_classification(
    receipt: PairedReceipt,
    outcome: PairOutcome,
) -> ClassificationEvidence:
    validity = receipt.arm_validity
    primary = receipt.primary_comparison
    guardrails = receipt.hard_guardrails
    efficiency = receipt.efficiency_comparison
    arms = (outcome.control, outcome.treatment)
    invalid_predicates = {
        "either_arm_status_not_valid": any(
            arm.status != validity.required_status for arm in arms
        ),
        "either_arm_attempt_counts_not_exact": any(
            arm.planned != validity.planned
            or arm.started != validity.started
            or arm.completed != validity.completed
            for arm in arms
        ),
        "either_arm_identity_verified_count_not_exact": any(
            arm.identity_verified != validity.identity_verified for arm in arms
        ),
        "either_arm_runtime_failures_above_maximum": any(
            arm.runtime_failures > validity.maximum_runtime_failures for arm in arms
        ),
        "either_arm_infrastructure_failures_above_maximum": any(
            arm.infrastructure_failures > validity.maximum_infrastructure_failures
            for arm in arms
        ),
        "either_arm_trace_failures_above_maximum": any(
            arm.trace_failures > validity.maximum_trace_failures for arm in arms
        ),
        "either_arm_grader_failures_above_maximum": any(
            arm.grader_failures > validity.maximum_grader_failures for arm in arms
        ),
        "required_artifact_evidence_missing": not outcome.required_artifact_evidence,
    }
    invalid = any(
        invalid_predicates[name]
        for name, enabled in receipt.decision_contract.invalid_if.items()
        if enabled
    )
    inventory_delta = (
        outcome.inventory_treatment_successes - outcome.inventory_control_successes
    )
    feature_delta = (
        outcome.feature_treatment_successes - outcome.feature_control_successes
    )
    overall_delta = (
        outcome.overall_treatment_successes - outcome.overall_control_successes
    )
    reject_predicates = {
        "inventory_treatment_below_minimum": outcome.inventory_treatment_successes
        < primary.inventory_lifecycle.treatment_minimum_successes,
        "inventory_delta_below_minimum": inventory_delta
        < primary.inventory_lifecycle.treatment_minus_control_minimum,
        "feature_delta_below_minimum": feature_delta
        < primary.feature_implementation.treatment_minus_control_minimum,
        "overall_delta_below_minimum": overall_delta
        < primary.overall.treatment_minus_control_minimum,
        "control_bug_fixing_not_required_value": outcome.control.bug_fixing_successes
        != guardrails.control_bug_fixing_successes_required,
        "treatment_bug_fixing_not_required_value": outcome.treatment.bug_fixing_successes
        != guardrails.treatment_bug_fixing_successes_required,
        "either_arm_timeouts_above_maximum": any(
            arm.timeouts > guardrails.maximum_timeouts_per_arm for arm in arms
        ),
        "treatment_timeouts_exceed_control": (
            guardrails.treatment_timeouts_must_not_exceed_control
            and outcome.treatment.timeouts > outcome.control.timeouts
        ),
    }
    reject = any(
        reject_predicates[name]
        for name, enabled in receipt.decision_contract.reject_if_both_arms_valid_and_any.items()
        if enabled
    )
    inventory_pass = not (
        reject_predicates["inventory_treatment_below_minimum"]
        or reject_predicates["inventory_delta_below_minimum"]
    )
    accept_predicates = {
        "no_reject_condition": not reject,
        "inventory_primary_pass": inventory_pass,
        "feature_primary_pass": (
            outcome.feature_treatment_successes
            >= primary.feature_implementation.treatment_minimum_successes
            and not reject_predicates["feature_delta_below_minimum"]
        ),
        "overall_primary_pass": (
            outcome.overall_treatment_successes
            >= primary.overall.treatment_minimum_successes
            and not reject_predicates["overall_delta_below_minimum"]
        ),
        "latency_ratio_at_or_below_maximum": outcome.latency_ratio is not None
        and outcome.latency_ratio
        <= efficiency.maximum_treatment_to_control_complete_median_latency_ratio,
        "token_ratio_at_or_below_maximum": outcome.token_ratio is not None
        and outcome.token_ratio
        <= efficiency.maximum_treatment_to_control_complete_median_input_output_token_ratio,
    }
    accept = all(
        accept_predicates[name]
        for name, enabled in receipt.decision_contract.accept_if_both_arms_valid_and_all.items()
        if enabled
    )
    mixed_predicates = {
        "inventory_primary_pass": inventory_pass,
        "no_reject_condition": not reject,
        "accept_condition_false": not accept,
    }
    mixed = all(
        mixed_predicates[name]
        for name, enabled in receipt.decision_contract.mixed_if_both_arms_valid_and_all.items()
        if enabled
    )
    matches = {
        "INVALID": invalid,
        "REJECT": not invalid and reject,
        "ACCEPT": not invalid and not reject and accept,
        "MIXED": not invalid and not reject and not accept and mixed,
    }
    if sum(matches.values()) != 1:
        raise ValueError("paired decision contract did not produce one verdict")
    verdict = select_classification_verdict(
        receipt.decision_contract.classification_order,
        matches,
    )
    return ClassificationEvidence(
        verdict=verdict,
        matches=matches,
        invalid_predicates=invalid_predicates,
        reject_predicates=reject_predicates,
        accept_predicates=accept_predicates,
        mixed_predicates=mixed_predicates,
    )


# 保留production classifier入口并委托唯一receipt-bound重算实现
def classify_pair(
    receipt: PairedReceipt,
    outcome: PairOutcome,
) -> ClassificationEvidence:
    return recompute_classification(receipt, outcome)


# 构建post-reservation失败的唯一脱敏terminal record
def build_terminal_record(
    *,
    terminal_id: str,
    created_at_utc: str,
    phase: TerminalPhase,
    receipt_sha256: str,
    preflight_sha256: str,
    authorization_sha256: str,
    authorization_use_sha256: str,
    transitions: list[PairTransition],
    failure_category: TerminalFailureCategory,
    control: ArmAudit | None = None,
    treatment: ArmAudit | None = None,
    control_child: ChildTerminationEvidence | None = None,
    treatment_child: ChildTerminationEvidence | None = None,
) -> PairTerminalRecord:
    provider_calls = sum(
        audit.provider_calls for audit in (control, treatment) if audit is not None
    )
    return PairTerminalRecord(
        schema_version=1,
        terminal_id=terminal_id,
        created_at_utc=created_at_utc,
        status="INVALID",
        phase=phase,
        receipt_sha256=receipt_sha256,
        preflight_sha256=preflight_sha256,
        authorization_sha256=authorization_sha256,
        authorization_use_sha256=authorization_use_sha256,
        transitions=transitions,
        control=control,
        treatment=treatment,
        control_child=control_child,
        treatment_child=treatment_child,
        provider_call_count=provider_calls,
        capability_delta_published=False,
        private_visibility=True,
        failure_category=failure_category,
    )


# 以create-once canonical文件持久化failure terminal evidence
def write_terminal_record(
    path: Path | str,
    record: PairTerminalRecord,
) -> None:
    write_canonical_artifact(path, record)


# 严格回读failure terminal并重新触发artifact-specific semantic validation
def read_terminal_record(path: Path | str) -> PairTerminalRecord:
    return read_strict_artifact(path, PairTerminalRecord)


# 构建唯一canonical paired result并冻结公开claim边界
def build_paired_result(
    *,
    result_id: str,
    created_at_utc: str,
    receipt_reference: ReceiptReference,
    preflight_commit: str,
    preflight_sha256: str,
    authorization_commit: str,
    authorization_sha256: str,
    authorization_use_sha256: str,
    receipt: PairedReceipt,
    control: ArmAudit,
    treatment: ArmAudit,
    control_child: ChildTerminationEvidence,
    treatment_child: ChildTerminationEvidence,
    transitions: list[PairTransition],
) -> PairedResult:
    outcome = derive_pair_outcome(control, treatment)
    classifier = recompute_classification(receipt, outcome)
    return PairedResult(
        schema_version=1,
        result_id=result_id,
        created_at_utc=created_at_utc,
        receipt_commit=receipt_reference.commit,
        receipt_path=receipt_reference.path,
        receipt_bytes=receipt_reference.bytes,
        receipt_sha256=receipt_reference.sha256,
        preflight_commit=preflight_commit,
        preflight_sha256=preflight_sha256,
        authorization_commit=authorization_commit,
        authorization_sha256=authorization_sha256,
        authorization_use_sha256=authorization_use_sha256,
        phase="paired_result_write",
        control=control,
        treatment=treatment,
        control_child=control_child,
        treatment_child=treatment_child,
        outcome=outcome,
        classifier=classifier,
        verdict=cast(CapabilityVerdict, classifier.verdict),
        transitions=transitions,
        provider_call_count=control.provider_calls + treatment.provider_calls,
        capability_delta_published=True,
        raw_provider_payload_persisted=False,
        raw_trace_visibility="private",
        public_visibility="private_observer_result",
        limitations=[
            "Fixed-task internal benchmark; not SWE-bench.",
            "Small repeated sample; not statistically significant.",
            "Process isolation is not a security sandbox.",
        ],
    )


# 从PairedResult同一模型生成不含raw payload的Markdown投影
def render_paired_markdown(result: PairedResult) -> str:
    rows = [
        "# KamaClaude Paired Observer Result",
        "",
        f"- Verdict: `{result.verdict}`",
        f"- Control status: `{result.control.status}`",
        f"- Treatment status: `{result.treatment.status}`",
        f"- Control successes: {result.outcome.overall_control_successes}",
        f"- Treatment successes: {result.outcome.overall_treatment_successes}",
        f"- Provider calls: {result.provider_call_count}",
        f"- Receipt commit: `{result.receipt_commit}`",
        f"- Receipt hash: `{result.receipt_sha256}`",
        "",
        "## Limitations",
        "",
    ]
    rows.extend(f"- {item}" for item in result.limitations)
    return "\n".join(rows) + "\n"


# 以单次unbuffered exact-length write写入并fsync一个private artifact
def _write_private_bytes(path: Path, payload: bytes) -> None:
    descriptor: int | None = None
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        written = os.write(descriptor, payload)
        if written != len(payload):
            raise ValueError("paired result artifact write was incomplete")
        os.fsync(descriptor)
    finally:
        if descriptor is not None:
            os.close(descriptor)


# 通过同父目录staging和atomic rename发布完整paired result bundle
def write_paired_result(
    output_root: Path | str,
    result: PairedResult,
    *,
    repository: Path | str,
    expected_receipt: ReceiptReference,
) -> ResultPublication:
    root = Path(output_root)
    if root.exists() or os.path.lexists(root):
        raise ValueError("paired result target already exists")
    parent = root.parent
    if not parent.is_dir() or parent.is_symlink():
        raise ValueError("paired result parent is invalid")
    staging = root.with_name(f".{root.name}.staging")
    if staging.exists() or os.path.lexists(staging):
        raise ValueError("paired result staging already exists")
    staging.mkdir(mode=0o700)
    json_path = staging / "paired-result.json"
    markdown_path = staging / "paired-result.md"
    manifest_path = staging / "manifest.json"
    json_payload = (canonical_json(result.model_dump(mode="json")) + "\n").encode(
        "utf-8"
    )
    markdown = render_paired_markdown(result)
    markdown_payload = markdown.encode("utf-8")
    _write_private_bytes(json_path, json_payload)
    _write_private_bytes(markdown_path, markdown_payload)
    manifest = _PairedResultManifest(
        schema_version=1,
        json_sha256=hashlib.sha256(json_payload).hexdigest(),
        markdown_sha256=hashlib.sha256(markdown_payload).hexdigest(),
    )
    manifest_payload = (
        canonical_json(manifest.model_dump(mode="json")) + "\n"
    ).encode("utf-8")
    _write_private_bytes(manifest_path, manifest_payload)
    _read_paired_result_bundle(
        staging,
        repository=repository,
        expected_receipt=expected_receipt,
    )
    _fsync_directory(staging)
    try:
        os.rename(staging, root)
    except OSError as exc:
        raise ValueError("paired result cannot be atomically published") from exc
    try:
        _fsync_directory(parent)
    except ValueError:
        return ResultPublication(durability_warning=True)
    return ResultPublication(durability_warning=False)


# 用冻结receipt重算并绑定result中的outcome、classifier与verdict
def _validate_paired_result_against_receipt(
    result: PairedResult,
    *,
    repository: Path | str,
    expected_receipt: ReceiptReference,
) -> None:
    receipt = _load_receipt_from_reference(repository, expected_receipt)
    if (
        result.receipt_commit != expected_receipt.commit
        or result.receipt_path != expected_receipt.path
        or result.receipt_bytes != expected_receipt.bytes
        or result.receipt_sha256 != expected_receipt.sha256
    ):
        raise ValueError("paired result receipt identity is invalid")
    canonical_outcome = derive_pair_outcome(result.control, result.treatment)
    canonical_classifier = recompute_classification(receipt, canonical_outcome)
    if (
        result.outcome != canonical_outcome
        or result.classifier != canonical_classifier
        or result.verdict != canonical_classifier.verdict
    ):
        raise ValueError("paired result receipt-bound evidence is invalid")


# 严格读取单个paired-result JSON并执行receipt-bound语义重算
def read_paired_result_json(
    path: Path | str,
    *,
    repository: Path | str,
    expected_receipt: ReceiptReference,
) -> PairedResult:
    result = read_strict_artifact(path, PairedResult)
    _validate_paired_result_against_receipt(
        result,
        repository=repository,
        expected_receipt=expected_receipt,
    )
    return result


# 严格回读完整bundle并用冻结receipt重算语义供publisher和final reader复用
def _read_paired_result_bundle(
    output_root: Path | str,
    *,
    repository: Path | str,
    expected_receipt: ReceiptReference,
) -> PairedResult:
    root = Path(output_root)
    try:
        json_path = root / "paired-result.json"
        markdown_path = root / "paired-result.md"
        manifest = read_strict_artifact(
            root / "manifest.json",
            _PairedResultManifest,
        )
        result = read_paired_result_json(
            json_path,
            repository=repository,
            expected_receipt=expected_receipt,
        )
        markdown = markdown_path.read_text(encoding="utf-8")
        if (
            manifest.json_sha256
            != hashlib.sha256(json_path.read_bytes()).hexdigest()
            or manifest.markdown_sha256
            != hashlib.sha256(markdown.encode("utf-8")).hexdigest()
            or markdown != render_paired_markdown(result)
        ):
            raise ValueError("paired result markdown diverges")
        return result
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise ValueError("paired result bundle is invalid") from exc


# 只接受final目录并用指定冻结receipt重算全部claim-bearing evidence
def read_paired_result_bundle(
    output_root: Path | str,
    *,
    repository: Path | str,
    expected_receipt: ReceiptReference,
) -> PairedResult:
    root = Path(output_root)
    if root.name.startswith(".") and root.name.endswith(".staging"):
        raise ValueError("paired result bundle is invalid")
    return _read_paired_result_bundle(
        root,
        repository=repository,
        expected_receipt=expected_receipt,
    )


# 验证一次已消费paired execution最终只能存在success或failure一种terminal
def validate_pair_terminal_exclusivity(
    paths: PrivateEvidencePaths,
    *,
    authorization_consumed: bool,
    execution_complete: bool,
    repository: Path | str | None = None,
    expected_receipt: ReceiptReference | None = None,
) -> Literal["success", "failure", "pending"]:
    reservation_exists = paths.authorization_use.is_file()
    if reservation_exists != authorization_consumed:
        raise ValueError("paired terminal authorization state is invalid")
    success = False
    failure = False
    if paths.paired_result.exists() or os.path.lexists(paths.paired_result):
        if repository is None or expected_receipt is None:
            raise ValueError("paired result receipt is unavailable")
        read_paired_result_bundle(
            paths.paired_result,
            repository=repository,
            expected_receipt=expected_receipt,
        )
        success = True
    if paths.terminal_record.exists() or os.path.lexists(paths.terminal_record):
        read_terminal_record(paths.terminal_record)
        failure = True
    if success and failure:
        raise ValueError("paired terminal evidence is not exclusive")
    if success:
        return "success"
    if failure:
        return "failure"
    if not execution_complete:
        return "pending"
    raise ValueError("paired terminal evidence is not exclusive")


# 验证独立授权artifact与receipt和preflight冻结引用完全一致
def validate_execution_authorization(
    authorization: ExecutionAuthorizationArtifact,
    *,
    preflight: FinalPreflightArtifact,
    receipt: PairedReceipt,
    receipt_sha256: str,
    preflight_commit: str,
    preflight_sha256: str,
) -> None:
    expected = (
        authorization.paired_receipt.commit == preflight.paired_receipt.commit
        and authorization.paired_receipt.sha256 == receipt_sha256
        and authorization.final_preflight.commit == preflight_commit
        and authorization.final_preflight.sha256 == preflight_sha256
        and authorization.control_commit == receipt.arms.control.commit
        and authorization.treatment_commit == receipt.arms.treatment.commit
        and authorization.provider == receipt.shared_identity.provider
        and authorization.model == receipt.shared_identity.model
        and authorization.protocol == receipt.shared_identity.protocol
        and authorization.attempts.control == receipt.execution_plan.attempts_per_arm
        and authorization.attempts.treatment == receipt.execution_plan.attempts_per_arm
        and authorization.attempts.total == receipt.execution_plan.total_attempts
        and authorization.maximum_authorized_attempts
        == receipt.execution_plan.total_attempts
        and authorization.output_parent_sha256
        == preflight.external_parent.canonical_path_sha256
        and authorization.logical_basenames.control
        == receipt.execution_plan.control_output_logical_root
        and authorization.logical_basenames.treatment
        == receipt.execution_plan.treatment_output_logical_root
    )
    if not expected:
        raise ValueError("authorization identity mismatch")


# 将已验证observer evidence组装为唯一strict final-preflight artifact
def build_final_preflight_artifact(
    *,
    preflight_id: str,
    created_at_utc: str,
    generator: GitSnapshot,
    paired_receipt: ReceiptReference,
    arms: PreflightArms,
    environment: EnvironmentSnapshot,
    shared_identity: SharedPreflightIdentity,
    external_parent: OutputParentEvidence,
    logical_roots: LogicalRootsEvidence,
    commands: CommandHashes,
    credential: CredentialEvidence,
    receipt: PairedReceipt,
) -> FinalPreflightArtifact:
    control = receipt.arms.control
    treatment = receipt.arms.treatment
    identities_match = (
        paired_receipt.authorization_remains_false_zero
        and arms.control.commit == control.commit
        and arms.control.profile_path == control.profile_path
        and arms.control.profile_id == control.profile_id
        and arms.control.profile_file_sha256 == control.profile_file_sha256
        and arms.control.profile_canonical_sha256
        == control.profile_canonical_sha256
        and arms.control.prompt_sha256 == control.prompt_sha256
        and arms.treatment.commit == treatment.commit
        and arms.treatment.profile_path == treatment.profile_path
        and arms.treatment.profile_id == treatment.profile_id
        and arms.treatment.profile_file_sha256 == treatment.profile_file_sha256
        and arms.treatment.profile_canonical_sha256
        == treatment.profile_canonical_sha256
        and arms.treatment.prompt_sha256 == treatment.prompt_sha256
        and shared_identity.provider == receipt.shared_identity.provider
        and shared_identity.model == receipt.shared_identity.model
        and shared_identity.protocol == receipt.shared_identity.protocol
        and shared_identity.suite_sha256 == receipt.shared_identity.suite_sha256
        and shared_identity.tool_schema_sha256
        == receipt.shared_identity.tool_schema_sha256
        and shared_identity.runtime_config_sha256
        == receipt.shared_identity.runtime_config_sha256
        and shared_identity.dependency_sha256
        == receipt.shared_identity.dependency_sha256
        and shared_identity.max_steps == receipt.shared_identity.max_steps
        and shared_identity.repeats == receipt.shared_identity.repeats
        and shared_identity.mcp_enabled == receipt.shared_identity.mcp_enabled
        and environment.python_version == receipt.host_policy.python
        and environment.os == receipt.host_policy.os
        and environment.architecture == receipt.host_policy.architecture
        and environment.sdk_distribution + "==" + environment.sdk_version
        == receipt.shared_identity.sdk
        and environment.dependency_sha256 == shared_identity.dependency_sha256
        and logical_roots.control
        == receipt.execution_plan.control_output_logical_root
        and logical_roots.treatment
        == receipt.execution_plan.treatment_output_logical_root
    )
    if not identities_match:
        raise ValueError("final preflight identity mismatch")
    return FinalPreflightArtifact(
        schema_version=1,
        preflight_id=preflight_id,
        status="READY_AWAITING_EXECUTION_AUTHORIZATION",
        created_at_utc=created_at_utc,
        generator=generator,
        paired_receipt=paired_receipt,
        arms=arms,
        environment=environment,
        shared_identity=shared_identity,
        external_parent=external_parent,
        logical_roots=logical_roots,
        commands=commands,
        credential=credential,
        network=NetworkEvidence(provider_calls=0, paid_smoke_calls=0),
        authorization="AWAITING_SEPARATE_EXECUTION_AUTHORIZATION",
    )
