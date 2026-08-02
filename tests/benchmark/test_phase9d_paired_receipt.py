from __future__ import annotations

import ast
import copy
import hashlib
import json
import platform
import subprocess
import traceback
import uuid
from collections.abc import Iterable
from importlib.metadata import version
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Never

import pytest

from kama_claude.benchmark.experiment import (
    ExperimentProfile,
    RepositoryIdentity,
    capture_declared_identity,
    load_experiment_profile,
)
from kama_claude.benchmark.schema import load_suite

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_RECEIPT_PATH = (
    _REPOSITORY_ROOT
    / "benchmarks"
    / "receipts"
    / "phase9d-repaired-v1-v2-paired-experiment.json"
)
_C1_COMMIT = "7e77478f988ca61cb0087a06c686c416a27544c3"
_C2_COMMIT = "f67c653b21cce5338c5cca76a3fa076371b015b9"
_SHARED_REPAIRED_STACK_COMMIT = "61d0266812034c9af9dd8a395b83b3da0965202f"
_C1_PROFILE_PATH = (
    "benchmarks/experiments/"
    "kama-coding-mvp-v1-deepseek-v4-pro-repaired-v1-control.json"
)
_C2_PROFILE_PATH = (
    "benchmarks/experiments/"
    "kama-coding-mvp-v1-deepseek-v4-pro-repaired-v2-treatment.json"
)
_LOOP_PATH = "src/kama_claude/core/loop.py"
_RUNNER_PATH = "src/kama_claude/core/runner.py"
_RUNNER_SHA256 = "4effa46bd0ff9dfb01aaef37bcd80a77a9cc970a9031b7fb87f0c5490c32f9f7"
_LIFECYCLE_CONTRACT_PATHS = (
    "src/kama_claude/eval/graders.py",
    "tests/unit/test_runner.py",
    "tests/eval/test_llm_error_lifecycle_contract.py",
    "tests/eval/test_phase8a_graders.py",
    "tests/eval/test_timeout_lifecycle_contract.py",
)
_TOP_LEVEL_KEYS = {
    "schema_version",
    "receipt_id",
    "status",
    "shared_repaired_stack_commit",
    "arms",
    "single_variable_contract",
    "shared_identity",
    "host_policy",
    "execution_plan",
    "arm_validity",
    "primary_comparison",
    "hard_guardrails",
    "secondary_reporting",
    "efficiency_comparison",
    "execution_state_machine",
    "decision_contract",
    "authorization",
    "immutability_policy",
    "logical_root_policy",
}


# 拒绝任意层级 JSON object 中的重复键
def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key") from None
        result[key] = value
    return result


# 拒绝标准 JSON 之外的非有限数值常量
def _reject_non_finite(_value: str) -> Never:
    raise ValueError("non-finite JSON constant")


# 使用拒绝重复键的标准库解析器读取 receipt object
def _strict_json_object(text: str | bytes) -> dict[str, Any]:
    if isinstance(text, bytes):
        text = text.decode("utf-8", errors="strict")
    value = json.loads(
        text,
        object_pairs_hook=_reject_duplicate_keys,
        parse_constant=_reject_non_finite,
    )
    if not isinstance(value, dict):
        raise ValueError("receipt root must be a JSON object")
    return value


# 读取当前候选 receipt，并把缺失收敛为明确 RED
def _receipt() -> dict[str, Any]:
    assert _RECEIPT_PATH.is_file(), "paired preregistration receipt is missing"
    return _strict_json_object(_RECEIPT_PATH.read_text(encoding="utf-8"))


# 从指定 Git commit 读取原始 blob bytes
def _git_blob(commit: str, path: str) -> bytes:
    return subprocess.run(
        ["git", "show", f"{commit}:{path}"],
        cwd=_REPOSITORY_ROOT,
        check=True,
        capture_output=True,
    ).stdout


# 返回指定 commit 的唯一直接 parent
def _git_parent(commit: str) -> str:
    return subprocess.run(
        ["git", "rev-parse", f"{commit}^"],
        cwd=_REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


# 返回两个 commit 在指定 paths 下的排序 changed-file 集合
def _git_changed_paths(
    older: str,
    newer: str,
    paths: Iterable[str],
) -> tuple[str, ...]:
    result = subprocess.run(
        ["git", "diff", "--name-only", older, newer, "--", *paths],
        cwd=_REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return tuple(line for line in result.stdout.splitlines() if line)


# 对 bytes 计算 SHA-256
def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


# 对 JSON-compatible value 计算排序键且紧凑编码的 canonical SHA-256
def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return _sha256(payload)


# 从 production AST 唯一提取 context.system_prompt 的默认常量
def _extract_default_prompt(source: bytes) -> str:
    tree = ast.parse(source.decode("utf-8"))
    prompts = [
        ast.literal_eval(node.args[0])
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "system_prompt"
    ]
    assert len(prompts) == 1
    assert isinstance(prompts[0], str)
    return prompts[0]


# 将唯一 default prompt constant 替换为 sentinel 后序列化完整 module AST
def _normalized_loop_ast(source: bytes) -> str:
    tree = ast.parse(source.decode("utf-8"))
    count = 0

    class _PromptSentinel(ast.NodeTransformer):
        # 仅替换 context.system_prompt 的第一个位置参数
        def visit_Call(self, node: ast.Call) -> ast.AST:
            nonlocal count
            self.generic_visit(node)
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "system_prompt"
            ):
                count += 1
                node.args[0] = ast.Constant(value="PROMPT_SENTINEL")
            return node

    _PromptSentinel().visit(tree)
    assert count == 1
    return ast.dump(tree, include_attributes=False)


# 递归收集 receipt 的全部 object keys
def _all_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return set(value) | {
            nested
            for child in value.values()
            for nested in _all_keys(child)
        }
    if isinstance(value, list):
        return {nested for child in value for nested in _all_keys(child)}
    return set()


# 递归返回 receipt 中全部字符串值
def _all_string_values(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, dict):
        return tuple(
            nested
            for child in value.values()
            for nested in _all_string_values(child)
        )
    if isinstance(value, list):
        return tuple(
            nested for child in value for nested in _all_string_values(child)
        )
    return ()


# 返回一个满足全部 ACCEPT 前置条件的成对实验结果
def _accepting_pair_outcome() -> dict[str, Any]:
    arm = {
        "status": "VALID",
        "planned": 27,
        "started": 27,
        "completed": 27,
        "identity_verified": 27,
        "runtime_failures": 0,
        "infrastructure_failures": 0,
        "trace_failures": 0,
        "grader_failures": 0,
        "timeouts": 1,
        "bug_fixing_successes": 9,
    }
    return {
        "control": copy.deepcopy(arm),
        "treatment": copy.deepcopy(arm),
        "required_artifact_evidence": True,
        "inventory_control_successes": 0,
        "inventory_treatment_successes": 1,
        "feature_control_successes": 6,
        "feature_treatment_successes": 6,
        "overall_control_successes": 20,
        "overall_treatment_successes": 20,
        "latency_ratio": 1.0,
        "token_ratio": 1.0,
        "atomic_oracle_disposition": "CONTRACT_AMBIGUITY_NOTE",
        "analysis_only_result": "unchanged",
    }


# 深复制 mapping 并设置指定嵌套路径的值
def _with_nested_value(
    value: dict[str, Any],
    path: tuple[str, ...],
    replacement: object,
) -> dict[str, Any]:
    mutated = copy.deepcopy(value)
    cursor: dict[str, Any] = mutated
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = replacement
    return mutated


# 返回两个同构 JSON object 之间发生变化的全部 leaf paths
def _changed_leaf_paths(
    before: object,
    after: object,
    prefix: tuple[str, ...] = (),
) -> set[tuple[str, ...]]:
    if isinstance(before, dict) and isinstance(after, dict):
        assert set(before) == set(after)
        return {
            changed
            for key in before
            for changed in _changed_leaf_paths(before[key], after[key], (*prefix, key))
        }
    if before == after:
        return set()
    return {prefix}


# 深复制 object 并在核对旧值后应用一组明确的 leaf mutations
def _with_verified_mutations(
    value: dict[str, Any],
    mutations: tuple[tuple[tuple[str, ...], object, object], ...],
) -> dict[str, Any]:
    paths = tuple(path for path, _old, _new in mutations)
    assert len(paths) == len(set(paths))
    mutated = copy.deepcopy(value)
    for path, expected_old, replacement in mutations:
        cursor: dict[str, Any] = mutated
        for key in path[:-1]:
            child = cursor[key]
            assert isinstance(child, dict)
            cursor = child
        assert cursor[path[-1]] == expected_old
        assert replacement != expected_old
        cursor[path[-1]] = replacement
    assert _changed_leaf_paths(value, mutated) == set(paths)
    return mutated


# 断言 tracked execution state machine 与冻结控制流逐字段一致
def _validate_execution_state_machine(receipt: dict[str, Any]) -> None:
    assert receipt["execution_state_machine"] == {
        "control_preflight_failed": {
            "create_control_output_root": False,
            "maximum_api_calls": 0,
            "run_treatment": False,
            "pair_status": "NOT_STARTED",
        },
        "control_started_then_invalid_or_incomplete": {
            "preserve_control_artifacts": True,
            "run_treatment": False,
            "rerun": False,
            "resume": False,
            "pair_status": "INVALID",
            "publish_paired_capability_delta": False,
        },
        "control_valid_and_complete": {
            "run_treatment": True,
            "progression_depends_on_control_capability_scores": False,
            "allow_experiment_changes_between_arms": False,
        },
        "treatment_started_then_invalid_or_incomplete": {
            "preserve_all_artifacts": True,
            "rerun": False,
            "resume": False,
            "pair_status": "INVALID",
            "publish_paired_capability_delta": False,
        },
        "both_arms_valid_and_complete": {
            "evaluate_decision_contract": True,
            "publish_paired_capability_delta": True,
        },
    }


# 断言 deterministic decision contract 的结构和优先级保持冻结
def _validate_decision_contract(receipt: dict[str, Any]) -> None:
    assert receipt["decision_contract"] == {
        "classification_order": ["INVALID", "REJECT", "ACCEPT", "MIXED"],
        "invalid_if": {
            "either_arm_status_not_valid": True,
            "either_arm_attempt_counts_not_exact": True,
            "either_arm_identity_verified_count_not_exact": True,
            "either_arm_runtime_failures_above_maximum": True,
            "either_arm_infrastructure_failures_above_maximum": True,
            "either_arm_trace_failures_above_maximum": True,
            "either_arm_grader_failures_above_maximum": True,
            "required_artifact_evidence_missing": True,
        },
        "reject_if_both_arms_valid_and_any": {
            "inventory_treatment_below_minimum": True,
            "inventory_delta_below_minimum": True,
            "feature_delta_below_minimum": True,
            "overall_delta_below_minimum": True,
            "control_bug_fixing_not_required_value": True,
            "treatment_bug_fixing_not_required_value": True,
            "either_arm_timeouts_above_maximum": True,
            "treatment_timeouts_exceed_control": True,
        },
        "accept_if_both_arms_valid_and_all": {
            "no_reject_condition": True,
            "inventory_primary_pass": True,
            "feature_primary_pass": True,
            "overall_primary_pass": True,
            "latency_ratio_at_or_below_maximum": True,
            "token_ratio_at_or_below_maximum": True,
        },
        "mixed_if_both_arms_valid_and_all": {
            "inventory_primary_pass": True,
            "no_reject_condition": True,
            "accept_condition_false": True,
        },
        "secondary_reporting_affects_classification": False,
    }


# 依据 receipt 自身阈值计算四种分类谓词是否成立
def _classification_matches(
    receipt: dict[str, Any],
    outcome: dict[str, Any],
) -> dict[str, bool]:
    _validate_decision_contract(receipt)
    contract = receipt["decision_contract"]
    validity = receipt["arm_validity"]
    primary = receipt["primary_comparison"]
    guardrails = receipt["hard_guardrails"]
    efficiency = receipt["efficiency_comparison"]
    arms = (outcome["control"], outcome["treatment"])
    expected_count = validity["planned"]

    invalid_predicates = {
        "either_arm_status_not_valid": any(
            arm["status"] != validity["required_status"] for arm in arms
        ),
        "either_arm_attempt_counts_not_exact": any(
            arm[field] != validity[field]
            for arm in arms
            for field in ("planned", "started", "completed")
        ),
        "either_arm_identity_verified_count_not_exact": any(
            arm["identity_verified"] != validity["identity_verified"] for arm in arms
        ),
        "either_arm_runtime_failures_above_maximum": any(
            arm["runtime_failures"] > validity["maximum_runtime_failures"]
            for arm in arms
        ),
        "either_arm_infrastructure_failures_above_maximum": any(
            arm["infrastructure_failures"]
            > validity["maximum_infrastructure_failures"]
            for arm in arms
        ),
        "either_arm_trace_failures_above_maximum": any(
            arm["trace_failures"] > validity["maximum_trace_failures"] for arm in arms
        ),
        "either_arm_grader_failures_above_maximum": any(
            arm["grader_failures"] > validity["maximum_grader_failures"]
            for arm in arms
        ),
        "required_artifact_evidence_missing": not outcome[
            "required_artifact_evidence"
        ],
    }
    assert expected_count == validity["started"] == validity["completed"]
    invalid = any(
        invalid_predicates[name]
        for name, enabled in contract["invalid_if"].items()
        if enabled
    )

    inventory = primary["inventory_lifecycle"]
    feature = primary["feature_implementation"]
    overall = primary["overall"]
    inventory_delta = (
        outcome["inventory_treatment_successes"]
        - outcome["inventory_control_successes"]
    )
    feature_delta = (
        outcome["feature_treatment_successes"] - outcome["feature_control_successes"]
    )
    overall_delta = (
        outcome["overall_treatment_successes"] - outcome["overall_control_successes"]
    )
    reject_predicates = {
        "inventory_treatment_below_minimum": (
            outcome["inventory_treatment_successes"]
            < inventory["treatment_minimum_successes"]
        ),
        "inventory_delta_below_minimum": (
            inventory_delta < inventory["treatment_minus_control_minimum"]
        ),
        "feature_delta_below_minimum": (
            feature_delta < feature["treatment_minus_control_minimum"]
        ),
        "overall_delta_below_minimum": (
            overall_delta < overall["treatment_minus_control_minimum"]
        ),
        "control_bug_fixing_not_required_value": (
            outcome["control"]["bug_fixing_successes"]
            != guardrails["control_bug_fixing_successes_required"]
        ),
        "treatment_bug_fixing_not_required_value": (
            outcome["treatment"]["bug_fixing_successes"]
            != guardrails["treatment_bug_fixing_successes_required"]
        ),
        "either_arm_timeouts_above_maximum": any(
            arm["timeouts"] > guardrails["maximum_timeouts_per_arm"] for arm in arms
        ),
        "treatment_timeouts_exceed_control": (
            guardrails["treatment_timeouts_must_not_exceed_control"]
            and outcome["treatment"]["timeouts"] > outcome["control"]["timeouts"]
        ),
    }
    reject = any(
        reject_predicates[name]
        for name, enabled in contract["reject_if_both_arms_valid_and_any"].items()
        if enabled
    )
    inventory_primary_pass = (
        not reject_predicates["inventory_treatment_below_minimum"]
        and not reject_predicates["inventory_delta_below_minimum"]
    )
    accept_predicates = {
        "no_reject_condition": not reject,
        "inventory_primary_pass": inventory_primary_pass,
        "feature_primary_pass": (
            outcome["feature_treatment_successes"]
            >= feature["treatment_minimum_successes"]
            and not reject_predicates["feature_delta_below_minimum"]
        ),
        "overall_primary_pass": (
            outcome["overall_treatment_successes"]
            >= overall["treatment_minimum_successes"]
            and not reject_predicates["overall_delta_below_minimum"]
        ),
        "latency_ratio_at_or_below_maximum": (
            outcome["latency_ratio"]
            <= efficiency[
                "maximum_treatment_to_control_complete_median_latency_ratio"
            ]
        ),
        "token_ratio_at_or_below_maximum": (
            outcome["token_ratio"]
            <= efficiency[
                "maximum_treatment_to_control_complete_median_input_output_token_ratio"
            ]
        ),
    }
    accept = all(
        accept_predicates[name]
        for name, enabled in contract["accept_if_both_arms_valid_and_all"].items()
        if enabled
    )
    mixed_predicates = {
        "inventory_primary_pass": inventory_primary_pass,
        "no_reject_condition": not reject,
        "accept_condition_false": not accept,
    }
    mixed = all(
        mixed_predicates[name]
        for name, enabled in contract["mixed_if_both_arms_valid_and_all"].items()
        if enabled
    )
    return {
        "INVALID": invalid,
        "REJECT": not invalid and reject,
        "ACCEPT": not invalid and not reject and accept,
        "MIXED": not invalid and not reject and not accept and mixed,
    }


# 按 receipt 冻结的优先级返回唯一 paired classification
def _classify_pair(receipt: dict[str, Any], outcome: dict[str, Any]) -> str:
    matches = _classification_matches(receipt, outcome)
    assert sum(matches.values()) == 1
    for classification in receipt["decision_contract"]["classification_order"]:
        assert isinstance(classification, str)
        if matches[classification]:
            return classification
    raise AssertionError("paired classification is unreachable")


# 断言 receipt 的授权值与永久不可变策略逐字段冻结
def _validate_authorization_immutability(receipt: dict[str, Any]) -> None:
    assert receipt["authorization"] == {
        "real_model_experiment_authorized": False,
        "authorized_attempts": 0,
        "authorized_data_egress": False,
        "authorized_cost": False,
    }
    assert receipt["immutability_policy"] == {
        "receipt_bytes_must_never_change_after_commit": True,
        "authorization_fields_in_this_receipt_must_remain_false_or_zero": True,
        "future_execution_authorization_requires_separate_tracked_artifact": True,
        "future_authorization_artifact_must_reference_receipt_git_commit": True,
        "future_authorization_artifact_must_reference_receipt_file_sha256": True,
        "future_authorization_must_not_amend_or_replace_this_receipt": True,
    }


# 验证 logical root 是跨 POSIX 与 Windows 语义的单一 basename
def _validate_logical_basename(value: str) -> None:
    assert value
    assert value not in {".", ".."}
    assert "/" not in value
    assert "\\" not in value
    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    assert not posix.is_absolute()
    assert not windows.is_absolute()
    assert not windows.drive
    assert len(posix.parts) == 1
    assert len(windows.parts) == 1


# 验证 logical root policy、互异性与选定临时 parent 下的不存在性
def _validate_logical_roots(
    receipt: dict[str, Any],
    external_parent: Path,
) -> None:
    assert receipt["logical_root_policy"] == {
        "values_are_frozen_identifiers_not_execution_dates": True,
        "must_be_single_basename": True,
        "must_not_be_empty": True,
        "must_not_equal_dot_or_dotdot": True,
        "must_not_contain_posix_separator": True,
        "must_not_contain_windows_separator": True,
        "must_not_be_absolute": True,
        "control_and_treatment_must_differ": True,
        "external_parent_selected_only_at_final_preflight": True,
        "final_preflight_must_preserve_frozen_basenames": True,
        "final_preflight_must_recheck_nonexistence": True,
    }
    execution = receipt["execution_plan"]
    roots = (
        execution["control_output_logical_root"],
        execution["treatment_output_logical_root"],
    )
    assert roots[0] != roots[1]
    for root in roots:
        _validate_logical_basename(root)
        assert not (external_parent / root).exists()


# 功能：验证 receipt 是未忽略、拒绝重复键且 top-level 字段精确的 JSON object
# 设计：直接解析 candidate bytes，并向同一 loader 注入重复 schema_version 证明 fail closed
def test_receipt_is_strict_nonignored_json_without_duplicate_keys() -> None:
    text = _RECEIPT_PATH.read_text(encoding="utf-8")
    receipt = _strict_json_object(text)
    ignored = subprocess.run(
        ["git", "check-ignore", "--quiet", str(_RECEIPT_PATH)],
        cwd=_REPOSITORY_ROOT,
        check=False,
    )

    assert ignored.returncode == 1
    assert set(receipt) == _TOP_LEVEL_KEYS
    assert receipt["schema_version"] == 1
    assert receipt["receipt_id"] == "phase9d-repaired-v1-v2-paired-experiment"
    assert receipt["status"] == (
        "preregistered_before_any_paired_real_model_execution"
    )
    duplicated = text.replace(
        '"schema_version": 1,',
        '"schema_version": 1, "schema_version": 1,',
        1,
    )
    with pytest.raises(ValueError, match="^duplicate JSON key$"):
        _strict_json_object(duplicated)


# 功能：验证 receipt 冻结已存在且线性的 repaired-stack C1/C2 Git 与 profile blobs
# 设计：完全从指定 commit 读取 profile bytes，不使用当前 worktree bytes 代替历史证据
def test_receipt_freezes_git_pair_and_profile_blobs() -> None:
    receipt = _receipt()
    control = receipt["arms"]["control"]
    treatment = receipt["arms"]["treatment"]

    assert _git_parent(_C2_COMMIT) == _C1_COMMIT
    assert _git_parent(_C1_COMMIT) == _SHARED_REPAIRED_STACK_COMMIT
    assert receipt["shared_repaired_stack_commit"] == _SHARED_REPAIRED_STACK_COMMIT
    assert control == {
        "label": "C1",
        "commit": _C1_COMMIT,
        "profile_path": _C1_PROFILE_PATH,
        "profile_id": "kama-coding-mvp-v1-deepseek-v4-pro-repaired-v1-control",
        "profile_bytes": 1443,
        "profile_file_sha256": (
            "74b6ab746694add2c833cd513b025e4cdadd8da1be65bfdb4605584c99208d05"
        ),
        "profile_canonical_sha256": (
            "93eb58024a559d6efdacb58f952eaf84696e853618a0bee5f6a7d39c9ca86f68"
        ),
        "prompt_words": 130,
        "prompt_utf8_bytes": 855,
        "prompt_sha256": (
            "b248587ef77d172cefb5e7b777a1523cf50978d6d273b466a8b6eb37349621eb"
        ),
    }
    assert treatment == {
        "label": "C2",
        "commit": _C2_COMMIT,
        "direct_parent": _C1_COMMIT,
        "profile_path": _C2_PROFILE_PATH,
        "profile_id": "kama-coding-mvp-v1-deepseek-v4-pro-repaired-v2-treatment",
        "profile_bytes": 1445,
        "profile_file_sha256": (
            "7e0885810adbf1ef4aa0cb15f5c3c897cc689b4c189e3f632ef4febf7acb0cfc"
        ),
        "profile_canonical_sha256": (
            "492bee94cf77e3fe998889ba57c8c4869cc82f8b0c0eb9c5779d046d91203a17"
        ),
        "prompt_words": 195,
        "prompt_utf8_bytes": 1297,
        "prompt_sha256": (
            "bc9d1a2fbcc3458efb5b153f4ff539050c96e28a9c76f3fcd8823588933eb6c0"
        ),
        "addition_words": 65,
        "addition_utf8_bytes": 440,
        "addition_sha256": (
            "310b308df8cad96fc20ff63b7f50f799ed717a16a65f67db392ca4f6eb7762a3"
        ),
    }
    for arm in (control, treatment):
        profile_bytes = _git_blob(arm["commit"], arm["profile_path"])
        profile_payload = _strict_json_object(profile_bytes.decode("utf-8"))
        assert len(profile_bytes) == arm["profile_bytes"]
        assert _sha256(profile_bytes) == arm["profile_file_sha256"]
        assert _canonical_sha256(profile_payload) == arm["profile_canonical_sha256"]
        assert profile_payload["profile_id"] == arm["profile_id"]


# 功能：验证 receipt 的唯一 behavior variable 是 default prompt 的精确 v2 addition
# 设计：从两个 Git blobs 提取 production AST，并独立比较 production/lifecycle/grader paths
def test_receipt_proves_single_variable_prompt_and_module_pair() -> None:
    receipt = _receipt()
    contract = receipt["single_variable_contract"]
    control = receipt["arms"]["control"]
    treatment = receipt["arms"]["treatment"]
    c1_loop = _git_blob(_C1_COMMIT, _LOOP_PATH)
    c2_loop = _git_blob(_C2_COMMIT, _LOOP_PATH)
    c1_prompt = _extract_default_prompt(c1_loop)
    c2_prompt = _extract_default_prompt(c2_loop)
    addition = c2_prompt[len(c1_prompt) + 2 :]

    assert len(c1_prompt.split()) == control["prompt_words"]
    assert len(c1_prompt.encode("utf-8")) == control["prompt_utf8_bytes"]
    assert _sha256(c1_prompt.encode("utf-8")) == control["prompt_sha256"]
    assert len(c2_prompt.split()) == treatment["prompt_words"]
    assert len(c2_prompt.encode("utf-8")) == treatment["prompt_utf8_bytes"]
    assert _sha256(c2_prompt.encode("utf-8")) == treatment["prompt_sha256"]
    assert len(addition.split()) == treatment["addition_words"]
    assert len(addition.encode("utf-8")) == treatment["addition_utf8_bytes"]
    assert _sha256(addition.encode("utf-8")) == treatment["addition_sha256"]
    assert c2_prompt == c1_prompt + "\n\n" + addition
    assert _normalized_loop_ast(c1_loop) == _normalized_loop_ast(c2_loop)
    assert contract == {
        "treatment_equals_control_plus_two_lf_and_addition": True,
        "normalized_full_module_ast_equal": True,
        "profiles_equal_after_normalizing_id_and_prompt_hash": True,
        "allowed_behavior_difference": "default_prompt_state_transition_addition_only",
    }
    assert _git_changed_paths(
        _C1_COMMIT,
        _C2_COMMIT,
        ("src/kama_claude",),
    ) == (_LOOP_PATH,)
    assert _sha256(_git_blob(_C1_COMMIT, _RUNNER_PATH)) == _RUNNER_SHA256
    assert _git_blob(_C1_COMMIT, _RUNNER_PATH) == _git_blob(
        _C2_COMMIT,
        _RUNNER_PATH,
    )
    assert _git_changed_paths(
        _C1_COMMIT,
        _C2_COMMIT,
        _LIFECYCLE_CONTRACT_PATHS,
    ) == ()


# 功能：验证 receipt shared identity 与两 profiles、suite、freeze 和 task/grader hashes 一致
# 设计：调用真实 strict profile/suite identity 路径，并比较两个 declaration 而非信任 receipt 自述
def test_receipt_shared_identity_matches_frozen_inputs() -> None:
    receipt = _receipt()
    shared = receipt["shared_identity"]
    c1_path = _REPOSITORY_ROOT / _C1_PROFILE_PATH
    c2_path = _REPOSITORY_ROOT / _C2_PROFILE_PATH
    c1_loaded = load_experiment_profile(c1_path)
    c2_loaded = load_experiment_profile(c2_path)
    c1_declared = capture_declared_identity(
        c1_loaded,
        repository_root=_REPOSITORY_ROOT,
        repository=RepositoryIdentity(commit=_C1_COMMIT, dirty=False),
        installed_sdk_version=version("anthropic"),
    )
    c2_declared = capture_declared_identity(
        c2_loaded,
        repository_root=_REPOSITORY_ROOT,
        repository=RepositoryIdentity(commit=_C2_COMMIT, dirty=False),
        installed_sdk_version=version("anthropic"),
    )
    c1_profile = c1_loaded.profile.model_dump(mode="json")
    c2_profile = c2_loaded.profile.model_dump(mode="json")
    normalized = json.loads(json.dumps(c2_profile))
    normalized["profile_id"] = c1_profile["profile_id"]
    normalized["expected_identity"]["prompt_hash"] = c1_profile[
        "expected_identity"
    ]["prompt_hash"]
    freeze = _strict_json_object(c1_loaded.freeze_path.read_text(encoding="utf-8"))
    suite = load_suite(c1_loaded.suite_path, c1_loaded.tasks_root)
    timeouts_by_difficulty: dict[str, set[int]] = {}
    for task, frozen_task in zip(suite.tasks, freeze["tasks"], strict=True):
        timeouts_by_difficulty.setdefault(frozen_task["difficulty"], set()).add(
            int(task.evaluation_task.public.timeout_s)
        )

    assert normalized == c1_profile
    assert c1_declared.suite == c2_declared.suite
    assert c1_declared.tool_schema_hash == c2_declared.tool_schema_hash
    assert c1_declared.runtime == c2_declared.runtime
    assert c1_declared.runtime_config_hash == c2_declared.runtime_config_hash
    assert c1_declared.dependency == c2_declared.dependency
    assert len(c1_declared.suite.task_hashes) == shared["task_hashes_verified"] == 9
    assert len(c1_declared.suite.grader_hashes) == (
        shared["grader_hashes_verified"]
    ) == 9
    assert shared == {
        "suite_id": (
            f"{c1_declared.suite.suite_id}@{c1_declared.suite.suite_version}"
        ),
        "suite_sha256": c1_declared.suite.suite_hash,
        "task_hashes_verified": 9,
        "grader_hashes_verified": 9,
        "provider": c1_declared.provider.service_provider,
        "protocol": c1_declared.provider.wire_protocol,
        "endpoint_id": c1_declared.provider.endpoint_id,
        "model": c1_declared.provider.model_id,
        "sdk": (
            f"{c1_declared.provider.sdk_distribution}"
            f"=={c1_declared.provider.sdk_version}"
        ),
        "tool_schema_sha256": c1_declared.tool_schema_hash,
        "runtime_config_sha256": c1_declared.runtime_config_hash,
        "dependency_sha256": c1_declared.dependency.dependency_hash,
        "max_steps": c1_declared.runtime.max_steps,
        "repeats": c1_declared.schedule.repeats,
        "execution_order": c1_declared.schedule.execution_order,
        "easy_timeout_seconds": 120,
        "medium_challenging_timeout_seconds": 180,
        "mcp_enabled": c1_declared.runtime.mcp_enabled,
        "raw_trace_visibility": c1_declared.artifacts.raw_trace_visibility,
    }
    assert freeze["suite_hash"] == shared["suite_sha256"]
    assert {
        task["task_id"]: task["task_hash"] for task in freeze["tasks"]
    } == c1_declared.suite.task_hashes
    assert {
        task["task_id"]: task["grader_hash"] for task in freeze["tasks"]
    } == c1_declared.suite.grader_hashes
    assert timeouts_by_difficulty == {
        "easy": {120},
        "medium": {180},
        "challenging": {180},
    }


# 功能：验证 receipt 冻结 control-first stop policy、全部 thresholds 与 deterministic contract
# 设计：用独立 literal 与结构 validator 锁定顺序、阈值和唯一分类规则
def test_receipt_freezes_execution_validity_and_decision_rules() -> None:
    receipt = _receipt()

    assert receipt["execution_plan"] == {
        "arm_order": ["control", "treatment"],
        "attempts_per_arm": 27,
        "total_attempts": 54,
        "control_output_logical_root": (
            "kama-coding-mvp-v1-deepseek-v4-pro-"
            "repaired-v1-control-20260730-001"
        ),
        "treatment_output_logical_root": (
            "kama-coding-mvp-v1-deepseek-v4-pro-"
            "repaired-v2-treatment-20260730-001"
        ),
        "separate_detached_worktrees": True,
        "output_roots_outside_repository": True,
        "output_roots_must_be_new": True,
        "no_paid_smoke_calls": True,
        "no_rerun": True,
        "no_resume": True,
        "no_result_aware_changes_between_arms": True,
        "treatment_runs_only_if_control_is_valid_and_complete": True,
        "treatment_runs_regardless_of_control_capability_scores": True,
    }
    assert receipt["arm_validity"] == {
        "required_status": "VALID",
        "planned": 27,
        "started": 27,
        "completed": 27,
        "identity_verified": 27,
        "maximum_runtime_failures": 0,
        "maximum_infrastructure_failures": 0,
        "maximum_trace_failures": 0,
        "maximum_grader_failures": 0,
    }
    assert receipt["primary_comparison"] == {
        "inventory_lifecycle": {
            "treatment_minimum_successes": 1,
            "treatment_minus_control_minimum": 1,
        },
        "feature_implementation": {
            "treatment_minimum_successes": 6,
            "treatment_minus_control_minimum": 0,
        },
        "overall": {
            "treatment_minimum_successes": 20,
            "treatment_minus_control_minimum": 0,
        },
    }
    assert receipt["hard_guardrails"] == {
        "control_bug_fixing_successes_required": 9,
        "treatment_bug_fixing_successes_required": 9,
        "maximum_timeouts_per_arm": 3,
        "treatment_timeouts_must_not_exceed_control": True,
    }
    assert receipt["secondary_reporting"] == {
        "atomic_plus_inventory_treatment_minimum": 3,
        "atomic_plus_inventory_attempts": 6,
        "atomic_oracle_disposition": "CONTRACT_AMBIGUITY_NOTE",
        "report_task_repeat_win_loss_tie": True,
        "analysis_only_stateful_tasks": [
            "feature-atomic-bulk-import",
            "feature-inventory-reservation-lifecycle",
            "bugfix-retry-state-idempotency",
        ],
        "analysis_only_fields": [
            "pre_state_identified",
            "mutation_points_enumerated",
            "later_failure_points_enumerated",
            "success_post_state_identified",
            "failure_post_state_identified",
            "rollback_or_compensation_planned",
            "mid_operation_failure_injected",
            "invariant_verified_after_injection",
        ],
    }
    assert receipt["efficiency_comparison"] == {
        "maximum_treatment_to_control_complete_median_latency_ratio": 1.1,
        "maximum_treatment_to_control_complete_median_input_output_token_ratio": 1.1,
        "median_algorithm": "python.statistics.median",
        "exclude_timeout_zero_token_placeholders": True,
        "report_total_experiment_wall_for_each_arm": True,
    }
    _validate_execution_state_machine(receipt)
    _validate_decision_contract(receipt)


# 功能：验证 host、authorization、output-root 与 loader/security 边界保持 fail closed
# 设计：检查真实本机身份和候选路径不存在，并证明 profile loader 拒绝 receipt schema/location
def test_receipt_freezes_host_authorization_and_security_boundaries() -> None:
    receipt = _receipt()
    execution = receipt["execution_plan"]
    roots = (
        execution["control_output_logical_root"],
        execution["treatment_output_logical_root"],
    )
    forbidden_keys = {
        "api_key",
        "credential",
        "secret",
        "system_prompt",
        "messages",
        "tool_schemas",
        "private_path",
        "absolute_path",
        "hidden_grader",
    }
    text = _RECEIPT_PATH.read_text(encoding="utf-8")

    assert receipt["host_policy"] == {
        "same_physical_host": True,
        "same_dependency_environment": True,
        "python": "3.12.13",
        "os": "Darwin",
        "architecture": "arm64",
        "no_environment_update_between_arms": True,
    }
    assert platform.python_version() == receipt["host_policy"]["python"]
    assert platform.system() == receipt["host_policy"]["os"]
    assert platform.machine() == receipt["host_policy"]["architecture"]
    assert roots[0] != roots[1]
    for logical_root in roots:
        assert not Path(logical_root).is_absolute()
        assert not (_REPOSITORY_ROOT / logical_root).exists()
        assert not (_REPOSITORY_ROOT.parent / logical_root).exists()
    assert receipt["authorization"] == {
        "real_model_experiment_authorized": False,
        "authorized_attempts": 0,
        "authorized_data_egress": False,
        "authorized_cost": False,
    }
    assert forbidden_keys.isdisjoint(_all_keys(receipt))
    assert "ANTHROPIC_API_KEY" not in text
    for value in _all_string_values(receipt):
        assert not PurePosixPath(value).is_absolute()
        assert not PureWindowsPath(value).is_absolute()
    with pytest.raises(ValueError, match="invalid experiment profile"):
        load_experiment_profile(_RECEIPT_PATH)
    with pytest.raises(Exception):
        ExperimentProfile.model_validate(receipt)


@pytest.mark.parametrize(
    "payload",
    (
        '{"value": NaN}',
        '{"value": Infinity}',
        '{"value": -Infinity}',
    ),
)
# 功能：验证 strict receipt JSON loader 拒绝全部非有限数值常量
# 设计：直接注入标准库默认接受的三个常量，证明 parser 必须通过 parse_constant fail closed
def test_strict_json_rejects_non_finite_constants(payload: str) -> None:
    with pytest.raises(ValueError, match="non-finite JSON constant"):
        _strict_json_object(payload)


# 功能：验证 receipt bytes 必须使用严格 UTF-8 解码且 root 必须是 object
# 设计：分别注入非法 UTF-8 bytes 与合法非 object JSON，锁定 encoding 和 shape 两层边界
def test_strict_json_requires_utf8_object_root() -> None:
    with pytest.raises(UnicodeDecodeError):
        _strict_json_object(b'{"value":"\xff"}')
    with pytest.raises(ValueError, match="receipt root must be a JSON object"):
        _strict_json_object("[]")


# 功能：验证 arm validity 是基础失败上限的唯一来源且 timeout 保持独立 REJECT 语义
# 设计：从 receipt 阈值驱动分类器，逐类注入失败并证明 INVALID 先于 hard comparison
def test_receipt_closes_validity_and_guardrail_semantics() -> None:
    receipt = _receipt()
    assert receipt["arm_validity"] == {
        "required_status": "VALID",
        "planned": 27,
        "started": 27,
        "completed": 27,
        "identity_verified": 27,
        "maximum_runtime_failures": 0,
        "maximum_infrastructure_failures": 0,
        "maximum_trace_failures": 0,
        "maximum_grader_failures": 0,
    }
    assert receipt["hard_guardrails"] == {
        "control_bug_fixing_successes_required": 9,
        "treatment_bug_fixing_successes_required": 9,
        "maximum_timeouts_per_arm": 3,
        "treatment_timeouts_must_not_exceed_control": True,
    }
    duplicated_limits = {
        "maximum_runtime_failures_per_arm",
        "maximum_infrastructure_failures_per_arm",
        "maximum_trace_failures_per_arm",
        "maximum_grader_failures_per_arm",
    }
    assert duplicated_limits.isdisjoint(receipt["hard_guardrails"])

    for failure_field in (
        "runtime_failures",
        "infrastructure_failures",
        "trace_failures",
        "grader_failures",
    ):
        invalid = _with_nested_value(
            _accepting_pair_outcome(),
            ("treatment", failure_field),
            1,
        )
        invalid["treatment"]["bug_fixing_successes"] = 8
        assert _classify_pair(receipt, invalid) == "INVALID"

    timeout_reject = _with_nested_value(
        _accepting_pair_outcome(),
        ("treatment", "timeouts"),
        4,
    )
    timeout_reject["control"]["timeouts"] = 4
    assert _classify_pair(receipt, timeout_reject) == "REJECT"


# 功能：验证 tracked execution state machine 冻结全部控制与停止 transition
# 设计：比较完整结构，确保 ignored devlog 不是 paid execution 控制流的唯一证据
def test_receipt_tracks_execution_state_machine() -> None:
    receipt = _receipt()
    assert "execution_state_machine" in receipt
    _validate_execution_state_machine(receipt)


_STATE_MACHINE_MUTATION_CASES = (
    pytest.param(
        (
            (
                (
                    "execution_state_machine",
                    "control_preflight_failed",
                    "create_control_output_root",
                ),
                False,
                True,
            ),
        ),
        id="preflight-create-output-root",
    ),
    pytest.param(
        (
            (
                (
                    "execution_state_machine",
                    "control_preflight_failed",
                    "maximum_api_calls",
                ),
                0,
                1,
            ),
        ),
        id="preflight-maximum-api-calls",
    ),
    pytest.param(
        (
            (
                (
                    "execution_state_machine",
                    "control_preflight_failed",
                    "run_treatment",
                ),
                False,
                True,
            ),
        ),
        id="preflight-run-treatment",
    ),
    pytest.param(
        (
            (
                (
                    "execution_state_machine",
                    "control_preflight_failed",
                    "pair_status",
                ),
                "NOT_STARTED",
                "INVALID",
            ),
        ),
        id="preflight-pair-status-invalid",
    ),
    pytest.param(
        (
            (
                (
                    "execution_state_machine",
                    "control_preflight_failed",
                    "pair_status",
                ),
                "NOT_STARTED",
                "VALID",
            ),
        ),
        id="preflight-pair-status-valid",
    ),
    pytest.param(
        (
            (
                (
                    "execution_state_machine",
                    "control_started_then_invalid_or_incomplete",
                    "preserve_control_artifacts",
                ),
                True,
                False,
            ),
        ),
        id="control-invalid-preserve-artifacts",
    ),
    pytest.param(
        (
            (
                (
                    "execution_state_machine",
                    "control_started_then_invalid_or_incomplete",
                    "run_treatment",
                ),
                False,
                True,
            ),
        ),
        id="control-invalid-run-treatment",
    ),
    pytest.param(
        (
            (
                (
                    "execution_state_machine",
                    "control_started_then_invalid_or_incomplete",
                    "rerun",
                ),
                False,
                True,
            ),
        ),
        id="control-invalid-rerun",
    ),
    pytest.param(
        (
            (
                (
                    "execution_state_machine",
                    "control_started_then_invalid_or_incomplete",
                    "resume",
                ),
                False,
                True,
            ),
        ),
        id="control-invalid-resume",
    ),
    pytest.param(
        (
            (
                (
                    "execution_state_machine",
                    "control_started_then_invalid_or_incomplete",
                    "pair_status",
                ),
                "INVALID",
                "NOT_STARTED",
            ),
        ),
        id="control-invalid-pair-status-not-started",
    ),
    pytest.param(
        (
            (
                (
                    "execution_state_machine",
                    "control_started_then_invalid_or_incomplete",
                    "pair_status",
                ),
                "INVALID",
                "VALID",
            ),
        ),
        id="control-invalid-pair-status-valid",
    ),
    pytest.param(
        (
            (
                (
                    "execution_state_machine",
                    "control_started_then_invalid_or_incomplete",
                    "publish_paired_capability_delta",
                ),
                False,
                True,
            ),
        ),
        id="control-invalid-publish",
    ),
    pytest.param(
        (
            (
                (
                    "execution_state_machine",
                    "control_valid_and_complete",
                    "run_treatment",
                ),
                True,
                False,
            ),
        ),
        id="control-valid-run-treatment",
    ),
    pytest.param(
        (
            (
                (
                    "execution_state_machine",
                    "control_valid_and_complete",
                    "progression_depends_on_control_capability_scores",
                ),
                False,
                True,
            ),
        ),
        id="control-valid-score-dependence",
    ),
    pytest.param(
        (
            (
                (
                    "execution_state_machine",
                    "control_valid_and_complete",
                    "allow_experiment_changes_between_arms",
                ),
                False,
                True,
            ),
        ),
        id="control-valid-experiment-changes",
    ),
    pytest.param(
        (
            (
                (
                    "execution_state_machine",
                    "treatment_started_then_invalid_or_incomplete",
                    "preserve_all_artifacts",
                ),
                True,
                False,
            ),
        ),
        id="treatment-invalid-preserve-artifacts",
    ),
    pytest.param(
        (
            (
                (
                    "execution_state_machine",
                    "treatment_started_then_invalid_or_incomplete",
                    "rerun",
                ),
                False,
                True,
            ),
        ),
        id="treatment-invalid-rerun",
    ),
    pytest.param(
        (
            (
                (
                    "execution_state_machine",
                    "treatment_started_then_invalid_or_incomplete",
                    "resume",
                ),
                False,
                True,
            ),
        ),
        id="treatment-invalid-resume",
    ),
    pytest.param(
        (
            (
                (
                    "execution_state_machine",
                    "treatment_started_then_invalid_or_incomplete",
                    "pair_status",
                ),
                "INVALID",
                "NOT_STARTED",
            ),
        ),
        id="treatment-invalid-pair-status-not-started",
    ),
    pytest.param(
        (
            (
                (
                    "execution_state_machine",
                    "treatment_started_then_invalid_or_incomplete",
                    "pair_status",
                ),
                "INVALID",
                "VALID",
            ),
        ),
        id="treatment-invalid-pair-status-valid",
    ),
    pytest.param(
        (
            (
                (
                    "execution_state_machine",
                    "treatment_started_then_invalid_or_incomplete",
                    "publish_paired_capability_delta",
                ),
                False,
                True,
            ),
        ),
        id="treatment-invalid-publish",
    ),
    pytest.param(
        (
            (
                (
                    "execution_state_machine",
                    "both_arms_valid_and_complete",
                    "evaluate_decision_contract",
                ),
                True,
                False,
            ),
        ),
        id="both-valid-evaluate",
    ),
    pytest.param(
        (
            (
                (
                    "execution_state_machine",
                    "both_arms_valid_and_complete",
                    "publish_paired_capability_delta",
                ),
                True,
                False,
            ),
        ),
        id="both-valid-publish",
    ),
    pytest.param(
        (
            (
                (
                    "execution_state_machine",
                    "control_started_then_invalid_or_incomplete",
                    "run_treatment",
                ),
                False,
                True,
            ),
            (
                (
                    "execution_state_machine",
                    "control_started_then_invalid_or_incomplete",
                    "pair_status",
                ),
                "INVALID",
                "VALID",
            ),
        ),
        id="joint-control-invalid",
    ),
    pytest.param(
        (
            (
                (
                    "execution_state_machine",
                    "treatment_started_then_invalid_or_incomplete",
                    "rerun",
                ),
                False,
                True,
            ),
            (
                (
                    "execution_state_machine",
                    "treatment_started_then_invalid_or_incomplete",
                    "publish_paired_capability_delta",
                ),
                False,
                True,
            ),
        ),
        id="joint-treatment-invalid",
    ),
    pytest.param(
        (
            (
                (
                    "execution_state_machine",
                    "control_valid_and_complete",
                    "run_treatment",
                ),
                True,
                False,
            ),
            (
                (
                    "execution_state_machine",
                    "control_valid_and_complete",
                    "allow_experiment_changes_between_arms",
                ),
                False,
                True,
            ),
        ),
        id="joint-control-valid",
    ),
)


@pytest.mark.parametrize("mutations", _STATE_MACHINE_MUTATION_CASES)
# 功能：验证 execution state machine 的危险 transition mutation 全部被合同测试杀死
# 设计：逐 case 核对旧值与 leaf diff 后调用正式 validator，覆盖 23 个单字段及 3 个 joint drift
def test_execution_state_machine_rejects_transition_mutations(
    mutations: tuple[tuple[tuple[str, ...], object, object], ...],
) -> None:
    receipt = _receipt()
    original = copy.deepcopy(receipt)
    mutated = _with_verified_mutations(receipt, mutations)
    assert receipt == original
    with pytest.raises(AssertionError):
        _validate_execution_state_machine(mutated)


_DECISION_CASES = (
    ("accept", (), "ACCEPT"),
    (
        "mixed_feature_absolute",
        (
            (("feature_control_successes",), 5),
            (("feature_treatment_successes",), 5),
        ),
        "MIXED",
    ),
    (
        "mixed_overall_absolute",
        (
            (("overall_control_successes",), 19),
            (("overall_treatment_successes",), 19),
        ),
        "MIXED",
    ),
    ("mixed_latency", ((("latency_ratio",), 1.11),), "MIXED"),
    ("mixed_token", ((("token_ratio",), 1.11),), "MIXED"),
    (
        "reject_inventory",
        ((("inventory_treatment_successes",), 0),),
        "REJECT",
    ),
    (
        "reject_inventory_delta",
        (
            (("inventory_control_successes",), 1),
            (("inventory_treatment_successes",), 1),
        ),
        "REJECT",
    ),
    (
        "reject_inventory_with_efficiency_regression",
        (
            (("inventory_treatment_successes",), 0),
            (("latency_ratio",), 1.11),
        ),
        "REJECT",
    ),
    (
        "reject_feature_regression",
        (
            (("feature_control_successes",), 7),
            (("feature_treatment_successes",), 6),
        ),
        "REJECT",
    ),
    (
        "reject_overall_regression",
        (
            (("overall_control_successes",), 21),
            (("overall_treatment_successes",), 20),
        ),
        "REJECT",
    ),
    (
        "reject_bug_fixing",
        ((("treatment", "bug_fixing_successes"), 8),),
        "REJECT",
    ),
    (
        "reject_timeout_above_maximum",
        (
            (("control", "timeouts"), 4),
            (("treatment", "timeouts"), 4),
        ),
        "REJECT",
    ),
    (
        "reject_treatment_timeout_regression",
        ((("treatment", "timeouts"), 2),),
        "REJECT",
    ),
    (
        "invalid_status",
        ((("treatment", "status"), "INVALID"),),
        "INVALID",
    ),
    (
        "invalid_attempt_count",
        ((("treatment", "completed"), 26),),
        "INVALID",
    ),
    (
        "invalid_planned_count",
        ((("control", "planned"), 26),),
        "INVALID",
    ),
    (
        "invalid_started_count",
        ((("control", "started"), 26),),
        "INVALID",
    ),
    (
        "invalid_identity_count",
        ((("treatment", "identity_verified"), 26),),
        "INVALID",
    ),
    (
        "invalid_runtime_failure",
        ((("treatment", "runtime_failures"), 1),),
        "INVALID",
    ),
    (
        "invalid_infrastructure_failure",
        ((("treatment", "infrastructure_failures"), 1),),
        "INVALID",
    ),
    (
        "invalid_trace_failure",
        ((("treatment", "trace_failures"), 1),),
        "INVALID",
    ),
    (
        "invalid_grader_failure",
        ((("treatment", "grader_failures"), 1),),
        "INVALID",
    ),
    (
        "invalid_missing_artifact",
        ((("required_artifact_evidence",), False),),
        "INVALID",
    ),
    (
        "secondary_independence",
        (
            (("atomic_oracle_disposition",), "OTHER"),
            (("analysis_only_result",), "changed"),
        ),
        "ACCEPT",
    ),
    (
        "secondary_cannot_rescue_reject",
        (
            (("inventory_treatment_successes",), 0),
            (("atomic_oracle_disposition",), "FAVORABLE"),
            (("analysis_only_result",), "favorable"),
        ),
        "REJECT",
    ),
)


@pytest.mark.parametrize(
    ("_case_name", "mutations", "expected"),
    _DECISION_CASES,
    ids=[case[0] for case in _DECISION_CASES],
)
# 功能：验证 deterministic classifier 对预注册边界案例产生唯一 ACCEPT/MIXED/REJECT/INVALID
# 设计：用手工 expected 与 receipt 阈值驱动的纯分类器，避免复制隐藏阈值或结果后解释
def test_decision_contract_classifies_table_cases_uniquely(
    _case_name: str,
    mutations: tuple[tuple[tuple[str, ...], object], ...],
    expected: str,
) -> None:
    receipt = _receipt()
    assert "decision_contract" in receipt
    outcome = _accepting_pair_outcome()
    for path, replacement in mutations:
        outcome = _with_nested_value(outcome, path, replacement)
    matches = _classification_matches(receipt, outcome)
    assert sum(matches.values()) == 1
    assert _classify_pair(receipt, outcome) == expected


# 功能：验证 classification order 精确且四种最终状态全部可达
# 设计：独立汇总 table expected，并冻结 INVALID-first 顺序以消除调用者选择空间
def test_decision_contract_order_and_reachability_are_frozen() -> None:
    receipt = _receipt()
    assert "decision_contract" in receipt
    _validate_decision_contract(receipt)
    assert receipt["decision_contract"]["classification_order"] == [
        "INVALID",
        "REJECT",
        "ACCEPT",
        "MIXED",
    ]
    assert {case[2] for case in _DECISION_CASES} == {
        "INVALID",
        "REJECT",
        "ACCEPT",
        "MIXED",
    }


@pytest.mark.parametrize(
    ("path", "replacement"),
    (
        (
            ("decision_contract", "classification_order"),
            ["REJECT", "INVALID", "ACCEPT", "MIXED"],
        ),
        (
            (
                "decision_contract",
                "invalid_if",
                "either_arm_runtime_failures_above_maximum",
            ),
            False,
        ),
        (
            (
                "decision_contract",
                "reject_if_both_arms_valid_and_any",
                "feature_delta_below_minimum",
            ),
            False,
        ),
        (
            ("decision_contract", "secondary_reporting_affects_classification"),
            True,
        ),
    ),
)
# 功能：验证 decision order、invalid/reject predicate 与 secondary independence 漂移都会失败
# 设计：对 tracked contract 做单变量 mutation，并由同一结构 validator 拒绝结果后自由度
def test_decision_contract_rejects_semantic_mutations(
    path: tuple[str, ...],
    replacement: object,
) -> None:
    receipt = _receipt()
    mutated = _with_nested_value(receipt, path, replacement)
    with pytest.raises(AssertionError):
        _validate_decision_contract(mutated)


_THRESHOLD_MUTATION_CASES = (
    pytest.param(
        (
            "primary_comparison",
            "feature_implementation",
            "treatment_minimum_successes",
        ),
        6,
        7,
        (),
        "ACCEPT",
        "MIXED",
        id="primary-feature-absolute",
    ),
    pytest.param(
        (
            "primary_comparison",
            "inventory_lifecycle",
            "treatment_minimum_successes",
        ),
        1,
        2,
        (),
        "ACCEPT",
        "REJECT",
        id="primary-inventory-treatment",
    ),
    pytest.param(
        (
            "primary_comparison",
            "inventory_lifecycle",
            "treatment_minus_control_minimum",
        ),
        1,
        2,
        (),
        "ACCEPT",
        "REJECT",
        id="primary-inventory-delta",
    ),
    pytest.param(
        ("primary_comparison", "overall", "treatment_minimum_successes"),
        20,
        21,
        (),
        "ACCEPT",
        "MIXED",
        id="primary-overall-absolute",
    ),
    pytest.param(
        (
            "primary_comparison",
            "feature_implementation",
            "treatment_minus_control_minimum",
        ),
        0,
        1,
        (),
        "ACCEPT",
        "REJECT",
        id="primary-feature-delta",
    ),
    pytest.param(
        ("primary_comparison", "overall", "treatment_minus_control_minimum"),
        0,
        1,
        (),
        "ACCEPT",
        "REJECT",
        id="primary-overall-delta",
    ),
    pytest.param(
        ("hard_guardrails", "control_bug_fixing_successes_required"),
        9,
        8,
        (),
        "ACCEPT",
        "REJECT",
        id="hard-control-bug-fixing",
    ),
    pytest.param(
        ("hard_guardrails", "treatment_bug_fixing_successes_required"),
        9,
        8,
        (),
        "ACCEPT",
        "REJECT",
        id="hard-treatment-bug-fixing",
    ),
    pytest.param(
        ("hard_guardrails", "maximum_timeouts_per_arm"),
        3,
        0,
        (),
        "ACCEPT",
        "REJECT",
        id="hard-maximum-timeout",
    ),
    pytest.param(
        ("hard_guardrails", "treatment_timeouts_must_not_exceed_control"),
        True,
        False,
        ((("treatment", "timeouts"), 2),),
        "REJECT",
        "ACCEPT",
        id="hard-timeout-relation",
    ),
    pytest.param(
        (
            "efficiency_comparison",
            "maximum_treatment_to_control_complete_median_latency_ratio",
        ),
        1.1,
        0.9,
        (),
        "ACCEPT",
        "MIXED",
        id="efficiency-latency-ratio",
    ),
    pytest.param(
        (
            "efficiency_comparison",
            "maximum_treatment_to_control_complete_median_input_output_token_ratio",
        ),
        1.1,
        0.9,
        (),
        "ACCEPT",
        "MIXED",
        id="efficiency-token-ratio",
    ),
    pytest.param(
        ("arm_validity", "maximum_runtime_failures"),
        0,
        1,
        ((("treatment", "runtime_failures"), 1),),
        "INVALID",
        "ACCEPT",
        id="validity-runtime-failure",
    ),
    pytest.param(
        ("arm_validity", "maximum_infrastructure_failures"),
        0,
        1,
        ((("treatment", "infrastructure_failures"), 1),),
        "INVALID",
        "ACCEPT",
        id="validity-infrastructure-failure",
    ),
    pytest.param(
        ("arm_validity", "maximum_trace_failures"),
        0,
        1,
        ((("treatment", "trace_failures"), 1),),
        "INVALID",
        "ACCEPT",
        id="validity-trace-failure",
    ),
    pytest.param(
        ("arm_validity", "maximum_grader_failures"),
        0,
        1,
        ((("treatment", "grader_failures"), 1),),
        "INVALID",
        "ACCEPT",
        id="validity-grader-failure",
    ),
)


@pytest.mark.parametrize(
    (
        "receipt_path",
        "old_value",
        "new_value",
        "outcome_mutations",
        "expected_original",
        "expected_mutated",
    ),
    _THRESHOLD_MUTATION_CASES,
)
# 功能：验证 classifier 从 receipt 读取 primary、hard、efficiency 与 validity thresholds
# 设计：固定同一 input，仅改变一个已核对 leaf，并同时证明原 object 与 receipt bytes 未被污染
def test_receipt_threshold_mutations_drive_classifier(
    receipt_path: tuple[str, ...],
    old_value: object,
    new_value: object,
    outcome_mutations: tuple[tuple[tuple[str, ...], object], ...],
    expected_original: str,
    expected_mutated: str,
) -> None:
    receipt_bytes = _RECEIPT_PATH.read_bytes()
    receipt = _receipt()
    original_receipt = copy.deepcopy(receipt)
    outcome = _accepting_pair_outcome()
    for outcome_path, replacement in outcome_mutations:
        outcome = _with_nested_value(outcome, outcome_path, replacement)
    fixed_outcome = copy.deepcopy(outcome)
    mutated = _with_verified_mutations(
        receipt,
        ((receipt_path, old_value, new_value),),
    )

    original_matches = _classification_matches(receipt, outcome)
    original_classification = _classify_pair(receipt, outcome)
    assert outcome == fixed_outcome
    mutated_matches = _classification_matches(mutated, outcome)
    mutated_classification = _classify_pair(mutated, outcome)

    assert sum(original_matches.values()) == 1
    assert sum(mutated_matches.values()) == 1
    assert original_classification == expected_original
    assert mutated_classification == expected_mutated
    assert outcome == fixed_outcome
    assert receipt == original_receipt
    assert _RECEIPT_PATH.read_bytes() == receipt_bytes
    assert _sha256(receipt_bytes) == (
        "58aaf8309d9e8eee1f64dc469453407e8c43c2eef884e7dfe00ced91ffd35958"
    )


# 功能：验证 authorization 保持 false/zero 且 receipt 永久不可变策略被 tracked contract 冻结
# 设计：同时检查当前授权与未来独立 artifact 引用要求，禁止后续把授权回写本 receipt
def test_receipt_freezes_authorization_immutability_policy() -> None:
    receipt = _receipt()
    assert "immutability_policy" in receipt
    _validate_authorization_immutability(receipt)


@pytest.mark.parametrize(
    "field",
    (
        "receipt_bytes_must_never_change_after_commit",
        "authorization_fields_in_this_receipt_must_remain_false_or_zero",
        "future_execution_authorization_requires_separate_tracked_artifact",
        "future_authorization_artifact_must_reference_receipt_git_commit",
        "future_authorization_artifact_must_reference_receipt_file_sha256",
        "future_authorization_must_not_amend_or_replace_this_receipt",
    ),
)
# 功能：验证删除或关闭任一 immutability 字段都会使 receipt 合同失败
# 设计：对深复制 receipt 做单字段 false mutation，并由完整 policy validator 拒绝
def test_immutability_policy_rejects_false_fields(field: str) -> None:
    receipt = _receipt()
    assert "immutability_policy" in receipt
    mutated = copy.deepcopy(receipt)
    mutated["immutability_policy"][field] = False
    with pytest.raises(AssertionError):
        _validate_authorization_immutability(mutated)


# 功能：验证缺失 immutability 字段与授权变 true/非零都会被合同拒绝
# 设计：分别删除 policy 字段和修改授权值，覆盖结构删除与未来回写两类 mutation
def test_authorization_immutability_rejects_missing_or_enabled_values() -> None:
    receipt = _receipt()
    assert "immutability_policy" in receipt
    missing = copy.deepcopy(receipt)
    del missing["immutability_policy"][
        "future_authorization_artifact_must_reference_receipt_git_commit"
    ]
    with pytest.raises(AssertionError):
        _validate_authorization_immutability(missing)
    for field, value in (
        ("real_model_experiment_authorized", True),
        ("authorized_attempts", 54),
        ("authorized_data_egress", True),
        ("authorized_cost", True),
    ):
        enabled = copy.deepcopy(receipt)
        enabled["authorization"][field] = value
        with pytest.raises(AssertionError):
            _validate_authorization_immutability(enabled)


@pytest.mark.parametrize(
    "invalid_root",
    (
        pytest.param("", id="empty"),
        pytest.param(".", id="dot"),
        pytest.param("..", id="dotdot"),
        pytest.param("../escape", id="posix-traversal"),
        pytest.param("nested/name", id="posix-separator"),
        pytest.param("nested\\name", id="windows-separator"),
        pytest.param("/absolute", id="posix-absolute"),
        pytest.param("C:\\absolute", id="drive-absolute-backslash"),
        pytest.param(
            "C:\\absolute\\child",
            id="drive-absolute-backslash-child",
        ),
        pytest.param("C:/absolute", id="drive-absolute-forward"),
        pytest.param("C:/absolute/child", id="drive-absolute-forward-child"),
        pytest.param("\\\\server\\share", id="unc-backslash-share"),
        pytest.param(
            "\\\\server\\share\\name",
            id="unc-backslash-child",
        ),
        pytest.param("//server/share", id="unc-forward-share"),
        pytest.param("//server/share/name", id="unc-forward-child"),
        pytest.param("C:relative", id="drive-relative-basename"),
        pytest.param(
            "C:relative\\child",
            id="drive-relative-backslash-child",
        ),
        pytest.param(
            "C:relative/child",
            id="drive-relative-forward-child",
        ),
    ),
)
# 功能：验证 logical root validator 拒绝遍历、UNC、绝对路径与 drive-relative shapes
# 设计：每个 persisted raw vector 都调用共享 basename consumer，不以 frozen literal 比较代替行为
def test_logical_root_validator_rejects_path_shapes(invalid_root: str) -> None:
    with pytest.raises(AssertionError):
        _validate_logical_basename(invalid_root)


# 功能：验证 logical root policy 冻结 basename、互异性及 final-preflight 不存在性语义
# 设计：只在 pytest 临时 parent 创建同名目录，证明 existing-root fail closed 且不触碰正式路径
def test_receipt_closes_logical_root_policy(tmp_path: Path) -> None:
    receipt = _receipt()
    assert "logical_root_policy" in receipt
    _validate_logical_roots(receipt, tmp_path)

    same_name = copy.deepcopy(receipt)
    execution = same_name["execution_plan"]
    execution["treatment_output_logical_root"] = execution[
        "control_output_logical_root"
    ]
    with pytest.raises(AssertionError):
        _validate_logical_roots(same_name, tmp_path)

    existing = tmp_path / execution["control_output_logical_root"]
    existing.mkdir()
    with pytest.raises(AssertionError):
        _validate_logical_roots(receipt, tmp_path)


# 功能：验证 duplicate-key 异常使用恒定诊断且不泄漏攻击者控制的 sentinel
# 设计：覆盖随机、普通与 nested duplicate，并检查 str、repr、traceback 和 exception chain
def test_duplicate_key_diagnostic_is_input_independent() -> None:
    sentinel = f"review-sentinel-{uuid.uuid4().hex}"
    payloads = (
        f'{{"{sentinel}": 1, "{sentinel}": 2}}',
        '{"ordinary": 1, "ordinary": 2}',
        '{"nested": {"duplicate": 1, "duplicate": 2}}',
    )

    for payload in payloads:
        with pytest.raises(ValueError) as exc_info:
            _strict_json_object(payload)

        rendered = "".join(traceback.format_exception(exc_info.value))
        assert str(exc_info.value) == "duplicate JSON key"
        assert sentinel not in str(exc_info.value)
        assert sentinel not in repr(exc_info.value)
        assert sentinel not in rendered
        assert exc_info.value.__cause__ is None
        assert exc_info.value.__context__ is None
