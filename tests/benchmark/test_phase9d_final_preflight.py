from __future__ import annotations

import importlib
import importlib.util
import json
import math
import os
import shlex
import subprocess
import sys
import venv
from pathlib import Path
from types import ModuleType

import pytest
from pydantic import ValidationError


# 加载待实现的 paired observer，并把缺失模块收敛为明确 RED
def _paired() -> ModuleType:
    try:
        return importlib.import_module("kama_claude.benchmark.paired")
    except ModuleNotFoundError:
        pytest.fail("paired observer module is missing")


# 从允许新增的script文件加载离线preflight边界
def _script() -> ModuleType:
    path = Path(__file__).resolve().parents[2] / "scripts" / "phase9d_paired.py"
    spec = importlib.util.spec_from_file_location("phase9d_preflight_script", path)
    if spec is None or spec.loader is None:
        pytest.fail("paired preflight script cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# 返回当前已提交 paired receipt 的仓库内路径
def _receipt_path() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "benchmarks"
        / "receipts"
        / "phase9d-repaired-v1-v2-paired-experiment.json"
    )


# 构造不含真实路径或 credential 的最小合法 final-preflight payload
def _preflight_payload() -> dict[str, object]:
    return {
        "schema_version": 1,
        "preflight_id": "phase9d-final-preflight-test",
        "status": "READY_AWAITING_EXECUTION_AUTHORIZATION",
        "created_at_utc": "2026-08-01T00:00:00Z",
        "generator": {
            "commit": "a" * 40,
            "branch": "codex/evaluation-harness",
            "remote_ref": "origin/codex/evaluation-harness",
            "remote_commit": "a" * 40,
            "ahead": 0,
            "behind": 0,
            "dirty": False,
            "git_operation_in_progress": False,
        },
        "paired_receipt": {
            "commit": "5af1ec2e1d235ab110314afe98b92e6702093657",
            "path": "benchmarks/receipts/phase9d-repaired-v1-v2-paired-experiment.json",
            "bytes": 9710,
            "sha256": "5" * 64,
            "authorization_remains_false_zero": True,
        },
        "arms": {
            "control": _arm_payload("control", "c" * 40, "control.json", "1"),
            "treatment": _arm_payload("treatment", "d" * 40, "treatment.json", "2"),
        },
        "environment": {
            "python_version": "3.12.13",
            "os": "Darwin",
            "os_release": "test",
            "architecture": "arm64",
            "sdk_distribution": "anthropic",
            "sdk_version": "0.111.0",
            "interpreter_path_sha256": "3" * 64,
            "interpreter_file_sha256": "4" * 64,
            "installed_distributions_sha256": "5" * 64,
            "installed_distribution_count": 3,
            "uv_version": "0.11.8",
            "pyproject_sha256": "6" * 64,
            "uv_lock_sha256": "7" * 64,
            "dependency_sha256": "8" * 64,
        },
        "shared_identity": {
            "provider": "deepseek",
            "model": "deepseek-v4-pro",
            "protocol": "anthropic_messages",
            "suite_sha256": "9" * 64,
            "tool_schema_sha256": "a" * 64,
            "runtime_config_sha256": "b" * 64,
            "dependency_sha256": "8" * 64,
            "max_steps": 20,
            "repeats": 3,
            "mcp_enabled": False,
        },
        "external_parent": {
            "env_name": "KAMA_PHASE9D_OUTPUT_PARENT",
            "label": "phase9d-external-output-parent",
            "canonical_path_sha256": "c" * 64,
            "canonical_object_sha256": "d" * 64,
            "absolute_path_persisted": False,
            "exists": True,
            "directory": True,
            "writable": True,
            "outside_repository": True,
            "outside_git_common_dir": True,
            "outside_all_worktrees": True,
            "canonical_resolution_stable": True,
        },
        "logical_roots": {
            "control": "control-root",
            "treatment": "treatment-root",
            "control_lexists": False,
            "treatment_lexists": False,
            "roots_created": 0,
        },
        "commands": {
            "control_spec_sha256": "e" * 64,
            "treatment_spec_sha256": "f" * 64,
            "raw_absolute_paths_persisted": False,
            "credential_value_persisted": False,
        },
        "credential": {
            "env_name": "ANTHROPIC_API_KEY",
            "present": True,
            "value_persisted": False,
            "hash_persisted": False,
        },
        "network": {"provider_calls": 0, "paid_smoke_calls": 0},
        "authorization": "AWAITING_SEPARATE_EXECUTION_AUTHORIZATION",
    }


# 构造一个不含绝对路径的 arm preflight payload
def _arm_payload(
    arm: str,
    commit: str,
    profile_path: str,
    marker: str,
) -> dict[str, object]:
    return {
        "arm": arm,
        "commit": commit,
        "profile_path": profile_path,
        "profile_id": f"{arm}-profile",
        "profile_file_sha256": marker * 64,
        "profile_canonical_sha256": marker * 64,
        "prompt_sha256": marker * 64,
        "worktree": {
            "label": f"{arm.upper()}_WORKTREE",
            "canonical_path_sha256": marker * 64,
            "absolute_path_persisted": False,
            "registered": True,
            "detached": True,
            "clean": True,
            "head_matches": True,
            "profile_exists": True,
            "source_import": {
                "source_root_sha256": marker * 64,
                "imported_module_path_sha256": marker * 64,
                "imported_module_file_sha256": marker * 64,
                "module_within_source_root": True,
                "absolute_path_persisted": False,
            },
            "outside_repository": True,
            "outside_output_parent": True,
        },
    }


# 构造一个严格 execution-authorization payload
def _authorization_payload() -> dict[str, object]:
    return {
        "schema_version": 1,
        "authorization_id": "phase9d-authorization-test",
        "status": "AUTHORIZED_FOR_ONE_PAIRED_EXECUTION",
        "created_at_utc": "2026-08-01T00:00:00Z",
        "paired_receipt": {
            "commit": "5af1ec2e1d235ab110314afe98b92e6702093657",
            "sha256": "1" * 64,
        },
        "final_preflight": {"commit": "a" * 40, "sha256": "2" * 64},
        "control_commit": "c" * 40,
        "treatment_commit": "d" * 40,
        "provider": "deepseek",
        "model": "deepseek-v4-pro",
        "protocol": "anthropic_messages",
        "attempts": {"control": 27, "treatment": 27, "total": 54},
        "maximum_authorized_attempts": 54,
        "data_egress_authorized": True,
        "cost_authorized": True,
        "cost_scope": "54 scheduled attempts under frozen profiles",
        "maximum_budget": None,
        "maximum_budget_declared_by_user": False,
        "no_paid_smoke_calls": True,
        "raw_trace_visibility": "private",
        "output_parent_sha256": "3" * 64,
        "logical_basenames": {
            "control": "control-root",
            "treatment": "treatment-root",
        },
        "no_rerun": True,
        "no_resume": True,
        "applies_once": True,
        "authorization_initially_unused": True,
        "expires_on": [
            "identity_drift",
            "root_existence",
            "environment_drift",
            "code_change",
            "credential_absence",
            "preflight_failure",
            "authorization_use_conflict",
        ],
        "human_authorization": {
            "recorded_at_utc": "2026-08-01T00:00:00Z",
            "normalized_scope": "One paired execution with cost and data egress.",
            "revoked_at_recording": False,
            "conversation_reference_persisted": False,
            "personal_identity_persisted": False,
            "cryptographic_signature_claimed": False,
        },
        "receipt_remains_immutable": True,
    }


# 功能：验证 canonical JSON固定键序、UTF-8和紧凑编码并拒绝非有限数值
# 设计：用手写literal避免调用被测hash helper计算期望值，并以NaN杀死默认json放行分支
def test_canonical_json_is_deterministic_and_rejects_non_finite() -> None:
    paired = _paired()

    assert paired.canonical_json({"z": 1, "a": "中文"}) == '{"a":"中文","z":1}'
    assert paired.canonical_sha256({"a": 1, "b": 2}) == (
        "43258cff783fe7036d8a43033f830adfc60ec037382473548ac742b888292777"
    )
    with pytest.raises(ValueError, match="canonical JSON"):
        paired.canonical_json({"bad": math.nan})


# 功能：验证paired receipt按完整严格schema加载并保持永久未授权状态
# 设计：读取真实commit工作树receipt，断言关键嵌套字段而非只检查JSON可解析
def test_load_paired_receipt_validates_frozen_contract() -> None:
    receipt = _paired().load_paired_receipt(_receipt_path())

    assert receipt.execution_plan.total_attempts == 54
    assert receipt.arms.control.commit == "7e77478f988ca61cb0087a06c686c416a27544c3"
    assert receipt.authorization.authorized_attempts == 0
    assert receipt.authorization.real_model_experiment_authorized is False


# 功能：验证receipt loader拒绝重复键、unknown字段和非有限数值
# 设计：三种损坏分别命中解析层与Pydantic层，防止宽松JSON掩盖覆盖攻击
@pytest.mark.parametrize(
    "text",
    [
        '{"schema_version":1,"schema_version":1}',
        '{"schema_version":1,"unknown":true}',
        '{"schema_version":1,"value":NaN}',
    ],
)
def test_load_paired_receipt_rejects_noncanonical_inputs(
    tmp_path: Path,
    text: str,
) -> None:
    path = tmp_path / "receipt.json"
    path.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match="paired receipt"):
        _paired().load_paired_receipt(path)


# 功能：验证FinalPreflightArtifact严格拒绝unknown字段和错误状态
# 设计：从完整合法payload分别单点mutation，证明嵌套strict模型真实生效
def test_final_preflight_artifact_is_strict_and_status_frozen() -> None:
    paired = _paired()
    payload = _preflight_payload()

    artifact = paired.FinalPreflightArtifact.model_validate(payload)
    assert artifact.network.provider_calls == 0
    invalid = json.loads(json.dumps(payload))
    invalid["unknown"] = True
    with pytest.raises(ValidationError):
        paired.FinalPreflightArtifact.model_validate(invalid)
    invalid = json.loads(json.dumps(payload))
    invalid["status"] = "AUTHORIZED_FOR_ONE_PAIRED_EXECUTION"
    with pytest.raises(ValidationError):
        paired.FinalPreflightArtifact.model_validate(invalid)


# 功能：验证ExecutionAuthorization允许null预算但拒绝未声明的非null预算
# 设计：只mutation预算字段，隔离“程序不得猜预算”的跨字段约束
def test_authorization_budget_requires_explicit_user_declaration() -> None:
    paired = _paired()
    payload = _authorization_payload()

    artifact = paired.ExecutionAuthorizationArtifact.model_validate(payload)
    assert artifact.maximum_budget is None
    payload["maximum_budget"] = {"amount": 25.0, "currency": "USD"}
    with pytest.raises(ValidationError, match="budget"):
        paired.ExecutionAuthorizationArtifact.model_validate(payload)
    payload["maximum_budget_declared_by_user"] = True
    artifact = paired.ExecutionAuthorizationArtifact.model_validate(payload)
    assert artifact.maximum_budget.amount == 25.0


# 功能：验证artifact时间戳必须是真实RFC3339 UTC而不只是匹配字符串形状
# 设计：用不存在日期和非UTC offset单点mutation，杀死仅正则校验的伪严格实现
@pytest.mark.parametrize(
    "timestamp",
    ["2026-02-30T00:00:00Z", "2026-08-01T08:00:00+08:00"],
)
def test_artifact_timestamps_require_valid_rfc3339_utc(timestamp: str) -> None:
    payload = _preflight_payload()
    payload["created_at_utc"] = timestamp

    with pytest.raises(ValidationError):
        _paired().FinalPreflightArtifact.model_validate(payload)


# 功能：验证所有artifact-facing profile/receipt paths只能保存规范仓库相对路径
# 设计：分别注入POSIX绝对路径、父目录逃逸和Windows绝对路径，覆盖跨平台路径泄漏
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("receipt", "/private/receipt.json"),
        ("profile", "../private/profile.json"),
        ("profile", "C:\\private\\profile.json"),
    ],
)
def test_artifact_paths_require_normalized_relative_values(
    field: str,
    value: str,
) -> None:
    payload = _preflight_payload()
    if field == "receipt":
        payload["paired_receipt"]["path"] = value
    else:
        payload["arms"]["control"]["profile_path"] = value

    with pytest.raises(ValidationError):
        _paired().FinalPreflightArtifact.model_validate(payload)


# 功能：验证authorization expiry conditions与human scope均为冻结规范值
# 设计：删除condition、加入重复项并制造多余空白，防止宽松list/string绕过一次性授权边界
def test_authorization_expiry_and_scope_are_canonical() -> None:
    paired = _paired()
    payload = _authorization_payload()

    missing = json.loads(json.dumps(payload))
    missing["expires_on"] = missing["expires_on"][:-1]
    with pytest.raises(ValidationError):
        paired.ExecutionAuthorizationArtifact.model_validate(missing)
    duplicated = json.loads(json.dumps(payload))
    duplicated["expires_on"].append("identity_drift")
    with pytest.raises(ValidationError):
        paired.ExecutionAuthorizationArtifact.model_validate(duplicated)
    unnormalized = json.loads(json.dumps(payload))
    unnormalized["human_authorization"]["normalized_scope"] = " One  paired execution "
    with pytest.raises(ValidationError):
        paired.ExecutionAuthorizationArtifact.model_validate(unnormalized)


# 功能：验证credential gate只返回存在布尔和固定隐私字段
# 设计：用高辨识canary检查模型、JSON和异常均不含value/length/hash
def test_credential_presence_gate_never_persists_value() -> None:
    paired = _paired()
    canary = "CREDENTIAL_SENTINEL_MUST_NOT_ESCAPE"

    evidence = paired.check_credential_presence(
        {"ANTHROPIC_API_KEY": canary},
        "ANTHROPIC_API_KEY",
    )

    text = evidence.model_dump_json()
    assert evidence.present is True
    assert set(evidence.model_dump()) == {
        "env_name",
        "present",
        "value_persisted",
        "hash_persisted",
    }
    assert canary not in text
    assert str(len(canary)) not in text


# 功能：验证credential gate对缺失和空字符串统一fail closed且异常脱敏
# 设计：分别传空mapping和空value，断言稳定错误不回显name以外的输入
@pytest.mark.parametrize("env", [{}, {"ANTHROPIC_API_KEY": ""}])
def test_credential_presence_gate_rejects_missing_or_empty(
    env: dict[str, str],
) -> None:
    with pytest.raises(ValueError, match="credential is missing") as exc_info:
        _paired().check_credential_presence(env, "ANTHROPIC_API_KEY")

    assert "ANTHROPIC_API_KEY=" not in str(exc_info.value)


# 功能：验证installed-distribution snapshot对输入顺序和PEP503名称变体稳定
# 设计：使用手写三包集合重排并替换大小写/下划线，断言hash/count相同
def test_distribution_snapshot_is_order_and_name_normalization_stable() -> None:
    paired = _paired()
    first = paired.snapshot_installed_distributions(
        [("Pydantic", "2.11.0"), ("anthropic", "0.111.0"), ("my_pkg", "1.0")]
    )
    second = paired.snapshot_installed_distributions(
        [("MY.PKG", "1.0"), ("ANTHROPIC", "0.111.0"), ("pydantic", "2.11.0")]
    )

    assert first == second
    assert first.count == 3
    assert paired.snapshot_installed_distributions(
        [("anthropic", "0.111.0"), ("Anthropic", "0.111.0")]
    ) == paired.snapshot_installed_distributions([("anthropic", "0.111.0")])


# 功能：验证distribution版本变化改变hash且冲突重复name被拒绝
# 设计：单点改变Anthropic版本并构造同名双版本，覆盖漂移和歧义两条风险
def test_distribution_snapshot_detects_version_drift_and_conflicts() -> None:
    paired = _paired()
    first = paired.snapshot_installed_distributions([("anthropic", "0.111.0")])
    second = paired.snapshot_installed_distributions([("anthropic", "0.112.0")])

    assert first.sha256 != second.sha256
    with pytest.raises(ValueError, match="distribution"):
        paired.snapshot_installed_distributions(
            [("anthropic", "0.111.0"), ("Anthropic", "0.112.0")]
        )


# 功能：验证environment snapshot只保存hash/version/count而不泄漏安装路径
# 设计：用临时interpreter/lock文件和fake distributions，扫描完整JSON排除tmp绝对路径
def test_environment_snapshot_is_deterministic_and_path_redacted(
    tmp_path: Path,
) -> None:
    paired = _paired()
    interpreter = tmp_path / "python"
    pyproject = tmp_path / "pyproject.toml"
    uv_lock = tmp_path / "uv.lock"
    interpreter.write_bytes(b"python-binary")
    pyproject.write_text("[project]\nname='fixture'\n", encoding="utf-8")
    uv_lock.write_text("version = 1\n", encoding="utf-8")

    snapshot = paired.capture_environment_snapshot(
        interpreter=interpreter,
        distributions=[("anthropic", "0.111.0")],
        python_version="3.12.13",
        system="Darwin",
        release="test",
        machine="arm64",
        sdk_distribution="anthropic",
        sdk_version="0.111.0",
        uv_version="0.11.8",
        pyproject=pyproject,
        uv_lock=uv_lock,
    )

    serialized = snapshot.model_dump_json()
    assert snapshot.installed_distribution_count == 1
    assert str(tmp_path) not in serialized
    assert set(snapshot.model_dump()) == {
        "python_version",
        "os",
        "os_release",
        "architecture",
        "sdk_distribution",
        "sdk_version",
        "interpreter_path_sha256",
        "interpreter_file_sha256",
        "installed_distributions_sha256",
        "installed_distribution_count",
        "uv_version",
        "pyproject_sha256",
        "uv_lock_sha256",
        "dependency_sha256",
    }

    interpreter.write_bytes(b"changed-python-binary")
    changed = paired.capture_environment_snapshot(
        interpreter=interpreter,
        distributions=[("anthropic", "0.111.0")],
        python_version="3.12.13",
        system="Darwin",
        release="test",
        machine="arm64",
        sdk_distribution="anthropic",
        sdk_version="0.111.0",
        uv_version="0.11.8",
        pyproject=pyproject,
        uv_lock=uv_lock,
    )
    assert changed.interpreter_path_sha256 == snapshot.interpreter_path_sha256
    assert changed.interpreter_file_sha256 != snapshot.interpreter_file_sha256


# 功能：验证output parent绑定canonical path、inode和冻结child basenames但不序列化绝对路径
# 设计：使用tmp外部目录与三个不相交边界，断言证据和root路径分离
def test_bind_output_parent_returns_redacted_stable_evidence(tmp_path: Path) -> None:
    paired = _paired()
    repository = tmp_path / "repo"
    git_common = repository / ".git"
    worktree = tmp_path / "worktree"
    parent = tmp_path / "outputs"
    for path in (git_common, worktree, parent):
        path.mkdir(parents=True)

    binding = paired.bind_output_parent(
        parent,
        repository=repository,
        git_common_dir=git_common,
        worktrees=[worktree],
        control_basename="control-root",
        treatment_basename="treatment-root",
    )

    serialized = binding.evidence.model_dump_json()
    assert binding.control_root == parent.resolve() / "control-root"
    assert binding.treatment_root == parent.resolve() / "treatment-root"
    assert str(parent.resolve()) not in serialized
    assert binding.evidence.absolute_path_persisted is False


# 功能：验证private evidence路径只由verified parent和冻结receipt ID机械派生
# 设计：重复调用检查稳定性，并扫描basename确保调用方不能注入任意authorization/result位置
def test_private_evidence_paths_are_frozen_identity_derivations(tmp_path: Path) -> None:
    parent = tmp_path / "outputs"
    parent.mkdir()
    paired = _paired()

    first = paired.derive_private_evidence_paths(parent.resolve(), "phase9d-pair-v1")
    second = paired.derive_private_evidence_paths(parent.resolve(), "phase9d-pair-v1")

    assert first == second
    assert first.root.parent == parent.resolve()
    assert first.authorization_use.parent == parent.resolve()
    assert first.authorization_use != first.root / "authorization-use.json"
    assert first.terminal_record.parent == parent.resolve()
    assert first.authorization_use != first.terminal_record
    assert first.paired_result == first.root / "paired-result"
    assert not first.root.exists()


# 功能：验证output parent拒绝repo内、包含worktree和非法logical basename
# 设计：参数化三类边界，覆盖canonical双向containment与basename逃逸
@pytest.mark.parametrize("case", ["inside_repo", "contains_worktree", "bad_basename"])
def test_bind_output_parent_rejects_unsafe_boundaries(
    tmp_path: Path,
    case: str,
) -> None:
    paired = _paired()
    repository = tmp_path / "repo"
    git_common = repository / ".git"
    repository.mkdir()
    git_common.mkdir()
    worktree = tmp_path / "worktree"
    worktree.mkdir()
    parent = tmp_path / "outputs"
    control = "control-root"
    if case == "inside_repo":
        parent = repository / "outputs"
    elif case == "contains_worktree":
        parent = tmp_path
    else:
        control = "../control-root"
    parent.mkdir(parents=True, exist_ok=True)

    with pytest.raises(ValueError, match="output parent|logical basename"):
        paired.bind_output_parent(
            parent,
            repository=repository,
            git_common_dir=git_common,
            worktrees=[worktree],
            control_basename=control,
            treatment_basename="treatment-root",
        )


# 功能：验证dangling root symlink由lexists识别为已占用
# 设计：在合法parent下创建悬空symlink，确保不存在目标也不能被当成可用root
def test_bind_output_parent_rejects_dangling_root_symlink(tmp_path: Path) -> None:
    paired = _paired()
    repository = tmp_path / "repo"
    git_common = repository / ".git"
    worktree = tmp_path / "worktree"
    parent = tmp_path / "outputs"
    for path in (git_common, worktree, parent):
        path.mkdir(parents=True)
    (parent / "control-root").symlink_to(parent / "missing")

    with pytest.raises(ValueError, match="output root already exists"):
        paired.bind_output_parent(
            parent,
            repository=repository,
            git_common_dir=git_common,
            worktrees=[worktree],
            control_basename="control-root",
            treatment_basename="treatment-root",
        )


# 功能：验证rebind检测同路径对象替换和symlink retarget
# 设计：先绑定目录再改名重建同名目录，path hash不变但inode fingerprint必须触发漂移
def test_rebind_output_parent_detects_object_replacement(tmp_path: Path) -> None:
    paired = _paired()
    repository = tmp_path / "repo"
    git_common = repository / ".git"
    worktree = tmp_path / "worktree"
    parent = tmp_path / "outputs"
    for path in (git_common, worktree, parent):
        path.mkdir(parents=True)
    binding = paired.bind_output_parent(
        parent,
        repository=repository,
        git_common_dir=git_common,
        worktrees=[worktree],
        control_basename="control-root",
        treatment_basename="treatment-root",
    )
    moved = tmp_path / "old-outputs"
    parent.rename(moved)
    parent.mkdir()

    with pytest.raises(ValueError, match="output parent identity drift"):
        paired.rebind_output_parent(
            binding.evidence,
            parent,
            repository=repository,
            git_common_dir=git_common,
            worktrees=[worktree],
            control_basename="control-root",
            treatment_basename="treatment-root",
        )


# 功能：验证output parent symlink在preflight后改指向另一目录会被rebind拒绝
# 设计：保持调用路径字符串不变而切换symlink target，覆盖纯inode替换之外的retarget攻击
def test_rebind_output_parent_detects_symlink_retarget(tmp_path: Path) -> None:
    paired = _paired()
    repository = tmp_path / "repo"
    git_common = repository / ".git"
    worktree = tmp_path / "worktree"
    first = tmp_path / "first-output"
    second = tmp_path / "second-output"
    link = tmp_path / "output-link"
    for path in (git_common, worktree, first, second):
        path.mkdir(parents=True)
    link.symlink_to(first, target_is_directory=True)
    binding = paired.bind_output_parent(
        link.absolute(),
        repository=repository,
        git_common_dir=git_common,
        worktrees=[worktree],
        control_basename="control-root",
        treatment_basename="treatment-root",
    )
    link.unlink()
    link.symlink_to(second, target_is_directory=True)

    with pytest.raises(ValueError, match="output parent identity drift"):
        paired.rebind_output_parent(
            binding.evidence,
            link.absolute(),
            repository=repository,
            git_common_dir=git_common,
            worktrees=[worktree],
            control_basename="control-root",
            treatment_basename="treatment-root",
        )


# 功能：验证logical command spec稳定且不包含absolute path或credential
# 设计：同一逻辑输入重复hash，再扫描JSON并改变arm/profile确认identity分离
def test_command_spec_is_logical_redacted_and_arm_specific(tmp_path: Path) -> None:
    paired = _paired()
    control = paired.build_command_spec(
        arm="control",
        interpreter_label="PHASE9D_PYTHON",
        interpreter_sha256="1" * 64,
        worktree_label="C1_WORKTREE",
        worktree_sha256="2" * 64,
        profile_path="benchmarks/experiments/control.json",
        output_basename="control-root",
        expected_attempts=27,
    )
    same = paired.build_command_spec(**control.model_dump(exclude={"spec_sha256"}))
    treatment = paired.build_command_spec(
        arm="treatment",
        interpreter_label="PHASE9D_PYTHON",
        interpreter_sha256="1" * 64,
        worktree_label="C2_WORKTREE",
        worktree_sha256="3" * 64,
        profile_path="benchmarks/experiments/treatment.json",
        output_basename="treatment-root",
        expected_attempts=27,
    )

    text = control.model_dump_json()
    assert control.spec_sha256 == same.spec_sha256
    assert control.spec_sha256 != treatment.spec_sha256
    assert str(tmp_path) not in text
    assert "ANTHROPIC_API_KEY" in control.allowed_env_names
    assert "HOME" in control.allowed_env_names
    assert "TMPDIR" in control.allowed_env_names
    assert "credential" not in text.lower()
    assert control.shell is False


# 功能：验证atomic artifact writer拒绝覆盖并可严格回读同一model
# 设计：在tmp target写一次后重复写，杀死replace覆盖和非原子宽松写入实现
def test_write_canonical_artifact_is_create_once_and_roundtrips(tmp_path: Path) -> None:
    paired = _paired()
    artifact = paired.FinalPreflightArtifact.model_validate(_preflight_payload())
    target = tmp_path / "preflight.json"

    paired.write_canonical_artifact(target, artifact)

    loaded = paired.read_strict_artifact(target, paired.FinalPreflightArtifact)
    assert loaded == artifact
    assert target.read_text(encoding="utf-8") == paired.canonical_json(
        artifact.model_dump(mode="json")
    ) + "\n"
    with pytest.raises(ValueError, match="artifact target already exists"):
        paired.write_canonical_artifact(target, artifact)


# 功能：验证artifact模型拒绝NaN且JSON中不出现主机绝对路径
# 设计：mutation环境count类型和非有限预算，覆盖strict与allow_inf_nan配置
def test_artifact_models_reject_non_finite_and_path_leakage(tmp_path: Path) -> None:
    paired = _paired()
    payload = _authorization_payload()
    payload["maximum_budget"] = {"amount": math.inf, "currency": "USD"}
    payload["maximum_budget_declared_by_user"] = True

    with pytest.raises(ValidationError):
        paired.ExecutionAuthorizationArtifact.model_validate(payload)
    text = paired.FinalPreflightArtifact.model_validate(
        _preflight_payload()
    ).model_dump_json()
    assert str(tmp_path) not in text


# 功能：验证output parent writable signal失败时拒绝而不创建probe文件
# 设计：monkeypatch os.access而保留真实目录/stat，隔离权限分支并检查目录仍为空
def test_bind_output_parent_rejects_nonwritable_signal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paired = _paired()
    repository = tmp_path / "repo"
    git_common = repository / ".git"
    worktree = tmp_path / "worktree"
    parent = tmp_path / "outputs"
    for path in (git_common, worktree, parent):
        path.mkdir(parents=True)
    monkeypatch.setattr(os, "access", lambda _path, _mode: False)

    with pytest.raises(ValueError, match="not writable"):
        paired.bind_output_parent(
            parent,
            repository=repository,
            git_common_dir=git_common,
            worktrees=[worktree],
            control_basename="control-root",
            treatment_basename="treatment-root",
        )

    assert list(parent.iterdir()) == []


# 功能：验证offline generate_final_preflight在tmp生成可严格回读的artifact
# 设计：仅fake Git/worktree observer，保留真实receipt、environment、path和atomic writer边界
def test_generate_final_preflight_offline_writes_redacted_tmp_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paired = _paired()
    script = _script()
    receipt = paired.load_paired_receipt(_receipt_path())
    repository = Path(__file__).resolve().parents[2]
    output_parent = tmp_path / "external-output"
    control_worktree = tmp_path / "control-worktree"
    treatment_worktree = tmp_path / "treatment-worktree"
    for path in (output_parent, control_worktree, treatment_worktree):
        path.mkdir()
    sentinel = "OFFLINE_PREFLIGHT_CREDENTIAL_SENTINEL"
    monkeypatch.setenv("KAMA_PHASE9D_OUTPUT_PARENT", str(output_parent))
    monkeypatch.setenv("ANTHROPIC_API_KEY", sentinel)
    generator = paired.GitSnapshot(
        commit="5" * 40,
        branch="codex/evaluation-harness",
        remote_ref="origin/codex/evaluation-harness",
        remote_commit="5" * 40,
        ahead=0,
        behind=0,
        dirty=False,
        git_operation_in_progress=False,
    )
    monkeypatch.setattr(script, "_capture_git_snapshot", lambda *_args: generator)

    # 将临时worktree观测投影为冻结receipt中的arm identity
    def capture_arm(**kwargs: object) -> tuple[object, object]:
        arm = str(kwargs["arm"])
        receipt_arm = getattr(receipt.arms, arm)
        marker = "1" if arm == "control" else "2"
        evidence = paired.ArmPreflightEvidence(
            arm=arm,
            commit=receipt_arm.commit,
            profile_path=receipt_arm.profile_path,
            profile_id=receipt_arm.profile_id,
            profile_file_sha256=receipt_arm.profile_file_sha256,
            profile_canonical_sha256=receipt_arm.profile_canonical_sha256,
            prompt_sha256=receipt_arm.prompt_sha256,
            worktree=paired.WorktreeEvidence(
                label=f"{receipt_arm.label}_WORKTREE",
                canonical_path_sha256=marker * 64,
                absolute_path_persisted=False,
                registered=True,
                detached=True,
                clean=True,
                head_matches=True,
                profile_exists=True,
                source_import=paired.SourceImportEvidence(
                    source_root_sha256=marker * 64,
                    imported_module_path_sha256=marker * 64,
                    imported_module_file_sha256=marker * 64,
                    module_within_source_root=True,
                    absolute_path_persisted=False,
                ),
                outside_repository=True,
                outside_output_parent=True,
            ),
        )
        execution_tests = importlib.import_module(
            "tests.benchmark.test_phase9d_paired_execution"
        )
        declared = execution_tests._identity(receipt, arm).declared
        return evidence, declared

    monkeypatch.setattr(script, "_capture_arm_preflight", capture_arm)
    artifact_path = tmp_path / "artifacts" / "final-preflight.json"

    artifact = script.generate_final_preflight(
        receipt_path=_receipt_path(),
        artifact_path=artifact_path,
        output_parent_env="KAMA_PHASE9D_OUTPUT_PARENT",
        control_worktree=control_worktree,
        treatment_worktree=treatment_worktree,
        interpreter=Path(sys.executable),
        repository=repository,
        remote_ref="origin/codex/evaluation-harness",
    )

    assert paired.read_strict_artifact(
        artifact_path,
        paired.FinalPreflightArtifact,
    ) == artifact
    assert sentinel not in artifact_path.read_text(encoding="utf-8")
    assert artifact.network.provider_calls == 0
    assert artifact.logical_roots.roots_created == 0
    assert not (output_parent / artifact.logical_roots.control).exists()
    assert not (output_parent / artifact.logical_roots.treatment).exists()


# 功能：验证final preflight禁止C1/C2绑定到同一canonical worktree
# 设计：使用同一tmp目录触发最早路径门禁，不需要Git mutation、credential或正式worktree
def test_generate_final_preflight_rejects_same_worktree_path(tmp_path: Path) -> None:
    worktree = tmp_path / "one-worktree"
    worktree.mkdir()

    with pytest.raises(ValueError, match="worktrees must be distinct"):
        _script().generate_final_preflight(
            receipt_path=_receipt_path(),
            artifact_path=tmp_path / "preflight.json",
            output_parent_env="MISSING_OFFLINE_ENV",
            control_worktree=worktree,
            treatment_worktree=worktree,
            interpreter=Path(sys.executable),
            repository=Path(__file__).resolve().parents[2],
            remote_ref="origin/codex/evaluation-harness",
        )


# 功能：验证真实解释器只能从指定arm source root导入kama_claude并返回脱敏hash证据
# 设计：创建最小真实package后运行subprocess probe，断言module bytes/path hash且artifact不含绝对路径
def test_source_import_probe_resolves_real_arm_package_without_path_leak(
    tmp_path: Path,
) -> None:
    package = tmp_path / "arm" / "src" / "kama_claude"
    package.mkdir(parents=True)
    module = package / "__init__.py"
    module.write_text("SOURCE_MARKER = 'arm'\n", encoding="utf-8")

    evidence = _script()._probe_arm_source_import(
        Path(sys.executable),
        package.parent,
        timeout_s=2.0,
    )

    assert evidence.source_root_sha256 == __import__("hashlib").sha256(
        str(package.parent.resolve()).encode("utf-8")
    ).hexdigest()
    assert evidence.imported_module_file_sha256 == __import__("hashlib").sha256(
        module.read_bytes()
    ).hexdigest()
    assert str(tmp_path) not in evidence.model_dump_json()


# 功能：验证真实subprocess可从包含中文字符的arm source路径完成正确import
# 设计：沿用production probe而不mock subprocess，直接杀死child/parent serializer不一致缺陷
def test_source_import_probe_accepts_unicode_source_path(tmp_path: Path) -> None:
    package = tmp_path / "工作树-控制臂" / "源码" / "kama_claude"
    package.mkdir(parents=True)
    module = package / "__init__.py"
    module.write_text("SOURCE_MARKER = 'unicode-arm'\n", encoding="utf-8")

    evidence = _script()._probe_arm_source_import(
        Path(sys.executable),
        package.parent,
        timeout_s=2.0,
    )

    assert evidence.imported_module_file_sha256 == __import__("hashlib").sha256(
        module.read_bytes()
    ).hexdigest()
    assert evidence.module_within_source_root is True
    assert evidence.absolute_path_persisted is False


# 功能：验证Unicode source中只namespace占位时真实解析到main checkout必须拒绝
# 设计：先用同一解释器证明fixture确实导入主checkout，再调production probe检查containment
def test_source_import_probe_rejects_real_unicode_main_checkout_resolution(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "工作树-错误主分支" / "src"
    (source_root / "kama_claude").mkdir(parents=True)
    environment = {
        "PYTHONPATH": str(source_root),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
    }
    observed = subprocess.run(
        [sys.executable, "-c", "import kama_claude; print(kama_claude.__file__)"],
        cwd=source_root,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    resolved = Path(observed.stdout.strip()).resolve(strict=True)
    assert not resolved.is_relative_to(source_root.resolve(strict=True))

    with pytest.raises(ValueError, match="source import probe"):
        _script()._probe_arm_source_import(
            Path(sys.executable),
            source_root,
            timeout_s=2.0,
        )


# 功能：验证真实解释器从隔离site-packages导入时production probe必须拒绝
# 设计：创建无pip复制型venv并写入最小包，避免mock subprocess掩盖真实sys.path顺序
def test_source_import_probe_rejects_real_site_packages_resolution(
    tmp_path: Path,
) -> None:
    environment_root = tmp_path / "isolated-venv"
    venv.EnvBuilder(with_pip=False, symlinks=True).create(environment_root)
    venv_interpreter = environment_root / "bin" / "python"
    if os.name == "nt":
        venv_interpreter = environment_root / "Scripts" / "python.exe"
    site_result = subprocess.run(
        [
            str(venv_interpreter),
            "-c",
            "import sysconfig; print(sysconfig.get_paths()['purelib'])",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    site_package = Path(site_result.stdout.strip()) / "kama_claude"
    site_package.mkdir()
    site_module = site_package / "__init__.py"
    site_module.write_text("SOURCE_MARKER = 'site-package'\n", encoding="utf-8")
    source_root = tmp_path / "工作树-site-packages" / "src"
    (source_root / "kama_claude").mkdir(parents=True)
    interpreter = environment_root / "selected-python"
    interpreter.write_text(
        "#!/bin/sh\nexec "
        + shlex.quote(str(venv_interpreter))
        + ' "$@"\n',
        encoding="utf-8",
    )
    interpreter.chmod(0o700)

    with pytest.raises(ValueError, match="source import probe"):
        _script()._probe_arm_source_import(
            interpreter,
            source_root,
            timeout_s=2.0,
        )


# 功能：验证合法但非canonical的child JSON经strict parse后仍能产生canonical hash evidence
# 设计：只替换subprocess transport输出格式，保留真实module containment与production解析逻辑
def test_source_import_probe_accepts_valid_noncanonical_child_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = tmp_path / "unicode-工作树" / "src" / "kama_claude"
    package.mkdir(parents=True)
    module = package / "__init__.py"
    module.write_text("SOURCE_MARKER = 'pretty-json'\n", encoding="utf-8")
    payload = json.dumps(
        {"module_file": str(module.resolve())},
        ensure_ascii=False,
        indent=2,
    ) + "\n"
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            args=[], returncode=0, stdout=payload, stderr=""
        ),
    )

    evidence = _script()._probe_arm_source_import(
        Path(sys.executable),
        package.parent,
        timeout_s=2.0,
    )

    assert evidence.imported_module_file_sha256 == __import__("hashlib").sha256(
        module.read_bytes()
    ).hexdigest()


# 功能：验证source目录存在但解释器实际导入main/site-packages checkout时必须拒绝
# 设计：用canonical subprocess结果分别返回两个错误checkout文件，隔离验证resolved module containment
@pytest.mark.parametrize("origin", ["main-checkout", "site-packages"])
def test_source_import_probe_rejects_wrong_checkout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    origin: str,
) -> None:
    source_root = tmp_path / "错误来源-工作树" / "src"
    (source_root / "kama_claude").mkdir(parents=True)
    wrong_module = tmp_path / origin / "kama_claude" / "__init__.py"
    wrong_module.parent.mkdir(parents=True)
    wrong_module.write_text("WRONG = True\n", encoding="utf-8")
    payload = json.dumps(
        {"module_file": str(wrong_module.resolve())},
        sort_keys=True,
        separators=(",", ":"),
    ) + "\n"
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            args=[], returncode=0, stdout=payload, stderr=""
        ),
    )

    with pytest.raises(ValueError, match="source import probe"):
        _script()._probe_arm_source_import(
            Path(sys.executable),
            source_root,
            timeout_s=2.0,
        )


# 功能：验证probe对缺失模块、损坏JSON、非零退出与超时统一fail closed
# 设计：参数化真实缺模块和受控subprocess异常，要求固定错误且不回显stderr/path
@pytest.mark.parametrize(
    "case",
    ["missing", "duplicate", "nonfinite", "extra", "invalid_utf8", "exit", "timeout"],
)
def test_source_import_probe_rejects_unverifiable_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
) -> None:
    source_root = tmp_path / "arm" / "src"
    source_root.mkdir(parents=True)
    if case in {"duplicate", "nonfinite", "extra"}:
        module_path = str(tmp_path / "wrong.py")
        payload = {
            "duplicate": (
                '{"module_file":'
                + json.dumps(module_path)
                + ',"module_file":'
                + json.dumps(module_path)
                + "}\n"
            ),
            "nonfinite": '{"module_file":NaN}\n',
            "extra": json.dumps(
                {"module_file": module_path, "unexpected": True},
                separators=(",", ":"),
            )
            + "\n",
        }[case]
        monkeypatch.setattr(
            subprocess,
            "run",
            lambda *_args, **_kwargs: subprocess.CompletedProcess(
                args=[], returncode=0, stdout=payload, stderr=""
            ),
        )
    elif case == "invalid_utf8":
        monkeypatch.setattr(
            subprocess,
            "run",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid")
            ),
        )
    elif case == "exit":
        monkeypatch.setattr(
            subprocess,
            "run",
            lambda *_args, **_kwargs: subprocess.CompletedProcess(
                args=[], returncode=1, stdout="", stderr="SECRET_STDERR_CANARY"
            ),
        )
    elif case == "timeout":
        def timeout(*_args: object, **_kwargs: object) -> object:
            raise subprocess.TimeoutExpired(cmd=["python"], timeout=2.0)

        monkeypatch.setattr(subprocess, "run", timeout)

    with pytest.raises(ValueError, match="source import probe") as exc_info:
        _script()._probe_arm_source_import(
            Path(sys.executable),
            source_root,
            timeout_s=2.0,
        )

    assert "SECRET_STDERR_CANARY" not in str(exc_info.value)
    assert str(tmp_path) not in str(exc_info.value)


# 功能：验证kama_claude package经symlink逃出arm source root时probe必须拒绝
# 设计：真实创建指向外部package的目录symlink，让Python成功导入后再由canonical containment杀死
def test_source_import_probe_rejects_symlink_escape(tmp_path: Path) -> None:
    source_root = tmp_path / "符号链接-工作树" / "src"
    outside = tmp_path / "outside" / "kama_claude"
    source_root.mkdir(parents=True)
    outside.mkdir(parents=True)
    (outside / "__init__.py").write_text("ESCAPED = True\n", encoding="utf-8")
    (source_root / "kama_claude").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="source import probe"):
        _script()._probe_arm_source_import(
            Path(sys.executable),
            source_root,
            timeout_s=2.0,
        )


# 在临时目录创建一个单commit Git仓库供真实operation marker测试
def _git_repository(path: Path) -> Path:
    path.mkdir()
    environment = {
        "GIT_AUTHOR_EMAIL": "phase9d@example.invalid",
        "GIT_AUTHOR_NAME": "Phase9D Test",
        "GIT_COMMITTER_EMAIL": "phase9d@example.invalid",
        "GIT_COMMITTER_NAME": "Phase9D Test",
        "PATH": os.environ.get("PATH", ""),
    }
    subprocess.run(
        ["git", "init", "-b", "main"],
        cwd=path,
        env=environment,
        check=True,
        capture_output=True,
    )
    (path / "tracked.txt").write_text("baseline\n", encoding="utf-8")
    subprocess.run(
        ["git", "add", "tracked.txt"],
        cwd=path,
        env=environment,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "commit", "-m", "baseline"],
        cwd=path,
        env=environment,
        check=True,
        capture_output=True,
    )
    return path


# 返回worktree-aware的Git operation marker路径
def _git_marker(repository: Path, marker: str) -> Path:
    result = subprocess.run(
        ["git", "rev-parse", "--git-path", marker],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
        env={"PATH": os.environ.get("PATH", "")},
    )
    path = Path(result.stdout.strip())
    return path if path.is_absolute() else repository / path


# 功能：验证preflight Git snapshot使用worktree-aware路径识别全部operation状态
# 设计：真实临时Git metadata逐个创建marker，防止遗漏rebase-apply、sequencer或bisect
@pytest.mark.parametrize(
    ("marker", "directory"),
    [
        ("MERGE_HEAD", False),
        ("CHERRY_PICK_HEAD", False),
        ("REVERT_HEAD", False),
        ("rebase-merge", True),
        ("rebase-apply", True),
        ("sequencer", True),
        ("BISECT_START", False),
        ("BISECT_LOG", False),
    ],
    ids=[
        "merge",
        "cherry-pick",
        "revert",
        "rebase-merge",
        "rebase-apply",
        "sequencer",
        "bisect-start",
        "bisect-log",
    ],
)
def test_git_snapshot_detects_worktree_aware_operation_markers(
    tmp_path: Path,
    marker: str,
    directory: bool,
) -> None:
    repository = _git_repository(tmp_path / "repository")
    marker_path = _git_marker(repository, marker)
    if directory:
        marker_path.mkdir(parents=True)
    else:
        marker_path.parent.mkdir(parents=True, exist_ok=True)
        marker_path.write_text("operation\n", encoding="utf-8")

    with pytest.raises(ValueError, match="git_operation_in_progress"):
        _script()._capture_git_snapshot(repository, "HEAD")


# 功能：验证无operation marker的clean Git snapshot保持false
# 设计：复用同一真实临时仓库作为negative control，避免marker测试只断言常量true
def test_git_snapshot_clean_repository_has_no_operation(tmp_path: Path) -> None:
    repository = _git_repository(tmp_path / "repository")

    snapshot = _script()._capture_git_snapshot(repository, "HEAD")

    assert snapshot.git_operation_in_progress is False
