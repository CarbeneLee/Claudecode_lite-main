from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import multiprocessing
import os
import time
from pathlib import Path

import pytest

import kama_claude.core.verification as verification_module
import kama_claude.core.verification_runner as verification_runner_module
from kama_claude.core.session.store import SessionStore
from kama_claude.core.verification import (
    ExecutionCompletionReceipt,
    InvalidVerificationTarget,
    SnapshotArtifact,
    SnapshotCaptureError,
    VerificationBinding,
    VerificationPathStateV1,
    VerificationResult,
    VerificationRuntimeProfileV1,
    VerificationSnapshotPolicyV1,
    VerificationSpecV1,
    _admit_verification_binding,
    _capture_execution_output_snapshot_with_policy,
    capture_execution_output_snapshot,
    materialize_snapshot_copy,
    validate_targets_in_snapshot,
)
from kama_claude.core.verification_runner import (
    DockerVerificationRunner,
    build_compileall_argv,
    build_docker_create_argv,
    build_docker_tool_probe_argv,
    build_pytest_argv,
)


# 组织 snapshot capture 测试所需的固定 identity 参数
def _capture_kwargs(root: Path, artifact_root: Path) -> dict[str, object]:
    app_digest = hashlib.sha256((root / "src/app.py").read_bytes()).hexdigest()
    return {
        "workspace_root": root,
        "verification_root": artifact_root,
        "session_id": "session-1",
        "request_id": "request-1",
        "execution_id": "execution-1",
        "execution_run_id": "run-1",
        "projection_key": "pv1:run-1:decision:v1",
        "decision_id": "decision",
        "decision_version": 1,
        "decision_content_digest": "decision-digest",
        "approval_record_digest": "approval-digest",
        "commit_receipt_digest": "receipt-digest",
        "execution_scope_digest": "scope-digest",
        "repository_snapshot_digest": "snapshot-digest",
        "workspace_id": "workspace-1",
        "expected_target_states": {
            "src/app.py": VerificationPathStateV1.create(
                exists=True,
                kind="file",
                content_digest=app_digest,
            )
        },
        "relevant_non_target_states": {},
    }


# 构造 runtime cleanup 行为测试共用的 sealed artifact、spec 和 runner
def _runtime_runner_inputs(
    tmp_path: Path,
) -> tuple[DockerVerificationRunner, VerificationSpecV1, SnapshotArtifact]:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "src").mkdir()
    (workspace / "src/app.py").write_text("print('ok')\n", encoding="utf-8")
    artifact = capture_execution_output_snapshot(
        **_capture_kwargs(workspace, tmp_path / "artifacts")
    )
    profile = VerificationRuntimeProfileV1.create(
        profile_id="python-test",
        image_ref="python@sha256:" + "a" * 64,
        expected_image_id="sha256:" + "b" * 64,
        python_executable="/usr/local/bin/python",
        expected_python_identity="Python 3.12",
    )
    spec = VerificationSpecV1.create(kind="compileall", targets=("src",), timeout_s=30)
    return DockerVerificationRunner(profile), spec, artifact


# 功能：验证 MVP VerificationSpecV1 只表示一次 kind/targets/invocation
# 设计：直接检查模型字段与 canonical digest，防止未来悄然恢复 steps 或 fail-fast 语义
def test_verification_spec_v1_is_single_step() -> None:
    spec = VerificationSpecV1.create(
        kind="pytest",
        targets=("tests/test_app.py",),
        timeout_s=30,
    )

    assert spec.schema_version == 1
    assert spec.kind == "pytest"
    assert spec.targets == ("tests/test_app.py",)
    assert not hasattr(spec, "steps")
    assert not hasattr(spec, "fail_fast")
    spec.verify_digest()


# 功能：验证 option-looking target 在 spec admission 前被拒绝
# 设计：覆盖 pytest/compileall 共用的路径安全边界，避免 shell=False 被误认为足够
@pytest.mark.parametrize("target", ["--help", "-p", "--maxfail=0", "tests/-case.py"])
def test_verification_spec_rejects_option_targets(target: str) -> None:
    with pytest.raises(ValueError, match="option"):
        VerificationSpecV1.create(kind="pytest", targets=(target,), timeout_s=30)


# 功能：验证 typed path state 拒绝缺少 digest 或自相矛盾的 malformed 状态
# 设计：直接覆盖 absent/file/present invariants，防止 snapshot expected-state 比较 fail open
def test_verification_path_state_is_strict() -> None:
    with pytest.raises(ValueError, match="SHA-256"):
        VerificationPathStateV1.create(exists=True, kind="file", content_digest="bad")
    with pytest.raises(ValueError, match="absent"):
        VerificationPathStateV1.create(exists=True, kind="absent")
    with pytest.raises(ValueError, match="present"):
        VerificationPathStateV1.create(exists=False, kind="directory")
    with pytest.raises(ValueError, match="only regular files"):
        VerificationPathStateV1.create(
            exists=True,
            kind="directory",
            content_digest="a" * 64,
        )


# 功能：验证 V1 snapshot private-root policy 不能被 production model 替换
# 设计：保留测试 helper 的预算覆盖，但拒绝通过 excluded_roots 暴露 runtime-private 内容
def test_snapshot_policy_roots_are_fixed() -> None:
    with pytest.raises(ValueError, match="roots are fixed"):
        VerificationSnapshotPolicyV1(excluded_roots=("custom-private",))


# 功能：验证 snapshot policy 版本进入 manifest digest 且普通 target 可被精确定位
# 设计：使用真实目录和 immutable artifact，区分 manifest authority 与 live workspace
def test_snapshot_manifest_binds_policy_and_target(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "src").mkdir()
    (workspace / "src/app.py").write_text("print('ok')\n", encoding="utf-8")
    artifact_root = tmp_path / "artifacts"

    artifact = capture_execution_output_snapshot(**_capture_kwargs(workspace, artifact_root))

    assert artifact.manifest.snapshot_policy_version == 1
    artifact.manifest.verify_digest()
    artifact.manifest.verify_artifact(artifact.tree_root)
    validate_targets_in_snapshot(artifact.manifest, ("src/app.py",))
    with pytest.raises(InvalidVerificationTarget):
        validate_targets_in_snapshot(artifact.manifest, ("src/missing.py",))


# 功能：验证 target 必须在 sealed snapshot 中存在后才能创建 VerificationBinding
# 设计：同时覆盖可用 target 与缺失 target，证明 live workspace 不能替代 immutable manifest
def test_verification_binding_requires_snapshot_targets(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "src").mkdir()
    (workspace / "src/app.py").write_text("print('ok')\n", encoding="utf-8")
    artifact = capture_execution_output_snapshot(
        **_capture_kwargs(workspace, tmp_path / "artifacts")
    )
    profile = VerificationRuntimeProfileV1.create(
        profile_id="python-test",
        image_ref="python@sha256:" + "a" * 64,
        expected_image_id="sha256:" + "b" * 64,
        python_executable="/usr/local/bin/python",
        expected_python_identity="Python 3.12",
    )
    spec = VerificationSpecV1.create(
        kind="compileall",
        targets=("src/app.py",),
        timeout_s=30,
    )
    binding = _admit_verification_binding(
        session_id="session-1",
        verification_request_id="request-1",
        verification_id="verification-1",
        execution_id="execution-1",
        spec=spec,
        snapshot_artifact=artifact,
        runtime_profile=profile,
    )
    assert binding.input_digest == artifact.manifest.manifest_digest
    missing = VerificationSpecV1.create(
        kind="compileall",
        targets=("src/missing.py",),
        timeout_s=30,
    )
    with pytest.raises(InvalidVerificationTarget):
        _admit_verification_binding(
            session_id="session-1",
            verification_request_id="request-2",
            verification_id="verification-2",
            execution_id="execution-2",
            spec=missing,
            snapshot_artifact=artifact,
            runtime_profile=profile,
        )


# 功能：验证 G1 不暴露看似 production 的 G2 admission alias
# 设计：直接检查模块符号边界，确保只有 provisional private helper 可被测试调用
def test_g1_admission_helper_is_private() -> None:
    assert not hasattr(verification_module, "admit_verification_binding")
    assert callable(verification_module._admit_verification_binding)


# 功能：验证 policy 排除 private roots、拒绝敏感 target 与 symlink/special file
# 设计：分别验证静默排除与 fail-closed 类型，避免把被拒绝内容复制进 verifier input
def test_snapshot_policy_excludes_private_and_rejects_symlink(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "src").mkdir()
    (workspace / "src/app.py").write_text("print('ok')\n", encoding="utf-8")
    private_root = workspace / ".git"
    private_root.mkdir()
    (private_root / "config").write_text("private\n", encoding="utf-8")
    (workspace / ".env").write_text("TOKEN=secret\n", encoding="utf-8")
    (workspace / "link.py").symlink_to(workspace / "src/app.py")

    with pytest.raises(SnapshotCaptureError, match="symlink"):
        capture_execution_output_snapshot(
            **_capture_kwargs(workspace, tmp_path / "artifacts")
        )


# 功能：验证固定 V1 policy 不排除合法 models/semantic 源码但排除嵌套 runtime cache
# 设计：用真实 nested cache 与敏感文件区分 source namespace 和 runtime-private namespace
def test_snapshot_policy_keeps_legitimate_source_names(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "src").mkdir()
    (workspace / "src/app.py").write_text("print('ok')\n", encoding="utf-8")
    (workspace / "models").mkdir()
    (workspace / "models/foo.py").write_text("MODEL = 1\n", encoding="utf-8")
    (workspace / "semantic").mkdir()
    (workspace / "semantic/parser.py").write_text("PARSER = 1\n", encoding="utf-8")
    cache_name = "_" * 2 + "pycache" + "_" * 2
    (workspace / "src" / cache_name).mkdir()
    (workspace / "src" / cache_name / "cache.pyc").write_bytes(b"cache")
    (workspace / ".env").write_text("SECRET=1\n", encoding="utf-8")
    (workspace / ".kama" / "planning").mkdir(parents=True)
    (workspace / ".kama" / "planning" / "session.json").write_text(
        "runtime\n",
        encoding="utf-8",
    )

    artifact = capture_execution_output_snapshot(
        **_capture_kwargs(workspace, tmp_path / "artifacts")
    )
    paths = {item.path for item in artifact.manifest.entries}
    assert "models/foo.py" in paths
    assert "semantic/parser.py" in paths
    assert f"src/{cache_name}/cache.pyc" not in paths
    assert ".env" not in paths
    assert ".kama/planning/session.json" not in paths


# 功能：验证小 budget 在读取前拒绝超额文件和文件总数
# 设计：通过内部测试 helper 注入固定 V1 的小上限，不暴露 production policy mutation seam
def test_snapshot_budget_rejects_before_full_read(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    for index in range(4):
        (workspace / f"file-{index}.txt").write_text("x", encoding="utf-8")
    capture_args = {
        "workspace_root": workspace,
        "verification_root": tmp_path / "artifacts",
        "session_id": "session-1",
        "request_id": "request-1",
        "execution_id": "execution-1",
        "execution_run_id": "run-1",
        "projection_key": "pv1:run-1:decision:v1",
        "decision_id": "decision",
        "decision_version": 1,
        "decision_content_digest": "decision-digest",
        "approval_record_digest": "approval-digest",
        "commit_receipt_digest": "receipt-digest",
        "execution_scope_digest": "scope-digest",
        "repository_snapshot_digest": "snapshot-digest",
        "workspace_id": "workspace-1",
        "expected_target_states": {},
        "relevant_non_target_states": {},
    }
    with pytest.raises(SnapshotCaptureError, match="file limit"):
        _capture_execution_output_snapshot_with_policy(
            **capture_args,
            policy=VerificationSnapshotPolicyV1(max_files=3),
        )

    (workspace / "large.txt").write_bytes(b"x" * 10)
    capture_args["verification_root"] = tmp_path / "artifacts-large"
    with pytest.raises(SnapshotCaptureError, match="byte limit"):
        _capture_execution_output_snapshot_with_policy(
            **capture_args,
            policy=VerificationSnapshotPolicyV1(max_bytes=5),
        )


# 功能：验证 planned-new 目标在 sealed snapshot 中仍缺失时不会被伪造为存在
# 设计：capture 可以保存 absent state，但 target admission 必须随后拒绝该路径
def test_snapshot_expected_planned_new_absent_is_typed(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "src").mkdir()
    (workspace / "src/app.py").write_text("print('ok')\n", encoding="utf-8")
    capture_args = _capture_kwargs(workspace, tmp_path / "artifacts")
    capture_args["expected_target_states"] = {
        "new.py": VerificationPathStateV1.create(exists=False, kind="absent")
    }

    artifact = capture_execution_output_snapshot(**capture_args)

    assert "new.py" not in {entry.path for entry in artifact.manifest.entries}
    with pytest.raises(InvalidVerificationTarget):
        validate_targets_in_snapshot(artifact.manifest, ("new.py",))


# 功能：验证 expected target 的内容、类型和 malformed digest 都 fail closed
# 设计：分别改变 source bytes、regular file 类型和 state digest，覆盖 capture 后不可绕过的状态比较
def test_snapshot_expected_state_mismatch_fails_closed(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "src").mkdir()
    app = workspace / "src/app.py"
    app.write_text("print('ok')\n", encoding="utf-8")
    original_digest = hashlib.sha256(app.read_bytes()).hexdigest()

    changed_args = _capture_kwargs(workspace, tmp_path / "changed")
    changed_args["expected_target_states"] = {
        "src/app.py": VerificationPathStateV1.create(
            exists=True,
            kind="file",
            content_digest=original_digest,
        )
    }
    app.write_text("print('changed')\n", encoding="utf-8")
    with pytest.raises(SnapshotCaptureError, match="expected state mismatch"):
        capture_execution_output_snapshot(**changed_args)

    type_args = _capture_kwargs(workspace, tmp_path / "type")
    type_args["expected_target_states"] = {
        "src/app.py": VerificationPathStateV1.create(
            exists=True,
            kind="file",
            content_digest=original_digest,
        )
    }
    app.unlink()
    app.mkdir()
    with pytest.raises(SnapshotCaptureError, match="expected state mismatch"):
        capture_execution_output_snapshot(**type_args)

    malformed = VerificationPathStateV1.model_validate(
        {
            "exists": True,
            "kind": "directory",
            "content_digest": None,
            "state_digest": "bad",
        }
    )
    malformed_args = type_args.copy()
    malformed_args["verification_root"] = tmp_path / "malformed"
    malformed_args["expected_target_states"] = {"src/app.py": malformed}
    with pytest.raises(SnapshotCaptureError, match="malformed"):
        capture_execution_output_snapshot(**malformed_args)


# 功能：验证 snapshot process 取消时收到 token、退出并向调用者传播原始 CancelledError
# 设计：用 fork 下的受控 finite process 验证 terminate/reap 边界，避免不可终止的无限测试进程
async def test_snapshot_capture_cancellation_is_owned_and_bounded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = multiprocessing.get_context("fork")
    started = context.Event()

    # 模拟可观察取消信号的慢 snapshot process
    def blocking_worker(
        _payload: object,
        result_path: str,
        cancel_event: object,
    ) -> None:
        started.set()
        while not cancel_event.is_set():  # type: ignore[attr-defined]
            time.sleep(0.01)
        Path(result_path).write_text(
            json.dumps({"ok": False, "error": "cancelled"}),
            encoding="utf-8",
        )

    monkeypatch.setattr(
        verification_module,
        "_SNAPSHOT_PROCESS_START_METHOD",
        "fork",
    )
    monkeypatch.setattr(
        verification_module,
        "_snapshot_worker_entry",
        blocking_worker,
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "src").mkdir()
    (workspace / "src/app.py").write_text("print('ok')\n", encoding="utf-8")
    capture_args = _capture_kwargs(workspace, tmp_path / "artifacts")
    task = asyncio.create_task(
        verification_module.capture_execution_output_snapshot_async(**capture_args)
    )
    assert await asyncio.to_thread(started.wait, 1.0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


# 功能：验证不合作 snapshot process 超过 grace 后会被强制终止且 staging 不残留
# 设计：受控 child 忽略 cancel token，但 terminate/kill 仍有毫秒级硬上限，避免无限挂起测试
async def test_snapshot_capture_noncooperative_worker_is_forced_terminated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    marker = tmp_path / "worker.pid"

    def noncooperative_worker(
        _payload: object,
        _result_path: str,
        _cancel_event: object,
    ) -> None:
        marker.write_text(str(os.getpid()), encoding="utf-8")
        while True:
            time.sleep(0.01)

    monkeypatch.setattr(verification_module, "_SNAPSHOT_PROCESS_START_METHOD", "fork")
    monkeypatch.setattr(verification_module, "_snapshot_worker_entry", noncooperative_worker)
    monkeypatch.setattr(verification_module, "SNAPSHOT_WORKER_CANCEL_WAIT_S", 0.05)
    monkeypatch.setattr(verification_module, "SNAPSHOT_WORKER_FORCE_WAIT_S", 0.05)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "src").mkdir()
    (workspace / "src/app.py").write_text("print('ok')\n", encoding="utf-8")
    task = asyncio.create_task(
        verification_module.capture_execution_output_snapshot_async(
            **_capture_kwargs(workspace, tmp_path / "artifacts")
        )
    )
    for _ in range(100):
        if marker.exists():
            break
        await asyncio.sleep(0.01)
    assert marker.exists()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=1.0)
    pid = int(marker.read_text(encoding="utf-8"))
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
    assert not (tmp_path / "artifacts" / ".staging").exists() or not list(
        (tmp_path / "artifacts" / ".staging").iterdir()
    )


# 功能：验证 disposable runtime copy cancellation 会终止 owned process 并删除 runtime parent
# 设计：让 verification-specific copy worker 忽略取消，检查 bounded kill 与 runtime cleanup 协同
async def test_runtime_copy_cancellation_is_bounded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "src").mkdir()
    (workspace / "src/app.py").write_text("print('ok')\n", encoding="utf-8")
    artifact = capture_execution_output_snapshot(**_capture_kwargs(workspace, tmp_path / "artifacts"))
    profile = VerificationRuntimeProfileV1.create(
        profile_id="python-test",
        image_ref="python@sha256:" + "a" * 64,
        expected_image_id="sha256:" + "b" * 64,
        python_executable="/usr/local/bin/python",
        expected_python_identity="Python 3.12",
    )
    runner = DockerVerificationRunner(profile)
    runtime_parent = tmp_path / "runtime-parent"
    runtime_parent.mkdir()
    destination = runtime_parent / "workspace"
    marker = tmp_path / "copy.pid"

    def noncooperative_copy(
        _manifest: dict[str, object],
        _artifact_dir: str,
        _destination: str,
        _result_path: str,
        _cancel_event: object,
    ) -> None:
        marker.write_text(str(os.getpid()), encoding="utf-8")
        while True:
            time.sleep(0.01)

    monkeypatch.setattr(verification_runner_module, "_RUNTIME_COPY_PROCESS_START_METHOD", "fork")
    monkeypatch.setattr(
        verification_runner_module,
        "_materialize_runtime_copy_worker",
        noncooperative_copy,
    )
    monkeypatch.setattr(verification_runner_module, "RUNTIME_COPY_CANCEL_WAIT_S", 0.05)
    monkeypatch.setattr(verification_runner_module, "RUNTIME_COPY_FORCE_WAIT_S", 0.05)
    task = asyncio.create_task(runner._materialize_runtime_copy(artifact, destination))
    for _ in range(100):
        if marker.exists():
            break
        await asyncio.sleep(0.01)
    assert marker.exists()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=1.0)
    assert not runtime_parent.exists()


# 功能：验证正常 runtime cleanup 也有有限 deadline，并强制终止不合作 child
# 设计：worker 忽略完成请求且不删除目录，断言 bounded SnapshotCaptureError 与无存活进程
async def test_runtime_cleanup_noncooperative_worker_is_bounded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    marker = tmp_path / "cleanup.pid"
    runtime_parent = tmp_path / "runtime-parent"
    runtime_parent.mkdir()

    def noncooperative_cleanup(_path: str, _result_path: str) -> None:
        marker.write_text(str(os.getpid()), encoding="utf-8")
        while True:
            time.sleep(0.01)

    monkeypatch.setattr(verification_runner_module, "_RUNTIME_COPY_PROCESS_START_METHOD", "fork")
    monkeypatch.setattr(
        verification_runner_module,
        "_remove_runtime_tree_worker",
        noncooperative_cleanup,
    )
    monkeypatch.setattr(verification_runner_module, "RUNTIME_COPY_CLEANUP_WAIT_S", 0.05)
    monkeypatch.setattr(verification_runner_module, "RUNTIME_COPY_FORCE_WAIT_S", 0.05)
    runner = DockerVerificationRunner(
        VerificationRuntimeProfileV1.create(
            profile_id="python-test",
            image_ref="python@sha256:" + "a" * 64,
            expected_image_id="sha256:" + "b" * 64,
            python_executable="/usr/local/bin/python",
            expected_python_identity="Python 3.12",
        )
    )
    task = asyncio.create_task(runner._remove_runtime_tree(runtime_parent))
    for _ in range(100):
        if marker.exists():
            break
        await asyncio.sleep(0.01)
    assert marker.exists()
    with pytest.raises(SnapshotCaptureError, match="cleanup"):
        await asyncio.wait_for(task, timeout=1.0)
    pid = int(marker.read_text(encoding="utf-8"))
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


# 功能：验证 primary CancelledError 不会被 runtime cleanup failure 替换
# 设计：让 verifier 取消、cleanup 同时失败，检查调用者仍收到原始取消异常
@pytest.mark.asyncio
async def test_runner_cancellation_survives_runtime_cleanup_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner, spec, artifact = _runtime_runner_inputs(tmp_path)

    async def materialize(_artifact: SnapshotArtifact, destination: Path) -> None:
        materialize_snapshot_copy(_artifact, destination)

    async def cancelled_run(*_args: object, **_kwargs: object) -> VerificationResult:
        raise asyncio.CancelledError("primary cancellation")

    async def failed_cleanup(_runtime_parent: Path) -> None:
        raise SnapshotCaptureError("cleanup failed")

    monkeypatch.setattr(runner, "_materialize_runtime_copy", materialize)
    monkeypatch.setattr(runner, "_run_from_runtime_copy", cancelled_run)
    monkeypatch.setattr(runner, "_remove_runtime_tree", failed_cleanup)
    with pytest.raises(asyncio.CancelledError, match="primary cancellation"):
        await runner.run(
            spec,
            snapshot_artifact=artifact,
            verification_id="verify-1",
            verification_request_id="request-1",
            execution_id="execution-1",
        )


# 功能：验证 pass 结果遇到 runtime cleanup failure 会转为稳定 verification_error
# 设计：使用完整 result identity，断言不把 verifier-owned temp 未清理报告成 passed
@pytest.mark.asyncio
async def test_runner_pass_becomes_cleanup_error_when_runtime_cleanup_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner, spec, artifact = _runtime_runner_inputs(tmp_path)
    expected_image_id = runner._profile.expected_image_id

    async def materialize(_artifact: SnapshotArtifact, destination: Path) -> None:
        materialize_snapshot_copy(_artifact, destination)

    async def passed_run(*_args: object, **_kwargs: object) -> VerificationResult:
        return VerificationResult.create(
            verification_id="verify-1",
            verification_request_id="request-1",
            execution_id="execution-1",
            input_digest=artifact.manifest.manifest_digest,
            spec_digest=spec.spec_digest,
            runtime_profile_digest=runner._profile.profile_digest,
            expected_image_id=expected_image_id,
            observed_container_image_id=expected_image_id,
            status="verification_passed",
            exit_code=0,
            started_at="2026-01-01T00:00:00+00:00",
            finished_at="2026-01-01T00:00:01+00:00",
        )

    async def failed_cleanup(_runtime_parent: Path) -> None:
        raise SnapshotCaptureError("cleanup failed")

    monkeypatch.setattr(runner, "_materialize_runtime_copy", materialize)
    monkeypatch.setattr(runner, "_run_from_runtime_copy", passed_run)
    monkeypatch.setattr(runner, "_remove_runtime_tree", failed_cleanup)
    result = await runner.run(
        spec,
        snapshot_artifact=artifact,
        verification_id="verify-1",
        verification_request_id="request-1",
        execution_id="execution-1",
    )
    assert result.status == "verification_error"
    assert result.reason == "runtime-copy-cleanup-unverified"


# 功能：验证 immutable snapshot 只能被复制成新的 writable runtime tree
# 设计：修改 runtime copy 后重新校验 authority artifact，证明 verifier 不会污染 source
def test_runtime_copy_is_disposable_and_source_remains_immutable(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "src").mkdir()
    (workspace / "src/app.py").write_text("print('ok')\n", encoding="utf-8")
    artifact = capture_execution_output_snapshot(
        **_capture_kwargs(workspace, tmp_path / "artifacts")
    )
    runtime_copy = tmp_path / "runtime-copy"

    materialize_snapshot_copy(artifact, runtime_copy)
    (runtime_copy / "src/app.py").write_text("print('changed')\n", encoding="utf-8")
    cache_dir = runtime_copy / ("_" * 2 + "pycache" + "_" * 2)
    cache_dir.mkdir()
    (cache_dir / "app.pyc").write_bytes(b"cache")

    artifact.manifest.verify_artifact(artifact.tree_root)
    assert (workspace / "src/app.py").read_text(encoding="utf-8") == "print('ok')\n"


# 功能：验证 completion receipt 只在后续 durable run.finished identity 已知时绑定
# 设计：用固定 manifest digest 与 event id 检查 receipt 的两阶段身份边界
def test_completion_receipt_binds_later_run_finished() -> None:
    receipt = ExecutionCompletionReceipt.create(
        session_id="session-1",
        request_id="request-1",
        execution_id="execution-1",
        execution_run_id="run-1",
        projection_key="pv1:run-1:decision:v1",
        snapshot_manifest_digest="manifest-digest",
        run_finished_event_id="evt-run-finished",
    )

    receipt.verify_digest()
    assert receipt.snapshot_manifest_digest == "manifest-digest"
    assert receipt.run_finished_event_id == "evt-run-finished"


# 功能：验证 VerificationRuntimeProfileV1 固定 image reference 与 local image id
# 设计：拒绝含糊 image digest，确保 admission 和 observed result 可比较同一身份
def test_runtime_profile_requires_exact_image_identity() -> None:
    profile = VerificationRuntimeProfileV1.create(
        profile_id="python-test",
        image_ref="python@sha256:" + "a" * 64,
        expected_image_id="sha256:" + "b" * 64,
        python_executable="/usr/local/bin/python",
        expected_python_identity="Python 3.12",
    )

    profile.verify_digest()
    assert profile.network_policy == "none"
    assert profile.image_ref.startswith("python@sha256:")
    assert profile.expected_image_id.startswith("sha256:")
    assert profile.resource_policy.pids_limit == 256


# 功能：验证 pytest/compileall 使用各自 canonical argv builder 且不接收任意 flags
# 设计：比较两个明确动作的 argv 形状，防止 generic shell/argv seam 扩大权限
def test_canonical_argv_builders_are_kind_specific() -> None:
    pytest_spec = VerificationSpecV1.create(
        kind="pytest",
        targets=("tests/test_app.py",),
        timeout_s=30,
    )
    compileall_spec = VerificationSpecV1.create(
        kind="compileall",
        targets=("src",),
        timeout_s=30,
    )

    assert build_pytest_argv(pytest_spec, "/usr/local/bin/python") == (
        "/usr/local/bin/python",
        "-m",
        "pytest",
        "-q",
        "--maxfail=1",
        "tests/test_app.py",
    )
    assert build_compileall_argv(compileall_spec, "/usr/local/bin/python") == (
        "/usr/local/bin/python",
        "-m",
        "compileall",
        "-q",
        "src",
    )


# 功能：验证 Docker verifier argv 固定为 pinned、隔离且不挂载 authority workspace
# 设计：只检查命令构造，不运行 Docker，确保 feasibility probe 与实际 runner 都拒绝 host fallback
def test_docker_argv_binds_runtime_isolation(tmp_path: Path) -> None:
    profile = VerificationRuntimeProfileV1.create(
        profile_id="python-test",
        image_ref="python@sha256:" + "a" * 64,
        expected_image_id="sha256:" + "b" * 64,
        python_executable="/usr/local/bin/python",
        expected_python_identity="Python 3.12",
    )
    spec = VerificationSpecV1.create(
        kind="compileall",
        targets=("src",),
        timeout_s=30,
    )
    runtime_copy = tmp_path / "runtime-copy"
    runtime_copy.mkdir()
    create_argv = build_docker_create_argv(
        profile,
        build_compileall_argv(spec, profile.python_executable),
        runtime_copy=runtime_copy,
        container_name="kama-test",
    )
    probe_argv = build_docker_tool_probe_argv(
        profile,
        (profile.python_executable, "-m", "compileall", "--help"),
        container_name="kama-probe",
    )
    assert "--pull=never" in create_argv
    assert "--network" in create_argv and "none" in create_argv
    assert "--cap-drop" in create_argv and "ALL" in create_argv
    assert "--read-only" in create_argv
    assert "--workdir" in create_argv and "/workspace" in create_argv
    assert "--mount" in create_argv
    assert "--pull=never" in probe_argv
    assert "--rm" in probe_argv
    assert "--mount" not in probe_argv


# 功能：验证 output limit helper 在超过上限时只保留 captured prefix
# 设计：用异步有限 stream，不启动 Docker/项目代码，证明 enforcement 发生在存储前
async def test_bounded_output_stops_at_hard_limit() -> None:
    from kama_claude.core.verification_runner import collect_bounded_output

    class _Stream:
        def __init__(self) -> None:
            self._chunks = [b"abc", b"def", b"ghi"]

        async def read(self, _size: int) -> bytes:
            await asyncio.sleep(0)
            return self._chunks.pop(0) if self._chunks else b""

    result = await collect_bounded_output(_Stream(), limit=5)  # type: ignore[arg-type]
    assert result.data == b"abcde"
    assert result.exceeded is True
    assert result.truncated is True


# 功能：验证 completion receipt 在 SessionStore 中可校验读取且不允许冲突覆盖
# 设计：使用真实文件 envelope，覆盖 derived receipt 的 durable boundary 而非只测内存模型
def test_store_persists_completion_receipt_create_once(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions")
    receipt = ExecutionCompletionReceipt.create(
        session_id="session-1",
        request_id="request-1",
        execution_id="execution-1",
        execution_run_id="run-1",
        projection_key="pv1:run-1:decision:v1",
        snapshot_manifest_digest="manifest-digest",
        run_finished_event_id="evt-run-finished",
    )

    store.write_execution_completion_receipt(receipt)
    assert store.read_execution_completion_receipt("session-1", "request-1") == receipt
    store.write_execution_completion_receipt(receipt)

    conflicting = ExecutionCompletionReceipt.create(
        session_id="session-1",
        request_id="request-1",
        execution_id="execution-1",
        execution_run_id="run-1",
        projection_key="pv1:run-1:decision:v1",
        snapshot_manifest_digest="manifest-digest",
        run_finished_event_id="evt-other",
    )
    with pytest.raises(ValueError, match="completion receipt conflict"):
        store.write_execution_completion_receipt(conflicting)


# 功能：验证 derived completion receipt 损坏后可由独立 canonical receipt 替换
# 设计：先破坏文件 bytes，再显式使用 replace 路径，确保 create-once user authority 规则不被复用
def test_corrupt_completion_receipt_can_be_replaced(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions")
    receipt = ExecutionCompletionReceipt.create(
        session_id="session-1",
        request_id="request-1",
        execution_id="execution-1",
        execution_run_id="run-1",
        projection_key="pv1:run-1:decision:v1",
        snapshot_manifest_digest="manifest-digest",
        run_finished_event_id="evt-run-finished",
    )
    store.write_execution_completion_receipt(receipt)
    store.execution_completion_receipt_path("session-1", "request-1").write_text(
        "corrupt\n",
        encoding="utf-8",
    )

    store.write_execution_completion_receipt(receipt, replace=True)

    assert store.read_execution_completion_receipt("session-1", "request-1") == receipt


# 功能：验证 Docker runner 只能从 sealed SnapshotArtifact 派生输入身份
# 设计：反射 public signature，防止 runtime copy 或 input digest 重新成为 caller 可控 authority
def test_docker_runner_requires_snapshot_artifact_input() -> None:
    parameters = inspect.signature(DockerVerificationRunner.run).parameters
    assert "snapshot_artifact" in parameters
    assert "runtime_copy" not in parameters
    assert "input_digest" not in parameters


# 功能：验证 container 尚未 inspect 成功时 startup error 不伪造 observed image identity
# 设计：stub 只推进到 create failure，直接检查 runner 返回的 infrastructure result 字段
@pytest.mark.asyncio
async def test_container_startup_error_has_no_observed_image_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = VerificationRuntimeProfileV1.create(
        profile_id="python-test",
        image_ref="python@sha256:" + "a" * 64,
        expected_image_id="sha256:" + "b" * 64,
        python_executable="/usr/local/bin/python",
        expected_python_identity="Python 3.12",
    )
    runtime_copy = tmp_path / "runtime-copy"
    runtime_copy.mkdir()
    spec = VerificationSpecV1.create(
        kind="compileall",
        targets=("src",),
        timeout_s=30,
    )
    runner = DockerVerificationRunner(profile)

    async def inspect_image() -> str:
        return profile.expected_image_id

    async def create_container(_argv: tuple[str, ...]) -> str:
        raise RuntimeError("create failed")

    async def cleanup(_name: str) -> bool:
        return True

    monkeypatch.setattr(runner, "inspect_image", inspect_image)
    monkeypatch.setattr(runner, "_create_container", create_container)
    monkeypatch.setattr(runner, "_cleanup_container", cleanup)
    result = await runner._run_from_runtime_copy(
        spec,
        runtime_copy=runtime_copy,
        verification_id="verify-1",
        verification_request_id="request-1",
        execution_id="execution-1",
        input_digest="input",
    )

    assert result.status == "verification_error"
    assert result.observed_container_image_id is None


# 功能：验证成功 container inspect 后 result 绑定精确 observed image ID
# 设计：stub create/inspect/start 三步并记录传入 identity，区分 local image inspect 与 container observation
@pytest.mark.asyncio
async def test_container_success_uses_exact_observed_image_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = VerificationRuntimeProfileV1.create(
        profile_id="python-test",
        image_ref="python@sha256:" + "a" * 64,
        expected_image_id="sha256:" + "b" * 64,
        python_executable="/usr/local/bin/python",
        expected_python_identity="Python 3.12",
    )
    runtime_copy = tmp_path / "runtime-copy"
    runtime_copy.mkdir()
    spec = VerificationSpecV1.create(kind="compileall", targets=("src",), timeout_s=30)
    runner = DockerVerificationRunner(profile)
    observed: list[str] = []

    async def inspect_image() -> str:
        return profile.expected_image_id

    async def create_container(_argv: tuple[str, ...]) -> str:
        return "container-id"

    async def inspect_container(_name: str) -> str:
        return profile.expected_image_id

    async def start_and_collect(
        _name: str,
        **kwargs: object,
    ) -> VerificationResult:
        image_id = kwargs["image_id"]
        assert isinstance(image_id, str)
        observed.append(image_id)
        return VerificationResult.create(
            verification_id="verify-1",
            verification_request_id="request-1",
            execution_id="execution-1",
            input_digest="input",
            spec_digest=spec.spec_digest,
            runtime_profile_digest=profile.profile_digest,
            expected_image_id=profile.expected_image_id,
            observed_container_image_id=image_id,
            status="verification_passed",
            exit_code=0,
            started_at="2026-01-01T00:00:00+00:00",
            finished_at="2026-01-01T00:00:01+00:00",
        )

    monkeypatch.setattr(runner, "inspect_image", inspect_image)
    monkeypatch.setattr(runner, "_create_container", create_container)
    monkeypatch.setattr(runner, "_inspect_container_image", inspect_container)
    monkeypatch.setattr(runner, "_start_and_collect", start_and_collect)
    result = await runner._run_from_runtime_copy(
        spec,
        runtime_copy=runtime_copy,
        verification_id="verify-1",
        verification_request_id="request-1",
        execution_id="execution-1",
        input_digest="input",
    )
    assert observed == [profile.expected_image_id]
    assert result.observed_container_image_id == profile.expected_image_id


# 功能：验证 named preflight 的异常路径会 stop/cleanup 并确认不留下 probe
# 设计：用受控 cancellation stub 检查 cleanup owner，不启动 Docker 或依赖 --rm 假设
@pytest.mark.asyncio
async def test_preflight_probe_cancellation_requires_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = VerificationRuntimeProfileV1.create(
        profile_id="python-test",
        image_ref="python@sha256:" + "a" * 64,
        expected_image_id="sha256:" + "b" * 64,
        python_executable="/usr/local/bin/python",
        expected_python_identity="Python 3.12",
    )
    runner = DockerVerificationRunner(profile)
    stopped: list[str] = []
    cleaned: list[str] = []

    async def cancelled_command(*_args: str, **_kwargs: object) -> tuple[int, bytes, bytes]:
        raise asyncio.CancelledError

    async def stop(name: str) -> None:
        stopped.append(name)

    async def cleanup(name: str) -> bool:
        cleaned.append(name)
        return True

    monkeypatch.setattr(runner, "_docker_command", cancelled_command)
    monkeypatch.setattr(runner, "_stop_container", stop)
    monkeypatch.setattr(runner, "_cleanup_container", cleanup)
    with pytest.raises(asyncio.CancelledError):
        await runner._run_named_probe(
            "probe-name",
            ("docker", "run", "--rm", "--name", "probe-name"),
        )
    assert stopped == ["probe-name"]
    assert cleaned == ["probe-name"]


# 功能：验证 probe timeout/error 也执行 stop/kill 与 rm/inspect absence 检查
# 设计：返回 bounded Docker error tuple，证明 cleanup 不依赖 cancellation callback 才触发
@pytest.mark.asyncio
async def test_preflight_probe_error_requires_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = VerificationRuntimeProfileV1.create(
        profile_id="python-test",
        image_ref="python@sha256:" + "a" * 64,
        expected_image_id="sha256:" + "b" * 64,
        python_executable="/usr/local/bin/python",
        expected_python_identity="Python 3.12",
    )
    runner = DockerVerificationRunner(profile)
    stopped: list[str] = []
    cleaned: list[str] = []

    async def failed_command(*_args: str, **_kwargs: object) -> tuple[int, bytes, bytes]:
        return -1, b"", b"timeout"

    async def stop(name: str) -> None:
        stopped.append(name)

    async def cleanup(name: str) -> bool:
        cleaned.append(name)
        return True

    monkeypatch.setattr(runner, "_docker_command", failed_command)
    monkeypatch.setattr(runner, "_stop_container", stop)
    monkeypatch.setattr(runner, "_cleanup_container", cleanup)
    result = await runner._run_named_probe(
        "probe-name",
        ("docker", "run", "--rm", "--name", "probe-name"),
    )
    assert result[0] == -1
    assert stopped == ["probe-name"]
    assert cleaned == ["probe-name"]


# 功能：验证 probe cancellation cleanup=false 保留取消主结果并发出二级诊断
# 设计：caplog 捕获明确 container identity，区分 cancellation 与 cleanup-unverified diagnosis
@pytest.mark.asyncio
async def test_preflight_probe_cancel_cleanup_false_is_diagnostic(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    profile = VerificationRuntimeProfileV1.create(
        profile_id="python-test",
        image_ref="python@sha256:" + "a" * 64,
        expected_image_id="sha256:" + "b" * 64,
        python_executable="/usr/local/bin/python",
        expected_python_identity="Python 3.12",
    )
    runner = DockerVerificationRunner(profile)

    async def cancelled_command(*_args: str, **_kwargs: object) -> tuple[int, bytes, bytes]:
        raise asyncio.CancelledError

    async def stop(_name: str) -> None:
        return None

    async def cleanup(_name: str) -> bool:
        return False

    monkeypatch.setattr(runner, "_docker_command", cancelled_command)
    monkeypatch.setattr(runner, "_stop_container", stop)
    monkeypatch.setattr(runner, "_cleanup_container", cleanup)
    with caplog.at_level("ERROR"):
        with pytest.raises(asyncio.CancelledError):
            await runner._run_named_probe(
                "probe-identity",
                ("docker", "run", "--rm", "--name", "probe-identity"),
            )
    assert "probe-identity" in caplog.text
    assert "cleanup-unverified" in caplog.text


# 功能：验证 VerificationBinding 是一次性 admission authority 且与 result 分离
# 设计：分别持久化 binding 和 terminal result，确保 result 不能替代 binding 身份
def test_store_persists_verification_binding_and_result(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions")
    binding = VerificationBinding.create(
        session_id="session-1",
        verification_request_id="verify-request-1",
        verification_id="verify-1",
        execution_id="execution-1",
        input_digest="input-digest",
        spec_digest="spec-digest",
        runtime_profile_digest="profile-digest",
        expected_image_id="sha256:" + "a" * 64,
    )
    store.write_verification_binding(binding)
    assert store.read_verification_binding("session-1", "verify-request-1") == binding
    store.write_verification_binding(binding)

    result = VerificationResult.create(
        verification_id="verify-1",
        verification_request_id="verify-request-1",
        execution_id="execution-1",
        input_digest="input-digest",
        spec_digest="spec-digest",
        runtime_profile_digest="profile-digest",
        expected_image_id="sha256:" + "a" * 64,
        observed_container_image_id="sha256:" + "a" * 64,
        status="verification_failed",
        reason="verification-command-failed",
        started_at="2026-01-01T00:00:00+00:00",
        finished_at="2026-01-01T00:00:01+00:00",
    )
    store.write_verification_result("session-1", result)
    assert store.read_verification_result("session-1", "verify-1") == result


# 功能：验证 startup 前未获得 container identity 的 verification_error 可持久化
# 设计：使用真实 binding/store 文件，区分 infrastructure error 与已执行 pass/fail 的强 identity 要求
def test_store_allows_pre_identity_verification_error(tmp_path: Path) -> None:
    store = SessionStore(tmp_path)
    expected = "sha256:" + "a" * 64
    binding = VerificationBinding.create(
        session_id="session-1",
        verification_request_id="request-1",
        verification_id="verify-1",
        execution_id="execution-1",
        input_digest="input",
        spec_digest="spec",
        runtime_profile_digest="profile",
        expected_image_id=expected,
    )
    store.write_verification_binding(binding)
    result = VerificationResult.create(
        verification_id="verify-1",
        verification_request_id="request-1",
        execution_id="execution-1",
        input_digest="input",
        spec_digest="spec",
        runtime_profile_digest="profile",
        expected_image_id=expected,
        observed_container_image_id=None,
        status="verification_error",
        reason="container-startup-failed",
        started_at="2026-01-01T00:00:00+00:00",
        finished_at="2026-01-01T00:00:01+00:00",
    )
    store.write_verification_result("session-1", result)
    assert store.read_verification_result("session-1", "verify-1") == result


# 功能：验证 pass/fail 没有实际 container identity 或使用错误 identity 时 fail closed
# 设计：同一 binding 覆盖两种 terminal observation，确保 pre-create error 规则不会削弱成功结果完整性
@pytest.mark.parametrize(
    ("status", "observed"),
    [
        ("verification_passed", None),
        ("verification_failed", "sha256:" + "b" * 64),
    ],
)
def test_store_rejects_terminal_result_without_exact_image(
    tmp_path: Path,
    status: str,
    observed: str | None,
) -> None:
    store = SessionStore(tmp_path)
    expected = "sha256:" + "a" * 64
    binding = VerificationBinding.create(
        session_id="session-1",
        verification_request_id="request-1",
        verification_id="verify-1",
        execution_id="execution-1",
        input_digest="input",
        spec_digest="spec",
        runtime_profile_digest="profile",
        expected_image_id=expected,
    )
    store.write_verification_binding(binding)
    result = VerificationResult.create(
        verification_id="verify-1",
        verification_request_id="request-1",
        execution_id="execution-1",
        input_digest="input",
        spec_digest="spec",
        runtime_profile_digest="profile",
        expected_image_id=expected,
        observed_container_image_id=observed,
        status=status,  # type: ignore[arg-type]
        reason="test",
        started_at="2026-01-01T00:00:00+00:00",
        finished_at="2026-01-01T00:00:01+00:00",
    )
    with pytest.raises(ValueError, match="identity"):
        store.write_verification_result("session-1", result)
