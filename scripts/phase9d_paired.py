from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from kama_claude.benchmark.experiment import (
    DeclaredExperimentIdentity,
    RepositoryIdentity,
    capture_declared_identity,
    load_experiment_profile,
)
from kama_claude.benchmark.paired import (
    ArmAudit,
    ArmPreflightEvidence,
    AuthorizationReservationError,
    AuthorizationUseRecord,
    BetweenArmEvidence,
    ChildTerminationEvidence,
    CommandHashes,
    ExecutionAuthorizationArtifact,
    ExpectedArmIdentity,
    FinalPreflightArtifact,
    FreshBetweenArmGitArtifacts,
    FreshTrackedArtifactEvidence,
    GitSnapshot,
    LogicalRootsEvidence,
    PairedReceipt,
    PairedResult,
    PairEvent,
    PairState,
    PairTerminalRecord,
    PairTransition,
    PreflightArms,
    PrivateEvidencePaths,
    SharedPreflightIdentity,
    SourceImportEvidence,
    WorktreeEvidence,
    WorktreeObservation,
    audit_arm_result,
    bind_output_parent,
    build_command_spec,
    build_final_preflight_artifact,
    build_paired_result,
    build_terminal_record,
    canonical_json,
    canonical_sha256,
    capture_environment_snapshot,
    check_credential_presence,
    derive_pair_outcome,
    derive_private_evidence_paths,
    load_paired_receipt,
    observe_receipt_reference,
    parse_strict_json_object,
    read_paired_result_bundle,
    read_strict_artifact,
    rebind_output_parent,
    recompute_classification,
    reduce_pair_transition,
    reserve_authorization_use,
    revalidate_before_treatment,
    terminal_event_for_failure,
    validate_execution_authorization,
    validate_pair_terminal_exclusivity,
    validate_worktree_binding,
    write_canonical_artifact,
    write_paired_result,
    write_terminal_record,
)


@dataclass(frozen=True)
class ChildResult:
    spawned: bool
    exit_code: int | None
    signal_number: int | None
    cancelled: bool
    process_group_id: int | None
    failure_category: str | None
    cleanup_term_sent: bool = False
    cleanup_kill_sent: bool = False
    process_group_gone: bool = True


@dataclass(frozen=True)
class _ProcessGroupCleanup:
    term_sent: bool
    kill_sent: bool
    gone: bool


@dataclass(frozen=True)
class ArmLaunch:
    expected: ExpectedArmIdentity
    argv: tuple[str, ...]
    cwd: Path
    env: dict[str, str]
    output_root: Path
    stdout_path: Path
    stderr_path: Path


@dataclass(frozen=True)
class PairExecutionSummary:
    state: PairState
    control: ArmAudit | None
    treatment: ArmAudit | None
    result: PairedResult | None
    terminal: PairTerminalRecord | None
    result_durability_warning: bool = False
    evidence_paths: PrivateEvidencePaths | None = None


@dataclass
class _PairExecutionCheckpoint:
    state: PairState
    transitions: list[PairTransition]
    authorization_consumed: bool = False
    control_audit: ArmAudit | None = None
    treatment_audit: ArmAudit | None = None
    control_child: ChildResult | None = None
    treatment_child: ChildResult | None = None
    parent_interrupt_handler: Callable[[KeyboardInterrupt | SystemExit], None] | None = (
        None
    )


# 将一个冻结arm身份物化为shell=false且source/private-output绑定的child launch
def _materialize_arm_launch(
    *,
    arm: str,
    receipt: PairedReceipt,
    declared: DeclaredExperimentIdentity,
    worktree: Path,
    output_root: Path,
    private_paths: PrivateEvidencePaths,
    interpreter: Path | str,
    source_environment: Mapping[str, str],
    credential_env: str,
) -> ArmLaunch:
    if arm not in {"control", "treatment"}:
        raise ValueError("paired arm is invalid")
    canonical_worktree = worktree.resolve(strict=True)
    source_root = (canonical_worktree / "src").resolve(strict=True)
    if not source_root.is_dir() or not source_root.is_relative_to(canonical_worktree):
        raise ValueError("arm source binding is invalid")
    receipt_arm = getattr(receipt.arms, arm)
    env = _build_child_environment(
        source_environment,
        credential_env=credential_env,
        source_root=source_root,
        home=private_paths.root / "home",
        tmpdir=private_paths.root / "tmp",
    )
    stdout_path = (
        private_paths.control_stdout
        if arm == "control"
        else private_paths.treatment_stdout
    )
    stderr_path = (
        private_paths.control_stderr
        if arm == "control"
        else private_paths.treatment_stderr
    )
    return ArmLaunch(
        expected=_expected_arm_identity(
            arm=arm,
            receipt=receipt,
            declared=declared,
        ),
        argv=(
            str(Path(interpreter).resolve(strict=True)),
            "-m",
            "kama_claude.benchmark.cli",
            "run",
            "--experiment",
            receipt_arm.profile_path,
            "--output",
            str(output_root),
        ),
        cwd=canonical_worktree,
        env=env,
        output_root=output_root,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
    )


# 从显式caller mapping构造child最小allowlist环境并覆盖所有路径身份
def _build_child_environment(
    source_environment: Mapping[str, str],
    *,
    credential_env: str,
    source_root: Path,
    home: Path,
    tmpdir: Path,
) -> dict[str, str]:
    credential = source_environment.get(credential_env)
    if not credential:
        raise ValueError("experiment credential is missing")
    canonical_source = source_root.resolve(strict=False)
    canonical_home = home.resolve(strict=False)
    canonical_tmp = tmpdir.resolve(strict=False)
    if (
        not canonical_source.is_absolute()
        or canonical_home.parent != canonical_tmp.parent
        or canonical_home == canonical_tmp
    ):
        raise ValueError("child environment path binding is invalid")
    environment = {
        credential_env: credential,
        "HOME": str(canonical_home),
        "PATH": source_environment.get("PATH") or os.defpath,
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": str(canonical_source),
        "TMPDIR": str(canonical_tmp),
    }
    for locale_name in ("LANG", "LC_ALL"):
        locale_value = source_environment.get(locale_name)
        if locale_value:
            environment[locale_name] = locale_value
    return environment


# 探测稳定process group是否仍有成员且不暴露PID到artifact
def _process_group_exists(process_group_id: int) -> bool:
    try:
        os.killpg(process_group_id, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


# 在给定deadline内等待稳定process group完全消失
def _wait_for_process_group_exit(process_group_id: int, timeout_s: float) -> bool:
    deadline = time.monotonic() + timeout_s
    while _process_group_exists(process_group_id):
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.02)
    return True


# 清理leader已退出仍存活的descendants并记录真实TERM/KILL动作
def _terminate_process_group(
    process: subprocess.Popen[bytes],
    process_group_id: int,
) -> _ProcessGroupCleanup:
    term_sent = False
    kill_sent = False
    if not _process_group_exists(process_group_id):
        if process.poll() is None:
            process.wait()
        return _ProcessGroupCleanup(False, False, True)
    try:
        os.killpg(process_group_id, signal.SIGTERM)
        term_sent = True
    except (ProcessLookupError, PermissionError):
        pass
    if not _wait_for_process_group_exit(process_group_id, 0.25):
        try:
            os.killpg(process_group_id, signal.SIGKILL)
            kill_sent = True
        except (ProcessLookupError, PermissionError):
            pass
    if process.poll() is None:
        try:
            process.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process_group_id, signal.SIGKILL)
                kill_sent = True
            except (ProcessLookupError, PermissionError):
                pass
            process.wait(timeout=1.0)
    gone = _wait_for_process_group_exit(process_group_id, 1.0)
    return _ProcessGroupCleanup(term_sent, kill_sent, gone)


# 以shell=false和独立session运行child，将raw输出仅写入private files
def run_private_child(
    argv: Sequence[str],
    *,
    cwd: Path | str,
    env: dict[str, str],
    stdout_path: Path | str,
    stderr_path: Path | str,
    cancel_event: threading.Event | None = None,
) -> ChildResult:
    stdout_target = Path(stdout_path)
    stderr_target = Path(stderr_path)
    stdout_target.parent.mkdir(parents=True, exist_ok=True)
    stderr_target.parent.mkdir(parents=True, exist_ok=True)
    stdout_fd = -1
    stderr_fd = -1
    process: subprocess.Popen[bytes] | None = None
    process_group_id: int | None = None
    try:
        stdout_fd = os.open(
            stdout_target,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            0o600,
        )
        stderr_fd = os.open(
            stderr_target,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            0o600,
        )
        with os.fdopen(stdout_fd, "wb", closefd=True) as stdout_stream:
            stdout_fd = -1
            with os.fdopen(stderr_fd, "wb", closefd=True) as stderr_stream:
                stderr_fd = -1
                try:
                    process = subprocess.Popen(
                        list(argv),
                        cwd=Path(cwd),
                        env=env,
                        stdin=subprocess.DEVNULL,
                        stdout=stdout_stream,
                        stderr=stderr_stream,
                        shell=False,
                        start_new_session=True,
                    )
                    process_group_id = os.getpgid(process.pid)
                    if process_group_id == os.getpgrp():
                        raise RuntimeError("child process group isolation failed")
                except OSError:
                    return ChildResult(
                        spawned=False,
                        exit_code=None,
                        signal_number=None,
                        cancelled=False,
                        process_group_id=None,
                        failure_category="spawn_failed",
                    )
                while process.poll() is None:
                    if cancel_event is not None and cancel_event.is_set():
                        assert process_group_id is not None
                        cleanup = _terminate_process_group(process, process_group_id)
                        if not cleanup.gone:
                            raise RuntimeError("child process group cleanup is unverifiable")
                        return_code = process.returncode
                        return ChildResult(
                            spawned=True,
                            exit_code=(
                                return_code
                                if return_code is not None and return_code >= 0
                                else None
                            ),
                            signal_number=(
                                -return_code
                                if return_code is not None and return_code < 0
                                else None
                            ),
                            cancelled=True,
                            process_group_id=process_group_id,
                            failure_category="cancelled",
                            cleanup_term_sent=cleanup.term_sent,
                            cleanup_kill_sent=cleanup.kill_sent,
                            process_group_gone=cleanup.gone,
                        )
                    time.sleep(0.02)
                return_code = process.returncode
                if return_code is None:
                    raise RuntimeError("child process did not expose a terminal status")
                assert process_group_id is not None
                cleanup = _terminate_process_group(process, process_group_id)
                if not cleanup.gone:
                    raise RuntimeError("child process group cleanup is unverifiable")
                return ChildResult(
                    spawned=True,
                    exit_code=return_code if return_code >= 0 else None,
                    signal_number=-return_code if return_code < 0 else None,
                    cancelled=False,
                    process_group_id=process_group_id,
                    failure_category=None,
                    cleanup_term_sent=cleanup.term_sent,
                    cleanup_kill_sent=cleanup.kill_sent,
                    process_group_gone=cleanup.gone,
                )
    finally:
        if process is not None and process.poll() is None:
            _terminate_process_group(process, process_group_id or process.pid)
        if stdout_fd >= 0:
            os.close(stdout_fd)
        if stderr_fd >= 0:
            os.close(stderr_fd)


# 将arm launch投影到私有child process边界
def _run_arm_launch(
    launch: ArmLaunch,
    cancel_event: threading.Event | None,
) -> ChildResult:
    return run_private_child(
        launch.argv,
        cwd=launch.cwd,
        env=launch.env,
        stdout_path=launch.stdout_path,
        stderr_path=launch.stderr_path,
        cancel_event=cancel_event,
    )


# 将private child结果投影为不含process path或raw error的terminal evidence
def _child_termination(result: ChildResult) -> ChildTerminationEvidence:
    return ChildTerminationEvidence(
        spawned=result.spawned,
        exit_code=result.exit_code,
        signal_number=result.signal_number,
        cancelled=result.cancelled,
        failure_category=result.failure_category,
        cleanup_term_sent=result.cleanup_term_sent,
        cleanup_kill_sent=result.cleanup_kill_sent,
        process_group_gone=result.process_group_gone,
    )


# 在共享checkpoint上按control-first顺序运行两个已有单臂CLI
def _execute_control_first_impl(
    *,
    repository: Path,
    receipt: PairedReceipt,
    preflight: FinalPreflightArtifact,
    authorization: ExecutionAuthorizationArtifact,
    receipt_sha256: str,
    preflight_commit: str,
    preflight_sha256: str,
    authorization_commit: str,
    authorization_sha256: str,
    authorization_use_record: AuthorizationUseRecord,
    receipt_path: Path,
    private_paths: PrivateEvidencePaths,
    control: ArmLaunch,
    treatment: ArmLaunch,
    between_expected: BetweenArmEvidence,
    observe_between: Callable[[], BetweenArmEvidence],
    result_id: str,
    created_at_utc: str,
    checkpoint: _PairExecutionCheckpoint,
    child_runner: Callable[
        [ArmLaunch, threading.Event | None], ChildResult
    ] = _run_arm_launch,
    cancel_event: threading.Event | None = None,
) -> PairExecutionSummary:
    validate_execution_authorization(
        authorization,
        preflight=preflight,
        receipt=receipt,
        receipt_sha256=receipt_sha256,
        preflight_commit=preflight_commit,
        preflight_sha256=preflight_sha256,
    )
    if (
        authorization_use_record.authorization_sha256 != authorization_sha256
        or authorization_use_record.paired_receipt_sha256 != receipt_sha256
        or authorization_use_record.output_parent_sha256
        != preflight.external_parent.canonical_path_sha256
    ):
        raise ValueError("authorization-use identity mismatch")
    if os.path.lexists(control.output_root) or os.path.lexists(treatment.output_root):
        raise ValueError("paired execution root already exists")
    state = checkpoint.state
    transitions = checkpoint.transitions

    # 只允许共享reducer改变state并同步追加可持久化history
    def advance(event: PairEvent) -> None:
        nonlocal state
        transition = reduce_pair_transition(
            state,
            event,
            receipt=receipt,
            authorization_use_reserved=True,
        )
        transitions.append(transition)
        state = PairState(transition.to_state)
        checkpoint.state = state

    expected_use_sha256 = hashlib.sha256(
        (canonical_json(authorization_use_record) + "\n").encode("utf-8")
    ).hexdigest()
    authorization_use_sha256 = expected_use_sha256

    # 将任一post-reservation失败经reducer终结并写create-once terminal record
    def terminalize(
        *,
        phase: str,
        failure_category: str,
        control_audit: ArmAudit | None = None,
        treatment_audit: ArmAudit | None = None,
        control_child: ChildResult | None = None,
        treatment_child: ChildResult | None = None,
    ) -> PairExecutionSummary:
        event = terminal_event_for_failure(phase, failure_category)
        advance(event)
        terminal = build_terminal_record(
            terminal_id=f"{result_id}-terminal",
            created_at_utc=created_at_utc,
            phase=phase,
            receipt_sha256=receipt_sha256,
            preflight_sha256=preflight_sha256,
            authorization_sha256=authorization_sha256,
            authorization_use_sha256=authorization_use_sha256,
            transitions=transitions,
            failure_category=failure_category,
            control=control_audit,
            treatment=treatment_audit,
            control_child=(
                _child_termination(control_child) if control_child is not None else None
            ),
            treatment_child=(
                _child_termination(treatment_child)
                if treatment_child is not None
                else None
            ),
        )
        write_terminal_record(private_paths.terminal_record, terminal)
        validate_pair_terminal_exclusivity(
            private_paths,
            authorization_consumed=True,
            execution_complete=True,
            repository=repository,
            expected_receipt=preflight.paired_receipt,
        )
        return PairExecutionSummary(
            state=state,
            control=control_audit,
            treatment=treatment_audit,
            result=None,
            terminal=terminal,
            evidence_paths=private_paths,
        )

    # 将reservation后的父进程中断收敛为固定脱敏terminal且不吞原BaseException
    def terminalize_parent_interrupt(
        error: KeyboardInterrupt | SystemExit,
        *,
        control_audit: ArmAudit | None = None,
        treatment_audit: ArmAudit | None = None,
        control_child: ChildResult | None = None,
        treatment_child: ChildResult | None = None,
    ) -> None:
        failure_category = (
            "parent_interrupted"
            if isinstance(error, KeyboardInterrupt)
            else "parent_system_exit"
        )
        if checkpoint.authorization_consumed and state is PairState.NOT_STARTED:
            advance(PairEvent.AUTHORIZATION_RESERVED)
        terminalize(
            phase="parent_interrupt",
            failure_category=failure_category,
            control_audit=(
                checkpoint.control_audit
                if control_audit is None
                else control_audit
            ),
            treatment_audit=(
                checkpoint.treatment_audit
                if treatment_audit is None
                else treatment_audit
            ),
            control_child=(
                checkpoint.control_child
                if control_child is None
                else control_child
            ),
            treatment_child=(
                checkpoint.treatment_child
                if treatment_child is None
                else treatment_child
            ),
        )

    checkpoint.parent_interrupt_handler = terminalize_parent_interrupt
    try:
        reserve_authorization_use(
            private_paths.authorization_use,
            authorization_use_record,
        )
    except AuthorizationReservationError as exc:
        if not exc.consumed:
            raise ValueError("authorization use reservation failed") from exc
        checkpoint.authorization_consumed = True
        advance(PairEvent.AUTHORIZATION_RESERVED)
        return terminalize(
            phase="control_spawn",
            failure_category="private_evidence_unavailable",
        )
    checkpoint.authorization_consumed = True
    advance(PairEvent.AUTHORIZATION_RESERVED)

    try:
        private_paths.root.mkdir(mode=0o700, exist_ok=False)
        (private_paths.root / "home").mkdir(mode=0o700)
        (private_paths.root / "tmp").mkdir(mode=0o700)
    except (KeyboardInterrupt, SystemExit) as exc:
        terminalize_parent_interrupt(exc)
        raise
    except OSError:
        return terminalize(
            phase="control_spawn",
            failure_category="private_evidence_unavailable",
        )
    try:
        advance(PairEvent.CONTROL_STARTED)
    except (KeyboardInterrupt, SystemExit) as exc:
        terminalize_parent_interrupt(exc)
        raise
    if os.path.lexists(control.output_root):
        return terminalize(
            phase="control_spawn",
            failure_category="control_spawn_failed",
        )
    try:
        control_child = child_runner(control, cancel_event)
        checkpoint.control_child = control_child
    except (KeyboardInterrupt, SystemExit) as exc:
        terminalize_parent_interrupt(exc)
        raise
    except (OSError, RuntimeError, ValueError):
        return terminalize(
            phase="control_spawn",
            failure_category="control_spawn_failed",
        )
    if not control_child.spawned:
        return terminalize(
            phase="control_spawn",
            failure_category="control_spawn_failed",
            control_child=control_child,
        )
    try:
        control_audit = audit_arm_result(
            expected=control.expected,
            exit_code=control_child.exit_code,
            signal_number=control_child.signal_number,
            output_root=control.output_root,
            receipt=receipt,
        )
        checkpoint.control_audit = control_audit
    except (KeyboardInterrupt, SystemExit) as exc:
        terminalize_parent_interrupt(exc, control_child=control_child)
        raise
    except (OSError, RuntimeError, ValueError):
        return terminalize(
            phase="control_audit",
            failure_category="control_invalid",
            control_child=control_child,
        )
    if control_audit.status != "VALID":
        return terminalize(
            phase="control_audit",
            failure_category="control_invalid",
            control_audit=control_audit,
            control_child=control_child,
        )
    try:
        advance(PairEvent.CONTROL_VALID)
    except (KeyboardInterrupt, SystemExit) as exc:
        terminalize_parent_interrupt(
            exc,
            control_audit=control_audit,
            control_child=control_child,
        )
        raise
    try:
        revalidate_before_treatment(between_expected, observe_between())
    except (KeyboardInterrupt, SystemExit) as exc:
        terminalize_parent_interrupt(
            exc,
            control_audit=control_audit,
            control_child=control_child,
        )
        raise
    except (OSError, RuntimeError, ValueError):
        return terminalize(
            phase="between_arms",
            failure_category="between_arm_invalid",
            control_audit=control_audit,
            control_child=control_child,
        )
    try:
        advance(PairEvent.TREATMENT_STARTED)
    except (KeyboardInterrupt, SystemExit) as exc:
        terminalize_parent_interrupt(
            exc,
            control_audit=control_audit,
            control_child=control_child,
        )
        raise
    try:
        treatment_child = child_runner(treatment, cancel_event)
        checkpoint.treatment_child = treatment_child
    except (KeyboardInterrupt, SystemExit) as exc:
        terminalize_parent_interrupt(
            exc,
            control_audit=control_audit,
            control_child=control_child,
        )
        raise
    except (OSError, RuntimeError, ValueError):
        return terminalize(
            phase="treatment_spawn",
            failure_category="treatment_spawn_failed",
            control_audit=control_audit,
            control_child=control_child,
        )
    if not treatment_child.spawned:
        return terminalize(
            phase="treatment_spawn",
            failure_category="treatment_spawn_failed",
            control_audit=control_audit,
            control_child=control_child,
            treatment_child=treatment_child,
        )
    try:
        treatment_audit = audit_arm_result(
            expected=treatment.expected,
            exit_code=treatment_child.exit_code,
            signal_number=treatment_child.signal_number,
            output_root=treatment.output_root,
            receipt=receipt,
        )
        checkpoint.treatment_audit = treatment_audit
    except (KeyboardInterrupt, SystemExit) as exc:
        terminalize_parent_interrupt(
            exc,
            control_audit=control_audit,
            control_child=control_child,
            treatment_child=treatment_child,
        )
        raise
    except (OSError, RuntimeError, ValueError):
        return terminalize(
            phase="treatment_audit",
            failure_category="treatment_invalid",
            control_audit=control_audit,
            control_child=control_child,
            treatment_child=treatment_child,
        )
    if treatment_audit.status != "VALID":
        return terminalize(
            phase="treatment_audit",
            failure_category="treatment_invalid",
            control_audit=control_audit,
            treatment_audit=treatment_audit,
            control_child=control_child,
            treatment_child=treatment_child,
        )
    try:
        advance(PairEvent.TREATMENT_VALID)
    except (KeyboardInterrupt, SystemExit) as exc:
        terminalize_parent_interrupt(
            exc,
            control_audit=control_audit,
            treatment_audit=treatment_audit,
            control_child=control_child,
            treatment_child=treatment_child,
        )
        raise
    try:
        outcome = derive_pair_outcome(control_audit, treatment_audit)
        recompute_classification(receipt, outcome)
    except (KeyboardInterrupt, SystemExit) as exc:
        terminalize_parent_interrupt(
            exc,
            control_audit=control_audit,
            treatment_audit=treatment_audit,
            control_child=control_child,
            treatment_child=treatment_child,
        )
        raise
    except (RuntimeError, ValueError):
        return terminalize(
            phase="paired_classification",
            failure_category="paired_classification_failed",
            control_audit=control_audit,
            treatment_audit=treatment_audit,
            control_child=control_child,
            treatment_child=treatment_child,
        )
    try:
        final_transition = reduce_pair_transition(
            state,
            PairEvent.FINALIZED,
            receipt=receipt,
            authorization_use_reserved=True,
        )
        result = build_paired_result(
            result_id=result_id,
            created_at_utc=created_at_utc,
            receipt_reference=preflight.paired_receipt,
            preflight_commit=preflight_commit,
            preflight_sha256=preflight_sha256,
            authorization_commit=authorization_commit,
            authorization_sha256=authorization_sha256,
            authorization_use_sha256=authorization_use_sha256,
            receipt=receipt,
            control=control_audit,
            treatment=treatment_audit,
            control_child=_child_termination(control_child),
            treatment_child=_child_termination(treatment_child),
            transitions=[*transitions, final_transition],
        )
    except (KeyboardInterrupt, SystemExit) as exc:
        terminalize_parent_interrupt(
            exc,
            control_audit=control_audit,
            treatment_audit=treatment_audit,
            control_child=control_child,
            treatment_child=treatment_child,
        )
        raise
    except (RuntimeError, ValueError):
        return terminalize(
            phase="paired_classification",
            failure_category="paired_classification_failed",
            control_audit=control_audit,
            treatment_audit=treatment_audit,
            control_child=control_child,
            treatment_child=treatment_child,
        )
    try:
        publication = write_paired_result(
            private_paths.paired_result,
            result,
            repository=repository,
            expected_receipt=preflight.paired_receipt,
        )
    except (KeyboardInterrupt, SystemExit) as exc:
        try:
            committed = read_paired_result_bundle(
                private_paths.paired_result,
                repository=repository,
                expected_receipt=preflight.paired_receipt,
            )
        except ValueError:
            terminalize_parent_interrupt(
                exc,
                control_audit=control_audit,
                treatment_audit=treatment_audit,
                control_child=control_child,
                treatment_child=treatment_child,
            )
        else:
            if canonical_json(committed) != canonical_json(result):
                raise ValueError("paired result commit identity drift") from exc
        raise
    except (OSError, RuntimeError, ValueError):
        try:
            committed = read_paired_result_bundle(
                private_paths.paired_result,
                repository=repository,
                expected_receipt=preflight.paired_receipt,
            )
        except ValueError:
            committed = None
        if committed is not None and canonical_json(committed) == canonical_json(result):
            transitions.append(final_transition)
            state = PairState(final_transition.to_state)
            checkpoint.state = state
            validate_pair_terminal_exclusivity(
                private_paths,
                authorization_consumed=True,
                execution_complete=True,
                repository=repository,
                expected_receipt=preflight.paired_receipt,
            )
            return PairExecutionSummary(
                state=state,
                control=control_audit,
                treatment=treatment_audit,
                result=result,
                terminal=None,
                result_durability_warning=True,
                evidence_paths=private_paths,
            )
        return terminalize(
            phase="paired_result_write",
            failure_category="paired_result_write_failed",
            control_audit=control_audit,
            treatment_audit=treatment_audit,
            control_child=control_child,
            treatment_child=treatment_child,
        )
    transitions.append(final_transition)
    state = PairState(final_transition.to_state)
    checkpoint.state = state
    validate_pair_terminal_exclusivity(
        private_paths,
        authorization_consumed=True,
        execution_complete=True,
        repository=repository,
        expected_receipt=preflight.paired_receipt,
    )
    return PairExecutionSummary(
        state=state,
        control=control_audit,
        treatment=treatment_audit,
        result=result,
        terminal=None,
        result_durability_warning=publication.durability_warning,
        evidence_paths=private_paths,
    )


# 在一次性授权消费后用最外层guard收敛任意父进程中断
def execute_control_first(
    *,
    repository: Path,
    receipt: PairedReceipt,
    preflight: FinalPreflightArtifact,
    authorization: ExecutionAuthorizationArtifact,
    receipt_sha256: str,
    preflight_commit: str,
    preflight_sha256: str,
    authorization_commit: str,
    authorization_sha256: str,
    authorization_use_record: AuthorizationUseRecord,
    receipt_path: Path,
    private_paths: PrivateEvidencePaths,
    control: ArmLaunch,
    treatment: ArmLaunch,
    between_expected: BetweenArmEvidence,
    observe_between: Callable[[], BetweenArmEvidence],
    result_id: str,
    created_at_utc: str,
    child_runner: Callable[
        [ArmLaunch, threading.Event | None], ChildResult
    ] = _run_arm_launch,
    cancel_event: threading.Event | None = None,
) -> PairExecutionSummary:
    checkpoint = _PairExecutionCheckpoint(PairState.NOT_STARTED, [])
    try:
        return _execute_control_first_impl(
            repository=repository,
            receipt=receipt,
            preflight=preflight,
            authorization=authorization,
            receipt_sha256=receipt_sha256,
            preflight_commit=preflight_commit,
            preflight_sha256=preflight_sha256,
            authorization_commit=authorization_commit,
            authorization_sha256=authorization_sha256,
            authorization_use_record=authorization_use_record,
            receipt_path=receipt_path,
            private_paths=private_paths,
            control=control,
            treatment=treatment,
            between_expected=between_expected,
            observe_between=observe_between,
            result_id=result_id,
            created_at_utc=created_at_utc,
            checkpoint=checkpoint,
            child_runner=child_runner,
            cancel_event=cancel_event,
        )
    except (KeyboardInterrupt, SystemExit) as exc:
        result_exists = False
        terminal_exists = False
        try:
            read_paired_result_bundle(
                private_paths.paired_result,
                repository=repository,
                expected_receipt=preflight.paired_receipt,
            )
        except ValueError:
            pass
        else:
            result_exists = True
        try:
            read_strict_artifact(private_paths.terminal_record, PairTerminalRecord)
        except ValueError:
            pass
        else:
            terminal_exists = True
        if result_exists or terminal_exists:
            validate_pair_terminal_exclusivity(
                private_paths,
                authorization_consumed=True,
                execution_complete=True,
                repository=repository,
                expected_receipt=preflight.paired_receipt,
            )
            raise
        if not checkpoint.authorization_consumed:
            observed_use: AuthorizationUseRecord | None = None
            try:
                observed_use = read_strict_artifact(
                    private_paths.authorization_use,
                    AuthorizationUseRecord,
                )
            except ValueError:
                pass
            if (
                observed_use is not None
                and canonical_json(observed_use)
                == canonical_json(authorization_use_record)
            ):
                checkpoint.authorization_consumed = True
        if not checkpoint.authorization_consumed:
            raise
        handler = checkpoint.parent_interrupt_handler
        if handler is None:
            raise RuntimeError("parent interrupt terminalizer is unavailable") from exc
        handler(exc)
        raise


# 执行不携带credential的只读Git命令并返回单行输出
def _git(repository: Path, *args: str, allow_failure: bool = False) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
        env={"PATH": os.environ.get("PATH", "")},
    )
    if result.returncode != 0 and not allow_failure:
        raise ValueError("Git observer command failed")
    return result.stdout.strip()


# 验证execute时main checkout与authorization commit和remote精确一致
def _validate_execution_main_identity(
    repository: Path,
    preflight: FinalPreflightArtifact,
    authorization_commit: str,
) -> str:
    remote_commit = _git(
        repository,
        "rev-parse",
        preflight.generator.remote_ref,
    )
    if (
        _git(repository, "rev-parse", "HEAD") != authorization_commit
        or remote_commit != authorization_commit
        or _git(repository, "branch", "--show-current")
        != preflight.generator.branch
        or bool(_git(repository, "status", "--porcelain", "--untracked-files=all"))
    ):
        raise ValueError("execution main checkout identity drift")
    return remote_commit


# 对当前checkout收集clean/ahead/behind的strict Git身份
def _capture_git_snapshot(repository: Path, remote_ref: str) -> GitSnapshot:
    commit = _git(repository, "rev-parse", "HEAD")
    branch = _git(repository, "branch", "--show-current")
    remote_commit = _git(repository, "rev-parse", remote_ref)
    counts = _git(repository, "rev-list", "--left-right", "--count", f"HEAD...{remote_ref}")
    try:
        ahead_text, behind_text = counts.split()
        ahead = int(ahead_text)
        behind = int(behind_text)
    except (TypeError, ValueError) as exc:
        raise ValueError("Git ahead/behind evidence is invalid") from exc
    operation_markers = (
        "MERGE_HEAD",
        "CHERRY_PICK_HEAD",
        "REVERT_HEAD",
        "rebase-merge",
        "rebase-apply",
        "sequencer",
        "BISECT_START",
        "BISECT_LOG",
    )
    operation = False
    for marker in operation_markers:
        marker_path = Path(_git(repository, "rev-parse", "--git-path", marker))
        if not marker_path.is_absolute():
            marker_path = repository / marker_path
        if marker_path.exists() or os.path.lexists(marker_path):
            operation = True
            break
    return GitSnapshot(
        commit=commit,
        branch=branch,
        remote_ref=remote_ref,
        remote_commit=remote_commit,
        ahead=ahead,
        behind=behind,
        dirty=bool(_git(repository, "status", "--porcelain", "--untracked-files=all")),
        git_operation_in_progress=operation,
    )


# 使用指定解释器和arm-local PYTHONPATH验证真实kama_claude import来源
def _probe_arm_source_import(
    interpreter: Path,
    source_root: Path,
    *,
    timeout_s: float = 5.0,
) -> SourceImportEvidence:
    try:
        canonical_interpreter = interpreter.resolve(strict=True)
        canonical_source = source_root.resolve(strict=True)
        package_root = (canonical_source / "kama_claude").resolve(strict=True)
        if (
            not canonical_interpreter.is_file()
            or not canonical_source.is_dir()
            or not package_root.is_dir()
            or not package_root.is_relative_to(canonical_source)
            or timeout_s <= 0
        ):
            raise ValueError("source import probe is invalid")
        probe_code = (
            "import json, kama_claude; "
            "print(json.dumps({'module_file': kama_claude.__file__}, "
            "sort_keys=True, separators=(',', ':')))"
        )
        result = subprocess.run(
            [str(canonical_interpreter), "-c", probe_code],
            cwd=canonical_source,
            env={
                "PYTHONPATH": str(canonical_source),
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONNOUSERSITE": "1",
            },
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        if result.returncode != 0 or result.stderr:
            raise ValueError("source import probe is invalid")
        payload = parse_strict_json_object(result.stdout)
        if (
            set(payload) != {"module_file"}
            or not isinstance(payload["module_file"], str)
        ):
            raise ValueError("source import probe is invalid")
        canonical_json(payload)
        module_file = Path(payload["module_file"]).resolve(strict=True)
        if (
            not module_file.is_file()
            or not module_file.is_relative_to(package_root)
            or not module_file.is_relative_to(canonical_source)
        ):
            raise ValueError("source import probe is invalid")
        return SourceImportEvidence(
            source_root_sha256=hashlib.sha256(
                str(canonical_source).encode("utf-8")
            ).hexdigest(),
            imported_module_path_sha256=hashlib.sha256(
                str(module_file).encode("utf-8")
            ).hexdigest(),
            imported_module_file_sha256=hashlib.sha256(
                module_file.read_bytes()
            ).hexdigest(),
            module_within_source_root=True,
            absolute_path_persisted=False,
        )
    except (
        OSError,
        RuntimeError,
        ValueError,
        json.JSONDecodeError,
        subprocess.TimeoutExpired,
    ) as exc:
        raise ValueError("source import probe failed") from exc


# 收集detached worktree状态而不修改其Git或文件内容
def _observe_worktree(
    *,
    label: str,
    worktree: Path,
    repository: Path,
    output_parent: Path,
    expected_commit: str,
    profile_path: str,
    interpreter: Path,
) -> WorktreeObservation:
    canonical = worktree.resolve(strict=True)
    source_import = _probe_arm_source_import(interpreter, canonical / "src")
    registered_rows = _git(repository, "worktree", "list", "--porcelain").splitlines()
    registered = f"worktree {canonical}" in registered_rows
    branch = _git(canonical, "symbolic-ref", "--quiet", "--short", "HEAD", allow_failure=True)
    observation = WorktreeObservation(
        label=label,
        path=canonical,
        canonical_path_sha256=hashlib.sha256(
            str(canonical).encode("utf-8")
        ).hexdigest(),
        source_root=canonical / "src",
        registered=registered,
        detached=not branch,
        clean=not bool(_git(canonical, "status", "--porcelain", "--untracked-files=all")),
        observed_head=_git(canonical, "rev-parse", "HEAD"),
        profile_exists=(canonical / profile_path).is_file(),
        source_import=source_import,
    )
    validate_worktree_binding(
        observation,
        expected_commit=expected_commit,
        repository=repository,
        output_parent=output_parent,
    )
    return observation


# 对指定arm生成与receipt绑定的declared identity与worktree evidence
def _capture_arm_preflight(
    *,
    arm: str,
    receipt: PairedReceipt,
    worktree: Path,
    repository: Path,
    output_parent: Path,
    sdk_version: str,
    interpreter: Path,
) -> tuple[ArmPreflightEvidence, DeclaredExperimentIdentity]:
    receipt_arm = getattr(receipt.arms, arm)
    observation = _observe_worktree(
        label=f"{receipt_arm.label}_WORKTREE",
        worktree=worktree,
        repository=repository,
        output_parent=output_parent,
        expected_commit=receipt_arm.commit,
        profile_path=receipt_arm.profile_path,
        interpreter=interpreter,
    )
    loaded = load_experiment_profile(worktree / receipt_arm.profile_path)
    declared = capture_declared_identity(
        loaded,
        repository_root=worktree,
        repository=RepositoryIdentity(commit=receipt_arm.commit, dirty=False),
        installed_sdk_version=sdk_version,
    )
    if (
        declared.profile_id != receipt_arm.profile_id
        or declared.profile_hash != receipt_arm.profile_canonical_sha256
        or declared.prompt_hash != receipt_arm.prompt_sha256
    ):
        raise ValueError("arm profile identity mismatch")
    path_hash = hashlib.sha256(str(worktree.resolve()).encode("utf-8")).hexdigest()
    evidence = ArmPreflightEvidence(
        arm=arm,
        commit=receipt_arm.commit,
        profile_path=receipt_arm.profile_path,
        profile_id=receipt_arm.profile_id,
        profile_file_sha256=hashlib.sha256(
            (worktree / receipt_arm.profile_path).read_bytes()
        ).hexdigest(),
        profile_canonical_sha256=declared.profile_hash,
        prompt_sha256=declared.prompt_hash,
        worktree=WorktreeEvidence(
            label=observation.label,
            canonical_path_sha256=path_hash,
            absolute_path_persisted=False,
            registered=True,
            detached=True,
            clean=True,
            head_matches=True,
            profile_exists=True,
            source_import=observation.source_import,
            outside_repository=True,
            outside_output_parent=True,
        ),
    )
    return evidence, declared


# 从当前解释器收集仅name/version的installed distribution pairs
def _installed_distributions() -> list[tuple[str, str]]:
    return [
        (distribution.metadata["Name"], distribution.version)
        for distribution in importlib.metadata.distributions()
        if distribution.metadata["Name"]
    ]


# 读取当前uv版本而不写入任何环境或cache
def _uv_version() -> str:
    result = subprocess.run(
        ["uv", "--version"],
        check=False,
        capture_output=True,
        text=True,
        env={"PATH": os.environ.get("PATH", "")},
    )
    if result.returncode != 0 or not result.stdout.strip():
        raise ValueError("uv version cannot be observed")
    return result.stdout.strip().removeprefix("uv ")


# 以显式路径和env name完全离线生成final-preflight artifact
def generate_final_preflight(
    *,
    receipt_path: Path,
    artifact_path: Path,
    output_parent_env: str,
    control_worktree: Path,
    treatment_worktree: Path,
    interpreter: Path,
    repository: Path,
    remote_ref: str,
) -> FinalPreflightArtifact:
    receipt_path = receipt_path.resolve(strict=True)
    artifact_path = artifact_path.absolute()
    receipt = load_paired_receipt(receipt_path)
    repository = repository.resolve(strict=True)
    if control_worktree.resolve(strict=True) == treatment_worktree.resolve(strict=True):
        raise ValueError("paired worktrees must be distinct")
    output_value = os.environ.get(output_parent_env)
    if output_value is None:
        raise ValueError("output parent environment is missing")
    output_parent = Path(output_value)
    if interpreter.resolve(strict=True) != Path(sys.executable).resolve(strict=True):
        raise ValueError("preflight interpreter must be the active interpreter")
    sdk_distribution, sdk_version = receipt.shared_identity.sdk.split("==", maxsplit=1)
    git_common = Path(_git(repository, "rev-parse", "--git-common-dir"))
    if not git_common.is_absolute():
        git_common = repository / git_common
    bound_parent = bind_output_parent(
        output_parent,
        repository=repository,
        git_common_dir=git_common,
        worktrees=[control_worktree, treatment_worktree],
        control_basename=receipt.execution_plan.control_output_logical_root,
        treatment_basename=receipt.execution_plan.treatment_output_logical_root,
    )
    control_evidence, control_declared = _capture_arm_preflight(
        arm="control",
        receipt=receipt,
        worktree=control_worktree,
        repository=repository,
        output_parent=bound_parent.path,
        sdk_version=sdk_version,
        interpreter=interpreter,
    )
    treatment_evidence, treatment_declared = _capture_arm_preflight(
        arm="treatment",
        receipt=receipt,
        worktree=treatment_worktree,
        repository=repository,
        output_parent=bound_parent.path,
        sdk_version=sdk_version,
        interpreter=interpreter,
    )
    if (
        control_declared.suite != treatment_declared.suite
        or control_declared.provider != treatment_declared.provider
        or control_declared.runtime != treatment_declared.runtime
        or control_declared.dependency != treatment_declared.dependency
        or control_declared.tool_schema_hash != treatment_declared.tool_schema_hash
        or control_declared.runtime_config_hash
        != treatment_declared.runtime_config_hash
    ):
        raise ValueError("paired arm shared identity mismatch")
    environment = capture_environment_snapshot(
        interpreter=interpreter,
        distributions=_installed_distributions(),
        python_version=platform.python_version(),
        system=platform.system(),
        release=platform.release(),
        machine=platform.machine(),
        sdk_distribution=sdk_distribution,
        sdk_version=importlib.metadata.version(sdk_distribution),
        uv_version=_uv_version(),
        pyproject=repository / "pyproject.toml",
        uv_lock=repository / "uv.lock",
    )
    control_command = build_command_spec(
        arm="control",
        interpreter_label="phase9d-python",
        interpreter_sha256=environment.interpreter_file_sha256,
        worktree_label=control_evidence.worktree.label,
        worktree_sha256=control_evidence.worktree.canonical_path_sha256,
        profile_path=control_evidence.profile_path,
        output_basename=receipt.execution_plan.control_output_logical_root,
        expected_attempts=receipt.execution_plan.attempts_per_arm,
    )
    treatment_command = build_command_spec(
        arm="treatment",
        interpreter_label="phase9d-python",
        interpreter_sha256=environment.interpreter_file_sha256,
        worktree_label=treatment_evidence.worktree.label,
        worktree_sha256=treatment_evidence.worktree.canonical_path_sha256,
        profile_path=treatment_evidence.profile_path,
        output_basename=receipt.execution_plan.treatment_output_logical_root,
        expected_attempts=receipt.execution_plan.attempts_per_arm,
    )
    receipt_commit, receipt_sha256 = _tracked_artifact_identity(
        repository,
        receipt_path,
    )
    receipt_reference = observe_receipt_reference(
        repository,
        receipt_commit,
        receipt_path,
    )
    if receipt_reference.sha256 != receipt_sha256:
        raise ValueError("receipt commit drift")
    artifact = build_final_preflight_artifact(
        preflight_id=f"{receipt.receipt_id}-final-preflight",
        created_at_utc=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        generator=_capture_git_snapshot(repository, remote_ref),
        paired_receipt=receipt_reference,
        arms=PreflightArms(
            control=control_evidence,
            treatment=treatment_evidence,
        ),
        environment=environment,
        shared_identity=SharedPreflightIdentity(
            provider=receipt.shared_identity.provider,
            model=receipt.shared_identity.model,
            protocol=receipt.shared_identity.protocol,
            suite_sha256=control_declared.suite.suite_hash,
            tool_schema_sha256=control_declared.tool_schema_hash,
            runtime_config_sha256=control_declared.runtime_config_hash,
            dependency_sha256=control_declared.dependency.dependency_hash,
            max_steps=control_declared.runtime.max_steps,
            repeats=control_declared.schedule.repeats,
            mcp_enabled=False,
        ),
        external_parent=bound_parent.evidence,
        logical_roots=LogicalRootsEvidence(
            control=receipt.execution_plan.control_output_logical_root,
            treatment=receipt.execution_plan.treatment_output_logical_root,
            control_lexists=False,
            treatment_lexists=False,
            roots_created=0,
        ),
        commands=CommandHashes(
            control_spec_sha256=control_command.spec_sha256,
            treatment_spec_sha256=treatment_command.spec_sha256,
            raw_absolute_paths_persisted=False,
            credential_value_persisted=False,
        ),
        credential=check_credential_presence(os.environ, "ANTHROPIC_API_KEY"),
        receipt=receipt,
    )
    write_canonical_artifact(artifact_path, artifact)
    return read_strict_artifact(artifact_path, FinalPreflightArtifact)


# 确认tracked artifact的引入commit与file SHA-256而不改变Git状态
def _tracked_artifact_identity(
    repository: Path,
    path: Path,
) -> tuple[str, str]:
    try:
        relative = path.resolve(strict=True).relative_to(repository.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise ValueError("tracked artifact path is invalid") from exc
    _git(repository, "ls-files", "--error-unmatch", str(relative))
    commit = _git(repository, "log", "-1", "--format=%H", "--", str(relative))
    result = subprocess.run(
        ["git", "show", f"HEAD:{relative.as_posix()}"],
        cwd=repository,
        check=False,
        capture_output=True,
        env={"PATH": os.environ.get("PATH", "")},
    )
    try:
        current = path.read_bytes()
    except OSError as exc:
        raise ValueError("tracked artifact bytes cannot be read") from exc
    if result.returncode != 0 or result.stdout != current:
        raise ValueError("tracked artifact differs from HEAD blob")
    return commit, hashlib.sha256(current).hexdigest()


# 从指定commit blob和当前worktree重新捕获单个tracked artifact身份
def _capture_fresh_tracked_artifact(
    *,
    repository: Path,
    approved_head: str,
    path: Path,
    expected_commit: str,
    expected_sha256: str,
) -> FreshTrackedArtifactEvidence:
    if path.is_symlink():
        raise ValueError("between-arm Git or artifact identity drift")
    canonical_repository = repository.resolve(strict=True)
    try:
        canonical_path = path.resolve(strict=True)
        relative = canonical_path.relative_to(canonical_repository)
    except (OSError, ValueError) as exc:
        raise ValueError("between-arm Git or artifact identity drift") from exc
    if not canonical_path.is_file():
        raise ValueError("between-arm Git or artifact identity drift")
    _git(repository, "cat-file", "-e", f"{expected_commit}^{{commit}}")
    blob = subprocess.run(
        ["git", "show", f"{expected_commit}:{relative.as_posix()}"],
        cwd=repository,
        check=False,
        capture_output=True,
        env={"PATH": os.environ.get("PATH", "")},
    )
    before_stat = canonical_path.stat()
    current = canonical_path.read_bytes()
    current_stat = canonical_path.stat()
    actual_sha256 = hashlib.sha256(blob.stdout).hexdigest()
    introducing_commit = _git(
        repository,
        "log",
        "-1",
        "--format=%H",
        approved_head,
        "--",
        relative.as_posix(),
    )
    if (
        blob.returncode != 0
        or not blob.stdout
        or current != blob.stdout
        or (before_stat.st_dev, before_stat.st_ino)
        != (current_stat.st_dev, current_stat.st_ino)
        or actual_sha256 != expected_sha256
        or introducing_commit != expected_commit
    ):
        raise ValueError("between-arm Git or artifact identity drift")
    return FreshTrackedArtifactEvidence(
        commit=introducing_commit,
        bytes=len(blob.stdout),
        sha256=actual_sha256,
        canonical_object_sha256=canonical_sha256(
            {"st_dev": current_stat.st_dev, "st_ino": current_stat.st_ino}
        ),
        current_matches_blob=True,
        symlink=False,
    )


# 重新读取main Git、三个tracked artifacts与C1到C2拓扑并返回无路径证据
def capture_between_arm_git_and_artifact_state(
    *,
    repository: Path,
    remote_ref: str,
    approved_head: str,
    approved_branch: str,
    receipt_path: Path,
    receipt_commit: str,
    receipt_sha256: str,
    preflight_path: Path,
    preflight_commit: str,
    preflight_sha256: str,
    authorization_path: Path,
    authorization_commit: str,
    authorization_sha256: str,
    control_commit: str,
    treatment_commit: str,
) -> FreshBetweenArmGitArtifacts:
    try:
        snapshot = _capture_git_snapshot(repository, remote_ref)
        if (
            snapshot.commit != approved_head
            or snapshot.branch != approved_branch
            or snapshot.remote_commit != approved_head
        ):
            raise ValueError("between-arm Git or artifact identity drift")
        receipt = _capture_fresh_tracked_artifact(
            repository=repository,
            approved_head=approved_head,
            path=receipt_path,
            expected_commit=receipt_commit,
            expected_sha256=receipt_sha256,
        )
        preflight = _capture_fresh_tracked_artifact(
            repository=repository,
            approved_head=approved_head,
            path=preflight_path,
            expected_commit=preflight_commit,
            expected_sha256=preflight_sha256,
        )
        authorization = _capture_fresh_tracked_artifact(
            repository=repository,
            approved_head=approved_head,
            path=authorization_path,
            expected_commit=authorization_commit,
            expected_sha256=authorization_sha256,
        )
        if (
            not _commit_exists(repository, control_commit)
            or not _commit_exists(repository, treatment_commit)
            or _git(repository, "rev-parse", f"{treatment_commit}^")
            != control_commit
        ):
            raise ValueError("between-arm Git or artifact identity drift")
        return FreshBetweenArmGitArtifacts(
            git=snapshot,
            receipt=receipt,
            preflight=preflight,
            authorization=authorization,
            control_commit_exists=True,
            treatment_commit_exists=True,
            treatment_parent_matches_control=True,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError("between-arm Git or artifact identity drift") from exc


# 将已捕获declared identity投影为arm audit的预期身份
def _expected_arm_identity(
    *,
    arm: str,
    receipt: PairedReceipt,
    declared: DeclaredExperimentIdentity,
) -> ExpectedArmIdentity:
    receipt_arm = getattr(receipt.arms, arm)
    return ExpectedArmIdentity(
        arm=arm,
        commit=receipt_arm.commit,
        profile_id=receipt_arm.profile_id,
        profile_hash=receipt_arm.profile_canonical_sha256,
        prompt_sha256=receipt_arm.prompt_sha256,
        suite_sha256=receipt.shared_identity.suite_sha256,
        tool_schema_sha256=receipt.shared_identity.tool_schema_sha256,
        runtime_config_sha256=receipt.shared_identity.runtime_config_sha256,
        dependency_sha256=receipt.shared_identity.dependency_sha256,
        provider=receipt.shared_identity.provider,
        model=receipt.shared_identity.model,
        protocol=receipt.shared_identity.protocol,
        sdk=receipt.shared_identity.sdk,
        max_steps=receipt.shared_identity.max_steps,
        repeats=receipt.shared_identity.repeats,
        task_ids=list(declared.suite.task_hashes),
    )


# 对指定commit执行只读object-existence probe
def _commit_exists(repository: Path, commit: str) -> bool:
    result = subprocess.run(
        ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
        cwd=repository,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env={"PATH": os.environ.get("PATH", "")},
    )
    return result.returncode == 0


# 收集control完成后、treatment spawn前的全量漂移证据
def _capture_between_arm_evidence(
    *,
    repository: Path,
    remote_ref: str,
    receipt: PairedReceipt,
    receipt_commit: str,
    receipt_sha256: str,
    preflight_commit: str,
    preflight_sha256: str,
    authorization_commit: str,
    authorization_sha256: str,
    receipt_path: Path,
    preflight_path: Path,
    authorization_path: Path,
    authorization_use_path: Path,
    main_ref_commit: str,
    treatment_worktree: Path,
    treatment_declared: DeclaredExperimentIdentity,
    preflight: FinalPreflightArtifact,
    output_parent: Path,
    treatment_root: Path,
) -> BetweenArmEvidence:
    stat = output_parent.stat()
    current_environment = capture_environment_snapshot(
        interpreter=sys.executable,
        distributions=_installed_distributions(),
        python_version=platform.python_version(),
        system=platform.system(),
        release=platform.release(),
        machine=platform.machine(),
        sdk_distribution=preflight.environment.sdk_distribution,
        sdk_version=importlib.metadata.version(preflight.environment.sdk_distribution),
        uv_version=_uv_version(),
        pyproject=repository / "pyproject.toml",
        uv_lock=repository / "uv.lock",
    )
    treatment_observation = _observe_worktree(
        label=preflight.arms.treatment.worktree.label,
        worktree=treatment_worktree,
        repository=repository,
        output_parent=output_parent,
        expected_commit=receipt.arms.treatment.commit,
        profile_path=receipt.arms.treatment.profile_path,
        interpreter=Path(sys.executable),
    )
    main_unchanged = (
        _git(repository, "rev-parse", remote_ref) == main_ref_commit
        and not bool(
            _git(repository, "status", "--porcelain", "--untracked-files=all")
        )
    )
    fresh_git_artifacts = capture_between_arm_git_and_artifact_state(
        repository=repository,
        remote_ref=remote_ref,
        approved_head=authorization_commit,
        approved_branch=preflight.generator.branch,
        receipt_path=receipt_path,
        receipt_commit=receipt_commit,
        receipt_sha256=receipt_sha256,
        preflight_path=preflight_path,
        preflight_commit=preflight_commit,
        preflight_sha256=preflight_sha256,
        authorization_path=authorization_path,
        authorization_commit=authorization_commit,
        authorization_sha256=authorization_sha256,
        control_commit=receipt.arms.control.commit,
        treatment_commit=receipt.arms.treatment.commit,
    )
    return BetweenArmEvidence(
        receipt_commit=receipt_commit,
        receipt_sha256=receipt_sha256,
        preflight_commit=preflight_commit,
        preflight_sha256=preflight_sha256,
        authorization_commit=authorization_commit,
        authorization_sha256=authorization_sha256,
        git_artifact_identity_sha256=canonical_sha256(fresh_git_artifacts),
        control_commit_exists=_commit_exists(repository, receipt.arms.control.commit),
        treatment_commit_exists=_commit_exists(repository, receipt.arms.treatment.commit),
        main_ref_sha256=canonical_sha256(main_ref_commit),
        treatment_worktree_sha256=hashlib.sha256(
            str(treatment_worktree.resolve(strict=True)).encode("utf-8")
        ).hexdigest(),
        treatment_profile_sha256=hashlib.sha256(
            (treatment_worktree / receipt.arms.treatment.profile_path).read_bytes()
        ).hexdigest(),
        treatment_prompt_sha256=treatment_declared.prompt_hash,
        source_binding_sha256=hashlib.sha256(
            canonical_json(treatment_observation.source_import).encode("utf-8")
        ).hexdigest(),
        treatment_source_import=treatment_observation.source_import,
        environment_sha256=canonical_sha256(current_environment),
        pyproject_sha256=current_environment.pyproject_sha256,
        uv_lock_sha256=current_environment.uv_lock_sha256,
        dependency_sha256=current_environment.dependency_sha256,
        suite_sha256=treatment_declared.suite.suite_hash,
        task_bundle_sha256=canonical_sha256(treatment_declared.suite.task_hashes),
        grader_bundle_sha256=canonical_sha256(treatment_declared.suite.grader_hashes),
        tool_schema_sha256=treatment_declared.tool_schema_hash,
        runtime_config_sha256=treatment_declared.runtime_config_hash,
        output_parent_path_sha256=hashlib.sha256(
            str(output_parent.resolve(strict=True)).encode("utf-8")
        ).hexdigest(),
        output_parent_object_sha256=canonical_sha256(
            {"st_dev": stat.st_dev, "st_ino": stat.st_ino}
        ),
        treatment_root_absent=not os.path.lexists(treatment_root),
        credential_present=check_credential_presence(
            os.environ,
            preflight.credential.env_name,
        ).present,
        authorization_use_sha256=hashlib.sha256(
            authorization_use_path.read_bytes()
        ).hexdigest(),
        experiment_unchanged=(
            current_environment == preflight.environment
            and main_unchanged
            and treatment_observation.registered
            and treatment_observation.detached
            and treatment_observation.clean
            and treatment_observation.source_import.module_within_source_root
        ),
    )


# 从已提交preflight/authorization物化并执行一次control-first pair
def execute_from_artifacts(args: argparse.Namespace) -> PairExecutionSummary:
    repository = Path(args.repository).resolve(strict=True)
    receipt_path = Path(args.receipt).resolve(strict=True)
    preflight_path = Path(args.preflight).resolve(strict=True)
    authorization_path = Path(args.authorization).resolve(strict=True)
    receipt = load_paired_receipt(receipt_path)
    preflight = read_strict_artifact(preflight_path, FinalPreflightArtifact)
    authorization = read_strict_artifact(
        authorization_path,
        ExecutionAuthorizationArtifact,
    )
    receipt_commit, receipt_sha256 = _tracked_artifact_identity(
        repository,
        receipt_path,
    )
    preflight_commit, preflight_sha256 = _tracked_artifact_identity(
        repository,
        preflight_path,
    )
    authorization_commit, authorization_sha256 = _tracked_artifact_identity(
        repository,
        authorization_path,
    )
    if (
        receipt_commit != preflight.paired_receipt.commit
        or receipt_sha256 != preflight.paired_receipt.sha256
        or receipt_path.stat().st_size != preflight.paired_receipt.bytes
        or str(receipt_path.relative_to(repository)) != preflight.paired_receipt.path
    ):
        raise ValueError("receipt commit drift")
    validate_execution_authorization(
        authorization,
        preflight=preflight,
        receipt=receipt,
        receipt_sha256=receipt_sha256,
        preflight_commit=preflight_commit,
        preflight_sha256=preflight_sha256,
    )
    output_value = os.environ.get(preflight.external_parent.env_name)
    if output_value is None:
        raise ValueError("output parent environment is missing")
    output_parent = Path(output_value)
    control_worktree = Path(args.control_worktree).resolve(strict=True)
    treatment_worktree = Path(args.treatment_worktree).resolve(strict=True)
    if control_worktree == treatment_worktree:
        raise ValueError("paired worktrees must be distinct")
    git_common = Path(_git(repository, "rev-parse", "--git-common-dir"))
    if not git_common.is_absolute():
        git_common = repository / git_common
    bound = rebind_output_parent(
        preflight.external_parent,
        output_parent,
        repository=repository,
        git_common_dir=git_common,
        worktrees=[control_worktree, treatment_worktree],
        control_basename=receipt.execution_plan.control_output_logical_root,
        treatment_basename=receipt.execution_plan.treatment_output_logical_root,
    )
    current_environment = capture_environment_snapshot(
        interpreter=sys.executable,
        distributions=_installed_distributions(),
        python_version=platform.python_version(),
        system=platform.system(),
        release=platform.release(),
        machine=platform.machine(),
        sdk_distribution=preflight.environment.sdk_distribution,
        sdk_version=importlib.metadata.version(preflight.environment.sdk_distribution),
        uv_version=_uv_version(),
        pyproject=repository / "pyproject.toml",
        uv_lock=repository / "uv.lock",
    )
    if current_environment != preflight.environment:
        raise ValueError("execution environment drift")
    sdk_version = importlib.metadata.version(preflight.environment.sdk_distribution)
    control_evidence, control_declared = _capture_arm_preflight(
        arm="control",
        receipt=receipt,
        worktree=control_worktree,
        repository=repository,
        output_parent=bound.path,
        sdk_version=sdk_version,
        interpreter=Path(sys.executable),
    )
    treatment_evidence, treatment_declared = _capture_arm_preflight(
        arm="treatment",
        receipt=receipt,
        worktree=treatment_worktree,
        repository=repository,
        output_parent=bound.path,
        sdk_version=sdk_version,
        interpreter=Path(sys.executable),
    )
    if (
        control_evidence != preflight.arms.control
        or treatment_evidence != preflight.arms.treatment
    ):
        raise ValueError("execution worktree identity drift")
    control_spec = build_command_spec(
        arm="control",
        interpreter_label="phase9d-python",
        interpreter_sha256=current_environment.interpreter_file_sha256,
        worktree_label=control_evidence.worktree.label,
        worktree_sha256=control_evidence.worktree.canonical_path_sha256,
        profile_path=control_evidence.profile_path,
        output_basename=receipt.execution_plan.control_output_logical_root,
        expected_attempts=receipt.execution_plan.attempts_per_arm,
    )
    treatment_spec = build_command_spec(
        arm="treatment",
        interpreter_label="phase9d-python",
        interpreter_sha256=current_environment.interpreter_file_sha256,
        worktree_label=treatment_evidence.worktree.label,
        worktree_sha256=treatment_evidence.worktree.canonical_path_sha256,
        profile_path=treatment_evidence.profile_path,
        output_basename=receipt.execution_plan.treatment_output_logical_root,
        expected_attempts=receipt.execution_plan.attempts_per_arm,
    )
    if (
        control_spec.spec_sha256 != preflight.commands.control_spec_sha256
        or treatment_spec.spec_sha256 != preflight.commands.treatment_spec_sha256
    ):
        raise ValueError("execution command identity drift")
    check_credential_presence(os.environ, preflight.credential.env_name)
    execution_main_ref = _validate_execution_main_identity(
        repository,
        preflight,
        authorization_commit,
    )
    initial_git_artifacts = capture_between_arm_git_and_artifact_state(
        repository=repository,
        remote_ref=preflight.generator.remote_ref,
        approved_head=authorization_commit,
        approved_branch=preflight.generator.branch,
        receipt_path=receipt_path,
        receipt_commit=receipt_commit,
        receipt_sha256=receipt_sha256,
        preflight_path=preflight_path,
        preflight_commit=preflight_commit,
        preflight_sha256=preflight_sha256,
        authorization_path=authorization_path,
        authorization_commit=authorization_commit,
        authorization_sha256=authorization_sha256,
        control_commit=receipt.arms.control.commit,
        treatment_commit=receipt.arms.treatment.commit,
    )
    private_paths = derive_private_evidence_paths(bound.path, receipt.receipt_id)
    created_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    use_record = AuthorizationUseRecord(
        schema_version=1,
        reservation_id=f"{authorization.authorization_id}-use",
        status="RESERVED_FOR_ONE_PAIRED_EXECUTION",
        created_at_utc=created_at,
        authorization_sha256=authorization_sha256,
        paired_receipt_sha256=receipt_sha256,
        output_parent_sha256=preflight.external_parent.canonical_path_sha256,
        absolute_path_persisted=False,
        credential_value_persisted=False,
    )
    source_environment = os.environ

    control_launch = _materialize_arm_launch(
        arm="control",
        receipt=receipt,
        declared=control_declared,
        worktree=control_worktree,
        output_root=bound.control_root,
        private_paths=private_paths,
        interpreter=sys.executable,
        source_environment=source_environment,
        credential_env=preflight.credential.env_name,
    )
    treatment_launch = _materialize_arm_launch(
        arm="treatment",
        receipt=receipt,
        declared=treatment_declared,
        worktree=treatment_worktree,
        output_root=bound.treatment_root,
        private_paths=private_paths,
        interpreter=sys.executable,
        source_environment=source_environment,
        credential_env=preflight.credential.env_name,
    )
    expected_between = BetweenArmEvidence(
        receipt_commit=receipt_commit,
        receipt_sha256=receipt_sha256,
        preflight_commit=preflight_commit,
        preflight_sha256=preflight_sha256,
        authorization_commit=authorization_commit,
        authorization_sha256=authorization_sha256,
        git_artifact_identity_sha256=canonical_sha256(initial_git_artifacts),
        control_commit_exists=True,
        treatment_commit_exists=True,
        main_ref_sha256=canonical_sha256(execution_main_ref),
        treatment_worktree_sha256=preflight.arms.treatment.worktree.canonical_path_sha256,
        treatment_profile_sha256=preflight.arms.treatment.profile_file_sha256,
        treatment_prompt_sha256=preflight.arms.treatment.prompt_sha256,
        source_binding_sha256=hashlib.sha256(
            canonical_json(preflight.arms.treatment.worktree.source_import).encode(
                "utf-8"
            )
        ).hexdigest(),
        treatment_source_import=preflight.arms.treatment.worktree.source_import,
        environment_sha256=canonical_sha256(preflight.environment),
        pyproject_sha256=preflight.environment.pyproject_sha256,
        uv_lock_sha256=preflight.environment.uv_lock_sha256,
        dependency_sha256=preflight.environment.dependency_sha256,
        suite_sha256=preflight.shared_identity.suite_sha256,
        task_bundle_sha256=canonical_sha256(treatment_declared.suite.task_hashes),
        grader_bundle_sha256=canonical_sha256(treatment_declared.suite.grader_hashes),
        tool_schema_sha256=preflight.shared_identity.tool_schema_sha256,
        runtime_config_sha256=preflight.shared_identity.runtime_config_sha256,
        output_parent_path_sha256=preflight.external_parent.canonical_path_sha256,
        output_parent_object_sha256=preflight.external_parent.canonical_object_sha256,
        treatment_root_absent=True,
        credential_present=True,
        authorization_use_sha256=hashlib.sha256(
            (canonical_json(use_record) + "\n").encode("utf-8")
        ).hexdigest(),
        experiment_unchanged=True,
    )

    # 在C1完成后重新收集全部between-arm evidence
    def observe_between() -> BetweenArmEvidence:
        return _capture_between_arm_evidence(
            repository=repository,
            remote_ref=preflight.generator.remote_ref,
            receipt=receipt,
            receipt_commit=receipt_commit,
            receipt_sha256=receipt_sha256,
            preflight_commit=preflight_commit,
            preflight_sha256=preflight_sha256,
            authorization_commit=authorization_commit,
            authorization_sha256=authorization_sha256,
            receipt_path=receipt_path,
            preflight_path=preflight_path,
            authorization_path=authorization_path,
            authorization_use_path=private_paths.authorization_use,
            main_ref_commit=execution_main_ref,
            treatment_worktree=treatment_worktree,
            treatment_declared=treatment_declared,
            preflight=preflight,
            output_parent=bound.path,
            treatment_root=bound.treatment_root,
        )

    return execute_control_first(
        repository=repository,
        receipt=receipt,
        preflight=preflight,
        authorization=authorization,
        receipt_sha256=receipt_sha256,
        preflight_commit=preflight_commit,
        preflight_sha256=preflight_sha256,
        authorization_commit=authorization_commit,
        authorization_sha256=authorization_sha256,
        authorization_use_record=use_record,
        receipt_path=receipt_path,
        private_paths=private_paths,
        control=control_launch,
        treatment=treatment_launch,
        between_expected=expected_between,
        observe_between=observe_between,
        result_id=f"{receipt.receipt_id}-result",
        created_at_utc=created_at,
    )


# 构建仅包含offline observer边界的Phase 9D参数解析器
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="phase9d-paired")
    subcommands = parser.add_subparsers(dest="command", required=True)
    preflight = subcommands.add_parser("preflight")
    preflight.add_argument("--receipt", required=True)
    preflight.add_argument("--artifact", required=True)
    preflight.add_argument("--output-parent-env", required=True)
    preflight.add_argument("--control-worktree", required=True)
    preflight.add_argument("--treatment-worktree", required=True)
    preflight.add_argument("--interpreter", required=True)
    preflight.add_argument("--repository", required=True)
    preflight.add_argument("--remote-ref", required=True)
    execute = subcommands.add_parser("execute")
    execute.add_argument("--receipt", required=True)
    execute.add_argument("--preflight", required=True)
    execute.add_argument("--authorization", required=True)
    execute.add_argument("--repository", required=True)
    execute.add_argument("--control-worktree", required=True)
    execute.add_argument("--treatment-worktree", required=True)
    inspect = subcommands.add_parser("inspect")
    inspect.add_argument("--artifact", required=True)
    inspect.add_argument("--receipt")
    inspect.add_argument("--repository")
    inspect.add_argument("--receipt-commit")
    inspect.add_argument(
        "--kind",
        choices=("preflight", "authorization", "result", "terminal"),
        default="preflight",
    )
    return parser


# 解析参数且不暴露model/provider/max_steps/repeat override
def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return _build_parser().parse_args(argv)


# 离线严格读取指定artifact用于inspect子命令
def _inspect(
    path: str,
    kind: str,
    receipt_path: str | None,
    repository: str | None,
    receipt_commit: str | None,
) -> None:
    models = {
        "preflight": FinalPreflightArtifact,
        "authorization": ExecutionAuthorizationArtifact,
        "result": PairedResult,
        "terminal": PairTerminalRecord,
    }
    if kind == "result":
        if receipt_path is None or repository is None or receipt_commit is None:
            raise ValueError("paired result receipt is required")
        expected_receipt = observe_receipt_reference(
            Path(repository),
            receipt_commit,
            Path(receipt_path),
        )
        read_paired_result_bundle(
            path,
            repository=Path(repository),
            expected_receipt=expected_receipt,
        )
    else:
        read_strict_artifact(path, models[kind])


# 执行离线CLI边界，未授权的preflight/execute不自动产生正式证据
def _main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.command == "inspect":
        _inspect(
            args.artifact,
            args.kind,
            args.receipt,
            args.repository,
            args.receipt_commit,
        )
        return 0
    if args.command == "preflight":
        generate_final_preflight(
            receipt_path=Path(args.receipt),
            artifact_path=Path(args.artifact),
            output_parent_env=args.output_parent_env,
            control_worktree=Path(args.control_worktree),
            treatment_worktree=Path(args.treatment_worktree),
            interpreter=Path(args.interpreter),
            repository=Path(args.repository),
            remote_ref=args.remote_ref,
        )
        return 0
    summary = execute_from_artifacts(args)
    if summary.evidence_paths is not None:
        observed_terminal = validate_pair_terminal_exclusivity(
            summary.evidence_paths,
            authorization_consumed=True,
            execution_complete=True,
            repository=Path(args.repository),
            expected_receipt=read_strict_artifact(
                Path(args.preflight),
                FinalPreflightArtifact,
            ).paired_receipt,
        )
        if (summary.result is not None) != (observed_terminal == "success"):
            raise ValueError("paired execution summary contradicts terminal evidence")
    return 0 if summary.result is not None else 2


# 将observer失败收敛为不回显路径、credential或底层异常的固定CLI错误
def main(argv: Sequence[str] | None = None) -> int:
    try:
        return _main(argv)
    except (OSError, RuntimeError, ValueError):
        print("phase9d paired observer failed", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
