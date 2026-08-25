from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import multiprocessing
import shutil
import tempfile
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from kama_claude.core.verification import (
    MAX_VERIFICATION_OUTPUT_BYTES,
    ExecutionOutputSnapshotManifest,
    SnapshotArtifact,
    SnapshotCaptureError,
    VerificationEnvironmentUnavailable,
    VerificationResult,
    VerificationRuntimeProfileV1,
    VerificationSpecV1,
    materialize_snapshot_copy,
)

logger = logging.getLogger(__name__)

RUNTIME_COPY_CANCEL_WAIT_S = 10.0
RUNTIME_COPY_FORCE_WAIT_S = 1.0
RUNTIME_COPY_MATERIALIZE_WAIT_S = 30.0
RUNTIME_COPY_CLEANUP_WAIT_S = 30.0
_RUNTIME_COPY_PROCESS_START_METHOD = "spawn"


# 在 owned process 中复制 immutable snapshot 到 disposable runtime copy
def _materialize_runtime_copy_worker(
    manifest_payload: dict[str, object],
    artifact_dir: str,
    destination: str,
    result_path: str,
    cancel_event: object,
) -> None:
    try:
        manifest = ExecutionOutputSnapshotManifest.model_validate(manifest_payload)
        artifact = SnapshotArtifact(
            manifest=manifest,
            artifact_dir=Path(artifact_dir),
            tree_root=Path(artifact_dir) / "tree",
            manifest_path=Path(artifact_dir) / "manifest.json",
        )
        materialize_snapshot_copy(
            artifact,
            Path(destination),
            cancel_event=cancel_event,
        )
        payload: dict[str, object] = {"ok": True}
    except Exception as exc:
        payload = {"ok": False, "error": type(exc).__name__}
    try:
        path = Path(result_path)
        path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    except OSError:
        pass


# 在 owned process 中删除 disposable runtime tree
def _remove_runtime_tree_worker(path: str, result_path: str) -> None:
    try:
        shutil.rmtree(path, ignore_errors=False)
        payload: dict[str, object] = {"ok": True}
    except FileNotFoundError:
        payload = {"ok": True}
    except Exception as exc:
        payload = {"ok": False, "error": type(exc).__name__}
    try:
        Path(result_path).write_text(
            json.dumps(payload, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except OSError:
        pass


# 轮询 owned process，保证 cancellation 不依赖不可控 join 线程
async def _wait_owned_process(process: object, timeout_s: float) -> bool:
    deadline = time.monotonic() + timeout_s
    is_alive = getattr(process, "is_alive")
    while is_alive():
        if time.monotonic() >= deadline:
            return False
        await asyncio.sleep(0.01)
    join = getattr(process, "join")
    join(0)
    return True


# 对不合作的 runtime process 执行 bounded terminate/kill/reap
async def _terminate_owned_process(process: object) -> None:
    is_alive = getattr(process, "is_alive")
    if not is_alive():
        getattr(process, "join")(0)
        return
    getattr(process, "terminate")()
    if await _wait_owned_process(process, RUNTIME_COPY_FORCE_WAIT_S):
        return
    kill = getattr(process, "kill", None)
    if callable(kill):
        kill()
    if not await _wait_owned_process(process, RUNTIME_COPY_FORCE_WAIT_S):
        raise SnapshotCaptureError("runtime worker termination could not be confirmed")


# 返回当前 UTC 时间的稳定 ISO 文本
def _now() -> str:
    return datetime.now(UTC).isoformat()


class _ReadableStream(Protocol):
    # 提供 bounded output reader 所需的 asyncio stream 接口
    async def read(self, size: int = -1) -> bytes: ...


@dataclass(frozen=True, slots=True)
class BoundedOutput:
    # 保存只包含 captured prefix 的 bounded stream 结果
    data: bytes
    exceeded: bool
    truncated: bool

    # 返回 captured prefix 的 digest，绝不声称覆盖未捕获 bytes
    @property
    def captured_digest(self) -> str:
        return hashlib.sha256(self.data).hexdigest()


# 在 stream 读取阶段达到 hard limit 时立即停止收集
async def collect_bounded_output(
    stream: _ReadableStream,
    *,
    limit: int = MAX_VERIFICATION_OUTPUT_BYTES,
) -> BoundedOutput:
    if limit < 1:
        raise ValueError("output limit must be positive")
    data = bytearray()
    while True:
        chunk = await stream.read(min(8192, limit + 1))
        if not chunk:
            return BoundedOutput(bytes(data), exceeded=False, truncated=False)
        remaining = limit - len(data)
        if len(chunk) > remaining:
            if remaining > 0:
                data.extend(chunk[:remaining])
            return BoundedOutput(bytes(data), exceeded=True, truncated=True)
        data.extend(chunk)
        if len(data) == limit:
            next_chunk = await stream.read(1)
            if next_chunk:
                return BoundedOutput(bytes(data), exceeded=True, truncated=True)
            return BoundedOutput(bytes(data), exceeded=False, truncated=False)


# 为 pytest 构造固定 argv，不接受 arbitrary flags 或 executable
def build_pytest_argv(
    spec: VerificationSpecV1,
    python_executable: str,
) -> tuple[str, ...]:
    if spec.kind != "pytest":
        raise ValueError("pytest argv builder requires pytest spec")
    return (
        python_executable,
        "-m",
        "pytest",
        "-q",
        "--maxfail=1",
        *spec.targets,
    )


# 为 compileall 构造固定 argv，不复用 pytest 或通用 shell builder
def build_compileall_argv(
    spec: VerificationSpecV1,
    python_executable: str,
) -> tuple[str, ...]:
    if spec.kind != "compileall":
        raise ValueError("compileall argv builder requires compileall spec")
    return (
        python_executable,
        "-m",
        "compileall",
        "-q",
        *spec.targets,
    )


# 为 verifier tool availability preflight 构造只访问 pinned runtime 的 argv
def build_tool_probe_argv(
    spec: VerificationSpecV1,
    python_executable: str,
) -> tuple[str, ...]:
    if spec.kind == "pytest":
        return (python_executable, "-m", "pytest", "--version")
    if spec.kind == "compileall":
        return (python_executable, "-m", "compileall", "--help")
    raise ValueError("unsupported verification kind")


# 为 trusted runtime identity probe 构造固定 Python 版本 argv
def build_python_identity_probe_argv(
    python_executable: str,
) -> tuple[str, ...]:
    return (python_executable, "--version")


# 构造一次 Docker create argv，集中绑定 profile 的安全参数
def build_docker_create_argv(
    profile: VerificationRuntimeProfileV1,
    command: tuple[str, ...],
    *,
    runtime_copy: Path,
    container_name: str,
    docker_executable: str = "docker",
) -> tuple[str, ...]:
    profile.verify_digest()
    runtime_root = runtime_copy.resolve(strict=True)
    if not runtime_root.is_dir():
        raise ValueError("verification runtime copy must be a directory")
    resources = profile.resource_policy
    pids = str(resources.pids_limit)
    memory = resources.memory
    cpus = resources.cpus
    tmpfs = resources.tmpfs
    env_args: list[str] = []
    for key, value in sorted(profile.env_policy.items()):
        env_args.extend(("--env", f"{key}={value}"))
    env_args.extend(
        (
            "--env",
            "HOME=/tmp",
            "--env",
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1",
            "--env",
            "PYTHONNOUSERSITE=1",
        )
    )
    return (
        docker_executable,
        "create",
        "--pull=never",
        "--name",
        container_name,
        "--label",
        f"kama.verification.attempt={container_name}",
        "--network",
        "none",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--read-only",
        "--workdir",
        "/workspace",
        "--user",
        profile.user_identity,
        "--pids-limit",
        pids,
        "--memory",
        memory,
        "--cpus",
        cpus,
        "--tmpfs",
        f"/tmp:rw,noexec,nosuid,size={tmpfs}",
        "--mount",
        f"type=bind,src={runtime_root},dst=/workspace,rw",
        *env_args,
        profile.image_ref,
        *command,
    )


# 为 verifier availability probe 构造无 workspace 挂载的安全 Docker argv
def build_docker_tool_probe_argv(
    profile: VerificationRuntimeProfileV1,
    command: tuple[str, ...],
    *,
    container_name: str,
    docker_executable: str = "docker",
) -> tuple[str, ...]:
    profile.verify_digest()
    resources = profile.resource_policy
    pids = str(resources.pids_limit)
    memory = resources.memory
    cpus = resources.cpus
    tmpfs = resources.tmpfs
    env_args: list[str] = []
    for key, value in sorted(profile.env_policy.items()):
        env_args.extend(("--env", f"{key}={value}"))
    env_args.extend(
        (
            "--env",
            "HOME=/tmp",
            "--env",
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1",
            "--env",
            "PYTHONNOUSERSITE=1",
        )
    )
    return (
        docker_executable,
        "run",
        "--rm",
        "--pull=never",
        "--name",
        container_name,
        "--label",
        f"kama.verification.attempt={container_name}",
        "--network",
        "none",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--read-only",
        "--workdir",
        "/tmp",
        "--user",
        profile.user_identity,
        "--pids-limit",
        pids,
        "--memory",
        memory,
        "--cpus",
        cpus,
        "--tmpfs",
        f"/tmp:rw,noexec,nosuid,size={tmpfs}",
        *env_args,
        profile.image_ref,
        *command,
    )


class DockerVerificationRunner:
    # 仅在 pinned profile 和 disposable runtime copy 上执行单次 verifier invocation
    def __init__(
        self,
        profile: VerificationRuntimeProfileV1,
        *,
        docker_executable: str = "docker",
        container_name_prefix: str = "kama-verification",
        cleanup_grace_s: float = 3.0,
    ) -> None:
        self._profile = profile
        self._docker = docker_executable
        self._name_prefix = container_name_prefix
        self._cleanup_grace_s = cleanup_grace_s

    # 运行 bounded Docker management command，避免 stop/rm/inspect 无限等待
    async def _docker_command(
        self,
        *args: str,
        timeout_s: float = 10.0,
    ) -> tuple[int, bytes, bytes]:
        proc = await asyncio.create_subprocess_exec(
            self._docker,
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout_s)
        except TimeoutError:
            if proc.returncode is None:
                proc.kill()
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), 1.0)
            except TimeoutError:
                stdout, stderr = b"", b"docker management timeout"
            return -1, stdout, stderr
        except asyncio.CancelledError:
            if proc.returncode is None:
                proc.kill()
            try:
                await asyncio.wait_for(proc.communicate(), 1.0)
            except (TimeoutError, asyncio.CancelledError):
                pass
            raise
        return proc.returncode or 0, stdout, stderr

    # 检查本地 image manifest 与 expected Docker image ID 完全一致
    async def inspect_image(self) -> str:
        self._profile.verify_digest()
        try:
            returncode, stdout, _stderr = await self._docker_command(
                "image",
                "inspect",
                "--format",
                "{{.Id}}",
                self._profile.image_ref,
            )
        except OSError as exc:
            raise VerificationEnvironmentUnavailable(
                "Docker executable is unavailable"
            ) from exc
        if returncode != 0:
            raise VerificationEnvironmentUnavailable(
                "verification image is unavailable locally"
            )
        observed = stdout.decode("utf-8", errors="replace").strip()
        if observed != self._profile.expected_image_id:
            raise VerificationEnvironmentUnavailable("verification image identity mismatch")
        return observed

    # 运行 profile tool probe，证明 pytest 或 compileall 本身存在
    async def preflight_tool(self, spec: VerificationSpecV1) -> None:
        image_id = await self.inspect_image()
        del image_id
        identity_name = f"{self._name_prefix}-identity-{uuid.uuid4().hex}"
        identity_argv = build_docker_tool_probe_argv(
            self._profile,
            build_python_identity_probe_argv(self._profile.python_executable),
            container_name=identity_name,
            docker_executable=self._docker,
        )
        returncode, stdout, stderr = await self._run_named_probe(identity_name, identity_argv)
        if returncode != 0 or len(stdout) + len(stderr) > MAX_VERIFICATION_OUTPUT_BYTES:
            raise VerificationEnvironmentUnavailable(
                "Python runtime identity probe failed"
            )
        observed_identity = (
            (stdout + stderr).decode("utf-8", errors="replace").strip().splitlines()[0]
            if (stdout + stderr).strip()
            else ""
        )
        expected_identity = self._profile.expected_python_identity.strip().splitlines()[0]
        if observed_identity != expected_identity:
            raise VerificationEnvironmentUnavailable("Python runtime identity mismatch")

        command = build_tool_probe_argv(spec, self._profile.python_executable)
        tool_name = f"{self._name_prefix}-probe-{uuid.uuid4().hex}"
        tool_argv = build_docker_tool_probe_argv(
            self._profile,
            command,
            container_name=tool_name,
            docker_executable=self._docker,
        )
        returncode, stdout, stderr = await self._run_named_probe(tool_name, tool_argv)
        if returncode != 0 or len(stdout) + len(stderr) > MAX_VERIFICATION_OUTPUT_BYTES:
            raise VerificationEnvironmentUnavailable(
                "verification tool is unavailable in pinned runtime"
            )

    # 执行 named probe 并在所有退出路径确认容器已移除
    async def _run_named_probe(
        self,
        name: str,
        argv: tuple[str, ...],
    ) -> tuple[int, bytes, bytes]:
        primary_error: Exception | None = None
        result: tuple[int, bytes, bytes] | None = None
        try:
            try:
                result = await self._docker_command(*argv[1:])
            except OSError as exc:
                primary_error = VerificationEnvironmentUnavailable(
                    "verification tool preflight could not start"
                )
                primary_error.__cause__ = exc
            except Exception as exc:
                primary_error = VerificationEnvironmentUnavailable(
                    "verification tool preflight failed"
                )
                primary_error.__cause__ = exc
        except asyncio.CancelledError:
            cleanup_ok = True
            try:
                await self._stop_container(name)
            except asyncio.CancelledError:
                cleanup_ok = False
                logger.error(
                    "verification probe stop cancellation cleanup-unverified name=%s",
                    name,
                )
            except Exception as exc:
                cleanup_ok = False
                logger.error(
                    "verification probe stop failed during cancellation name=%s",
                    name,
                    exc_info=exc,
                )
            try:
                cleanup_ok = await self._cleanup_container(name) and cleanup_ok
            except asyncio.CancelledError:
                cleanup_ok = False
                logger.error(
                    "verification probe cleanup cancellation cleanup-unverified name=%s",
                    name,
                )
            except Exception as exc:
                cleanup_ok = False
                logger.error(
                    "verification probe cleanup failed during cancellation name=%s",
                    name,
                    exc_info=exc,
                )
            if not cleanup_ok:
                logger.error(
                    "verification probe cleanup-unverified during cancellation name=%s",
                    name,
                )
            raise
        if result is None:
            result = (-1, b"", b"probe failed")
        if primary_error is not None or result[0] != 0:
            try:
                await self._stop_container(name)
            except asyncio.CancelledError:
                pass
        cleanup_ok = await self._cleanup_container(name)
        if not cleanup_ok:
            raise VerificationEnvironmentUnavailable("probe-cleanup-unverified") from primary_error
        if primary_error is not None:
            raise primary_error
        return result

    # 执行一次 bounded verification attempt 并保证 container cleanup
    async def run(
        self,
        spec: VerificationSpecV1,
        *,
        snapshot_artifact: SnapshotArtifact,
        verification_id: str,
        verification_request_id: str,
        execution_id: str,
    ) -> VerificationResult:
        spec.verify_digest()
        try:
            snapshot_artifact.manifest.verify_artifact(snapshot_artifact.tree_root)
        except (OSError, ValueError, SnapshotCaptureError) as exc:
            raise SnapshotCaptureError("verification input snapshot is corrupt") from exc
        input_digest = snapshot_artifact.manifest.manifest_digest
        runtime_parent = Path(tempfile.mkdtemp(prefix="kama-verification-runtime-"))
        runtime_copy = runtime_parent / "workspace"
        artifact_root = snapshot_artifact.artifact_dir.resolve()
        if runtime_parent.resolve().is_relative_to(artifact_root):
            await self._remove_runtime_tree(runtime_parent)
            raise SnapshotCaptureError("verification runtime copy overlaps snapshot artifact")
        result: VerificationResult | None = None
        primary_error: BaseException | None = None
        cancellation: asyncio.CancelledError | None = None
        cleanup_error: BaseException | None = None
        try:
            await self._materialize_runtime_copy(snapshot_artifact, runtime_copy)
            try:
                snapshot_artifact.manifest.verify_artifact(runtime_copy)
            except (OSError, ValueError, SnapshotCaptureError) as exc:
                raise SnapshotCaptureError("verification runtime copy mismatch") from exc
            result = await self._run_from_runtime_copy(
                spec,
                runtime_copy=runtime_copy,
                verification_id=verification_id,
                verification_request_id=verification_request_id,
                execution_id=execution_id,
                input_digest=input_digest,
            )
        except asyncio.CancelledError as exc:
            cancellation = exc
        except Exception as exc:
            primary_error = exc
        finally:
            try:
                await self._remove_runtime_tree(runtime_parent)
            except asyncio.CancelledError as exc:
                if cancellation is None:
                    cancellation = exc
                cleanup_error = exc
            except Exception as exc:
                cleanup_error = exc

        if cancellation is not None:
            if cleanup_error is not None:
                logger.error(
                    "runtime-copy-cleanup-unverified during cancellation",
                    exc_info=cleanup_error,
                )
            raise cancellation
        if primary_error is not None:
            if cleanup_error is not None:
                logger.error(
                    "runtime-copy-cleanup-unverified after primary failure",
                    exc_info=cleanup_error,
                )
            raise primary_error
        if cleanup_error is not None:
            if result is None:
                raise cleanup_error
            values = result.model_dump(mode="json", exclude={"result_digest"})
            return VerificationResult.create(
                **{
                    **values,
                    "status": "verification_error",
                    "reason": "runtime-copy-cleanup-unverified",
                }
            )
        if result is None:
            raise VerificationEnvironmentUnavailable("verification result is missing")
        return result

    # 将 immutable snapshot 复制到 owned disposable runtime directory
    async def _materialize_runtime_copy(
        self,
        artifact: SnapshotArtifact,
        destination: Path,
    ) -> None:
        result_path = destination.parent / f".{destination.name}.copy-result-{uuid.uuid4().hex}"
        context = multiprocessing.get_context(_RUNTIME_COPY_PROCESS_START_METHOD)
        cancel_event = context.Event()
        process = context.Process(  # type: ignore[attr-defined]
            target=_materialize_runtime_copy_worker,
            args=(
                artifact.manifest.model_dump(mode="json"),
                str(artifact.artifact_dir),
                str(destination),
                str(result_path),
                cancel_event,
            ),
            daemon=True,
        )
        started = False
        try:
            process.start()
            started = True
            completed = await _wait_owned_process(process, RUNTIME_COPY_MATERIALIZE_WAIT_S)
            if not completed:
                raise SnapshotCaptureError("runtime copy worker did not finish")
        except asyncio.CancelledError:
            cancel_event.set()
            try:
                try:
                    completed = await _wait_owned_process(
                        process,
                        RUNTIME_COPY_CANCEL_WAIT_S,
                    )
                    if not completed:
                        await _terminate_owned_process(process)
                except asyncio.CancelledError:
                    logger.error(
                        "runtime-copy worker cancellation cleanup-unverified",
                        exc_info=True,
                    )
                except Exception:
                    logger.error(
                        "runtime-copy worker cleanup-unverified during cancellation",
                        exc_info=True,
                    )
            finally:
                try:
                    await asyncio.shield(self._remove_runtime_tree(destination.parent))
                except Exception:
                    pass
                result_path.unlink(missing_ok=True)
            raise
        except Exception:
            if started and process.is_alive():
                try:
                    await _terminate_owned_process(process)
                except Exception:
                    pass
            try:
                await self._remove_runtime_tree(destination.parent)
            except Exception:
                pass
            result_path.unlink(missing_ok=True)
            raise
        finally:
            if started and process.is_alive():
                cancel_event.set()
                await _terminate_owned_process(process)
            if started:
                process.close()
        try:
            raw = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SnapshotCaptureError("runtime copy worker result is unavailable") from exc
        finally:
            result_path.unlink(missing_ok=True)
        if not isinstance(raw, dict) or raw.get("ok") is not True:
            raise SnapshotCaptureError("runtime copy worker failed")

    # 在 runner-owned runtime copy 上执行一次 Docker attempt
    async def _run_from_runtime_copy(
        self,
        spec: VerificationSpecV1,
        *,
        runtime_copy: Path,
        verification_id: str,
        verification_request_id: str,
        execution_id: str,
        input_digest: str,
    ) -> VerificationResult:
        await self.inspect_image()
        command = (
            build_pytest_argv(spec, self._profile.python_executable)
            if spec.kind == "pytest"
            else build_compileall_argv(spec, self._profile.python_executable)
        )
        container_name = f"{self._name_prefix}-{uuid.uuid4().hex}"
        create_argv = build_docker_create_argv(
            self._profile,
            command,
            runtime_copy=runtime_copy,
            container_name=container_name,
            docker_executable=self._docker,
        )
        started_at = _now()
        observed_container_image_id: str | None = None
        try:
            await self._create_container(create_argv)
            observed_container_image_id = await self._inspect_container_image(container_name)
            if observed_container_image_id != self._profile.expected_image_id:
                raise VerificationEnvironmentUnavailable("container image identity mismatch")
            if observed_container_image_id is None:
                raise VerificationEnvironmentUnavailable("container image identity is missing")
            return await self._start_and_collect(
                container_name,
                verification_id=verification_id,
                verification_request_id=verification_request_id,
                execution_id=execution_id,
                input_digest=input_digest,
                spec=spec,
                image_id=observed_container_image_id,
                started_at=started_at,
            )
        except asyncio.CancelledError:
            await self._cleanup_container(container_name)
            raise
        except VerificationEnvironmentUnavailable:
            await self._cleanup_container(container_name)
            raise
        except Exception:
            cleanup_ok = await self._cleanup_container(container_name)
            return VerificationResult.create(
                verification_id=verification_id,
                verification_request_id=verification_request_id,
                execution_id=execution_id,
                input_digest=input_digest,
                spec_digest=spec.spec_digest,
                runtime_profile_digest=self._profile.profile_digest,
                expected_image_id=self._profile.expected_image_id,
                observed_container_image_id=observed_container_image_id,
                status="verification_error",
                reason=("container-startup-failed" if cleanup_ok else "cleanup_unverified"),
                started_at=started_at,
                finished_at=_now(),
            )

    # 删除 runner-owned runtime tree，不让 verifier bytes 留在工作区或 artifact
    async def _remove_runtime_tree(self, runtime_parent: Path) -> None:
        if not runtime_parent.exists():
            return
        result_path = runtime_parent.parent / (
            f".{runtime_parent.name}.remove-result-{uuid.uuid4().hex}"
        )
        context = multiprocessing.get_context(_RUNTIME_COPY_PROCESS_START_METHOD)
        process = context.Process(  # type: ignore[attr-defined]
            target=_remove_runtime_tree_worker,
            args=(str(runtime_parent), str(result_path)),
            daemon=True,
        )
        started = False
        try:
            process.start()
            started = True
            completed = await _wait_owned_process(process, RUNTIME_COPY_CLEANUP_WAIT_S)
            if not completed:
                raise SnapshotCaptureError("runtime cleanup worker did not finish")
        except asyncio.CancelledError:
            try:
                try:
                    if started:
                        completed = await _wait_owned_process(
                            process,
                            RUNTIME_COPY_CANCEL_WAIT_S,
                        )
                        if not completed:
                            await _terminate_owned_process(process)
                except asyncio.CancelledError:
                    logger.error(
                        "runtime-copy worker cancellation cleanup-unverified",
                        exc_info=True,
                    )
                except Exception:
                    logger.error(
                        "runtime-copy cleanup worker cleanup-unverified during cancellation",
                        exc_info=True,
                    )
            finally:
                result_path.unlink(missing_ok=True)
            raise
        except Exception:
            if started and process.is_alive():
                try:
                    await _terminate_owned_process(process)
                except Exception:
                    pass
            result_path.unlink(missing_ok=True)
            raise
        finally:
            if started and process.is_alive():
                await _terminate_owned_process(process)
            if started:
                process.close()
        if runtime_parent.exists():
            result_path.unlink(missing_ok=True)
            raise SnapshotCaptureError("runtime cleanup could not be confirmed")
        try:
            raw = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SnapshotCaptureError("runtime cleanup worker result is unavailable") from exc
        finally:
            result_path.unlink(missing_ok=True)
        if not isinstance(raw, dict) or raw.get("ok") is not True:
            raise SnapshotCaptureError("runtime cleanup failed")

    # 创建 named container 并检查 Docker 返回的 container identity
    async def _create_container(self, argv: tuple[str, ...]) -> str:
        returncode, stdout, _stderr = await self._docker_command(*argv[1:])
        if returncode != 0:
            raise VerificationEnvironmentUnavailable("Docker container creation failed")
        container_id = stdout.decode("utf-8", errors="replace").strip()
        if not container_id:
            raise VerificationEnvironmentUnavailable("Docker returned no container identity")
        return container_id

    # 读取 container 实际 image ID，防止 create 参数与运行对象脱节
    async def _inspect_container_image(self, name: str) -> str:
        returncode, stdout, _stderr = await self._docker_command(
            "inspect",
            "--format",
            "{{.Image}}",
            name,
        )
        if returncode != 0:
            raise VerificationEnvironmentUnavailable("Docker container inspect failed")
        return stdout.decode("utf-8", errors="replace").strip()

    # attach 到 container process，实时执行 timeout/output bound 和 cleanup
    async def _start_and_collect(
        self,
        name: str,
        *,
        verification_id: str,
        verification_request_id: str,
        execution_id: str,
        input_digest: str,
        spec: VerificationSpecV1,
        image_id: str,
        started_at: str,
    ) -> VerificationResult:
        proc = await asyncio.create_subprocess_exec(
            self._docker,
            "start",
            "--attach",
            name,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        if proc.stdout is None or proc.stderr is None:
            raise VerificationEnvironmentUnavailable("Docker output pipes are unavailable")
        stdout_task = asyncio.create_task(collect_bounded_output(proc.stdout))
        stderr_task = asyncio.create_task(collect_bounded_output(proc.stderr))
        wait_task = asyncio.create_task(proc.wait())
        timed_out = False
        output_exceeded = False
        watched_tasks: set[asyncio.Task[object]] = {
            wait_task,
            stdout_task,
            stderr_task,
        }
        result: VerificationResult | None = None
        try:
            deadline = asyncio.get_running_loop().time() + spec.timeout_s
            while not wait_task.done():
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    timed_out = True
                    break
                done, _pending = await asyncio.wait(
                    watched_tasks,
                    timeout=remaining,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for task in done:
                    watched_tasks.discard(task)
                if stdout_task in done and stdout_task.result().exceeded:
                    output_exceeded = True
                if stderr_task in done and stderr_task.result().exceeded:
                    output_exceeded = True
                if output_exceeded:
                    break
            for output_task in (stdout_task, stderr_task):
                if output_task.done() and output_task.result().exceeded:
                    output_exceeded = True
            if timed_out or output_exceeded or not wait_task.done():
                await self._stop_container(name)
            try:
                return_code = await asyncio.wait_for(
                    asyncio.shield(wait_task),
                    max(1.0, self._cleanup_grace_s + 2.0),
                )
            except TimeoutError:
                if proc.returncode is None:
                    proc.kill()
                return_code = await asyncio.wait_for(
                    asyncio.shield(wait_task),
                    1.0,
                )
            stdout = await asyncio.wait_for(
                asyncio.shield(stdout_task),
                1.0,
            )
            stderr = await asyncio.wait_for(
                asyncio.shield(stderr_task),
                1.0,
            )
            if output_exceeded:
                status = "verification_error"
                reason = "output_limit_exceeded"
            elif timed_out:
                status = "verification_error"
                reason = "verification-timeout"
            elif return_code == 0:
                status = "verification_passed"
                reason = None
            else:
                status = "verification_failed"
                reason = "verification-command-failed"
            result = VerificationResult.create(
                verification_id=verification_id,
                verification_request_id=verification_request_id,
                execution_id=execution_id,
                input_digest=input_digest,
                spec_digest=spec.spec_digest,
                runtime_profile_digest=self._profile.profile_digest,
                expected_image_id=self._profile.expected_image_id,
                observed_container_image_id=image_id,
                status=status,
                exit_code=return_code,
                reason=reason,
                stdout_preview=stdout.data.decode("utf-8", errors="replace"),
                stderr_preview=stderr.data.decode("utf-8", errors="replace"),
                stdout_truncated=stdout.truncated,
                stderr_truncated=stderr.truncated,
                captured_stdout_digest=stdout.captured_digest,
                captured_stderr_digest=stderr.captured_digest,
                started_at=started_at,
                finished_at=_now(),
            )
        except asyncio.CancelledError:
            await self._stop_container(name)
            if proc.returncode is None:
                proc.kill()
            raise
        finally:
            for task in (stdout_task, stderr_task, wait_task):
                if not task.done():
                    task.cancel()
            try:
                await asyncio.wait_for(
                    asyncio.gather(
                        stdout_task,
                        stderr_task,
                        wait_task,
                        return_exceptions=True,
                    ),
                    1.0,
                )
            except TimeoutError:
                if proc.returncode is None:
                    proc.kill()
            cleanup_ok = await self._cleanup_container(name)
            if result is not None and not cleanup_ok:
                values = result.model_dump(mode="json", exclude={"result_digest"})
                result = VerificationResult.create(
                    **{
                        **values,
                        "status": "verification_error",
                        "reason": "cleanup_unverified",
                    }
                )
        if result is None:
            raise VerificationEnvironmentUnavailable("verification result is missing")
        return result

    # 停止 container 并在 grace 后 kill，确保不是只停止 docker CLI
    async def _stop_container(self, name: str) -> None:
        try:
            await self._docker_command(
                "stop",
                "-t",
                str(int(self._cleanup_grace_s)),
                name,
            )
        except Exception:
            logger.warning("verification container stop failed name=%s", name, exc_info=True)
        try:
            await self._docker_command(
                "kill",
                name,
            )
        except Exception:
            logger.warning("verification container kill failed name=%s", name, exc_info=True)

    # 幂等 rm container 并确认 stale container 不会被复用
    async def _cleanup_container(self, name: str) -> bool:
        try:
            remove_code, _remove_stdout, remove_stderr = await self._docker_command(
                "rm",
                "-f",
                name,
            )
            inspect_code, _stdout, inspect_stderr = await self._docker_command(
                "inspect",
                name,
            )
            if inspect_code == 0:
                logger.warning(
                    "verification container still exists after cleanup name=%s",
                    name,
                )
                return False
            if remove_code == 0:
                return True
            missing_markers = (b"no such object", b"no such container", b"not found")
            combined_error = (remove_stderr + inspect_stderr).lower()
            if any(marker in combined_error for marker in missing_markers):
                return True
            logger.warning(
                "verification container cleanup command failed name=%s remove_code=%s",
                name,
                remove_code,
            )
            return False
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("verification container cleanup failed name=%s", name, exc_info=True)
            return False

    # 清理重启后遗留的带 verification attempt label 的容器
    async def cleanup_stale_containers(self, attempt_label: str) -> None:
        try:
            returncode, stdout, _stderr = await self._docker_command(
                "ps",
                "-aq",
                "--filter",
                f"label=kama.verification.attempt={attempt_label}",
            )
        except OSError as exc:
            raise VerificationEnvironmentUnavailable(
                "Docker executable is unavailable for stale cleanup"
            ) from exc
        if returncode != 0:
            raise VerificationEnvironmentUnavailable(
                "stale verification container lookup failed"
            )
        for name in stdout.decode("utf-8", errors="replace").splitlines():
            if name.strip():
                await self._stop_container(name.strip())
                if not await self._cleanup_container(name.strip()):
                    raise VerificationEnvironmentUnavailable(
                        "stale verification container cleanup could not be confirmed"
                    )
