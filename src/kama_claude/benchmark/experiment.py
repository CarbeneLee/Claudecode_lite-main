from __future__ import annotations

import hashlib
import json
import os
import platform as platform_module
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal
from urllib.parse import urlsplit

from dotenv import dotenv_values
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from kama_claude.benchmark.schema import LoadedBenchmarkSuite, load_suite


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
SafeIdentifier = Annotated[str, Field(min_length=1, pattern=r"^[A-Za-z0-9._-]+$")]


class SuiteProfile(_StrictModel):
    manifest: str = Field(min_length=1)
    freeze_manifest: str = Field(min_length=1)
    tasks_root: str = Field(min_length=1)
    expected_suite_hash: Sha256


class ProviderProfile(_StrictModel):
    service_provider: Literal["deepseek"]
    wire_protocol: Literal["anthropic_messages"]
    endpoint_id: Literal["deepseek-anthropic-compatible"]
    endpoint: str = Field(min_length=1)
    model_id: Literal["deepseek-v4-pro"]
    sdk_distribution: Literal["anthropic"]
    sdk_version: str = Field(pattern=r"^[0-9]+\.[0-9]+\.[0-9]+$")
    credential_env: Literal["ANTHROPIC_API_KEY"]

    @field_validator("endpoint")
    @classmethod
    # 只接受无 credential、query 或 fragment 的 HTTPS provider endpoint
    def _endpoint_is_public_https(cls, value: str) -> str:
        parsed = urlsplit(value)
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("provider endpoint must be a public HTTPS URL")
        normalized = value.rstrip("/")
        if normalized != "https://api.deepseek.com/anthropic":
            raise ValueError("provider endpoint does not match endpoint identity")
        return normalized


class RuntimeProfile(_StrictModel):
    max_steps: Annotated[int, Field(ge=1)]
    router: Literal["static"]
    compaction_threshold: Annotated[float, Field(ge=0.0, le=1.0)]
    tool_result_limit: Annotated[int, Field(ge=1)]
    tool_result_keep: Annotated[int, Field(ge=1)]
    mcp_enabled: Literal[False]
    trace_enabled: Literal[True]
    include_llm_payload: Literal[True]


class ScheduleProfile(_StrictModel):
    repeats: Annotated[int, Field(ge=1, le=3)]
    execution_order: Literal["suite_task_then_repeat_ascending"]


class ArtifactProfile(_StrictModel):
    output_root_must_be_new: Literal[True]
    output_root_must_be_outside_repository: Literal[True]
    retain_all_attempts: Literal[True]
    raw_trace_visibility: Literal["private"]


class ExpectedIdentity(_StrictModel):
    prompt_hash: Sha256
    tool_schema_hash: Sha256


class ExperimentProfile(_StrictModel):
    schema_version: Literal[1]
    profile_id: SafeIdentifier
    suite: SuiteProfile
    provider: ProviderProfile
    runtime: RuntimeProfile
    schedule: ScheduleProfile
    artifacts: ArtifactProfile
    expected_identity: ExpectedIdentity


class RepositoryIdentity(_StrictModel):
    commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    dirty: bool


class ProviderIdentity(_StrictModel):
    service_provider: str
    wire_protocol: str
    endpoint_id: str
    endpoint: str
    model_id: str
    sdk_distribution: str
    sdk_version: str


class RuntimeIdentity(_StrictModel):
    max_steps: int
    router: str
    compaction_threshold: float
    tool_result_limit: int
    tool_result_keep: int
    mcp_enabled: bool
    trace_enabled: bool
    include_llm_payload: bool


class SuiteIdentity(_StrictModel):
    suite_id: str
    suite_version: int
    suite_hash: Sha256
    task_hashes: dict[str, Sha256]
    grader_hashes: dict[str, Sha256]


class DependencyIdentity(_StrictModel):
    pyproject_hash: Sha256
    uv_lock_hash: Sha256
    dependency_hash: Sha256


class HostIdentity(_StrictModel):
    python_version: str
    os: str
    os_release: str
    architecture: str


class DeclaredExperimentIdentity(_StrictModel):
    schema_version: Literal[1] = 1
    profile_id: str
    profile_hash: Sha256
    git: RepositoryIdentity
    suite: SuiteIdentity
    provider: ProviderIdentity
    prompt_hash: Sha256
    tool_schema_hash: Sha256
    runtime: RuntimeIdentity
    runtime_config_hash: Sha256
    dependency: DependencyIdentity
    host: HostIdentity
    schedule: ScheduleProfile
    artifacts: ArtifactProfile


class ObservedExperimentIdentity(_StrictModel):
    run_id: str = Field(min_length=1)
    provider: ProviderIdentity
    prompt_hash: Sha256
    tool_schema_hash: Sha256
    runtime: RuntimeIdentity
    runtime_config_hash: Sha256
    api_call_count: Annotated[int, Field(ge=1)]
    model_event_ids: Annotated[list[str], Field(min_length=1)]


class IdentityVerification(_StrictModel):
    valid: bool
    mismatches: list[str]


class ObservedIdentitySummary(_StrictModel):
    provider: ProviderIdentity
    prompt_hash: Sha256
    tool_schema_hash: Sha256
    runtime: RuntimeIdentity
    runtime_config_hash: Sha256
    attempts: Annotated[int, Field(ge=1)]
    api_calls: Annotated[int, Field(ge=1)]
    model_event_ids: Annotated[list[str], Field(min_length=1)]


class ExperimentVerificationSummary(_StrictModel):
    status: Literal["match"]
    verified_attempts: Annotated[int, Field(ge=1)]
    mismatches: list[str]


class VerifiedExperimentIdentity(_StrictModel):
    status: Literal["valid"] = "valid"
    declared: DeclaredExperimentIdentity
    observed: ObservedIdentitySummary
    verification: ExperimentVerificationSummary


class ExperimentIdentityMismatch(ValueError):
    # 保存稳定 mismatch 字段，错误文本不包含 trace payload 或路径
    def __init__(self, mismatches: list[str]) -> None:
        self.mismatches = tuple(mismatches)
        super().__init__(
            "declared and observed experiment identity mismatch: "
            + ",".join(mismatches)
        )


class _FrozenTask(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True, strict=True)
    task_id: SafeIdentifier
    task_hash: Sha256
    grader_hash: Sha256


class _FreezeManifest(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True, strict=True)
    schema_version: Literal[1]
    suite_id: SafeIdentifier
    suite_version: Annotated[int, Field(ge=1)]
    suite_hash: Sha256
    tasks: Annotated[list[_FrozenTask], Field(min_length=1)]


@dataclass(frozen=True)
class LoadedExperimentProfile:
    profile_path: Path
    benchmark_root: Path
    suite_path: Path
    freeze_path: Path
    tasks_root: Path
    profile: ExperimentProfile


# 对 JSON-compatible identity payload 生成保留 list 顺序的 canonical SHA-256
def canonical_hash(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


# 对单个文件 bytes 计算 SHA-256，并把文件系统错误净化为 identity error
def _hash_file(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise ValueError("experiment identity input cannot be hashed") from exc


# 对 task directory 的 path-content 计算与 frozen suite 相同的稳定哈希
def _hash_task_directory(task_dir: Path) -> str:
    digest = hashlib.sha256()
    try:
        paths = sorted(
            task_dir.rglob("*"),
            key=lambda path: path.relative_to(task_dir).as_posix(),
        )
        for path in paths:
            if path.is_symlink():
                raise ValueError("benchmark task contains a symlink")
            if path.is_dir():
                continue
            if not path.is_file():
                raise ValueError("benchmark task contains a non-regular entry")
            digest.update(path.relative_to(task_dir).as_posix().encode("utf-8"))
            digest.update(b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
    except OSError as exc:
        raise ValueError("benchmark task cannot be hashed") from exc
    return digest.hexdigest()


# 对 grader JSON 与 hidden tests 的 path-content 计算稳定私有 oracle 哈希
def _hash_grader_bundle(task_dir: Path) -> str:
    private = task_dir / "private"
    hidden = private / "hidden_tests"
    paths = [private / "grader.json"]
    if hidden.is_dir():
        paths.extend(path for path in hidden.rglob("*") if path.is_file())
    digest = hashlib.sha256()
    try:
        for path in sorted(paths, key=lambda item: item.relative_to(task_dir).as_posix()):
            if path.is_symlink() or not path.is_file():
                raise ValueError("benchmark grader contains an invalid entry")
            digest.update(path.relative_to(task_dir).as_posix().encode("utf-8"))
            digest.update(b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
    except OSError as exc:
        raise ValueError("benchmark grader cannot be hashed") from exc
    return digest.hexdigest()


# 读取 strict freeze evidence 并隐藏底层 JSON/validation 细节
def _load_freeze(path: Path) -> _FreezeManifest:
    try:
        return _FreezeManifest.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValidationError) as exc:
        raise ValueError("invalid frozen suite evidence") from exc


# 将 profile provider 字段投影为不含 credential variable 的公开身份
def _provider_identity(profile: ProviderProfile) -> ProviderIdentity:
    return ProviderIdentity(
        service_provider=profile.service_provider,
        wire_protocol=profile.wire_protocol,
        endpoint_id=profile.endpoint_id,
        endpoint=profile.endpoint,
        model_id=profile.model_id,
        sdk_distribution=profile.sdk_distribution,
        sdk_version=profile.sdk_version,
    )


# 将 profile runtime 字段投影为 worker 必须观察到的行为配置身份
def _runtime_identity(profile: RuntimeProfile) -> RuntimeIdentity:
    return RuntimeIdentity(**profile.model_dump(mode="python"))


# 校验 frozen suite 与当前 suite/task/grader 完全一致并生成公开 identity
def _suite_identity(
    loaded: LoadedExperimentProfile,
    suite: LoadedBenchmarkSuite,
) -> SuiteIdentity:
    freeze = _load_freeze(loaded.freeze_path)
    suite_hash = canonical_hash(suite.manifest.model_dump(mode="json"))
    if suite_hash != loaded.profile.suite.expected_suite_hash:
        raise ValueError("experiment suite hash does not match profile")
    if (
        freeze.suite_id != suite.manifest.suite_id
        or freeze.suite_version != suite.manifest.suite_version
        or freeze.suite_hash != suite_hash
    ):
        raise ValueError("frozen suite identity mismatch")
    task_ids = [task.metadata.task_id for task in suite.tasks]
    if [task.task_id for task in freeze.tasks] != task_ids:
        raise ValueError("frozen suite task order mismatch")
    task_hashes = {
        task.metadata.task_id: _hash_task_directory(task.task_dir)
        for task in suite.tasks
    }
    grader_hashes = {
        task.metadata.task_id: _hash_grader_bundle(task.task_dir)
        for task in suite.tasks
    }
    for frozen in freeze.tasks:
        if (
            frozen.task_hash != task_hashes[frozen.task_id]
            or frozen.grader_hash != grader_hashes[frozen.task_id]
        ):
            raise ValueError("frozen task or grader hash mismatch")
    return SuiteIdentity(
        suite_id=suite.manifest.suite_id,
        suite_version=suite.manifest.suite_version,
        suite_hash=suite_hash,
        task_hashes=task_hashes,
        grader_hashes=grader_hashes,
    )


# 对 dependency lock inputs 计算独立文件哈希与组合哈希
def _dependency_identity(repository_root: Path) -> DependencyIdentity:
    pyproject_hash = _hash_file(repository_root / "pyproject.toml")
    uv_lock_hash = _hash_file(repository_root / "uv.lock")
    return DependencyIdentity(
        pyproject_hash=pyproject_hash,
        uv_lock_hash=uv_lock_hash,
        dependency_hash=canonical_hash(
            {
                "pyproject.toml": pyproject_hash,
                "uv.lock": uv_lock_hash,
            }
        ),
    )


# 在任何 attempt 前绑定 clean Git、frozen suite、provider/runtime、dependencies 与 host
def capture_declared_identity(
    loaded: LoadedExperimentProfile,
    *,
    repository_root: Path | str,
    repository: RepositoryIdentity,
    installed_sdk_version: str,
) -> DeclaredExperimentIdentity:
    if repository.dirty:
        raise ValueError("baseline experiment requires a clean Git repository")
    if installed_sdk_version != loaded.profile.provider.sdk_version:
        raise ValueError("installed SDK version does not match experiment profile")
    root = Path(repository_root).resolve(strict=True)
    suite = load_suite(loaded.suite_path, loaded.tasks_root)
    runtime = _runtime_identity(loaded.profile.runtime)
    return DeclaredExperimentIdentity(
        profile_id=loaded.profile.profile_id,
        profile_hash=canonical_hash(loaded.profile.model_dump(mode="json")),
        git=repository,
        suite=_suite_identity(loaded, suite),
        provider=_provider_identity(loaded.profile.provider),
        prompt_hash=loaded.profile.expected_identity.prompt_hash,
        tool_schema_hash=loaded.profile.expected_identity.tool_schema_hash,
        runtime=runtime,
        runtime_config_hash=canonical_hash(runtime.model_dump(mode="json")),
        dependency=_dependency_identity(root),
        host=HostIdentity(
            python_version=platform_module.python_version(),
            os=platform_module.system(),
            os_release=platform_module.release(),
            architecture=platform_module.machine(),
        ),
        schedule=loaded.profile.schedule,
        artifacts=loaded.profile.artifacts,
    )


# 原子写入 canonical declared-experiment.json 并要求 experiment output root 全新
def write_declared_experiment(
    output_root: Path | str,
    identity: DeclaredExperimentIdentity,
) -> None:
    root = Path(output_root)
    try:
        root.mkdir(parents=True, exist_ok=False)
        target = root / "declared-experiment.json"
        temporary = root / "declared-experiment.json.tmp"
        temporary.write_text(
            json.dumps(
                identity.model_dump(mode="json"),
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(target)
    except OSError as exc:
        raise ValueError("experiment output root must be new and writable") from exc


# 校验 baseline output 是 repository 外部的全新路径，避免 artifacts 改写声明的 Git identity
def validate_experiment_output(
    output_root: Path | str,
    repository_root: Path | str,
    policy: ArtifactProfile,
) -> Path:
    output = Path(output_root).resolve(strict=False)
    repository = Path(repository_root).resolve(strict=True)
    if policy.output_root_must_be_outside_repository and output.is_relative_to(
        repository
    ):
        raise ValueError("experiment output root must be outside repository")
    if policy.output_root_must_be_new and output.exists():
        raise ValueError("experiment output root must be new")
    return output


@contextmanager
# 在串行 benchmark scope 内用 profile 覆盖行为环境，并在退出时完整恢复 caller 环境
def scoped_experiment_environment(
    profile: ExperimentProfile,
    *,
    credential: str | None = None,
) -> Iterator[None]:
    behavior_environment = {
        "ANTHROPIC_BASE_URL": profile.provider.endpoint,
        "KAMA_LLM_DEFAULT_MODEL": profile.provider.model_id,
        "KAMA_MAX_STEPS": str(profile.runtime.max_steps),
        "KAMA_TRACE_ENABLED": "true",
        "KAMA_TRACE_INCLUDE_LLM_PAYLOAD": "true",
        "KAMA_COMPACT_THRESHOLD": str(profile.runtime.compaction_threshold),
        "KAMA_COMPACT_TOOL_LIMIT": str(profile.runtime.tool_result_limit),
        "KAMA_COMPACT_TOOL_KEEP": str(profile.runtime.tool_result_keep),
    }
    if credential is not None:
        behavior_environment[profile.provider.credential_env] = credential
    keys = (*behavior_environment, "KAMA_CONFIG")
    previous = {key: os.environ.get(key) for key in keys}
    try:
        os.environ.update(behavior_environment)
        os.environ.pop("KAMA_CONFIG", None)
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


# 读取非空 JSONL artifact，并拒绝任何损坏或非 object frame
def _read_jsonl(path: Path) -> list[dict[str, object]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        frames = [json.loads(line) for line in lines if line.strip()]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("observed identity evidence is invalid") from exc
    if not frames or any(not isinstance(frame, dict) for frame in frames):
        raise ValueError("observed identity evidence is invalid")
    return frames


# 从 trace api_call 提取唯一 effective prompt 与 ordered tool schema hash
def _api_call_identity(
    records: list[dict[str, object]],
    run_id: str,
) -> tuple[str, str, int]:
    calls = [
        record
        for record in records
        if record.get("kind") == "api_call"
        and record.get("direction") == "CORE→LLM"
    ]
    if not calls or any(record.get("run_id") != run_id for record in calls):
        raise ValueError("observed identity evidence is invalid")
    prompt_hashes: set[str] = set()
    tool_hashes: set[str] = set()
    for record in calls:
        data = record.get("data")
        if not isinstance(data, dict):
            raise ValueError("observed identity evidence is invalid")
        prompt = data.get("system")
        tools = data.get("tool_schemas")
        if not isinstance(prompt, str) or not isinstance(tools, list):
            raise ValueError("observed identity evidence is invalid")
        prompt_hashes.add(hashlib.sha256(prompt.encode("utf-8")).hexdigest())
        tool_hashes.add(canonical_hash(tools))
    if len(prompt_hashes) != 1 or len(tool_hashes) != 1:
        raise ValueError("observed identity changed within an attempt")
    return prompt_hashes.pop(), tool_hashes.pop(), len(calls)


# 从 run journal 提取同一 run 的全部 model-selected identity
def _model_event_identity(
    frames: list[dict[str, object]],
    run_id: str,
) -> list[str]:
    models: list[str] = []
    for frame in frames:
        event = frame.get("event")
        if not isinstance(event, dict) or event.get("type") != "llm.model_selected":
            continue
        if (
            frame.get("stream_id") != f"run:{run_id}"
            or event.get("run_id") != run_id
            or not isinstance(event.get("model"), str)
        ):
            raise ValueError("observed identity evidence is invalid")
        models.append(event["model"])
    if not models:
        raise ValueError("observed identity evidence is invalid")
    return models


# 从单 attempt 已落盘 trace/journal 机械收集 provider、runtime、prompt、tools 与 model identity
def collect_observed_identity(
    attempt_root: Path | str,
) -> ObservedExperimentIdentity:
    runtime_dir = Path(attempt_root) / "runtime"
    trace_records = _read_jsonl(runtime_dir / "trace.jsonl")
    runtime_records = [
        record
        for record in trace_records
        if record.get("kind") == "runtime_identity"
        and record.get("direction") == "CORE"
    ]
    if len(runtime_records) != 1:
        raise ValueError("observed identity evidence is invalid")
    runtime_record = runtime_records[0]
    run_id = runtime_record.get("run_id")
    data = runtime_record.get("data")
    if not isinstance(run_id, str) or not run_id or not isinstance(data, dict):
        raise ValueError("observed identity evidence is invalid")
    try:
        provider = ProviderIdentity.model_validate(data.get("provider"))
        runtime = RuntimeIdentity.model_validate(data.get("runtime"))
    except ValidationError as exc:
        raise ValueError("observed identity evidence is invalid") from exc
    prompt_hash, tool_schema_hash, api_call_count = _api_call_identity(
        trace_records,
        run_id,
    )
    models = _model_event_identity(
        _read_jsonl(runtime_dir / "events.v2.jsonl"),
        run_id,
    )
    return ObservedExperimentIdentity(
        run_id=run_id,
        provider=provider,
        prompt_hash=prompt_hash,
        tool_schema_hash=tool_schema_hash,
        runtime=runtime,
        runtime_config_hash=canonical_hash(runtime.model_dump(mode="json")),
        api_call_count=api_call_count,
        model_event_ids=models,
    )


# 按稳定字段顺序比较 declaration 与单 attempt observation
def verify_declared_observed(
    declared: DeclaredExperimentIdentity,
    observed: ObservedExperimentIdentity,
) -> IdentityVerification:
    mismatches: list[str] = []
    if observed.provider != declared.provider:
        mismatches.append("provider")
    if observed.runtime != declared.runtime:
        mismatches.append("runtime")
    if observed.runtime_config_hash != declared.runtime_config_hash:
        mismatches.append("runtime_config_hash")
    if observed.prompt_hash != declared.prompt_hash:
        mismatches.append("prompt_hash")
    if observed.tool_schema_hash != declared.tool_schema_hash:
        mismatches.append("tool_schema_hash")
    if any(model != declared.provider.model_id for model in observed.model_event_ids):
        mismatches.append("model_events")
    return IdentityVerification(valid=not mismatches, mismatches=mismatches)


# 对任一 behavior identity mismatch fail closed，不把 invalid experiment 交给 metrics
def require_identity_match(
    declared: DeclaredExperimentIdentity,
    observed: ObservedExperimentIdentity,
) -> None:
    verification = verify_declared_observed(declared, observed)
    if not verification.valid:
        raise ExperimentIdentityMismatch(verification.mismatches)


# 将全部匹配的 attempt observations 收敛为 report-facing canonical experiment identity
def build_verified_experiment_identity(
    declared: DeclaredExperimentIdentity,
    observations: list[ObservedExperimentIdentity],
) -> VerifiedExperimentIdentity:
    if not observations:
        raise ValueError("verified experiment identity requires observations")
    for observed in observations:
        require_identity_match(declared, observed)
    first = observations[0]
    identity_key = (
        first.provider,
        first.prompt_hash,
        first.tool_schema_hash,
        first.runtime,
        first.runtime_config_hash,
    )
    if any(
        (
            observed.provider,
            observed.prompt_hash,
            observed.tool_schema_hash,
            observed.runtime,
            observed.runtime_config_hash,
        )
        != identity_key
        for observed in observations[1:]
    ):
        raise ExperimentIdentityMismatch(["cross_attempt_identity"])
    return VerifiedExperimentIdentity(
        declared=declared,
        observed=ObservedIdentitySummary(
            provider=first.provider,
            prompt_hash=first.prompt_hash,
            tool_schema_hash=first.tool_schema_hash,
            runtime=first.runtime,
            runtime_config_hash=first.runtime_config_hash,
            attempts=len(observations),
            api_calls=sum(observed.api_call_count for observed in observations),
            model_event_ids=sorted(
                {
                    model
                    for observed in observations
                    for model in observed.model_event_ids
                }
            ),
        ),
        verification=ExperimentVerificationSummary(
            status="match",
            verified_attempts=len(observations),
            mismatches=[],
        ),
    )


# 只从 process env 或 repository .env 读取 profile 指定 credential，不加载任何行为配置
def resolve_experiment_credential(
    profile: ExperimentProfile,
    repository_root: Path | str,
) -> str:
    name = profile.provider.credential_env
    ambient = os.environ.get(name)
    if ambient:
        return ambient
    try:
        candidate = dotenv_values(Path(repository_root) / ".env").get(name)
    except OSError as exc:
        raise ValueError("experiment credential cannot be read") from exc
    if not isinstance(candidate, str) or not candidate:
        raise ValueError("experiment credential is missing")
    return candidate


# 写入不含 payload 的 invalid receipt，并确保 invalid experiment 不留下 baseline report
def write_invalid_experiment(
    output_root: Path | str,
    mismatches: list[str],
) -> None:
    root = Path(output_root)
    payload = {
        "schema_version": 1,
        "experiment_status": "invalid",
        "reason": "identity_mismatch",
        "mismatches": mismatches,
    }
    try:
        (root / "baseline.json").unlink(missing_ok=True)
        (root / "baseline.md").unlink(missing_ok=True)
        target = root / "experiment-invalid.json"
        temporary = root / "experiment-invalid.json.tmp"
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(target)
    except OSError as exc:
        raise ValueError("invalid experiment receipt cannot be written") from exc


# 将 profile 的相对引用限制在同一 benchmarks root 内并验证目标类型
def _resolve_profile_path(
    profile_dir: Path,
    benchmark_root: Path,
    value: str,
    *,
    directory: bool,
) -> Path:
    raw = Path(value)
    if raw.is_absolute():
        raise ValueError("experiment profile paths must be relative")
    try:
        resolved = (profile_dir / raw).resolve(strict=True)
    except OSError as exc:
        raise ValueError("experiment profile reference is missing") from exc
    if not resolved.is_relative_to(benchmark_root):
        raise ValueError("experiment profile reference escapes benchmark root")
    if directory and not resolved.is_dir():
        raise ValueError("experiment profile directory reference is invalid")
    if not directory and not resolved.is_file():
        raise ValueError("experiment profile file reference is invalid")
    return resolved


# 加载 strict versioned experiment profile 并解析受 containment 保护的 benchmark 引用
def load_experiment_profile(path: Path | str) -> LoadedExperimentProfile:
    try:
        profile_path = Path(path).resolve(strict=True)
        profile = ExperimentProfile.model_validate_json(
            profile_path.read_text(encoding="utf-8")
        )
    except (OSError, UnicodeDecodeError, ValidationError) as exc:
        raise ValueError("invalid experiment profile") from exc
    if not profile_path.is_file() or profile_path.parent.name != "experiments":
        raise ValueError("invalid experiment profile location")
    benchmark_root = profile_path.parent.parent.resolve(strict=True)
    try:
        suite_path = _resolve_profile_path(
            profile_path.parent,
            benchmark_root,
            profile.suite.manifest,
            directory=False,
        )
        freeze_path = _resolve_profile_path(
            profile_path.parent,
            benchmark_root,
            profile.suite.freeze_manifest,
            directory=False,
        )
        tasks_root = _resolve_profile_path(
            profile_path.parent,
            benchmark_root,
            profile.suite.tasks_root,
            directory=True,
        )
    except ValueError as exc:
        raise ValueError("invalid experiment profile") from exc
    return LoadedExperimentProfile(
        profile_path=profile_path,
        benchmark_root=benchmark_root,
        suite_path=suite_path,
        freeze_path=freeze_path,
        tasks_root=tasks_root,
        profile=profile,
    )
