from __future__ import annotations

import hashlib
import json
import platform as platform_module
import subprocess
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from kama_claude.benchmark.analyzers import (
    AttemptAnalysis,
    BenchmarkMetrics,
)
from kama_claude.benchmark.experiment import VerifiedExperimentIdentity
from kama_claude.benchmark.schema import LoadedBenchmarkSuite


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class RepositoryState(_StrictModel):
    commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    dirty: bool


class ExperimentIdentity(_StrictModel):
    suite_id: str
    suite_version: int = Field(ge=1)
    suite_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    task_hashes: dict[str, str]
    repeats: int = Field(ge=1, le=3)
    model_id: str = Field(min_length=1)
    git_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    git_dirty: bool
    python_version: str = Field(min_length=1)
    platform: str = Field(min_length=1)


class BaselineReport(_StrictModel):
    artifact_version: Literal[2] = 2
    scope: Literal["fixed_task_internal_benchmark"] = "fixed_task_internal_benchmark"
    security_boundary: Literal[
        "process_isolation_not_security_sandbox"
    ] = "process_isolation_not_security_sandbox"
    statistical_claim: Literal[
        "descriptive_not_statistically_significant"
    ] = "descriptive_not_statistically_significant"
    external_benchmark_claim: Literal[
        "not_swe_bench_or_general_coding_ability"
    ] = "not_swe_bench_or_general_coding_ability"
    experiment: VerifiedExperimentIdentity
    metrics: BenchmarkMetrics
    attempts: list[AttemptAnalysis]


# 对 task directory 的路径与文件内容计算稳定哈希，不将 private 内容写入 report
def _hash_task_directory(task_dir: Path) -> str:
    digest = hashlib.sha256()
    try:
        paths = sorted(task_dir.rglob("*"), key=lambda path: path.relative_to(task_dir).as_posix())
        for path in paths:
            if path.is_symlink():
                raise ValueError("benchmark task contains a symlink")
            if path.is_dir():
                continue
            if not path.is_file():
                raise ValueError("benchmark task contains a non-regular entry")
            relative = path.relative_to(task_dir).as_posix().encode("utf-8")
            digest.update(relative)
            digest.update(b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
    except OSError as exc:
        raise ValueError("benchmark task cannot be hashed") from exc
    return digest.hexdigest()


# 对 canonical suite manifest 计算稳定哈希
def _hash_suite(suite: LoadedBenchmarkSuite) -> str:
    payload = json.dumps(
        suite.manifest.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


# 只读获取 full Git commit 和 dirty 布尔值，不保留文件路径
def probe_repository(repository_root: Path | str) -> RepositoryState:
    root = Path(repository_root)
    try:
        commit_result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
        status_result = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError("repository identity cannot be observed") from exc
    return RepositoryState(
        commit=commit_result.stdout.strip().lower(),
        dirty=bool(status_result.stdout.strip()),
    )


# 捕获固定 suite、task、model、Git 和本机解释器身份供 baseline 公平比较
def capture_experiment_identity(
    suite: LoadedBenchmarkSuite,
    *,
    repeats: int,
    model_id: str,
    repository: RepositoryState,
) -> ExperimentIdentity:
    return ExperimentIdentity(
        suite_id=suite.manifest.suite_id,
        suite_version=suite.manifest.suite_version,
        suite_hash=_hash_suite(suite),
        task_hashes={
            task.metadata.task_id: _hash_task_directory(task.task_dir)
            for task in suite.tasks
        },
        repeats=repeats,
        model_id=model_id,
        git_commit=repository.commit,
        git_dirty=repository.dirty,
        python_version=platform_module.python_version(),
        platform=f"{platform_module.system()}-{platform_module.machine()}",
    )


# 构建完整且无重复的 task × repeat baseline report
def build_baseline_report(
    identity: VerifiedExperimentIdentity,
    attempts: list[AttemptAnalysis],
    metrics: BenchmarkMetrics,
) -> BaselineReport:
    expected = {
        (task_id, repeat)
        for task_id in identity.declared.suite.task_hashes
        for repeat in range(1, identity.declared.schedule.repeats + 1)
    }
    observed = {(attempt.task_id, attempt.repeat) for attempt in attempts}
    if (
        not attempts
        or len(observed) != len(attempts)
        or observed != expected
        or metrics.overall.scheduled_attempts != len(attempts)
        or identity.observed.attempts != len(attempts)
        or identity.verification.verified_attempts != len(attempts)
    ):
        raise ValueError("baseline attempt matrix is incomplete or inconsistent")
    return BaselineReport(
        experiment=identity,
        metrics=metrics,
        attempts=attempts,
    )


# 将 baseline report 序列化为排序 canonical JSON
def render_json(report: BaselineReport) -> str:
    return (
        json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True)
        + "\n"
    )


# 从同一个 canonical report 模型渲染带明确 claim boundary 的 Markdown
def render_markdown(report: BaselineReport) -> str:
    overall = report.metrics.overall
    declared = report.experiment.declared
    observed = report.experiment.observed
    rows = [
        "# KamaClaude Fixed-task Internal Benchmark",
        "",
        "> This is a Fixed-task internal benchmark; it is not SWE-bench and does not",
        "> establish general coding ability. Process isolation is not a security sandbox,",
        "> and this small repeated sample is not statistically significant.",
        "",
        "## Experiment",
        "",
        f"- Identity status: `{report.experiment.status}`",
        f"- Verification: `{report.experiment.verification.status}`",
        f"- Suite: `{declared.suite.suite_id}@{declared.suite.suite_version}`",
        f"- Suite hash: `{declared.suite.suite_hash}`",
        f"- Provider: `{declared.provider.service_provider}`",
        f"- Protocol: `{declared.provider.wire_protocol}`",
        f"- Endpoint ID: `{declared.provider.endpoint_id}`",
        f"- Endpoint: `{declared.provider.endpoint}`",
        f"- Model: `{declared.provider.model_id}`",
        f"- SDK: `{declared.provider.sdk_distribution}=={declared.provider.sdk_version}`",
        f"- Prompt hash: `{declared.prompt_hash}`",
        f"- Tool schema hash: `{declared.tool_schema_hash}`",
        f"- Max steps: {declared.runtime.max_steps}",
        f"- Runtime config hash: `{declared.runtime_config_hash}`",
        f"- Git commit: `{declared.git.commit}`",
        f"- Git dirty: `{str(declared.git.dirty).lower()}`",
        f"- Repeats: {declared.schedule.repeats}",
        f"- Python: `{declared.host.python_version}`",
        f"- Platform: `{declared.host.os}-{declared.host.architecture}`",
        f"- Verified attempts/API calls: {observed.attempts}/{observed.api_calls}",
        "",
        "## Overall metrics",
        "",
        f"- Success: {overall.successful_attempts}/{overall.scheduled_attempts}",
        f"- Success rate: {overall.success_rate:.3f}",
        f"- Runtime success: {overall.runtime_successful_attempts}",
        f"- Median wall latency: {overall.median_wall_latency_ms:.3f} ms",
        f"- Input/output/cache tokens: {overall.total_input_tokens}/"
        f"{overall.total_output_tokens}/{overall.total_cache_tokens}",
        f"- Steps/tools/retries: {overall.total_steps}/"
        f"{overall.total_tool_calls}/{overall.total_retries}",
        f"- Timeouts: {overall.timeout_count}",
        "",
        "## Category metrics",
        "",
        "| Category | Success | Rate | Median latency (ms) |",
        "| --- | ---: | ---: | ---: |",
    ]
    rows.extend(
        f"| `{category}` | {metrics.successful_attempts}/"
        f"{metrics.scheduled_attempts} | {metrics.success_rate:.3f} | "
        f"{metrics.median_wall_latency_ms:.3f} |"
        for category, metrics in report.metrics.categories.items()
    )
    rows.extend(
        [
            "",
            "## Attempt matrix",
            "",
            "| Task | Category | Repeat | Task success | Failure |",
            "| --- | --- | ---: | --- | --- |",
        ]
    )
    rows.extend(
        f"| `{attempt.task_id}` | `{attempt.category}` | {attempt.repeat} | "
        f"`{str(attempt.task_success).lower()}` | "
        f"`{attempt.failure_category.value}` |"
        for attempt in report.attempts
    )
    return "\n".join(rows) + "\n"


# 将 canonical JSON 与同源 Markdown 写入 experiment root
def write_baseline_report(
    output_root: Path | str,
    report: BaselineReport,
) -> None:
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    (root / "baseline.json").write_text(render_json(report), encoding="utf-8")
    (root / "baseline.md").write_text(render_markdown(report), encoding="utf-8")
