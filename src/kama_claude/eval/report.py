from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from kama_claude.eval.failure import FailureCategory
from kama_claude.eval.metrics import BasicMetrics


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class PublicCriterionResult(_StrictModel):
    id: str
    kind: str
    passed: bool


class EvaluationReport(_StrictModel):
    task_id: str
    attempt_id: str
    task_success: bool
    runtime_success: bool
    trace_sanity_passed: bool
    failure_category: FailureCategory
    criteria: list[PublicCriterionResult]
    metrics: BasicMetrics


# 将 report 模型序列化为 canonical、排序且无 private detail 的 JSON
def render_json(report: EvaluationReport) -> str:
    return (
        json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True)
        + "\n"
    )


# 从同一个 report 模型渲染人类可读 Markdown，不读取 private grader artifact
def render_markdown(report: EvaluationReport) -> str:
    rows = [
        "# KamaClaude Evaluation Report",
        "",
        f"- Task: `{report.task_id}`",
        f"- Attempt: `{report.attempt_id}`",
        f"- Task success: `{str(report.task_success).lower()}`",
        f"- Runtime success: `{str(report.runtime_success).lower()}`",
        f"- Trace sanity: `{str(report.trace_sanity_passed).lower()}`",
        f"- Failure category: `{report.failure_category.value}`",
        "",
        "## Basic metrics",
        "",
        f"- Steps: {report.metrics.step_count}",
        f"- Tool calls: {report.metrics.tool_count}",
        f"- Retries: {report.metrics.retry_count}",
        f"- Wall latency: {report.metrics.wall_latency_ms:.3f} ms",
        f"- Input tokens: {report.metrics.token_usage.input_tokens}",
        f"- Output tokens: {report.metrics.token_usage.output_tokens}",
        f"- Cache tokens: {report.metrics.token_usage.cache_tokens}",
        "",
        "## Criteria",
        "",
        "| ID | Kind | Passed |",
        "| --- | --- | --- |",
    ]
    rows.extend(
        f"| `{criterion.id}` | `{criterion.kind}` | "
        f"`{str(criterion.passed).lower()}` |"
        for criterion in report.criteria
    )
    return "\n".join(rows) + "\n"


# 写入单 task/attempt 的 manifest、canonical JSON 和 Markdown report
def write_report(output_root: Path | str, report: EvaluationReport) -> None:
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "artifact_version": 1,
        "task_id": report.task_id,
        "attempt_id": report.attempt_id,
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (root / "report.json").write_text(render_json(report), encoding="utf-8")
    (root / "report.md").write_text(render_markdown(report), encoding="utf-8")
