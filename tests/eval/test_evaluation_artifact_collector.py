from __future__ import annotations

import json
from pathlib import Path

import pytest

from kama_claude.eval.collector import (
    ArtifactCollectionError,
    collect_artifacts,
    snapshot_workspace,
)
from kama_claude.eval.failure import FailureCategory
from kama_claude.eval.models import WorkerResult
from kama_claude.eval.runner import AttemptExecution, prepare_attempt
from kama_claude.eval.task import load_task


# 创建包含公开 fixture 与私有 marker 的最小任务目录
def _task_dir(tmp_path: Path) -> Path:
    task_dir = tmp_path / "task-a"
    workspace = task_dir / "public" / "workspace"
    private = task_dir / "private"
    workspace.mkdir(parents=True)
    private.mkdir()
    (workspace / "input.txt").write_text("before\n", encoding="utf-8")
    (task_dir / "public" / "task.json").write_text(
        json.dumps(
            {
                "id": "task-a",
                "goal": "Update input.txt.",
                "workspace_fixture": "public/workspace",
                "timeout_s": 30.0,
            }
        ),
        encoding="utf-8",
    )
    (private / "grader.json").write_text(
        json.dumps(
            {
                "criteria": [
                    {
                        "id": "hidden-value",
                        "kind": "file_contains",
                        "path": "input.txt",
                        "text": "PRIVATE_EXPECTED_VALUE",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return task_dir


# 构造已有 canonical journal、trace 和 worker result 的完成 attempt
def _execution(tmp_path: Path) -> tuple[AttemptExecution, Path]:
    loaded = load_task(_task_dir(tmp_path))
    prepared = prepare_attempt(loaded, tmp_path / "output")
    run_dir = prepared.runs_dir / prepared.request.run_id
    run_dir.mkdir()
    (run_dir / "events.v2.jsonl").write_text('{"record":"journal"}\n', encoding="utf-8")
    prepared.trace_path.write_text('{"record":"trace"}\n', encoding="utf-8")
    (prepared.workspace / "input.txt").write_text("after\n", encoding="utf-8")
    result = WorkerResult(
        run_id=prepared.request.run_id,
        runtime_status="success",
        result="PRIVATE_FINAL_TEXT",
    )
    execution = AttemptExecution(
        prepared=prepared,
        worker_result=result,
        failure_category=FailureCategory.NONE,
        wall_latency_ms=12.5,
    )
    return execution, loaded.workspace_fixture


# 功能：验证 workspace snapshot 使用稳定相对路径和流式哈希并拒绝任何 symlink
# 设计：先断言普通文件 manifest，再加入指向根外的 symlink，锁定 collector 的证据边界
def test_snapshot_workspace_is_stable_and_rejects_symlinks(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "a.txt").write_text("a", encoding="utf-8")

    manifest = snapshot_workspace(workspace)

    assert [item.path for item in manifest.files] == ["a.txt"]
    assert manifest.total_bytes == 1
    (workspace / "escape").symlink_to(tmp_path)
    with pytest.raises(ArtifactCollectionError, match="symlink"):
        snapshot_workspace(workspace)


# 功能：验证 collector 保存 outcome、canonical journal/trace、双 manifest 和 bounded diff
# 设计：使用完成 attempt 的真实目录布局，断言输出不包含 private criterion 且对象没有评分字段
def test_collect_artifacts_collects_evidence_without_grading(tmp_path: Path) -> None:
    execution, initial_fixture = _execution(tmp_path)

    artifacts = collect_artifacts(execution, initial_fixture)

    assert artifacts.journal_path.read_text(encoding="utf-8") == '{"record":"journal"}\n'
    assert artifacts.trace_path.read_text(encoding="utf-8") == '{"record":"trace"}\n'
    assert artifacts.initial_manifest.files[0].sha256 != artifacts.final_manifest.files[0].sha256
    assert "before" in artifacts.workspace_diff
    assert "after" in artifacts.workspace_diff
    assert "PRIVATE_EXPECTED_VALUE" not in artifacts.workspace_diff
    assert "PRIVATE_FINAL_TEXT" not in artifacts.outcome_path.read_text(encoding="utf-8")
    assert not hasattr(artifacts, "task_success")


# 功能：验证完成 worker 缺少 journal 或 trace 时 collector fail closed
# 设计：逐项删除 canonical 证据并重新收集，防止部分 artifact 被误当成完整 attempt
def test_collect_artifacts_rejects_missing_runtime_evidence(tmp_path: Path) -> None:
    execution, initial_fixture = _execution(tmp_path)
    journal = (
        execution.prepared.runs_dir
        / execution.prepared.request.run_id
        / "events.v2.jsonl"
    )
    journal.unlink()
    with pytest.raises(ArtifactCollectionError, match="journal"):
        collect_artifacts(execution, initial_fixture)

    journal.write_text("{}\n", encoding="utf-8")
    execution.prepared.trace_path.unlink()
    with pytest.raises(ArtifactCollectionError, match="trace"):
        collect_artifacts(execution, initial_fixture)


# 功能：验证 workspace diff 达到 UTF-8 byte cap 后只写一次明确 truncation marker
# 设计：用小 cap 和多字节文本触发边界，断言编码后大小有界且 marker 唯一
def test_collect_artifacts_bounds_workspace_diff_bytes(tmp_path: Path) -> None:
    execution, initial_fixture = _execution(tmp_path)
    (execution.prepared.workspace / "input.txt").write_text("变" * 500, encoding="utf-8")

    artifacts = collect_artifacts(execution, initial_fixture, max_diff_bytes=128)
    encoded = artifacts.workspace_diff.encode("utf-8")

    assert len(encoded) <= 160
    assert artifacts.workspace_diff.count("diff truncated") == 1
