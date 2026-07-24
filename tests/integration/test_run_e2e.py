"""
End-to-end integration test for the S1 agent pipeline.

Requires a real ANTHROPIC_API_KEY — skipped automatically when absent.
Run explicitly:
    uv run pytest tests/integration/test_run_e2e.py -v
Or with the marker:
    uv run pytest -m integration -v
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

from kama_claude.core.config import KamaConfig
from kama_claude.core.runner import AgentRunner

# Load project .env so ANTHROPIC_API_KEY is available without going through get_config()
load_dotenv(Path(__file__).parent.parent.parent / ".env", override=False)

pytestmark = pytest.mark.integration


# 创建包含固定 magic number 的真实 workspace 文件
@pytest.fixture()
def sample_file(tmp_path: Path) -> Path:
    f = tmp_path / "sample.txt"
    f.write_text(
        "# Test Document\n\nThe magic number mentioned in this file is 7391.\n",
        encoding="utf-8",
    )
    return f


# 功能：验证完整的端到端 agent 链路：调用真实 LLM → 执行 read_file → 成功完成并写入 events.v2.jsonl
# 设计：用真实 provider 与固定 magic number 验证工具闭环，再解析 v2 wrapper 事件序列并证明 legacy 文件不再写入
async def test_run_e2e_reads_file_and_succeeds(
    sample_file: Path,
    tmp_path: Path,
) -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")

    goal = (
        "Use the read_file tool to read the file 'sample.txt' "
        "and report the magic number it mentions."
    )
    runs_dir = tmp_path / "runs"

    config = KamaConfig()
    config.agent.max_steps = 5

    runner = AgentRunner(
        config,
        workspace_root=tmp_path.resolve(),
        runs_dir=runs_dir,
    )
    await runner.run(goal)

    # ── events.v2.jsonl must be the only newly written journal ───────────────
    jsonl_files = list(runs_dir.rglob("events.v2.jsonl"))
    assert len(jsonl_files) == 1, "expected exactly one events.v2.jsonl"
    assert list(runs_dir.rglob("events.jsonl")) == []

    rows = [
        json.loads(line)
        for line in jsonl_files[0].read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert all(row["schema_version"] == 2 for row in rows)
    assert [row["seq"] for row in rows] == list(range(1, len(rows) + 1))
    assert len({row["event_id"] for row in rows}) == len(rows)
    events = [row["event"] for row in rows]
    types = [e["type"] for e in events]

    # ── event sequence assertions (from §6.4) ────────────────────────────────
    assert types[0] == "run.started"
    assert types[-1] == "run.finished"
    assert "step.started" in types
    assert "tool.call_started" in types
    assert "tool.call_finished" in types
    assert "llm.usage" in types

    # ── run completed successfully ────────────────────────────────────────────
    finished = events[-1]
    assert finished["status"] == "success", (
        f"run finished with status={finished['status']!r}, reason={finished.get('reason')!r}"
    )

    # ── read_file was actually invoked ────────────────────────────────────────
    tool_starts = [e for e in events if e["type"] == "tool.call_started"]
    assert any(e["tool_name"] == "read_file" for e in tool_starts), (
        "expected at least one read_file tool call"
    )

    # ── run_id is consistent across the event stream ─────────────────────────
    run_id = events[0]["run_id"]
    assert all(e["run_id"] == run_id for e in events), "run_id must be the same in every event"

    # ── LLM cache stats are present ──────────────────────────────────────────
    usage_events = [e for e in events if e["type"] == "llm.usage"]
    assert len(usage_events) >= 1
    for ue in usage_events:
        assert "input_tokens" in ue
        assert "output_tokens" in ue
        assert "cache_read_input_tokens" in ue
