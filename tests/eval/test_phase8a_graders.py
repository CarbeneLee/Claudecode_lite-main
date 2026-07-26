from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

from kama_claude.core.bus.events import RunFinishedEvent, RunStartedEvent
from kama_claude.core.events.journal import EventJournalCoordinator
from kama_claude.eval.graders import GraderExecutionError, grade_rules, grade_trace
from kama_claude.eval.task import load_task


# 创建包含四种客观 criterion 与 hidden pytest 的任务目录
def _task_dir(tmp_path: Path) -> Path:
    task_dir = tmp_path / "task-a"
    public_workspace = task_dir / "public" / "workspace"
    private = task_dir / "private"
    hidden = private / "hidden_tests"
    public_workspace.mkdir(parents=True)
    hidden.mkdir(parents=True)
    (public_workspace / "solution.py").write_text("VALUE = 'old'\n", encoding="utf-8")
    (task_dir / "public" / "task.json").write_text(
        json.dumps(
            {
                "id": "task-a",
                "goal": "Update solution.py.",
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
                    {"id": "exists", "kind": "file_exists", "path": "solution.py"},
                    {
                        "id": "contains",
                        "kind": "file_contains",
                        "path": "solution.py",
                        "text": "VALUE = 'new'",
                    },
                    {
                        "id": "no-todo",
                        "kind": "file_not_contains",
                        "path": "solution.py",
                        "text": "TODO",
                    },
                    {
                        "id": "hidden-tests",
                        "kind": "command_exit",
                        "argv": ["python", "-m", "pytest", "-q", "hidden_tests"],
                        "expected_exit_code": 0,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    (hidden / "test_solution.py").write_text(
        "\n".join(
            (
                "from solution import VALUE",
                "",
                "",
                "# 功能：验证 Agent 最终文件满足隐藏行为要求",
                "# 设计：只导入公开产物，不暴露实现文本或 grader 路径",
                "def test_value() -> None:",
                "    assert VALUE == 'new'",
                "",
            )
        ),
        encoding="utf-8",
    )
    return task_dir


# 构建一个 canonical v2 journal wrapper
def _record(seq: int, event: dict[str, object], run_id: str = "run-a") -> dict[str, object]:
    return {
        "schema_version": 2,
        "event_id": f"evt-{seq}",
        "stream_id": f"run:{run_id}",
        "seq": seq,
        "event": event,
    }


# 将事件序列写为 UTF-8 JSONL journal
def _write_journal(path: Path, events: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(_record(index, event)) + "\n" for index, event in enumerate(events, 1)),
        encoding="utf-8",
    )


# 返回最小合法 run lifecycle 事件
def _valid_events() -> list[dict[str, object]]:
    return [
        {"type": "run.started", "run_id": "run-a", "goal": "goal", "ts": "t1"},
        {"type": "step.started", "run_id": "run-a", "step": 1, "ts": "t2"},
        {
            "type": "tool.call_started",
            "run_id": "run-a",
            "tool_use_id": "tool-1",
            "tool_name": "read_file",
            "params": {"path": "solution.py"},
            "ts": "t3",
        },
        {
            "type": "tool.call_failed",
            "run_id": "run-a",
            "tool_use_id": "tool-1",
            "tool_name": "read_file",
            "error_class": "transient",
            "error_message": "retry",
            "elapsed_ms": 1,
            "attempt": 1,
            "ts": "t4",
        },
        {
            "type": "tool.call_finished",
            "run_id": "run-a",
            "tool_use_id": "tool-1",
            "tool_name": "read_file",
            "elapsed_ms": 2,
            "output": "ok",
            "ts": "t5",
        },
        {"type": "step.finished", "run_id": "run-a", "step": 1, "ts": "t6"},
        {
            "type": "run.finished",
            "run_id": "run-a",
            "status": "success",
            "reason": None,
            "steps": 1,
            "ts": "t7",
        },
    ]


# 用真实 coordinator 写出最小完整 run journal
async def _write_real_journal(path: Path, run_id: str) -> Path:
    coordinator = EventJournalCoordinator()
    await coordinator.register_run(run_id, path, session_id=None)
    await coordinator.handle(
        RunStartedEvent(
            run_id=run_id,
            goal="Verify the journal observer contract.",
            ts="2026-07-26T00:00:00Z",
        )
    )
    await coordinator.handle(
        RunFinishedEvent(
            run_id=run_id,
            status="success",
            reason=None,
            steps=0,
            ts="2026-07-26T00:00:01Z",
        )
    )
    await coordinator.flush_all()
    await coordinator.close()
    return path / "events.v2.jsonl"


# 功能：验证 filesystem 与 hidden command grader 全部在 private grading copy 上得到客观结果
# 设计：比较 final workspace 前后 manifest，并确认 hidden tests 和 pytest cache 从未写入原 workspace
@pytest.mark.asyncio
async def test_rule_graders_use_private_copy_and_preserve_final_workspace(
    tmp_path: Path,
) -> None:
    loaded = load_task(_task_dir(tmp_path))
    final_workspace = tmp_path / "final"
    final_workspace.mkdir()
    (final_workspace / "solution.py").write_text("VALUE = 'new'\n", encoding="utf-8")
    before = sorted(path.relative_to(final_workspace) for path in final_workspace.rglob("*"))

    grade = await grade_rules(loaded, final_workspace, tmp_path / "private-artifacts")
    after = sorted(path.relative_to(final_workspace) for path in final_workspace.rglob("*"))

    assert grade.passed is True
    assert [item.passed for item in grade.criteria] == [True, True, True, True]
    assert before == after
    assert not (final_workspace / "hidden_tests").exists()
    assert not (final_workspace / ".pytest_cache").exists()
    assert grade.command_results_path.is_file()


# 功能：验证 filesystem grader 将不满足的 required criterion 报为失败而不是 grader error
# 设计：提供合法但内容错误的最终文件，区分 task failure 与 grader infrastructure failure
@pytest.mark.asyncio
async def test_rule_graders_report_objective_failure(tmp_path: Path) -> None:
    loaded = load_task(_task_dir(tmp_path))
    final_workspace = tmp_path / "final"
    final_workspace.mkdir()
    (final_workspace / "solution.py").write_text("VALUE = 'wrong'\n", encoding="utf-8")

    grade = await grade_rules(loaded, final_workspace, tmp_path / "private-artifacts")

    assert grade.passed is False
    assert next(item for item in grade.criteria if item.id == "contains").passed is False
    assert next(item for item in grade.criteria if item.id == "hidden-tests").passed is False


# 功能：验证 command grader timeout 会失败 criterion 并清理派生进程组
# 设计：命令派生延迟 marker 子进程后长时间等待，越过 marker 窗口仍无文件即证明 child 被回收
@pytest.mark.asyncio
async def test_command_grader_timeout_reaps_child_process_group(tmp_path: Path) -> None:
    task_dir = _task_dir(tmp_path)
    marker = tmp_path / "command-child-leaked.txt"
    child = (
        "import signal,time; from pathlib import Path; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        f"time.sleep(0.5); Path({str(marker)!r}).write_text('leaked')"
    )
    parent = (
        "import subprocess,sys,time; "
        f"subprocess.Popen([sys.executable,'-c',{child!r}]); "
        "time.sleep(30)"
    )
    (task_dir / "private" / "grader.json").write_text(
        json.dumps(
            {
                "criteria": [
                    {
                        "id": "timeout",
                        "kind": "command_exit",
                        "argv": [sys.executable, "-c", parent],
                        "expected_exit_code": 0,
                        "timeout_s": 0.1,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    loaded = load_task(task_dir)
    final_workspace = tmp_path / "final-timeout"
    final_workspace.mkdir()
    (final_workspace / "solution.py").write_text("VALUE = 'new'\n", encoding="utf-8")

    grade = await grade_rules(loaded, final_workspace, tmp_path / "timeout-artifacts")
    await asyncio.sleep(0.6)

    assert grade.passed is False
    assert not marker.exists()


# 功能：验证 command leader 正常退出后 grader 仍终止持有输出管道的后台子进程
# 设计：child 继承 stdout/stderr 并延迟写 marker，锁定 drain 与进程组 cleanup 属于同一完成屏障
@pytest.mark.asyncio
async def test_command_grader_normal_exit_reaps_pipe_holding_child(tmp_path: Path) -> None:
    task_dir = _task_dir(tmp_path)
    marker = tmp_path / "normal-command-child-leaked.txt"
    child = (
        "import signal,time; from pathlib import Path; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        f"time.sleep(0.5); Path({str(marker)!r}).write_text('leaked')"
    )
    parent = (
        "import subprocess,sys; "
        f"subprocess.Popen([sys.executable,'-c',{child!r}])"
    )
    (task_dir / "private" / "grader.json").write_text(
        json.dumps(
            {
                "criteria": [
                    {
                        "id": "normal-exit",
                        "kind": "command_exit",
                        "argv": [sys.executable, "-c", parent],
                        "expected_exit_code": 0,
                        "timeout_s": 1.0,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    loaded = load_task(task_dir)
    final_workspace = tmp_path / "final-normal-command"
    final_workspace.mkdir()
    (final_workspace / "solution.py").write_text("VALUE = 'new'\n", encoding="utf-8")

    grade = await grade_rules(
        loaded,
        final_workspace,
        tmp_path / "normal-command-artifacts",
    )
    await asyncio.sleep(0.6)

    assert grade.passed is True
    assert not marker.exists()


# 功能：验证 command grader 无法启动 argv 时转换为稳定 grader execution error
# 设计：使用确定不存在的 executable 触发 create_subprocess_exec OSError，防止异常绕过中央分类
@pytest.mark.asyncio
async def test_command_grader_missing_executable_is_grader_error(tmp_path: Path) -> None:
    task_dir = _task_dir(tmp_path)
    (task_dir / "private" / "grader.json").write_text(
        json.dumps(
            {
                "criteria": [
                    {
                        "id": "missing-command",
                        "kind": "command_exit",
                        "argv": ["kama-eval-executable-that-does-not-exist"],
                        "expected_exit_code": 0,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    loaded = load_task(task_dir)
    final_workspace = tmp_path / "final-missing-command"
    final_workspace.mkdir()
    (final_workspace / "solution.py").write_text("VALUE = 'new'\n", encoding="utf-8")

    with pytest.raises(GraderExecutionError, match="grader command could not start"):
        await grade_rules(loaded, final_workspace, tmp_path / "missing-command-artifacts")


# 功能：验证合法 journal 通过 schema、sequence、run/step/tool lifecycle sanity 检查
# 设计：包含一次 failed retry 后成功的 tool lifecycle，证明 grader 不把可恢复失败误判为非法
def test_trace_grader_accepts_valid_lifecycle(tmp_path: Path) -> None:
    journal = tmp_path / "events.v2.jsonl"
    _write_journal(journal, _valid_events())

    grade = grade_trace(journal, expected_run_id="run-a")

    assert grade.passed is True
    assert grade.errors == []


# 功能：验证真实 EventJournalCoordinator 写出的 v2 journal 可被 trace grader 直接接受
# 设计：贯通 production writer 与 observer，不用手写 wrapper，锁定两端 schema contract
@pytest.mark.asyncio
async def test_trace_grader_accepts_real_journal_writer_output(tmp_path: Path) -> None:
    run_id = "producer-contract"
    run_path = tmp_path / run_id
    journal_path = await _write_real_journal(run_path, run_id)

    grade = grade_trace(
        journal_path,
        expected_run_id=run_id,
        expected_terminal_status="success",
    )

    assert grade.passed is True
    assert grade.errors == []


# 功能：验证 observer 严格拒绝缺失、旧版、未来版及非整数 v2 schema_version
# 设计：仅变异合法 wrapper 的版本字段，证明 fail-closed 不依赖 event lifecycle 错误
@pytest.mark.parametrize(
    "schema_version",
    (None, 1, 3, "2", True),
)
@pytest.mark.asyncio
async def test_trace_grader_rejects_unknown_or_malformed_schema_version(
    tmp_path: Path,
    schema_version: object,
) -> None:
    run_id = "version-contract"
    journal = await _write_real_journal(tmp_path / run_id, run_id)
    rows = [
        json.loads(line)
        for line in journal.read_text(encoding="utf-8").splitlines()
    ]
    if schema_version is None:
        rows[0].pop("schema_version")
    else:
        rows[0]["schema_version"] = schema_version
    journal.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    grade = grade_trace(journal, expected_run_id=run_id)

    assert grade.passed is False
    assert grade.errors == ["invalid journal schema"]


# 功能：验证 trace grader 拒绝 sequence gap、无 start 的 finish、非法 retry 与 terminal 后事件
# 设计：逐个破坏一个状态机不变式，断言仅返回结构错误且不出现工具质量评价
@pytest.mark.parametrize(
    "mutate",
    ("sequence_gap", "finish_without_start", "retry_gap", "post_terminal"),
)
def test_trace_grader_rejects_impossible_transitions(
    tmp_path: Path, mutate: str
) -> None:
    journal = tmp_path / f"{mutate}.jsonl"
    events = _valid_events()
    if mutate == "sequence_gap":
        rows = [_record(index, event) for index, event in enumerate(events, 1)]
        rows[2]["seq"] = 4
        journal.write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )
    elif mutate == "finish_without_start":
        events.pop(2)
        _write_journal(journal, events)
    elif mutate == "retry_gap":
        failed = dict(events[3])
        failed["attempt"] = 3
        events.insert(4, failed)
        _write_journal(journal, events)
    else:
        events.append(
            {"type": "step.started", "run_id": "run-a", "step": 2, "ts": "late"}
        )
        _write_journal(journal, events)

    grade = grade_trace(journal, expected_run_id="run-a")

    assert grade.passed is False
    assert grade.errors
    assert all("intelligent" not in error and "unnecessary" not in error for error in grade.errors)


# 功能：验证未知 event schema、错误 stream/run identity 和损坏 JSON 都稳定返回 trace failure
# 设计：覆盖解析层而非状态机层，确保 malformed journal 不会抛出到 report 并伪装成成功
@pytest.mark.parametrize("case", ("unknown_event", "wrong_run", "broken_json"))
def test_trace_grader_rejects_invalid_schema_and_identity(
    tmp_path: Path, case: str
) -> None:
    journal = tmp_path / f"{case}.jsonl"
    if case == "broken_json":
        journal.write_text("{broken\n", encoding="utf-8")
    elif case == "wrong_run":
        journal.write_text(
            json.dumps(
                _record(
                    1,
                    {"type": "run.started", "run_id": "other", "goal": "g", "ts": "t"},
                )
            )
            + "\n",
            encoding="utf-8",
        )
    else:
        journal.write_text(
            json.dumps(
                _record(1, {"type": "agent.was-clever", "run_id": "run-a"})
            )
            + "\n",
            encoding="utf-8",
        )

    grade = grade_trace(journal, expected_run_id="run-a")

    assert grade.passed is False
    assert grade.errors


@pytest.mark.parametrize(
    "case",
    (
        "first_step_not_one",
        "overlapping_steps",
        "tool_outside_step",
        "tool_failure_name_mismatch",
        "tool_name_mismatch",
        "step_count_mismatch",
        "terminal_status_mismatch",
    ),
)
# 功能：验证 trace grader 拒绝 step/tool/terminal 元数据中的结构性矛盾
# 设计：逐一构造当前 schema 可解析但 lifecycle 不可能成立的 journal，不涉及工具质量或计划判断
def test_trace_grader_rejects_structural_contradictions(
    tmp_path: Path, case: str
) -> None:
    journal = tmp_path / f"{case}.jsonl"
    events = _valid_events()
    if case == "first_step_not_one":
        events[1]["step"] = 2
        events[5]["step"] = 2
    elif case == "overlapping_steps":
        events.insert(
            2,
            {"type": "step.started", "run_id": "run-a", "step": 2, "ts": "overlap"},
        )
        events.insert(
            7,
            {"type": "step.finished", "run_id": "run-a", "step": 2, "ts": "overlap-end"},
        )
        events[-1]["steps"] = 2
    elif case == "tool_outside_step":
        tool_events = events[2:5]
        del events[2:5]
        events[1:1] = tool_events
    elif case == "tool_failure_name_mismatch":
        events[3]["tool_name"] = "bash"
    elif case == "tool_name_mismatch":
        events[4]["tool_name"] = "bash"
    elif case == "step_count_mismatch":
        events[-1]["steps"] = 999
    else:
        events[-1]["status"] = "failed"
    _write_journal(journal, events)

    grade = grade_trace(
        journal,
        expected_run_id="run-a",
        expected_terminal_status="success",
    )

    assert grade.passed is False
    assert grade.errors
