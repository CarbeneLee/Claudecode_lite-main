from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

from kama_claude.core.bus.events import RunFinishedEvent, RunStartedEvent
from kama_claude.core.events.journal import EventJournalCoordinator
from kama_claude.eval.graders import (
    GraderExecutionError,
    grade_rules,
    grade_timeout_trace_prefix,
    grade_trace,
)
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


# 构造不含 terminal 的最小 run prefix
def _run_prefix() -> list[dict[str, object]]:
    return [
        {"type": "run.started", "run_id": "run-a", "goal": "goal", "ts": "t1"},
    ]


# 构造指定 step 的 started 事件
def _step_started(step: int, ts: str) -> dict[str, object]:
    return {
        "type": "step.started",
        "run_id": "run-a",
        "step": step,
        "ts": ts,
    }


# 构造指定 step 的 finished 事件
def _step_finished(step: int, ts: str) -> dict[str, object]:
    return {
        "type": "step.finished",
        "run_id": "run-a",
        "step": step,
        "ts": ts,
    }


# 构造不泄露真实参数的 tool started 事件
def _tool_started(
    tool_id: str,
    ts: str,
    *,
    name: str = "fake_tool",
) -> dict[str, object]:
    return {
        "type": "tool.call_started",
        "run_id": "run-a",
        "tool_use_id": tool_id,
        "tool_name": name,
        "params": {},
        "ts": ts,
    }


# 构造指定连续 attempt 的 tool failed 事件
def _tool_failed(
    tool_id: str,
    attempt: int,
    ts: str,
    *,
    name: str = "fake_tool",
) -> dict[str, object]:
    return {
        "type": "tool.call_failed",
        "run_id": "run-a",
        "tool_use_id": tool_id,
        "tool_name": name,
        "error_class": "command_failed",
        "error_message": "expected failure",
        "elapsed_ms": attempt,
        "attempt": attempt,
        "ts": ts,
    }


# 构造不含虚构 attempt 字段的 tool finished 事件
def _tool_finished(
    tool_id: str,
    ts: str,
    *,
    name: str = "fake_tool",
) -> dict[str, object]:
    return {
        "type": "tool.call_finished",
        "run_id": "run-a",
        "tool_use_id": tool_id,
        "tool_name": name,
        "elapsed_ms": 1,
        "output": "ok",
        "ts": ts,
    }


# 构造 complete policy 所需的唯一 run terminal 事件
def _run_finished(steps: int, ts: str) -> dict[str, object]:
    return {
        "type": "run.finished",
        "run_id": "run-a",
        "status": "success",
        "reason": None,
        "steps": steps,
        "ts": ts,
    }


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


# 功能：验证 final tool failure 可在 step barrier 关闭并允许进入后续 step
# 设计：复现真实 baseline 的 failed→step.finished→next step 前缀，直接锁定 timeout observer 根因
def test_timeout_trace_accepts_final_failure_then_later_step(tmp_path: Path) -> None:
    journal = tmp_path / "final-failure-later-step.jsonl"
    events = [
        *_run_prefix(),
        _step_started(1, "t2"),
        _tool_started("tool-a", "t3"),
        _tool_failed("tool-a", 1, "t4"),
        _step_finished(1, "t5"),
        _step_started(2, "t6"),
    ]
    _write_journal(journal, events)

    grade = grade_timeout_trace_prefix(journal, expected_run_id="run-a")

    assert grade.passed is True
    assert grade.errors == []


# 功能：验证连续多次 failure 在 step barrier 处可收敛为 final failure
# 设计：使用 attempts 1、2、3 且没有 success finish，区分 retry 记录与最终 barrier outcome
def test_timeout_trace_accepts_multiple_failures_at_step_barrier(tmp_path: Path) -> None:
    journal = tmp_path / "multiple-failures.jsonl"
    events = [
        *_run_prefix(),
        _step_started(1, "t2"),
        _tool_started("tool-a", "t3"),
        _tool_failed("tool-a", 1, "t4"),
        _tool_failed("tool-a", 2, "t5"),
        _tool_failed("tool-a", 3, "t6"),
        _step_finished(1, "t7"),
    ]
    _write_journal(journal, events)

    grade = grade_timeout_trace_prefix(journal, expected_run_id="run-a")

    assert grade.passed is True
    assert grade.errors == []


@pytest.mark.parametrize("retry_succeeds", (False, True))
# 功能：验证同一步多个 tool 可以交错，且 failure 与 success outcome 独立归属
# 设计：A/B 都先 started，再交错 A failed 与 B finished；参数化 A 最终失败或随后成功两条合法路径
def test_timeout_trace_accepts_same_step_interleaved_tools(
    tmp_path: Path,
    retry_succeeds: bool,
) -> None:
    journal = tmp_path / f"interleaved-{retry_succeeds}.jsonl"
    events = [
        *_run_prefix(),
        _step_started(1, "t2"),
        _tool_started("tool-a", "t3", name="tool_a"),
        _tool_started("tool-b", "t4", name="tool_b"),
        _tool_failed("tool-a", 1, "t5", name="tool_a"),
        _tool_finished("tool-b", "t6", name="tool_b"),
    ]
    if retry_succeeds:
        events.append(_tool_finished("tool-a", "t7", name="tool_a"))
    events.append(_step_finished(1, "t8"))
    _write_journal(journal, events)

    grade = grade_timeout_trace_prefix(journal, expected_run_id="run-a")

    assert grade.passed is True
    assert grade.errors == []


@pytest.mark.parametrize(
    "case",
    (
        "missing_outcome",
        "overlapping_step",
        "event_after_barrier",
        "finished_before_start",
        "failed_before_start",
        "failure_after_finish",
    ),
)
# 功能：验证共享 lifecycle reducer 继续拒绝真实不可能 transition
# 设计：每个 case 只破坏一条 ownership/barrier 规则，防止 final-failure 修复放宽整体 strictness
def test_timeout_trace_rejects_invalid_tool_and_step_transitions(
    tmp_path: Path,
    case: str,
) -> None:
    journal = tmp_path / f"invalid-{case}.jsonl"
    events = [*_run_prefix(), _step_started(1, "t2")]
    if case == "missing_outcome":
        events.extend((_tool_started("tool-a", "t3"), _step_finished(1, "t4")))
    elif case == "overlapping_step":
        events.append(_step_started(2, "t3"))
    elif case == "event_after_barrier":
        events.extend(
            (
                _tool_started("tool-a", "t3"),
                _tool_finished("tool-a", "t4"),
                _step_finished(1, "t5"),
                _tool_finished("tool-a", "t6"),
            )
        )
    elif case == "finished_before_start":
        events.append(_tool_finished("tool-a", "t3"))
    elif case == "failed_before_start":
        events.append(_tool_failed("tool-a", 1, "t3"))
    else:
        events.extend(
            (
                _tool_started("tool-a", "t3"),
                _tool_finished("tool-a", "t4"),
                _tool_failed("tool-a", 1, "t5"),
            )
        )
    _write_journal(journal, events)

    grade = grade_timeout_trace_prefix(journal, expected_run_id="run-a")

    assert grade.passed is False
    assert grade.errors


@pytest.mark.parametrize(
    "attempts",
    (
        (0,),
        (1, 1),
        (1, 3),
        (1, 2, 1),
    ),
)
# 功能：验证 failure attempt 必须从1开始并严格连续递增
# 设计：覆盖零值、重复、跳号与回退，直接杀死使用非严格比较或仅检查正数的实现
def test_timeout_trace_rejects_invalid_failure_attempt_sequence(
    tmp_path: Path,
    attempts: tuple[int, ...],
) -> None:
    journal = tmp_path / f"attempts-{'-'.join(map(str, attempts))}.jsonl"
    events = [
        *_run_prefix(),
        _step_started(1, "t2"),
        _tool_started("tool-a", "t3"),
        *[
            _tool_failed("tool-a", attempt, f"t{index + 4}")
            for index, attempt in enumerate(attempts)
        ],
    ]
    _write_journal(journal, events)

    grade = grade_timeout_trace_prefix(journal, expected_run_id="run-a")

    assert grade.passed is False
    assert grade.errors


@pytest.mark.parametrize("open_tool", (False, True))
# 功能：验证 timeout EOF 可停在 open step 或尚无 outcome 的 open tool
# 设计：只改变是否发布 tool.started，证明 partial policy不把截断时刻误当作已完成 barrier
def test_timeout_trace_allows_open_lifecycle_at_eof(
    tmp_path: Path,
    open_tool: bool,
) -> None:
    journal = tmp_path / f"open-eof-{open_tool}.jsonl"
    events = [*_run_prefix(), _step_started(1, "t2")]
    if open_tool:
        events.append(_tool_started("tool-a", "t3"))
    _write_journal(journal, events)

    grade = grade_timeout_trace_prefix(journal, expected_run_id="run-a")

    assert grade.passed is True
    assert grade.errors == []


# 功能：验证 complete 与 timeout 对共享合法 prefix 使用相同 transition 语义
# 设计：prefix含 final failure barrier；timeout直接评分，complete仅追加 terminal，二者都必须通过
def test_complete_and_timeout_graders_share_lifecycle_semantics(tmp_path: Path) -> None:
    prefix = [
        *_run_prefix(),
        _step_started(1, "t2"),
        _tool_started("tool-a", "t3"),
        _tool_failed("tool-a", 1, "t4"),
        _step_finished(1, "t5"),
    ]
    timeout_journal = tmp_path / "shared-timeout.jsonl"
    complete_journal = tmp_path / "shared-complete.jsonl"
    _write_journal(timeout_journal, prefix)
    _write_journal(complete_journal, [*prefix, _run_finished(1, "t6")])

    timeout_grade = grade_timeout_trace_prefix(
        timeout_journal,
        expected_run_id="run-a",
    )
    complete_grade = grade_trace(
        complete_journal,
        expected_run_id="run-a",
        expected_terminal_status="success",
    )

    assert timeout_grade.passed is True
    assert timeout_grade.errors == []
    assert complete_grade.passed is True
    assert complete_grade.errors == []


# 功能：验证 complete 缺 terminal 与 timeout 包含 terminal 都继续 fail closed
# 设计：对同一零 step run分别遗漏和注入 run.finished，锁定两种 completion policy 的唯一差异
def test_completion_policies_enforce_terminal_contract(tmp_path: Path) -> None:
    prefix = _run_prefix()
    complete_journal = tmp_path / "complete-missing-terminal.jsonl"
    timeout_journal = tmp_path / "timeout-with-terminal.jsonl"
    _write_journal(complete_journal, prefix)
    _write_journal(timeout_journal, [*prefix, _run_finished(0, "t2")])

    complete_grade = grade_trace(complete_journal, expected_run_id="run-a")
    timeout_grade = grade_timeout_trace_prefix(
        timeout_journal,
        expected_run_id="run-a",
    )

    assert complete_grade.passed is False
    assert "run terminal event is missing" in complete_grade.errors
    assert timeout_grade.passed is False
    assert "timeout prefix contains terminal run event" in timeout_grade.errors


# 功能：验证 timeout subagent lifecycle 使用真实事件合同中的 run_id 作为 child identity
# 设计：写入合法 parent_run_id与child run_id配对，防止 observer继续读取不存在的child_run_id字段
def test_timeout_trace_uses_subagent_run_id_as_child_identity(tmp_path: Path) -> None:
    journal = tmp_path / "subagent-prefix.jsonl"
    events = [
        *_run_prefix(),
        {
            "type": "subagent.started",
            "run_id": "child-a",
            "parent_run_id": "run-a",
            "description": "trusted child",
            "ts": "t2",
        },
        {
            "type": "subagent.finished",
            "run_id": "child-a",
            "parent_run_id": "run-a",
            "status": "success",
            "ts": "t3",
        },
    ]
    _write_journal(journal, events)

    grade = grade_timeout_trace_prefix(journal, expected_run_id="run-a")

    assert grade.passed is True
    assert grade.errors == []


# 功能：验证相同 tool_use_id 可在不同 step重新开始而不依赖 run-wide唯一性
# 设计：两个 step各自完整拥有同一ID，锁定 observer只按(step,tool_use_id)解释 ownership
def test_trace_allows_tool_id_reuse_across_steps(tmp_path: Path) -> None:
    events = [*_run_prefix()]
    for step in (1, 2):
        events.extend(
            (
                _step_started(step, f"s{step}"),
                _tool_started("reused-tool", f"start{step}"),
                _tool_finished("reused-tool", f"finish{step}"),
                _step_finished(step, f"done{step}"),
            )
        )
    events.append(_run_finished(2, "terminal"))
    journal = tmp_path / "reuse-across-steps.jsonl"
    _write_journal(journal, events)

    grade = grade_trace(journal, expected_run_id="run-a")

    assert grade.passed is True
    assert grade.errors == []
