from __future__ import annotations

import asyncio
import json
import shutil
import sys
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal, NoReturn

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from kama_claude.core.bus.events import Event
from kama_claude.eval.collector import ArtifactCollectionError, snapshot_workspace
from kama_claude.eval.runner import _cleanup_process_group
from kama_claude.eval.schema import (
    CommandExitCriterion,
    FileContainsCriterion,
    FileExistsCriterion,
    FileNotContainsCriterion,
)
from kama_claude.eval.task import LoadedTask

MAX_GRADED_FILE_BYTES = 8 * 1024 * 1024
MAX_COMMAND_OUTPUT_BYTES = 64 * 1024


class GraderExecutionError(RuntimeError):
    pass


@dataclass(frozen=True)
class CriterionGrade:
    id: str
    kind: str
    passed: bool
    detail: str | None = None


@dataclass(frozen=True)
class CommandResult:
    criterion_id: str
    exit_code: int | None
    timed_out: bool
    stdout: str
    stderr: str


@dataclass(frozen=True)
class RuleGrade:
    passed: bool
    criteria: tuple[CriterionGrade, ...]
    command_results_path: Path


@dataclass(frozen=True)
class TraceGrade:
    passed: bool
    errors: list[str]
    events: list[dict[str, Any]]


class _JournalRow(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal[2]
    event_id: str = Field(min_length=1)
    stream_id: str = Field(min_length=1)
    seq: int = Field(gt=0)
    event: dict[str, Any]


_EVENT_ADAPTER: TypeAdapter[Any] = TypeAdapter(Event)


# 将 criterion 相对路径解析到 final workspace 内并拒绝任何 symlink
def _resolve_workspace_path(workspace: Path, relative: str) -> Path:
    root = workspace.resolve(strict=True)
    candidate = root
    for part in Path(relative).parts:
        candidate /= part
        if candidate.is_symlink():
            raise GraderExecutionError("graded path contains a symlink")
    resolved = candidate.resolve(strict=False)
    if not resolved.is_relative_to(root):
        raise GraderExecutionError("graded path escapes workspace")
    return resolved


# 有界读取 grader 文本文件，过大或非 UTF-8 内容视为 grader error
def _read_graded_text(path: Path) -> str:
    try:
        size = path.stat(follow_symlinks=False).st_size
        if size > MAX_GRADED_FILE_BYTES:
            raise GraderExecutionError("graded file exceeds byte limit")
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise GraderExecutionError("graded file is not UTF-8") from exc
    except OSError as exc:
        raise GraderExecutionError("graded file cannot be read") from exc


# 在不修改 workspace 的前提下执行一个 filesystem criterion
def _grade_filesystem(
    criterion: FileExistsCriterion | FileContainsCriterion | FileNotContainsCriterion,
    workspace: Path,
) -> CriterionGrade:
    path = _resolve_workspace_path(workspace, criterion.path)
    if isinstance(criterion, FileExistsCriterion):
        passed = path.is_file()
    elif not path.is_file():
        passed = False
    else:
        content = _read_graded_text(path)
        passed = (
            criterion.text in content
            if isinstance(criterion, FileContainsCriterion)
            else criterion.text not in content
        )
    return CriterionGrade(
        id=criterion.id,
        kind=criterion.kind,
        passed=passed,
        detail=None if passed else "filesystem criterion failed",
    )


# 持续 drain 子进程输出并仅保留前 N 个 UTF-8 bytes
async def _drain_bounded(
    stream: asyncio.StreamReader | None, limit: int = MAX_COMMAND_OUTPUT_BYTES
) -> str:
    if stream is None:
        return ""
    retained = bytearray()
    while chunk := await stream.read(8192):
        if len(retained) < limit:
            retained.extend(chunk[: limit - len(retained)])
    return bytes(retained).decode("utf-8", errors="replace")


# 等待 command leader 退出，不依赖可能被 descendants 持有的 stdout/stderr transport
async def _wait_for_command_leader(process: asyncio.subprocess.Process) -> int:
    while process.returncode is None:
        await asyncio.sleep(0.01)
    return process.returncode


# 执行一个无 shell 的 argv command，并在 timeout 时清理整个进程组
async def _run_command(criterion: CommandExitCriterion, cwd: Path) -> CommandResult:
    argv = list(criterion.argv)
    if argv[0] == "python":
        argv[0] = sys.executable
    try:
        process = await asyncio.create_subprocess_exec(
            *argv,
            cwd=cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
    except (OSError, ValueError) as exc:
        raise GraderExecutionError("grader command could not start") from exc
    stdout_task = asyncio.create_task(_drain_bounded(process.stdout))
    stderr_task = asyncio.create_task(_drain_bounded(process.stderr))
    timed_out = False
    try:
        try:
            async with asyncio.timeout(criterion.timeout_s):
                exit_code = await _wait_for_command_leader(process)
        except TimeoutError:
            timed_out = True
            await _cleanup_process_group(process)
            exit_code = None
        else:
            await _cleanup_process_group(process)
        stdout, stderr = await asyncio.gather(stdout_task, stderr_task)
    except asyncio.CancelledError:
        await _cleanup_process_group(process)
        await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)
        raise
    return CommandResult(
        criterion_id=criterion.id,
        exit_code=exit_code,
        timed_out=timed_out,
        stdout=stdout,
        stderr=stderr,
    )


# 将 dataclass rows 以稳定 JSON 写入 private artifact
def _write_private_rows(
    path: Path, rows: list[CriterionGrade] | list[CommandResult]
) -> None:
    path.write_text(
        json.dumps([asdict(row) for row in rows], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


# 在 worker 退出后创建 private grading copy 并执行全部客观 criteria
async def grade_rules(
    task: LoadedTask,
    final_workspace: Path,
    private_artifact_dir: Path,
) -> RuleGrade:
    try:
        snapshot_workspace(final_workspace)
        private_artifact_dir.mkdir(parents=True, exist_ok=True)
        grading_workspace = private_artifact_dir / "grading-workspace"
        shutil.copytree(final_workspace, grading_workspace, symlinks=True)
        if task.hidden_tests is not None:
            snapshot_workspace(task.hidden_tests)
            hidden_target = grading_workspace / "hidden_tests"
            if hidden_target.exists():
                raise GraderExecutionError("hidden test destination already exists")
            shutil.copytree(task.hidden_tests, hidden_target, symlinks=True)
    except ArtifactCollectionError as exc:
        raise GraderExecutionError("grading workspace is invalid") from exc
    except OSError as exc:
        raise GraderExecutionError("grading workspace cannot be prepared") from exc

    grades: list[CriterionGrade] = []
    commands: list[CommandResult] = []
    for criterion in task.private.criteria:
        if isinstance(criterion, CommandExitCriterion):
            command = await _run_command(criterion, grading_workspace)
            commands.append(command)
            passed = (
                not command.timed_out
                and command.exit_code == criterion.expected_exit_code
            )
            grades.append(
                CriterionGrade(
                    id=criterion.id,
                    kind=criterion.kind,
                    passed=passed,
                    detail=None if passed else "command criterion failed",
                )
            )
        else:
            grades.append(_grade_filesystem(criterion, final_workspace))
    command_results_path = private_artifact_dir / "command-results.json"
    _write_private_rows(command_results_path, list(commands))
    grades_path = private_artifact_dir / "grades.json"
    _write_private_rows(grades_path, list(grades))
    return RuleGrade(
        passed=all(grade.passed for grade in grades),
        criteria=tuple(grades),
        command_results_path=command_results_path,
    )


# 将单条 trace 错误追加到结果，保持错误文本稳定且不包含 event payload
def _trace_error(errors: list[str], message: str) -> None:
    errors.append(message)


# 严格解析 canonical v2 wrapper，并验证 sequence、stream identity 与 event schema
def validate_journal_evidence(
    journal_path: Path,
    *,
    expected_run_id: str,
) -> TraceGrade:
    errors: list[str] = []
    events: list[dict[str, Any]] = []
    rows: list[_JournalRow] = []
    try:
        lines = journal_path.read_text(encoding="utf-8").splitlines()
        if not lines:
            raise ValueError("empty journal")
        for line in lines:
            rows.append(_JournalRow.model_validate_json(line))
    except (OSError, UnicodeDecodeError, ValidationError, ValueError):
        return TraceGrade(passed=False, errors=["invalid journal schema"], events=[])

    seen_event_ids: set[str] = set()
    expected_stream = f"run:{expected_run_id}"
    for expected_seq, row in enumerate(rows, 1):
        if row.seq != expected_seq:
            _trace_error(errors, "journal sequence is not contiguous")
        if row.stream_id != expected_stream:
            _trace_error(errors, "journal stream identity mismatch")
        if row.event_id in seen_event_ids:
            _trace_error(errors, "duplicate journal event identity")
        seen_event_ids.add(row.event_id)
        try:
            event = _EVENT_ADAPTER.validate_python(row.event)
            events.append(event.model_dump(mode="json"))
        except ValidationError:
            _trace_error(errors, "unknown or invalid event schema")

    return TraceGrade(passed=not errors, errors=errors, events=events)


class _CompletionPolicy(StrEnum):
    COMPLETE = "complete"
    TIMEOUT_PREFIX = "timeout_prefix"


@dataclass
class _ToolLifecycle:
    step: int
    tool_use_id: str
    name: str
    last_failure_attempt: int = 0
    has_failed_outcome: bool = False
    success: bool = False


@dataclass
class _LifecycleState:
    run_started: bool = False
    run_finished: bool = False
    current_step: int | None = None
    completed_steps: int = 0
    tools: dict[tuple[int, str], _ToolLifecycle] = field(default_factory=dict)
    open_subagents: set[str] = field(default_factory=set)


class _LifecycleViolation(RuntimeError):
    pass


# 以首个稳定错误终止 lifecycle reduction，避免残留状态产生级联诊断
def _lifecycle_violation(message: str) -> NoReturn:
    raise _LifecycleViolation(message)


# 返回当前 step 拥有的指定 tool state，不把 tool ID解释为 run-wide唯一
def _current_tool(
    state: _LifecycleState,
    event: dict[str, Any],
    *,
    missing_message: str,
) -> _ToolLifecycle:
    if state.current_step is None:
        _lifecycle_violation("tool event appears outside an open step")
    tool_id = str(event["tool_use_id"])
    tool = state.tools.get((state.current_step, tool_id))
    if tool is None:
        _lifecycle_violation(missing_message)
    return tool


# 用共享状态机验证完整 run 或 timeout prefix 的全部 lifecycle transition
def _reduce_lifecycle(
    events: list[dict[str, Any]],
    *,
    expected_run_id: str,
    policy: _CompletionPolicy,
    expected_terminal_status: str | None,
) -> list[str]:
    state = _LifecycleState()
    try:
        for index, event in enumerate(events):
            event_type = str(event["type"])
            event_run_id = event.get("run_id")
            if state.run_finished:
                _lifecycle_violation("event appears after terminal run event")
            if event_type == "run.started":
                if (
                    index != 0
                    or state.run_started
                    or event_run_id != expected_run_id
                ):
                    _lifecycle_violation("invalid run start transition")
                state.run_started = True
                continue
            if not state.run_started:
                _lifecycle_violation("event appears before run start")
            if event_type.startswith("subagent."):
                if event.get("parent_run_id") != expected_run_id:
                    _lifecycle_violation("subagent parent identity mismatch")
            elif event_run_id is not None and event_run_id != expected_run_id:
                _lifecycle_violation("event run identity mismatch")

            if event_type == "run.finished":
                if policy is _CompletionPolicy.TIMEOUT_PREFIX:
                    _lifecycle_violation(
                        "timeout prefix contains terminal run event"
                    )
                if state.current_step is not None:
                    _lifecycle_violation("run finished with open step")
                if int(event["steps"]) != state.completed_steps:
                    _lifecycle_violation("run terminal step count mismatch")
                if (
                    expected_terminal_status is not None
                    and str(event["status"]) != expected_terminal_status
                ):
                    _lifecycle_violation("run terminal status mismatch")
                if state.open_subagents:
                    _lifecycle_violation("run finished with open subagent")
                state.run_finished = True
            elif event_type == "step.started":
                step = int(event["step"])
                if state.current_step is not None:
                    _lifecycle_violation("overlapping step start")
                if step != state.completed_steps + 1:
                    _lifecycle_violation("step sequence is not contiguous")
                state.current_step = step
            elif event_type == "step.finished":
                step = int(event["step"])
                if state.current_step != step:
                    _lifecycle_violation("step finish without start")
                if any(
                    not tool.success and not tool.has_failed_outcome
                    for tool in state.tools.values()
                ):
                    _lifecycle_violation("step finished with active tool")
                state.tools.clear()
                state.current_step = None
                state.completed_steps += 1
            elif event_type == "tool.call_started":
                if state.current_step is None:
                    _lifecycle_violation(
                        "tool event appears outside an open step"
                    )
                tool_id = str(event["tool_use_id"])
                key = (state.current_step, tool_id)
                if key in state.tools:
                    _lifecycle_violation("duplicate tool start")
                state.tools[key] = _ToolLifecycle(
                    step=state.current_step,
                    tool_use_id=tool_id,
                    name=str(event["tool_name"]),
                )
            elif event_type == "tool.call_failed":
                tool = _current_tool(
                    state,
                    event,
                    missing_message="tool failure without active start",
                )
                if tool.success:
                    _lifecycle_violation("tool failure without active start")
                if str(event["tool_name"]) != tool.name:
                    _lifecycle_violation("tool lifecycle name mismatch")
                attempt = int(event["attempt"])
                if (
                    attempt <= 0
                    or attempt != tool.last_failure_attempt + 1
                ):
                    _lifecycle_violation(
                        "tool retry attempt is not contiguous"
                    )
                tool.last_failure_attempt = attempt
                tool.has_failed_outcome = True
            elif event_type == "tool.call_finished":
                tool = _current_tool(
                    state,
                    event,
                    missing_message="tool finish without active start",
                )
                if tool.success:
                    _lifecycle_violation("tool finish without active start")
                if str(event["tool_name"]) != tool.name:
                    _lifecycle_violation("tool lifecycle name mismatch")
                tool.success = True
            elif event_type == "subagent.started":
                child_run_id = str(event["run_id"])
                if child_run_id in state.open_subagents:
                    _lifecycle_violation("duplicate subagent start")
                state.open_subagents.add(child_run_id)
            elif event_type == "subagent.finished":
                child_run_id = str(event["run_id"])
                if child_run_id not in state.open_subagents:
                    _lifecycle_violation("subagent finish without start")
                state.open_subagents.remove(child_run_id)
    except _LifecycleViolation as exc:
        return [str(exc)]

    if not state.run_started:
        return [
            "missing run start"
            if policy is _CompletionPolicy.TIMEOUT_PREFIX
            else "run start is missing"
        ]
    if policy is _CompletionPolicy.COMPLETE and not state.run_finished:
        return ["run terminal event is missing"]
    return []


# 验证 timeout 截断前缀不存在已发生的非法 lifecycle transition
def grade_timeout_trace_prefix(
    journal_path: Path,
    *,
    expected_run_id: str,
) -> TraceGrade:
    parsed = validate_journal_evidence(
        journal_path,
        expected_run_id=expected_run_id,
    )
    if not parsed.passed:
        return parsed
    errors = _reduce_lifecycle(
        parsed.events,
        expected_run_id=expected_run_id,
        policy=_CompletionPolicy.TIMEOUT_PREFIX,
        expected_terminal_status=None,
    )
    return TraceGrade(
        passed=not errors,
        errors=errors,
        events=parsed.events,
    )


# 验证 canonical journal 的 schema、sequence 和 run/step/tool/subagent lifecycle
def grade_trace(
    journal_path: Path,
    *,
    expected_run_id: str,
    expected_terminal_status: str | None = None,
) -> TraceGrade:
    parsed = validate_journal_evidence(
        journal_path,
        expected_run_id=expected_run_id,
    )
    if not parsed.passed:
        return parsed
    errors = _reduce_lifecycle(
        parsed.events,
        expected_run_id=expected_run_id,
        policy=_CompletionPolicy.COMPLETE,
        expected_terminal_status=expected_terminal_status,
    )
    return TraceGrade(passed=not errors, errors=errors, events=parsed.events)
