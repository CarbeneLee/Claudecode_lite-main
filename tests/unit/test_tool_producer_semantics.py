from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import pytest
from pydantic import BaseModel

import kama_claude.core.sandbox.executors as exec_mod
from kama_claude.core.bus.events import (
    ToolCallFailedEvent,
    ToolCallFinishedEvent,
    ToolCallStartedEvent,
)
from kama_claude.core.events.bus import EventBus
from kama_claude.core.llm.types import ToolCallBlock
from kama_claude.core.sandbox.executors import HostExecutor
from kama_claude.core.session.store import SessionStore
from kama_claude.core.tools.base import BaseTool, ToolResult
from kama_claude.core.tools.builtin.bash import BashTool
from kama_claude.core.tools.builtin.note_save import NoteSaveTool
from kama_claude.core.tools.builtin.write_file import WriteFileTool
from kama_claude.core.tools.invocation import invoke_tool
from kama_claude.core.tools.registry import ToolRegistry
from kama_claude.core.workspace.policy import WorkspaceAccessPolicy
from kama_claude.core.workspace.resolver import WorkspacePathResolver

_ONE_MIB = 1024 * 1024


class _FakeProcess:
    # 初始化可重复的 communicate 结果与资源清理计数
    def __init__(
        self,
        errors: list[BaseException | None],
        *,
        returncode: int | None = None,
    ) -> None:
        self._errors = errors
        self.kill_calls = 0
        self.communicate_calls = 0
        self.returncode = returncode

    # 记录 kill 次数并模拟进程进入被信号终止状态
    def kill(self) -> None:
        self.kill_calls += 1

    # 按配置依次抛出异常或返回空输出，成功 reap 后设置 returncode
    async def communicate(self) -> tuple[bytes, None]:
        self.communicate_calls += 1
        index = self.communicate_calls - 1
        error = self._errors[index] if index < len(self._errors) else None
        if error is not None:
            raise error
        if self.returncode is None:
            self.returncode = -9 if self.kill_calls else 0
        return b"", None


# 功能：构造绑定真实 workspace resolver 和 policy 的 write_file 工具
# 设计：复用生产依赖而非模拟写文件逻辑，使边界测试覆盖真实副作用顺序
def _write_tool(workspace: Path) -> WriteFileTool:
    return WriteFileTool(
        WorkspacePathResolver(workspace),
        WorkspaceAccessPolicy(workspace),
    )


# 功能：通过真实 invoke_tool 执行任意工具并收集完整生命周期事件
# 设计：注册真实工具和 EventBus，联合断言重试次数与 started/failed/finished 语义
async def _invoke(
    tool: BaseTool,
    params: dict[str, object],
) -> tuple[ToolResult, list[BaseModel]]:
    registry = ToolRegistry()
    registry.register(tool)
    bus = EventBus()
    events: list[BaseModel] = []

    # 收集当前 logical invocation 发布的全部事件
    async def _collect(event: BaseModel) -> None:
        events.append(event)

    bus.subscribe(_collect)
    result = await invoke_tool(
        registry,
        ToolCallBlock(id="tool-1", name=tool.name, input=params),
        bus,
        run_id="run-1",
    )
    return result, events


# 功能：验证 write_file 恰好 1 MiB 的 UTF-8 内容仍能成功写入
# 设计：使用单字节 ASCII 精确命中上界，断言返回成功且磁盘字节数未被截断
async def test_write_file_accepts_exactly_one_mib(tmp_path: Path) -> None:
    result = await _write_tool(tmp_path).invoke(
        {"path": "exact.bin", "content": "x" * _ONE_MIB}
    )

    assert not result.is_error
    assert (tmp_path / "exact.bin").stat().st_size == _ONE_MIB


# 功能：验证 write_file 超过 1 MiB 一个字节时返回 invalid_input 且不创建目标
# 设计：使用不存在的目标冻结校验先于 mkdir/write 的副作用顺序
async def test_write_file_rejects_one_mib_plus_one_without_creating(
    tmp_path: Path,
) -> None:
    result = await _write_tool(tmp_path).invoke(
        {"path": "new/oversize.bin", "content": "x" * (_ONE_MIB + 1)}
    )

    assert result.is_error
    assert result.error_type == "invalid_input"
    assert not (tmp_path / "new").exists()


# 功能：验证超限写入不会覆盖已经存在的目标文件
# 设计：预置哨兵内容后提交超限 payload，断言错误类型和原文件字节均保持不变
async def test_write_file_oversize_does_not_overwrite_existing_file(
    tmp_path: Path,
) -> None:
    target = tmp_path / "existing.txt"
    target.write_text("sentinel", encoding="utf-8")

    result = await _write_tool(tmp_path).invoke(
        {"path": "existing.txt", "content": "x" * (_ONE_MIB + 1)}
    )

    assert result.error_type == "invalid_input"
    assert target.read_text(encoding="utf-8") == "sentinel"


# 功能：验证 write_file 按 UTF-8 字节数而不是 Python 字符数执行上限
# 设计：选择三字节汉字使字符数低于上限但编码后超限，断言拒绝且不创建文件
async def test_write_file_counts_multibyte_utf8_bytes(tmp_path: Path) -> None:
    content = "界" * (_ONE_MIB // 3 + 1)
    assert len(content) < _ONE_MIB
    assert len(content.encode("utf-8")) > _ONE_MIB

    result = await _write_tool(tmp_path).invoke(
        {"path": "multibyte.txt", "content": content}
    )

    assert result.error_type == "invalid_input"
    assert not (tmp_path / "multibyte.txt").exists()


# 功能：验证 write_file 的 invalid_input 经真实 invocation 只产生一个失败 attempt
# 设计：用真实工具和事件总线锁定一个 started、一个 failed(attempt=1)、零 finished
async def test_write_file_invalid_input_is_not_retried(tmp_path: Path) -> None:
    result, events = await _invoke(
        _write_tool(tmp_path),
        {"path": "oversize.txt", "content": "x" * (_ONE_MIB + 1)},
    )

    assert result.error_type == "invalid_input"
    assert len([e for e in events if isinstance(e, ToolCallStartedEvent)]) == 1
    assert [e.attempt for e in events if isinstance(e, ToolCallFailedEvent)] == [1]
    assert not any(isinstance(e, ToolCallFinishedEvent) for e in events)


# 功能：验证 note_save 的空白业务输入通过 invocation 后仍不重试
# 设计：使用真实 SessionStore 并联合断言事件、最终类型和 notes 无副作用
async def test_note_save_invalid_input_is_not_retried(tmp_path: Path) -> None:
    store = SessionStore(tmp_path)
    result, events = await _invoke(
        NoteSaveTool(store, "session-1", "run-1"),
        {"content": " \n\t "},
    )

    assert result.error_type == "invalid_input"
    assert [e.attempt for e in events if isinstance(e, ToolCallFailedEvent)] == [1]
    assert store.read_notes("session-1") == ""


# 功能：验证 Bash 的两个常见非零退出码稳定映射为 command_failed
# 设计：参数化普通失败和 command-not-found 等价退出码，同时冻结退出码对 Agent 可见
@pytest.mark.parametrize("returncode", [1, 127])
async def test_bash_nonzero_exit_is_command_failed(
    tmp_path: Path,
    returncode: int,
) -> None:
    result = await BashTool(HostExecutor(), workspace_root=tmp_path).invoke({"command": f"exit {returncode}"})

    assert result.error_type == "command_failed"
    assert f"[exit {returncode}]" in result.content


# 功能：验证 Bash 非零退出仍向 Agent 展示退出码和合并后的 stdout/stderr
# 设计：同一命令写两个输出流后 exit 7，冻结 command_failed 的可诊断内容契约
async def test_bash_command_failed_keeps_combined_output(tmp_path: Path) -> None:
    command = (
        "printf 'stdout-visible\\n'; "
        "printf 'stderr-visible\\n' >&2; "
        "exit 7"
    )

    result = await BashTool(HostExecutor(), workspace_root=tmp_path).invoke({"command": command})

    assert result.error_type == "command_failed"
    assert "[exit 7]" in result.content
    assert "stdout-visible" in result.content
    assert "stderr-visible" in result.content


# 功能：验证 Bash direct invoke 不吞掉 subprocess 创建阶段的未知 RuntimeError
# 设计：注入含敏感路径和 token 的异常，期望原样抛出以交由上层边界分类
async def test_bash_direct_invoke_bubbles_unknown_subprocess_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "/private/workspace/.env token=bash-secret"

    # 模拟 subprocess 创建阶段的未知内部故障
    async def _explode(*args: object, **kwargs: object) -> object:
        raise RuntimeError(secret)

    monkeypatch.setattr(exec_mod.asyncio, "create_subprocess_shell", _explode)

    with pytest.raises(RuntimeError, match="bash-secret"):
        await BashTool(HostExecutor(), workspace_root=tmp_path).invoke({"command": "echo hi"})


# 功能：验证 Bash 未知异常在中央边界安全化、记 traceback 且只执行一次
# 设计：计数 subprocess 创建调用并检查结果/事件不含秘密，而 caplog 保留诊断堆栈
async def test_bash_unknown_error_is_safely_classified_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    secret = "/private/workspace/.env token=bash-secret"
    calls = 0

    # 模拟只应执行一次的未知 subprocess 故障
    async def _explode(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        raise RuntimeError(secret)

    monkeypatch.setattr(exec_mod.asyncio, "create_subprocess_shell", _explode)

    with caplog.at_level(logging.ERROR, logger="kama_claude.core.tools.errors"):
        result, events = await _invoke(BashTool(HostExecutor(), workspace_root=tmp_path), {"command": "echo hi"})

    failed = [e for e in events if isinstance(e, ToolCallFailedEvent)]
    assert calls == 1
    assert result.error_type == "execution_error"
    assert result.content == "tool execution failed"
    assert secret not in result.content
    assert [event.attempt for event in failed] == [1]
    assert all(secret not in event.error_message for event in failed)
    assert "Traceback" in caplog.text
    assert secret in caplog.text


# 功能：验证 Bash 在 subprocess 创建阶段取消时无进程需要清理
# 设计：spawn 直接抛出固定 CancelledError，断言异常身份不变且未返回 fake process
async def test_bash_spawn_cancelled_error_propagates_without_kill(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = _FakeProcess([])
    cancellation = asyncio.CancelledError()

    # 模拟任务在子进程创建阶段被上层取消
    async def _cancel(*args: object, **kwargs: object) -> object:
        raise cancellation

    monkeypatch.setattr(exec_mod.asyncio, "create_subprocess_shell", _cancel)

    with pytest.raises(asyncio.CancelledError) as exc_info:
        await BashTool(HostExecutor(), workspace_root=tmp_path).invoke({"command": "echo hi"})

    assert exc_info.value is cancellation
    assert process.kill_calls == 0
    assert process.communicate_calls == 0


# 功能：验证 communicate 阶段取消会 kill/reap 后原样传播且不发布失败事件
# 设计：fake process 首次 communicate 取消、第二次完成 reap，并通过真实 invocation 检查事件
async def test_bash_communicate_cancelled_error_cleans_up_and_propagates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cancellation = asyncio.CancelledError()
    process = _FakeProcess([cancellation, None])

    # 将已成功创建的 fake process 交给 BashTool
    async def _spawn(*args: object, **kwargs: object) -> object:
        return process

    monkeypatch.setattr(exec_mod.asyncio, "create_subprocess_shell", _spawn)
    registry = ToolRegistry()
    registry.register(BashTool(HostExecutor(), workspace_root=tmp_path))
    bus = EventBus()
    events: list[BaseModel] = []

    # 收集 cancellation 路径发布的生命周期事件
    async def _collect(event: BaseModel) -> None:
        events.append(event)

    bus.subscribe(_collect)
    with pytest.raises(asyncio.CancelledError) as exc_info:
        await invoke_tool(
            registry,
            ToolCallBlock(id="tool-cancel", name="bash", input={"command": "echo hi"}),
            bus,
            run_id="run-1",
        )

    assert exc_info.value is cancellation
    assert process.kill_calls == 1
    assert process.communicate_calls == 2
    assert process.returncode == -9
    assert [type(event) for event in events] == [ToolCallStartedEvent]


# 功能：验证 communicate 未知异常清理后 direct 仍抛原异常，中央边界安全分类一次
# 设计：两个 fake process 分别覆盖 direct/invocation，并锁定 kill/reap、秘密净化与 attempt
async def test_bash_communicate_runtime_error_cleans_up_and_is_safely_classified(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "/private/workspace/.env token=communicate-secret"
    direct_error = RuntimeError(secret)
    invoked_error = RuntimeError(secret)
    processes = [
        _FakeProcess([direct_error, None]),
        _FakeProcess([invoked_error, None]),
    ]

    # 每次 spawn 返回独立 fake，避免 direct 路径消费 invocation 的异常序列
    async def _spawn(*args: object, **kwargs: object) -> object:
        return processes.pop(0)

    monkeypatch.setattr(exec_mod.asyncio, "create_subprocess_shell", _spawn)
    direct_process = processes[0]
    with pytest.raises(RuntimeError) as exc_info:
        await BashTool(HostExecutor(), workspace_root=tmp_path).invoke({"command": "echo hi"})

    assert exc_info.value is direct_error
    assert direct_process.kill_calls == 1
    assert direct_process.communicate_calls == 2
    invoked_process = processes[0]
    result, events = await _invoke(BashTool(HostExecutor(), workspace_root=tmp_path), {"command": "echo hi"})
    failed = [event for event in events if isinstance(event, ToolCallFailedEvent)]
    assert invoked_process.kill_calls == 1
    assert invoked_process.communicate_calls == 2
    assert result.error_type == "execution_error"
    assert result.content == "tool execution failed"
    assert secret not in result.content
    assert [event.attempt for event in failed] == [1]
    assert all(secret not in event.error_message for event in failed)


# 功能：验证 deterministic timeout 会 kill/reap 并保持 timeout ToolResult
# 设计：fake communicate 首次抛 TimeoutError、第二次成功，避免真实 sleep 参与资源测试
async def test_bash_timeout_kills_and_reaps_fake_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = _FakeProcess([TimeoutError(), None])

    # 返回已成功 spawn 的 timeout fake process
    async def _spawn(*args: object, **kwargs: object) -> object:
        return process

    monkeypatch.setattr(exec_mod.asyncio, "create_subprocess_shell", _spawn)

    result = await BashTool(HostExecutor(), workspace_root=tmp_path).invoke({"command": "echo hi"})

    assert result.error_type == "timeout"
    assert process.kill_calls == 1
    assert process.communicate_calls == 2
    assert process.returncode == -9


# 功能：验证 cleanup 自身失败不会覆盖 communicate 的原始异常
# 设计：第二次 communicate 模拟 reap 故障，断言原始异常身份保留且清理故障被记录
async def test_bash_cleanup_failure_logs_and_preserves_original_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    original = RuntimeError("original communicate failure")
    cleanup = OSError("cleanup reap failure")
    process = _FakeProcess([original, cleanup])

    # 返回 cleanup 也会失败的 fake process
    async def _spawn(*args: object, **kwargs: object) -> object:
        return process

    monkeypatch.setattr(exec_mod.asyncio, "create_subprocess_shell", _spawn)

    with caplog.at_level(logging.ERROR, logger="kama_claude.core.sandbox.executors"):
        with pytest.raises(RuntimeError) as exc_info:
            await BashTool(HostExecutor(), workspace_root=tmp_path).invoke({"command": "echo hi"})

    assert exc_info.value is original
    assert process.kill_calls == 1
    assert process.communicate_calls == 2
    assert "cleanup reap failure" in caplog.text


# 功能：验证 Bash command_failed 通过真实 invoke_tool 不会重复执行命令
# 设计：写入 append-only 哨兵文件；若误重试行数会大于一，同时断言单个失败 attempt
async def test_bash_command_failed_is_not_retried(tmp_path: Path) -> None:
    marker = tmp_path / "attempts.txt"
    command = f"printf 'attempt\\n' >> {marker.name}; exit 1"

    result, events = await _invoke(BashTool(HostExecutor(), workspace_root=tmp_path), {"command": command})

    assert result.error_type == "command_failed"
    assert marker.read_text(encoding="utf-8").splitlines() == ["attempt"]
    assert [e.attempt for e in events if isinstance(e, ToolCallFailedEvent)] == [1]
