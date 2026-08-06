from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from kama_claude.core.sandbox.config import SandboxConfig
from kama_claude.core.sandbox.errors import (
    SandboxCreationFailedError,
    SandboxError,
    SandboxTimeoutError,
    SandboxUnavailableError,
)
from kama_claude.core.sandbox.executors import (
    ContainerExecutor,
    ExecResult,
    HostExecutor,
    build_executor,
)
from kama_claude.core.sandbox.manager import SandboxManager


# 功能：验证 HostExecutor 在宿主 cwd 执行命令并合并 stdout/stderr，返回码透传
# 设计：参数化退出码与输出组合，用真实 subprocess 验证行为与既有 BashTool 语义一致
@pytest.mark.parametrize(
    ("command", "expected_code", "expected_output"),
    [
        ("echo hello", 0, b"hello\n"),
        ("exit 3", 3, b""),
        ("echo out; echo err >&2", 0, b"out\nerr\n"),
        ("true", 0, b""),
    ],
)
async def test_host_executor_runs_commands(
    tmp_path: Path,
    command: str,
    expected_code: int,
    expected_output: bytes,
) -> None:
    result = await HostExecutor().exec(command, cwd=tmp_path, timeout=10)
    assert result.returncode == expected_code
    assert result.output == expected_output
    assert result.timed_out is False


# 功能：验证 HostExecutor 超时返回 timed_out=True 且清理子进程
# 设计：用 sleep 命令 + 短超时触发超时路径，断言 ExecResult 语义而非异常
async def test_host_executor_timeout(tmp_path: Path) -> None:
    result = await HostExecutor().exec("sleep 10", cwd=tmp_path, timeout=0.2)
    assert result.timed_out is True
    assert result.returncode == -1


# 功能：验证 HostExecutor 被取消时 CancelledError 原样传播且子进程被清理
# 设计：启动长任务后 cancel，断言 CancelledError 传播——取消是控制流而非工具失败
async def test_host_executor_cancellation_propagates(tmp_path: Path) -> None:
    executor = HostExecutor()
    task = asyncio.create_task(executor.exec("sleep 10", cwd=tmp_path, timeout=30))
    await asyncio.sleep(0.1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


class _FakeManager:
    # 记录转发调用并返回可配置结果，用于隔离 ContainerExecutor 的转发语义
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, float]] = []
        self.failure: SandboxError | None = None

    async def exec(self, command: str, *, cwd: str, timeout: float) -> ExecResult:
        self.calls.append((command, cwd, timeout))
        if self.failure is not None:
            raise self.failure
        return ExecResult(output=b"ok", returncode=0, timed_out=False)


# 功能：验证 ContainerExecutor 把宿主 cwd 映射为容器内路径后转发
# 设计：参数化根/子目录/嵌套路径，断言映射结果与调用透传，映射是纯函数逻辑
@pytest.mark.parametrize(
    ("cwd", "expected_container_cwd"),
    [
        ("/ws", "/workspace"),
        ("/ws/sub", "/workspace/sub"),
        ("/ws/a/b", "/workspace/a/b"),
    ],
)
async def test_container_executor_maps_cwd(
    tmp_path: Path,
    cwd: str,
    expected_container_cwd: str,
) -> None:
    manager = _FakeManager()
    executor = ContainerExecutor(
        manager,  # type: ignore[arg-type]
        workspace_root=tmp_path,
    )
    await executor.exec("ls", cwd=tmp_path / Path(cwd).relative_to("/ws"), timeout=5)
    assert manager.calls[0][0] == "ls"
    assert manager.calls[0][1] == expected_container_cwd
    assert manager.calls[0][2] == 5


# 功能：验证 ContainerExecutor 对 workspace 外 cwd 拒绝执行
# 设计：workspace 外路径无法映射进容器（容器只挂载 workspace），防御性拒绝
async def test_container_executor_rejects_outside_workspace(tmp_path: Path) -> None:
    manager = _FakeManager()
    executor = ContainerExecutor(
        manager,  # type: ignore[arg-type]
        workspace_root=tmp_path,
    )
    with pytest.raises(ValueError, match="outside workspace"):
        await executor.exec("ls", cwd=tmp_path.parent, timeout=5)


# 功能：验证装配工厂按 manager 注入与否选择宿主或容器执行器
# 设计：None → HostExecutor；注入 manager → ContainerExecutor（绑定同一 workspace）
def test_build_executor_selects_by_manager(tmp_path: Path) -> None:
    assert isinstance(
        build_executor(None, workspace_root=tmp_path), HostExecutor
    )
    manager = SandboxManager(
        config=SandboxConfig(image="python:3.12-slim"),
        workspace_root=tmp_path,
    )
    executor = build_executor(manager, workspace_root=tmp_path)
    assert isinstance(executor, ContainerExecutor)
    assert executor._workspace_root == tmp_path


# 功能：验证 ContainerExecutor 把沙箱故障异常原样传播给调用方
# 设计：参数化 unavailable / creation_failed / timeout 三类故障注入 fake manager，
#       断言异常透传而非被吞掉——分类决策留给 BashTool 层
@pytest.mark.parametrize(
    "failure",
    [
        SandboxUnavailableError("no daemon"),
        SandboxCreationFailedError("no image"),
        SandboxTimeoutError("slow"),
    ],
)
async def test_container_executor_propagates_sandbox_errors(
    tmp_path: Path,
    failure: SandboxError,
) -> None:
    manager = _FakeManager()
    manager.failure = failure
    executor = ContainerExecutor(
        manager,  # type: ignore[arg-type]
        workspace_root=tmp_path,
    )
    with pytest.raises(type(failure)):
        await executor.exec("ls", cwd=tmp_path, timeout=5)
