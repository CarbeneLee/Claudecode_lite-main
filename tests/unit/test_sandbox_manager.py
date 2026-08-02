from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from kama_claude.core.sandbox.config import SandboxConfig
from kama_claude.core.sandbox.errors import (
    ContainerNotReadyError,
    SandboxCreationFailedError,
    SandboxError,
    SandboxImageError,
    SandboxTimeoutError,
    SandboxUnavailableError,
)
from kama_claude.core.sandbox.executors import ExecResult
from kama_claude.core.sandbox.manager import SandboxManager
from kama_claude.core.sandbox.runtime import ContainerRuntime


class FakeDockerRuntime(ContainerRuntime):
    # 进程内 fault injection（设计层 2）：ensure/exec 故障独立注入，创建耗时可调
    def __init__(self) -> None:
        self.ensure_calls = 0
        self.exec_calls = 0
        self.close_calls = 0
        self.ensure_failure: SandboxError | None = None
        self.exec_failure: SandboxError | None = None
        self.ensure_delay: float = 0.0
        self.last_exec: tuple[str, str, float] | None = None

    async def ensure_running(self) -> None:
        self.ensure_calls += 1
        if self.ensure_delay:
            await asyncio.sleep(self.ensure_delay)
        if self.ensure_failure is not None:
            raise self.ensure_failure

    async def exec(self, command: str, *, cwd: str, timeout: float) -> ExecResult:
        self.exec_calls += 1
        self.last_exec = (command, cwd, timeout)
        if self.exec_failure is not None:
            raise self.exec_failure
        return ExecResult(output=b"ok", returncode=0, timed_out=False)

    async def close(self) -> None:
        self.close_calls += 1


@pytest.fixture
def fake_runtime() -> FakeDockerRuntime:
    return FakeDockerRuntime()


def _manager(runtime: ContainerRuntime) -> SandboxManager:
    return SandboxManager(
        config=SandboxConfig(image="python:3.12-slim"),
        workspace_root=Path("/ws"),
        runtime=runtime,
    )


# 功能：首次 exec 触发懒创建，就绪后复用同一容器
# 设计：断言 ensure 恰一次、exec 恰两次、结果透传——懒创建 + ready 状态复用路径
async def test_exec_lazy_creates_then_reuses(
    fake_runtime: FakeDockerRuntime,
) -> None:
    manager = _manager(fake_runtime)
    first = await manager.exec("a", cwd="/workspace", timeout=5)
    second = await manager.exec("b", cwd="/workspace", timeout=5)
    assert first.output == b"ok" and second.output == b"ok"
    assert fake_runtime.ensure_calls == 1
    assert fake_runtime.exec_calls == 2
    assert manager.state == SandboxManager.READY


# 功能：并发 exec 只触发一次容器创建（asyncio.Lock + lock 内 double-check）
# 设计：注入创建耗时，两个 task 同时 exec，断言 ensure 仅一次——
#       若无双重检查，第二个 task 会重复创建
async def test_concurrent_exec_creates_once(
    fake_runtime: FakeDockerRuntime,
) -> None:
    fake_runtime.ensure_delay = 0.05
    manager = _manager(fake_runtime)
    results = await asyncio.gather(
        manager.exec("a", cwd="/workspace", timeout=5),
        manager.exec("b", cwd="/workspace", timeout=5),
    )
    assert [r.output for r in results] == [b"ok", b"ok"]
    assert fake_runtime.ensure_calls == 1
    assert fake_runtime.exec_calls == 2


# 功能：容器创建阶段故障时 exec 原样抛异常（fail closed），绝不静默降级宿主
# 设计：参数化 unavailable / image_error / creation_failed 三类创建故障，
#       断言异常类型透传且状态机进入 failed
@pytest.mark.parametrize(
    "failure",
    [
        SandboxUnavailableError("no daemon"),
        SandboxImageError("no image"),
        SandboxCreationFailedError("create failed"),
    ],
)
async def test_ensure_failure_fails_closed(
    fake_runtime: FakeDockerRuntime,
    failure: SandboxError,
) -> None:
    fake_runtime.ensure_failure = failure
    manager = _manager(fake_runtime)
    with pytest.raises(type(failure)):
        await manager.exec("ls", cwd="/workspace", timeout=5)
    assert fake_runtime.exec_calls == 0  # 未降级到容器外执行
    assert manager.state == SandboxManager.FAILED


# 功能：exec 阶段故障原样传播且状态机保持 ready（超时/未就绪不代表容器死亡）
# 设计：参数化 timeout / not_ready 两类 exec 故障，断言异常透传且后续 exec 正常
@pytest.mark.parametrize(
    "failure",
    [
        SandboxTimeoutError("slow"),
        ContainerNotReadyError("not running"),
    ],
)
async def test_exec_failure_propagates_keeps_ready(
    fake_runtime: FakeDockerRuntime,
    failure: SandboxError,
) -> None:
    fake_runtime.exec_failure = failure
    manager = _manager(fake_runtime)
    with pytest.raises(type(failure)):
        await manager.exec("ls", cwd="/workspace", timeout=5)
    assert manager.state == SandboxManager.READY
    fake_runtime.exec_failure = None
    result = await manager.exec("ls", cwd="/workspace", timeout=5)
    assert result.returncode == 0


# 功能：创建失败后故障恢复，再次 exec 可重建容器（failed 非终态）
# 设计：先注入故障验证 failed 状态，再清除故障断言重建成功——瞬时故障不永久瘫痪 bash
async def test_failed_state_retries_after_recovery(
    fake_runtime: FakeDockerRuntime,
) -> None:
    fake_runtime.ensure_failure = SandboxUnavailableError("daemon down")
    manager = _manager(fake_runtime)
    with pytest.raises(SandboxUnavailableError):
        await manager.exec("ls", cwd="/workspace", timeout=5)
    assert manager.state == SandboxManager.FAILED
    fake_runtime.ensure_failure = None
    result = await manager.exec("ls", cwd="/workspace", timeout=5)
    assert result.returncode == 0
    assert fake_runtime.ensure_calls == 2
    assert manager.state == SandboxManager.READY


# 功能：close 幂等——重复调用只清理一次容器
# 设计：两次 close 断言 runtime.close 恰一次，且状态进入 closed 终态
async def test_close_idempotent(fake_runtime: FakeDockerRuntime) -> None:
    manager = _manager(fake_runtime)
    await manager.exec("ls", cwd="/workspace", timeout=5)
    await manager.close()
    await manager.close()
    assert fake_runtime.close_calls == 1
    assert manager.state == SandboxManager.CLOSED


# 功能：close 后 exec 被拒绝（closed 是终态，容器已删除）
# 设计：closed 状态是生命周期顺序错误的程序性错误，断言抛 RuntimeError 而非沙箱类型
async def test_exec_after_close_rejected(fake_runtime: FakeDockerRuntime) -> None:
    manager = _manager(fake_runtime)
    await manager.close()
    with pytest.raises(RuntimeError, match="closed"):
        await manager.exec("ls", cwd="/workspace", timeout=5)


# 功能：exec 取消时 CancelledError 原样传播（取消是控制流，manager 不吞不转）
# 设计：创建阶段注入耗时，取消发生在 lock 内，断言异常类型与状态机进入 failed 可重试
async def test_exec_cancellation_propagates(
    fake_runtime: FakeDockerRuntime,
) -> None:
    fake_runtime.ensure_delay = 1.0
    manager = _manager(fake_runtime)
    task = asyncio.create_task(manager.exec("ls", cwd="/workspace", timeout=30))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert manager.state == SandboxManager.FAILED


# 功能：manager 把 timeout 参数原样转发给 runtime
# 设计：断言 fake 收到的 timeout 与调用一致——超时策略由调用方（BashTool）决定
async def test_exec_forwards_timeout(fake_runtime: FakeDockerRuntime) -> None:
    manager = _manager(fake_runtime)
    await manager.exec("ls", cwd="/workspace", timeout=42)
    assert fake_runtime.last_exec == ("ls", "/workspace", 42)
