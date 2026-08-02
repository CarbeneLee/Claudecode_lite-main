from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from kama_claude.core.sandbox.config import SandboxConfig
from kama_claude.core.sandbox.manager import SandboxManager


# 检查 docker CLI 与 daemon 是否可用；不可用时自动跳过真实冒烟（fault injection 第 3 层）
def _docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        proc = subprocess.run(["docker", "info"], capture_output=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return proc.returncode == 0


pytestmark = pytest.mark.skipif(
    not _docker_available(),
    reason="docker daemon unavailable; start docker to run real sandbox smoke tests",
)


# 功能：验证真实 docker 环境下沙箱 exec 往返（懒创建容器 + 命令执行 + 输出返回）
# 设计：SandboxManager exec echo，断言退出码/输出正确且 state 流转到 ready
async def test_sandbox_exec_echo_roundtrip(tmp_path: Path) -> None:
    manager = SandboxManager(config=SandboxConfig(), workspace_root=tmp_path)
    try:
        result = await manager.exec("echo smoke-ok", cwd="/workspace", timeout=120)

        assert result.returncode == 0
        assert b"smoke-ok" in result.output
        assert manager.state == SandboxManager.READY
    finally:
        await manager.close()


# 功能：验证 workspace bind mount 真实映射（宿主写文件、容器内可见）
# 设计：宿主侧创建 probe 文件后用容器内 cat 读取，证明隔离环境共享同一 workspace
async def test_sandbox_bind_mount_shares_workspace(tmp_path: Path) -> None:
    probe = tmp_path / "probe.txt"
    probe.write_text("bind-mounted", encoding="utf-8")
    manager = SandboxManager(config=SandboxConfig(), workspace_root=tmp_path)
    try:
        result = await manager.exec(
            "cat /workspace/probe.txt", cwd="/workspace", timeout=120
        )

        assert result.returncode == 0
        assert b"bind-mounted" in result.output
    finally:
        await manager.close()


# 功能：验证 close 幂等且关闭后拒绝继续执行
# 设计：两次 close 不抛异常，closed 状态后 exec 抛 RuntimeError（程序性错误）
async def test_sandbox_close_idempotent_and_rejects_exec(tmp_path: Path) -> None:
    manager = SandboxManager(config=SandboxConfig(), workspace_root=tmp_path)
    await manager.exec("echo warm", cwd="/workspace", timeout=120)
    await manager.close()
    await manager.close()  # 幂等：第二次 close 不抛

    assert manager.state == SandboxManager.CLOSED
    with pytest.raises(RuntimeError, match="closed"):
        await manager.exec("echo after-close", cwd="/workspace", timeout=30)
