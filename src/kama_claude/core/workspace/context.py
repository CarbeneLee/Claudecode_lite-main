from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from kama_claude.core.git.manager import GitManager
from kama_claude.core.sandbox.manager import SandboxManager
from kama_claude.core.semantic.service import SemanticRetrievalService


@dataclass(frozen=True)
class WorkspaceContext:
    # 一个 workspace 的全部运行时管理器（root 为 canonical 路径）
    root: Path
    sandbox: SandboxManager | None
    git: GitManager | None
    semantic: SemanticRetrievalService | None = None

    # 级联关闭各 manager；底层 close 均有状态护栏，重复调用幂等
    async def close(self) -> None:
        for manager in (self.sandbox, self.git, self.semantic):
            if manager is not None:
                await manager.close()
