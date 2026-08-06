from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from kama_claude.core.git.manager import GitManager
from kama_claude.core.tools.base import BaseTool, ToolResult


class GitDiffParams(BaseModel):
    model_config = ConfigDict(extra="ignore")
    ref: str | None = Field(
        default=None,
        description="Optional ref to compare against; default compares working tree to HEAD.",
    )


class GitCheckpointParams(BaseModel):
    model_config = ConfigDict(extra="ignore")
    label: str = Field(default="auto-step", min_length=1, max_length=100)


class GitCommitParams(BaseModel):
    model_config = ConfigDict(extra="ignore")
    summary: str = Field(min_length=1, max_length=200)


class GitRollbackParams(BaseModel):
    model_config = ConfigDict(extra="ignore")
    step: int = Field(ge=0, description="Checkpoint step number to restore.")


class GitStatusTool(BaseTool):
    # 只读：展示分支与工作树状态，权限默认 auto_allow
    name = "git_status"
    description = (
        "Show the git workspace status: current branch, changed files, "
        "ahead/behind counts. Read-only."
    )
    input_schema: dict[str, object] = {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }

    def __init__(self, manager: GitManager) -> None:
        self._manager = manager

    async def invoke(self, params: dict[str, object]) -> ToolResult:
        status = await self._manager.status()
        lines = [
            f"branch: {status.branch}",
            f"ahead: {status.ahead} behind: {status.behind}",
        ]
        for code, path in status.entries:
            lines.append(f"{code} {path}")
        if not status.entries:
            lines.append("(clean)")
        return ToolResult(content="\n".join(lines))


class GitDiffTool(BaseTool):
    # 只读：diff stat 摘要供 commit 前审查，权限默认 auto_allow
    name = "git_diff"
    description = (
        "Show a diff stat summary of current changes (or against an explicit ref). "
        "Read-only."
    )
    input_schema: dict[str, object] = {
        "type": "object",
        "properties": {
            "ref": {
                "type": "string",
                "description": "Optional ref to compare against; default compares "
                "working tree to HEAD.",
            },
        },
    }

    def __init__(self, manager: GitManager) -> None:
        self._manager = manager

    async def invoke(self, params: dict[str, object]) -> ToolResult:
        p = GitDiffParams.model_validate(params)
        diff = await self._manager.diff(p.ref)
        content = diff.stat or "[no changes]"
        if diff.truncated:
            content += "\n[truncated]"
        return ToolResult(content=content)


class GitCheckpointTool(BaseTool):
    # 写操作：显式存档点，权限默认 ASK
    name = "git_checkpoint"
    description = (
        "Create an internal checkpoint: commit all current changes and register "
        "a recovery point that git_rollback can restore. Requires approval."
    )
    input_schema: dict[str, object] = {
        "type": "object",
        "properties": {
            "label": {
                "type": "string",
                "description": "Short label describing this checkpoint.",
            },
        },
    }

    def __init__(self, manager: GitManager, run_id: str) -> None:
        self._manager = manager
        self._run_id = run_id

    async def invoke(self, params: dict[str, object]) -> ToolResult:
        p = GitCheckpointParams.model_validate(params)
        latest = await self._manager.latest_checkpoint(self._run_id)
        step = (latest.step + 1) if latest is not None else 1
        cp = await self._manager.create_checkpoint(self._run_id, step, p.label)
        if cp is None:
            return ToolResult(content="workspace clean; nothing to checkpoint")
        return ToolResult(content=f"checkpoint {cp.short_sha} step {cp.step}: {cp.label}")


class GitCommitTool(BaseTool):
    # 写操作：唯一用户可见 commit（squash finalize），权限强制 ASK
    name = "git_commit"
    description = (
        "Finalize the run: squash all internal checkpoints into a single visible "
        "commit on the task branch. Requires approval."
    )
    input_schema: dict[str, object] = {
        "type": "object",
        "properties": {
            "summary": {
                "type": "string",
                "description": "Commit summary describing the completed work.",
            },
        },
        "required": ["summary"],
    }

    def __init__(self, manager: GitManager, run_id: str) -> None:
        self._manager = manager
        self._run_id = run_id

    async def invoke(self, params: dict[str, object]) -> ToolResult:
        p = GitCommitParams.model_validate(params)
        result = await self._manager.finalize(self._run_id, p.summary)
        return ToolResult(content=f"committed {result.short_sha}: {p.summary}")


class GitRollbackTool(BaseTool):
    # 写操作：恢复工作树到 checkpoint，权限强制 ASK
    name = "git_rollback"
    description = (
        "Roll back the working tree to a previous checkpoint step. "
        "Requires approval; review the target step before restoring."
    )
    input_schema: dict[str, object] = {
        "type": "object",
        "properties": {
            "step": {
                "type": "integer",
                "description": "Checkpoint step number to restore (0 = run baseline).",
            },
        },
        "required": ["step"],
    }

    def __init__(self, manager: GitManager, run_id: str) -> None:
        self._manager = manager
        self._run_id = run_id

    async def invoke(self, params: dict[str, object]) -> ToolResult:
        p = GitRollbackParams.model_validate(params)
        cp = await self._manager.get_checkpoint(self._run_id, p.step)
        if cp is None:
            return ToolResult(
                content=f"no checkpoint at step {p.step}",
                is_error=True,
                error_type="not_found",
            )
        result = await self._manager.restore(cp)
        return ToolResult(
            content=(
                f"rolled back to step {p.step} "
                f"({result.checkpoint_sha[:7]}) via {result.strategy}"
            )
        )
