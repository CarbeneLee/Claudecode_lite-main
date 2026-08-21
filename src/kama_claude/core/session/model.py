from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

SessionStatus = Literal["active", "waiting_for_input", "closed"]
SessionMode = Literal["one_shot", "chat"]
AgentMode = Literal["direct", "plan"]
MAX_AGENT_MODE_REVISION = 2**63 - 1


class ModeSnapshotRelation(StrEnum):
    NEWER = "newer"
    EQUAL_SAME = "equal_same"
    EQUAL_CONFLICT = "equal_conflict"
    OLDER = "older"


# 校验 agent mode 的持久化 domain 值并保留 Literal 类型
def validate_agent_mode(value: object) -> AgentMode:
    if value not in ("direct", "plan"):
        raise ValueError("invalid agent_mode")
    return value


# 校验单调 mode revision，拒绝 bool、负数和超过协议上限的值
def validate_agent_mode_revision(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("invalid agent_mode_revision")
    if value < 0 or value > MAX_AGENT_MODE_REVISION:
        raise ValueError("invalid agent_mode_revision")
    return value


@dataclass(frozen=True, slots=True)
class AgentModeSnapshot:
    agent_mode: AgentMode
    revision: int

    # 构造 mode snapshot 时统一执行 domain invariant 校验
    def __post_init__(self) -> None:
        object.__setattr__(self, "agent_mode", validate_agent_mode(self.agent_mode))
        object.__setattr__(self, "revision", validate_agent_mode_revision(self.revision))


# 只比较本地与 incoming snapshot，不执行 RPC、渲染或状态变更
def compare_agent_mode_snapshots(
    local: AgentModeSnapshot | None,
    incoming: AgentModeSnapshot,
) -> ModeSnapshotRelation:
    if local is None or incoming.revision > local.revision:
        return ModeSnapshotRelation.NEWER
    if incoming.revision < local.revision:
        return ModeSnapshotRelation.OLDER
    if incoming.agent_mode == local.agent_mode:
        return ModeSnapshotRelation.EQUAL_SAME
    return ModeSnapshotRelation.EQUAL_CONFLICT


@dataclass
class Session:
    id: str
    mode: SessionMode
    status: SessionStatus
    title: str
    created_at: str
    updated_at: str
    workspace_root: Path
    agent_mode: AgentMode = "direct"
    agent_mode_revision: int = 0
    run_ids: list[str] = field(default_factory=list)
    active_run_id: str | None = None

    # Session 构造完成时确保持久化 mode 与 revision 已满足统一 invariant
    def __post_init__(self) -> None:
        self.agent_mode = validate_agent_mode(self.agent_mode)
        self.agent_mode_revision = validate_agent_mode_revision(self.agent_mode_revision)

    # 将 Session 转为可写入 meta.json 的普通 dict
    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "mode": self.mode,
            "agent_mode": self.agent_mode,
            "agent_mode_revision": self.agent_mode_revision,
            "status": self.status,
            "title": self.title,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "workspace_root": str(self.workspace_root),
            "run_ids": list(self.run_ids),
            "active_run_id": self.active_run_id,
        }

    # 从 meta.json 的 dict 还原 Session 对象
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Session:
        return cls(
            id=str(data["id"]),
            mode=data["mode"],
            agent_mode=data.get("agent_mode", "direct"),
            agent_mode_revision=data.get("agent_mode_revision", 0),
            status=data["status"],
            title=str(data.get("title", "")),
            created_at=str(data["created_at"]),
            updated_at=str(data["updated_at"]),
            workspace_root=Path(data["workspace_root"]),
            run_ids=[str(x) for x in data.get("run_ids", [])],
            active_run_id=(
                str(data["active_run_id"])
                if data.get("active_run_id") is not None
                else None
            ),
        )
