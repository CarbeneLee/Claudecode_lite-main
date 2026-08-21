from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from kama_claude.core.workspace.policy import WorkspaceAccessPolicy
from kama_claude.core.workspace.resolver import WorkspacePathResolver

if TYPE_CHECKING:
    from kama_claude.core.tools.base import BaseTool

ExecutionStatus = Literal[
    "admitted",
    "running",
    "completed_unverified",
    "failed",
    "cancelled",
    "scope_denied",
    "inconclusive",
    "interrupted",
]

TERMINAL_EXECUTION_STATUSES = frozenset(
    {
        "completed_unverified",
        "failed",
        "cancelled",
        "scope_denied",
        "inconclusive",
        "interrupted",
    }
)


# 返回稳定 UTC 时间文本，供 admission/status artifact 使用
def _now() -> str:
    return datetime.now(UTC).isoformat()


# 对结构化 execution artifact 计算 canonical SHA-256 digest
def execution_digest(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class ApprovedExecutionBinding(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    session_id: str
    request_id: str
    execution_id: str
    run_id: str
    projection_key: str
    decision_id: str
    decision_version: int = Field(ge=1)
    decision_content_digest: str
    approval_record_digest: str
    commit_receipt_digest: str
    snapshot_digest: str
    workspace_id: str
    admitted_at: str
    binding_digest: str

    # 从已重新验证的 approval/receipt/snapshot 字段创建 immutable admission binding
    @classmethod
    def create(
        cls,
        *,
        session_id: str,
        request_id: str,
        execution_id: str,
        run_id: str,
        projection_key: str,
        decision_id: str,
        decision_version: int,
        decision_content_digest: str,
        approval_record_digest: str,
        commit_receipt_digest: str,
        snapshot_digest: str,
        workspace_id: str,
        admitted_at: str | None = None,
    ) -> ApprovedExecutionBinding:
        payload: dict[str, Any] = {
            "schema_version": 1,
            "session_id": session_id,
            "request_id": request_id,
            "execution_id": execution_id,
            "run_id": run_id,
            "projection_key": projection_key,
            "decision_id": decision_id,
            "decision_version": decision_version,
            "decision_content_digest": decision_content_digest,
            "approval_record_digest": approval_record_digest,
            "commit_receipt_digest": commit_receipt_digest,
            "snapshot_digest": snapshot_digest,
            "workspace_id": workspace_id,
            "admitted_at": admitted_at or _now(),
        }
        return cls.model_validate(
            {**payload, "binding_digest": execution_digest(payload)}
        )

    # 校验 immutable binding 自身 canonical digest，损坏时 fail closed
    def verify_digest(self) -> None:
        payload = self.model_dump(mode="json", exclude={"binding_digest"})
        if execution_digest(payload) != self.binding_digest:
            raise ValueError("approved execution binding digest mismatch")


class ExecutionStatusProjection(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    execution_id: str
    run_id: str
    request_id: str
    projection_key: str
    status: ExecutionStatus
    status_revision: int = Field(ge=0)
    reason: str | None = None
    updated_at: str


class ExecutionSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    execution_id: str
    run_id: str
    request_id: str
    projection_key: str
    status: ExecutionStatus
    status_revision: int = Field(ge=0)
    reason: str | None = None
    status_digest: str | None = None


class ExecutionSnapshotRelation:
    # 仅作为稳定字符串常量，避免客户端状态机依赖 arrival order
    NEWER = "newer"
    IDEMPOTENT = "idempotent"
    STALE = "stale"
    CONFLICT = "conflict"
    IDENTITY_MISMATCH = "identity_mismatch"


# 比较 execution snapshot 的生命周期单调关系和 terminal 冲突
def compare_execution_snapshots(
    local: ExecutionSnapshot | None,
    incoming: ExecutionSnapshot,
) -> str:
    if local is None:
        return ExecutionSnapshotRelation.NEWER
    if (
        local.execution_id != incoming.execution_id
        or local.run_id != incoming.run_id
        or local.request_id != incoming.request_id
        or local.projection_key != incoming.projection_key
    ):
        return ExecutionSnapshotRelation.IDENTITY_MISMATCH
    same_terminal_payload = (
        local.status == incoming.status
        and local.reason == incoming.reason
        and local.status_digest == incoming.status_digest
    )
    if same_terminal_payload and local.status in TERMINAL_EXECUTION_STATUSES:
        if incoming.status_revision <= local.status_revision:
            return ExecutionSnapshotRelation.IDEMPOTENT
        return ExecutionSnapshotRelation.NEWER
    if local.status == incoming.status and local.status_revision == incoming.status_revision:
        if local.reason == incoming.reason and local.status_digest == incoming.status_digest:
            return ExecutionSnapshotRelation.IDEMPOTENT
        if local.status in TERMINAL_EXECUTION_STATUSES:
            return ExecutionSnapshotRelation.CONFLICT
    if (
        local.status in TERMINAL_EXECUTION_STATUSES
        and incoming.status not in TERMINAL_EXECUTION_STATUSES
    ):
        return ExecutionSnapshotRelation.STALE
    if local.status == "running" and incoming.status == "admitted":
        return ExecutionSnapshotRelation.STALE
    if (
        local.status in TERMINAL_EXECUTION_STATUSES
        and incoming.status in TERMINAL_EXECUTION_STATUSES
    ):
        return ExecutionSnapshotRelation.CONFLICT
    if incoming.status_revision <= local.status_revision:
        return ExecutionSnapshotRelation.STALE
    return ExecutionSnapshotRelation.NEWER


@dataclass(slots=True)
class ExecutionSnapshotState:
    # 合并 approved execution live/replay 状态并隔离 terminal conflict refresh
    owner: ExecutionRequestOwner
    snapshot: ExecutionSnapshot | None = None
    refresh_required: bool = False
    refresh_in_flight: bool = False
    conflicted: bool = False
    conflict_epoch: int | None = None
    _next_conflict_epoch: int = 0

    # 合并普通 live/replay snapshot，不使用 last-arrival-wins
    def merge(
        self,
        incoming: ExecutionSnapshot,
        *,
        owner: ExecutionRequestOwner | None = None,
    ) -> str:
        if owner is not None and owner != self.owner:
            return ExecutionSnapshotRelation.IDENTITY_MISMATCH
        if (
            incoming.request_id != self.owner.request_id
            or incoming.projection_key != self.owner.projection_key
            or incoming.execution_id != self.owner.execution_id
        ):
            return ExecutionSnapshotRelation.IDENTITY_MISMATCH
        relation = compare_execution_snapshots(self.snapshot, incoming)
        if relation == ExecutionSnapshotRelation.CONFLICT:
            if self.conflict_epoch is None:
                self._next_conflict_epoch += 1
                self.conflict_epoch = self._next_conflict_epoch
                self.refresh_required = True
            self.conflicted = True
            return relation
        if relation == ExecutionSnapshotRelation.NEWER:
            self.snapshot = incoming
        return relation

    # 为当前 execution terminal conflict 启动最多一个 authority refresh
    def begin_refresh(self) -> int | None:
        if self.conflict_epoch is None or self.refresh_in_flight or not self.refresh_required:
            return None
        self.refresh_in_flight = True
        self.refresh_required = False
        return self.conflict_epoch

    # 直接应用 authority query 结果，绕过普通冲突分类器
    def apply_authoritative(self, incoming: ExecutionSnapshot, *, epoch: int) -> bool:
        if self.conflict_epoch != epoch or not self.refresh_in_flight:
            return False
        if (
            incoming.request_id != self.owner.request_id
            or incoming.projection_key != self.owner.projection_key
            or incoming.execution_id != self.owner.execution_id
        ):
            return False
        self.snapshot = incoming
        self.conflicted = False
        self.conflict_epoch = None
        self.refresh_in_flight = False
        self.refresh_required = False
        return True

    @property
    # 返回客户端可展示的 lifecycle status，冲突期间只允许 unknown
    def status(self) -> str:
        if self.conflicted:
            return "conflicted/unknown"
        return self.snapshot.status if self.snapshot is not None else "admitted"

    # refresh 失败时保持 conflict unknown，禁止自动形成 refresh loop
    def fail_refresh(self, *, epoch: int) -> bool:
        if self.conflict_epoch != epoch:
            return False
        self.refresh_in_flight = False
        self.refresh_required = False
        return True


class ExecutionRequestOwner(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    client_object: object
    session_id: str
    daemon_instance_id: str
    projection_key: str
    request_id: str
    execution_id: str | None = None

    # 验证异步 response 是否仍属于当前 client/session/projection owner
    def accepts(
        self,
        *,
        client_object: object,
        session_id: str,
        daemon_instance_id: str,
        projection_key: str,
        request_id: str,
        execution_id: str | None = None,
    ) -> bool:
        return (
            client_object is self.client_object
            and session_id == self.session_id
            and daemon_instance_id == self.daemon_instance_id
            and projection_key == self.projection_key
            and request_id == self.request_id
            and (execution_id is None or execution_id == self.execution_id)
        )


class TrustedScopedToolRegistry:
    # approved execution 专用的 exact builtin mapping，构造后不可替换

    @staticmethod
    def _expected_classes() -> tuple[type[Any], ...]:
        # 延迟导入 builtin，避免 registry 类定义阶段触发包初始化循环
        from kama_claude.core.tools.builtin import (
            ListDirTool,
            ReadFileTool,
            SearchCodeTool,
            WriteFileTool,
        )

        return (ReadFileTool, ListDirTool, SearchCodeTool, WriteFileTool)

    # 根据 canonical resolver/policy 创建不可变的 approved scoped tool mapping
    @classmethod
    def create(
        cls,
        resolver: WorkspacePathResolver,
        access_policy: WorkspaceAccessPolicy,
    ) -> TrustedScopedToolRegistry:
        # 延迟导入 builtin，避免 execution 与 bus/session 包初始化形成循环
        from kama_claude.core.tools.builtin import (
            ListDirTool,
            ReadFileTool,
            SearchCodeTool,
            WriteFileTool,
        )

        policy_root = access_policy.root
        if policy_root != resolver.root:
            raise ValueError("trusted scoped registry workspace roots do not match")
        tools = (
            ReadFileTool(resolver, access_policy),
            ListDirTool(resolver, access_policy),
            SearchCodeTool(resolver, access_policy),
            WriteFileTool(resolver, access_policy),
        )
        if tuple(type(tool) for tool in tools) != cls._expected_classes():
            raise TypeError("trusted scoped registry builtin identity mismatch")
        return cls(resolver.root, tools)

    # 物化已验证的 exact builtin mapping，禁止外部直接构造任意工具
    def __init__(self, root: Any, tools: tuple[BaseTool, ...]) -> None:
        expected = self._expected_classes()
        if len(tools) != len(expected) or tuple(type(tool) for tool in tools) != expected:
            raise TypeError("trusted scoped registry requires exact builtin tools")
        if not isinstance(root, Path) or root != root.resolve() or not root.is_dir():
            raise ValueError("trusted scoped registry root must be canonical directory")
        names = tuple(tool.name for tool in tools)
        if names != ("read_file", "list_dir", "search_code", "write_file"):
            raise ValueError("trusted scoped registry tool names are not canonical")
        for tool in tools:
            resolver = getattr(tool, "_resolver", None)
            policy = getattr(tool, "_access_policy", None)
            if (
                resolver is None
                or policy is None
                or resolver.root != root
                or policy.root != root
            ):
                raise ValueError("trusted scoped registry tool root mismatch")
        self._root = root
        self._tools = MappingProxyType({tool.name: tool for tool in tools})

    # 按 canonical name 查找 approved builtin，未知工具返回 None
    def get(self, name: str) -> BaseTool | None:
        return self._tools.get(name)

    # 返回固定 mapping 的只读 schema，供 approved AgentLoop provider 使用
    def tool_schemas(self) -> list[dict[str, object]]:
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.input_schema,
            }
            for tool in self._tools.values()
        ]

    # 返回固定工具名称，供 admission/test 审计使用
    def tool_names(self) -> tuple[str, ...]:
        return tuple(self._tools)

    # 明确拒绝构造后 same-name replacement，防止安全边界被可变 registry 绕过
    def register(self, tool: BaseTool) -> None:
        del tool
        raise TypeError("trusted scoped registry is sealed")

    # 返回 canonical workspace root，供 approved context binding 校验
    @property
    def workspace_root(self) -> Any:
        return self._root
