from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, cast

_UNSET = object()


@dataclass(frozen=True, slots=True)
class SurfaceSnapshot:
    """只描述当前 semantic/projected conversation surface 的 immutable snapshot。"""

    surface_revision: int = 0
    active_head_id: str | None = None
    contract_version: int = 0
    contract_digest: str = ""
    pending_reconciliation_watermark: str = ""
    active_checkpoint_id: str | None = None
    active_checkpoint_status: str = "NONE"
    pending_digest: str = ""


@dataclass(frozen=True, slots=True)
class CompactionCandidate:
    """在 snapshot 之后选择的 compactable span 及其 CAS 身份。"""

    base_surface_revision: int
    base_active_head_id: str | None
    base_contract_digest: str
    base_pending_watermark: str
    base_checkpoint_id: str | None
    selected_unit_ids: tuple[str, ...]
    selected_span_digest: str
    route_epoch: str
    prefix_epoch: str
    request_envelope_id: str
    base_contract_version: int = 0
    base_pending_digest: str = ""
    base_checkpoint_status: str = "NONE"

    # 根据初始 surface snapshot 创建事后选择的 compaction candidate
    @classmethod
    def from_snapshot(
        cls,
        snapshot: SurfaceSnapshot,
        *,
        selected_unit_ids: tuple[str, ...],
        selected_span_digest: str,
        route_epoch: str,
        prefix_epoch: str,
        request_envelope_id: str,
    ) -> CompactionCandidate:
        return cls(
            base_surface_revision=snapshot.surface_revision,
            base_active_head_id=snapshot.active_head_id,
            base_contract_digest=snapshot.contract_digest,
            base_pending_watermark=snapshot.pending_reconciliation_watermark,
            base_checkpoint_id=snapshot.active_checkpoint_id,
            base_checkpoint_status=snapshot.active_checkpoint_status,
            selected_unit_ids=tuple(selected_unit_ids),
            selected_span_digest=selected_span_digest,
            route_epoch=route_epoch,
            prefix_epoch=prefix_epoch,
            request_envelope_id=request_envelope_id,
            base_contract_version=snapshot.contract_version,
            base_pending_digest=snapshot.pending_digest,
        )

    # 计算 selected unit sequence 的稳定 digest，防止异步期间内容被替换
    @staticmethod
    def digest_units(units: list[dict[str, Any]]) -> str:
        encoded = json.dumps(
            units,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass
class SurfaceState:
    """单 session 的 revision/epoch 状态与异步 mutation lane。"""

    surface_revision: int = 0
    active_head_id: str | None = None
    contract_version: int = 0
    contract_digest: str = ""
    pending_reconciliation_watermark: str = ""
    pending_digest: str = ""
    active_checkpoint_id: str | None = None
    active_checkpoint_status: str = "NONE"
    route_epoch: str = "route-0"
    prefix_epoch: str = "prefix-0"
    mutation_lane: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False, compare=False)
    _route_counter: int = field(default=0, init=False, repr=False, compare=False)
    _prefix_counter: int = field(default=0, init=False, repr=False, compare=False)

    # 捕获当前 semantic/projected conversation state，不选择 compactable span
    def snapshot(self) -> SurfaceSnapshot:
        return SurfaceSnapshot(
            surface_revision=self.surface_revision,
            active_head_id=self.active_head_id,
            contract_version=self.contract_version,
            contract_digest=self.contract_digest,
            pending_reconciliation_watermark=self.pending_reconciliation_watermark,
            active_checkpoint_id=self.active_checkpoint_id,
            active_checkpoint_status=self.active_checkpoint_status,
            pending_digest=self.pending_digest,
        )

    # 记录用户、contract、checkpoint 或 steering 等 semantic surface mutation
    def note_surface_mutation(
        self,
        active_head_id: str | None | object = _UNSET,
        *,
        contract_version: int | None = None,
        contract_digest: str | None = None,
        pending_watermark: str | None = None,
        pending_digest: str | None = None,
        checkpoint_id: str | None | object = _UNSET,
        checkpoint_status: str | None | object = _UNSET,
    ) -> None:
        self.surface_revision += 1
        if active_head_id is not _UNSET:
            self.active_head_id = cast(str | None, active_head_id)
        if contract_version is not None:
            self.contract_version = contract_version
        if contract_digest is not None:
            self.contract_digest = contract_digest
        if pending_watermark is not None:
            self.pending_reconciliation_watermark = pending_watermark
        if pending_digest is not None:
            self.pending_digest = pending_digest
        if checkpoint_id is not _UNSET:
            self.active_checkpoint_id = cast(str | None, checkpoint_id)
        if checkpoint_status is not _UNSET:
            self.active_checkpoint_status = (
                "NONE" if checkpoint_status is None else str(checkpoint_status)
            )

    # 更新 route epoch 而不伪装成 conversation surface mutation
    def change_route(self, route_identity: str) -> None:
        self._route_counter += 1
        digest = hashlib.sha256(
            f"{self._route_counter}:{route_identity}".encode()
        ).hexdigest()[:16]
        self.route_epoch = f"route-{digest}"

    # 更新 prompt A/B 内容 epoch 而不增加 surface revision
    def change_prefix(self, prefix_identity: str) -> None:
        self._prefix_counter += 1
        digest = hashlib.sha256(
            f"{self._prefix_counter}:{prefix_identity}".encode()
        ).hexdigest()[:16]
        self.prefix_epoch = f"prefix-{digest}"

    # 将 gateway 已计算的 route/prefix epoch 同步到 surface，不伪造 conversation mutation
    def synchronize_epochs(self, *, route_epoch: str, prefix_epoch: str) -> None:
        self.route_epoch = route_epoch
        self.prefix_epoch = prefix_epoch

    # 比较 candidate 所捕获的所有 semantic surface 与 route/prefix identity
    def candidate_matches(self, candidate: CompactionCandidate) -> bool:
        snapshot = self.snapshot()
        return (
            snapshot.surface_revision == candidate.base_surface_revision
            and snapshot.active_head_id == candidate.base_active_head_id
            and snapshot.contract_version == candidate.base_contract_version
            and snapshot.contract_digest == candidate.base_contract_digest
            and snapshot.pending_reconciliation_watermark == candidate.base_pending_watermark
            and snapshot.pending_digest == candidate.base_pending_digest
            and snapshot.active_checkpoint_id == candidate.base_checkpoint_id
            and snapshot.active_checkpoint_status == candidate.base_checkpoint_status
            and self.route_epoch == candidate.route_epoch
            and self.prefix_epoch == candidate.prefix_epoch
        )

    # 在 mutation lane 内以 CAS 方式提交 candidate，冲突时拒绝旧摘要
    def commit_candidate(self, candidate: CompactionCandidate) -> bool:
        if not self.candidate_matches(candidate):
            return False
        self.surface_revision += 1
        self.active_head_id = candidate.selected_span_digest
        self.active_checkpoint_id = candidate.selected_span_digest
        self.active_checkpoint_status = "ACTIVE_CHECKPOINT"
        return True


# 纯函数比较当前 snapshot 与 candidate 的 semantic/epoch 身份
def semantic_surface_matches(
    snapshot: SurfaceSnapshot,
    candidate: CompactionCandidate,
    *,
    route_epoch: str,
    prefix_epoch: str,
) -> bool:
    return (
        snapshot.surface_revision == candidate.base_surface_revision
        and snapshot.active_head_id == candidate.base_active_head_id
        and snapshot.contract_version == candidate.base_contract_version
        and snapshot.contract_digest == candidate.base_contract_digest
        and snapshot.pending_reconciliation_watermark == candidate.base_pending_watermark
        and snapshot.pending_digest == candidate.base_pending_digest
        and snapshot.active_checkpoint_id == candidate.base_checkpoint_id
        and snapshot.active_checkpoint_status == candidate.base_checkpoint_status
        and route_epoch == candidate.route_epoch
        and prefix_epoch == candidate.prefix_epoch
    )
