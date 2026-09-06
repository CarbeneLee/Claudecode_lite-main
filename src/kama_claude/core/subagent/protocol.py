from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class SubagentOutcome:
    """Bounded terminal child result exposed to a parent tool invocation."""

    status: str
    error_code: str | None
    child_run_id: str
    provider: str = ""
    model: str = ""
    attempt_count: int = 1
    last_committed_step: int = 0
    short_message: str = ""
    preview: str = ""
    evidence_ref: str | None = None

    # 将 child terminal state 转成严格 bounded 的 parent-facing JSON receipt
    def to_parent_receipt(self, max_chars: int = 4_000) -> str:
        limit = max(2, max_chars)
        payload: dict[str, Any] = {
            "status": self.status,
            "error_code": self.error_code,
            "child_run_id": self.child_run_id,
            "provider": self.provider,
            "model": self.model,
            "attempt_count": self.attempt_count,
            "last_committed_step": self.last_committed_step,
            "short_message": self.short_message[:512],
            "preview": self.preview,
            "evidence_ref": self.evidence_ref,
        }
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        if len(encoded) <= limit:
            return encoded
        # 先缩短 preview/short_message，再按需缩短身份字段，始终返回合法 JSON
        for key in (
            "preview",
            "short_message",
            "evidence_ref",
            "model",
            "provider",
            "child_run_id",
            "error_code",
            "status",
        ):
            value = payload[key]
            if not isinstance(value, str):
                continue
            while value and len(encoded) > limit:
                value = value[: max(0, len(value) - max(1, len(encoded) - limit))]
                payload[key] = value
                encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        if len(encoded) <= limit:
            return encoded
        # 删除可选字段后再尝试保留核心状态；极小 limit 退化为合法空 JSON
        for key in ("evidence_ref", "preview", "short_message", "model", "provider"):
            payload.pop(key, None)
            encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True)
            if len(encoded) <= limit:
                return encoded
        minimal = {"status": self.status, "error_code": self.error_code}
        encoded = json.dumps(minimal, ensure_ascii=False, sort_keys=True)
        if len(encoded) <= limit:
            return encoded
        return "{}"

    # 根据 child context 构造 bounded terminal outcome，不携带 traceback/history
    @classmethod
    def from_context(
        cls,
        context: Any,
        *,
        child_run_id: str,
        error_code: str | None = None,
        evidence_ref: str | None = None,
        provider: str | None = None,
        model: str | None = None,
        attempt_count: int | None = None,
    ) -> SubagentOutcome:
        status = str(getattr(context, "status", "failed"))
        result = str(getattr(context, "result", "") or "")
        reason = str(getattr(context, "reason", "") or "")
        return cls(
            status=status,
            error_code=error_code or (reason if status != "success" else None),
            child_run_id=child_run_id,
            provider=(
                provider
                if provider is not None
                else str(getattr(context, "provider_name", ""))
            ),
            model=(
                model
                if model is not None
                else str(getattr(context, "provider_model", ""))
            ),
            attempt_count=max(
                1,
                attempt_count
                if attempt_count is not None
                else int(getattr(context, "provider_attempt_count", 1) or 1),
            ),
            last_committed_step=int(getattr(context, "step", 0) or 0),
            short_message=(result or reason or "Subagent completed")[:512],
            preview=result[:4_000],
            evidence_ref=evidence_ref,
        )


# 将 child-owned evidence reference 限制为带授权前缀的 proxy reference
def authorize_child_evidence(reference: str, *, parent_run_id: str, child_run_id: str) -> str:
    if not reference or not reference.startswith(f"child:{child_run_id}:"):
        raise ValueError("unauthorized child evidence reference")
    return f"parent:{parent_run_id}:{reference}"
