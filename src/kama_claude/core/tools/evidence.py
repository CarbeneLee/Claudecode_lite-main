from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO


@dataclass(frozen=True, slots=True)
class ToolEvidenceBudget:
    """工具 producer、durable evidence 与 model ingress 的独立上限。"""

    producer_max_bytes: int = 256 * 1024
    durable_max_bytes: int = 4 * 1024 * 1024
    model_max_chars: int = 8_000
    preview_head_chars: int = 4_000
    preview_tail_chars: int = 2_000


@dataclass(frozen=True, slots=True)
class ToolEvidenceReceipt:
    """Bounded tool result receipt with truthful truncation metadata."""

    preview: str
    raw_truncated: bool
    original_size: int | None
    captured_size: int
    evidence_ref: str | None = None
    model_truncated: bool = False

    # 将 receipt 转成 provider 可见的有限文本并保留 truncation truth
    def to_model_text(self) -> str:
        if (
            not self.raw_truncated
            and not self.model_truncated
            and self.original_size == self.captured_size
        ):
            return self.preview
        metadata = (
            f"\n[evidence raw_truncated={str(self.raw_truncated).lower()} "
            f"original_size={self.original_size} captured_size={self.captured_size}]"
        )
        return self.preview + metadata


# 将任意 producer output 裁切到 durable cap，再生成 head/tail model receipt
def bound_tool_output(
    output: bytes | str,
    *,
    budget: ToolEvidenceBudget = ToolEvidenceBudget(),
    evidence_ref: str | None = None,
    original_size: int | None = None,
    raw_truncated: bool | None = None,
) -> ToolEvidenceReceipt:
    raw = output.encode("utf-8", errors="replace") if isinstance(output, str) else output
    observed_size = len(raw)
    known_original_size = (
        max(observed_size, original_size) if original_size is not None else observed_size
    )
    capture_limit = min(max(0, budget.producer_max_bytes), max(0, budget.durable_max_bytes))
    captured = raw[:capture_limit]
    producer_truncated = bool(raw_truncated) if raw_truncated is not None else False
    was_truncated = producer_truncated or len(captured) < known_original_size
    text = captured.decode("utf-8", errors="replace")
    model_truncated = len(text) > budget.model_max_chars
    if model_truncated:
        marker_template = (
            "\n[... {omitted} chars omitted; "
            "{source}]\n"
        )
        source = (
            f"see evidence artifact {evidence_ref}"
            if evidence_ref
            else "the captured raw output is bounded and no full artifact was retained"
        )
        marker = marker_template.format(omitted=0, source=source)
        available = max(0, budget.model_max_chars - len(marker))
        head_size = min(max(0, budget.preview_head_chars), available)
        tail_size = min(
            max(0, budget.preview_tail_chars),
            max(0, available - head_size),
        )
        # Preserve a useful head/tail receipt while guaranteeing the configured cap.
        head = text[:head_size]
        tail = text[-tail_size:] if tail_size else ""
        omitted = max(0, len(text) - len(head) - len(tail))
        marker = marker_template.format(omitted=omitted, source=source)
        if len(head) + len(marker) + len(tail) > budget.model_max_chars:
            marker = marker[: max(0, budget.model_max_chars - len(head) - len(tail))]
        text = (head + marker + tail)[: budget.model_max_chars]
    return ToolEvidenceReceipt(
        preview=text,
        raw_truncated=was_truncated,
        original_size=(
            max(observed_size, original_size)
            if original_size is not None
            else (None if producer_truncated else observed_size)
        ),
        captured_size=len(captured),
        evidence_ref=evidence_ref,
        model_truncated=model_truncated,
    )


class EvidenceSpool:
    """Bounded file spool that never buffers an unbounded producer output in memory."""

    # 初始化受限 spool 文件并延迟写入
    def __init__(self, path: Path, *, max_bytes: int) -> None:
        self._path = path
        self._max_bytes = max(0, max_bytes)
        self._captured = 0
        self._original = 0
        self._file: BinaryIO | None = None

    # 打开 spool 文件并准备流式写入
    def __enter__(self) -> EvidenceSpool:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self._path.open("wb")
        return self

    # 流式写入当前 chunk，只保留 durable cap 内的 bytes
    def write(self, chunk: bytes) -> None:
        if self._file is None:
            raise RuntimeError("evidence spool is not open")
        self._original += len(chunk)
        remaining = max(0, self._max_bytes - self._captured)
        if remaining:
            retained = chunk[:remaining]
            self._file.write(retained)
            self._captured += len(retained)

    # fsync 并关闭 spool 文件
    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        if self._file is None:
            return
        self._file.flush()
        os.fsync(self._file.fileno())
        self._file.close()
        self._file = None

    # 返回 spool 的真实原始/捕获字节数
    @property
    def sizes(self) -> tuple[int, int]:
        return self._original, self._captured
