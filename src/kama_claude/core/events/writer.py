from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path


class EventWriter:
    # 初始化只接受 v2 bytes rows 的低层单文件 sink
    def __init__(self, path: Path) -> None:
        self._path = path

    # 在 worker thread 中批量追加完整 JSONL rows 并 flush
    def append_and_flush(self, rows: Iterable[bytes]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("ab") as file:
            for row in rows:
                if not row.endswith(b"\n"):
                    raise ValueError("journal row must end with newline")
                file.write(row)
            file.flush()
