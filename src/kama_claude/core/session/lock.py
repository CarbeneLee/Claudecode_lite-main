from __future__ import annotations

import os
from pathlib import Path
from types import TracebackType
from typing import Self

try:
    import fcntl
except ImportError:  # pragma: no cover - macOS/Linux are the supported targets
    fcntl = None  # type: ignore[assignment]


class DaemonRootLock:
    """sessions root 的进程级 advisory lock，防止多个 daemon 同时写入。"""

    # 初始化 lock 文件路径并延迟打开 descriptor
    def __init__(self, sessions_root: Path) -> None:
        self._root = sessions_root.expanduser()
        self._path = self._root / ".daemon.lock"
        self._fd: int | None = None

    # 以非阻塞独占 flock 获取 daemon 根目录所有权
    def acquire(self) -> None:
        if fcntl is None:
            raise RuntimeError("DaemonRootLock requires POSIX fcntl locking")
        if self._fd is not None:
            raise RuntimeError("daemon root lock already acquired")
        self._root.mkdir(parents=True, exist_ok=True)
        fd = os.open(self._path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (BlockingIOError, OSError) as exc:
            os.close(fd)
            raise RuntimeError("sessions root is already owned by another daemon") from exc
        self._fd = fd

    # 释放 flock 并关闭 descriptor，崩溃时由 OS 自动释放
    def release(self) -> None:
        if self._fd is None:
            return
        fd = self._fd
        self._fd = None
        if fcntl is not None:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)
        else:  # pragma: no cover - defensive fallback
            os.close(fd)

    # 返回当前进程是否持有 lock
    @property
    def held(self) -> bool:
        return self._fd is not None

    # 支持 with 语法获取 root lock
    def __enter__(self) -> Self:
        self.acquire()
        return self

    # 支持 with 语法释放 root lock
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.release()
