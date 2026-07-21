from __future__ import annotations

from pathlib import Path

import pytest

from kama_claude.core.events.writer import EventWriter


# 功能：验证低层 v2 sink 原样追加完整 UTF-8 JSONL bytes
# 设计：使用真实 tmp_path 并比较精确 bytes，避免把 schema/identity 职责重新塞回 sink
def test_event_writer_appends_complete_rows(tmp_path: Path) -> None:
    path = tmp_path / "events.v2.jsonl"
    writer = EventWriter(path)

    writer.append_and_flush((b'{"seq":1}\n', b'{"seq":2}\n'))

    assert path.read_bytes() == b'{"seq":1}\n{"seq":2}\n'


# 功能：验证 sink 自动创建 v2 journal 的多级父目录
# 设计：直接写一条完整 row 并只断言最终文件，隔离 coordinator 的 stream 注册逻辑
def test_event_writer_creates_parent_directories(tmp_path: Path) -> None:
    path = tmp_path / "runs" / "run-1" / "events.v2.jsonl"

    EventWriter(path).append_and_flush((b'{"seq":1}\n',))

    assert path.exists()


# 功能：验证多次 batch append 不覆盖先前 durable prefix
# 设计：通过两个独立 append_and_flush 调用锁定 append 模式，模拟 worker 的相邻 flush batch
def test_event_writer_multiple_batches_preserve_prefix(tmp_path: Path) -> None:
    path = tmp_path / "events.v2.jsonl"
    writer = EventWriter(path)

    writer.append_and_flush((b'{"seq":1}\n',))
    writer.append_and_flush((b'{"seq":2}\n',))

    assert path.read_bytes().splitlines() == [b'{"seq":1}', b'{"seq":2}']


# 功能：验证 sink 拒绝缺少换行的部分 row 且不写入该 row
# 设计：传入 crash-tail 形态的 bytes 并断言 ValueError，保证 worker 永不主动制造不完整 JSONL 行
def test_event_writer_rejects_partial_row(tmp_path: Path) -> None:
    path = tmp_path / "events.v2.jsonl"

    with pytest.raises(ValueError, match="newline"):
        EventWriter(path).append_and_flush((b'{"seq":1}',))

    assert path.read_bytes() == b""
