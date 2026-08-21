from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import Iterable
from pathlib import Path
from typing import Any, cast

import pytest

from kama_claude.core.bus.events import (
    LogLineEvent,
    PlannerDecisionReadyEvent,
    RunFinishedEvent,
    RunStartedEvent,
)
from kama_claude.core.events import journal as journal_module
from kama_claude.core.events.journal import (
    DuplicateStreamOwnerError,
    EventJournalCoordinator,
    JournalCapacityError,
    JournalCorruptionError,
    JournalError,
    JournalRecord,
    ReplayBatch,
    StreamLifecycle,
    UnknownStreamError,
)
from kama_claude.core.events.writer import EventWriter
from kama_claude.core.plan_view import LegacyPlanViewV0


# 构造字段稳定的 run.started 事件，便于比较跨 stream identity
def _run_started(run_id: str = "run-1") -> RunStartedEvent:
    return RunStartedEvent(
        run_id=run_id,
        goal="journal test",
        ts="2026-07-21T00:00:00Z",
    )


# 构造双 durable stream 使用的结构化 PlanReady 事件
def _plan_ready(event_id: str = "plan-1", *, approach: str = "reuse") -> PlannerDecisionReadyEvent:
    plan = LegacyPlanViewV0(
        plan_key="decision-1:v1",
        goal="journal plan",
        selected_approach=approach,
    )
    return PlannerDecisionReadyEvent(
        event_id=event_id,
        run_id="run-1",
        planner_run_id="planner-1",
        session_id="sess-1",
        plan=plan,
        plan_key=plan.plan_key,
        decision_id="decision-1",
        decision_version=1,
        ts="2026-07-21T00:00:00Z",
        snapshot_digest="snapshot",
        content_digest="content",
    )


# 读取 JSONL 文件并解析为对象列表
def _read_rows(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


# 功能：验证同一事件写入 run/session journal 时共享全局 event_id 但各自使用 stream-local seq
# 设计：注册真实双 stream 并读取两个 v2 文件，直接比较持久化行而非依赖内存回调
async def test_dual_stream_records_share_event_id_and_keep_local_seq(tmp_path: Path) -> None:
    delivered: list[object] = []

    # 记录两个 durable stream 回调以核对 overlap 次数
    async def on_durable(record: JournalRecord) -> None:
        delivered.append(record)

    coordinator = EventJournalCoordinator(on_durable=on_durable)
    session_path = tmp_path / "sessions" / "sess-1"
    run_path = session_path / "runs" / "run-1"
    await coordinator.register_session("sess-1", session_path)
    await coordinator.register_run("run-1", run_path, session_id="sess-1")

    await coordinator.handle(_run_started())
    await coordinator.flush_all()
    await coordinator.close()

    run_rows = _read_rows(run_path / "events.v2.jsonl")
    session_rows = _read_rows(session_path / "events.v2.jsonl")
    assert run_rows[0]["event_id"] == session_rows[0]["event_id"]
    assert run_rows[0]["stream_id"] == "run:run-1"
    assert session_rows[0]["stream_id"] == "session:sess-1"
    assert run_rows[0]["seq"] == 1
    assert session_rows[0]["seq"] == 1
    assert len(delivered) == 2


# 功能：验证 PlanReady 必须同时 durable 到 top-level run 与 session stream，并支持精确幂等重放
# 设计：先发布相同 event_id 两次再发布冲突 payload，直接读取两个 journal 的严格 replay 结果
async def test_plan_ready_requires_dual_durable_route_and_deduplicates_payload(
    tmp_path: Path,
) -> None:
    coordinator = EventJournalCoordinator()
    session_path = tmp_path / "sessions" / "sess-1"
    run_path = session_path / "runs" / "run-1"
    await coordinator.register_session("sess-1", session_path)
    await coordinator.register_run("run-1", run_path, session_id="sess-1")

    event = _plan_ready()
    first, second = await asyncio.gather(
        coordinator.publish_required_durable(event),
        coordinator.publish_required_durable(event),
    )
    assert first == second

    run_replay = await coordinator.read_replay("run:run-1", after_seq=0, high_watermark=1)
    session_replay = await coordinator.read_replay(
        "session:sess-1",
        after_seq=0,
        high_watermark=1,
    )
    assert run_replay.records[0].event_id == event.event_id
    assert session_replay.records[0].event_id == event.event_id

    with pytest.raises(JournalError, match="payload conflict"):
        await coordinator.publish_required_durable(_plan_ready(approach="different"))
    await coordinator.close()


# 功能：验证两个并发 run 各自只写入所属 stream，不交叉污染 journal
# 设计：并发发布两个不同 run_id 事件，从两个真实 v2 文件核对独立 seq 和 payload
async def test_concurrent_runs_keep_single_stream_ownership(tmp_path: Path) -> None:
    coordinator = EventJournalCoordinator()
    await coordinator.register_run("run-a", tmp_path / "run-a", session_id=None)
    await coordinator.register_run("run-b", tmp_path / "run-b", session_id=None)

    await asyncio.gather(
        coordinator.handle(_run_started("run-a")),
        coordinator.handle(_run_started("run-b")),
    )
    await coordinator.flush_all()
    await coordinator.close()

    rows_a = _read_rows(tmp_path / "run-a" / "events.v2.jsonl")
    rows_b = _read_rows(tmp_path / "run-b" / "events.v2.jsonl")
    assert [(row["stream_id"], row["seq"]) for row in rows_a] == [("run:run-a", 1)]
    assert [(row["stream_id"], row["seq"]) for row in rows_b] == [("run:run-b", 1)]
    assert rows_a[0]["event"]["run_id"] == "run-a"
    assert rows_b[0]["event"]["run_id"] == "run-b"


# 功能：验证同一 durable stream 不能被第二个 writer owner 重复打开
# 设计：连续注册相同 run_id 并断言稳定内部异常，防止两个 worker 同时追加同一文件
async def test_duplicate_stream_owner_is_rejected(tmp_path: Path) -> None:
    coordinator = EventJournalCoordinator()
    await coordinator.register_run("run-1", tmp_path / "run-1", session_id=None)

    with pytest.raises(DuplicateStreamOwnerError, match="run:run-1"):
        await coordinator.register_run("run-1", tmp_path / "run-1", session_id=None)

    await coordinator.close()


# 功能：验证未注册 run 的 durable event fail closed，不得降级为 seq-less live-only
# 设计：注入 live-only collector 后直接发布带 run_id 事件，同时断言异常与零泄漏回调
async def test_unknown_run_event_is_not_downgraded_to_live_only() -> None:
    delivered: list[object] = []

    # 记录只允许无 durable identity 事件使用的回调
    async def on_live_only(event: object) -> None:
        delivered.append(event)

    coordinator = EventJournalCoordinator(on_live_only=on_live_only)

    with pytest.raises(UnknownStreamError, match="run:run-unknown"):
        await coordinator.handle(_run_started("run-unknown"))

    assert delivered == []
    await coordinator.close()


# 功能：验证 run 不能绑定尚未注册的 parent session
# 设计：直接走真实 identity registry 入口，断言失败后 run stream 与 mapping 均未创建
async def test_run_rejects_unknown_parent_session(tmp_path: Path) -> None:
    coordinator = EventJournalCoordinator()

    with pytest.raises(UnknownStreamError, match="session:sess-missing"):
        await coordinator.register_run(
            "run-1",
            tmp_path / "run-1",
            session_id="sess-missing",
        )

    assert not coordinator.has_stream("run:run-1")
    assert coordinator.identities.lookup_run("run-1") == (False, None)
    await coordinator.close()


# 功能：验证 legacy raw rows 获得稳定 synthetic seq/event_id 且 v2 延续该 stream 的下一序号
# 设计：先写真实 legacy 前缀再注册并追加 v2 事件，跨两次 replay 比较 identity 与连续序号
async def test_legacy_prefix_has_stable_identity_and_v2_continues_seq(tmp_path: Path) -> None:
    run_path = tmp_path / "run-legacy"
    run_path.mkdir(parents=True)
    legacy = _run_started("run-legacy").model_dump_json()
    (run_path / "events.jsonl").write_text(legacy + "\n", encoding="utf-8")

    coordinator = EventJournalCoordinator()
    await coordinator.register_run("run-legacy", run_path, session_id=None)
    first = await coordinator.read_replay("run:run-legacy", after_seq=0, high_watermark=1)
    second = await coordinator.read_replay("run:run-legacy", after_seq=0, high_watermark=1)

    await coordinator.handle(_run_started("run-legacy"))
    await coordinator.flush_all()
    combined = await coordinator.read_replay(
        "run:run-legacy",
        after_seq=0,
        high_watermark=2,
    )
    await coordinator.close()

    assert first.records[0].event_id == second.records[0].event_id
    assert first.records[0].event_id.startswith("legacy-")
    assert [record.seq for record in combined.records] == [1, 2]


@pytest.mark.parametrize("tail", [b"", b"\n{"])
# 功能：验证 legacy 合法无尾换行可读，无效 crash tail 被忽略且不占 seq
# 设计：对同一合法 raw Event 分别保留 EOF 或追加不完整 JSON，回放均只得一条记录
async def test_legacy_eof_and_truncated_tail_keep_complete_prefix(
    tmp_path: Path,
    tail: bytes,
) -> None:
    run_path = tmp_path / "run-legacy-tail"
    run_path.mkdir(parents=True)
    raw = _run_started("run-legacy-tail").model_dump_json().encode()
    (run_path / "events.jsonl").write_bytes(raw + tail)
    coordinator = EventJournalCoordinator()

    await coordinator.register_run("run-legacy-tail", run_path, session_id=None)
    replay = await coordinator.read_replay(
        "run:run-legacy-tail",
        after_seq=0,
        high_watermark=1,
    )

    assert len(replay.records) == 1
    assert replay.records[0].seq == 1
    await coordinator.close()


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (b"{\n", "complete invalid row"),
        (b"[]\n", "must be an Event object"),
        (b'{"type":"run.started"}\n', "invalid Event"),
        (
            json.dumps(
                {
                    **_run_started("run-bad-legacy").model_dump(mode="json"),
                    "schema_version": 2,
                }
            ).encode()
            + b"\n",
            "mixed legacy/v2",
        ),
    ],
)
# 功能：验证 legacy 完整坏行、标量、无效 Event 与混合 v2 标记全部 fail closed
# 设计：每个 case 只变更一类语法或 schema 不变量，在 stream owner 打开前断言稳定分类
async def test_legacy_invalid_complete_rows_are_rejected(
    tmp_path: Path,
    content: bytes,
    message: str,
) -> None:
    run_path = tmp_path / "run-bad-legacy"
    run_path.mkdir(parents=True)
    (run_path / "events.jsonl").write_bytes(content)
    coordinator = EventJournalCoordinator()

    with pytest.raises(JournalCorruptionError, match=message):
        await coordinator.register_run("run-bad-legacy", run_path, session_id=None)

    await coordinator.close()


# 功能：验证 replay 在 v2 中间行损坏时整批失败而不是返回损坏前的部分记录
# 设计：构造一条合法行加一条带换行的非法行，断言 reader 只抛 corruption 且不暴露 ReplayBatch
async def test_replay_rejects_complete_corrupt_row_without_partial_batch(tmp_path: Path) -> None:
    run_path = tmp_path / "run-corrupt"
    run_path.mkdir(parents=True)
    valid = {
        "schema_version": 2,
        "event_id": "evt-1",
        "stream_id": "run:run-corrupt",
        "seq": 1,
        "event": _run_started("run-corrupt").model_dump(mode="json"),
    }
    (run_path / "events.v2.jsonl").write_text(
        json.dumps(valid) + "\n{" + "\n",
        encoding="utf-8",
    )
    coordinator = EventJournalCoordinator()

    with pytest.raises(JournalCorruptionError, match="complete invalid row"):
        await coordinator.register_run("run-corrupt", run_path, session_id=None)

    await coordinator.close()


# 功能：验证 worker 已 pop 但尚未 flush 的 in-flight batch 仍占 journal frame reservation
# 设计：在线程 append gate 上阻塞第一条，随后填满总计 4096 条并断言第 4097 条立即 fail closed
async def test_queue_frame_limit_counts_inflight_batch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entered = threading.Event()
    release = threading.Event()
    original = EventWriter.append_and_flush

    # 阻塞第一个 batch flush，制造 deque 已出队但 reservation 未释放的窗口
    def blocked_append(writer: EventWriter, rows: Iterable[bytes]) -> None:
        entered.set()
        release.wait(timeout=5)
        original(writer, rows)

    monkeypatch.setattr(EventWriter, "append_and_flush", blocked_append)
    coordinator = EventJournalCoordinator()
    await coordinator.register_run("run-1", tmp_path / "run-1", session_id=None)
    await coordinator.handle(_run_started())
    assert await asyncio.to_thread(entered.wait, 1)

    for _ in range(journal_module.JOURNAL_QUEUE_MAX_FRAMES - 1):
        await coordinator.handle(_run_started())

    with pytest.raises(JournalCapacityError, match="capacity"):
        await coordinator.handle(_run_started())

    release.set()
    await coordinator.close()


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda rows: rows[1].update(seq=3), "seq continuity"),
        (
            lambda rows: rows[1].update(event_id=rows[0]["event_id"]),
            "duplicate event identity",
        ),
    ],
)
# 功能：验证 seq corruption 与同 stream duplicate event_id 都使整个 v2 stream 拒绝打开
# 设计：从两个结构合法 wrapper 只变异一个不变量，证明失败来自 identity/ordering 校验而非 JSON 解析
async def test_v2_identity_and_sequence_corruption_rejected(
    tmp_path: Path,
    mutate: Any,
    message: str,
) -> None:
    run_path = tmp_path / "run-mutated"
    run_path.mkdir(parents=True)
    rows = [
        {
            "schema_version": 2,
            "event_id": f"evt-{index}",
            "stream_id": "run:run-mutated",
            "seq": index,
            "event": _run_started("run-mutated").model_dump(mode="json"),
        }
        for index in (1, 2)
    ]
    mutate(rows)
    (run_path / "events.v2.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    coordinator = EventJournalCoordinator()

    with pytest.raises(JournalCorruptionError, match=message):
        await coordinator.register_run("run-mutated", run_path, session_id=None)

    await coordinator.close()


# 功能：验证请求的 high watermark 超过完整 durable prefix 时 replay 明确失败
# 设计：只持久化 seq=1 却请求 through=2，锁定 missing-watermark mutation 不得静默返回短 batch
async def test_replay_rejects_missing_high_watermark(tmp_path: Path) -> None:
    coordinator = EventJournalCoordinator()
    await coordinator.register_run("run-1", tmp_path / "run-1", session_id=None)
    await coordinator.handle(_run_started())
    await coordinator.flush_all()

    with pytest.raises(JournalCorruptionError, match="missing high watermark"):
        await coordinator.read_replay("run:run-1", after_seq=0, high_watermark=2)

    await coordinator.close()


# 功能：验证 legacy 与 v2 共享同一 16 MiB replay actual-byte budget
# 设计：用读取函数 spy 让 legacy 消耗 limit-7，断言 v2 只收到剩余 7 bytes 而非新的完整限额
def test_replay_legacy_and_v2_share_total_byte_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_v2_budget: list[int] = []

    # 模拟 legacy 已消耗绝大部分全局 replay bytes
    def read_legacy(
        stream_id: str,
        path: Path | None,
        *,
        stop_event: threading.Event | None,
        max_bytes: int,
    ) -> tuple[list[JournalRecord], int]:
        assert max_bytes == journal_module.MAX_REPLAY_BYTES
        return [], journal_module.MAX_REPLAY_BYTES - 7

    # 记录 v2 reader 获得的真实剩余 budget
    def read_v2(
        stream_id: str,
        path: Path,
        *,
        expected_first_seq: int,
        stop_event: threading.Event | None,
        max_bytes: int,
    ) -> tuple[list[JournalRecord], int]:
        observed_v2_budget.append(max_bytes)
        return [], 0

    monkeypatch.setattr(journal_module, "_read_legacy_records", read_legacy)
    monkeypatch.setattr(journal_module, "_read_v2_records", read_v2)
    batch = journal_module._read_replay_batch(
        "run:run-1",
        tmp_path / "events.v2.jsonl",
        tmp_path / "events.jsonl",
        0,
        0,
        threading.Event(),
    )

    assert observed_v2_budget == [7]
    assert batch.examined_bytes == journal_module.MAX_REPLAY_BYTES - 7


# 功能：验证底层 replay reader 按实际 bytes 接受 limit 并在 limit+1 立即拒绝
# 设计：对同一小文件分别传入精确长度和少一的 budget，避免构造 16 MiB fixture
def test_replay_file_reader_enforces_actual_byte_limit(tmp_path: Path) -> None:
    path = tmp_path / "bytes.bin"
    path.write_bytes(b"abcd")

    assert journal_module._read_file_bytes(
        path,
        None,
        max_bytes=4,
    ) == b"abcd"
    with pytest.raises(JournalCapacityError, match="16 MiB"):
        journal_module._read_file_bytes(path, None, max_bytes=3)


# 功能：验证 replay thread 在首次和重复 cancellation 后 cooperative terminal 再恢复取消
# 设计：fake reader 在 stop_event 后仍受 release gate 控制，二次 cancel 期间断言 task 未提前完成
async def test_replay_cancellation_waits_for_worker_terminal_under_repeated_cancel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entered = threading.Event()
    stop_seen = threading.Event()
    release = threading.Event()

    # 在 cooperative stop 后继续等待受控 release，暴露 worker 是否提前退出
    def blocked_reader(
        stream_id: str,
        v2_path: Path,
        legacy_path: Path | None,
        after_seq: int,
        high_watermark: int,
        stop_event: threading.Event,
    ) -> ReplayBatch:
        entered.set()
        stop_event.wait(timeout=5)
        stop_seen.set()
        release.wait(timeout=5)
        return ReplayBatch(records=(), examined_bytes=0)

    monkeypatch.setattr(journal_module, "_read_replay_batch", blocked_reader)
    coordinator = EventJournalCoordinator()
    await coordinator.register_run("run-1", tmp_path / "run-1", session_id=None)
    task = asyncio.create_task(
        coordinator.read_replay("run:run-1", after_seq=0, high_watermark=0)
    )
    assert await asyncio.to_thread(entered.wait, 1)

    task.cancel("first")
    assert await asyncio.to_thread(stop_seen.wait, 1)
    task.cancel("repeated")
    assert not task.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task

    await coordinator.close()
    assert task.done()


@pytest.mark.parametrize(
    "failed_roles",
    [
        frozenset({"session"}),
        frozenset({"run"}),
        frozenset({"run", "session"}),
    ],
    ids=["session-only", "run-only", "both"],
)
# 功能：验证 run/session 任一方或双方 append 失败时只保留各自真实 durable prefix
# 设计：按 path 对三种 failure matrix 定向注入，核对 watermark、lifecycle、文件与后续原子拒绝
async def test_partial_stream_failure_preserves_each_real_prefix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failed_roles: frozenset[str],
) -> None:
    original = EventWriter.append_and_flush
    session_path = tmp_path / "session"
    run_path = session_path / "runs" / "run-1"

    # 按测试矩阵只让选定 stream 的 append 失败
    def fail_selected(writer: EventWriter, rows: Iterable[bytes]) -> None:
        role = "session" if writer._path == session_path / "events.v2.jsonl" else "run"
        if role in failed_roles:
            raise OSError("journal-disk-secret")
        original(writer, rows)

    monkeypatch.setattr(EventWriter, "append_and_flush", fail_selected)
    coordinator = EventJournalCoordinator()
    await coordinator.register_session("sess-1", session_path)
    await coordinator.register_run(
        "run-1",
        run_path,
        session_id="sess-1",
    )

    await coordinator.handle(_run_started())
    with pytest.raises(JournalError, match="journal append failed"):
        await coordinator.flush_all()

    expectations = {
        "run": ("run:run-1", run_path / "events.v2.jsonl"),
        "session": ("session:sess-1", session_path / "events.v2.jsonl"),
    }
    for role, (stream_id, path) in expectations.items():
        succeeded = role not in failed_roles
        assert coordinator.high_watermark(stream_id) == int(succeeded)
        expected_lifecycle = StreamLifecycle.OPEN if succeeded else StreamLifecycle.DEGRADED
        assert coordinator.stream_lifecycle(stream_id) is expected_lifecycle
        assert path.exists() is succeeded

    before = {
        stream_id: coordinator.high_watermark(stream_id)
        for stream_id, _path in expectations.values()
    }
    with pytest.raises(JournalError, match="degraded"):
        await coordinator.handle(_run_started())
    assert {
        stream_id: coordinator.high_watermark(stream_id)
        for stream_id, _path in expectations.values()
    } == before
    await coordinator.close()


# 功能：验证 append 已成功后的 live callback 失败不会把 journal stream 降级
# 设计：durable callback 固定抛连接错误，随后读取 watermark 与 v2 文件证明事实源仍可 replay
async def test_live_delivery_failure_does_not_degrade_durable_stream(tmp_path: Path) -> None:
    # 模拟 durable 之后的单客户端 delivery failure
    async def fail_live(record: JournalRecord) -> None:
        raise ConnectionError("live-connection-secret")

    coordinator = EventJournalCoordinator(on_durable=fail_live)
    await coordinator.register_run("run-1", tmp_path / "run-1", session_id=None)

    await coordinator.handle(_run_started())
    await coordinator.flush_all()

    assert coordinator.high_watermark("run:run-1") == 1
    assert coordinator.stream_lifecycle("run:run-1") is StreamLifecycle.OPEN
    assert (tmp_path / "run-1" / "events.v2.jsonl").exists()
    await coordinator.close()


# 功能：验证 v2 streaming encoder 在 1 MiB 精确接受并在 limit+1 立即拒绝
# 设计：先测空 message wrapper 开销，再用 ASCII 线性补齐到精确 bytes，避免依赖字符数近似
def test_single_frame_byte_limit_exact_boundary() -> None:
    event = LogLineEvent(
        run_id="run-1",
        level="INFO",
        source="test",
        message="",
        ts="2026-07-21T00:00:00Z",
    )
    base = journal_module._build_v2_record(
        event_id="evt-fixed",
        stream_id="run:run-1",
        seq=1,
        event=event.model_dump(mode="json"),
    )
    remaining = journal_module.JOURNAL_MAX_SINGLE_FRAME_BYTES - len(base.serialized)
    event.message = "a" * remaining

    exact = journal_module._build_v2_record(
        event_id="evt-fixed",
        stream_id="run:run-1",
        seq=1,
        event=event.model_dump(mode="json"),
    )
    assert len(exact.serialized) == journal_module.JOURNAL_MAX_SINGLE_FRAME_BYTES

    event.message += "a"
    with pytest.raises(JournalCapacityError, match="1 MiB"):
        journal_module._build_v2_record(
            event_id="evt-fixed",
            stream_id="run:run-1",
            seq=1,
            event=event.model_dump(mode="json"),
        )


# 功能：验证 journal queue bytes 精确容纳一条 frame，再增加一条立即拒绝
# 设计：阻塞首批 flush 保留 in-flight reservation，将 byte cap 设为已知序列化长度以锁定等号边界
async def test_queue_byte_limit_counts_inflight_exact_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entered = threading.Event()
    release = threading.Event()
    original = EventWriter.append_and_flush
    event = _run_started()
    exact_record = journal_module._build_v2_record(
        event_id="evt-" + "a" * 32,
        stream_id="run:run-1",
        seq=1,
        event=event.model_dump(mode="json"),
    )

    # 阻塞已出队 batch，证明 byte reservation 在 durable 前不释放
    def blocked_append(writer: EventWriter, rows: Iterable[bytes]) -> None:
        entered.set()
        release.wait(timeout=5)
        original(writer, rows)

    monkeypatch.setattr(EventWriter, "append_and_flush", blocked_append)
    monkeypatch.setattr(
        journal_module,
        "JOURNAL_QUEUE_MAX_BYTES",
        len(exact_record.serialized),
    )
    coordinator = EventJournalCoordinator()
    await coordinator.register_run("run-1", tmp_path / "run-1", session_id=None)
    await coordinator.handle(event)
    assert await asyncio.to_thread(entered.wait, 1)
    assert coordinator.stream_metrics("run:run-1")["reserved_bytes"] == len(
        exact_record.serialized
    )

    with pytest.raises(JournalCapacityError, match="capacity"):
        await coordinator.handle(event)

    release.set()
    await coordinator.close()


# 功能：验证 shutdown timeout 记录真实 pending stream，且 close 在底层 append 终态前不返回
# 设计：阻塞 to_thread append 并将 timeout 置零，以日志 Event 作为门闩核对无 late write
async def test_shutdown_timeout_joins_inflight_append_before_return(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entered = threading.Event()
    release = threading.Event()
    timeout_logged = asyncio.Event()
    timeout_args: list[tuple[object, ...]] = []
    original_append = EventWriter.append_and_flush
    original_error = journal_module.logger.error

    # 阻塞真实 append，直到测试确认 close 仍在等待
    def blocked_append(writer: EventWriter, rows: Iterable[bytes]) -> None:
        entered.set()
        release.wait(timeout=5)
        original_append(writer, rows)

    # 捕获 shutdown timeout 记录时点与结构化参数
    def record_error(message: str, *args: object, **kwargs: object) -> None:
        if message.startswith("journal shutdown timeout"):
            timeout_args.append(args)
            timeout_logged.set()
        original_error(message, *args, **kwargs)

    monkeypatch.setattr(EventWriter, "append_and_flush", blocked_append)
    monkeypatch.setattr(journal_module, "JOURNAL_SHUTDOWN_TIMEOUT_S", 0.0)
    monkeypatch.setattr(journal_module.logger, "error", record_error)
    coordinator = EventJournalCoordinator()
    run_path = tmp_path / "run-1"
    await coordinator.register_run("run-1", run_path, session_id=None)
    await coordinator.handle(_run_started())
    assert await asyncio.to_thread(entered.wait, 1)
    close_task = asyncio.create_task(coordinator.close())

    try:
        await asyncio.wait_for(timeout_logged.wait(), timeout=1.0)
        await asyncio.sleep(0)
        assert not close_task.done()
        assert not (run_path / "events.v2.jsonl").exists()
    finally:
        release.set()
        await asyncio.gather(close_task, return_exceptions=True)

    assert (run_path / "events.v2.jsonl").exists()
    assert timeout_args
    assert "run:run-1" in str(timeout_args[0])


# 功能：验证 run.finished durable terminal 后拒绝 late event 且不会重开已关闭 stream
# 设计：先完成真实 terminal flush，再尝试相同 run 的新 run.started并比较 v2 行数不变
async def test_terminal_stream_rejects_late_event_without_reopen(tmp_path: Path) -> None:
    run_path = tmp_path / "run-1"
    coordinator = EventJournalCoordinator()
    await coordinator.register_run("run-1", run_path, session_id=None)
    await coordinator.handle(_run_started())
    await coordinator.handle(
        RunFinishedEvent(
            run_id="run-1",
            status="success",
            steps=1,
            ts="2026-07-21T00:00:01Z",
        )
    )
    before = (run_path / "events.v2.jsonl").read_bytes()

    with pytest.raises(JournalError, match="closed"):
        await coordinator.handle(_run_started())

    assert (run_path / "events.v2.jsonl").read_bytes() == before
    assert coordinator.stream_lifecycle("run:run-1") is StreamLifecycle.CLOSED
    await coordinator.close()


# 功能：验证 10000 个 token event enqueue/flush 期间 event loop heartbeat 持续推进
# 设计：独立零休眠 heartbeat 与分批 flush 并行，只断言出现进展而不建立机器相关性能阈值
async def test_ten_thousand_token_events_keep_event_loop_responsive(
    tmp_path: Path,
) -> None:
    from kama_claude.core.bus.events import LlmTokenEvent

    coordinator = EventJournalCoordinator()
    await coordinator.register_run("run-1", tmp_path / "run-1", session_id=None)
    stop = asyncio.Event()
    ticks = 0

    # 用零休眠协程记录 journal enqueue/flush 期间 event loop 是否持续调度
    async def heartbeat() -> None:
        nonlocal ticks
        while not stop.is_set():
            ticks += 1
            await asyncio.sleep(0)

    heartbeat_task = asyncio.create_task(heartbeat())
    for index in range(10_000):
        await coordinator.handle(
            LlmTokenEvent(
                run_id="run-1",
                token=str(index),
                ts="2026-07-21T00:00:00Z",
            )
        )
        if (index + 1) % 2048 == 0:
            await coordinator.flush_all()
    await coordinator.flush_all()
    stop.set()
    await heartbeat_task
    await coordinator.close()

    assert ticks > 0


# 功能：验证一个目标 stream 容量耗尽只降级该 stream，其他未入队目标保持 OPEN
# 设计：在双 stream 原子 reservation 前只预占 session frame cap，断言事件零部分入队且 run lifecycle 不受牵连
async def test_capacity_failure_degrades_only_limiting_stream(tmp_path: Path) -> None:
    coordinator = EventJournalCoordinator()
    await coordinator.register_session("sess-1", tmp_path / "session")
    await coordinator.register_run(
        "run-1",
        tmp_path / "session" / "runs" / "run-1",
        session_id="sess-1",
    )
    session_state = cast(Any, coordinator)._streams["session:sess-1"]
    session_state.reserved_frames = journal_module.JOURNAL_QUEUE_MAX_FRAMES

    with pytest.raises(JournalCapacityError, match="capacity"):
        await coordinator.handle(_run_started())

    assert coordinator.stream_lifecycle("run:run-1") is StreamLifecycle.OPEN
    assert coordinator.stream_lifecycle("session:sess-1") is StreamLifecycle.DEGRADED
    assert coordinator.stream_metrics("run:run-1")["reserved_frames"] == 0
    await coordinator.close()


@pytest.mark.parametrize(
    "mutate",
    [
        lambda row: row.update(seq=True),
        lambda row: row.update(seq=1.0),
        lambda row: row.update(unexpected="wrapper"),
        lambda row: row["event"].update(unexpected="event"),
    ],
)
# 功能：验证 v2 wrapper 与 Event 使用 strict 类型并拒绝所有未知字段
# 设计：逐次只变异 bool/float seq、wrapper extra 或 event extra，证明 parser 不依赖 Python 宽松相等和 Pydantic 默认忽略
async def test_v2_wrapper_and_event_schema_are_strict(
    tmp_path: Path,
    mutate: Any,
) -> None:
    run_path = tmp_path / "run-strict"
    run_path.mkdir(parents=True)
    row: dict[str, Any] = {
        "schema_version": 2,
        "event_id": "evt-1",
        "stream_id": "run:run-strict",
        "seq": 1,
        "event": _run_started("run-strict").model_dump(mode="json"),
    }
    mutate(row)
    (run_path / "events.v2.jsonl").write_text(
        json.dumps(row) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(JournalCorruptionError):
        await EventJournalCoordinator().register_run(
            "run-strict",
            run_path,
            session_id=None,
        )


# 功能：验证 v2 event_id 不得复用同 stream legacy prefix 的 synthetic identity
# 设计：先通过真实 legacy replay取得 deterministic identity，再把它写入 seq=2 wrapper并要求 owner open fail closed
async def test_v2_identity_cannot_duplicate_legacy_prefix(tmp_path: Path) -> None:
    run_path = tmp_path / "run-cross-identity"
    run_path.mkdir(parents=True)
    (run_path / "events.jsonl").write_text(
        _run_started("run-cross-identity").model_dump_json() + "\n",
        encoding="utf-8",
    )
    first = EventJournalCoordinator()
    await first.register_run("run-cross-identity", run_path, session_id=None)
    replay = await first.read_replay(
        "run:run-cross-identity",
        after_seq=0,
        high_watermark=1,
    )
    await first.close()
    duplicate_id = replay.records[0].event_id
    row = {
        "schema_version": 2,
        "event_id": duplicate_id,
        "stream_id": "run:run-cross-identity",
        "seq": 2,
        "event": _run_started("run-cross-identity").model_dump(mode="json"),
    }
    (run_path / "events.v2.jsonl").write_text(
        json.dumps(row) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(JournalCorruptionError, match="duplicate event identity"):
        await EventJournalCoordinator().register_run(
            "run-cross-identity",
            run_path,
            session_id=None,
        )


@pytest.mark.parametrize("tail", [b"", b'{\"schema_version\":2'])
# 功能：验证 owner open 会规范化合法无换行末行或移除无效 crash tail 后再安全续写
# 设计：从一个完整 seq=1 row 构造两类无换行尾部，注册后追加 seq=2并逐行解析，防止 bytes 拼接破坏 journal
async def test_v2_tail_is_normalized_before_append(tmp_path: Path, tail: bytes) -> None:
    run_path = tmp_path / "run-tail"
    run_path.mkdir(parents=True)
    row = {
        "schema_version": 2,
        "event_id": "evt-1",
        "stream_id": "run:run-tail",
        "seq": 1,
        "event": _run_started("run-tail").model_dump(mode="json"),
    }
    first_row = json.dumps(row, separators=(",", ":")).encode()
    initial = first_row if not tail else first_row + b"\n" + tail
    (run_path / "events.v2.jsonl").write_bytes(initial)
    coordinator = EventJournalCoordinator()
    await coordinator.register_run("run-tail", run_path, session_id=None)
    await coordinator.handle(_run_started("run-tail"))
    await coordinator.flush_all()
    await coordinator.close()

    rows = _read_rows(run_path / "events.v2.jsonl")
    assert [row["seq"] for row in rows] == [1, 2]
