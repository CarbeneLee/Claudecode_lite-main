import pytest
from inventory import (
    InsufficientStock,
    InvalidQuantity,
    InvalidTransition,
    InventoryStore,
    RequestConflict,
    ReservationService,
    ReservationStatus,
)


# 构造带初始库存的 reservation 服务
def _service(
    *items: tuple[str, int],
) -> tuple[ReservationService, InventoryStore]:
    store = InventoryStore()
    for sku, available in items:
        store.add_item(sku, available)
    return ReservationService(store), store


# 功能：验证 reserve 扣减库存并创建 pending 记录
# 设计：联合断言 record identity、状态和 exact stock delta
def test_reserve_reduces_available_and_creates_pending_record() -> None:
    service, store = _service(("pen", 5))

    reservation = service.reserve("req-1", "pen", 2)

    assert reservation.reservation_id == "r-1"
    assert reservation.request_key == "req-1"
    assert reservation.status is ReservationStatus.PENDING
    assert store.get("pen").available == 3


# 功能：验证相同请求完全幂等而参数冲突不会修改库存
# 设计：分别改变 quantity 和 SKU，并在每次异常后检查双方库存
def test_request_key_idempotency_and_conflict() -> None:
    service, store = _service(("pen", 8), ("book", 4))
    first = service.reserve("req-1", "pen", 2)

    repeated = service.reserve("req-1", "pen", 2)
    assert repeated is first
    assert store.get("pen").available == 6

    with pytest.raises(RequestConflict):
        service.reserve("req-1", "pen", 3)
    with pytest.raises(RequestConflict):
        service.reserve("req-1", "book", 2)
    assert store.get("pen").available == 6
    assert store.get("book").available == 4


# 功能：验证不足和非法数量均不留下可观察 reservation 或库存变化
# 设计：失败后补货并复用同一 key，暴露不可直接读取的 ghost record
def test_failed_reserve_has_no_partial_mutation() -> None:
    service, store = _service(("pen", 1))

    with pytest.raises(InsufficientStock):
        service.reserve("req-low", "pen", 2)
    assert store.get("pen").available == 1

    store.adjust("pen", 2)
    reservation = service.reserve("req-low", "pen", 2)
    assert reservation.status is ReservationStatus.PENDING
    assert store.get("pen").available == 1

    with pytest.raises(InvalidQuantity):
        service.reserve("req-zero", "pen", 0)
    assert store.get("pen").available == 1


# 功能：验证 confirm 在同一终态重复调用时幂等且不改变库存
# 设计：比较两次返回 identity、终态和 reserve 后的 stock
def test_confirm_is_idempotent_without_stock_change() -> None:
    service, store = _service(("pen", 5))
    reservation = service.reserve("req-1", "pen", 2)

    first = service.confirm(reservation.reservation_id)
    second = service.confirm(reservation.reservation_id)

    assert first is second
    assert second.status is ReservationStatus.CONFIRMED
    assert store.get("pen").available == 3


# 功能：验证 release 恢复库存恰好一次
# 设计：重复 release 后断言库存回到初始值而不是增加两次
def test_release_is_idempotent_and_restores_exactly_once() -> None:
    service, store = _service(("pen", 5))
    reservation = service.reserve("req-1", "pen", 2)

    first = service.release(reservation.reservation_id)
    second = service.release(reservation.reservation_id)

    assert first is second
    assert second.status is ReservationStatus.RELEASED
    assert store.get("pen").available == 5


# 功能：验证 confirmed reservation 不得转为 released 或恢复库存
# 设计：同时断言异常、原终态和库存，排除先恢复后报错
def test_confirmed_reservation_cannot_be_released() -> None:
    service, store = _service(("pen", 5))
    reservation = service.reserve("req-1", "pen", 2)
    service.confirm(reservation.reservation_id)

    with pytest.raises(InvalidTransition):
        service.release(reservation.reservation_id)

    assert reservation.status is ReservationStatus.CONFIRMED
    assert store.get("pen").available == 3


# 功能：验证不同 SKU 和请求的 reservation 状态彼此隔离
# 设计：一条 release、一条 confirm，分别核对 record 和 stock
def test_reservations_keep_skus_and_requests_independent() -> None:
    service, store = _service(("pen", 5), ("book", 7))
    pen = service.reserve("pen-request", "pen", 2)
    book = service.reserve("book-request", "book", 3)

    service.release(pen.reservation_id)
    service.confirm(book.reservation_id)

    assert pen.status is ReservationStatus.RELEASED
    assert book.status is ReservationStatus.CONFIRMED
    assert store.get("pen").available == 5
    assert store.get("book").available == 4
