from __future__ import annotations

from inventory.errors import UnknownReservation
from inventory.models import Reservation
from inventory.store import InventoryStore


class ReservationService:
    # 初始化 reservation 生命周期服务
    def __init__(self, store: InventoryStore) -> None:
        self._store = store
        self._reservations: dict[str, Reservation] = {}
        self._by_request: dict[str, str] = {}
        self._next_id = 1

    # 按 reservation ID 查询记录
    def get(self, reservation_id: str) -> Reservation:
        try:
            return self._reservations[reservation_id]
        except KeyError as exc:
            raise UnknownReservation(reservation_id) from exc

    # 预留库存并保证 request key 幂等
    def reserve(
        self,
        request_key: str,
        sku: str,
        quantity: int,
    ) -> Reservation:
        raise NotImplementedError

    # 确认一个待处理 reservation
    def confirm(self, reservation_id: str) -> Reservation:
        raise NotImplementedError

    # 释放一个待处理 reservation 并恢复库存
    def release(self, reservation_id: str) -> Reservation:
        raise NotImplementedError
