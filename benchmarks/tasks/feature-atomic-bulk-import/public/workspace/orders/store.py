from collections.abc import Iterable

from orders.errors import DuplicateOrderError, StoreCapacityError
from orders.models import Order


class InMemoryOrderStore:
    def __init__(self, *, max_orders: int | None = None) -> None:
        if max_orders is not None and max_orders < 0:
            raise ValueError("max_orders must be non-negative")
        self._max_orders = max_orders
        self._orders: dict[str, Order] = {}

    def add(self, order: Order) -> None:
        if order.order_id in self._orders:
            raise DuplicateOrderError(order.order_id)
        if self._max_orders is not None and len(self._orders) >= self._max_orders:
            raise StoreCapacityError("store capacity reached")
        self._orders[order.order_id] = order

    def get(self, order_id: str) -> Order:
        return self._orders[order_id]

    def all_orders(self) -> tuple[Order, ...]:
        return tuple(self._orders.values())

    def replace_all(self, orders: Iterable[Order]) -> None:
        replacement: dict[str, Order] = {}
        for order in orders:
            if order.order_id in replacement:
                raise DuplicateOrderError(order.order_id)
            replacement[order.order_id] = order
        if self._max_orders is not None and len(replacement) > self._max_orders:
            raise StoreCapacityError("store capacity reached")
        self._orders = replacement
