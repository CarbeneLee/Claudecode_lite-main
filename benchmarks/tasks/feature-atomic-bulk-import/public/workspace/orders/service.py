from collections.abc import Mapping

from orders.models import Order
from orders.parser import parse_order_row
from orders.store import InMemoryOrderStore


def create_order(
    store: InMemoryOrderStore,
    row: Mapping[str, str],
) -> Order:
    order = parse_order_row(row)
    store.add(order)
    return order
