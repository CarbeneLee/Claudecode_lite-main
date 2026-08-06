from collections.abc import Mapping

from orders.models import Order


def parse_order_row(row: Mapping[str, str]) -> Order:
    try:
        order_id = row["order_id"].strip()
        sku = row["sku"].strip()
        quantity = int(row["quantity"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("invalid order row") from exc
    if not order_id or not sku or quantity <= 0:
        raise ValueError("invalid order row")
    return Order(order_id=order_id, sku=sku, quantity=quantity)
