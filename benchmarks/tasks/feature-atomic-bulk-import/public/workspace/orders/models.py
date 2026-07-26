from dataclasses import dataclass


@dataclass(frozen=True)
class Order:
    order_id: str
    sku: str
    quantity: int
