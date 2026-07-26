from collections.abc import Iterable

from inventory.models import Product


def total_units(products: Iterable[Product]) -> int:
    return sum(product.quantity for product in products)
