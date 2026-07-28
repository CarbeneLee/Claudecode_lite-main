import pytest
from inventory import Product, low_stock_names
from inventory.service import low_stock_names as service_low_stock_names


def test_low_stock_names_is_sorted_and_uses_strict_threshold() -> None:
    products = [
        Product("pencil", 1),
        Product("eraser", 5),
        Product("notebook", 2),
    ]

    assert low_stock_names(products, 5) == ["notebook", "pencil"]
    assert service_low_stock_names(products, 2) == ["pencil"]


def test_low_stock_names_rejects_negative_threshold() -> None:
    with pytest.raises(ValueError):
        low_stock_names([], -1)
