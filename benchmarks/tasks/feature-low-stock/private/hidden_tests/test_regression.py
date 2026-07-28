from inventory import Product, total_units


def test_total_units_and_product_remain_available() -> None:
    assert total_units([Product("pen", 2), Product("paper", 8)]) == 10
