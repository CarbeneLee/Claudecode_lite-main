from inventory import Product, total_units


def test_total_units() -> None:
    products = [Product("pen", 3), Product("notebook", 4)]

    assert total_units(products) == 7
