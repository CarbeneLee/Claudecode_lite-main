from calculator import add, multiply


def test_existing_operations_remain_unchanged() -> None:
    assert add(10, 5) == 15
    assert multiply(6, 7) == 42
