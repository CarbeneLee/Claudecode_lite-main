from calculator import subtract


def test_subtract_positive_and_negative_values() -> None:
    assert subtract(9, 4) == 5
    assert subtract(-2, 3) == -5
