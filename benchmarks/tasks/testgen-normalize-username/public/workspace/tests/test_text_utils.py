from text_utils import normalize_username


def test_normalize_username_trims_and_lowercases() -> None:
    assert normalize_username(" Alice ") == "alice"
