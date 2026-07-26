from configkit import resolve_settings


def test_falsey_values_are_explicit_at_each_precedence_layer() -> None:
    assert resolve_settings(user={"timeout_s": 0})["timeout_s"] == 0
    assert (
        resolve_settings(
            user={"verbose": True},
            project={"verbose": False},
        )["verbose"]
        is False
    )
    assert resolve_settings(
        project={"timeout_s": 12, "verbose": True},
        environ={"APP_TIMEOUT_S": "0", "APP_VERBOSE": "false"},
    ) == {
        "timeout_s": 0,
        "verbose": False,
        "profile": "standard",
    }


def test_missing_and_none_values_inherit_lower_precedence() -> None:
    assert resolve_settings(
        user={"timeout_s": 9, "profile": "user"},
        project={"timeout_s": None},
        environ={},
    ) == {
        "timeout_s": 9,
        "verbose": False,
        "profile": "user",
    }
