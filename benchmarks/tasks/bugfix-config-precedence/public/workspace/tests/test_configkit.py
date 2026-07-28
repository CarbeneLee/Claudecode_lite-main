import pytest
from configkit import UnknownSettingError, resolve_settings


def test_defaults_are_available_without_overrides() -> None:
    assert resolve_settings() == {
        "timeout_s": 30,
        "verbose": False,
        "profile": "standard",
    }


def test_environment_has_highest_precedence_for_truthy_values() -> None:
    resolved = resolve_settings(
        user={"timeout_s": 10},
        project={"timeout_s": 20, "profile": "project"},
        environ={"APP_TIMEOUT_S": "45"},
    )

    assert resolved == {
        "timeout_s": 45,
        "verbose": False,
        "profile": "project",
    }


def test_unknown_file_setting_is_rejected() -> None:
    with pytest.raises(UnknownSettingError):
        resolve_settings(user={"retries": 3})
