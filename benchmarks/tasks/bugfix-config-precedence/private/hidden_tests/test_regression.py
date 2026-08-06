import pytest
from configkit import UnknownSettingError, resolve_settings


def test_defaults_and_truthy_environment_precedence_remain_stable() -> None:
    assert resolve_settings() == {
        "timeout_s": 30,
        "verbose": False,
        "profile": "standard",
    }
    assert resolve_settings(
        user={"timeout_s": 8},
        project={"timeout_s": 16, "profile": "project"},
        environ={"APP_TIMEOUT_S": "60", "APP_VERBOSE": "true"},
    ) == {
        "timeout_s": 60,
        "verbose": True,
        "profile": "project",
    }


def test_validation_remains_active() -> None:
    with pytest.raises(UnknownSettingError):
        resolve_settings(project={"unknown": "value"})
    with pytest.raises(ValueError):
        resolve_settings(environ={"APP_VERBOSE": "sometimes"})
