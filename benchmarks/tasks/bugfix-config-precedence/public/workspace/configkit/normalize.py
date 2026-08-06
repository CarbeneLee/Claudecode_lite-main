from collections.abc import Mapping

from configkit.errors import UnknownSettingError

_ALLOWED_SETTINGS = frozenset({"timeout_s", "verbose", "profile"})


def normalize_mapping(values: Mapping[str, object]) -> dict[str, object]:
    unknown = set(values) - _ALLOWED_SETTINGS
    if unknown:
        raise UnknownSettingError(f"unknown setting: {sorted(unknown)[0]}")
    return dict(values)


def normalize_environment(environ: Mapping[str, str]) -> dict[str, object]:
    normalized: dict[str, object] = {}
    if "APP_TIMEOUT_S" in environ:
        normalized["timeout_s"] = int(environ["APP_TIMEOUT_S"])
    if "APP_VERBOSE" in environ:
        value = environ["APP_VERBOSE"].strip().lower()
        if value not in {"true", "false"}:
            raise ValueError("APP_VERBOSE must be true or false")
        normalized["verbose"] = value == "true"
    if "APP_PROFILE" in environ:
        normalized["profile"] = environ["APP_PROFILE"]
    return normalized
