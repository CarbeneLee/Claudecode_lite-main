from collections.abc import Mapping

from configkit.defaults import DEFAULT_SETTINGS
from configkit.merge import merge_layers
from configkit.normalize import normalize_environment, normalize_mapping


def resolve_settings(
    *,
    user: Mapping[str, object] | None = None,
    project: Mapping[str, object] | None = None,
    environ: Mapping[str, str] | None = None,
) -> dict[str, object]:
    return merge_layers(
        DEFAULT_SETTINGS,
        normalize_mapping(user or {}),
        normalize_mapping(project or {}),
        normalize_environment(environ or {}),
    )
