from collections.abc import Mapping


def merge_layers(
    defaults: Mapping[str, object],
    *overrides: Mapping[str, object],
) -> dict[str, object]:
    resolved = dict(defaults)
    for layer in overrides:
        for key, value in layer.items():
            if value:
                resolved[key] = value
    return resolved
