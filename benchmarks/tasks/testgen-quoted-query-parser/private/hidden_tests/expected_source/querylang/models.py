from dataclasses import dataclass


@dataclass(frozen=True)
class Term:
    field: str
    value: str
