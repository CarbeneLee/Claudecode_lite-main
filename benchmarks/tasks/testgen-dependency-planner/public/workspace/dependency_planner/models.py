from dataclasses import dataclass


@dataclass(frozen=True)
class Task:
    name: str
    dependencies: tuple[str, ...] = ()
