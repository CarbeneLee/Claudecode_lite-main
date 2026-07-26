from __future__ import annotations

from collections.abc import Sequence

from dependency_planner.errors import DuplicateTaskError
from dependency_planner.models import Task


# 错误实现：只校验重复名称而忽略未知依赖
def validate_tasks(tasks: Sequence[Task]) -> tuple[Task, ...]:
    ordered = tuple(tasks)
    by_name: dict[str, Task] = {}
    for task in ordered:
        if task.name in by_name:
            raise DuplicateTaskError(task.name)
        by_name[task.name] = task
    return ordered
