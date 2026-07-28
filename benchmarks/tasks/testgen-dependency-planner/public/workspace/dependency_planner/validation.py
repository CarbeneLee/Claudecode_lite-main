from __future__ import annotations

from collections.abc import Sequence

from dependency_planner.errors import DuplicateTaskError, UnknownDependencyError
from dependency_planner.models import Task


# 校验名称唯一和依赖存在并返回稳定任务副本
def validate_tasks(tasks: Sequence[Task]) -> tuple[Task, ...]:
    ordered = tuple(tasks)
    by_name: dict[str, Task] = {}
    for task in ordered:
        if task.name in by_name:
            raise DuplicateTaskError(task.name)
        by_name[task.name] = task

    for task in ordered:
        for dependency in task.dependencies:
            if dependency not in by_name:
                raise UnknownDependencyError(dependency)
    return ordered
