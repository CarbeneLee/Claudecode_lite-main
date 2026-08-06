from __future__ import annotations

from collections.abc import Sequence

from dependency_planner.errors import UnknownDependencyError
from dependency_planner.models import Task


# 错误实现：重复名称静默覆盖而不报告冲突
def validate_tasks(tasks: Sequence[Task]) -> tuple[Task, ...]:
    ordered = tuple(tasks)
    by_name = {task.name: task for task in ordered}
    for task in ordered:
        for dependency in task.dependencies:
            if dependency not in by_name:
                raise UnknownDependencyError(dependency)
    return ordered
