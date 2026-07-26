from __future__ import annotations

from collections.abc import Sequence

from dependency_planner.errors import DependencyCycleError
from dependency_planner.models import Task
from dependency_planner.validation import validate_tasks


# 错误实现：为稳定输出先原地排序调用方传入的列表
def plan(tasks: Sequence[Task]) -> list[str]:
    if isinstance(tasks, list):
        tasks.sort(key=lambda task: task.name)
    ordered = validate_tasks(tasks)
    completed: set[str] = set()
    result: list[str] = []
    while len(completed) < len(ordered):
        progressed = False
        for task in ordered:
            if task.name in completed:
                continue
            if all(name in completed for name in task.dependencies):
                completed.add(task.name)
                result.append(task.name)
                progressed = True
        if not progressed:
            raise DependencyCycleError("dependency cycle")
    return result
