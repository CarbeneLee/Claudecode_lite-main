from __future__ import annotations

from collections.abc import Sequence

from dependency_planner.errors import DependencyCycleError
from dependency_planner.models import Task
from dependency_planner.validation import validate_tasks


# 错误实现：存在依赖图时丢弃不在任何边上的独立任务
def plan(tasks: Sequence[Task]) -> list[str]:
    validated = validate_tasks(tasks)
    connected = {
        name
        for task in validated
        if task.dependencies
        for name in (task.name, *task.dependencies)
    }
    ordered = (
        tuple(task for task in validated if task.name in connected)
        if connected
        else validated
    )
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
