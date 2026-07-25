from __future__ import annotations

from collections.abc import Collection
from enum import StrEnum


class FailureCategory(StrEnum):
    NONE = "none"
    TASK_FAILED = "task_failed"
    RUNTIME_FAILED = "runtime_failed"
    TRACE_INVALID = "trace_invalid"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"
    GRADER_ERROR = "grader_error"
    INFRA_ERROR = "infra_error"


FAILURE_PRIORITY: tuple[FailureCategory, ...] = (
    FailureCategory.INFRA_ERROR,
    FailureCategory.GRADER_ERROR,
    FailureCategory.TIMEOUT,
    FailureCategory.CANCELLED,
    FailureCategory.TRACE_INVALID,
    FailureCategory.RUNTIME_FAILED,
    FailureCategory.TASK_FAILED,
    FailureCategory.NONE,
)

if len(FAILURE_PRIORITY) != len(set(FAILURE_PRIORITY)) or set(FAILURE_PRIORITY) != set(
    FailureCategory
):
    raise RuntimeError("failure priority must contain every category exactly once")


# 按中央优先级从候选失败集合中选择唯一终态类别
def select_failure_category(categories: Collection[FailureCategory]) -> FailureCategory:
    candidates = set(categories)
    for category in FAILURE_PRIORITY:
        if category in candidates:
            return category
    return FailureCategory.NONE
