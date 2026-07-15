from __future__ import annotations


class TaskError(ValueError):
    """Base exception for task domain failures."""


class TaskNotFoundError(TaskError):
    """Raised when a requested task does not exist."""


class TaskValidationError(TaskError):
    """Raised when a task business rule rejects an input."""
