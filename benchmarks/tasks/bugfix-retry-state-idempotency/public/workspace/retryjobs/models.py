from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class JobState(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


@dataclass
class Job:
    job_id: str
    state: JobState = JobState.PENDING
    attempts: int = 0
    last_error: str | None = None

    # 判断任务是否已经进入不可继续执行的终态
    def is_terminal(self) -> bool:
        return self.state in {JobState.SUCCEEDED, JobState.FAILED}
