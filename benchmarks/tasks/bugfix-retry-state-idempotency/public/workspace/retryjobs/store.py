from __future__ import annotations

from retryjobs.models import Job


class JobStore:
    # 初始化内存任务存储
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}

    # 新建并保存任务
    def create(self, job_id: str) -> Job:
        if job_id in self._jobs:
            raise ValueError("job already exists")
        job = Job(job_id=job_id)
        self._jobs[job_id] = job
        return job

    # 按 ID 返回已有任务
    def get(self, job_id: str) -> Job:
        return self._jobs[job_id]

    # 保存任务最新状态
    def save(self, job: Job) -> None:
        self._jobs[job.job_id] = job
