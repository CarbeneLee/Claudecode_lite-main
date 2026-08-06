from __future__ import annotations

from retryjobs.executor import ScriptedExecutor
from retryjobs.models import Job, JobState
from retryjobs.policy import RetryPolicy
from retryjobs.store import JobStore


class JobScheduler:
    # 组装任务存储、执行器和重试策略
    def __init__(
        self,
        store: JobStore,
        executor: ScriptedExecutor,
        policy: RetryPolicy,
    ) -> None:
        self._store = store
        self._executor = executor
        self._policy = policy

    # 执行任务并根据结果推进其生命周期
    def run(self, job_id: str) -> Job:
        job = self._store.get(job_id)
        if job.is_terminal():
            return job

        while job.attempts < self._policy.max_attempts:
            job.state = JobState.RUNNING
            result = self._executor.execute(job.job_id)
            if result.success:
                job.state = JobState.SUCCEEDED
                job.last_error = None
                self._store.save(job)
                return job

            job.attempts += 1
            job.last_error = result.error
            if not result.retryable or job.attempts >= self._policy.max_attempts:
                job.state = JobState.FAILED
                self._store.save(job)
                return job

        job.state = JobState.FAILED
        self._store.save(job)
        return job
