from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass


@dataclass(frozen=True)
class ExecutionResult:
    success: bool
    retryable: bool = False
    error: str | None = None


class ScriptedExecutor:
    # 初始化每个任务的确定性执行结果脚本
    def __init__(self, outcomes: dict[str, list[ExecutionResult]]) -> None:
        self._outcomes = {
            job_id: list(results) for job_id, results in outcomes.items()
        }
        self.calls: dict[str, int] = defaultdict(int)
        self.committed_effects: dict[str, int] = defaultdict(int)

    # 执行下一条脚本结果，并在成功时提交一次可观察效果
    def execute(self, job_id: str) -> ExecutionResult:
        call_index = self.calls[job_id]
        self.calls[job_id] += 1
        scripted = self._outcomes.get(job_id, [])
        result = (
            scripted[call_index]
            if call_index < len(scripted)
            else ExecutionResult(success=True)
        )
        if result.success:
            self.committed_effects[job_id] += 1
        return result
