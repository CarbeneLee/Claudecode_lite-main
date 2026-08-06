from retryjobs import (
    ExecutionResult,
    JobScheduler,
    JobState,
    JobStore,
    RetryPolicy,
    ScriptedExecutor,
)


# 构造带单个任务和确定性结果脚本的调度器
def _scheduler(
    job_id: str,
    outcomes: list[ExecutionResult],
    *,
    max_attempts: int = 3,
) -> tuple[JobScheduler, JobStore, ScriptedExecutor]:
    store = JobStore()
    store.create(job_id)
    executor = ScriptedExecutor({job_id: outcomes})
    return (
        JobScheduler(store, executor, RetryPolicy(max_attempts=max_attempts)),
        store,
        executor,
    )


# 功能：验证短暂失败后成功会记录两次调用且只提交一次效果
# 设计：把 attempt、executor call 和 committed effect 绑定到同一轨迹
def test_retry_then_success_counts_every_invocation_and_commits_once() -> None:
    scheduler, _, executor = _scheduler(
        "invoice",
        [
            ExecutionResult(False, retryable=True, error="busy"),
            ExecutionResult(True),
        ],
    )

    result = scheduler.run("invoice")

    assert result.state is JobState.SUCCEEDED
    assert result.attempts == 2
    assert result.last_error is None
    assert executor.calls["invoice"] == 2
    assert executor.committed_effects["invoice"] == 1


# 功能：验证连续短暂失败严格停在配置的最大调用次数
# 设计：准备多于上限的失败结果以暴露 max+1 的边界错误
def test_retryable_failures_stop_at_maximum_attempts() -> None:
    scheduler, _, executor = _scheduler(
        "sync",
        [ExecutionResult(False, retryable=True, error="busy")] * 4,
        max_attempts=3,
    )

    result = scheduler.run("sync")

    assert result.state is JobState.FAILED
    assert result.attempts == 3
    assert executor.calls["sync"] == 3
    assert executor.committed_effects["sync"] == 0


# 功能：验证永久失败不重试且仍记录本次执行
# 设计：在永久失败后准备成功结果，调用数可证明未消费后续结果
def test_permanent_failure_stops_immediately() -> None:
    scheduler, _, executor = _scheduler(
        "charge",
        [
            ExecutionResult(False, retryable=False, error="invalid"),
            ExecutionResult(True),
        ],
    )

    result = scheduler.run("charge")

    assert result.state is JobState.FAILED
    assert result.attempts == 1
    assert result.last_error == "invalid"
    assert executor.calls["charge"] == 1
    assert executor.committed_effects["charge"] == 0


# 功能：验证成功终态重复运行不会新增调用或重复副作用
# 设计：连续运行同一任务并同时比较对象、attempt、call 和 effect
def test_terminal_success_is_idempotent() -> None:
    scheduler, _, executor = _scheduler(
        "publish",
        [ExecutionResult(True), ExecutionResult(True)],
    )

    first = scheduler.run("publish")
    second = scheduler.run("publish")

    assert second is first
    assert second.state is JobState.SUCCEEDED
    assert second.attempts == 1
    assert executor.calls["publish"] == 1
    assert executor.committed_effects["publish"] == 1


# 功能：验证失败终态重复运行不会继续消耗执行结果
# 设计：永久失败后保留成功脚本，证明第二次 run 未重新执行
def test_terminal_failure_is_idempotent() -> None:
    scheduler, _, executor = _scheduler(
        "delete",
        [
            ExecutionResult(False, retryable=False, error="denied"),
            ExecutionResult(True),
        ],
    )

    scheduler.run("delete")
    result = scheduler.run("delete")

    assert result.state is JobState.FAILED
    assert result.attempts == 1
    assert executor.calls["delete"] == 1
    assert executor.committed_effects["delete"] == 0


# 功能：验证一个任务的重试状态不会污染另一个任务
# 设计：在共享 scheduler 中交错两条不同轨迹并逐项比较最终状态
def test_jobs_keep_independent_state_and_effects() -> None:
    store = JobStore()
    store.create("alpha")
    store.create("beta")
    executor = ScriptedExecutor(
        {
            "alpha": [
                ExecutionResult(False, retryable=True, error="busy"),
                ExecutionResult(True),
            ],
            "beta": [ExecutionResult(True)],
        }
    )
    scheduler = JobScheduler(store, executor, RetryPolicy(max_attempts=3))

    scheduler.run("alpha")
    scheduler.run("beta")

    assert store.get("alpha").attempts == 2
    assert store.get("beta").attempts == 1
    assert executor.committed_effects == {"alpha": 1, "beta": 1}
