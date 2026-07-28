from retryjobs import (
    ExecutionResult,
    JobScheduler,
    JobState,
    JobStore,
    RetryPolicy,
    ScriptedExecutor,
)


# 功能：验证首次成功仍保持现有成功终态和一次效果
# 设计：联合断言状态、调用数和效果数，保护原有 happy path
def test_first_attempt_success_preserves_behavior() -> None:
    store = JobStore()
    store.create("email")
    executor = ScriptedExecutor({"email": [ExecutionResult(success=True)]})
    scheduler = JobScheduler(store, executor, RetryPolicy(max_attempts=3))

    result = scheduler.run("email")

    assert result.state is JobState.SUCCEEDED
    assert executor.calls["email"] == 1
    assert executor.committed_effects["email"] == 1


# 功能：验证任务创建和查询接口保持稳定
# 设计：用两个对象的 identity 排除跨任务存储污染
def test_store_keeps_jobs_independent() -> None:
    store = JobStore()
    first = store.create("first")
    second = store.create("second")

    assert store.get("first") is first
    assert store.get("second") is second
