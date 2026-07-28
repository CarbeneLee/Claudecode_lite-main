import pytest
from retryjobs import (
    ExecutionResult,
    JobScheduler,
    JobState,
    JobStore,
    RetryPolicy,
    ScriptedExecutor,
)


# 功能：验证首次成功仍只执行一次并进入成功终态
# 设计：只保护既有 happy path，不把新 attempt contract 混入 regression
def test_first_success_state_and_effect_remain_stable() -> None:
    store = JobStore()
    store.create("one")
    executor = ScriptedExecutor({"one": [ExecutionResult(True)]})

    result = JobScheduler(store, executor, RetryPolicy(2)).run("one")

    assert result.state is JobState.SUCCEEDED
    assert executor.calls["one"] == 1
    assert executor.committed_effects["one"] == 1


# 功能：验证存储拒绝重复任务且不同任务保持独立
# 设计：同时检查 identity 和重复创建异常，覆盖 store 原有边界
def test_store_contract_remains_stable() -> None:
    store = JobStore()
    first = store.create("first")
    second = store.create("second")

    assert store.get("first") is first
    assert store.get("second") is second
    with pytest.raises(ValueError, match="job already exists"):
        store.create("first")
