from dependency_planner import Task, plan


# 功能：验证单个无依赖任务可以被规划
# 设计：保留最小 happy path，使新增测试质量由 private oracle 判断
def test_single_task() -> None:
    assert plan([Task("build")]) == ["build"]
