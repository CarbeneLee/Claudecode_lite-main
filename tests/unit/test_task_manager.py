from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

from kama_claude.core.task.errors import (
    TaskError,
    TaskNotFoundError,
    TaskValidationError,
)
from kama_claude.core.task.manager import TaskManager
from kama_claude.core.task.model import TaskStatus


# 功能：验证 task domain exceptions 存在并保持 ValueError 兼容关系
# 设计：直接锁定两类具体异常经 TaskError 继承 ValueError 的兼容契约
def test_task_domain_errors_are_value_errors() -> None:
    assert issubclass(TaskError, ValueError)
    assert issubclass(TaskNotFoundError, TaskError)
    assert issubclass(TaskValidationError, TaskError)


# 功能：验证 create 写入 JSON 文件并返回正确的 Task 对象
# 设计：用 tmp_path 隔离文件系统，断言文件存在且字段值正确
def test_create_writes_file(tmp_path: Path) -> None:
    mgr = TaskManager(tmp_path)
    task = mgr.create("do something")
    assert task.id == 1
    assert task.subject == "do something"
    assert task.status == "pending"
    assert (tmp_path / "task_1.json").exists()


# 功能：验证多次 create 的 ID 递增
# 设计：连续创建两个任务，断言 ID 分别为 1 和 2
def test_create_increments_id(tmp_path: Path) -> None:
    mgr = TaskManager(tmp_path)
    t1 = mgr.create("first")
    t2 = mgr.create("second")
    assert t1.id == 1
    assert t2.id == 2


# 功能：验证 create 传入不存在的 blocked_by 抛出 TaskValidationError
# 设计：blocked_by=[99] 引用不存在的任务，锁定业务输入与资源不存在的类型边界
def test_create_invalid_blocked_by_raises(tmp_path: Path) -> None:
    mgr = TaskManager(tmp_path)
    with pytest.raises(TaskValidationError, match="not found"):
        mgr.create("dependent", blocked_by=[99])


# 功能：验证 get 返回正确的 Task
# 设计：create 后立即 get，断言 subject 一致
def test_get_returns_task(tmp_path: Path) -> None:
    mgr = TaskManager(tmp_path)
    mgr.create("hello")
    task = mgr.get(1)
    assert task.subject == "hello"


# 功能：验证 get 不存在的 ID 抛出 TaskNotFoundError
# 设计：不创建任何任务，直接 get(999)，锁定资源缺失的 domain exception
def test_get_nonexistent_raises(tmp_path: Path) -> None:
    mgr = TaskManager(tmp_path)
    with pytest.raises(TaskNotFoundError):
        mgr.get(999)


# 功能：验证 update 非法业务状态抛出 TaskValidationError
# 设计：用 cast 绕过静态 Literal 边界模拟外部坏输入，断言 manager 不靠字符串分类
def test_update_invalid_status_raises_task_validation_error(tmp_path: Path) -> None:
    mgr = TaskManager(tmp_path)
    mgr.create("work")

    with pytest.raises(TaskValidationError, match="invalid status"):
        mgr.update(1, status=cast(TaskStatus, "paused"))


# 功能：验证 update 修改 status 并写回文件
# 设计：create 后 update status="in_progress"，重新 get 断言状态已变更
def test_update_status(tmp_path: Path) -> None:
    mgr = TaskManager(tmp_path)
    mgr.create("work")
    mgr.update(1, status="in_progress")
    assert mgr.get(1).status == "in_progress"


# 功能：验证 update status="completed" 会从其他任务的 blocked_by 中清除该 ID
# 设计：创建任务 1，再创建被 1 阻塞的任务 2，完成任务 1 后断言任务 2 的 blocked_by 为空
def test_update_completed_clears_dependency(tmp_path: Path) -> None:
    mgr = TaskManager(tmp_path)
    mgr.create("step 1")
    mgr.create("step 2", blocked_by=[1])
    mgr.update(1, status="completed")
    assert mgr.get(2).blocked_by == []


# 功能：验证 update add_blocked_by 正确追加依赖
# 设计：先创建两个任务，再为任务 2 追加对任务 1 的依赖
def test_update_add_blocked_by(tmp_path: Path) -> None:
    mgr = TaskManager(tmp_path)
    mgr.create("a")
    mgr.create("b")
    mgr.update(2, add_blocked_by=[1])
    assert 1 in mgr.get(2).blocked_by


# 功能：验证 update 拒绝不存在的依赖且所有相关 task 文件完全不变
# 设计：同时请求 completed 与缺失依赖，证明校验先于 status、updated_at、clear dependency 和 save
def test_update_missing_blocked_by_is_atomic(tmp_path: Path) -> None:
    mgr = TaskManager(tmp_path)
    mgr.create("step 1")
    mgr.create("step 2", blocked_by=[1])
    task_1_path = tmp_path / "task_1.json"
    task_2_path = tmp_path / "task_2.json"
    task_1_before = task_1_path.read_text(encoding="utf-8")
    task_2_before = task_2_path.read_text(encoding="utf-8")

    with pytest.raises(TaskValidationError, match="blocked_by task 999 not found"):
        mgr.update(1, status="completed", add_blocked_by=[999])

    assert task_1_path.read_text(encoding="utf-8") == task_1_before
    assert task_2_path.read_text(encoding="utf-8") == task_2_before
    assert mgr.get(1).status == "pending"
    assert mgr.get(1).blocked_by == []
    assert mgr.get(2).blocked_by == [1]


# 功能：验证 update remove_blocked_by 正确移除依赖
# 设计：创建带依赖的任务，再移除依赖，断言 blocked_by 为空
def test_update_remove_blocked_by(tmp_path: Path) -> None:
    mgr = TaskManager(tmp_path)
    mgr.create("a")
    mgr.create("b", blocked_by=[1])
    mgr.update(2, remove_blocked_by=[1])
    assert mgr.get(2).blocked_by == []


# 功能：验证 list_all 返回所有任务，按 ID 升序排列
# 设计：创建三个任务后 list_all，断言数量为 3 且 ID 顺序正确
def test_list_all_ordered(tmp_path: Path) -> None:
    mgr = TaskManager(tmp_path)
    mgr.create("x")
    mgr.create("y")
    mgr.create("z")
    tasks = mgr.list_all()
    assert len(tasks) == 3
    assert [t.id for t in tasks] == [1, 2, 3]


# 功能：验证 format_list 输出包含状态标记和任务名
# 设计：创建两个任务并更新其中一个，检查 format_list 字符串内容
def test_format_list_content(tmp_path: Path) -> None:
    mgr = TaskManager(tmp_path)
    mgr.create("alpha")
    mgr.create("beta")
    mgr.update(1, status="completed")
    result = mgr.format_list()
    assert "[x]" in result
    assert "alpha" in result
    assert "beta" in result


# 功能：验证 TaskManager 重新实例化时能从现有文件恢复 next_id
# 设计：第一个 mgr 创建 2 个任务，第二个 mgr 读取同目录，新任务 ID 应为 3
def test_manager_resumes_id_from_existing_files(tmp_path: Path) -> None:
    mgr1 = TaskManager(tmp_path)
    mgr1.create("first")
    mgr1.create("second")

    mgr2 = TaskManager(tmp_path)
    task = mgr2.create("third")
    assert task.id == 3
