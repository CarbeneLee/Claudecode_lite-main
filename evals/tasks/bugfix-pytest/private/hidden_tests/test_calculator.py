from calculator import add


# 功能：验证公开 fixture 中的 add 函数返回两个整数之和
# 设计：隐藏测试只断言外部行为，不向 Agent 暴露预期实现文本
def test_adds_two_numbers() -> None:
    assert add(2, 3) == 5
