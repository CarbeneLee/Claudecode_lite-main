# 模拟错误实现：下划线未被规范化
def normalize_username(value: str) -> str:
    cleaned = value.strip().lower()
    if not cleaned:
        raise ValueError("username cannot be empty")
    if " " in cleaned:
        raise ValueError("username cannot contain spaces")
    return cleaned
