# 模拟错误实现：内部空格未被拒绝
def normalize_username(value: str) -> str:
    cleaned = value.strip().lower()
    if not cleaned:
        raise ValueError("username cannot be empty")
    return cleaned.replace("_", "-")
