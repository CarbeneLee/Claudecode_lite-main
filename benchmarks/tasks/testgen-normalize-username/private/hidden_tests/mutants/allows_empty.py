# 模拟错误实现：空白输入未被拒绝
def normalize_username(value: str) -> str:
    cleaned = value.strip().lower()
    if " " in cleaned:
        raise ValueError("username cannot contain spaces")
    return cleaned.replace("_", "-")
