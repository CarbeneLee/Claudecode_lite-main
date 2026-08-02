from __future__ import annotations

import pytest

from kama_claude.core.git.secrets import RegexScanner, SecretFinding, SecretScanner


# 功能：验证三条内置正则规则各自命中对应的 secret 形态
# 设计：参数化 (期望 rule, 命中行)；line 以 + 前缀模拟新增行
@pytest.mark.parametrize(
    ("rule", "line"),
    [
        ("aws_access_key", "+AKIAIOSFODNN7EXAMPLE"),
        ("private_key", "+-----BEGIN RSA PRIVATE KEY-----"),
        ("private_key", "+-----BEGIN OPENSSH PRIVATE KEY-----"),
        ("private_key", "+-----BEGIN EC PRIVATE KEY-----"),
        ("password_token", "+password = 'hunter2secret'"),
        ("password_token", "+token=abcdefgh"),
        ("password_token", "+api_key: supersecret123"),
        ("password_token", "+secret: 0xDEADBEEF12345"),
    ],
)
def test_regex_rules_hit(rule: str, line: str) -> None:
    scanner = RegexScanner()

    findings = scanner.scan_diff(line)

    assert findings
    assert findings[0].rule == rule


# 功能：验证相似但不构成 secret 的内容不误报
# 设计：参数化各规则的反例（长度不足 / 错误密钥类型 / 非密钥字段名）
@pytest.mark.parametrize(
    "line",
    [
        "+AKIA12345678",  # AWS key 长度不足
        "+BEGIN RSA PRIVATE KEY",  # 缺少完整 ----- 围栏
        "+-----BEGIN PGP MESSAGE-----",  # 非密钥类型
        "+token = ab",  # 值太短
        "+username = hunter2secret",  # 非敏感字段名
    ],
)
def test_regex_rules_miss(line: str) -> None:
    scanner = RegexScanner()

    assert scanner.scan_diff(line) == []


# 功能：验证扫描只检查新增行，并按 hunk 头正确计算行号
# 设计：构造双 hunk diff（a.py 改 1 行 + b.py 新增 2 行），断言 file/line/rule
def test_scan_reports_file_and_line_numbers() -> None:
    diff = (
        "diff --git a/a.py b/a.py\n"
        "index 1111111..2222222 100644\n"
        "--- a/a.py\n"
        "+++ b/a.py\n"
        "@@ -1,2 +1,3 @@\n"
        " x=1\n"
        "-y=2\n"
        "+AKIAIOSFODNN7EXAMPLE\n"
        "+z=3\n"
        "diff --git a/b.py b/b.py\n"
        "new file mode 100644\n"
        "--- /dev/null\n"
        "+++ b/b.py\n"
        "@@ -0,0 +1,2 @@\n"
        "+password = 'hunter2secret'\n"
        "+ok\n"
    )

    findings = RegexScanner().scan_diff(diff)

    assert findings == [
        SecretFinding(file="a.py", line=2, rule="aws_access_key"),
        SecretFinding(file="b.py", line=1, rule="password_token"),
    ]


# 功能：验证 SecretFinding 不携带 secret 值本身（detail 可安全展示给 LLM）
# 设计：扫描含两个 secret 的 diff，断言 finding 全部字段均不含值
def test_finding_does_not_leak_secret_value() -> None:
    diff = (
        "+++ b/conf.py\n"
        "@@ -0,0 +1 @@\n"
        "+AKIAIOSFODNN7EXAMPLE\n"
        "+++ b/secret.txt\n"
        "@@ -0,0 +1 @@\n"
        "+password = 'hunter2secret'\n"
    )

    findings = RegexScanner().scan_diff(diff)

    assert len(findings) == 2
    for finding in findings:
        for value in vars(finding).values():
            assert "AKIAIOSFODNN7EXAMPLE" not in str(value)
            assert "hunter2secret" not in str(value)


# 功能：验证上下文行与删除行不参与扫描（只有 + 行代表进入文件的内容）
# 设计：同内容出现在 context（前导空格）与删除行（- 前缀）均不产生 finding
def test_context_and_removed_lines_are_not_scanned() -> None:
    diff = (
        " AKIAIOSFODNN7EXAMPLE\n"
        "-AKIAIOSFODNN7EXAMPLE\n"
        "-password = 'hunter2secret'\n"
    )

    assert RegexScanner().scan_diff(diff) == []


# 功能：验证 SecretScanner 是抽象基类（不可直接实例化）
# 设计：直接构造 ABC 期待 TypeError，防止绕过子类实现
def test_scanner_is_abstract() -> None:
    with pytest.raises(TypeError):
        SecretScanner()  # type: ignore[abstract]
