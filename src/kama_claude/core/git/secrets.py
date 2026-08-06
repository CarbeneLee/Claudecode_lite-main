from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class SecretFinding:
    # finalize 扫描产物：命中位置（file:line）+ 规则名；刻意不含 secret 值，
    # 保证 detail 可安全展示给 LLM 与用户（设计 §10.4）
    file: str
    line: int
    rule: str


class SecretScanner(ABC):
    # 扫描 staged diff 文本，返回命中位置；子类实现具体检测规则
    @abstractmethod
    def scan_diff(self, diff: str) -> list[SecretFinding]: ...


_AWS_ACCESS_KEY_RE = re.compile(r"\bAKIA[0-9A-Z]{16}\b")
_PRIVATE_KEY_RE = re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")
_CREDENTIAL_RE = re.compile(
    r"\b(?:password|passwd|token|api_key|secret)\s*[=:]\s*['\"]?[^\s'\"]{8,}"
)


class RegexScanner(SecretScanner):
    # 基于正则的内置扫描器：只扫新增行（+ 前缀），按 hunk 头计算命中行号
    _RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
        ("aws_access_key", _AWS_ACCESS_KEY_RE),
        ("private_key", _PRIVATE_KEY_RE),
        ("password_token", _CREDENTIAL_RE),
    )

    def scan_diff(self, diff: str) -> list[SecretFinding]:
        findings: list[SecretFinding] = []
        file = ""
        new_line = 0
        consumed = 0
        for raw in diff.splitlines():
            if raw.startswith("+++ "):
                file = raw[4:].removeprefix("b/")
                consumed = 0
                continue
            hunk = re.match(r"@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@", raw)
            if hunk:
                new_line = int(hunk.group(1))
                consumed = 0
                continue
            if raw.startswith("+") and not raw.startswith("+++"):
                consumed += 1
                for rule, pattern in self._RULES:
                    if pattern.search(raw[1:]):
                        findings.append(
                            SecretFinding(file=file, line=new_line + consumed - 1, rule=rule)
                        )
            elif raw.startswith(" "):
                consumed += 1
        return findings
