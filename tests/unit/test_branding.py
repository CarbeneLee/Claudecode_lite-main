from __future__ import annotations

import sys
import tomllib
from pathlib import Path
from unittest.mock import Mock

import pytest

import kama_claude.cli.main as cli_main_module
import kama_claude.tui.__main__ as tui_main_module
from kama_claude.tui.app import KamaTuiApp

_REPO_ROOT = Path(__file__).resolve().parents[2]


# 功能：验证 Li Code 新入口与 Kama legacy 入口逐组复用完全相同的实现函数
# 设计：直接解析 PEP 621 scripts 映射，锁定打包边界且避免通过重复 wrapper 获得表面兼容
def test_console_script_aliases_share_existing_entry_functions() -> None:
    pyproject = tomllib.loads((_REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    scripts = pyproject["project"]["scripts"]

    assert scripts["li"] == scripts["kama"] == "kama_claude.cli.main:main"
    assert scripts["li-core"] == scripts["kama-core"] == "kama_claude.core.app:run"
    assert scripts["li-tui"] == scripts["kama-tui"] == "kama_claude.tui.__main__:main"


@pytest.mark.parametrize("program", ["li", "kama"])
# 功能：验证新旧主 CLI help 都使用实际入口名并显示 Li Code 产品描述
# 设计：替换 argv 后调用真实 argparse 入口，同时覆盖 native prog-name 与 legacy alias 兼容性
def test_cli_help_uses_invoked_alias_and_li_code_brand(
    program: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(sys, "argv", [program, "--help"])

    with pytest.raises(SystemExit) as exc_info:
        cli_main_module.main()

    assert exc_info.value.code == 0
    output = capsys.readouterr().out
    assert output.startswith(f"usage: {program} ")
    assert "Li Code CLI" in output


@pytest.mark.parametrize("program", ["li-tui", "kama-tui"])
# 功能：验证新旧 TUI help 都使用实际入口名并显示 Li Code 产品描述
# 设计：让 --help 在配置和 UI 启动前退出，聚焦验证 parser 品牌与 argv[0] 兼容行为
def test_tui_help_uses_invoked_alias_and_li_code_brand(
    program: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(sys, "argv", [program, "--help"])

    with pytest.raises(SystemExit) as exc_info:
        tui_main_module.main()

    assert exc_info.value.code == 0
    output = capsys.readouterr().out
    assert output.startswith(f"usage: {program} ")
    assert "Li Code TUI" in output


# 功能：验证 TUI title、banner、初始 header 与连接态 header 统一展示 Li Code
# 设计：直接检查 compose 结果并替换 header 查询边界，不建立 socket 或修改 session 生命周期
def test_tui_visible_branding_is_li_code(monkeypatch: pytest.MonkeyPatch) -> None:
    app = KamaTuiApp("127.0.0.1", 7437)
    initial_header = list(app.compose())[0]

    assert app.TITLE == "Li Code"
    assert "Li Code" in str(initial_header.content)
    assert "Li Code" in app._BANNER

    connected_header = Mock()
    monkeypatch.setattr(app, "query_one", Mock(return_value=connected_header))
    app._session_id = "sess-brand"
    app._update_header("ready")

    connected_header.update.assert_called_once()
    rendered_header = connected_header.update.call_args.args[0]
    assert "Li Code" in rendered_header
    assert "sess-brand" in rendered_header
    assert "ready" in rendered_header
