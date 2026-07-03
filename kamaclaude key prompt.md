我们正在推进 KamaClaude / Agent Runtime 学习项目。我的目标不是单纯让 Codex 帮我写功能，而是通过源码阅读、功能扩展、测试、debug 和复盘训练工程能力，并为实习/秋招面试准备项目表达。

请你后续规划时遵守：
1. 每一步都要帮助我拆解并理解项目模块、数据流和设计原因；
2. 不要直接跳到实现，要先做源码定位、设计草案和风险分析；
3. 每个功能都要包含 unit test、integration test、ruff/mypy/pytest、git commit；
4. 每一步都要给我理解检查题；
5. 每一步都要从面试官角度拆解可能被问到的问题；
6. 我是 Python/工程新手，需要在项目推进中补工程化知识，但不要陷入纯教程学习；
7. 当前阶段我们准备做 `core.echo`，用于训练 CLI → JSON-RPC → SocketServer → CoreApp handler → Pydantic model → response 的最小闭环。