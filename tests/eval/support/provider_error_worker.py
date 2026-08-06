from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from kama_claude.core.bus.events import LlmModelSelectedEvent
from kama_claude.core.config import KamaConfig
from kama_claude.core.events.bus import EventBus
from kama_claude.core.llm.types import LlmResponse
from kama_claude.core.runner import AgentRunner
from kama_claude.eval.models import WorkerRequest, WorkerResult
from kama_claude.eval.worker import execute_request


class _ModelThenErrorProvider:
    # 保存experiment profile配置的模型身份
    def __init__(self, model: str) -> None:
        self._model = model

    # 发布合法model identity后抛出稳定本地异常且不访问网络
    async def chat(
        self,
        messages: list[dict[str, object]],
        tool_schemas: list[dict[str, object]],
        bus: EventBus,
        run_id: str,
        *,
        step: int = 0,
        system: str | None = None,
    ) -> LlmResponse:
        await bus.publish(
            LlmModelSelectedEvent(
                run_id=run_id,
                model=self._model,
                strategy="static",
                ts="2026-07-29T00:00:00+00:00",
            )
        )
        raise RuntimeError("local provider failure")


# 解析production worker兼容的request/result参数
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    parser.add_argument("--result", required=True)
    return parser.parse_args()


# 使用fake provider构造真实AgentRunner并保留production observer wiring
def _runner_factory(config: KamaConfig, **kwargs: object) -> AgentRunner:
    return AgentRunner(
        config,
        provider=_ModelThenErrorProvider(config.llm.default_model),
        **kwargs,  # type: ignore[arg-type]
    )


# 原子写入test worker结果供production parent读取
def _write_result(path: Path, result: WorkerResult) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(result.model_dump_json(), encoding="utf-8")
    temporary.replace(path)


# 执行真实worker observer链路并只注入无网络provider seam
def main() -> None:
    args = _parse_args()
    request = WorkerRequest.model_validate_json(
        Path(args.request).read_text(encoding="utf-8")
    )
    result = asyncio.run(execute_request(request, runner_factory=_runner_factory))
    _write_result(Path(args.result), result)


if __name__ == "__main__":
    main()
