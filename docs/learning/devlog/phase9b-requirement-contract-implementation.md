# Phase 9B In-Context Requirement-Contract Checklist Implementation

## 0. Metadata

- Phase：`Phase 9B / In-Context Requirement-Contract Checklist`
- Date：`2026-07-27`
- Baseline：`b6653581ce8046d7b890208f3d3855b24a152b75`
- Branch：`codex/evaluation-harness`
- Current experiment host：`Python 3.12.13`
- Scope：默认 effective base system prompt 的严格单变量改动及其本地可审计 identity
- Out of scope：真实模型、网络 API、benchmark、task/grader修改、runtime gate、commit/push
- External report：等待 external diff review

本轮没有把 Python 3.13 写成本机实验身份。项目兼容范围与当前 host identity 是两个不同
概念；本记录只声明实际执行测试的 Python 3.12.13。

## 1. Problem and frozen hypothesis

VALID Third Baseline 在27次attempt中成功17次。Bug fixing为9/9，但feature
implementation为4/9；Phase 9A的主要证据指向显式requirement遗漏与failure-state
invariant遗漏，而不是缺少工具或Evaluation错误。

本阶段只验证以下实现假设：

> 在所有 default-prompt Agent runs 的 effective base system prompt 中追加一个简短、
> 通用、可审计的 in-context requirement contract，可能改善后续真实实验中的需求完整性。

Fake provider只能证明prompt wiring和identity，不能证明Agent能力改善。本轮没有运行真实
Phase 9B实验。

## 2. Baseline efficiency reference

### 2.1 Frozen values

| Metric | Frozen Third Baseline value |
| --- | ---: |
| Complete attempts | `23` |
| Timeout attempts | `4` |
| Complete-attempt median wall latency | `66078.2359589357 ms` |
| Complete-attempt median input + output tokens | `9859` |
| Total experiment wall time | `2071.197958742 s` |
| Sum of all 27 attempt wall latencies, diagnostic only | `2048.964364751242 s` |

### 2.2 Deterministic calculation

数据源是Third Baseline canonical `baseline.json`。算法：

```python
attempts = baseline["attempts"]
complete = [a for a in attempts if a["failure_category"] != "timeout"]
latencies = [float(a["wall_latency_ms"]) for a in complete]
tokens = [
    int(a["token_usage"]["input_tokens"])
    + int(a["token_usage"]["output_tokens"])
    for a in complete
]
median_latency = statistics.median(latencies)
median_tokens = statistics.median(tokens)
attempt_latency_sum = sum(float(a["wall_latency_ms"]) for a in attempts)
experiment_wall = (
    baseline_json.stat().st_mtime_ns
    - declared_experiment_json.stat().st_mtime_ns
) / 1_000_000_000
```

`statistics.median` 对奇数样本取排序后中间值，对偶数样本取中间两项算术平均。本次complete
数量为23。4个timeout的input/output token均为0，且全部排除在complete token median
之外。

`total experiment wall time`冻结为 `declared-experiment.json` 到canonical
`baseline.json` 的mtime差，包含attempt间调度和report收敛开销；attempt latency sum只是
诊断值，不能替代experiment wall-clock。

只读证据：

- Third Baseline artifact文件数：`1364`；
- canonical `baseline.json` SHA-256：
  `9b89907b838c31f597cec38682a978f081e8e844cf7ae28e22dec87918909d3e`；
- `declared-experiment.json` SHA-256：
  `3abf138fa5e489152670394b8ade29a8c76f6027066d5d507e77b0a951209fc5`。

本轮没有写入、移动、删除或覆盖Third Baseline artifacts。

## 3. Architecture role and prompt data flow

### 3.1 Module boundaries

| Module | Role | Must remain unchanged |
| --- | --- | --- |
| `core/loop.py` | 提供default base prompt并传入每次provider call | step/tool/stopping/event逻辑 |
| `core/context.py` | 选择base或override，再拼接global/project/session context | override完全替换base的合同 |
| `core/runner.py` | 创建ExecutionContext、registry、compactor和AgentLoop | runtime wiring与max_steps |
| `core/trace/provider.py` | 观察实际system与ordered tool schemas | provider行为 |
| `eval/worker.py` | benchmark/eval worker复用真实AgentRunner | 不增加prompt override |
| `benchmark/experiment.py` | 从实际trace计算observed identity | 不改变evaluation语义 |

### 3.2 Data flow

```text
goal / session history
  → AgentRunner.run_and_capture()
  → ExecutionContext(
        messages,
        global_context,
        project_context,
        session_notes,
        optional system_prompt_override
    )
  → AgentLoop.run()
  → context.system_prompt(default base + requirement contract)
  → TracingProvider records actual system and ordered tool schemas
  → scripted or real provider
```

`ExecutionContext.system_prompt()`保持现有顺序：

```text
default base or override
  → Global Context
  → Project Context
  → Session Notes
```

Compactor只改写`context.messages`并使用独立summarizer prompt；provider fallback只在
`system is None`时生效。两者本轮字节未改。

## 4. Exact 97-word intervention

既有default base末尾追加两个换行，随后按字节冻结：

```text
Before changing the workspace, create a concise requirement contract from every explicit acceptance criterion. For each item, record the required observable behavior, relevant failure or invalid-input behavior, any side-effect or state invariant, and the evidence you plan to use for verification. Keep this checklist visible in the conversation as you work, and update each item as implemented, verified, or unchecked. Before finishing, review every item. Do not assume unchecked items are complete: verify them when possible, otherwise clearly report the limitation. Keep the contract brief and auditable; do not expose private chain-of-thought or force any particular tool.
```

AST只读检查证明：

- contract为`97`词；
- separator精确为两个换行；
- default base literal内contract只出现一次；
- 没有新增函数、参数、state、event或branch。

每次provider API call按协议重新收到完整system；“只注入一次”指每个effective system
字符串内contract出现一次，不是整次run只传一次system。

## 5. RED evidence

Production prompt尚未修改时运行：

```text
uv run pytest \
  tests/unit/test_loop.py::test_default_prompt_contains_requirement_contract_once_per_call \
  tests/unit/test_loop.py::test_requirement_contract_preserves_runtime_inputs_and_lifecycle \
  tests/eval/test_phase8a_runner.py::test_worker_and_direct_runner_share_requirement_contract_prompt \
  tests/unit/test_spawn_agent_tool.py::test_unprofiled_subagent_inherits_requirement_contract \
  tests/unit/test_spawn_agent_tool.py::test_subagents_isolate_profile_and_context_by_workspace \
  -q
```

Observed：

```text
4 failed, 1 passed
```

四个default-path失败的第一目标断言均为：

```text
expected requirement contract occurrence: 1
actual occurrence: 0
```

Default root、worker/direct root和unprofiled subagent都准确命中contract缺失。Profiled
subagent测试通过，证明RED没有误把现有override完全替换语义当成缺陷。没有fixture、
tool schema、event、journal或provider stub错误。

## 6. GREEN evidence

只修改`core/loop.py`后重跑同一命令：

```text
5 passed in 0.51s
```

GREEN证明：

- 每个default effective system中contract恰好出现一次；
- goal和首轮messages逐字不变；
- global/project/session内容和顺序不变；
- ordered tool schemas内容和顺序不变；
- max_steps仍为测试输入值；
- tool-use → end-turn仍以两步success结束；
- event序列仍为step/tool原有lifecycle；
- worker与direct AgentRunner收到逐字相同的default system；
- unprofiled subagent继承，profiled subagent不继承。

## 7. Default/override behavior matrix

| Run path | Effective first prompt | Contract |
| --- | --- | --- |
| Benchmark root | default base | inherited once |
| Normal root | default base | inherited once |
| Normal session | default base + existing context layers | inherited once |
| Unprofiled subagent | default base | inherited once |
| Skill run | skill `system_prompt_override` | not inherited |
| Profiled subagent | profile `system_prompt_override` | not inherited |

本轮没有改变prompt merge策略。

## 8. Experiment identity

### 8.1 Actual trace-derived prompt hash

Hash来自以下真实本地、无网络链路：

```text
eval.worker.execute_request()
  → real AgentRunner
  → real AgentLoop
  → TracingProvider
  → scripted fake provider
  → trace api_call.data.system
  → UTF-8 SHA-256
```

结果：

```text
b248587ef77d172cefb5e7b777a1523cf50978d6d273b466a8b6eb37349621eb
```

没有根据源码文字手工猜hash。真实journal和trace随后由
`collect_observed_identity()`读取，`require_identity_match()`通过；把declared
prompt hash突变为全0后，observer以`prompt_hash` mismatch fail closed。

### 8.2 Old/new identity diff

| Field | Third Baseline | Phase 9B |
| --- | --- | --- |
| Git commit | `b6653581…` | 尚未提交；external review后产生新commit |
| Profile ID | `kama-coding-mvp-v1-deepseek-v4-pro` | `kama-coding-mvp-v1-deepseek-v4-pro-requirement-contract-v1` |
| Prompt hash | `73308ba3…61816` | `b248587e…621eb` |
| Suite hash | `6e1b554d…44d7` | unchanged |
| Task/grader hashes | `9 / 9 frozen` | unchanged and verified |
| Provider/model/protocol/SDK | DeepSeek / v4-pro / Anthropic messages / 0.111.0 | unchanged |
| Tool-schema hash | `8ea67642…35f9f` | unchanged |
| Runtime-config hash | `bd7d46cf…35a64` | unchanged |
| Dependency hash | `dfdbce9b…6793` | unchanged |
| Max steps / repeats / order | `20 / 3 / suite-task-repeat` | unchanged |
| MCP / artifact policy | disabled / private raw trace | unchanged |

新profile file SHA-256：

```text
9346d813af69ea1d3bc3d16a5f7e9b5cecc2f601f69e40aa1c138799ed2e1da0
```

## 9. Old profile immutability

旧profile bytes SHA-256保持：

```text
1b35206b1d0ef4449a3773cc17025c3149d797b49e9d022c707f9c9ee9fa4aa7
```

测试把新profile的`profile_id`与`expected_identity.prompt_hash`归一化回旧值后，完整JSON
树与旧profile相等。旧profile继续绑定历史prompt和历史commit；在新prompt commit上不得
把它当作可运行profile，也不得修改旧hash来适配新代码。

## 10. Holdout rubric freeze

冻结文件：

```text
docs/learning/devlog/phase9b-holdout-authoring-rubric-v1.md
```

SHA-256：

```text
4e98b45c59c1d9247be4ed4635780ee91d7e5ea752a6e174d38058b642623d14
```

Rubric要求3 categories × 2 difficulties，每格至少2题；每题具有public/private分离、
reference、alternative、至少4个wrong probes和三-copy determinism。本轮没有创建
holdout task。候选intervention冻结后只运行一次正式holdout矩阵，结果不得用于同轮调参
或补跑。

## 11. Changed files

Production：

- `src/kama_claude/core/loop.py`：仅追加两个换行和精确97词文本。

Tests：

- `tests/unit/test_loop.py`
- `tests/unit/test_spawn_agent_tool.py`
- `tests/eval/test_phase8a_runner.py`
- `tests/benchmark/test_experiment_identity.py`

Experiment：

- `benchmarks/experiments/kama-coding-mvp-v1-deepseek-v4-pro-requirement-contract-v1.json`

Learning：

- `docs/learning/devlog/phase9b-requirement-contract-implementation.md`
- `docs/learning/devlog/phase9b-holdout-authoring-rubric-v1.md`

`docs/`受现有ignore规则管理。本轮没有stage、commit或push。

## 12. Complete validation

Focused：

| Command | Result |
| --- | --- |
| `pytest tests/unit/test_loop.py -q` | `15 passed` |
| `pytest tests/unit/test_context_system_prompt.py -q` | `4 passed` |
| `pytest tests/unit/test_runner.py -q` | `23 passed` |
| `pytest tests/unit/test_spawn_agent_tool.py -q` | `21 passed` |
| `pytest tests/eval/test_phase8a_runner.py -q` | `10 passed` |
| `pytest tests/benchmark/test_experiment_identity.py -q` | `29 passed` |
| `pytest tests/benchmark/test_report.py -q` | `10 passed` |

Suite：

| Command | Result |
| --- | --- |
| `pytest tests/eval -q` | `74 passed` |
| `pytest tests/benchmark -q` | `84 passed` |
| `pytest tests/ -q` | `977 passed` |
| `ruff check src tests scripts` | all checks passed |
| `mypy src` | no issues in 115 source files |
| `python scripts/gen_protocol_doc.py --check` | WIRE_PROTOCOL up to date |
| `git diff --check` | empty |

以下scope diff全部为空：

- `core/context.py`
- `core/runner.py`
- `core/llm/provider.py`
- `core/subagent/**`
- `core/compact/**`
- `core/tools/**`
- `eval/**`
- `benchmark/**`
- `benchmarks/tasks/**`
- `benchmarks/suites/**`
- 旧DeepSeek profile

## 13. Security, lifecycle and resource boundaries

- Scripted provider不读取credential、不初始化真实Anthropic client、不访问网络；
- 测试trace只位于pytest临时目录，不写入本devlog；
- 本devlog不含credential、private task context、hidden grader或raw trace；
- raw trace policy仍为private；
- process isolation仍不是security sandbox；
- AgentLoop的tool、permission、cancellation、terminal和max-step分支没有变化；
- prompt增加会产生固定context开销，但没有新增tool call、文件或event；
- 真实token/latency影响只能由后续已授权paired experiment测量。

## 14. Manual and negative probe evidence

| Probe | Expected killer | Result |
| --- | --- | --- |
| 删除default contract | root/worker/subagent prompt tests | RED中killed |
| default中重复contract | occurrence严格等于1 | killed by assertion |
| 给profile override合并default | profiled subagent absence assertion | killed by assertion |
| 改ordered tools | worker/direct与loop完整list比较 | killed by assertion |
| 修改declared prompt hash | `require_identity_match` | fail closed |
| 修改旧profile任一其他字段 | normalized full-tree equality | killed by assertion |

这些是局部合同probe，不宣称全仓库mutation coverage。

## 15. Remaining risks and rollback

Risks：

- Prompt instruction不等于runtime enforcement，模型可能忽略或机械执行checklist；
- 97词会在每次provider call增加固定input context；
- 更多显式规划可能提高latency或timeout；
- override runs故意不继承contract，不能把root结果外推到skill/profiled subagent；
- development suite已参与问题分析，存在过拟合风险；
- fake-provider测试只证明wiring，不能证明success rate提升。

Rollback范围严格为：

1. 删除`core/loop.py`中的两个换行和97词addition；
2. 删除Phase 9B新profile；
3. 保留失败实验artifact和结果，不修改task/grader来挽救分数。

Rollback不得改变tool schema、timeout、max_steps、Evaluation或Benchmark semantics。

## 16. Resume claim impact

| Claim | Status | Safe wording |
| --- | --- | --- |
| Requirement-contract prompt wiring | Implemented + Tested | Implemented and regression-tested an in-context requirement-contract instruction |
| Declared/observed prompt identity | Tested | Verified actual trace-derived prompt identity with fail-closed mismatch detection |
| Agent capability improvement | Not benchmarked | Benchmark impact has not yet been measured |
| General coding ability | Unsupported | Do not claim before independent holdout evidence |

安全表述：

> Implemented and locally verified a single-variable requirement-contract prompt
> intervention; real-model benchmark impact is not yet measured.

不得声称success从17/27提高、不得声称generalization，也不得把977个测试描述成coding
benchmark结果。

## 17. 90-second interview explanation

第三次固定benchmark是17/27，bug fixing为9/9，但feature只有4/9。Trace与grader分析
显示主要失败来自显式需求和failure-state invariant遗漏，而不是缺少工具。因此我做了
严格单变量干预：只在AgentLoop的默认base system prompt后追加97词的通用requirement
contract，要求模型在改workspace前提取observable behavior、failure behavior、state
invariant和验证证据，并在结束前标记verified或unchecked。ExecutionContext的override
与context拼接、tools、max steps、stopping和events都没有改。测试先得到4 fail/1 pass的
RED：root、benchmark worker和unprofiled subagent都缺contract，而profile override保持
原语义；最小prompt改动后同组5项全绿。随后通过real AgentRunner、AgentLoop、
TracingProvider和scripted provider生成actual trace，得到prompt hash并与新profile
declaration匹配，tool/runtime/suite identity保持不变。完整回归977项通过。当前只能说
wiring已实现并测试，是否改善17/27必须由另行授权的真实paired experiment和冻结holdout
证明。

## 18. Comprehension questions and standard answers

### Q1. 为什么只修改AgentLoop中的literal？

标准答案：它是default-prompt run的单一事实源。改ExecutionContext会同时改变override
merge合同，改worker会产生benchmark-only行为，都会引入第二变量。

### Q2. 为什么每次API call都包含contract不算重复注入？

标准答案：provider协议每轮都接收完整system。一次注入定义为单个effective system字符串
中只出现一次，不是整个run只发送一次。

### Q3. 为什么profiled subagent不继承？

标准答案：现有`system_prompt_override`完全替换base。改变它会扩大实验变量；本轮用测试
冻结旧语义。

### Q4. 如何证明hash不是根据源码猜的？

标准答案：测试让real AgentRunner经过AgentLoop和TracingProvider，读取实际
`api_call.data.system`的UTF-8 bytes后计算SHA-256。

### Q5. 为什么fake provider不能证明能力提升？

标准答案：它返回预编排的end_turn，只能验证wiring、消息和identity；它没有完成开放式
coding任务。

### Q6. 为什么timeout token不进入效率median？

标准答案：timeout partial使用canonical zero-token占位，并不是完整usage观测；混入会
系统性压低token median。

### Q7. 为什么保留旧profile而不更新prompt hash？

标准答案：旧profile是历史实验identity的一部分。原地更新会破坏Third Baseline可审计性
并让旧commit与新prompt混淆。

### Q8. 为什么还需要holdout？

标准答案：本干预来自已见suite的失败分析，development suite结果可能过拟合。只有事先
冻结且不用于同轮调参的独立holdout才能提供更强的一般化证据。

## 19. Interview follow-up questions and answer frameworks

### Follow-up 1. 如何证明这是单变量实验？

回答框架：列出允许变化的prompt/profile identity；逐项给出suite、tools、runtime、
provider与grader的hash/diff证据；最后说明Git commit变化是代码身份而非第二行为参数。

### Follow-up 2. Prompt instruction与structured state有什么区别？

回答框架：说明owner、持久化、可验证性和failure semantics；本轮checklist由模型维护，
runtime没有结构化字段或final gate，因此不能保证遵守。

### Follow-up 3. 如何防止benchmark leakage？

回答框架：public/private数据流分离；prompt保持任务无关；holdout不复用领域/API/边界/
failure mechanism；冻结后只运行一次。

### Follow-up 4. 如果Phase 9B成功率提高但latency增加怎么办？

回答框架：使用预注册primary与efficiency rollback；complete median latency或token增加
超过20%且overall改善少于3次，则判为不值得采用。

### Follow-up 5. 下一步如何运行真实实验？

回答框架：先external diff review和clean commit；fresh preflight验证declared/observed
identity与新output root；再单独获得数据外发、27 attempts、费用和private artifacts授权。

## 20. Final evidence checklist

- [x] Baseline效率口径与数值已在真实实验前冻结。
- [x] Host identity记录为Python 3.12.13。
- [x] 唯一production diff是default base prompt文本。
- [x] RED与GREEN均为fresh evidence。
- [x] Default/override矩阵有测试覆盖。
- [x] Prompt hash来自真实本地trace链。
- [x] Tool/runtime/suite/task/grader/dependency identity保持不变。
- [x] 旧profile bytes未改。
- [x] Holdout rubric及hash已冻结，未创建task。
- [x] Focused、Eval、Benchmark、977-test full gates全部通过。
- [x] Ruff、mypy、protocol和scope diff通过。
- [x] 无secret、raw private trace、hidden grader或主机私有绝对路径。
- [x] 未stage、commit或push。
- [x] 未调用真实模型/API，未运行真实benchmark（只运行了benchmark测试套件）。

```text
Implementation: COMPLETE
Ready for External Diff Review: YES
Real model experiment: NOT AUTHORIZED
```
