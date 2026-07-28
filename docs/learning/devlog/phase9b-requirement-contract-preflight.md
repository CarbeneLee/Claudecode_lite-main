# Phase 9B Requirement-Contract Experiment Final Preflight

## 0. Metadata

- Phase：Phase 9B final preflight
- Date：2026-07-28
- Approved HEAD：`7aa4b7512fadbc87b389a8d0106a23274265fd42`
- Branch：`codex/evaluation-harness`
- Scope：在无网络条件下验证提交链、单变量 prompt、实验 identity、测试、历史 artifact 和输出策略
- Out of scope：调用模型、运行真实 benchmark、修改实现或冻结输入、创建正式输出目录、stage/commit/push
- Decision：`Technical preflight: NO-GO`

## 1. Problem

Phase 9B 计划对所有默认 root Agent run 的 effective base system prompt 做一次严格单变量干预。真实实验启动前，不仅要证明实现、identity 和测试门禁稳定，还必须证明结果判定标准已经在真实结果产生前成为不可变证据。

本轮其余技术门禁通过，但四项 Primary thresholds、完整 hard guardrails 和 27-attempt 完整性要求没有出现在当前 HEAD 的已提交文档中。它们只存在于本地未跟踪且被忽略的 planning 文档，不能满足“真实结果前已提交”的预注册要求。因此本轮必须 fail closed。

## 2. Why it matters to an Agent

这次实验试图回答：短小、通用的 in-context requirement-contract 是否能改善 Agent 对显式验收条件、失败行为和状态不变式的覆盖。若 identity 不完整或阈值未预注册，即使最终分数上升，也无法排除事后选择指标或改变判定口径，实验不能作为可信 benchmark 证据。

## 3. Current Git chain

| Check | Observed | Result |
| --- | --- | --- |
| HEAD | `7aa4b7512fadbc87b389a8d0106a23274265fd42` | PASS |
| Remote tracking commit | 与 HEAD 相同 | PASS |
| Direct parent | `088fe0314d060b1e519b1c4861e595b3d20e7d3e` | PASS |
| Third Baseline parent | `b6653581ce8046d7b890208f3d3855b24a152b75` | PASS |
| Branch | `codex/evaluation-harness` | PASS |
| Tracked/staged/untracked | 空 | PASS |
| Dirty | false | PASS |

Implementation commit相对Third Baseline只包含批准的六个文件：

- `src/kama_claude/core/loop.py`
- `tests/unit/test_loop.py`
- `tests/unit/test_spawn_agent_tool.py`
- `tests/eval/test_phase8a_runner.py`
- `tests/benchmark/test_experiment_identity.py`
- `benchmarks/experiments/kama-coding-mvp-v1-deepseek-v4-pro-requirement-contract-v1.json`

Docs commit只包含：

- `docs/learning/devlog/phase9b-requirement-contract-implementation.md`
- `docs/learning/devlog/phase9b-holdout-authoring-rubric-v1.md`

Docs commit相对implementation commit的 `src tests benchmarks` diff为空。

## 4. Committed documentation evidence

从 Git committed blobs重新计算：

| Blob | SHA-256 | Result |
| --- | --- | --- |
| Requirement-contract implementation devlog | `5746701ac31b242a21474771722440c7d23fd0d88e415ade5e80218d99f78a4d` | PASS |
| Holdout authoring rubric v1 | `4e98b45c59c1d9247be4ed4635780ee91d7e5ea752a6e174d38058b642623d14` | PASS |

Rubric在真实Phase 9B结果前冻结，并要求：

- 至少12个holdout tasks；
- 3 categories × 2 difficulties；
- public/private分离；
- reference与alternative valid solution；
- 每题至少4个wrong probes；
- 三份fresh-copy determinism；
- 候选干预冻结后只运行一次正式holdout；
- 不允许同轮调参或补跑。

## 5. Production single-variable audit

相对Third Baseline，唯一production行为差异位于默认base system prompt字面量：

- 末尾以精确 `\n\n` 分隔；
- 追加97个英文单词的通用requirement-contract；
- default base中只出现一次；
- 未增加函数、constant owner、branch、state、event或tool call；
- 未改写messages；
- 未改变stopping、max steps、permission或cancellation；
- 归一化该prompt literal后，AST与Third Baseline相同；
- 未包含当前9个任务、业务对象、grader、mutant或具体failure硬编码。

本文件只记录prompt hash，不保存prompt原文。

## 6. Prompt composition matrix

| Run path | Observed contract composition | Result |
| --- | --- | --- |
| Benchmark root | default base包含一次 | PASS |
| Normal root/session | default base包含一次 | PASS |
| Unprofiled subagent |继承default base一次 | PASS |
| Skill override | override完全替换default base，不继承 | PASS |
| Profiled subagent | profile override完全替换default base，不继承 | PASS |

同时确认：

- goal与messages不变；
- global/project/session context内容和顺序不变；
- ordered tool schemas内容和顺序不变；
- compactor summarizer prompt不变；
- provider fallback prompt不变；
- 每次API call发送完整system不等于同一system字符串内重复注入。

## 7. Profile normalized comparison

| Item | Value | Result |
| --- | --- | --- |
| New profile ID | `kama-coding-mvp-v1-deepseek-v4-pro-requirement-contract-v1` | PASS |
| New profile SHA-256 | `9346d813af69ea1d3bc3d16a5f7e9b5cecc2f601f69e40aa1c138799ed2e1da0` | PASS |
| Old profile SHA-256 | `1b35206b1d0ef4449a3773cc17025c3149d797b49e9d022c707f9c9ee9fa4aa7` | PASS |
| Normalized JSON tree | 只回退profile ID和expected prompt hash后与旧profile相同 | PASS |

旧profile只保留为Third Baseline历史identity，本轮没有运行。

## 8. Declared and observed identity

使用本地scripted provider走过真实链路：

```text
new profile
  → eval worker
  → real AgentRunner
  → real AgentLoop
  → TracingProvider
  → scripted provider
  → actual trace
  → observed identity collector
```

该probe没有读取credential，也没有访问网络。

| Identity | Value | Result |
| --- | --- | --- |
| Effective prompt SHA-256 | `b248587ef77d172cefb5e7b777a1523cf50978d6d273b466a8b6eb37349621eb` | PASS |
| Prompt hash cardinality within attempt | 1 | PASS |
| Declared vs observed | match | PASS |
| Mutated observed prompt hash | fail closed | PASS |
| Public report raw prompt | absent | PASS |

## 9. Frozen experiment identity

| Field | Frozen value | Result |
| --- | --- | --- |
| Suite | `kama-coding-mvp@1` | PASS |
| Suite hash | `6e1b554df12962fc454a5b61c6e6145bd022041ec77b24149d73bd616e6044d7` | PASS |
| Task hashes | 9/9 match | PASS |
| Grader hashes | 9/9 match | PASS |
| Provider | `deepseek` | PASS |
| Protocol | `anthropic_messages` | PASS |
| Endpoint ID | `deepseek-anthropic-compatible` | PASS |
| Model | `deepseek-v4-pro` | PASS |
| SDK | `anthropic==0.111.0` | PASS |
| Max steps | 20 | PASS |
| Repeats | 3 | PASS |
| Execution order | suite task, then repeat, ascending | PASS |
| Task timeout | easy 120s；medium/challenging 180s | PASS |
| MCP | disabled | PASS |
| Raw trace visibility | private | PASS |
| Ordered tool-schema hash | `8ea67642a27db476c048575e9a532c41a044e39f63f729260428cdce0fd35f9f` | PASS |
| Runtime-config hash | `bd7d46cf2e3139369704dcc67ade71c58a9d9e2247d7912abae1c378a7b35a64` | PASS |
| Dependency hash | `dfdbce9b7ec2a3164390e373d13a2f547280db34e96d58f66d20255f5f436793` | PASS |

相对Third Baseline，预期只变化Git commit、profile ID/hash、prompt hash和新output root。其余behavior identity保持不变。

## 10. Preregistered metrics audit

计划中的Primary thresholds为：

- overall success ≥ 20/27；
- feature success ≥ 6/9；
- atomic + inventory ≥ 3/6；
- inventory ≥ 1/3。

计划中的Hard guardrails为：

- bug fixing保持9/9；
- timeout不高于4/27；
- runtime、infra、trace、grader failures为0；
- experiment为VALID；
- 27 planned/started/completed矩阵完整。

审计结果：**FAIL**。

对当前HEAD的两个已提交Phase 9B文档执行精确内容搜索，仅找到Third Baseline的17/27和9/9背景描述；没有找到四项Primary thresholds、完整hard guardrails或27-attempt完整性要求。相关标准只存在于本地忽略的planning文档，未形成Git committed blob。因此它们不能作为不可变的预注册实验判定证据。

这是本轮唯一确定性NO-GO blocker。修复需要新的、经人工review的文档提交和随后全新preflight；本轮禁止修改或提交，因此没有自动修复。

## 11. Efficiency baseline and aggregation contract

Third Baseline冻结口径：

| Metric | Value |
| --- | ---: |
| Complete attempts | 23 |
| Timeout attempts | 4 |
| Complete median latency | 66078.2359589357 ms |
| Complete median input+output tokens | 9859 |
| Total experiment wall | 2071.197958742 s |
| Attempt latency sum, diagnostic only | 2048.964364751242 s |

若未来获得授权，Phase 9B必须继续使用complete-attempt median、Python `statistics.median`和相同experiment wall起止定义。Timeout的canonical zero-token值不得进入complete token median。

## 12. Fresh test and static-gate evidence

所有测试均为本轮fresh执行。首次启动full suite时，沙箱不能读取现有uv cache，命令在测试收集前以exit 2立即退出；授权读取现有cache后重新执行并通过。这是执行环境门禁，不是测试失败。

| Command scope | Exact result | Test/runtime exit |
| --- | --- | ---: |
| Loop | 15 passed in 0.52s | 0 |
| Context system prompt | 4 passed in 0.02s | 0 |
| Runner | 23 passed in 0.63s | 0 |
| Spawn agent | 21 passed in 0.47s | 0 |
| Eval runner | 10 passed in 2.76s | 0 |
| Experiment identity | 29 passed in 0.66s | 0 |
| Report | 10 passed in 0.24s | 0 |
| Evaluation suite | 74 passed in 6.05s | 0 |
| Benchmark suite | 84 passed in 212.04s | 0 |
| Full suite | 977 passed in 227.07s | 0 |
| Ruff | All checks passed | 0 |
| Mypy | No issues in 115 source files | 0 |
| Protocol doc check | Up to date | 0 |
| Git diff check | Empty | 0 |

测试数量与预期完全一致，无skip或failure。

## 13. Scope gates

相对Third Baseline，以下scope diff均为空：

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

当前docs commit相对implementation commit的 `src tests benchmarks` diff为空。

## 14. Historical artifact integrity

| Experiment | Files | Canonical evidence | Status |
| --- | ---: | --- | --- |
| First invalid | 280 | Tree hash `79e1ff156c72687d0ab0e49472f2a200c9d058d015ae1e7b8fc5ecbf352b84ea` | INVALID；无baseline report |
| Second invalid | 848 | Tree hash `95443fe3a3dccd1dcd337e7ebea0bb288311b7a1a6c1f7b143061c9e2063255b` | INVALID；无baseline report |
| Third Baseline | 1364 | Tree hash `09a1010f2d1f1f9de8c92e6bdb7ee410985b2e5b8d20910cfec652c6eb7adf8c` | VALID |

Third Baseline进一步确认：

- `baseline.json` SHA-256：`9b89907b838c31f597cec38682a978f081e8e844cf7ae28e22dec87918909d3e`
- `declared-experiment.json` SHA-256：`3abf138fa5e489152670394b8ade29a8c76f6027066d5d507e77b0a951209fc5`
- 结果仍为17/27
- 本轮没有重算、补跑、移动或覆盖任何历史artifact

## 15. Output policy

预留逻辑名称：

`kama-coding-mvp-v1-deepseek-v4-pro-requirement-contract-v1-20260728-001`

已验证：

- 对应canonical path位于repository外；
- 当前不存在；
- 不同于前三次experiment roots；
- 不属于任何Git worktree，不会默认上传；
- parent可写且可用空间约687.6 GB；
- 本轮没有创建该目录。

任何未来经授权的真实运行必须在启动前再次检查目录不存在，以关闭preflight与执行之间的TOCTOU窗口。

## 16. Credential redaction and security boundaries

| Boundary | Result | Limitation |
| --- | --- | --- |
| Credential presence | production resolver确认存在 | 未输出值、未计算公开hash、未持久化 |
| Fake-provider probes | 未读取credential、未访问网络 | 只验证wiring与identity，不证明真实provider可用 |
| Public report | 只包含prompt/tool identity hashes | private raw trace仍可能含敏感runtime内容 |
| Raw trace | private policy保持 | private不等于安全隔离 |
| Worker | lifecycle isolation | 不是OS security sandbox |
| Output root | repository外且不自动上传 | 运行前仍需重新做TOCTOU检查 |

## 17. Remaining external risks

- In-context instruction是模型指导，不是runtime强制执行。
- 当前suite是9个synthetic Python fixtures，不能代表通用coding能力。
- 相同suite上的干预存在过拟合风险，必须依赖已冻结holdout policy约束解释。
- Provider latency与模型随机性仍会影响timeout和paired结果。
- Process isolation不是恶意代码sandbox。
- Credential存在只证明配置可解析，不证明未来网络、计费或provider availability。
- 最关键的当前blocker是预注册阈值没有进入已提交HEAD。

## 18. Decision and stop condition

```text
Technical preflight: NO-GO
Real model experiment: PAUSED
Reason: preregistered primary and hard-guardrail thresholds are not present in committed HEAD
Real model experiment: NOT AUTHORIZED
```

解除blocker的最小条件：

1. 将四项Primary thresholds、全部hard guardrails、27-attempt完整性和效率口径写入一个受review的tracked文档；
2. 形成新的clean commit；
3. 重新执行完整final preflight；
4. 只有得到新的显式人类授权后才能运行真实模型。

## 19. Resume claim impact

| Candidate claim | Status | Evidence | Gap | Safe wording |
| --- | --- | --- | --- | --- |
| 实现通用requirement-contract prompt干预 | Implemented and tested | 单变量diff、identity probe、977 tests | 尚无真实Phase 9B结果 | “实现并离线验证了可审计的单变量prompt实验配置” |
| 该干预改善coding success | Not benchmarked | 无真实Phase 9B API run | 预注册证据blocker且无结果 | 不得声称改善 |
| 实验可复现 | Partially ready | frozen identity、profile、tests | 阈值未成为committed evidence | “技术identity已冻结，实验门禁尚未全部通过” |

## 20. 90-second interview explanation

我为coding Agent设计了一次严格单变量实验：只在默认base system prompt中加入一个97词的通用requirement-contract，保持suite、provider、tools、timeouts和grader不变。Preflight从Git提交链、AST归一化、profile tree、真实本地AgentRunner链路和trace identity验证这个边界，并用scripted provider保证不访问网络。所有focused、evaluation、benchmark和977项全量测试都通过，历史artifact也保持只读。但preflight仍然判为NO-GO，因为四项主指标阈值与hard guardrails没有进入已提交HEAD，只存在于本地planning文档。这个决定体现了实验可信度的核心：实现正确不等于实验可以运行；如果成功标准没有在结果前成为不可变证据，就必须fail closed，先补齐预注册提交，再重新preflight。

## 21. Comprehension questions and standard answers

### Q1. 为什么测试全绿仍是NO-GO？

标准答案：测试证明实现和身份链路满足技术契约，但不能替代预注册。成功阈值未进入committed evidence时，结果解释仍可能被事后调整。

### Q2. 为什么本地忽略的planning文档不够？

标准答案：它不是不可变、可由commit定位和复核的实验前证据，无法证明内容在看到真实结果前没有变化。

### Q3. 单变量边界如何验证？

标准答案：比较commit文件范围、归一化prompt literal后的AST、profile归一化tree，并从真实本地runtime链路收集observed identity。

### Q4. 为什么需要declared与observed identity？

标准答案：profile声明的是意图，trace观察的是实际执行；二者必须匹配，行为影响字段的任何不一致都使实验无效。

### Q5. 为什么timeout zero-token不进入token median？

标准答案：timeout attempt的canonical zero是缺失完整usage证据的占位，不代表真实零消耗；混入会系统性压低complete-attempt成本。

### Q6. 为什么output root在preflight中不能创建？

标准答案：正式runner要求全新目录以避免覆盖和污染；preflight只验证候选路径，执行前再次检查不存在以降低TOCTOU风险。

### Q7. private trace是否等于安全？

标准答案：不是。private是artifact visibility policy，worker也只提供lifecycle isolation；二者都不是针对恶意代码的OS sandbox。

### Q8. 如何安全描述当前成果？

标准答案：可以说单变量实验实现与离线身份验证完成，但不能说干预提高了能力，也不能把synthetic development suite解释为通用coding能力。

## 22. Interview follow-up questions and answer frameworks

### Follow-up 1. 如何避免prompt实验过拟合？

回答框架：冻结development suite解释边界；在看到结果前冻结holdout rubric；holdout只运行一次；禁止同轮调参和补跑。

### Follow-up 2. 真实模型随机性如何处理？

回答框架：固定identity与27-attempt矩阵；保持task/repeat order；使用paired category结果；同时报告timeout、latency和tokens；不把单次变化外推为普遍能力。

### Follow-up 3. 为什么hash本身不能证明正确？

回答框架：hash只证明bytes identity；还需验证内容语义、声明与观察一致、privacy边界和统计口径。

### Follow-up 4. 什么情况下必须判experiment invalid？

回答框架：behavior identity mismatch、trace/evidence不完整、runtime/infra/grader错误、attempt矩阵不完整，或未满足预注册的VALID guardrail。

### Follow-up 5. 下一步最小动作是什么？

回答框架：只补齐tracked preregistration文档；人工review；clean commit；重新preflight；再单独请求真实模型授权。

## 23. Commands and files to revisit

### Commands

```bash
uv run pytest tests/unit/test_loop.py -q
uv run pytest tests/unit/test_context_system_prompt.py -q
uv run pytest tests/unit/test_runner.py -q
uv run pytest tests/unit/test_spawn_agent_tool.py -q
uv run pytest tests/eval/test_phase8a_runner.py -q
uv run pytest tests/benchmark/test_experiment_identity.py -q
uv run pytest tests/benchmark/test_report.py -q
uv run pytest tests/eval -q
uv run pytest tests/benchmark -q
uv run pytest tests/ -q
uv run ruff check src tests scripts
uv run mypy src
uv run python scripts/gen_protocol_doc.py --check
git diff --check
```

### Evidence files

- `src/kama_claude/core/loop.py`
- `benchmarks/experiments/kama-coding-mvp-v1-deepseek-v4-pro-requirement-contract-v1.json`
- `docs/learning/devlog/phase9b-requirement-contract-implementation.md`
- `docs/learning/devlog/phase9b-holdout-authoring-rubric-v1.md`

## 24. Final evidence checklist

- [x] Git chain、branch、remote和clean状态匹配。
- [x] Commit职责与blob hashes匹配。
- [x] 单变量production diff和prompt composition完成审计。
- [x] Declared/observed identity通过真实本地无网络链路验证。
- [x] Focused、eval、benchmark和977项full suite fresh通过。
- [x] Static、protocol、diff和scope gates通过。
- [x] 三次历史experiment artifacts保持完整。
- [x] 新output root未创建。
- [x] Credential值未输出或持久化。
- [ ] Primary与hard guardrails已成为Git committed evidence。
- [x] 本轮没有stage、commit、push或运行真实模型。
