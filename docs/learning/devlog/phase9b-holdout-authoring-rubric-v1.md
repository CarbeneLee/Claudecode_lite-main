# Phase 9B Holdout Authoring Rubric v1

## 0. Freeze metadata

- Version：`phase9b-holdout-authoring-rubric@1`
- Freeze date：`2026-07-27`
- Status：rubric frozen before any Phase 9B real-model result
- Scope：定义后续独立 coding-agent holdout 的选题、oracle 和 authoring evidence
- Out of scope：本轮不创建 task、不运行模型、不修改 Agent runtime

本 rubric 是反过拟合审计产物。它不改变 Agent 行为，也不允许根据 Phase 9B
development/regression suite 的结果回写本轮干预。

## 1. Required matrix

Holdout 必须覆盖以下矩阵：

| Category | Medium | Challenging |
| --- | ---: | ---: |
| Bug fixing | 至少2题 | 至少2题 |
| Feature implementation | 至少2题 | 至少2题 |
| Test generation | 至少2题 | 至少2题 |

正式矩阵至少包含12题。难度必须来自：

- 跨模块定位与推理；
- 可观察状态不变量；
- 输入、失败和副作用合同的组合复杂度。

难度不得来自无意义的仓库体积、重复样板、随机环境失败或未公开的产品需求。

## 2. Independence and leakage rules

每个 holdout task 必须满足：

- 不复用当前9题的业务领域；
- 不复用当前9题的函数名、类名或模块名；
- 不复用当前9题的 fixture API；
- 不复用当前9题的主要边界形式；
- 不复用当前9题的主要 failure mechanism；
- public issue/request 不包含 grader 路径、hidden tests、expected patch 或答案暗示；
- private evidence 不进入 Agent workspace、prompt、trace-facing task input 或公开报告；
- task authoring、reference validation 和 review 在 Phase 9B 真实结果解释前独立完成。

仅替换标识符、数字或领域名但保留相同控制流与 oracle，不构成独立任务。

## 3. Public/private contract

Public 部分只提供：

- 稳定 task ID；
- issue、feature request 或 test-generation request；
- 可执行 repository fixture；
- 明确且完整的 acceptance criteria；
- 运行所需、但不泄露答案的环境说明。

Private 部分必须包含：

- 自动 grader；
- hidden tests 或 validation scripts；
- requirement-to-test matrix；
- reference solution；
- alternative valid solution；
- wrong-probe evidence；
- pristine/reference/alternative/probe determinism receipts。

Process isolation 只用于 lifecycle isolation，不是恶意 shell 的安全 sandbox。Holdout
只能使用可信 fixture。

## 4. Oracle requirements

每个 task 的 oracle 必须：

1. 对每条公开 acceptance criterion 至少有一个独立验证项；
2. 区分 target behavior、failure/invalid-input behavior 与 regression；
3. 对 stateful task 明确验证失败后的状态与副作用；
4. 对 test-generation task 验证生成测试能杀死有代表性的行为 mutant，不能只看 coverage；
5. 不要求唯一 patch 或唯一代码结构；
6. reference 和 alternative valid solution 都必须通过；
7. pristine fixture 必须按任务合同失败，避免无效 benchmark；
8. grader infrastructure failure 必须与 Agent task failure 分离并 fail closed。

## 5. Required authoring evidence

每个 task 必须完成完整 lifecycle：

```text
Idea
  → behavioral contract
  → public fixture
  → private oracle
  → reference solution
  → alternative valid solution
  → wrong probes
  → determinism
  → independent review
```

Authoring package 至少包含：

- task summary；
- public/private tree；
- 独立 requirement-test matrix；
- reference validation receipt；
- alternative implementation validation receipt；
- 至少4个 plausible wrong probes；
- 每个 probe 对应的被杀死 requirement；
- 三份 fresh-copy determinism evidence；
- reviewer 对 leakage、oracle 完整性与难度来源的结论。

Wrong probes 必须代表现实中可能出现的实现错误，不能只制造语法错误或环境损坏。

## 6. Determinism policy

每个 task 必须在三份全新副本上分别验证：

- pristine 结果一致；
- reference 结果一致；
- alternative 结果一致；
- 每个 wrong probe 的失败 criterion 一致；
- command exit、grader输出与文件 hash 一致；
- 不依赖网络、外部时钟、随机顺序或主机私有状态。

若必须使用随机数据，必须冻结 seed，并证明同一 seed 的输出与 grader 结果可重放。
任一三-copy 不一致会阻止 task 进入 holdout。

## 7. Freeze and execution policy

在正式 holdout 前必须冻结并记录：

- rubric hash；
- task、grader、reference 与 suite hashes；
- Agent commit；
- experiment profile；
- provider/model/protocol/SDK；
- prompt 与 ordered tool-schema hashes；
- runtime config、max steps、timeout、repeats和execution order；
- primary metrics 与 guardrails。

候选 intervention 冻结后，只允许运行一次正式 holdout 矩阵：

- 不补跑失败 attempt；
- 不删除或替换失败 task；
- 不根据 holdout 结果在同轮修改 prompt、runtime 或 grader；
- holdout 结果不得用于同轮调参；
- invalid experiment 只报告 invalid，不发布能力分数；
- 如需新干预，必须创建新版本并使用新的、未见的 holdout。

## 8. Review gate

Task 只有在以下条件全部满足时才能进入 frozen holdout：

- category/difficulty 标签由独立 reviewer 接受；
- 难度来自推理、不变量或合同复杂度；
- public/private leakage audit 通过；
- requirement-test matrix 无遗漏；
- reference 与 alternative 均通过；
- 至少4个 wrong probes 被预期 oracle 杀死；
- 三-copy determinism 通过；
- environment 与其他 holdout task 一致；
- 没有从 Phase 9B 实验结果派生的新提示或 grader 调整。

## 9. Safe interpretation

Holdout 结果只能证明冻结 Agent 在这组预注册、独立 synthetic repository tasks 上的表现。
它不是 SWE-bench，也不能单独证明通用 coding ability、真实生产安全性或对不可信仓库的
隔离能力。
