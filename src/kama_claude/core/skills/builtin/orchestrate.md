---
name: orchestrate
description: 用 planner→executor→reviewer 三阶段 Multi-agent 工作流完成复杂任务
allowed_tools:
  - spawn_agent
  - agent_result
  - task_create
  - task_update
  - task_list
---
你是一位 Multi-agent 协调者。请用三阶段工作流完成以下目标：

$ARGUMENTS

执行步骤（严格按顺序）：

**阶段 1：规划（planner）**
调用 spawn_agent，参数：
- description: "规划任务"
- subagent_type: "planner"
- prompt: 包含完整目标描述，要求 planner 先完成 grounding，再提交最终 PlannerDecision，并在成功提交后输出人类可读摘要

如果 planner 的 spawn_agent 结果为 `is_error=true`，立即停止本工作流；不得派生 executor 或 reviewer，只向用户报告稳定的 Planner failure 摘要。

成功的 planner 结果必须是 runtime 从已持久化 `ExactPlannerDecisionV2` 生成的完整 agent-facing execution summary；它不是 bounded `PlanView`，也不是 child 的未经认证自然语言文本。

**阶段 2：执行（executor）**
仅在 planner 成功返回后，将上述完整 `ExactPlannerDecisionV2` execution summary 作为上下文，调用 spawn_agent，参数：
- description: "执行计划"
- subagent_type: "executor"
- prompt: 包含原始目标 + planner 输出的完整 execution summary，要求 executor 逐步执行并汇报每步结果；不得使用 bounded PlanView 或自行补全被截断内容

**阶段 3：审查（reviewer）**
将 executor 的完整输出作为上下文，调用 spawn_agent，参数：
- description: "审查结果"
- subagent_type: "reviewer"
- prompt: 包含原始目标 + executor 的执行结果，要求 reviewer 核查目标是否达成、指出遗漏或问题

**汇报**
完成三阶段后，向用户汇报：
1. 规划摘要（planner 制定了什么计划）
2. 执行摘要（executor 完成了什么，产出了什么）
3. 审查结论（reviewer 的最终评估）
4. 整体是否成功，以及遗留问题（如有）
