# dependency_planner

`plan(tasks)` returns task names in an executable order.

- Every dependency appears before its dependent.
- When several tasks are ready, their original input order is preserved.
- Disconnected tasks remain in the plan.
- Duplicate task names raise `DuplicateTaskError`.
- A dependency that names no task raises `UnknownDependencyError`.
- Cycles raise `DependencyCycleError`.
- Planning does not mutate the caller's sequence or task objects.

Add tests for this contract. Do not modify production files in
`dependency_planner/`.
