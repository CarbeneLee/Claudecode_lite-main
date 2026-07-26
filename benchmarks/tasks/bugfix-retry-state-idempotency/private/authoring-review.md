# Retry State and Idempotency — Author Review

## Idea and task design rationale

This challenging bug-fixing task exposes one observable lifecycle defect across
the scheduler, job state, retry policy, executor effects, and store. Difficulty
comes from reconstructing state invariants and call accounting, not repository
size or an undisclosed requirement.

## Behavioral contract

- Every executor invocation increments `attempts`.
- Retryable failures stop at the configured maximum.
- Permanent failures stop immediately.
- Success commits one effect; terminal reruns commit no further effect.
- Different jobs retain independent state.
- Existing first-success and store behavior remains stable.

## Oracle design

Target tests observe only public types and pair state with executor call/effect
counters. Separate cases cover retry-then-success, exhaustion, permanent
failure, both terminal states, and cross-job isolation. Regression tests cover
the existing successful path and store contract.

## Alternative and wrong probes

The alternative moves attempt accounting into a job lifecycle method. Four
plausible defects are rejected independently: counting only success, allowing
one extra attempt, retrying permanent failures, and rerunning terminal jobs.

## Failure modes and security

- State-only assertions could miss duplicate side effects.
- Call-count-only assertions could miss incorrect terminal state.
- A single fail-then-success test would not establish retry bounds.
- The trusted offline fixture uses lifecycle isolation only, not a sandbox.

## Review status

Author-validated and pending external review. Formal suite membership remains
unfrozen until the suite-level balance review and Batch 3.
