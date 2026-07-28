# Bugfix Subtract — Author Review

## Task design rationale

This easy anchor isolates a one-location arithmetic defect. The issue states the
observable contract without naming the faulty expression. Signed input prevents
an absolute-difference implementation from satisfying the oracle accidentally.

## Benchmark methodology

- Pristine workspace: target group fails; regression group passes.
- Reference patch: every criterion group passes.
- Alternative implementation: algebraically equivalent implementation passes.
- Wrong probes: reversed operands and absolute difference both fail target behavior.
- Every state is graded from three fresh workspace copies.

## Oracle design

The target oracle checks positive and signed subtraction. The regression oracle
checks the two existing operations independently. Both use argv-based pytest
commands through the Phase 8A rule grader; no shell, network, clock, or random
input is used.

## Failure modes

- A too-narrow positive-only oracle could accept absolute difference.
- A public root-cause hint could turn localization into transcription.
- Reusing a mutated workspace could hide state leakage between probes.
- Process isolation protects lifecycle cleanup only; this trusted fixture is not
  an untrusted-code sandbox.

## Review status

Author-validated and pending external review. The task version is not frozen
until the full nine-task suite reaches Batch 3.
