# Normalize Username Test Generation — Author Review

## Task design rationale

This easy test-generation anchor asks the agent to understand an existing pure
function and improve tests without modifying production code. The public request
names three behavioral dimensions but does not reveal assertions or validation
mechanics.

## Benchmark methodology

- Pristine workspace passes existing tests but fails generated-test and coverage groups.
- Reference and parametrized alternative test suites pass every criterion.
- Two plausible suites with enough passing tests and line coverage deliberately
  omit one requested behavior each and must fail the generated-test group.
- Every state is graded from three fresh workspace copies.

## Oracle design

The grader first freezes production source, then requires candidate tests to pass
the correct implementation. It also runs those tests against three deterministic
single-fault mutants for empty input, internal spaces, and underscore handling.
Coverage remains a separate numeric signal rather than a substitute for behavior.

## Failure modes

- Test count plus line coverage can reward assertions that miss required behavior.
- Running mutants in the same interpreter can leak imported modules.
- Exposing mutant source or names publicly would reveal the private oracle.
- Mutant subprocesses isolate Python imports, not malicious code; fixtures remain trusted.

## Review status

Author-validated and pending external review. During Batch 0, wrong-patch probes
exposed and corrected a real grader defect: the former oracle accepted suites
that reached lines without testing all requested behavior. Public success
criteria were unchanged.
