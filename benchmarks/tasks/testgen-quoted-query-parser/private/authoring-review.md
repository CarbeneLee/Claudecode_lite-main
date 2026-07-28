# Quoted Query Parser Test Generation — Author Review

## Idea and task design rationale

This medium test-generation task asks the solver to read a documented grammar
and trace behavior across tokenizer and parser modules. Difficulty comes from
cross-module escape/quote semantics and from writing assertions that detect
behavioral faults, not from a large repository.

## Behavioral contract

- Production files under `querylang/` remain byte-identical.
- Candidate tests pass on the correct implementation.
- Tests distinguish quoted delimiters, escapes, empty quoted values, and
  unclosed quotes.
- Boundary/error coverage increases while existing tests remain green.

## Oracle design

The generated-tests validator hashes the complete production Python file set,
runs candidate tests in a clean process, and reruns them against four
single-fault mutants. Each mutant replaces only one parser or tokenizer file in
a fresh temporary package. Coverage is measured separately and cannot turn a
surviving mutant into success.

## Alternative and wrong probes

The reference uses explicit tests; the alternative uses table-driven valid and
invalid cases. Three wrong suites execute many of the same lines but omit a
meaningful assertion for escape decoding, empty quoted values, or unclosed
quotes. Each must pass correct-source tests yet fail behavior sensitivity.

## Failure modes and security

- Test count and line coverage alone can reward assertion-free execution.
- Reusing one interpreter across mutants can leak imported correct modules.
- A source-content check that ignores added files would allow production
  replacement through a new module.
- Candidate tests are trusted inputs. Temporary processes isolate imports and
  lifecycle only; they are not a hostile-code sandbox.

## Review status

Author-validated and pending external review. Task version and suite membership
remain unfrozen until Batch 3.
