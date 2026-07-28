# Dependency Planner Test Generation — Author Review

## Idea and task design rationale

This challenging test-generation task asks the solver to derive a graph contract
spread across the README, model, validation, and planning modules. Difficulty
comes from cross-module behavior and oracle design—not from modifying
production code, repository size, or unstated requirements.

## Behavioral contract

- Dependencies precede dependents and ready ties preserve input order.
- Disconnected tasks remain in the plan.
- Duplicate names, unknown dependencies, and cycles raise distinct errors.
- The caller's sequence and immutable task values remain unchanged.
- Production source remains byte-identical.

## Oracle design

The primary oracle hashes the complete production file set, runs candidate
tests on correct source, then runs them in fresh processes against six
single-defect mutants. A separate trace-based coverage criterion requires the
critical validation and cycle branches, but mutation sensitivity—not coverage
alone—establishes behavioral quality.

## Alternative and wrong probes

The alternative uses table-driven valid and invalid graph matrices. Four
plausible incomplete suites each omit one contract family: cycles, unknown
dependencies, duplicates, or disconnected nodes. They preserve source and meet
the coverage threshold but fail mutation sensitivity.

## Failure modes and security

- High line coverage can still accept a wrong exception class or silent cycle.
- Running mutants in one interpreter could leak import cache state.
- Hashing only one production file would permit answer-by-source-edit.
- The offline fixture is trusted; process isolation is not a hostile-code sandbox.

## Review status

Author-validated and pending external review. Formal suite membership remains
unfrozen until the suite-level balance review and Batch 3.
