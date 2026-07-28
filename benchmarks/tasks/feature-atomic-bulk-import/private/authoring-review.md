# Atomic Bulk Order Import — Author Review

## Idea and task design rationale

This medium feature task extends an existing parser, model, service, and
in-memory store. Difficulty comes from preserving a state invariant across
multiple responsibility layers, not from repository size or external systems.

## Behavioral contract

- Every row uses the existing parser.
- Valid rows commit together and preserve input order in the summary.
- Invalid rows, duplicates, and store write failures leave the store unchanged.
- Empty input succeeds.
- The single-order API remains compatible.

## Oracle design

Target tests snapshot store state before mixed-validity, duplicate, and capacity
failures, then compare the complete post-failure state. A valid batch begins
with an existing order so a replacement that drops prior data cannot pass.
Regression tests exercise single-order create/read, validation, and capacity.

## Alternative and wrong probes

The reference stages parsed rows and calls one validated replacement. The valid
alternative performs sequential writes but restores the snapshot on any
exception. Wrong probes represent partial writes, replacement that loses
existing rows, and dictionary deduplication that silently accepts duplicate IDs.

## Failure modes and security

- Happy-path-only tests cannot prove atomicity.
- Checking only order count can miss replacement of existing records.
- A database or concurrency fixture would add environment noise, so the store is
  deterministic and in memory.
- This trusted fixture is not a sandbox for untrusted generated code.

## Review status

Author-validated and pending external review. Task version and suite membership
remain unfrozen until Batch 3.
