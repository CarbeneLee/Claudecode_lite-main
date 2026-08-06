# Inventory Reservation Lifecycle — Author Review

## Idea and task design rationale

This challenging feature task extends a small inventory service with a
reservation state machine. Difficulty comes from coordinating stock mutation,
request idempotency, and legal transitions across models, store behavior, and a
new service contract—not from repository volume or hidden requirements.

## Behavioral contract

- Reserve reduces available stock atomically and creates a pending reservation.
- Equal request retries are idempotent; changed parameters conflict.
- Insufficient or invalid requests leave no partial reservation or stock change.
- Confirm and release are idempotent only in their same terminal state.
- Release restores exactly once; confirmed reservations cannot release.
- Existing inventory query and adjustment behavior remains stable.

## Oracle design

The target oracle observes returned public records and stock after every
transition. It retries a previously insufficient request after replenishment to
detect invisible ghost records, and it pairs idempotency assertions with stock
counts. Regression tests independently protect the existing service and store.

## Alternative and wrong probes

The alternative centralizes existing-request and pending-transition checks.
Four plausible implementations are rejected: ignoring request parameter
conflicts, restoring twice, storing before a failing stock mutation, and
releasing after confirmation.

## Failure modes and security

- Happy-path-only grading would miss all atomicity and idempotency defects.
- Checking status without stock would miss duplicate restoration.
- Checking stock without record identity would miss request-key conflicts.
- The offline fixture is trusted; process isolation is not a security sandbox.

## Review status

Author-validated and pending external review. Formal suite membership remains
unfrozen until the suite-level balance review and Batch 3.
