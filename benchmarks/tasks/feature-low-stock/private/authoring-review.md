# Low Stock Feature — Author Review

## Task design rationale

This easy feature anchor requires a small cross-file API change: implement one
pure function and export it. The request states observable behavior and keeps
storage, I/O, and dependency concerns out of scope.

## Benchmark methodology

- Pristine workspace: feature group fails to import; regression group passes.
- Reference and structurally different list-then-sort implementations pass.
- An inclusive-boundary probe fails strict-threshold behavior.
- An input-order probe fails the alphabetical-order requirement.
- Every state is graded from three fresh workspace copies.

## Oracle design

The feature oracle deliberately uses a non-alphabetical fixture order so sorting
is observed rather than accidentally implied by the input. It also checks a
quantity equal to the threshold, a negative threshold, and both package-level
and service-level imports. Regression grading preserves Product and total_units.

## Failure modes

- Alphabetically ordered input can let an unsorted implementation pass.
- Omitting an equality-boundary item can let `<=` pass as `<`.
- Checking only the service module can miss the public export requirement.
- The fixture is trusted; process-group isolation is not a security sandbox.

## Review status

Author-validated and pending external review. During Batch 0, the authoring
probe exposed and corrected a real oracle defect: the original input order was
already sorted after filtering. The public success contract was not changed.
