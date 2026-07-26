# Configuration Precedence — Author Review

## Idea and task design rationale

This medium bug-fixing task presents one observable precedence regression across
defaults, file-like mappings, and environment normalization. The issue names the
public API and contract but not the merge implementation. A solver must inspect
the resolver, normalizers, defaults, and merge boundary before changing one
causal condition.

## Behavioral contract

- Precedence is defaults, user, project, then environment.
- `timeout_s=0` and `verbose=False` are valid explicit values.
- Missing keys and `None` inherit lower-precedence values.
- Existing defaults, truthy overrides, and validation remain unchanged.

## Oracle design

Target tests isolate falsey values at user, project, and environment layers and
separately exercise `None`. Regression tests cover defaults, truthy multi-layer
precedence, unknown keys, and invalid environment booleans. The oracle calls
only the public `resolve_settings` API.

## Alternative and wrong probes

The alternative uses a filtered mapping update instead of the reference branch.
Three plausible shortcuts are rejected: reversing user/project, special-casing
only timeout zero, and applying `None` as an explicit override.

## Failure modes and security

- A truthy-only test matrix would miss the causal defect.
- Testing environment alone would not prove project-over-user precedence.
- Inspecting a private helper would overfit the reference structure.
- The fixture is offline and trusted. Process isolation is lifecycle isolation,
  not a sandbox for hostile code.

## Review status

Author-validated and pending external review. Task version and suite membership
remain unfrozen until Batch 3.
