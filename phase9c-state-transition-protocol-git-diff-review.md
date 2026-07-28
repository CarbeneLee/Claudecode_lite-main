# Phase 9C-B State-Transition Protocol — External Diff Review

## Review verdict

- External Diff Review: **BLOCKED**
- Ready to Commit: **NO**
- Ready for Preflight: **NO**
- Real model experiment: **NOT AUTHORIZED**

The implementation is technically narrow and all fresh validation gates pass, but the review contract is not yet satisfied:

1. an existing real-trace fail-closed assertion was removed while its test was replaced;
2. the preregistration and implementation records are ignored/untracked while describing the preregistration evidence as frozen or immutable.

This review is read-only with respect to the implementation. It does not repair either blocker.

## 1. Review identity and base

| Field | Value |
| --- | --- |
| Base commit | `966cfa4072c63fd8abc05870a57f0dd8f6a1ab8a` |
| Branch | `codex/evaluation-harness` |
| Remote branch | `origin/codex/evaluation-harness` |
| Production files changed | 1 |
| Review-set files | 9 |
| Patch additions/deletions | `1150 / 26` |
| Patch SHA-256 | `245effa29b6baa6199c6edac99a898239bb31b2ece4b197494051efee5225028` |

The real Git index remained empty throughout review preparation. No staging, commit, push, amend, rebase, model call, benchmark run, holdout run, or historical-artifact mutation occurred.

## 2. Exact nine-file review set

| File | Add/Delete | Bytes | SHA-256 |
| --- | ---: | ---: | --- |
| `src/kama_claude/core/loop.py` | `8 / 1` | 9,129 | `d9c4e0f0b731f4a09e254b10f16a417a4a946b76bc869be36194631e6d7c9fe9` |
| `tests/benchmark/test_experiment_identity.py` | `89 / 14` | 43,131 | `366ef4365c4e2645c420ac2eb77e12f7127e206c17789df487728c4e41291509` |
| `tests/unit/test_context_system_prompt.py` | `59 / 0` | 5,120 | `43b19e83abd0a09aaf5591296d2d7634c5a3fe442e7f057a2ad7ffec304f3e37` |
| `tests/unit/test_loop.py` | `93 / 9` | 20,254 | `519ab5cfa0b8ea36b4d83ddd25610443f5990333b39b3139f40a47d693357122` |
| `tests/unit/test_session_manager.py` | `55 / 0` | 14,344 | `3fd8b30ba9bfb71015535e9d6dacba212320ddb5967b4fda67c7ca04ec7c2413` |
| `tests/unit/test_spawn_agent_tool.py` | `14 / 2` | 23,702 | `4992c90b6c43653e4b558d921f6e001d7b385edfe4460e690ea08032f9c41253` |
| `benchmarks/experiments/kama-coding-mvp-v1-deepseek-v4-pro-requirement-contract-v2.json` | `44 / 0` | 1,447 | `5b29ee53229f89fac9fd7298f38f92122b8fe24bb2284f459185af8b5136c4ef` |
| `docs/learning/devlog/phase9c-state-transition-protocol-implementation.md` | `527 / 0` | 22,028 | `3a9864858ee4ad37f4bf4c38e4183b61ef6d1e15b650b1a24bae241de844d129` |
| `docs/learning/devlog/phase9c-state-transition-protocol-preregistration.md` | `261 / 0` | 8,495 | `e2cab23a2631522cc47812e2c43a43efd3f276c443633f0764312cc420871778` |

No benchmark task, hidden grader, suite manifest, historical experiment profile, Evaluation runtime, provider behavior, tool schema, max-step policy, or unrelated production file is part of the patch.

## 3. Patch reproducibility

The review patch was assembled from read-only Git diff fragments without modifying the real index.

Validation against a clean export of the base commit:

- `git apply --check`: passed;
- patch application: passed;
- exact file count: 9;
- every replayed file is byte-identical to the current review set;
- patch contains no extra file;
- current worktree application was intentionally not used as the authoritative check because the changes are already present there.

The patch is therefore a reproducible representation of the reviewed implementation state.

## 4. Production change audit

The only production change is the default system prompt literal in `AgentLoop.run()`.

### Byte identity

| Prompt component | Words | UTF-8 bytes | SHA-256 |
| --- | ---: | ---: | --- |
| Phase 9B base requirement contract | 97 | 676 | `90fc45897aa8323175ceb0e5be6af3561ae44b2d0ca9bcb64244411d65561812` |
| Phase 9C state-transition addition | 65 | 440 | `310b308df8cad96fc20ff63b7f50f799ed717a16a65f67db392ca4f6eb7762a3` |
| Full effective default prompt | — | — | `bc9d1a2fbcc3458efb5b153f4ff539050c96e28a9c76f3fcd8823588933eb6c0` |

Verified properties:

- the Phase 9B prompt bytes are unchanged;
- the Phase 9C addition occurs exactly once;
- the Phase 9B text precedes the Phase 9C text;
- the separator is exact;
- normalizing the one changed prompt literal back to the base value produces an AST dump identical to the base commit;
- no task name, benchmark category, failure fingerprint, expected patch, grader rule, or domain-specific workaround was added.

### Runtime boundary

Source and call-path review confirmed:

- `ExecutionContext.system_prompt` inheritance remains unchanged;
- global, project, and session prompt composition remains unchanged;
- worker, trace writer, collector, runner, tools, provider, stopping conditions, and Evaluation behavior are unchanged;
- the treatment observes the existing runtime path and changes only the default prompt content.

## 5. Prompt inheritance test matrix

The focused tests cover the treatment at the key prompt boundaries:

| Boundary | Evidence |
| --- | --- |
| Default AgentLoop prompt | exact full prompt and ordering assertions |
| Explicit system prompt override | override remains authoritative |
| ExecutionContext composition | global/project/session layers retain ordering |
| SessionManager path | effective prompt propagates without alternate construction |
| Subagent path | child execution receives the same inherited contract |
| Trace/identity path | observed prompt hash matches the treatment profile |

No test is skipped or marked expected-failure. Assertion counts increase in every modified test file, but that aggregate fact does not cure the deleted assertion blocker below.

## 6. Blocking test-contract finding

### Finding F1 — existing fail-closed assertion removed

`tests/benchmark/test_experiment_identity.py` replaces the prior Phase 9B real-trace test. The prior test explicitly:

1. produced a real traced identity;
2. corrupted its prompt hash;
3. asserted `ExperimentIdentityMismatch`.

The replacement Phase 9C real-trace test stops after the matching identity case and omits that direct mismatch assertion.

A generic parameterized mismatch test remains and still verifies fail-closed production behavior. Therefore this is not evidence that the runtime now accepts mismatches. It is nevertheless a deterministic review blocker because the review contract explicitly prohibits deleting or weakening existing assertions.

Required remediation in a later authorized implementation turn:

- restore the direct mismatched-prompt assertion in the Phase 9C real-trace identity test;
- retain the generic mismatch coverage;
- rerun the complete review and validation gates.

## 7. Treatment profile identity

The v1 profile remains unchanged:

`9346d813af69ea1d3bc3d16a5f7e9b5cecc2f601f69e40aa1c138799ed2e1da0`

The v2 treatment profile hash is:

`5b29ee53229f89fac9fd7298f38f92122b8fe24bb2284f459185af8b5136c4ef`

Normalized comparison shows that the treatment profile differs from the control only in:

- `profile_id`;
- `expected_identity.prompt_hash`.

Frozen behavior identity:

| Field | Value |
| --- | --- |
| Provider | DeepSeek |
| Protocol | `anthropic_messages` |
| SDK | `anthropic==0.111.0` |
| Prompt hash | `bc9d1a2fbcc3458efb5b153f4ff539050c96e28a9c76f3fcd8823588933eb6c0` |
| Ordered tool-schema hash | `8ea67642a27db476c048575e9a532c41a044e39f63f729260428cdce0fd35f9f` |
| Suite hash | `6e1b554df12962fc454a5b61c6e6145bd022041ec77b24149d73bd616e6044d7` |
| Max steps | 20 |
| Repeats | 3 |
| MCP | disabled |

Artifact policy and all other normalized profile fields remain unchanged.

## 8. Preregistration audit

The preregistration contains the required fixed decision contract:

- primary thresholds:
  - inventory success at least `1/3`;
  - feature success at least `6/9`;
  - overall success at least `20/27`;
- atomic plus inventory combined threshold at least `3/6`;
- hard reliability thresholds:
  - bug-fixing success `9/9`;
  - timeout count at most `3`;
  - runtime, infrastructure, trace, and grader failures all `0`;
  - scheduled, attempted, and accounted matrix `27/27/27`;
- efficiency references:
  - total token reference `88061.8772`;
  - median successful-attempt latency reference `11976.25`;
  - timeout attempts excluded where specified;
- outcomes: `ACCEPT`, `MIXED`, `REJECT`, `INVALID`;
- an explicit contract-ambiguity note;
- real-model execution remains unauthorized.

### Finding F2 — claimed immutability is not yet durable

The implementation and preregistration records are currently ignored/untracked, and the treatment profile is untracked. Consequently, the preregistration bytes have not yet been bound to a Git commit even though the documents use `FROZEN` or immutable-evidence language.

Required remediation in a later authorized implementation/commit-preparation turn:

- reconcile wording so it distinguishes content-frozen from Git-immutable;
- ensure the exact reviewed implementation record, preregistration, and treatment profile are included in the eventual immutable commit;
- regenerate the review patch and verify all hashes after remediation.

Until that happens, experiment preflight and real-model execution are NO-GO.

## 9. Fresh validation evidence

All commands below were run fresh against the reviewed worktree:

| Gate | Result |
| --- | --- |
| `tests/unit/test_loop.py` | 18 passed |
| `tests/unit/test_context_system_prompt.py` | 6 passed |
| `tests/unit/test_spawn_agent_tool.py` | 21 passed |
| `tests/unit/test_session_manager.py` | 11 passed |
| `tests/benchmark/test_experiment_identity.py` | 31 passed |
| Focused combined total | 87 passed, 0 skipped |
| Benchmark suite | 86 passed |
| Full test suite | 985 passed |
| Ruff | passed |
| mypy | passed, 115 source files |
| Protocol document check | passed |
| `git diff --check` | passed |

Passing tests demonstrate implementation behavior and regression compatibility. They do not override the explicit assertion-preservation or preregistration-immutability review requirements.

## 10. Scope and security audit

Confirmed empty scope diffs:

- benchmark tasks;
- benchmark graders;
- benchmark suites;
- historical experiment profiles other than the new treatment profile.

Security review:

- no credential value appears in the patch;
- no credential assignment or private-key material appears;
- no private trace or historical invalid experiment artifact appears;
- no host-private absolute path appears;
- no model or API call was made;
- fake providers used by tests do not read credentials or access the network;
- process isolation is not described as a security sandbox.

## 11. Independent external review

An independent reviewer reproduced the core evidence:

- patch hash and clean-base replay;
- exact nine-file reproduction;
- production byte/AST identity;
- prompt inheritance and profile comparison;
- focused and benchmark test results;
- absence of credentials, private traces, task/grader/suite mutations, and unrelated production changes.

The reviewer independently identified both blocking findings:

1. deletion of the direct real-trace prompt-mismatch assertion;
2. preregistration evidence not yet Git-immutable despite current wording.

The independent recommendation is `BLOCKED`.

## 12. Required next step

The smallest compliant next implementation turn should:

1. restore the deleted direct fail-closed assertion without weakening generic coverage;
2. correct the preregistration/implementation wording and make the exact documents and treatment profile commit-bound;
3. regenerate the complete nine-file-or-updated review patch;
4. repeat clean-base replay, independent review, focused tests, benchmark tests, full tests, Ruff, mypy, protocol check, and scope/security audits.

No experiment authorization should be considered until that remediated diff receives a new external review verdict of `PASS`.

## Final conclusion

The state-transition prompt treatment itself is narrow, reproducible, and fully green under fresh validation. The current change set is nevertheless not review-complete because it violates the assertion-preservation contract and has not yet made its preregistration evidence Git-immutable.

**External Diff Review: BLOCKED**  
**Ready to Commit: NO**  
**Ready for Preflight: NO**  
**Real model experiment: NOT AUTHORIZED**
