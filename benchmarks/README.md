# KamaClaude Fixed-task Internal Benchmark

This directory contains trusted, local fixtures for the Phase 8B internal benchmark.

It is not SWE-bench, does not measure general coding ability, and does not provide a
security sandbox. The frozen `kama-coding-mvp-v1` suite contains nine trusted tasks:

- bug fixing at easy, medium, and challenging difficulty;
- feature implementation at easy, medium, and challenging difficulty;
- test generation at easy, medium, and challenging difficulty.

Agents receive only each task's `public/task.json` goal and `public/workspace`.
Private graders, hidden tests, validation scripts, and reference patches are injected
only after the Phase 8A worker has exited.

`suites/kama-coding-mvp-v1.freeze.json` records the ordered task inventory and
content hashes used to detect post-freeze drift. It is audit evidence, not an
additional runtime configuration file.
