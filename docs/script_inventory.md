# Script Inventory

New train/dev work starts with `python scripts\experiment.py`. Other scripts are implementation details or retained historical workflows; none should be deleted solely because its name contains `smoke`.

| Group | Scripts | Status |
| --- | --- | --- |
| Public | `experiment.py` | Supported train/dev entry for doctor, initial-state runs, one checkpoint pair, offline evaluation, analysis, and listing. Real model runs require `--execute`. |
| Runtime drivers | `run_live_controller_initial_task.py`, `run_checkpoint_dual_branch_smoke.py`, `run_live_controller_checkpoint_smoke.py`, `run_official_evaluation.py` | Called by the public entry; keep. |
| Shared legacy code | `run_controller_api_smoke.py`, `run_checkpoint_restore_continue.py` | Despite their names, other scripts import model configuration, AppWorld setup, checkpoint, and file helpers from them. Extract these helpers into library modules before any move or deletion. |
| Older experiment collectors | `collect_with_checkpoints.py`, `collect_oracle_instruction_branches.py`, `run_lightweight_branch_runs.py`, `run_single_intervention_branches.py`, `scan_single_intervention_checkpoints.py` | Retained for audit and potential checkpoint studies. Not part of the new train/dev CLI; do not use historical oracle annotations for prompt development. |
| Analysis helpers | `analyze_branch_experiment.py`, `build_end_to_end_token_usage.py`, `build_branch_token_usage.py`, `build_checkpoint_comparison.py` | Preserve while output readers and existing tests depend on them. |

The `controller_agent/` folders are reusable components with separate contracts. `branching/` owns checkpoint state and continuations, `evaluation/` owns official evaluation, and `results/` owns result readers and token ledgers. A common artifact root and manifest links make results browsable without merging those code modules or duplicating AppWorld raw logs.

Tests are organized by behavior and dataset contract. File count alone is not evidence of redundant coverage; remove an older test only after its behavior is covered by the replacement entry and the full suite still passes.
