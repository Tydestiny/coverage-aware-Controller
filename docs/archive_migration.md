# Historical Study Migration

The historical study is stored in an external archive outside both Git repositories. Pass its location explicitly with `experiment.py list --archive-root <ARCHIVE_ROOT>`. The official AppWorld dataset was not moved. The archive preserves the per-file `files.sha256.jsonl` index, `archive_manifest.json`, and `source_snapshot_HEAD.zip`.

| Old location | Archived location | Active location |
| --- | --- | --- |
| `appworld-controller-agent/results/` | `controller_results/` | New runs: `artifacts/experiments/` |
| `appworld/experiments/outputs/` | `appworld_outputs/` | Empty until new train/dev runs |
| `controller_agent/controller/prompts/` (G0/G1/G2 and templates) | `prompts/` | No default prompt |
| `oracle_instructions.jsonl` | Archive root | None |
| `replay_experiment_audit.md`, `task_overlap_audit.md` | Archive root | None |

Old test-specific collector, G0 policy, prompt optimizer, replay repetition runner, migration-only script, obsolete team-work/status/dependency plans, and their dedicated tests were removed from the active checkout after reference checks. The source ZIP preserves their earlier tracked versions. The reusable Controller, Trigger, StateBuilder, AppWorld integration, checkpoint, official evaluator and analysis modules remain.

Exact active-source removals:

- `controller_agent/controller/policy_g0.py` and the six files formerly under `controller_agent/controller/prompts/`.
- `oracle_instructions.jsonl`, `replay_experiment_audit.md`, and `task_overlap_audit.md` (all separately preserved in the archive).
- `scripts/collect_test_baseline_controller.py`, `scripts/optimize_controller_prompt.py`, `scripts/run_guidance_replay_repetitions.py`, and `scripts/migrate_branch_results_layout.py`.
- `controller_agent/tests/test_collect_test_baseline_controller.py` and `controller_agent/tests/test_optimize_controller_prompt.py`, which tested only the removed scripts.
- `docs/team_work_plan.md`, `docs/project_status_audit.md`, and `docs/module_dependency_map.md`; the old tracked copies remain in the source ZIP.
- The one-time archive helper was removed after successful verification to prevent accidental re-archiving over the index.

## Verification

- Every archived file was SHA-256 compared with its source before the two output roots were cleared. The original non-output prompts/notes were removed only after archiving; two untracked audit documents were individually rehashed before removal.
- The v2 study summary `controller_results/end_to_end/test_qwen3coderflash_deepseekv4pro_g2_fullcontext_v2/analysis/summary.json` matches its SHA-256 index (`87d7521b304ac643a7696ac7ce47539b0c6ae77f339bfb6e8da2416950739d67`). It still reports 60 tasks, 11 baseline successes, and 17 Controller successes.
- The archived `appworld_outputs/controlled_react_checkpoint_collection/tasks/07b42fd_1/checkpoints/manifest.json` lists `07b42fd_1_post_0001`; the corresponding file exists under `post_observation/`. **Its stored `path` points to a former teammate's absolute path**, so old checkpoints need an explicit archive-root-relative resolver before replay. File preservation is verified, direct restoration from every old manifest is not.

No old result was moved into new active experiments. Archived test results are read-only historical material, not a development input. The parent directory is not a backup service; make a separate backup before changing or deleting the archive.
