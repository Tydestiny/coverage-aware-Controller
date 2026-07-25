# Runtime Dense Checkpoint Collection

This document describes the minimal runtime checkpoint collection path for future AppWorld baseline runs. It does not implement new triggers, Controller prompts, or BranchRunner logic.

## Scope

The collection layer is implemented under:

```text
controller_agent/branching/
  checkpoint_capture_hook.py
  runtime_checkpoint_manager.py
  checkpoint_storage.py
  namespace_adapter.py
```

`ControlledAppWorldRunner` accepts an optional `checkpoint_capture_hook`. When the hook is `None`, behavior remains unchanged. Hook exceptions are caught and recorded on `runner.checkpoint_hook_errors`; they do not interrupt the Agent task.

No Controller, Trigger, or Agent public interface was changed.

## Capture Phases

### pre_action

Captured after an Agent produces proposed execution and before environment execution.

Fields:

- `record_id`
- `task_id`
- `agent_turn_index`
- `proposed_execution`
- `messages`
- `last_execution_outputs`
- `latest_post_observation_checkpoint_id`
- `whether_contains_complete_task`
- `whether_contains_probable_write_operation`
- `previous_execution_had_error`
- `phase = pre_action`

Pre-action records do not save AppWorld database state. They only reference the latest post-observation checkpoint.

In the current runner, the pre-action hook is invoked after `ControlledReActAgent.next_execution_inputs_usage_and_status(...)` returns. For blocked `complete_task`, this means Controller inspection has already happened, but the proposed code is still recorded before any environment execution.

### post_observation

Captured after `world.batch_execute(...)` returns observation.

Fields:

- `checkpoint_id`
- `task_id`
- `agent_turn_index`
- `environment_execution_index`
- `appworld_state_id`
- `agent_messages`
- `last_execution_outputs`
- `executed_code`
- `executed_code_prefix`
- `cumulative_agent_tokens`
- `cumulative_controller_tokens`
- `cumulative_cost`
- `controller_runtime_state`
- `appworld_code_version`
- `appworld_data_version`
- `timestamp`
- `phase = post_observation`
- `namespace_restore_mode`

The manager calls `world.save_state(checkpoint_id)` when available. This avoids the ambiguity of AppWorld `save_state()` returning `None`.

### terminal

Captured when the runner exits.

Fields:

- `before_complete_task`
- `complete_task_blocked_by_controller`
- `complete_task_entered_world_batch_execute`
- `task_completed`
- `supervisor_status`
- `termination_reason`
- `evaluator_executed`

This explicitly covers the earlier logging gap where final `complete_task` output could be missing from `logger.jsonl`.

## Storage Layout

The default runtime collection layout is task-local. New AppWorld trajectory
collection runs should keep controller records and checkpoints next to the
official AppWorld task logs:

```text
<appworld_root>/experiments/outputs/<experiment_name>/tasks/<task_id>/
  logs/
  evaluation/
  dbs/
  misc/
  version/
  controller/
    config.json
    turn_events.jsonl
    controller_events.jsonl
    evaluator_result.json
    final_smoke_report.md
  checkpoints/
    manifest.json
    post_observation/
    pre_action/
    terminal.json
```

`manifest.json` indexes every pre-action record and post-observation checkpoint, so offline trigger screening can locate the closest runnable state for any Agent turn.

The older central layout is still supported for tests and early validation:

```text
results/checkpoints/<run_id>/<task_id>/
```

Use central layout only for mock runs or temporary validation artifacts. Use
task-local layout for any trajectory collection intended for downstream
checkpoint selection or branch experiments.

## Namespace Restore

Current conclusion: Python namespace restore is not supported.

The AppWorld simplified ReAct Agent exposes message state, but the Python execution namespace is owned by the AppWorld environment/requester layer. I did not find a stable public API for serializing and restoring the Python REPL local variables. Therefore:

- `ReplayRequiredNamespaceAdapter.is_supported()` returns `False`.
- Checkpoints record `namespace_restore_mode = "replay_required"`.
- Each post-observation checkpoint stores `executed_code_prefix`.

Future branch continuation should either replay the code prefix after loading AppWorld state or add a separately reviewed namespace capture mechanism.

## Restored-Branch Analysis Artifacts

Restored branch runs keep two deliberately different trajectory views:

- `continuation.jsonl` is the authoritative checkpoint-after branch. Branch
  tokens, cost, success, and baseline-versus-Controller comparisons use only
  this scope.
- `trajectory.jsonl` is the directly readable complete view. It materializes the
  common `checkpoint_prefix` followed by the branch-specific
  `branch_continuation` and marks both segments explicitly.
- `checkpoint_context.json` preserves the exact checkpoint-visible messages,
  executed-code prefix, boundary indices, and cumulative prefix usage needed to
  interpret the complete view.

The prefix is built from the source collection's `turn_events.jsonl` when
available so that blocked turns are retained. Post-observation checkpoints add
the exact Agent output and observation. Each full row has a global
`sequence_index`, a segment-local `segment_sequence_index`, and a stable
`event_id`. Validation requires both experimental branches to have identical
prefixes and requires the continuation segment to match `continuation.jsonl`.

Checkpoint cumulative token and cost values remain prefix metadata. They are not
inserted into branch `token_usage.jsonl`, preventing the common prefix from being
counted twice in comparisons. This is recorded as
`branch_continuation_only` accounting scope.

Controller data is separated for later analysis without duplicating the token
source of truth:

- `controller_calls.jsonl` stores exact runtime calls and responses;
- `controller_runtime_events.jsonl` stores non-call fail-open and budget events
  at occurrence;
- `token_usage.jsonl` remains the unique per-call ledger and distinguishes
  Controller rows with `source=controller`;
- `controller_summary.json` derives calls, blocks, fail-open reasons, the fixed
  action histogram, tokens, cost availability, and latency availability;
- root `controller_results.jsonl` provides one batch-readable summary row per
  branch, including an explicit zero-call baseline row.

These additions do not change the checkpoint schema, Trigger, Controller Prompt,
Action enum, fail-open behavior, AppWorld main loop, or evaluator.

## Safety Rules

- The hook must not read evaluator results, ground truth answers, hidden task state, or future trajectory.
- Pre-action records must not duplicate AppWorld database state.
- Hook failures must not stop the Agent task.
- Blocked `complete_task` generates pre-action and terminal metadata, but no post-observation checkpoint.
- Allowed `complete_task` enters `world.batch_execute` and is reflected in terminal metadata.

## Mock Collection Experiment

A minimal mock/cached collection run was executed without model API calls.

Directory:

```text
results/checkpoints/mock_runtime_collection_demo/scripted-task/
```

Observed:

- pre-action records: 4
- post-observation checkpoints: 3
- terminal record: present
- first checkpoint id: `scripted-task_post_0001`
- `load_state` restored the first execution prefix in the mock world
- hook errors: none

This validates collection structure and manifest indexing. It does not validate real AppWorld Python namespace restoration.
