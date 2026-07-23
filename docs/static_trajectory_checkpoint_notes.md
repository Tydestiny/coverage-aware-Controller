# Static AppWorld Trajectory Checkpoint Notes

This note records what is actually available in the current AppWorld baseline outputs and what the offline checkpoint replay prototype can validate.

## Inspected Data

Primary inspected baseline:

```text
D:\TY\SJTU\CIS\adaptive_compaction_validation\appworld\experiments\outputs\simplified_react_code_agent\alibaba\qwen3-coder-flash-2025-07-28\dev\tasks\23cf851_1
```

Available files:

- `logs/logger.jsonl`
- `logs/environment_io.md`
- `logs/lm_calls.jsonl`
- `logs/api_calls.jsonl`
- `misc/usage.json`
- `evaluation/report.md`
- `version/code.txt`
- `version/data.txt`
- `dbs/*.jsonl`

Version fields found:

- AppWorld code version: `0.2.0.dev0:a072b7a`
- AppWorld data version: `0.2.0`

## Field Availability

`task_id`

- Available from the task output directory name.
- Also appears in aggregate evaluation paths.

Per-step Python code

- Available in `logs/logger.jsonl` agent records as fenced Python code blocks.
- Also available in `logs/environment_io.md`.
- The new loader extracts the first Python code block from each agent message.

Per-step environment output

- Available in `logs/logger.jsonl` environment records for ordinary steps.
- Also available in `logs/environment_io.md`.
- For `23cf851_1`, `logger.jsonl` has 10 agent records and 9 environment records, while `environment_io.md` has 10 environment interactions. The final `complete_task` output is therefore not paired in `logger.jsonl`.

Agent messages / reconstructable visible history

- `logs/lm_calls.jsonl` stores full request messages for each model call.
- `logs/logger.jsonl` stores agent and environment messages sufficient to reconstruct visible history for ordinary intermediate checkpoints.
- The prototype rebuilds messages from `logger.jsonl` without reading evaluator or ground truth files.

Task initial state

- The official AppWorld task data is available under the AppWorld repository `data/tasks`.
- Existing task output directories contain final `dbs/*.jsonl` snapshots, but those are final output snapshots and must not be used to construct an initial checkpoint.
- The replay prototype creates a fresh `AppWorld(task_id=...)` instead of using final DB files as state.

Complete task execution position

- Detectable in the final agent message in `logs/logger.jsonl`.
- Detectable in `logs/environment_io.md` as the final environment interaction.
- Detectable in `logs/api_calls.jsonl` as `POST /supervisor/message`.
- Because `logger.jsonl` may not pair the final environment output, final-step checkpoint reconstruction should parse `environment_io.md` or use online checkpoint capture.

Evaluator result

- Available in `evaluation/report.md` and evaluation JSON files.
- Not used by checkpoint reconstruction.
- Must not be included in `ControllerInput`, checkpoint Agent messages, or replay state construction.

## Prototype Result

The minimal experiment loaded `23cf851_1` from `logger.jsonl`, selected `checkpoint_step = 4`, and replayed static code into a fresh AppWorld instance without calling any LLM backend.

Output:

```text
results/static_checkpoint_replay/minimal_replay_result.json
```

Observed result:

- Loaded replayable steps: 9
- Replayed through checkpoint step: 4
- Environment executions: 5
- Replay status: `valid`
- Validation errors: none
- Saved state id: `replay_23cf851_1_step_4`
- `load_state` plus read-only probe: no errors

The read-only probe validates AppWorld API/database restore behavior. It does not prove that the Python REPL local variable namespace is serialized across `load_state`. Static replay currently reconstructs visible Agent messages and `last_execution_outputs`, but it does not serialize local variables such as `passwords`, `login_result`, or app access tokens except insofar as they appear in visible history.

## Important Restore Detail

AppWorld `save_state()` without an argument returns `None` and saves to the output DB directory. AppWorld `save_state(state_id)` returns the provided state id and stores a named checkpoint under the output checkpoint directory. The replay builder therefore uses a deterministic named state id when the world supports that signature.

`load_state(state_id)` internally calls `AppWorld.close_all()`. When it is used inside an active `with AppWorld(...)` context manager, this can trigger a double-close/freezegun lifecycle error on Windows. The minimal experiment therefore used a non-context-manager world lifecycle and then called `AppWorld.close_all()` explicitly.

## Current Sufficiency Assessment

Static trajectories are sufficient for intermediate checkpoint reconstruction when:

- the step has a Python code block;
- `logger.jsonl` contains the matching environment output;
- the checkpoint is before the unpaired final `complete_task` gap;
- replayed output exactly matches the logged output.

Static trajectories are not yet fully sufficient for robust final-step reconstruction because the final completion output may be present in `environment_io.md` but absent as a paired `logger.jsonl` environment record.

## Risks And Missing Pieces

- Final `complete_task` output can be missing from `logger.jsonl`.
- `environment_io.md` is Markdown and requires stricter parsing before using it as replay ground truth.
- Final `dbs/*.jsonl` snapshots are useful for audit but must not be used as initial state.
- Existing logs do not provide an explicit online checkpoint id for each step.
- AppWorld restore lifecycle needs careful handling outside context managers on Windows.
- Python REPL locals are not yet proven to survive `load_state`; future branch continuation may need either prefix replay in the same process or an explicit Python namespace capture mechanism.

## Recommendation

Keep the offline replay module for auditing and retroactive intermediate checkpoints. For future baselines, add online checkpoint capture through the reserved `CheckpointCaptureHook.capture(world, agent_state, step_index)` protocol so each step has:

- explicit state id;
- exact Agent messages;
- last execution outputs;
- environment execution count;
- model usage counters;
- restore validation metadata.

This future hook should be wired in a new baseline runner or adapter layer, not by changing the frozen Controller main loop.
