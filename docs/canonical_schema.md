# Canonical Trajectory Schema v1.1

`CanonicalTrajectory` is the storage and evaluation object. Future Compactors and Verifiers must receive only the separate `ModelInputView` from `src/agent_error_compaction/views.py`.

## Top-level structure

```json
{
  "schema_version": "1.1",
  "trajectory_id": "...",
  "trajectory_family_id": "...",
  "source": {"dataset": "AFTraj-2K", "split": "train", "record_id": "..."},
  "task": {"task_id": null, "instruction": "...", "gold_answer": "..."},
  "environment": {"domain": "math", "tools": []},
  "steps": [],
  "outcome": {"status": "success", "score": null, "termination_reason": null, "raw": "success"},
  "annotations": {"primary_error": null, "error_events": []},
  "derived_annotations": {"first_error_step": null, "earliest_detectable_step": null, "evidence_steps": [], "provenance": null},
  "metadata": {}
}
```

`derived_annotations` remains empty during preprocessing. No first-error, earliest-detectable, evidence, or LLM-derived label is generated.

## PrimaryErrorAnnotation

`annotations.primary_error` is null for successful trajectories. For AFTraj failures it contains:

| Field | Meaning |
|---|---|
| `step_index` | Canonical 0-based `mistake_step`. |
| `source_step_index` | Original source step index. |
| `actor_id` | Original `mistake_agent`. |
| `reason` | Original `mistake_reason`. |
| `error_type_original` | Original error type, if supplied. |
| `error_type_unified` | Reserved; currently null. |
| `label_semantics` | Always `decisive_mistake_step`. |
| `annotation_source` | Always `dataset_author`. |
| `annotation_method` | `injected` becomes `controlled_injection`; `diagnosed` becomes `post_hoc_diagnosis`. |
| `confidence` | Currently null. |

The name `error_onset_step` is not used in v1.1. The current annotation does not claim to be the first or earliest error.

## CanonicalStep

Each original turn maps to exactly one step. The following source-preservation fields remain explicit:

| Field | Meaning |
|---|---|
| `step_index` | Continuous canonical 0-based index. |
| `source_step_index` | Original source index. |
| `source_step_id` | Original step ID, if present. |
| `event_type` | Explicit or conservative source-derived event type. |
| `content` | Original content. |
| `raw_step` | Complete original turn, unchanged. |
| `token_count` | Source token count, if present. |

The normalized interaction fields are:

```json
{
  "actor": {"id": "MathSolver", "type": "agent"},
  "message_role": "assistant",
  "reasoning": "...",
  "tool_call": {"call_id": null, "tool_name": null, "arguments": null, "raw": null},
  "tool_result": {"call_id": null, "content": null, "status": null, "error_code": null, "raw": null},
  "decision_step_index": 3
}
```

Actor type is assigned only from explicit role/actor values: `environment`, `tool`, `user`, a concrete agent name as `agent`, otherwise `unknown`. AFTraj sets `decision_step_index == step_index`. Action and observation values are always retained in `tool_call.raw` and `tool_result.raw`; unknown structured fields remain null.

## Outcome

The old string outcome is migrated to an object:

```json
{"status": "failure", "score": null, "termination_reason": null, "raw": "failure"}
```

Supported status values are `success`, `failure`, `partial_success`, `timeout`, `aborted`, and `unknown`; current AFTraj records use only `success` and `failure`.

## PrefixSample

Prefix samples contain `prefix_id`, trajectory/family identifiers, `source_split`, `prefix_end_step`, the prefix steps, `critical_error_occurred`, `critical_error_step`, `distance_to_critical_error`, `responsible_actor`, and metadata.

For a failure, `critical_error_step` is `annotations.primary_error.step_index` and:

```python
critical_error_occurred = prefix_end_step >= critical_error_step
distance_to_critical_error = prefix_end_step - critical_error_step
```

For a safe trajectory, `critical_error_step` and distance are null and `critical_error_occurred` is always false. Prefix steps are exactly `trajectory.steps[:prefix_end_step + 1]`.

## Validation and migration

Validation rejects missing or non-continuous step indices, invalid outcomes, failure records without `primary_error`, success records with `primary_error`, out-of-range primary error positions, non-empty derived annotations, and mismatched source/canonical step counts.

Use `scripts/migrate_schema_v1_to_v1_1.py` to migrate the existing canonical JSONL. Use `scripts/check_migration_consistency.py` to compare trajectory order, IDs, family/split values, source indices, raw steps, critical error positions, prefix IDs, and boolean prefix labels before and after migration. The migration does not read or rewrite Parquet.

