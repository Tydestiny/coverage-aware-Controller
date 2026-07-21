# Canonical Trajectory Schema

## 1. Purpose

The canonical schema is the stable data boundary between raw AFTraj-2K records and later compaction, verification, or modeling experiments. It preserves source evidence and labels while making trajectory boundaries, family membership, split membership, and prefix semantics explicit.

The schema is implemented with Python standard-library `dataclasses` in `src/agent_error_compaction/schemas.py`. Each dataclass exposes `to_dict()` and can be serialized as one JSON object per JSONL line.

## 2. CanonicalTrajectory

Each canonical JSONL line is a `CanonicalTrajectory` object.

| Field | Type | Meaning |
|---|---|---|
| `trajectory_id` | string | Stable source identifier, mapped from `conv_id`. |
| `trajectory_family_id` | string or null | Group identifier used for leakage-safe splitting. |
| `source_dataset` | string | Source dataset name, currently `AFTraj-2K`. |
| `source_split` | string or null | Split assigned by the group splitter. |
| `domain` | string or null | Original domain, such as `math`, `coding`, or `agentic`. |
| `task_id` | string or null | Optional source task identifier. |
| `task` | JSON value | Original task without semantic rewriting. |
| `gold_answer` | JSON value | Original reference answer. |
| `outcome` | `success` or `failure` | Normalized trajectory outcome. |
| `tools` | JSON value | Original tool declarations. |
| `steps` | array | Exactly one `CanonicalStep` for every original turn. |
| `labels` | object | Source error annotations and canonical error position. |
| `metadata` | object | Conversion provenance and invariant checks. |

## 3. CanonicalStep

`CanonicalStep` contains normalized access fields and the complete original turn.

| Field | Type | Meaning |
|---|---|---|
| `step_index` | integer | Canonical 0-based position, continuous from zero. |
| `source_step_index` | integer | Original source position. |
| `source_step_id` | string or null | Explicit source step or turn ID when present. |
| `actor` | string or null | Source actor; falls back to `role` when no actor field exists. |
| `role` | string or null | Original source role. |
| `event_type` | string | Explicit type, or conservative message/action/observation classification. Empty action strings are not actions. |
| `content` | string or null | Original content in a JSON-safe representation. |
| `thought` | string or null | Original thought or reasoning field. |
| `action` | JSON value | Original action, including string, object, or null values. |
| `observation` | JSON value | Explicit observation; environment content is used only if no explicit observation exists. |
| `raw_step` | JSON object | Complete original turn, preserved losslessly. |
| `token_count` | integer or null | Source token count when supplied. |

The one-turn/one-step invariant is checked using both `metadata.num_steps` and `metadata.source_turn_count`.

## 4. TrajectoryLabels

| Field | Type | Meaning |
|---|---|---|
| `is_failure` | boolean | Must agree with `outcome`. |
| `error_onset_step` | integer or null | Canonical 0-based error position. For AFTraj-2K, this equals the source `mistake_step`. |
| `source_error_onset_step` | integer or null | Original `mistake_step`, never overwritten. |
| `responsible_actor` | string or null | Mapped from `mistake_agent`. |
| `error_type_original` | string or null | Original error type if supplied. |
| `error_type_unified` | string or null | Reserved for a later taxonomy; no inference is performed. |
| `error_reason` | string or null | Original `mistake_reason`, not a generated diagnosis. |
| `label_source` | string or null | Label origin. Safe records use `safe`; unsafe records use source `unsafe_source`. |
| `unsafe_source` | string or null | Original unsafe source; safe records are normalized to `safe`. |

## 5. PrefixSample

`PrefixSample` is used for later prefix-based experiments.

| Field | Type | Meaning |
|---|---|---|
| `prefix_id` | string | Deterministic trajectory/end-step identifier. |
| `trajectory_id` | string | Parent trajectory. |
| `trajectory_family_id` | string or null | Parent family. |
| `source_split` | string or null | Parent split. |
| `prefix_end_step` | integer | Inclusive canonical endpoint. |
| `steps` | array | Exactly `trajectory.steps[:prefix_end_step + 1]`; no future step is allowed. |
| `error_occurred` | boolean | True when a failure prefix reaches or passes the onset; always false for safe prefixes. |
| `distance_to_error_onset` | integer or null | `prefix_end_step - error_onset_step`; null for safe prefixes. |
| `metadata` | object | Prefix mode and fixed seed provenance. |

## 6. Family and split design

Family assignment happens before splitting. If a source family linkage is available, it is preferred. Otherwise a deterministic hash is computed from:

- normalized task;
- normalized gold answer;
- domain;
- normalized text of the first three raw steps.

Normalization only strips leading/trailing whitespace, collapses consecutive whitespace, and lowercases text. No semantic rewriting is performed.

Train, validation, and test are assigned by family group with a fixed random seed. A strict validation rejects any family appearing in more than one split.

## 7. Validation invariants

The validator rejects records when:

- `trajectory_id` is empty or `steps` is empty;
- `step_index` is not continuous from zero;
- canonical and source turn counts differ;
- `outcome` and labels disagree;
- a failure has no error onset, or a success has one;
- the error onset is outside the trajectory;
- `error_onset_step` does not equal `source_error_onset_step` for the confirmed 0-based AFTraj-2K data;
- a family appears in more than one split.

The preprocessing stage does not infer error reasons, rewrite human labels, merge turns, split turns, or use future steps to construct a current prefix.

