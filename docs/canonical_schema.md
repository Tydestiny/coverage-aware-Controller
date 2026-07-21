# Canonical Trajectory Schema v1.1

`CanonicalTrajectory` is the storage and offline-evaluation object. It is not a model prompt. Compactors and Verifiers may receive only the whitelist view created by `build_model_input_view`.

## Top level

```text
schema_version: "1.1"
trajectory_id
trajectory_family_id
source: {dataset, original_split, record_id}
experiment_split: train | validation | test | null
task: {task_id, instruction, gold_answer}
environment: {domain, tools}
steps: CanonicalStep[]
outcome: {status, score, termination_reason, raw}
annotations: {annotation_status, primary_error, error_events}
derived_annotations
metadata
```

`source.original_split` is original dataset provenance. `experiment_split` is assigned later by family-disjoint splitting.

## Step and event representation

Each `CanonicalStep` preserves one source boundary and contains:

```text
step_index, source_step_index, source_step_id, source_unit_type,
decision_step_index, actor: {id, type}, message_role, event_type,
content, reasoning, events, raw_step, token_count
```

`events` is an ordered list. Each `CanonicalEvent` has `event_index`, `event_type`, `actor`, `content`, optional `tool_call` and optional `tool_result`, plus source provenance fields. Tool calls and results are represented only inside this ordered list; one step may contain multiple events. `raw_step` and `raw_event` preserve source values but are never exposed to model input.

The generic validator does not require one source turn to equal one canonical step. `AFTrajValidator` adds that dataset-specific invariant.

## Annotations

`primary_error.alignment_status` is one of `exact`, `span`, `ambiguous`, or `unmapped`; `step_span` is used for span alignment. AFTraj `mistake_step` is confirmed 0-based and is mapped without offset guessing.

`annotation_status` is one of `source_provided`, `rule_mapped`, `not_provided`, `unavailable`, or `not_applicable`. Preprocessing does not create LLM-derived labels. Every `derived_annotations` field is an object with `value`, `status`, and `provenance`, defaulting to `status: not_annotated`.

## Model input

The model view contains only task instruction, environment tools, and the requested step prefix. Visible step fields are `step_index`, `decision_step_index`, `actor`, `message_role`, `event_type`, `content`, `reasoning`, and `events`. Visible event fields are `event_index`, `event_type`, `actor`, `content`, `tool_call`, and `tool_result`. Source indexes, raw values, outcomes, gold answers, families, splits, annotations, and derived fields are forbidden.
