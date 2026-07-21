# AFTraj-2K Mapping to Canonical Schema v1.1

The adapter uses the confirmed `mistake_step` base of 0. It does not infer an offset, error cause, or corrected label.

| AFTraj field | Canonical field |
|---|---|
| `conv_id` | `trajectory_id` |
| `split` | `experiment_split` during migration/splitting |
| source split, when available | `source.original_split` |
| `domain` | `environment.domain` |
| `task` | `task.instruction` |
| `gold_answer` | `task.gold_answer` |
| `tools` | `environment.tools` |
| each `turn` | exactly one `CanonicalStep`, `source_unit_type: turn` |
| `mistake_step` | exact `annotations.primary_error.step_index` and `source_step_index` |
| `mistake_agent` | `primary_error.actor_id` |
| `mistake_reason` | `primary_error.reason` |
| `unsafe_source` | `primary_error.annotation_method` / source metadata |

Turn actor, role, content, thought, action, and observation are retained. Tool actions and observations become ordered `tool_call` and `tool_result` events; reasoning and message content become events when present. The complete source turn remains in `raw_step`. No source turn is merged or split.

The adapter accepts common string, dictionary, and null action/observation forms. Unsupported structures are rejected with a record-specific reason in `rejected_records.jsonl`. `AFTrajValidator` checks one-turn-one-step, aligned source indices, and exact error labels. The generic `CanonicalValidator` intentionally does not impose those AFTraj rules.
