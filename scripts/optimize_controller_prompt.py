#!/usr/bin/env python3
"""Generate a candidate Controller system prompt from checkpoint branch pairs.

The script diagnoses Controller decisions and guidance before deciding whether the
system prompt should change. When no change is justified, the model must not return
the current prompt again, and this script will not create a candidate prompt file.

Requirements:
    pip install openai

Environment:
    OPENROUTER_API_KEY=...
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


OPTIMIZER_SYSTEM_PROMPT = """You review and, only when justified, optimize the
system prompt of a runtime Controller that supervises a ReAct-style code agent.

The Controller observes the current execution state, decides whether the Agent may
proceed or should be interrupted, and may provide corrective guidance.

You will receive:
1. the current Controller system prompt;
2. contrastive checkpoint branches, normally including baseline and controller
   continuations and sometimes an oracle-instruction branch;
3. evaluator results and Controller call records.

Your first responsibility is diagnosis, not prompt rewriting.

For every checkpoint, determine:
1. whether the Controller should have intervened;
2. whether its selected action was appropriate;
3. whether its guidance was sufficiently specific and actionable;
4. whether the Agent followed and correctly executed that guidance;
5. whether any subsequent failure is attributable to:
   - an incorrect Controller decision;
   - insufficient or vague Controller guidance;
   - the Agent ignoring or incorrectly executing adequate guidance;
   - infrastructure or evaluation failure;
   - insufficient evidence to decide.

A failed controller branch is not automatically evidence that the Controller prompt
is defective. A successful branch is not automatically evidence that the Controller
was correct. Judge the reasoning process, intervention, guidance, and Agent response,
not only the final evaluator outcome.

Critical no-change rule:
If the Controller selected an appropriate action and provided guidance that was
already clear, targeted, and actionable, but the Agent ignored, misunderstood, or
incorrectly executed it, attribute that failure to the Agent. Do not modify the
Controller system prompt based on that sample.

Modify the Controller system prompt only when the evidence reveals a reusable
Controller-side defect, such as:
- failing to intervene when intervention was necessary;
- intervening when the Agent should have been allowed to proceed;
- selecting an inappropriate Controller action;
- giving guidance too vague for the Agent to act on;
- failing to identify the relevant unsupported assumption, missing evidence,
  target-set error, API uncertainty, or unsafe operation;
- showing the same general Controller failure across multiple checkpoints.

Prefer no change over speculative change. Do not add a rule for an isolated,
task-specific failure unless it clearly exposes a general Controller defect.

When an oracle instruction is present, use it only to judge whether the Controller
missed an important corrective direction. Do not copy task-specific oracle content
into the candidate system prompt, and do not make the Controller reveal an answer or
an exact task-specific action sequence.

Hard constraints:
1. Do not encode task-specific answers, entities, identifiers, user data, API results,
   exact action sequences, or facts copied from supplied checkpoints.
2. Preserve the Controller action vocabulary and required output schema exactly.
3. Preserve effective existing rules unless evidence clearly supports changing them.
4. Guidance should identify the incorrect assumption, missing evidence, unsafe action,
   or next type of verification without solving the task for the Agent.
5. Avoid unnecessary blocking when the Agent already has sufficient evidence.
6. Distinguish a valid result supported by sound reasoning from a correct result
   reached accidentally or through unsupported reasoning.
7. Prefer concise, testable, task-agnostic rules.
8. Return valid JSON only.

Use one of these classifications for each checkpoint:
- "controller_helpful"
- "controller_sufficient_but_agent_failed"
- "controller_guidance_insufficient"
- "controller_decision_incorrect"
- "controller_unnecessary_or_harmful"
- "controller_missed_intervention"
- "no_controller_prompt_issue"
- "infrastructure_or_evaluation_issue"
- "ambiguous"

If no reusable Controller-side defect justifies a prompt change, return only:
{
  "sample_assessments": [
    {
      "pair_id": "...",
      "classification": "...",
      "controller_should_intervene": true,
      "controller_action_correct": true,
      "guidance_sufficient": true,
      "agent_followed_guidance": false,
      "failure_attribution": "agent",
      "prompt_change_needed": false,
      "reason": "brief explanation"
    }
  ],
  "prompt_change_needed": false,
  "no_change_reason": "brief overall explanation"
}

In the no-change case:
- do not return candidate_system_prompt;
- do not repeat, quote, or paraphrase the current system prompt;
- do not return changes, principles, or a validation plan.

If a reusable Controller-side defect does justify a change, return:
{
  "sample_assessments": [
    {
      "pair_id": "...",
      "classification": "...",
      "controller_should_intervene": true,
      "controller_action_correct": false,
      "guidance_sufficient": false,
      "agent_followed_guidance": null,
      "failure_attribution": "controller",
      "prompt_change_needed": true,
      "reason": "brief explanation"
    }
  ],
  "reusable_controller_failures": [
    {
      "pattern": "...",
      "supporting_pair_ids": ["...", "..."],
      "proposed_general_rule": "..."
    }
  ],
  "prompt_change_needed": true,
  "diagnosis": ["general Controller-side failure", "..."],
  "principles": ["task-agnostic principle", "..."],
  "changes": [
    {
      "change": "...",
      "evidence_pair_ids": ["..."],
      "expected_effect": "...",
      "regression_risk": "..."
    }
  ],
  "candidate_system_prompt": "...",
  "leakage_check": {
    "contains_task_specific_content": false,
    "notes": []
  },
  "validation_plan": ["held-out check", "..."]
}
"""


@dataclass(frozen=True)
class BranchRecord:
    task_id: str
    checkpoint_id: str
    branch: str
    branch_dir: Path
    manifest: dict[str, Any]
    config: dict[str, Any]
    result: dict[str, Any]
    evaluator_result: dict[str, Any]
    continuation: list[dict[str, Any]]
    controller_calls: list[dict[str, Any]]


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        value = json.load(f)
    return value if isinstance(value, dict) else {}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSONL: {path}:{line_number}: {exc}"
                ) from exc
            if isinstance(value, dict):
                rows.append(value)
    return rows


def first_nonempty(*values: Any) -> Any:
    for value in values:
        if value not in (None, ""):
            return value
    return None


def nested_get(data: dict[str, Any], *paths: tuple[str, ...]) -> Any:
    for path in paths:
        current: Any = data
        valid = True
        for key in path:
            if not isinstance(current, dict) or key not in current:
                valid = False
                break
            current = current[key]
        if valid and current is not None:
            return current
    return None


def normalize_branch_name(name: str) -> str:
    normalized = name.strip().lower().replace("-", "_")
    aliases = {
        "baseline": "baseline",
        "controller": "controller",
        "oracle": "oracle",
        "oracle_instruction": "oracle",
        "oracle_instructions": "oracle",
    }
    return aliases.get(normalized, normalized)


def infer_identity(branch_dir: Path) -> tuple[str, str, str] | None:
    manifest = read_json(branch_dir / "manifest.json")
    config = read_json(branch_dir / "config.json")
    result = read_json(branch_dir / "result.json")

    task_id = first_nonempty(
        manifest.get("task_id"),
        config.get("task_id"),
        result.get("task_id"),
    )
    checkpoint_id = first_nonempty(
        manifest.get("checkpoint_id"),
        config.get("checkpoint_id"),
        result.get("checkpoint_id"),
    )
    branch = first_nonempty(
        manifest.get("branch"),
        config.get("branch"),
        result.get("branch"),
        manifest.get("logical_branch"),
        result.get("logical_branch"),
    )

    if task_id and checkpoint_id and branch:
        return str(task_id), str(checkpoint_id), str(branch)

    parts = branch_dir.name.rsplit("__", 2)
    if len(parts) == 3:
        return parts[0], parts[1], parts[2]
    return None


def discover_branches(experiment_root: Path) -> list[BranchRecord]:
    tasks_root = experiment_root / "tasks"
    if not tasks_root.exists():
        raise FileNotFoundError(f"Missing tasks directory: {tasks_root}")

    records: list[BranchRecord] = []
    for branch_dir in sorted(path for path in tasks_root.glob("*/*") if path.is_dir()):
        if branch_dir.parent.name == "checkpoints":
            continue
        if not (branch_dir / "result.json").exists():
            continue

        identity = infer_identity(branch_dir)
        if identity is None:
            continue

        task_id, checkpoint_id, branch = identity
        records.append(
            BranchRecord(
                task_id=task_id,
                checkpoint_id=checkpoint_id,
                branch=branch,
                branch_dir=branch_dir,
                manifest=read_json(branch_dir / "manifest.json"),
                config=read_json(branch_dir / "config.json"),
                result=read_json(branch_dir / "result.json"),
                evaluator_result=read_json(branch_dir / "evaluator_result.json"),
                continuation=read_jsonl(branch_dir / "continuation.jsonl"),
                controller_calls=read_jsonl(branch_dir / "controller_calls.jsonl"),
            )
        )
    return records


def branch_outcome(record: BranchRecord) -> dict[str, Any]:
    result = record.result
    evaluator = record.evaluator_result

    return {
        "success": first_nonempty(
            result.get("evaluator_success"),
            nested_get(result, ("evaluation", "success"), ("success",)),
            evaluator.get("evaluator_success"),
            evaluator.get("success"),
        ),
        "score": first_nonempty(
            result.get("evaluator_score"),
            nested_get(result, ("evaluation", "score"), ("score",)),
            evaluator.get("evaluator_score"),
            evaluator.get("score"),
        ),
        "termination_reason": result.get("termination_reason"),
        "task_completed": result.get("task_completed"),
        "continuation_tokens": first_nonempty(
            result.get("continuation_combined_total_tokens"),
            nested_get(result, ("tokens", "continuation", "combined_total")),
            result.get("continuation_total_tokens"),
        ),
        "effective_total_tokens": first_nonempty(
            result.get("effective_total_tokens"),
            nested_get(result, ("tokens", "effective_total", "total")),
        ),
    }


def clip(value: Any, limit: int) -> Any:
    if value is None or not isinstance(value, str):
        return value

    value = value.strip()
    if limit <= 0 or len(value) <= limit:
        return value
    return value[:limit] + " …[truncated]"


def select_rows(
    rows: list[dict[str, Any]],
    max_rows: int,
) -> list[dict[str, Any]]:
    if max_rows <= 0 or len(rows) <= max_rows:
        return rows

    head_count = max_rows // 2
    tail_count = max_rows - head_count
    return rows[:head_count] + rows[-tail_count:]


def compact_continuation(
    rows: list[dict[str, Any]],
    max_rows: int,
    text_limit: int,
) -> list[dict[str, Any]]:
    fields = (
        "step_number",
        "agent_call_id",
        "agent_output",
        "generated_code",
        "controller_triggered",
        "controller_call_id",
        "controller_action",
        "guidance",
        "execution_allowed",
        "environment_observation",
        "task_completed",
        "termination_reason",
    )
    return [
        {
            key: clip(row.get(key), text_limit)
            for key in fields
            if key in row
        }
        for row in select_rows(rows, max_rows)
    ]


def compact_controller_calls(
    rows: list[dict[str, Any]],
    max_rows: int,
    text_limit: int,
) -> list[dict[str, Any]]:
    fields = (
        "controller_call_id",
        "step_number",
        "controller_input",
        "controller_response",
        "raw_response",
        "action",
        "reason",
        "guidance",
        "blocked",
        "fail_open",
        "record_origin",
        "prompt_hash",
    )

    compacted: list[dict[str, Any]] = []
    for row in select_rows(rows, max_rows):
        item: dict[str, Any] = {}
        for key in fields:
            if key not in row:
                continue
            value = row.get(key)
            if isinstance(value, (dict, list)):
                value = json.dumps(value, ensure_ascii=False)
            item[key] = clip(value, text_limit)
        compacted.append(item)
    return compacted


def extract_oracle_reference(record: BranchRecord) -> dict[str, Any] | None:
    sources = (record.manifest, record.config, record.result)

    def find_value(*paths: tuple[str, ...]) -> Any:
        for source in sources:
            value = nested_get(source, *paths)
            if value is not None:
                return value
        return None

    action = find_value(
        ("oracle_action",),
        ("oracle", "action"),
        ("injection", "oracle_action"),
    )
    instruction = find_value(
        ("oracle_instruction",),
        ("oracle", "instruction"),
        ("injection", "oracle_instruction"),
    )

    if action is None and instruction is None:
        return None
    return {
        "oracle_action": action,
        "oracle_instruction": instruction,
    }


def build_pairs(
    records: Iterable[BranchRecord],
    max_pairs: int,
    max_steps_per_branch: int,
    text_limit: int,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, BranchRecord]] = {}

    for record in records:
        key = (record.task_id, record.checkpoint_id)
        branch_name = normalize_branch_name(record.branch)
        grouped.setdefault(key, {})[branch_name] = record

    pairs: list[dict[str, Any]] = []
    for (task_id, checkpoint_id), branches in sorted(grouped.items()):
        baseline = branches.get("baseline")
        controller = branches.get("controller")
        oracle = branches.get("oracle")

        if baseline is None or controller is None:
            continue

        sample: dict[str, Any] = {
            "pair_id": f"{task_id}::{checkpoint_id}",
            "task_id": task_id,
            "checkpoint_id": checkpoint_id,
            "branches": {
                "baseline": {
                    "outcome": branch_outcome(baseline),
                    "continuation": compact_continuation(
                        baseline.continuation,
                        max_steps_per_branch,
                        text_limit,
                    ),
                    "evaluator_result": baseline.evaluator_result,
                },
                "controller": {
                    "outcome": branch_outcome(controller),
                    "continuation": compact_continuation(
                        controller.continuation,
                        max_steps_per_branch,
                        text_limit,
                    ),
                    "controller_calls": compact_controller_calls(
                        controller.controller_calls,
                        max_steps_per_branch,
                        text_limit,
                    ),
                    "evaluator_result": controller.evaluator_result,
                },
            },
        }

        if oracle is not None:
            sample["branches"]["oracle"] = {
                "outcome": branch_outcome(oracle),
                "continuation": compact_continuation(
                    oracle.continuation,
                    max_steps_per_branch,
                    text_limit,
                ),
                "controller_calls": compact_controller_calls(
                    oracle.controller_calls,
                    max_steps_per_branch,
                    text_limit,
                ),
                "evaluator_result": oracle.evaluator_result,
            }
            oracle_reference = extract_oracle_reference(oracle)
            if oracle_reference is not None:
                sample["oracle_reference"] = oracle_reference

        pairs.append(sample)

    if max_pairs > 0:
        return pairs[:max_pairs]
    return pairs


def build_user_prompt(
    current_prompt: str,
    pairs: list[dict[str, Any]],
    single_shot: str | None,
) -> str:
    payload = {
        "objective": (
            "Review all checkpoint pairs, judge whether each Controller decision and "
            "guidance was sufficient, attribute later failures correctly, and revise "
            "the Controller system prompt only when reusable Controller-side defects "
            "justify a change."
        ),
        "current_system_prompt": current_prompt,
        "controller_contract": {
            "actions": [
                "PROCEED",
                "VERIFY_TARGET_SET",
                "LOOKUP_API_SPEC",
                "REPLAN",
                "BLOCK_RISKY_ACTION",
            ],
            "non_negotiable": (
                "Preserve the existing action vocabulary and output schema."
            ),
        },
        "contrastive_branch_samples": pairs,
        "single_shot_example": single_shot,
        "instructions": [
            "Assess every checkpoint independently before making an overall decision.",
            "Do not infer prompt failure merely from evaluator failure.",
            "If guidance was adequate but the Agent failed to follow or execute it, "
            "attribute the failure to the Agent and do not change the prompt for that sample.",
            "Modify the prompt only for reusable Controller-side defects.",
            "Prefer no change over speculative or task-specific optimization.",
            "If no change is justified, omit candidate_system_prompt and return only "
            "sample_assessments, prompt_change_needed=false, and no_change_reason.",
            "Do not copy task-specific facts or oracle instructions into the prompt.",
            "Return the JSON object required by the system message.",
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        value = json.loads(text)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass

    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if fenced:
        value = json.loads(fenced.group(1))
        if isinstance(value, dict):
            return value

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        value = json.loads(text[start : end + 1])
        if isinstance(value, dict):
            return value

    raise ValueError("Optimizer response did not contain a valid JSON object.")


def validate_optimizer_output(
    output: dict[str, Any],
    output_dir: Path,
) -> tuple[bool, Path | None]:
    prompt_change_needed = output.get("prompt_change_needed")
    if not isinstance(prompt_change_needed, bool):
        raise ValueError("Optimizer response requires boolean prompt_change_needed.")

    candidate_path = output_dir / "candidate_system_prompt.txt"

    if not prompt_change_needed:
        # Remove contradictory or unnecessarily repeated prompt content.
        output.pop("candidate_system_prompt", None)
        output.pop("changes", None)
        output.pop("principles", None)
        output.pop("validation_plan", None)

        no_change_reason = output.get("no_change_reason")
        if not isinstance(no_change_reason, str) or not no_change_reason.strip():
            raise ValueError(
                "No-change response requires a non-empty no_change_reason."
            )

        # Avoid leaving a stale candidate from a previous run in the same directory.
        if candidate_path.exists():
            candidate_path.unlink()

        return False, None

    candidate = output.get("candidate_system_prompt")
    if not isinstance(candidate, str) or not candidate.strip():
        raise ValueError(
            "Prompt-change response requires candidate_system_prompt."
        )

    changes = output.get("changes")
    if not isinstance(changes, list) or not changes:
        raise ValueError("Prompt-change response requires a non-empty changes list.")

    leakage_check = output.get("leakage_check")
    if not isinstance(leakage_check, dict):
        raise ValueError("Prompt-change response requires leakage_check.")

    if leakage_check.get("contains_task_specific_content") is True:
        raise ValueError(
            "Candidate prompt failed leakage check: task-specific content detected."
        )

    candidate_path.write_text(
        candidate.strip() + "\n",
        encoding="utf-8",
    )
    return True, candidate_path


def load_repo_env(start: Path) -> None:
    """Load OPENROUTER_API_KEY from the nearest .env without overwriting env vars."""
    for directory in (start.resolve(), *start.resolve().parents):
        env_path = directory / ".env"
        if not env_path.exists():
            continue

        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
        return


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--current-prompt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="openai/gpt-5.2")
    parser.add_argument("--base-url", default="https://openrouter.ai/api/v1")
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=0,
        help="Maximum checkpoint pairs; 0 uses all available pairs.",
    )
    parser.add_argument(
        "--max-steps-per-branch",
        type=int,
        default=0,
        help="Maximum rows per branch; 0 keeps the complete continuation.",
    )
    parser.add_argument(
        "--text-limit",
        type=int,
        default=0,
        help="Maximum characters per text field; 0 disables truncation.",
    )
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--single-shot-file", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    current_prompt = args.current_prompt.read_text(encoding="utf-8").strip()
    if not current_prompt:
        raise ValueError("Current system prompt is empty.")

    single_shot = None
    if args.single_shot_file:
        single_shot = args.single_shot_file.read_text(encoding="utf-8").strip()

    records = discover_branches(args.experiment_root)
    pairs = build_pairs(
        records=records,
        max_pairs=args.max_pairs,
        max_steps_per_branch=args.max_steps_per_branch,
        text_limit=args.text_limit,
    )
    if not pairs:
        raise ValueError(
            "No checkpoint with both baseline and controller branches was found."
        )

    user_prompt = build_user_prompt(current_prompt, pairs, single_shot)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    input_path = args.output_dir / "optimizer_input.json"
    output_path = args.output_dir / "optimizer_output.json"
    metadata_path = args.output_dir / "run_metadata.json"

    input_payload = {
        "model": args.model,
        "system_prompt": OPTIMIZER_SYSTEM_PROMPT,
        "user_prompt": json.loads(user_prompt),
        "source_experiment": str(args.experiment_root),
        "selected_pair_count": len(pairs),
    }
    input_path.write_text(
        json.dumps(input_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if args.dry_run:
        print(f"Dry run complete: {input_path}")
        return 0

    load_repo_env(Path.cwd())
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENROUTER_API_KEY is not set.")

    # Delay import so --dry-run does not require the SDK.
    from openai import OpenAI

    client = OpenAI(
        api_key=api_key,
        base_url=args.base_url,
        default_headers={
            "HTTP-Referer": "https://github.com/local/controller-prompt-optimizer",
            "X-Title": "Controller Prompt Optimizer",
        },
    )

    response = client.chat.completions.create(
        model=args.model,
        messages=[
            {"role": "system", "content": OPTIMIZER_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=args.temperature,
    )

    content = response.choices[0].message.content
    if not content:
        raise RuntimeError("Optimizer returned an empty response.")

    output = extract_json_object(content)
    changed, candidate_path = validate_optimizer_output(
        output=output,
        output_dir=args.output_dir,
    )

    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    usage = response.usage
    metadata = {
        "model": args.model,
        "selected_pair_count": len(pairs),
        "prompt_change_needed": changed,
        "candidate_system_prompt": (
            str(candidate_path) if candidate_path is not None else None
        ),
        "usage": {
            "input_tokens": getattr(usage, "prompt_tokens", None),
            "output_tokens": getattr(usage, "completion_tokens", None),
            "total_tokens": getattr(usage, "total_tokens", None),
        },
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if changed:
        print(f"Prompt change recommended: {candidate_path}")
    else:
        print("No Controller prompt change recommended.")
    print(f"Details: {output_path}")
    print(f"Metadata: {metadata_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise