"""Offline replay of PolicyG0 on annotated Part3 checkpoints.

For each annotation checkpoint `{task}_post_{N}` (intervene BEFORE turn N+1):
- proposed code = turn_events generated_code at agent_turn_index = N+1
- history/errors from checkpoint messages + prior turn outputs
- compare PolicyG0 action vs annotated targeted_action

Does not call the LLM. confidence is ignored.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(REPO_ROOT))

from controller_agent.controller.policy_g0 import PolicyG0Backend, infer_case_type  # noqa: E402
from controller_agent.state.schemas import Action, ControllerInput  # noqa: E402


ACTION_ALIASES = {
    "CONTINUE": Action.PROCEED,
    "PROCEED": Action.PROCEED,
    "VERIFY_TARGET_SET": Action.VERIFY_TARGET_SET,
    "VERIFY_GOALS": Action.VERIFY_TARGET_SET,
    "LOOKUP_API_SPEC": Action.LOOKUP_API_SPEC,
    "QUERY_API_SPEC": Action.LOOKUP_API_SPEC,
    "REPLAN": Action.REPLAN,
    "BLOCK_RISKY_ACTION": Action.BLOCK_RISKY_ACTION,
    "BLOCK_RISKY": Action.BLOCK_RISKY_ACTION,
}

# Soft agreement families (same intervention intent)
FAMILIES = {
    Action.PROCEED: "proceed",
    Action.VERIFY_TARGET_SET: "verify_or_block",
    Action.BLOCK_RISKY_ACTION: "verify_or_block",
    Action.REPLAN: "replan_or_block",
    Action.LOOKUP_API_SPEC: "lookup",
}
# complete-before: BLOCK and REPLAN are both "stop finish" intents
COMPLETE_STOP = {Action.BLOCK_RISKY_ACTION, Action.REPLAN, Action.VERIFY_TARGET_SET}


def main() -> int:
    args = parse_args()
    anns = json.loads(Path(args.annotations).read_text(encoding="utf-8"))
    split_root = Path(args.split_root)
    backend = PolicyG0Backend()
    rows: list[dict[str, Any]] = []

    for ann in anns:
        row = replay_one(ann, split_root, backend)
        rows.append(row)
        mark = "OK" if row.get("exact_match") else ("SOFT" if row.get("soft_match") else "DIFF")
        print(
            f"[{mark}] {row['task_id']} {row['case_type']}: "
            f"ann={row['annotated_action']} g0={row['g0_action']} "
            f"reason={row.get('g0_reason')}"
        )

    n = len(rows)
    exact = sum(1 for r in rows if r.get("exact_match"))
    soft = sum(1 for r in rows if r.get("soft_match"))
    missing = sum(1 for r in rows if r.get("error"))
    summary = {
        "n": n,
        "exact_match": exact,
        "exact_rate": exact / n if n else 0.0,
        "soft_match": soft,
        "soft_rate": soft / n if n else 0.0,
        "errors": missing,
        "rows": rows,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_md(summary, out.with_suffix(".md"))
    print(f"\nExact {exact}/{n} ({summary['exact_rate']:.0%}) | Soft {soft}/{n} ({summary['soft_rate']:.0%})")
    print(f"Wrote {out}")
    print(f"Wrote {out.with_suffix('.md')}")
    return 0


def replay_one(ann: dict[str, Any], split_root: Path, backend: PolicyG0Backend) -> dict[str, Any]:
    task_id = ann["task_id"]
    checkpoint_id = ann["checkpoint_id"]
    case_type = ann.get("case_type", "")
    annotated = normalize_action(ann.get("targeted_action"))
    task_dir = split_root / task_id
    ck_path = task_dir / "checkpoints" / "post_observation" / f"{checkpoint_id}.json"
    events_path = task_dir / "collection" / "turn_events.jsonl"

    base = {
        "task_id": task_id,
        "checkpoint_id": checkpoint_id,
        "case_type": case_type,
        "annotated_action": annotated.value if annotated else ann.get("targeted_action"),
    }
    if not ck_path.exists():
        return base | {"error": f"missing checkpoint {ck_path}", "exact_match": False, "soft_match": False}
    if not events_path.exists():
        return base | {"error": f"missing turn_events", "exact_match": False, "soft_match": False}

    checkpoint = json.loads(ck_path.read_text(encoding="utf-8"))
    turn_n = int(checkpoint.get("agent_turn_index") or 0)
    # Prefer the first subsequent turn that matches case_type intent
    # (annotation is "before" a later risky step, not necessarily N+1).
    proposed, used_turn = load_relevant_proposed(
        events_path, turn_n + 1, case_type
    )
    if not proposed:
        proposed = last_assistant_code(checkpoint.get("agent_messages") or [])
        used_turn = turn_n + 1

    task_text = extract_task(checkpoint.get("agent_messages") or [])
    history = compact_history(checkpoint.get("agent_messages") or [])
    errors = extract_errors(checkpoint.get("last_execution_outputs") or [], history)
    completed = list(checkpoint.get("executed_code_prefix") or [])[-6:]

    # Map case_type to a trigger_reason PolicyG0 understands
    trigger = case_type_to_trigger(case_type, proposed)

    cin = ControllerInput(
        task=task_text,
        trigger_reason=trigger,
        recent_history=history,
        current_agent_output=proposed or "",
        known_entities=[],
        completed_operations=completed,
        recent_errors=errors,
    )
    decision = backend.decide(cin)
    inferred = infer_case_type(cin)
    exact = annotated is not None and decision.action == annotated
    soft = soft_agree(decision.action, annotated, case_type)

    return base | {
        "trigger_reason": trigger,
        "inferred_case_type": inferred,
        "proposed_code_preview": (proposed or "")[:300],
        "proposed_turn_index": used_turn,
        "g0_action": decision.action.value,
        "g0_reason": decision.reason,
        "g0_instruction_preview": (decision.instruction or "")[:200],
        "exact_match": exact,
        "soft_match": soft,
        "error": None,
    }


def case_type_to_trigger(case_type: str, proposed: str | None) -> str:
    ct = (case_type or "").lower()
    if "complete" in ct:
        return "before_finish"
    if "write" in ct or "high_risk" in ct:
        return "before_first_write"
    if "error" in ct:
        return "after_api_or_execution_error"
    code = proposed or ""
    if re.search(r"complete_task\s*\(", code, re.I):
        return "before_finish"
    if re.search(r"remove_|delete_|move_|create_|add_", code, re.I):
        return "before_first_write"
    return case_type or "unknown"


def normalize_action(name: str | None) -> Action | None:
    if not name:
        return None
    return ACTION_ALIASES.get(str(name).strip().upper())


def soft_agree(g0: Action, ann: Action | None, case_type: str) -> bool:
    if ann is None:
        return False
    if g0 == ann:
        return True
    # same soft family
    if FAMILIES.get(g0) == FAMILIES.get(ann) and FAMILIES.get(g0) != "lookup":
        return True
    if "complete" in (case_type or "") and g0 in COMPLETE_STOP and ann in COMPLETE_STOP:
        return True
    if "write" in (case_type or "") and {g0, ann} <= {Action.VERIFY_TARGET_SET, Action.BLOCK_RISKY_ACTION}:
        return True
    return False


def load_turn_code(events_path: Path, turn_index: int) -> str | None:
    for line in events_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        ev = json.loads(line)
        if int(ev.get("agent_turn_index") or -1) == turn_index:
            code = ev.get("generated_code")
            return str(code) if code else None
    return None


def load_relevant_proposed(
    events_path: Path, start_turn: int, case_type: str
) -> tuple[str | None, int]:
    """Find first turn >= start_turn whose code matches the annotated intent."""
    ct = (case_type or "").lower()
    events: list[tuple[int, str]] = []
    for line in events_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        ev = json.loads(line)
        idx = int(ev.get("agent_turn_index") or -1)
        code = str(ev.get("generated_code") or "")
        if idx >= start_turn and code:
            events.append((idx, code))

    def is_write(code: str) -> bool:
        return bool(
            re.search(
                r"\b(remove_|delete_|move_|create_|add_|update_|send_|pay_|"
                r"create_transaction|create_playlist|compress_)\w*\s*\(",
                code,
                re.I,
            )
        )

    def is_complete(code: str) -> bool:
        return bool(re.search(r"\b(?:apis\.)?supervisor\.complete_task\s*\(", code, re.I))

    if "write" in ct or "high_risk" in ct:
        for idx, code in events:
            if is_write(code):
                return code, idx
    if "complete" in ct or "coverage_incomplete" in ct or "finish" in ct:
        for idx, code in events:
            if is_complete(code):
                return code, idx
    # fallback: immediate next turn
    if events:
        return events[0][1], events[0][0]
    return None, start_turn


def last_assistant_code(messages: list[dict[str, Any]]) -> str:
    for msg in reversed(messages):
        if msg.get("role") != "assistant":
            continue
        content = str(msg.get("content") or "")
        blocks = re.findall(r"```(?:python)?\n(.*?)```", content, flags=re.S)
        if blocks:
            return blocks[-1].strip()
    return ""


def extract_task(messages: list[dict[str, Any]]) -> str:
    for msg in messages:
        if msg.get("role") != "user":
            continue
        content = str(msg.get("content") or "")
        if "Task:" in content and "Using these APIs" in content:
            m = re.search(r"Task:\s*(.+)$", content, flags=re.S)
            if m:
                return m.group(1).strip()
        if content.startswith("Task:") or "\nTask:" in content:
            m = re.search(r"Task:\s*(.+?)(?:\n\n|\Z)", content, flags=re.S)
            if m:
                return m.group(1).strip()
    # fallback: longest user message with Task
    for msg in reversed(messages):
        content = str(msg.get("content") or "")
        if "Task:" in content:
            m = re.search(r"Task:\s*(.+)$", content, flags=re.S)
            if m:
                return m.group(1).strip()[:2000]
    return ""


def compact_history(messages: list[dict[str, Any]], max_msgs: int = 12) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for msg in messages[-max_msgs:]:
        role = msg.get("role")
        content = str(msg.get("content") or "")
        if len(content) > 800:
            content = content[:800] + "...[truncated]"
        out.append({"role": role, "content": content})
    return out


def extract_errors(last_outputs: list[Any], history: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    for out in last_outputs:
        text = str(out)
        if re.search(r"Execution failed|Traceback|Error|Exception|401|422|unexpected keyword", text, re.I):
            errors.append(text[:500])
    for msg in history:
        text = str(msg.get("content") or "")
        if re.search(r"Execution failed|Traceback|unexpected keyword|Not authorized", text, re.I):
            errors.append(text[:500])
    return errors[-5:]


def write_md(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# PolicyG0 离线回放（Part3 标注）",
        "",
        f"- n = {summary['n']}",
        f"- exact match = {summary['exact_match']}/{summary['n']} ({summary['exact_rate']:.0%})",
        f"- soft match = {summary['soft_match']}/{summary['n']} ({summary['soft_rate']:.0%})",
        f"- errors = {summary['errors']}",
        "",
        "> soft：写前 VERIFY↔BLOCK；结束前 BLOCK↔REPLAN↔VERIFY 视为同向干预。",
        "",
        "| task | case_type | annotated | G0 | exact | soft | g0_reason |",
        "|------|-----------|-----------|----|-------|------|-----------|",
    ]
    for r in summary["rows"]:
        lines.append(
            f"| `{r['task_id']}` | `{r.get('case_type')}` | `{r.get('annotated_action')}` | "
            f"`{r.get('g0_action')}` | {r.get('exact_match')} | {r.get('soft_match')} | "
            f"{r.get('g0_reason') or r.get('error') or ''} |"
        )
    lines.extend(
        [
            "",
            "## 怎么读",
            "",
            "- **exact 高**：规则与人工标注对齐好",
            "- **soft 高、exact 低**：动作族对，具体枚举可再调",
            "- **都低**：触发映射或规则先验需改（看 DIFF 行的 proposed_code_preview）",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--annotations",
        default=str(
            Path(r"C:\Users\86137\OneDrive\Desktop\branch_case_type_runs\annotations\selected_checkpoints_part3.json")
        ),
    )
    p.add_argument(
        "--split-root",
        default=str(Path(r"C:\Users\86137\OneDrive\Desktop\tasks_split_3\tasks")),
    )
    p.add_argument(
        "--out",
        default=str(
            Path(r"C:\Users\86137\OneDrive\Desktop\branch_case_type_runs\results\policy_g0_offline_replay.json")
        ),
    )
    return p.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
