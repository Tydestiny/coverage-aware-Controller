"""Branch online compare: CONTINUE vs PolicyG0 advice at annotated checkpoints.

Does NOT require live triggers. Uses restore→inject→continue→evaluate.
Convenient for Controller-side validation before Trigger integration.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_policy_g0_offline_replay import (  # noqa: E402
    compact_history,
    extract_errors,
    extract_task,
    load_relevant_proposed,
    case_type_to_trigger,
)
from controller_agent.controller.policy_g0 import PolicyG0Backend  # noqa: E402
from controller_agent.state.schemas import Action, ControllerInput  # noqa: E402

RESTORE = REPO_ROOT / "scripts" / "run_checkpoint_restore_continue.py"
DEFAULT_SPLIT = Path(r"C:\Users\86137\OneDrive\Desktop\tasks_split_3\tasks")
DEFAULT_APPWORLD = Path(r"C:\Users\86137\OneDrive\Desktop\controller\appworld")
DEFAULT_ANNS = Path(
    r"C:\Users\86137\OneDrive\Desktop\branch_case_type_runs\annotations\selected_checkpoints_part3.json"
)
DEFAULT_OUT = Path(
    r"C:\Users\86137\OneDrive\Desktop\branch_case_type_runs\results\policy_g0_branch_compare.json"
)


def main() -> int:
    args = parse_args()
    anns = json.loads(Path(args.annotations).read_text(encoding="utf-8"))
    if args.tasks:
        want = set(args.tasks)
        anns = [a for a in anns if a["task_id"] in want]
    if args.limit and args.limit > 0:
        anns = anns[: args.limit]

    backend = PolicyG0Backend()
    rows: list[dict[str, Any]] = []
    for ann in anns:
        plan = plan_g0(ann, Path(args.split_root), backend)
        print(
            f"\n## {ann['task_id']} @ {ann['checkpoint_id']} | "
            f"G0={plan.get('g0_action')} | {plan.get('g0_reason')}"
        )
        if plan.get("error"):
            rows.append(plan)
            print(f"SKIP: {plan['error']}")
            continue

        cont = run_branch(
            ann=ann,
            strategy="CONTINUE",
            guidance=None,
            split_root=Path(args.split_root),
            appworld_root=Path(args.appworld_root),
            max_turns=args.max_turns,
            max_execs=args.max_execs,
            max_cost=args.max_cost_cny,
        )
        g0 = run_branch(
            ann=ann,
            strategy=str(plan["g0_action"]),
            guidance=plan.get("g0_instruction") or None,
            split_root=Path(args.split_root),
            appworld_root=Path(args.appworld_root),
            max_turns=args.max_turns,
            max_execs=args.max_execs,
            max_cost=args.max_cost_cny,
        )
        row = {
            **{k: plan[k] for k in ("task_id", "checkpoint_id", "case_type", "g0_action", "g0_reason", "g0_instruction")},
            "continue": cont,
            "g0": g0,
            "delta_pass_rate": None,
            "label": "unknown",
        }
        if cont.get("pass_rate") is not None and g0.get("pass_rate") is not None:
            delta = g0["pass_rate"] - cont["pass_rate"]
            row["delta_pass_rate"] = delta
            if g0.get("success") and not cont.get("success"):
                row["label"] = "positive"
            elif delta > 1e-9:
                row["label"] = "positive"
            elif delta < -1e-9:
                row["label"] = "negative"
            else:
                row["label"] = "neutral"
        rows.append(row)
        print(
            f"CONTINUE pass={cont.get('num_passed')}/{cont.get('num_total')} "
            f"success={cont.get('success')} | "
            f"G0 pass={g0.get('num_passed')}/{g0.get('num_total')} "
            f"success={g0.get('success')} | label={row['label']}"
        )

    payload = {"n": len(rows), "rows": rows}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_md(payload, out.with_suffix(".md"))
    print(f"\nWrote {out}")
    print(f"Wrote {out.with_suffix('.md')}")
    return 0


def plan_g0(ann: dict[str, Any], split_root: Path, backend: PolicyG0Backend) -> dict[str, Any]:
    task_id = ann["task_id"]
    checkpoint_id = ann["checkpoint_id"]
    case_type = ann.get("case_type", "")
    task_dir = split_root / task_id
    ck_path = task_dir / "checkpoints" / "post_observation" / f"{checkpoint_id}.json"
    events_path = task_dir / "collection" / "turn_events.jsonl"
    base = {
        "task_id": task_id,
        "checkpoint_id": checkpoint_id,
        "case_type": case_type,
    }
    if not ck_path.exists() or not events_path.exists():
        return base | {"error": "missing checkpoint or turn_events"}

    checkpoint = json.loads(ck_path.read_text(encoding="utf-8"))
    turn_n = int(checkpoint.get("agent_turn_index") or 0)
    proposed, _ = load_relevant_proposed(events_path, turn_n + 1, case_type)
    task_text = extract_task(checkpoint.get("agent_messages") or [])
    history = compact_history(checkpoint.get("agent_messages") or [])
    errors = extract_errors(checkpoint.get("last_execution_outputs") or [], history)
    completed = list(checkpoint.get("executed_code_prefix") or [])[-6:]
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
    d = backend.decide(cin)
    return base | {
        "g0_action": d.action.value,
        "g0_reason": d.reason,
        "g0_instruction": d.instruction,
        "trigger_reason": trigger,
        "error": None,
    }


def run_branch(
    *,
    ann: dict[str, Any],
    strategy: str,
    guidance: str | None,
    split_root: Path,
    appworld_root: Path,
    max_turns: int,
    max_execs: int,
    max_cost: float,
) -> dict[str, Any]:
    task_id = ann["task_id"]
    checkpoint_id = ann["checkpoint_id"]
    branch = strategy.lower()
    exp = f"policy_g0_branch_{branch}"
    out = (
        appworld_root
        / "experiments"
        / "outputs"
        / exp
        / "tasks"
        / task_id
        / "restore_continue"
        / f"{checkpoint_id}__{branch}"
    )
    cmd = [
        sys.executable,
        str(RESTORE),
        "--execute",
        "--source-task-dir",
        str(split_root / task_id),
        "--checkpoint-id",
        checkpoint_id,
        "--experiment-name",
        exp,
        "--appworld-root",
        str(appworld_root),
        "--output-dir",
        str(out),
        "--max-agent-turns-after-restore",
        str(max_turns),
        "--max-environment-executions-after-restore",
        str(max_execs),
        "--max-cost-cny",
        str(max_cost),
        "--strategy-name",
        strategy,
    ]
    if guidance and strategy != "CONTINUE" and strategy != Action.PROCEED.value:
        cmd.extend(["--guidance", guidance])

    print(f"  -> run {strategy}")
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT))
    return summarize(out, proc.returncode)


def summarize(output_dir: Path, exit_code: int) -> dict[str, Any]:
    eval_path = output_dir / "evaluator_result.json"
    success = num_passed = num_failed = num_total = pass_rate = None
    if eval_path.exists():
        payload = json.loads(eval_path.read_text(encoding="utf-8"))
        result = payload.get("result") if isinstance(payload, dict) else None
        if isinstance(result, dict):
            success = bool(result.get("success")) if "success" in result else None
            num_passed = len(result.get("passes") or [])
            num_failed = len(result.get("failures") or [])
            num_total = result.get("num_tests")
            if isinstance(num_total, int) and num_total > 0:
                pass_rate = num_passed / num_total
    return {
        "success": success,
        "num_passed": num_passed,
        "num_failed": num_failed,
        "num_total": num_total,
        "pass_rate": pass_rate,
        "exit_code": exit_code,
        "output_dir": str(output_dir),
    }


def write_md(payload: dict[str, Any], path: Path) -> None:
    lines = [
        "# PolicyG0 分支 online 对照（CONTINUE vs G0）",
        "",
        "不依赖 Trigger：restore → 注入 G0 Advice → evaluate。",
        "",
        "| task | case_type | G0 action | CONTINUE | G0 | Δpass | label |",
        "|------|-----------|-----------|----------|----|-------|-------|",
    ]
    for r in payload["rows"]:
        c = r.get("continue") or {}
        g = r.get("g0") or {}
        ct = (
            f"{c.get('num_passed')}/{c.get('num_total')}"
            if c.get("num_passed") is not None
            else "-"
        )
        gt = (
            f"{g.get('num_passed')}/{g.get('num_total')}"
            if g.get("num_passed") is not None
            else "-"
        )
        d = r.get("delta_pass_rate")
        ds = "-" if d is None else f"{d:+.2f}"
        lines.append(
            f"| `{r.get('task_id')}` | `{r.get('case_type')}` | `{r.get('g0_action')}` | "
            f"{ct} | {gt} | {ds} | {r.get('label')} |"
        )
    lines.extend(
        [
            "",
            "## 读法",
            "",
            "- `positive`：G0 注入后优于 CONTINUE（Controller 有增益）",
            "- `negative`：有害，需收紧该类规则",
            "- `neutral`：无差，可降干预或改 instruction",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--annotations", default=str(DEFAULT_ANNS))
    p.add_argument("--split-root", default=str(DEFAULT_SPLIT))
    p.add_argument("--appworld-root", default=str(DEFAULT_APPWORLD))
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--tasks", nargs="*", default=None, help="Optional task_id filter")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--max-turns", type=int, default=12)
    p.add_argument("--max-execs", type=int, default=12)
    p.add_argument("--max-cost-cny", type=float, default=1.0)
    return p.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
