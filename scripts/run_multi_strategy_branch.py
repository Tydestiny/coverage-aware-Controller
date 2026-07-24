"""Single-checkpoint multi-strategy branch experiment.

Restore one post_observation checkpoint, then continue under different
manually defined control strategies. Compare official evaluator outcomes
to decide which action/guidance templates to bake into the Controller.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
RESTORE_SCRIPT = REPO_ROOT / "scripts" / "run_checkpoint_restore_continue.py"
DEFAULT_SPLIT = Path(r"C:\Users\86137\OneDrive\Desktop\tasks_split_3\tasks")
DEFAULT_APPWORLD = Path(r"C:\Users\86137\OneDrive\Desktop\controller\appworld")
DEFAULT_STRATEGIES = REPO_ROOT / "scripts" / "strategy_packs" / "229360a_2_write_before.json"


def main() -> int:
    args = parse_args()
    pack = json.loads(Path(args.strategies).read_text(encoding="utf-8"))
    task_id = pack["task_id"]
    checkpoint_id = pack["checkpoint_id"]
    case_type = pack.get("case_type", "")
    strategies: list[dict[str, Any]] = pack["strategies"]
    source = Path(args.split_root) / task_id
    if not (source / "checkpoints" / "manifest.json").exists():
        raise SystemExit(f"Missing source checkpoints: {source}")

    rows: list[dict[str, Any]] = []
    for item in strategies:
        name = str(item["name"])
        guidance = item.get("guidance")
        branch = name.lower()
        exp = f"strategy_branch_{branch}"
        out = (
            Path(args.appworld_root)
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
            str(RESTORE_SCRIPT),
            "--execute",
            "--source-task-dir",
            str(source),
            "--checkpoint-id",
            checkpoint_id,
            "--experiment-name",
            exp,
            "--appworld-root",
            str(args.appworld_root),
            "--output-dir",
            str(out),
            "--max-agent-turns-after-restore",
            str(args.max_turns),
            "--max-environment-executions-after-restore",
            str(args.max_execs),
            "--max-cost-cny",
            str(args.max_cost_cny),
            "--strategy-name",
            name,
        ]
        if guidance:
            cmd.extend(["--guidance", str(guidance)])

        print(f"\n===== {task_id} | {checkpoint_id} | {name} =====")
        proc = subprocess.run(cmd, cwd=str(REPO_ROOT))
        row = summarize_run(
            task_id=task_id,
            checkpoint_id=checkpoint_id,
            case_type=case_type,
            strategy=name,
            guidance=guidance,
            output_dir=out,
            exit_code=proc.returncode,
        )
        rows.append(row)
        print(
            f"DONE {name}: success={row.get('success')} "
            f"pass={row.get('num_passed')}/{row.get('num_total')} exit={proc.returncode}"
        )

    baseline = next((r for r in rows if r["strategy"] == "CONTINUE"), None)
    base_rate = baseline.get("pass_rate") if baseline else None
    for r in rows:
        if base_rate is None or r.get("pass_rate") is None:
            r["delta_pass_rate"] = None
            r["label"] = "unknown"
        else:
            delta = r["pass_rate"] - base_rate
            r["delta_pass_rate"] = delta
            if r.get("success") is True and not (baseline or {}).get("success"):
                r["label"] = "positive"
            elif delta > 1e-9:
                r["label"] = "positive"
            elif delta < -1e-9:
                r["label"] = "negative"
            else:
                r["label"] = "neutral"

    out_json = Path(args.out)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "task_id": task_id,
        "checkpoint_id": checkpoint_id,
        "case_type": case_type,
        "strategies_file": str(Path(args.strategies).resolve()),
        "rows": rows,
    }
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_md(payload, out_json.with_suffix(".md"))
    print(f"Wrote {out_json}")
    print(f"Wrote {out_json.with_suffix('.md')}")
    return 0


def summarize_run(
    *,
    task_id: str,
    checkpoint_id: str,
    case_type: str,
    strategy: str,
    guidance: str | None,
    output_dir: Path,
    exit_code: int,
) -> dict[str, Any]:
    eval_path = output_dir / "evaluator_result.json"
    report_path = output_dir / "restore_continue_report.md"
    success = None
    category = None
    num_passed = num_failed = num_total = None
    pass_rate = None
    if eval_path.exists():
        payload = json.loads(eval_path.read_text(encoding="utf-8"))
        result = payload.get("result") if isinstance(payload, dict) else None
        if isinstance(result, dict):
            if "success" in result:
                success = bool(result["success"])
            num_passed = len(result.get("passes") or [])
            num_failed = len(result.get("failures") or [])
            num_total = result.get("num_tests")
            if isinstance(num_total, int) and num_total > 0:
                pass_rate = num_passed / num_total
    if report_path.exists():
        text = report_path.read_text(encoding="utf-8", errors="ignore")
        for line in text.splitlines():
            if line.startswith("- Category:") and "`" in line:
                category = line.split("`")[1]
                break
    return {
        "task_id": task_id,
        "checkpoint_id": checkpoint_id,
        "case_type": case_type,
        "strategy": strategy,
        "guidance": guidance,
        "success": success,
        "num_passed": num_passed,
        "num_failed": num_failed,
        "num_total": num_total,
        "pass_rate": pass_rate,
        "category": category,
        "exit_code": exit_code,
        "output_dir": str(output_dir),
    }


def write_md(payload: dict[str, Any], path: Path) -> None:
    rows = payload["rows"]
    lines = [
        "# 单 Checkpoint 多策略分支实验",
        "",
        f"- task: `{payload['task_id']}`",
        f"- checkpoint: `{payload['checkpoint_id']}`",
        f"- case_type: `{payload.get('case_type')}`",
        f"- strategies file: `{payload.get('strategies_file')}`",
        "",
        "| strategy | success | passed/total | pass_rate | Δpass vs CONTINUE | label |",
        "|----------|---------|--------------|-----------|-------------------|-------|",
    ]
    for r in rows:
        pr = "-" if r.get("pass_rate") is None else f"{r['pass_rate']:.2f}"
        delta = r.get("delta_pass_rate")
        ds = "-" if delta is None else f"{delta:+.2f}"
        pt = (
            f"{r.get('num_passed')}/{r.get('num_total')}"
            if r.get("num_passed") is not None and r.get("num_total") is not None
            else "-"
        )
        lines.append(
            f"| `{r['strategy']}` | {r.get('success')} | {pt} | {pr} | {ds} | {r.get('label')} |"
        )
    lines.extend(
        [
            "",
            "## 怎么用于 Controller",
            "",
            "- `positive`：该动作+guidance 优于 CONTINUE，可写成规则/Prompt 模板",
            "- `negative`：该动作有害，避免在同类 checkpoint 默认使用",
            "- `neutral`：无差异，可降成本（continue）或改写 guidance 再试",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--strategies", default=str(DEFAULT_STRATEGIES))
    p.add_argument("--split-root", default=str(DEFAULT_SPLIT))
    p.add_argument("--appworld-root", default=str(DEFAULT_APPWORLD))
    p.add_argument("--max-turns", type=int, default=12)
    p.add_argument("--max-execs", type=int, default=12)
    p.add_argument("--max-cost-cny", type=float, default=1.0)
    p.add_argument(
        "--out",
        default=str(
            Path(r"C:\Users\86137\OneDrive\Desktop\branch_case_type_runs")
            / "results"
            / "scratch"
            / "branch_multi_strategy.json"
        ),
    )
    return p.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
