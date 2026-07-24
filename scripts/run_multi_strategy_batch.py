"""Run multiple strategy packs and aggregate case_type conclusions."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
BRANCH_SCRIPT = REPO_ROOT / "scripts" / "run_multi_strategy_branch.py"
DEFAULT_OUT_DIR = Path(r"C:\Users\86137\OneDrive\Desktop\branch_case_type_runs\results")


def main() -> int:
    args = parse_args()
    packs = [Path(p) for p in args.packs]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    result_paths: list[Path] = []
    for pack in packs:
        stem = pack.stem
        out_json = out_dir / f"branch_multi_strategy_{stem}.json"
        cmd = [
            sys.executable,
            str(BRANCH_SCRIPT),
            "--strategies",
            str(pack),
            "--split-root",
            args.split_root,
            "--appworld-root",
            args.appworld_root,
            "--max-turns",
            str(args.max_turns),
            "--max-execs",
            str(args.max_execs),
            "--max-cost-cny",
            str(args.max_cost_cny),
            "--out",
            str(out_json),
        ]
        print(f"\n######## PACK {pack.name} -> {out_json.name} ########")
        proc = subprocess.run(cmd, cwd=str(REPO_ROOT))
        if proc.returncode != 0:
            print(f"WARN: pack {pack.name} exited {proc.returncode}")
        if out_json.exists():
            result_paths.append(out_json)

    for extra in args.include_json:
        p = Path(extra)
        if p.exists():
            result_paths.append(p)

    summary = aggregate(result_paths)
    summary_path = out_dir / "case_type_strategy_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path = out_dir / "case_type_strategy_summary.md"
    write_summary_md(summary, md_path)
    print(f"\nWrote {summary_path}")
    print(f"Wrote {md_path}")
    return 0


def aggregate(result_paths: list[Path]) -> dict[str, Any]:
    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    tasks: list[dict[str, Any]] = []
    for path in result_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        case_type = payload.get("case_type") or "unknown"
        tasks.append(
            {
                "source": str(path),
                "task_id": payload.get("task_id"),
                "checkpoint_id": payload.get("checkpoint_id"),
                "case_type": case_type,
                "rows": payload.get("rows") or [],
            }
        )
        for row in payload.get("rows") or []:
            if row.get("strategy") == "CONTINUE":
                continue
            by_case[case_type].append(
                {
                    "task_id": payload.get("task_id"),
                    "strategy": row.get("strategy"),
                    "label": row.get("label"),
                    "delta_pass_rate": row.get("delta_pass_rate"),
                    "success": row.get("success"),
                    "pass_rate": row.get("pass_rate"),
                }
            )

    case_stats: dict[str, Any] = {}
    for case_type, rows in by_case.items():
        strat_stats: dict[str, dict[str, Any]] = {}
        for r in rows:
            s = r["strategy"]
            st = strat_stats.setdefault(
                s,
                {
                    "n": 0,
                    "positive": 0,
                    "negative": 0,
                    "neutral": 0,
                    "unknown": 0,
                    "delta_sum": 0.0,
                    "delta_n": 0,
                    "success_flips": 0,
                },
            )
            st["n"] += 1
            label = r.get("label") or "unknown"
            st[label] = st.get(label, 0) + 1
            if r.get("delta_pass_rate") is not None:
                st["delta_sum"] += float(r["delta_pass_rate"])
                st["delta_n"] += 1
            if r.get("success") is True:
                st["success_flips"] += 1
        ranked = []
        for s, st in strat_stats.items():
            mean_delta = (st["delta_sum"] / st["delta_n"]) if st["delta_n"] else None
            ranked.append(
                {
                    "strategy": s,
                    "n_tasks": st["n"],
                    "positive": st["positive"],
                    "negative": st["negative"],
                    "neutral": st["neutral"],
                    "mean_delta_pass": mean_delta,
                    "success_true_count": st["success_flips"],
                    "recommendation": recommend(st, mean_delta),
                }
            )
        ranked.sort(
            key=lambda x: (x["positive"] - x["negative"], x["mean_delta_pass"] or -99),
            reverse=True,
        )
        case_stats[case_type] = ranked

    return {"tasks": tasks, "by_case_type": case_stats}


def recommend(st: dict[str, Any], mean_delta: float | None) -> str:
    n = st["n"] or 1
    if st["positive"] >= max(2, (n + 1) // 2) and st["negative"] == 0:
        return "default_candidate"
    if st["negative"] > st["positive"]:
        return "avoid_default"
    if mean_delta is not None and mean_delta > 0:
        return "optional"
    return "inconclusive"


def write_summary_md(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# Case-type 多策略汇总（跨任务）",
        "",
        "## 任务覆盖",
        "",
    ]
    for t in summary.get("tasks") or []:
        lines.append(
            f"- `{t.get('case_type')}` / `{t.get('task_id')}` @ `{t.get('checkpoint_id')}`"
        )
    lines.extend(["", "## 按 case_type 的策略排名", ""])
    for case_type, ranked in (summary.get("by_case_type") or {}).items():
        lines.extend(
            [
                f"### `{case_type}`",
                "",
                "| strategy | n | + | - | ~ | mean Δpass | success=True | recommendation |",
                "|----------|---|---|---|---|------------|--------------|----------------|",
            ]
        )
        for r in ranked:
            md = "-" if r.get("mean_delta_pass") is None else f"{r['mean_delta_pass']:+.2f}"
            lines.append(
                f"| `{r['strategy']}` | {r['n_tasks']} | {r['positive']} | {r['negative']} | "
                f"{r['neutral']} | {md} | {r['success_true_count']} | {r['recommendation']} |"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--packs", nargs="+", required=True)
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    p.add_argument("--include-json", nargs="*", default=[])
    p.add_argument(
        "--split-root",
        default=str(Path(r"C:\Users\86137\OneDrive\Desktop\tasks_split_3\tasks")),
    )
    p.add_argument(
        "--appworld-root",
        default=str(Path(r"C:\Users\86137\OneDrive\Desktop\controller\appworld")),
    )
    p.add_argument("--max-turns", type=int, default=12)
    p.add_argument("--max-execs", type=int, default=12)
    p.add_argument("--max-cost-cny", type=float, default=1.0)
    return p.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
