"""Run repeated no-guidance and one-shot-guidance checkpoint continuations."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from controller_agent.evaluation.official_evaluator import run_official_evaluation
from scripts.collect_oracle_instruction_branches import _normalize_evaluator
from scripts.run_single_intervention_branches import (
    _materialize_prefix_checkpoint,
    _read_jsonl,
    _write_json,
)


CONCISE_GUIDANCE = (
    'Keep only threads that are both starred and have any priority label.\n'
    'Do not restrict the priority label to "priority-3".\n'
    'Update the existing filter accordingly.'
)


def parse_args() -> argparse.Namespace:
    """Parse a bounded repeated checkpoint-guidance experiment."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--checkpoint-id", required=True)
    parser.add_argument("--current-guidance-source", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--appworld-root", default=str(REPO_ROOT.parent / "appworld"))
    parser.add_argument("--agent-model", default="qwen3-coder-flash-2025-07-28")
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def main() -> int:
    """Run every group and write an experiment-level manifest."""
    args = parse_args()
    if args.execute and importlib.util.find_spec("munch") is None:
        appworld_python = Path(args.appworld_root).resolve() / ".venv" / "Scripts" / "python.exe"
        if not appworld_python.is_file():
            raise RuntimeError(f"AppWorld Python runtime not found: {appworld_python}")
        return subprocess.run(
            [str(appworld_python), str(Path(__file__).resolve()), *sys.argv[1:]],
            cwd=REPO_ROOT,
            check=False,
        ).returncode
    if args.repetitions < 1:
        raise ValueError("--repetitions must be at least 1")
    candidate = _find_candidate(Path(args.candidates), args.task_id, args.checkpoint_id)
    current_guidance, current_action = _read_current_guidance(Path(args.current_guidance_source))
    root = Path(args.output_root).resolve() / args.experiment_id
    groups = {
        "no_guidance": (None, None),
        "current_guidance": (current_guidance, current_action),
        "concise_guidance": (CONCISE_GUIDANCE, "REPLAN"),
    }
    manifest = {
        "experiment_id": args.experiment_id,
        "task_id": args.task_id,
        "checkpoint_id": args.checkpoint_id,
        "agent_model": args.agent_model,
        "seed": args.seed,
        "repetitions": args.repetitions,
        "groups": list(groups),
        "current_guidance_source": str(Path(args.current_guidance_source).resolve()),
        "execute": args.execute,
    }
    _write_json(root / "manifest.json", manifest)
    if not args.execute:
        return 0

    outcomes: list[dict[str, Any]] = []
    for group, (guidance, action) in groups.items():
        for run_number in range(1, args.repetitions + 1):
            outcomes.append(_run_one(args, candidate, root, group, run_number, guidance, action))
    _write_json(root / "summary.json", {"runs": outcomes})
    return 0 if all(row["status"] == "completed" for row in outcomes) else 1


def _find_candidate(path: Path, task_id: str, checkpoint_id: str) -> dict[str, Any]:
    """Find exactly one requested checkpoint candidate."""
    matches = [
        row for row in _read_jsonl(path)
        if row.get("task_id") == task_id and row.get("checkpoint_id") == checkpoint_id
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one candidate for {task_id}/{checkpoint_id}; found {len(matches)}")
    return matches[0]


def _read_current_guidance(path: Path) -> tuple[str, str | None]:
    """Read the one previously generated Controller guidance record."""
    rows = _read_jsonl(path)
    if len(rows) != 1:
        raise ValueError(f"Expected exactly one Controller call in {path}; found {len(rows)}")
    guidance = rows[0].get("guidance")
    if not isinstance(guidance, str) or not guidance.strip():
        raise ValueError(f"Controller call has no guidance: {path}")
    action = rows[0].get("action")
    return guidance, action if isinstance(action, str) else None


def _run_one(
    args: argparse.Namespace,
    candidate: dict[str, Any],
    root: Path,
    group: str,
    run_number: int,
    guidance: str | None,
    action: str | None,
) -> dict[str, Any]:
    """Materialize one independent checkpoint and execute one continuation."""
    destination = root / "tasks" / args.task_id / args.checkpoint_id / group / f"run_{run_number:02d}"
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing run: {destination}")
    destination.mkdir(parents=True)
    stage_args = SimpleNamespace(
        appworld_root=args.appworld_root,
        seed=args.seed,
        appworld_experiment_prefix=f"{args.experiment_id}_{group}_{run_number:02d}",
    )
    stage = None
    try:
        stage = _materialize_prefix_checkpoint(stage_args, candidate)
        command = _runner_command(args, candidate, stage, destination, group, run_number, guidance, action)
        completed = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True, check=False)
        official = run_official_evaluation(Path(args.appworld_root).resolve(), stage.branch_experiment_name, args.task_id)
        evaluator = _normalize_evaluator(args.task_id, official)
        _write_json(destination / "evaluator_result.json", evaluator)
        result = {
            "task_id": args.task_id,
            "checkpoint_id": args.checkpoint_id,
            "group": group,
            "run_number": run_number,
            "restore_success": _read_json(destination / "restore_result.json").get("success"),
            "termination_reason": _final_termination(destination / "continuation.jsonl"),
            "evaluator": evaluator,
            "status": "completed" if completed.returncode == 0 else "error",
            "subprocess_returncode": completed.returncode,
            "subprocess_stderr": completed.stderr[-4000:],
        }
        _write_json(destination / "result.json", result)
        return {"group": group, "run_number": run_number, "status": result["status"], "path": str(destination)}
    except Exception as exc:
        _write_json(destination / "result.json", {"task_id": args.task_id, "checkpoint_id": args.checkpoint_id, "group": group, "run_number": run_number, "status": "error", "error": f"{type(exc).__name__}: {exc}"})
        return {"group": group, "run_number": run_number, "status": "error", "path": str(destination), "error": f"{type(exc).__name__}: {exc}"}
    finally:
        if stage is not None and stage.stage_task_dir.exists():
            shutil.rmtree(stage.stage_task_dir.parent)


def _runner_command(args: argparse.Namespace, candidate: dict[str, Any], stage: Any, destination: Path, group: str, run_number: int, guidance: str | None, action: str | None) -> list[str]:
    """Build a no-Controller continuation command using the existing Runner."""
    python = Path(args.appworld_root).resolve() / ".venv" / "Scripts" / "python.exe"
    command = [
        str(python), str(REPO_ROOT / "scripts" / "run_live_controller_checkpoint_smoke.py"), "--execute",
        "--task-id", args.task_id, "--agent-model", args.agent_model,
        "--checkpoint-id", args.checkpoint_id, "--source-task-dir", str(stage.source_task_dir),
        "--appworld-root", str(Path(args.appworld_root).resolve()), "--output-dir", str(destination),
        "--experiment-name", f"{args.experiment_id}__{args.task_id}__{args.checkpoint_id}__{group}__{run_number:02d}",
        "--branch", "baseline", "--disable-controller", "--skip-inprocess-evaluate", "--log-lm-calls",
        "--max-agent-turns", "10000", "--max-environment-executions", "10000",
        "--max-controller-calls", "10000", "--max-controller-blocks", "10000",
        "--max-extra-agent-turns-after-block", "10000", "--max-cost-cny", "inf",
        "--seed", str(args.seed),
    ]
    if guidance is not None:
        command.extend(["--external-action", action or "REPLAN", "--external-guidance", guidance])
    return command


def _read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object or return an empty object."""
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _final_termination(path: Path) -> str | None:
    """Return the continuation's final recorded termination reason."""
    rows = _read_jsonl(path)
    return rows[-1].get("termination_reason") if rows else None


if __name__ == "__main__":
    raise SystemExit(main())
