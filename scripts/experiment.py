"""Small, guarded CMD-friendly interface for train/dev experiments."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from controller_agent.paths import appworld_python, appworld_root, artifact_root, official_task_ids

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def task_ids(split: str, root: Path) -> set[str]:
    """Read official task IDs without loading hidden task data."""
    return official_task_ids(root, split)


def run_path(experiment_id: str, task_id: str, branch: str, root: Path) -> Path:
    """Return a deterministic per-task run directory."""
    if not all(_IDENTIFIER.fullmatch(value) for value in (experiment_id, task_id, branch)):
        raise ValueError("experiment_id, task_id, and branch must use letters, digits, _, ., or -")
    return root / experiment_id / "tasks" / task_id / branch


def parser() -> argparse.ArgumentParser:
    """Build the guarded command line parser."""
    cli = argparse.ArgumentParser(description=__doc__)
    sub = cli.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor")
    prepare = sub.add_parser("prepare", help="Fix train/dev metadata-only selection without model calls")
    prepare.add_argument("--output", type=Path, default=ROOT / "configs" / "studies" / "train_dev_v1.json")
    prepare.add_argument("--seed", type=int, default=20260722)
    listing = sub.add_parser("list")
    listing.add_argument("--archive-root", type=Path)
    review = sub.add_parser("review", help="Review trajectories locally and save per-run annotations")
    review.add_argument("--experiment-id")
    review.add_argument("--task-id")
    review.add_argument("--branch")
    review.add_argument("--output", type=Path)
    review.add_argument("--serve", action="store_true", help="Browse all active experiments locally")
    review.add_argument("--port", type=int, default=8765)
    run = sub.add_parser("run")
    run.add_argument("scenario", nargs="?", choices=("initial", "checkpoint"), default="initial")
    run.add_argument("--split", choices=("train", "dev"), required=True)
    run.add_argument("--task-id", required=True)
    run.add_argument("--experiment-id", required=True)
    run.add_argument("--branch", choices=("baseline", "controller"))
    run.add_argument("--checkpoint-id")
    run.add_argument("--source-task-dir", type=Path)
    run.add_argument("--prompt", type=Path)
    run.add_argument("--agent-model", required=True)
    run.add_argument("--controller-model")
    run.add_argument("--controller-base-url", default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    run.add_argument("--controller-api-key-env", default="DASHSCOPE_API_KEY")
    run.add_argument("--seed", type=int, default=20260722)
    run.add_argument("--max-controller-calls", type=int, default=4)
    run.add_argument("--max-controller-blocks", type=int, default=4)
    run.add_argument("--trigger-profile", choices=("before_finish", "all"), default="all")
    run.add_argument("--execute", action="store_true")
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--experiment-id", required=True)
    evaluate.add_argument("--task-id", required=True)
    evaluate.add_argument("--branch", choices=("baseline", "controller"), required=True)
    evaluate.add_argument("--execute", action="store_true")
    analyze = sub.add_parser("analyze")
    analyze.add_argument("--experiment-id", required=True)
    return cli


def main(argv: list[str] | None = None) -> int:
    """Run only explicitly authorized model calls; other commands stay offline."""
    args = parser().parse_args(argv)
    appworld = appworld_root()
    artifacts = artifact_root()
    if args.command == "prepare":
        from controller_agent.development_study import prepare_study, save_study
        study = prepare_study(appworld, args.seed)
        save_study(args.output, study)
        print(json.dumps({"manifest": str(args.output), "train": [r["task_id"] for r in study["train"]],
                          "reserved_dev": [r["task_id"] for r in study["reserved_dev"]],
                          "status": study["status"]}, indent=2))
        return 0
    if args.command == "doctor":
        agent_python = appworld_python(appworld)
        import_check = subprocess.run(
            [str(agent_python), "-c", "import appworld"], cwd=appworld,
            capture_output=True, text=True, check=False,
        ) if appworld.is_dir() else None
        checks = {
            "python_3_11_plus": sys.version_info >= (3, 11),
            "appworld_checkout": (appworld / "pyproject.toml").is_file(),
            "train_ids": (appworld / "data" / "datasets" / "train.txt").is_file(),
            "dev_ids": (appworld / "data" / "datasets" / "dev.txt").is_file(),
            "agent_python": str(agent_python),
            "appworld_importable": import_check is not None and import_check.returncode == 0,
            "dashscope_key_present": bool(os.environ.get("DASHSCOPE_API_KEY")),
            "openrouter_key_present": bool(os.environ.get("OPENROUTER_API_KEY")),
            "appworld_env_file_present": (appworld / "experiments" / ".env").is_file(),
            "appworld_root": str(appworld),
            "artifact_root": str(artifacts),
        }
        print(json.dumps(checks, indent=2))
        return 0 if all(checks[key] for key in ("python_3_11_plus", "appworld_checkout", "train_ids", "dev_ids", "appworld_importable")) else 1
    if args.command == "list":
        runs = sorted(str(path.relative_to(artifacts)) for path in artifacts.glob("*/tasks/**/manifest.json"))
        historical = []
        if args.archive_root:
            for summary_file in sorted((args.archive_root / "controller_results" / "end_to_end").glob("*/analysis/summary.json")):
                summary = json.loads(summary_file.read_text(encoding="utf-8"))
                historical.append({"experiment": summary_file.parents[1].name,
                                   "task_count": summary.get("overall", {}).get("task_count"),
                                   "baseline_success": summary.get("overall", {}).get("baseline_success_count"),
                                   "controller_success": summary.get("overall", {}).get("controller_success_count"),
                                   "path": str(summary_file)})
        print(json.dumps({"active_runs": runs, "historical_summaries": historical}, indent=2))
        return 0
    if args.command == "review":
        if args.serve:
            from controller_agent.results.review_server import serve_reviews

            serve_reviews(artifacts, args.port)
            return 0
        if not all((args.experiment_id, args.task_id, args.branch)):
            raise ValueError("Use review --serve, or provide --experiment-id, --task-id and --branch")
        from controller_agent.results.reviewer import write_review_page

        source = run_path(args.experiment_id, args.task_id, args.branch, artifacts)
        destination = args.output or artifacts / "reviews" / f"{args.experiment_id}__{args.task_id}__{args.branch}.html"
        if destination.resolve().is_relative_to(source.resolve()):
            raise ValueError("Review page must not be written into the source run directory")
        page = write_review_page(source, destination)
        print(json.dumps({"review_page": str(page), "source_run": str(source)}, indent=2))
        return 0
    if args.command == "analyze":
        if not _IDENTIFIER.fullmatch(args.experiment_id):
            raise ValueError("Invalid experiment_id")
        from scripts.analyze_branch_experiment import analyze as analyze_results
        result = analyze_results(artifacts / args.experiment_id, artifacts / args.experiment_id / "analysis")
        print(json.dumps(result["summary"], indent=2))
        return 0
    if args.command == "run":
        if not _IDENTIFIER.fullmatch(args.experiment_id) or not _IDENTIFIER.fullmatch(args.task_id):
            raise ValueError("Invalid experiment or task ID")
        if args.task_id not in task_ids(args.split, appworld):
            raise ValueError(f"{args.task_id} is not in official {args.split} IDs")
        if args.scenario == "initial" and args.branch is None:
            raise ValueError("Initial runs require --branch baseline or controller")
        needs_prompt = args.scenario == "checkpoint" or args.branch == "controller"
        if needs_prompt and (args.prompt is None or not args.prompt.is_file()):
            raise ValueError("Controller runs require an existing --prompt; historical G0/G1/G2 are not defaults")
        if needs_prompt and not args.prompt.resolve().is_relative_to(ROOT / "prompts" / "train_dev"):
            raise ValueError("Development prompts must be stored under prompts/train_dev")
        if args.max_controller_calls < 0 or args.max_controller_blocks < 0:
            raise ValueError("Controller limits must be nonnegative")
        if args.scenario == "checkpoint":
            if args.branch is not None:
                raise ValueError("Checkpoint runs generate both baseline and controller; omit --branch")
            if not args.checkpoint_id or not _IDENTIFIER.fullmatch(args.checkpoint_id) or args.source_task_dir is None:
                raise ValueError("Checkpoint runs require --checkpoint-id and --source-task-dir")
            source = args.source_task_dir.resolve()
            outputs = (appworld / "experiments" / "outputs").resolve()
            if (source.name != args.task_id or source.parent.name != "tasks"
                    or source.parent.parent.parent != outputs):
                raise ValueError("Checkpoint source must be this task's AppWorld output directory")
            source_experiment = source.parent.parent.name
            source_manifests = artifacts.glob(f"*/tasks/{args.task_id}/baseline/manifest.json")
            if not any(
                (manifest_data := json.loads(manifest.read_text(encoding="utf-8"))).get("split") == args.split
                and manifest_data.get("appworld_experiment") == source_experiment
                for manifest in source_manifests
            ):
                raise ValueError("Checkpoint source must match an active train/dev baseline manifest")
            checkpoint_file = source / "checkpoints" / "post_observation" / f"{args.checkpoint_id}.json"
            if not checkpoint_file.is_file():
                raise FileNotFoundError(checkpoint_file)
            checkpoint = json.loads(checkpoint_file.read_text(encoding="utf-8"))
            if checkpoint.get("task_id") != args.task_id or checkpoint.get("checkpoint_id") != args.checkpoint_id:
                raise ValueError("Checkpoint identity does not match task ID and checkpoint ID")
            path = artifacts / args.experiment_id
            command = [sys.executable, str(ROOT / "scripts" / "run_checkpoint_dual_branch_smoke.py"),
                       "--experiment-id", args.experiment_id, "--split", args.split,
                       "--task-id", args.task_id, "--checkpoint-id", args.checkpoint_id,
                       "--source-task-dir", str(source), "--appworld-root", str(appworld),
                       "--controller-system-prompt", str(args.prompt.resolve()),
                       "--agent-model", args.agent_model, "--controller-model", args.controller_model or "deepseek-v4-pro",
                       "--controller-base-url", args.controller_base_url,
                       "--controller-api-key-env", args.controller_api_key_env,
                       "--seed", str(args.seed), "--trigger-profile", args.trigger_profile,
                       "--max-controller-calls", str(args.max_controller_calls),
                       "--max-controller-blocks", str(args.max_controller_blocks)]
            if path.exists():
                raise FileExistsError(f"Experiment directory already exists: {path}")
            print(json.dumps({"mode": "execute" if args.execute else "dry-run", "command": command,
                              "output": str(path)}, indent=2))
            if not args.execute:
                return 0
            return subprocess.run(command + ["--execute"], cwd=ROOT, check=False).returncode
        path = run_path(args.experiment_id, args.task_id, args.branch, artifacts)
        experiment_name = f"{args.experiment_id}__{args.task_id}__{args.branch}"
        command = [str(appworld_python(appworld)), str(ROOT / "scripts" / "run_live_controller_initial_task.py"),
                   "--task-id", args.task_id, "--dataset-name", args.split, "--split", args.split,
                   "--branch", args.branch, "--experiment-name", experiment_name,
                   "--output-dir", str(path), "--appworld-root", str(appworld),
                   "--agent-model", args.agent_model, "--seed", str(args.seed),
                   "--max-controller-calls", str(args.max_controller_calls),
                   "--max-controller-blocks", str(args.max_controller_blocks),
                   "--trigger-profile", args.trigger_profile]
        if args.branch == "baseline":
            command.append("--disable-controller")
        else:
            command += ["--controller-system-prompt", str(args.prompt.resolve())]
            if args.controller_model:
                command += ["--controller-model", args.controller_model]
            command += ["--controller-base-url", args.controller_base_url,
                        "--controller-api-key-env", args.controller_api_key_env]
        if path.exists():
            raise FileExistsError(f"Run directory already exists: {path}")
        print(json.dumps({"mode": "execute" if args.execute else "dry-run", "command": command, "output": str(path)}, indent=2))
        if not args.execute:
            return 0
        return subprocess.run(command + ["--execute"], cwd=ROOT, check=False).returncode
    if args.branch is None:
        raise ValueError("Evaluation requires --branch baseline or controller")
    path = run_path(args.experiment_id, args.task_id, args.branch, artifacts)
    if not path.is_dir():
        raise FileNotFoundError(path)
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    command = [sys.executable, str(ROOT / "scripts" / "run_official_evaluation.py"),
               "--experiment-name", manifest["appworld_experiment"], "--task-id", args.task_id,
               "--appworld-root", str(appworld), "--summary", str(path / "run_summary.json")]
    if not args.execute:
        print(json.dumps({"mode": "dry-run", "command": command}, indent=2))
        return 0
    return subprocess.run(command, cwd=ROOT, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
