"""Deterministic train/dev selection from a narrow official metadata allowlist."""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from controller_agent.paths import git_commit, official_task_ids


def metadata_pool(root: Path, split: str) -> list[dict[str, Any]]:
    """Read IDs, difficulty and app metadata, never solutions or evaluations."""
    records = []
    for task_id in sorted(official_task_ids(root, split)):
        directory = root / "data" / "tasks" / task_id / "ground_truth"
        metadata_bytes = (directory / "metadata.json").read_bytes()
        apps_bytes = (directory / "required_apps.json").read_bytes()
        metadata = json.loads(metadata_bytes)
        apps = json.loads(apps_bytes)
        if (type(metadata.get("difficulty")) is not int
                or type(metadata.get("num_apps")) is not int
                or not isinstance(apps, list)
                or not all(isinstance(app, str) for app in apps)):
            raise ValueError(f"Invalid selection metadata: {task_id}")
        records.append({
            "task_id": task_id, "split": split,
            "family_id": task_id.rsplit("_", 1)[0],
            "difficulty": metadata["difficulty"], "num_apps": metadata["num_apps"],
            "apps": sorted(set(apps)),
            "metadata_sha256": hashlib.sha256(metadata_bytes).hexdigest(),
            "required_apps_sha256": hashlib.sha256(apps_bytes).hexdigest(),
        })
    return records


def select_tasks(
    records: list[dict[str, Any]], count: int, seed: int, excluded_families: set[str]
) -> list[dict[str, Any]]:
    """Balance difficulty and app-count strata with seeded ties and unique families."""
    if count < 1:
        raise ValueError("Task count must be positive")
    pool = sorted(records, key=lambda row: row["task_id"])
    random.Random(seed).shuffle(pool)
    difficulties: Counter[int] = Counter()
    app_counts: Counter[str] = Counter()
    apps: set[str] = set()
    families = set(excluded_families)
    selected = []
    for _ in range(count):
        candidates = [row for row in pool if row["family_id"] not in families]
        if not candidates:
            raise ValueError("Not enough distinct task families for the requested selection")
        row = min(candidates, key=lambda row: (
            difficulties[row["difficulty"]],
            app_counts["single" if row["num_apps"] == 1 else "multiple"],
            -len(set(row["apps"]) - apps),
        ))
        selected.append(row)
        families.add(row["family_id"])
        difficulties[row["difficulty"]] += 1
        app_counts["single" if row["num_apps"] == 1 else "multiple"] += 1
        apps.update(row["apps"])
    return selected


def prepare_study(root: Path, seed: int = 20260722, count: int = 8) -> dict[str, Any]:
    """Reserve dev before train without inspecting prompts or historical runs."""
    dev_pool = metadata_pool(root, "dev")
    train_pool = metadata_pool(root, "train")
    dev = select_tasks(dev_pool, count, seed, set())
    train = select_tasks(train_pool, count, seed, {row["family_id"] for row in dev})
    sources = {
        split: hashlib.sha256((root / "data" / "datasets" / f"{split}.txt").read_bytes()).hexdigest()
        for split in ("train", "dev")
    }
    versions = {}
    for kind in ("code", "data"):
        path = root / "version" / f"{kind}.txt"
        versions[kind] = path.read_text(encoding="utf-8").strip() if path.exists() else None
    return {
        "study_id": "train_dev_v1", "status": "task_selection_fixed_prompt_pending",
        "selection_seed": seed,
        "selection_rule": "dev first; unique ID-prefix families; least represented difficulty, then single/multiple apps, then new apps; seeded tie order",
        "metadata_boundary": "Only train/dev IDs, difficulty, num_apps and required_apps. No instructions, answers, solutions, evaluator results, test files or archived runs.",
        "dataset_list_sha256": sources, "appworld_versions": versions,
        "appworld_commit": git_commit(root),
        "train": train, "reserved_dev": dev,
        "smoke_task_ids": [row["task_id"] for row in train[:2]],
        "settings": {
            "agent_model": "qwen3-coder-flash-2025-07-28",
            "controller_model": "deepseek-v4-pro",
            "agent_base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "controller_base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "api_key_env": "DASHSCOPE_API_KEY",
            "temperature": 0.0, "seed": seed, "controller_enable_thinking": False,
            "controller_system_prompt": None,
            "max_controller_calls": 4, "max_controller_blocks": 4,
            "trigger_profile": "all",
            "agent_max_output_tokens_per_call": 65536,
            "controller_max_output_tokens_per_call": 512,
            "controller_request_timeout_seconds": 60.0,
            "max_agent_turns": 10000, "max_environment_executions": 10000,
            "max_extra_agent_turns_after_block": 10000,
            "max_consecutive_identical_executions": 3,
        },
    }


def save_study(path: Path, study: dict[str, Any]) -> None:
    """Create a fixed manifest or verify an identical existing selection."""
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != study:
            raise FileExistsError(f"Refusing to replace a fixed study: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(study, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
