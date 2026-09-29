"""Local paths for development runs; historical data lives outside both repositories."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def appworld_root() -> Path:
    """Return the configured external AppWorld checkout."""
    return Path(os.environ.get("APPWORLD_ROOT") or REPO_ROOT.parent / "appworld").resolve()


def artifact_root() -> Path:
    """Return the root for new train/dev experiment artifacts."""
    return Path(os.environ.get("CONTROLLER_ARTIFACT_ROOT") or REPO_ROOT / "artifacts" / "experiments").resolve()


def appworld_python(root: Path) -> Path:
    """Use the AppWorld environment for real task execution when it exists."""
    candidate = root / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    return candidate if candidate.is_file() else Path(sys.executable)


def official_task_ids(root: Path, split: str) -> set[str]:
    """Read task IDs from the official train or dev split only."""
    if split not in {"train", "dev"}:
        raise ValueError("Development runs accept only train/dev splits")
    path = root / "data" / "datasets" / f"{split}.txt"
    if not path.is_file():
        raise FileNotFoundError(f"Official {split} task list is missing: {path}")
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def git_commit(root: Path) -> str | None:
    """Return a repository revision when Git metadata is available."""
    try:
        completed = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=False)
    except OSError:
        return None
    return completed.stdout.strip() if completed.returncode == 0 else None
