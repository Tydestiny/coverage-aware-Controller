"""Utilities for restoring AppWorld Python executor namespace from checkpoints."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class NamespaceRestoreResult:
    """Result of rebuilding the Python executor namespace."""

    success: bool
    mode: str
    steps: int
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable result dictionary."""
        return asdict(self)


def restore_python_namespace_from_prefix(world: Any, checkpoint: dict[str, Any]) -> NamespaceRestoreResult:
    """Replay saved execution code to rebuild AppWorld REPL globals before state load."""
    mode = str(checkpoint.get("namespace_restore_mode", "replay_required"))
    if mode != "replay_required":
        return NamespaceRestoreResult(success=True, mode=mode, steps=0)

    if "executed_code_prefix" not in checkpoint:
        return NamespaceRestoreResult(
            success=False,
            mode=mode,
            steps=0,
            error="Checkpoint is missing executed_code_prefix; cannot rebuild Python namespace.",
        )
    prefix = checkpoint["executed_code_prefix"]
    if not isinstance(prefix, list):
        return NamespaceRestoreResult(
            success=False,
            mode=mode,
            steps=0,
            error="Checkpoint executed_code_prefix must be a list.",
        )

    steps = 0
    try:
        for index, code in enumerate(prefix):
            if not isinstance(code, str) or not code.strip():
                return NamespaceRestoreResult(
                    success=False,
                    mode=mode,
                    steps=steps,
                    error=f"Checkpoint executed_code_prefix[{index}] is empty or not a string.",
                )
            output = str(world.execute(code))
            steps += 1
            if output.lstrip().startswith("Execution failed."):
                return NamespaceRestoreResult(
                    success=False,
                    mode=mode,
                    steps=steps,
                    error=f"Replay failed at executed_code_prefix[{index}]: {first_line(output)}",
                )
    except Exception as exc:
        return NamespaceRestoreResult(
            success=False,
            mode=mode,
            steps=steps,
            error=f"{type(exc).__name__}: {exc}",
        )
    finally:
        reset_world_interaction_logs(world)

    return NamespaceRestoreResult(success=True, mode=mode, steps=steps)


def reset_world_interaction_logs(world: Any) -> None:
    """Clear restore-only AppWorld logs so branch artifacts contain continuation only."""
    if hasattr(world, "environment_io"):
        world.environment_io = []
    if hasattr(world, "num_interactions"):
        world.num_interactions = 0
    if hasattr(world, "num_sub_interactions"):
        world.num_sub_interactions = 0
    requester = getattr(world, "requester", None)
    if requester is not None and hasattr(requester, "reset_requests"):
        requester.reset_requests()
    if hasattr(world, "save_logs"):
        world.save_logs()


def first_line(text: str) -> str:
    """Return the first non-empty line from text."""
    for line in text.splitlines():
        if line.strip():
            return line.strip()
    return ""
