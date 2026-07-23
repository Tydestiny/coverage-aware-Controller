"""Filesystem storage for runtime checkpoint collection."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
from typing import Any


FORBIDDEN_KEYS = {"ground_truth", "evaluator", "evaluation", "hidden_state"}


class CheckpointStorage:
    """Store runtime checkpoints and an index manifest."""

    def __init__(self, root_dir: Path, run_id: str, task_id: str) -> None:
        self.task_id = task_id
        self.base_dir = root_dir / run_id / task_id
        self.post_dir = self.base_dir / "post_observation"
        self.pre_dir = self.base_dir / "pre_action"
        self.post_dir.mkdir(parents=True, exist_ok=True)
        self.pre_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.base_dir / "manifest.json"
        self.manifest: dict[str, Any] = {
            "run_id": run_id,
            "task_id": task_id,
            "post_observation": [],
            "pre_action": [],
            "terminal": None,
        }
        self._write_manifest()

    def write_post_observation(self, checkpoint: Any) -> Path:
        """Write a post-observation checkpoint and update manifest."""
        data = self._to_jsonable(checkpoint)
        self._assert_no_forbidden_keys(data)
        path = self.post_dir / f"{data['checkpoint_id']}.json"
        self._write_json(path, data)
        self.manifest["post_observation"].append(
            {
                "checkpoint_id": data["checkpoint_id"],
                "agent_turn_index": data["agent_turn_index"],
                "environment_execution_index": data["environment_execution_index"],
                "path": str(path),
            }
        )
        self._write_manifest()
        return path

    def write_pre_action(self, record: Any) -> Path:
        """Write a pre-action record and update manifest."""
        data = self._to_jsonable(record)
        self._assert_no_forbidden_keys(data)
        path = self.pre_dir / f"{data['record_id']}.json"
        self._write_json(path, data)
        self.manifest["pre_action"].append(
            {
                "record_id": data["record_id"],
                "agent_turn_index": data["agent_turn_index"],
                "latest_post_observation_checkpoint_id": data["latest_post_observation_checkpoint_id"],
                "path": str(path),
            }
        )
        self._write_manifest()
        return path

    def write_terminal(self, record: Any) -> Path:
        """Write terminal lifecycle record and update manifest."""
        data = self._to_jsonable(record)
        self._assert_no_forbidden_keys(data)
        path = self.base_dir / "terminal.json"
        self._write_json(path, data)
        self.manifest["terminal"] = {"path": str(path), "termination_reason": data["termination_reason"]}
        self._write_manifest()
        return path

    def _write_manifest(self) -> None:
        """Write manifest JSON."""
        self._write_json(self.manifest_path, self.manifest)

    def _write_json(self, path: Path, data: Any) -> None:
        """Write JSON data."""
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    def _to_jsonable(self, value: Any) -> Any:
        """Convert dataclasses and nested objects into JSON-compatible data."""
        if is_dataclass(value):
            return self._to_jsonable(asdict(value))
        if isinstance(value, dict):
            return {str(key): self._to_jsonable(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._to_jsonable(item) for item in value]
        if isinstance(value, tuple):
            return [self._to_jsonable(item) for item in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)

    def _assert_no_forbidden_keys(self, value: Any, path: str = "") -> None:
        """Reject records containing known forbidden hidden/evaluator keys."""
        if isinstance(value, dict):
            for key, item in value.items():
                key_path = f"{path}.{key}" if path else str(key)
                if str(key).lower() in FORBIDDEN_KEYS:
                    raise ValueError(f"Forbidden checkpoint field: {key_path}")
                self._assert_no_forbidden_keys(item, key_path)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                self._assert_no_forbidden_keys(item, f"{path}[{index}]")

