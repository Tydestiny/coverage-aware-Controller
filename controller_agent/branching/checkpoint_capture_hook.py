"""Runtime checkpoint capture hook used by ControlledAppWorldRunner."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from controller_agent.branching.runtime_checkpoint_manager import RuntimeCheckpointManager


class CheckpointCaptureHook(Protocol):
    """Protocol consumed by the AppWorld runner."""

    def on_pre_action(self, **kwargs: Any) -> Any:
        """Capture a pre-action record."""

    def on_post_observation(self, **kwargs: Any) -> Any:
        """Capture a full post-observation checkpoint."""

    def on_terminal(self, **kwargs: Any) -> Any:
        """Capture terminal lifecycle metadata."""


class RuntimeCheckpointCaptureHook:
    """Thin hook wrapper around RuntimeCheckpointManager."""

    def __init__(
        self,
        root_dir: Path,
        run_id: str,
        task_id: str,
        appworld_code_version: str | None = None,
        appworld_data_version: str | None = None,
        base_dir: Path | None = None,
    ) -> None:
        self.manager = RuntimeCheckpointManager(
            root_dir=root_dir,
            run_id=run_id,
            task_id=task_id,
            appworld_code_version=appworld_code_version,
            appworld_data_version=appworld_data_version,
            base_dir=base_dir,
        )

    def on_pre_action(self, **kwargs: Any) -> Any:
        """Capture pre-action metadata."""
        return self.manager.capture_pre_action(**kwargs)

    def on_post_observation(self, **kwargs: Any) -> Any:
        """Capture a full checkpoint."""
        return self.manager.capture_post_observation(**kwargs)

    def on_terminal(self, **kwargs: Any) -> Any:
        """Capture terminal metadata."""
        return self.manager.capture_terminal(**kwargs)
