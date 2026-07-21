"""Causally valid trajectory-prefix construction for schema v1.1."""

from __future__ import annotations

import random

from .schemas import CanonicalTrajectory, PrefixSample

DEFAULT_ONSET_WINDOW = (-3, -2, -1, 0, 1, 2, 3)


def _sample(trajectory: CanonicalTrajectory, end_step: int, mode: str, seed: int) -> PrefixSample:
    """Build one prefix ending at an inclusive canonical step index."""
    primary = trajectory.annotations.primary_error
    critical_step = primary.step_index if primary is not None and primary.alignment_status == "exact" else None
    occurred = critical_step is not None and end_step >= critical_step
    distance = end_step - critical_step if critical_step is not None else None
    return PrefixSample(
        prefix_id=f"{trajectory.trajectory_id}:end:{end_step}",
        trajectory_id=trajectory.trajectory_id,
        trajectory_family_id=trajectory.trajectory_family_id,
        experiment_split=trajectory.experiment_split,
        prefix_end_step=end_step,
        steps=trajectory.steps[: end_step + 1],
        critical_error_occurred=occurred,
        critical_error_step=critical_step,
        distance_to_critical_error=distance,
        responsible_actor=primary.actor_id if primary is not None else None,
        metadata={"mode": mode, "seed": seed},
    )


def build_prefixes(trajectories: list[CanonicalTrajectory], mode: str = "all", window: tuple[int, ...] = DEFAULT_ONSET_WINDOW, seed: int = 42) -> list[PrefixSample]:
    """Create all prefixes or a critical-error window without future steps."""
    if mode not in {"all", "onset_window"}:
        raise ValueError("mode must be 'all' or 'onset_window'")
    if mode == "all":
        return [_sample(trajectory, end, mode, seed) for trajectory in trajectories for end in range(len(trajectory.steps))]
    failures = [item for item in trajectories if item.annotations.primary_error is not None and item.annotations.primary_error.alignment_status == "exact"]
    safe = [item for item in trajectories if item.annotations.primary_error is None]
    samples: list[PrefixSample] = []
    relative_targets: list[float] = []
    for trajectory in failures:
        assert trajectory.annotations.primary_error is not None
        for distance in window:
            end = trajectory.annotations.primary_error.step_index + distance
            if 0 <= end < len(trajectory.steps):
                samples.append(_sample(trajectory, end, mode, seed))
                relative_targets.append((end + 1) / len(trajectory.steps))
    randomizer = random.Random(seed)
    randomizer.shuffle(relative_targets)
    if safe and relative_targets:
        for index, target in enumerate(relative_targets):
            trajectory = safe[index % len(safe)]
            end = min(len(trajectory.steps) - 1, max(0, round(target * len(trajectory.steps) - 1)))
            samples.append(_sample(trajectory, end, mode, seed))
    return samples
