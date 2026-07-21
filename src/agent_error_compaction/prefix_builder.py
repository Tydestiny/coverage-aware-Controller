"""Causally valid trajectory-prefix construction."""

from __future__ import annotations

from collections import defaultdict
import random

from .schemas import CanonicalTrajectory, PrefixSample

DEFAULT_ONSET_WINDOW = (-3, -2, -1, 0, 1, 2, 3)


def _sample(trajectory: CanonicalTrajectory, end_step: int, mode: str, seed: int) -> PrefixSample:
    """Build one prefix ending at an inclusive canonical step index."""
    onset = trajectory.labels.error_onset_step
    occurred = onset is not None and end_step >= onset
    distance = end_step - onset if onset is not None else None
    return PrefixSample(
        prefix_id=f"{trajectory.trajectory_id}:end:{end_step}",
        trajectory_id=trajectory.trajectory_id,
        trajectory_family_id=trajectory.trajectory_family_id,
        source_split=trajectory.source_split,
        prefix_end_step=end_step,
        steps=trajectory.steps[: end_step + 1],
        error_occurred=occurred,
        distance_to_error_onset=distance,
        metadata={"mode": mode, "seed": seed},
    )


def build_prefixes(trajectories: list[CanonicalTrajectory], mode: str = "all", window: tuple[int, ...] = DEFAULT_ONSET_WINDOW, seed: int = 42) -> list[PrefixSample]:
    """Create all prefixes or an onset-window subset without exposing future steps."""
    if mode not in {"all", "onset_window"}:
        raise ValueError("mode must be 'all' or 'onset_window'")
    if mode == "all":
        return [_sample(trajectory, end, mode, seed) for trajectory in trajectories for end in range(len(trajectory.steps))]
    failures = [item for item in trajectories if item.labels.is_failure]
    safe = [item for item in trajectories if not item.labels.is_failure]
    samples: list[PrefixSample] = []
    relative_targets: list[float] = []
    for trajectory in failures:
        assert trajectory.labels.error_onset_step is not None
        for distance in window:
            end = trajectory.labels.error_onset_step + distance
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

