"""Leakage-safe, deterministic trajectory-family split construction."""

from __future__ import annotations

from collections import defaultdict
import random
from typing import Iterable

from .schemas import CanonicalTrajectory

SPLIT_NAMES = ("train", "validation", "test")
SPLIT_RATIOS = {"train": 0.60, "validation": 0.20, "test": 0.20}


def _stratum(trajectory: CanonicalTrajectory) -> tuple[str, str, str]:
    """Return the coarse attributes used to approximate balanced assignment."""
    annotation_method = trajectory.annotations.primary_error.annotation_method if trajectory.annotations.primary_error else "safe"
    return (trajectory.environment.domain or "<missing>", trajectory.outcome.status, annotation_method or "<missing>")


def build_group_splits(trajectories: Iterable[CanonicalTrajectory], seed: int = 42) -> dict[str, list[CanonicalTrajectory]]:
    """Assign each family once while approximately preserving coarse distributions."""
    items = list(trajectories)
    if len({item.trajectory_id for item in items}) != len(items):
        raise ValueError("trajectory_id values must be unique before splitting")
    families: dict[str, list[CanonicalTrajectory]] = defaultdict(list)
    for item in items:
        if not item.trajectory_family_id:
            raise ValueError(f"trajectory {item.trajectory_id} has no trajectory_family_id")
        families[item.trajectory_family_id].append(item)
    randomizer = random.Random(seed)
    family_items = list(families.items())
    family_items.sort(key=lambda item: item[0])
    randomizer.shuffle(family_items)
    targets = {name: len(items) * ratio for name, ratio in SPLIT_RATIOS.items()}
    assigned_counts = {name: 0 for name in SPLIT_NAMES}
    stratum_totals: dict[tuple[str, str, str], int] = defaultdict(int)
    for item in items:
        stratum_totals[_stratum(item)] += 1
    stratum_assigned: dict[str, dict[tuple[str, str, str], int]] = {name: defaultdict(int) for name in SPLIT_NAMES}
    result = {name: [] for name in SPLIT_NAMES}
    for family_id, members in family_items:
        profile: dict[tuple[str, str, str], int] = defaultdict(int)
        for member in members:
            profile[_stratum(member)] += 1
        def score(name: str) -> tuple[float, float, str]:
            total_deficit = targets[name] - assigned_counts[name]
            balance_deficit = sum(profile[key] * ((stratum_totals[key] * SPLIT_RATIOS[name]) - stratum_assigned[name][key]) for key in profile)
            return (balance_deficit, total_deficit, name)
        destination = max(SPLIT_NAMES, key=score)
        result[destination].extend(members)
        assigned_counts[destination] += len(members)
        for key, count in profile.items():
            stratum_assigned[destination][key] += count
    assert_no_family_leakage(result)
    return result


def assert_no_family_leakage(splits: dict[str, list[CanonicalTrajectory]]) -> None:
    """Raise when a trajectory family appears in more than one split."""
    seen: dict[str, str] = {}
    for split_name, trajectories in splits.items():
        for trajectory in trajectories:
            family = trajectory.trajectory_family_id
            if family is None:
                raise ValueError(f"trajectory {trajectory.trajectory_id} has no family")
            previous = seen.setdefault(family, split_name)
            if previous != split_name:
                raise ValueError(f"family {family} leaks between {previous} and {split_name}")
