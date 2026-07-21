"""Trajectory family construction without semantic rewriting."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import replace
from typing import Any

from .schemas import CanonicalTrajectory


def normalize_text(value: Any) -> str:
    """Normalize only edge whitespace, internal whitespace, and casing."""
    if value is None:
        return ""
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return re.sub(r"\s+", " ", value.strip()).lower()


def _explicit_family_id(trajectory: CanonicalTrajectory) -> str | None:
    """Return a declared linkage ID from source metadata when available."""
    candidates = ("trajectory_family_id", "parent_trajectory_id", "parent_id", "related_trajectory_id", "family_id")
    for key in candidates:
        value = trajectory.metadata.get(key)
        if isinstance(value, (str, int)) and str(value).strip():
            return str(value)
    for key in candidates:
        value = trajectory.metadata.get("source_record", {}).get(key) if isinstance(trajectory.metadata.get("source_record"), dict) else None
        if isinstance(value, (str, int)) and str(value).strip():
            return str(value)
    return None


def assign_family(trajectory: CanonicalTrajectory) -> CanonicalTrajectory:
    """Assign an explicit family ID or a deterministic non-semantic candidate hash."""
    explicit = _explicit_family_id(trajectory)
    metadata = dict(trajectory.metadata)
    if explicit is not None:
        metadata["family_builder"] = {"method": "explicit_source_link", "fields": ["trajectory_family_id", "parent_trajectory_id", "parent_id", "related_trajectory_id", "family_id"]}
        return replace(trajectory, trajectory_family_id=explicit, metadata=metadata)
    first_steps = [normalize_text(step.raw_step) for step in trajectory.steps[:3]]
    material = {"task": normalize_text(trajectory.task), "gold_answer": normalize_text(trajectory.gold_answer), "domain": normalize_text(trajectory.domain), "first_three_raw_steps": first_steps}
    encoded = json.dumps(material, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    family_id = "hash:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:24]
    metadata["family_builder"] = {"method": "normalized_candidate_hash", "normalization": "strip, collapse whitespace, lowercase", "hash_fields": list(material)}
    return replace(trajectory, trajectory_family_id=family_id, metadata=metadata)


def assign_families(trajectories: list[CanonicalTrajectory]) -> list[CanonicalTrajectory]:
    """Assign a family to every supplied canonical trajectory."""
    return [assign_family(trajectory) for trajectory in trajectories]

