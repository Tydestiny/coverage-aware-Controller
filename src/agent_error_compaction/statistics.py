"""Dataset and prefix statistics reporting for schema v1.1."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from statistics import mean, median
from typing import Any

from .schemas import CanonicalTrajectory, PrefixSample
from .split_builder import assert_no_family_leakage


def _distribution(values: list[Any]) -> dict[str, int]:
    """Count values under readable JSON object keys."""
    return dict(sorted(Counter("<missing>" if value is None else str(value) for value in values).items()))


def _quantile(sorted_values: list[float], probability: float) -> float | None:
    """Return a nearest-rank quantile without third-party dependencies."""
    if not sorted_values:
        return None
    return sorted_values[round((len(sorted_values) - 1) * probability)]


def _length_summary(values: list[int]) -> dict[str, float | int | None]:
    """Summarize trajectory lengths and selected quantiles."""
    ordered = sorted(values)
    return {"mean": mean(values) if values else None, "median": median(values) if values else None, "min": min(values) if values else None, "max": max(values) if values else None, "p25": _quantile(ordered, 0.25), "p75": _quantile(ordered, 0.75), "p90": _quantile(ordered, 0.90)}


def build_report(trajectories_by_split: dict[str, list[CanonicalTrajectory]], prefixes: list[PrefixSample], rejected_records: list[dict[str, Any]]) -> dict[str, Any]:
    """Create a JSON-compatible v1.1 report."""
    all_trajectories = [item for values in trajectories_by_split.values() for item in values]
    assert_no_family_leakage(trajectories_by_split)
    failures = [item for item in all_trajectories if item.annotations.primary_error is not None]
    onset_abs = [item.annotations.primary_error.step_index for item in failures if item.annotations.primary_error]
    onset_rel = [item.annotations.primary_error.step_index / len(item.steps) for item in failures if item.annotations.primary_error]
    return {
        "schema_version": "1.1",
        "trajectory_count": len(all_trajectories),
        "outcome_distribution": _distribution([item.outcome.status for item in all_trajectories]),
        "domain_distribution": _distribution([item.environment.domain for item in all_trajectories]),
        "annotation_method_distribution": _distribution([item.annotations.primary_error.annotation_method if item.annotations.primary_error else "safe" for item in all_trajectories]),
        "trajectory_length": _length_summary([len(item.steps) for item in all_trajectories]),
        "critical_error_absolute_distribution": _distribution(onset_abs),
        "critical_error_relative_distribution": _distribution([round(value, 4) for value in onset_rel]),
        "responsible_actor_distribution": _distribution([item.annotations.primary_error.actor_id for item in failures if item.annotations.primary_error]),
        "splits": {name: {"trajectory_count": len(values), "outcome_distribution": _distribution([item.outcome.status for item in values]), "domain_distribution": _distribution([item.environment.domain for item in values])} for name, values in trajectories_by_split.items()},
        "family_leakage_check": {"passed": True, "family_count": len({item.trajectory_family_id for item in all_trajectories})},
        "prefix_count": len(prefixes),
        "critical_error_occurred_distribution": _distribution([item.critical_error_occurred for item in prefixes]),
        "distance_to_critical_error_distribution": _distribution([item.distance_to_critical_error for item in prefixes]),
        "rejected_records": {"count": len(rejected_records), "reasons": _distribution([item.get("reason") for item in rejected_records])},
    }


def report_markdown(report: dict[str, Any]) -> str:
    """Render a compact v1.1 Markdown report."""
    selected = {key: report[key] for key in ("outcome_distribution", "domain_distribution", "annotation_method_distribution", "critical_error_occurred_distribution", "distance_to_critical_error_distribution")}
    lines = ["# Dataset Report", "", f"- Schema: {report['schema_version']}", f"- Trajectories: {report['trajectory_count']}", f"- Prefixes: {report['prefix_count']}", f"- Family leakage check: {'passed' if report['family_leakage_check']['passed'] else 'failed'}", f"- Rejected records: {report['rejected_records']['count']}", "", "## Distributions", "", "```json", json.dumps(selected, ensure_ascii=False, indent=2, sort_keys=True), "```", "", "## Split Summary", "", "```json", json.dumps(report["splits"], ensure_ascii=False, indent=2, sort_keys=True), "```", ""]
    return "\n".join(lines)


def write_report(report: dict[str, Any], output_json: Path, output_md: Path) -> None:
    """Write JSON and Markdown report artifacts."""
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_md.write_text(report_markdown(report), encoding="utf-8")

