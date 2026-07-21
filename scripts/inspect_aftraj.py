"""Inspect unknown AFTraj JSON/JSONL fields without deciding label indexing."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from agent_error_compaction.io_utils import read_records


def _type_name(value: Any) -> str:
    """Return a stable JSON-oriented runtime type name."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, dict):
        return "object"
    if isinstance(value, list):
        return "array"
    return type(value).__name__


def inspect_records(input_path: Path, num_samples: int) -> dict[str, Any]:
    """Collect field/type evidence needed before AFTraj adapter configuration."""
    top_fields: dict[str, Counter[str]] = {}
    turn_fields: dict[str, Counter[str]] = {}
    key_types: dict[str, Counter[str]] = {key: Counter() for key in ("action", "content", "thought")}
    lengths: list[int] = []
    onset_values: list[int] = []
    out_of_bounds_zero: list[dict[str, Any]] = []
    out_of_bounds_one: list[dict[str, Any]] = []
    safe_samples: list[dict[str, Any]] = []
    failure_samples: list[dict[str, Any]] = []
    actor_evidence = {"zero_based_eligible": 0, "zero_based_matches": 0, "one_based_eligible": 0, "one_based_matches": 0}
    total = safe_count = failure_count = 0
    for record in read_records(input_path):
        total += 1
        for key, value in record.items():
            top_fields.setdefault(key, Counter())[_type_name(value)] += 1
        turns = record.get("turns")
        length = len(turns) if isinstance(turns, list) else 0
        lengths.append(length)
        if isinstance(turns, list):
            for turn in turns:
                if not isinstance(turn, dict):
                    continue
                for key, value in turn.items():
                    turn_fields.setdefault(key, Counter())[_type_name(value)] += 1
                for key in key_types:
                    if key in turn:
                        key_types[key][_type_name(turn[key])] += 1
        mistake = record.get("mistake_step")
        is_failure = mistake is not None
        if is_failure:
            failure_count += 1
            if isinstance(mistake, int) and not isinstance(mistake, bool):
                onset_values.append(mistake)
                evidence = {"conv_id": record.get("conv_id"), "mistake_step": mistake, "turn_count": length}
                if not 0 <= mistake < length:
                    out_of_bounds_zero.append(evidence)
                if not 1 <= mistake <= length:
                    out_of_bounds_one.append(evidence)
                mistake_agent = record.get("mistake_agent")
                if isinstance(turns, list) and 0 <= mistake < length and isinstance(turns[mistake], dict):
                    actor_evidence["zero_based_eligible"] += 1
                    actor_evidence["zero_based_matches"] += turns[mistake].get("role") == mistake_agent
                if isinstance(turns, list) and 1 <= mistake <= length and isinstance(turns[mistake - 1], dict):
                    actor_evidence["one_based_eligible"] += 1
                    actor_evidence["one_based_matches"] += turns[mistake - 1].get("role") == mistake_agent
            if len(failure_samples) < num_samples:
                failure_samples.append(record)
        else:
            safe_count += 1
            if len(safe_samples) < num_samples:
                safe_samples.append(record)
    return {"input": str(input_path), "trajectory_count": total, "safe_count": safe_count, "failure_count": failure_count, "top_level_field_types": {key: dict(value) for key, value in sorted(top_fields.items())}, "turn_field_types": {key: dict(value) for key, value in sorted(turn_fields.items())}, "key_field_types": {key: dict(value) for key, value in key_types.items()}, "length_distribution": {"min": min(lengths) if lengths else None, "max": max(lengths) if lengths else None, "mean": sum(lengths) / len(lengths) if lengths else None, "counts": dict(Counter(lengths))}, "mistake_step": {"min": min(onset_values) if onset_values else None, "max": max(onset_values) if onset_values else None, "out_of_bounds_if_zero_based": out_of_bounds_zero, "out_of_bounds_if_one_based": out_of_bounds_one, "actor_match_evidence": actor_evidence, "note": "This report does not infer whether mistake_step is 0-based or 1-based."}, "safe_samples": safe_samples, "failure_samples": failure_samples}


def main() -> None:
    """Run the AFTraj structural inspector."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = inspect_records(args.input, args.num_samples)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"trajectories={report['trajectory_count']} safe={report['safe_count']} failure={report['failure_count']}")
    print(f"length_range={report['length_distribution']['min']}..{report['length_distribution']['max']} mistake_step_range={report['mistake_step']['min']}..{report['mistake_step']['max']}")
    print(f"saved={args.output}")


if __name__ == "__main__":
    main()
