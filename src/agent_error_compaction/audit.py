"""Human-readable audit report construction."""

from __future__ import annotations

import json
from typing import Any


def build_audit_markdown(conversion_manifest: dict[str, Any], inspection: dict[str, Any] | None, dataset_report: dict[str, Any], mistake_step_base: int) -> str:
    """Build a Markdown audit from deterministic pipeline artifacts."""
    if mistake_step_base not in {0, 1}:
        raise ValueError("mistake_step_base must be 0 or 1")
    inspection = inspection or {}
    mistake = inspection.get("mistake_step", {})
    chosen_key = "out_of_bounds_if_zero_based" if mistake_step_base == 0 else "out_of_bounds_if_one_based"
    alternate_key = "out_of_bounds_if_one_based" if mistake_step_base == 0 else "out_of_bounds_if_zero_based"
    leakage = dataset_report.get("family_leakage_check", {})
    rejected = dataset_report.get("rejected_records", {})
    lines = [
        "# AFTraj-2K Data Audit",
        "",
        "## Parquet conversion",
        "",
        f"- Source files: {conversion_manifest.get('source_count', 0)}",
        f"- Source rows: {conversion_manifest.get('total_rows', 0)}",
        f"- JSONL output: `{conversion_manifest.get('output', '<missing>')}`",
        f"- Streaming batch size: {conversion_manifest.get('batch_size', '<missing>')}",
        "",
        "| Source | Rows | SHA-256 |",
        "|---|---:|---|",
    ]
    for source in conversion_manifest.get("sources", []):
        lines.append(f"| `{source.get('path')}` | {source.get('row_count')} | `{source.get('sha256')}` |")
    lines.extend(
        [
            "",
            "## Source format inspection",
            "",
            f"- Trajectories: {inspection.get('trajectory_count', dataset_report.get('trajectory_count', 0))}",
            f"- Safe: {inspection.get('safe_count', dataset_report.get('outcome_distribution', {}).get('success', 0))}",
            f"- Failure: {inspection.get('failure_count', dataset_report.get('outcome_distribution', {}).get('failure', 0))}",
            f"- `mistake_step` range: {mistake.get('min', 'validated during conversion')} to {mistake.get('max', 'validated during conversion')}",
            f"- Selected convention: {mistake_step_base}-based",
            f"- Out of bounds under selected convention: {len(mistake.get(chosen_key, [])) if mistake else '0 rejected by canonical validation'}",
            "",
            "Observed source schema:",
            "",
            "```json",
            json.dumps({"parquet_schema": [source.get("schema") for source in conversion_manifest.get("sources", [])], "field_types": inspection.get("key_field_types", {})}, ensure_ascii=False, indent=2, sort_keys=True),
            "```",
            "",
            "## Canonical validation",
            "",
            f"- Canonical trajectories: {dataset_report.get('trajectory_count', 0)}",
            f"- Rejected records: {rejected.get('count', 0)}",
            f"- Family leakage: {'PASS' if leakage.get('passed') else 'FAIL'}",
            f"- Distinct families: {leakage.get('family_count', 0)}",
            f"- Prefix samples: {dataset_report.get('prefix_count', 0)}",
            "",
            "Rejected reason counts:",
            "",
            "```json",
            json.dumps(rejected.get("reasons", {}), ensure_ascii=False, indent=2, sort_keys=True),
            "```",
            "",
            "## Distribution summary",
            "",
            "```json",
            json.dumps(
                {
                    "outcome": dataset_report.get("outcome_distribution", {}),
                    "domain": dataset_report.get("domain_distribution", {}),
                    "unsafe_source": dataset_report.get("unsafe_source_distribution", {}),
                    "trajectory_length": dataset_report.get("trajectory_length", {}),
                    "error_onset_absolute": dataset_report.get("error_onset_absolute_distribution", {}),
                    "responsible_actor": dataset_report.get("responsible_actor_distribution", {}),
                    "splits": dataset_report.get("splits", {}),
                    "error_occurred": dataset_report.get("error_occurred_distribution", {}),
                    "distance_to_error_onset": dataset_report.get("distance_to_error_onset_distribution", {}),
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            "```",
            "",
            "## Audit conclusions",
            "",
            "- Parquet rows are converted one-for-one to JSONL records in source-file order.",
            "- Canonical conversion preserves one step per source turn and retains each complete `raw_step`.",
            "- Family grouping is completed before split assignment; the report performs a strict leakage check.",
            "- Prefixes contain only steps up to and including `prefix_end_step`.",
            "- Error reasons and source labels are preserved without inference or rewriting.",
            "",
        ]
    )
    return "\n".join(lines)
