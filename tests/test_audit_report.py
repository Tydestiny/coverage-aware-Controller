from agent_error_compaction.audit import build_audit_markdown


def test_audit_markdown_contains_conversion_and_validation_evidence():
    manifest = {"total_rows": 2, "sources": [{"path": "safe.parquet", "row_count": 1}, {"path": "unsafe.parquet", "row_count": 1}]}
    inspection = {
        "trajectory_count": 2,
        "safe_count": 1,
        "failure_count": 1,
        "mistake_step": {"min": 0, "max": 0, "out_of_bounds_if_zero_based": [], "out_of_bounds_if_one_based": [{"conv_id": "bad"}]},
        "key_field_types": {"action": {"str": 2}},
    }
    report = {"trajectory_count": 2, "prefix_count": 4, "family_leakage_check": {"passed": True}, "rejected_records": {"count": 0, "reasons": {}}}

    markdown = build_audit_markdown(manifest, inspection, report, mistake_step_base=0)

    assert "Parquet conversion" in markdown
    assert "0-based" in markdown
    assert "Family leakage" in markdown
    assert "Rejected records: 0" in markdown
