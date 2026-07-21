from dataclasses import replace

from agent_error_compaction.adapters.aftraj import AFTrajAdapter
from agent_error_compaction.split_builder import build_group_splits


def item(identifier, family, failure=False):
    source = {"conv_id": identifier, "domain": "math" if identifier[-1] in "024" else "coding", "turns": [{"content": "a"}], "mistake_step": 0 if failure else None, "unsafe_source": "injected" if failure else None}
    return replace(AFTrajAdapter().convert_record(source), trajectory_family_id=family)


def test_family_never_crosses_split():
    splits = build_group_splits([item("a0", "shared"), item("a1", "shared"), item("b2", "other", True), item("c3", "third")])
    appearances = [name for name, values in splits.items() if any(value.trajectory_family_id == "shared" for value in values)]
    assert len(appearances) == 1


def test_seed_is_reproducible():
    records = [item(f"id{i}", f"family{i}", i % 2 == 0) for i in range(10)]
    first = build_group_splits(records, 9)
    second = build_group_splits(records, 9)
    assert {key: [value.trajectory_id for value in values] for key, values in first.items()} == {key: [value.trajectory_id for value in values] for key, values in second.items()}


def test_all_trajectories_are_assigned_once():
    records = [item(f"id{i}", f"family{i}") for i in range(6)]
    splits = build_group_splits(records)
    assert sorted(value.trajectory_id for values in splits.values() for value in values) == sorted(value.trajectory_id for value in records)
