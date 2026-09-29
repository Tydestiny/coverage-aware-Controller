"""Task-selection invariants without any real task content or model calls."""

import json
from pathlib import Path

import pytest

from controller_agent.development_study import prepare_study, save_study


def _fixture(root: Path) -> None:
    datasets = root / "data" / "datasets"
    datasets.mkdir(parents=True)
    for split in ("train", "dev"):
        ids = []
        for family in range(12):
            for variant in (1, 2):
                task_id = f"{split}{family:02d}_{variant}"
                ids.append(task_id)
                gt = root / "data" / "tasks" / task_id / "ground_truth"
                gt.mkdir(parents=True)
                (gt / "metadata.json").write_text(json.dumps({
                    "difficulty": family % 3 + 1, "num_apps": family % 2 + 1,
                    "solution": "DO_NOT_USE_THIS_FIELD",
                }), encoding="utf-8")
                (gt / "required_apps.json").write_text(json.dumps([f"app{family % 4}"]), encoding="utf-8")
        (datasets / f"{split}.txt").write_text("\n".join(ids), encoding="utf-8")


def test_selection_is_reproducible_balanced_and_family_disjoint(tmp_path: Path) -> None:
    _fixture(tmp_path)
    first = prepare_study(tmp_path)
    assert first == prepare_study(tmp_path)
    assert len(first["train"]) == len(first["reserved_dev"]) == 8
    all_rows = first["train"] + first["reserved_dev"]
    assert len({row["family_id"] for row in all_rows}) == 16
    for split in ("train", "reserved_dev"):
        assert {row["difficulty"] for row in first[split]} == {1, 2, 3}
        assert {row["num_apps"] for row in first[split]} == {1, 2}
    assert "DO_NOT_USE_THIS_FIELD" not in json.dumps(first)
    assert first["settings"]["controller_system_prompt"] is None


def test_existing_task_selection_is_immutable(tmp_path: Path) -> None:
    _fixture(tmp_path)
    study = prepare_study(tmp_path)
    path = tmp_path / "study.json"
    save_study(path, study)
    before = path.read_bytes()
    save_study(path, study)
    assert path.read_bytes() == before
    with pytest.raises(FileExistsError):
        save_study(path, {**study, "selection_seed": 1})


def test_short_pool_fails_instead_of_reusing_families(tmp_path: Path) -> None:
    _fixture(tmp_path)
    with pytest.raises(ValueError, match="distinct task families"):
        prepare_study(tmp_path, count=13)
