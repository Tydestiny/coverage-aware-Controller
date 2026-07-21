"""Adapter contract for lossless trajectory conversion."""

from __future__ import annotations

from abc import ABC, abstractmethod
import json
from pathlib import Path
from typing import Any, Iterator

from ..io_utils import read_records
from ..schemas import CanonicalTrajectory
from ..validation import CanonicalValidator, validate_trajectory


class BaseTrajectoryAdapter(ABC):
    """Convert a source dataset into validated canonical JSONL records."""

    validator = CanonicalValidator()

    def load_records(self, path: Path) -> Iterator[dict[str, Any]]:
        """Stream source object records from JSON or JSONL."""
        return read_records(path)

    @abstractmethod
    def convert_record(self, record: dict[str, Any]) -> CanonicalTrajectory:
        """Convert one source record into a canonical trajectory."""

    def convert_file(self, input_path: Path, output_path: Path, rejected_path: Path) -> dict[str, int]:
        """Stream-convert an input file and write rejected records with reasons."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        rejected_path.parent.mkdir(parents=True, exist_ok=True)
        stats = {"total": 0, "converted": 0, "rejected": 0}
        with output_path.open("w", encoding="utf-8", newline="\n") as output, rejected_path.open("w", encoding="utf-8", newline="\n") as rejected:
            try:
                records = self.load_records(input_path)
                for index, record in enumerate(records):
                    stats["total"] += 1
                    try:
                        trajectory = self.convert_record(record)
                        self.validator.validate(trajectory)
                    except Exception as exc:  # Per-record conversion failures must be recorded.
                        rejected.write(json.dumps({"record_index": index, "reason": str(exc), "raw_record": record}, ensure_ascii=False) + "\n")
                        stats["rejected"] += 1
                        continue
                    output.write(json.dumps(trajectory.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
                    stats["converted"] += 1
            except Exception as exc:
                rejected.write(json.dumps({"record_index": None, "reason": str(exc), "raw_record": None}, ensure_ascii=False) + "\n")
                stats["rejected"] += 1
        return stats
