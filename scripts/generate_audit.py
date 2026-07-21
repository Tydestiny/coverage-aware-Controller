"""Generate a consolidated AFTraj-2K Markdown audit document."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from agent_error_compaction.audit import build_audit_markdown


def _load_json(path: Path) -> dict[str, object]:
    """Load a JSON object or raise a clear artifact error."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def main() -> None:
    """Combine conversion, inspection, and dataset reports into Markdown."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--conversion-manifest", type=Path, required=True)
    parser.add_argument("--inspection", type=Path)
    parser.add_argument("--dataset-report", type=Path, required=True)
    parser.add_argument("--mistake-step-base", type=int, choices=(0, 1), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inspection = _load_json(args.inspection) if args.inspection else None
    markdown = build_audit_markdown(_load_json(args.conversion_manifest), inspection, _load_json(args.dataset_report), args.mistake_step_base)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(markdown, encoding="utf-8")
    print(f"saved={args.output}")


if __name__ == "__main__":
    main()

