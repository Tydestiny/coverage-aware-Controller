# diagnosis-aware-compaction

Preprocessing-only pipeline for converting AFTraj-2K JSON/JSONL exports into a diagnosis-aware canonical trajectory format. It does not implement a compactor, verifier, LLM call, or model training.

The repository deliberately does not track raw or processed dataset payloads. It includes only code and a small synthetic fixture.

## Prerequisites

- Python 3.11+
- `pytest` for the test suite

The source parser supports JSON and JSONL. The supplied files under `dataset/AFTraj-2K/` are streamed to JSONL with PyArrow; source Parquet files remain untouched and generated data stays ignored by Git.

## Required AFTraj confirmation

Before converting real data, confirm with `inspect_aftraj.py`:

- top-level `turns` is an array of source turns;
- the actual field types for `action`, `content`, and `thought`;
- whether `mistake_step` is 0-based or 1-based;
- whether `mistake_step = null` reliably denotes a safe trajectory;
- whether the source provides a parent/family linkage ID and its exact field name.

The adapter does not guess these facts. It rejects malformed records and records the reason in `rejected_records.jsonl`.

## Workflow

```bash
python scripts/parquet_to_jsonl.py \
  --input dataset/AFTraj-2K/aftraj_safe.parquet dataset/AFTraj-2K/aftraj_unsafe.parquet \
  --output data/raw/aftraj.jsonl \
  --manifest data/manifests/parquet_conversion.json

python scripts/inspect_aftraj.py \
  --input data/raw/aftraj.jsonl \
  --num-samples 20 \
  --output data/processed/inspection.json

python scripts/convert_aftraj.py \
  --input data/raw/aftraj.jsonl \
  --output data/processed/aftraj_all.jsonl \
  --rejected data/processed/rejected_records.jsonl \
  --mistake-step-base 0

python scripts/build_splits.py \
  --input data/processed/aftraj_all.jsonl \
  --output-dir data/processed \
  --manifest-dir data/manifests \
  --seed 42

python scripts/build_prefixes.py \
  --input-dir data/processed \
  --output-dir data/processed/prefixes \
  --mode onset_window \
  --window=-3,-2,-1,0,1,2,3 \
  --seed 42

python scripts/generate_report.py \
  --trajectory-dir data/processed \
  --prefix-dir data/processed/prefixes \
  --output-json data/processed/dataset_report.json \
  --output-md data/processed/dataset_report.md

python scripts/generate_audit.py \
  --conversion-manifest data/manifests/parquet_conversion.json \
  --inspection data/processed/inspection.json \
  --dataset-report data/processed/dataset_report.json \
  --mistake-step-base 0 \
  --output data/processed/aftraj_data_audit.md
```

Canonical conversion preserves exactly one `CanonicalStep` per source turn, keeps every `raw_step`, uses the configured source index base without auto-correction, assigns families before splitting, and creates every prefix with `steps[:prefix_end_step + 1]` only.
