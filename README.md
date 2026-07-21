# diagnosis-aware-compaction

Preprocessing-only pipeline for converting AFTraj-2K JSON/JSONL exports into Canonical Trajectory Schema v1.1. It does not implement a compactor, verifier, LLM call, or model training.

The repository may contain a temporary dataset snapshot; generated artifacts remain reproducible from the scripts.

## Prerequisites

- Python 3.11+
- `pytest` for the test suite

The source parser supports JSON and JSONL. The supplied files under `dataset/AFTraj-2K/` are streamed to JSONL with PyArrow; source Parquet files remain untouched and generated data stays ignored by Git.

## Confirmed AFTraj conventions

The current AFTraj-2K export uses 0-based `mistake_step`. Safe records have no `mistake_step`; canonical conversion writes their `unsafe_source` and `label_source` as `safe`. Malformed records are rejected and recorded in `rejected_records.jsonl`.

## Workflow

```bash
python scripts/parquet_to_jsonl.py \
  --input dataset/AFTraj-2K/aftraj_safe.parquet dataset/AFTraj-2K/aftraj_unsafe.parquet \
  --output data/raw/aftraj.jsonl \
  --manifest data/manifests/parquet_conversion.json

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
  --dataset-report data/processed/dataset_report.json \
  --mistake-step-base 0 \
  --output data/processed/aftraj_data_audit.md
```

Canonical conversion preserves exactly one `CanonicalStep` per source turn, keeps every `raw_step`, uses the confirmed 0-based AFTraj label, assigns families before splitting, and creates every prefix with `steps[:prefix_end_step + 1]` only.

## Schema migration

Migrate an existing v1.0 canonical JSONL without reading Parquet:

```bash
python scripts/migrate_schema_v1_to_v1_1.py \
  --input data/processed/aftraj_all.jsonl \
  --output data/processed/aftraj_all_v1_1.jsonl \
  --rejected data/processed/migration_rejected_records.jsonl \
  --stats-output data/manifests/schema_migration_v1_1.json

python scripts/build_splits.py \
  --input data/processed/aftraj_all_v1_1.jsonl \
  --output-dir data/processed/v1_1 \
  --manifest-dir data/manifests/v1_1 \
  --seed 42

python scripts/build_prefixes.py \
  --input-dir data/processed/v1_1 \
  --output-dir data/processed/v1_1/prefixes \
  --mode onset_window \
  --window=-3,-2,-1,0,1,2,3 \
  --seed 42

python scripts/check_migration_consistency.py \
  --before data/processed/aftraj_all.jsonl \
  --after data/processed/aftraj_all_v1_1.jsonl \
  --before-prefix-dir data/processed/prefixes \
  --after-prefix-dir data/processed/v1_1/prefixes \
  --output data/manifests/schema_migration_consistency.json
```

See [docs/canonical_schema.md](docs/canonical_schema.md) for the v1.1 field contract. Old v1.0 labels are migration input only; new generated JSONL uses `annotations.primary_error` and the structured outcome/step schemas.

## Model input boundary

`CanonicalTrajectory` is a storage and evaluation object. `ModelInputView` from `src/agent_error_compaction/views.py` is the only valid input to future Compactors and Verifiers. It contains only `task`, `tools`, and the requested prefix of whitelisted step fields. Labels, family/split identifiers, final answers, outcomes, raw steps, metadata, and diagnostic fields are reserved for `build_evaluation_view` and must not be passed to a model.
