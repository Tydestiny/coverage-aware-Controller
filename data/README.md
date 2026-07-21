# Data Layout

`raw/` contains source exports. `processed/` contains canonical trajectories, splits, prefixes, reports, and rejected records. `manifests/` contains conversion and split ID manifests. `samples/` is for small synthetic, non-source examples only.

The supplied `dataset/AFTraj-2K/*.parquet` files are intentionally untouched. This standard-library pipeline accepts JSON and JSONL exports. Run `scripts/inspect_aftraj.py` on an exported source file before selecting `--mistake-step-base`; the script reports evidence but does not infer 0- versus 1-based labels.

