# Train/Dev Prompts

Store newly written Controller system prompts here. Name each candidate explicitly (for example `candidate_v1.txt`), record its SHA-256 in the run manifest, and freeze the selected file before any final test evaluation.

## train_dev_v1

- File: `train_dev_v1.txt` (UTF-8, English).
- Created: 2026-09-29. Status: initial development candidate; not performance-validated or frozen for final evaluation.
- SHA-256 of file bytes: `4ae520e75984232bf7d8e18ca8282897e87f4aa4413f500b8e7c5822d04bb597`.
- Basis: the design discussion now consolidated in `docs/research_protocol.md`, the ControllerInput/Action contract, and guidance delivery behavior. No task examples or historical prompt/trajectory files were used to draft this file. This records development inputs, not a claim of historically unexposed authors or models.
- Hypothesis: evidence-based local corrections and explicit permission to PROCEED can reduce unnecessary intervention while preserving useful corrections. This remains untested.
- Response: exactly `action`, `reason`, `instruction`; PROCEED has an empty instruction. Other actions supply standalone advice for the next Agent generation.
- Validation scope: offline loading, parsing and CLI compatibility only. Mock responses cannot establish how a real model follows this prompt.

From the repository root, preview the first fixed train Controller run:

```cmd
python scripts\experiment.py run --split train --task-id 3c13f5a_2 --experiment-id train_dev_v1_smoke --branch controller --agent-model qwen3-coder-flash-2025-07-28 --controller-model deepseek-v4-pro --prompt prompts\train_dev\train_dev_v1.txt --seed 20260722 --max-controller-calls 4 --max-controller-blocks 4
```

The command is a dry-run; adding `--execute` makes real model requests. The study-selection manifest remains unchanged and is not loaded automatically by this CLI. Select this prompt explicitly. Preserve this candidate when iterating: use a new version filename and record its hypothesis and hash.
