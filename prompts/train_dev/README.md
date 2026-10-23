# Train/Dev Prompts

Store newly written Controller system prompts here. Name each candidate explicitly (for example `candidate_v1.txt`), record its SHA-256 in the run manifest, and freeze the selected file before any final test evaluation.

## train_dev_v1

- File: `train_dev_v1.txt` (UTF-8, English).
- Created: 2026-09-29. Status: initial development candidate; not performance-validated or frozen for final evaluation.
- SHA-256 of file bytes: `4ae520e75984232bf7d8e18ca8282897e87f4aa4413f500b8e7c5822d04bb597`.
- Basis: the design discussion now consolidated in `docs/research_protocol.md`, the ControllerInput/Action contract, and guidance delivery behavior. No task examples or historical prompt/trajectory files were used to draft this file. This records development inputs, not a claim of historically unexposed authors or models.
- Hypothesis: evidence-based local corrections and explicit permission to PROCEED can reduce unnecessary intervention while preserving useful corrections. The fixed eight-task train pilot is complete: baseline 1/8 and Controller 3/8 exact success. These development results do not establish general effectiveness or individual-intervention causality.
- Response: exactly `action`, `reason`, `instruction`; PROCEED has an empty instruction. Other actions supply standalone advice for the next Agent generation.
- Validation scope: offline tests, eight paired train trajectories, official evaluation and per-call audit. Mock responses cannot establish how a real model follows this prompt.

From the repository root, preview the first fixed train Controller run:

```cmd
python scripts\experiment.py run --split train --task-id 3c13f5a_2 --experiment-id train_dev_v1_smoke --branch controller --agent-model qwen3-coder-flash-2025-07-28 --controller-model deepseek-v4-pro --prompt prompts\train_dev\train_dev_v1.txt --seed 20260722 --max-controller-calls 4 --max-controller-blocks 4
```

The command is a dry-run; adding `--execute` makes real model requests. The study-selection manifest remains unchanged and is not loaded automatically by this CLI. Select this prompt explicitly. Preserve this candidate when iterating: use a new version filename and record its hypothesis and hash.

## train_dev_v2

- File: `train_dev_v2.txt`. Created: 2026-10-02. Development candidate only.
- SHA-256: `75510166d97685c98db226a983812def3e487ae4edcffff10726037788fae2b3`.
- Only change: an explicit task-condition evidence standard. Check the requested scope, target predicates and confirmed execution; do not invent stricter requirements or accept unsupported proxies. Guidance names a specific remaining check rather than repeating generic advice.
- Basis: the eight v1 train pairs and their visible action/observation audit. No archived prompts, test trajectories, hidden answers or reserved dev results were used.
- Unchanged: Agent/Controller models, input schema/rendering, actions, triggers, decoding, seed and four-call/four-block policy.
- Experiment: `train_dev_v2_evidence`; reuse the original eight baselines read-only and run one new Controller trajectory per task. Preserve all failures and format errors. This is not a held-out comparison or an estimate of within-condition variance.
- Completed result: 3/8 exact success, unchanged from v1; macro requirement completion 66.3% versus v1's 68.3%. Combined tokens increased from 1,464,804 to 1,676,270. This candidate has not shown an additional benefit. Three account-arrearage interruptions were restarted after recharge; task failures were not retried.

```cmd
python scripts\experiment.py run --split train --task-id 3c13f5a_2 --experiment-id train_dev_v2_evidence --branch controller --agent-model qwen3-coder-flash-2025-07-28 --controller-model deepseek-v4-pro --prompt prompts\train_dev\train_dev_v2.txt --seed 20260722 --max-controller-calls 4 --max-controller-blocks 4
```

Add `--execute` only for a real run. Evaluate each finished run with `experiment.py evaluate`, then use `experiment.py analyze --experiment-id train_dev_v2_evidence` for the referenced-baseline comparison.
