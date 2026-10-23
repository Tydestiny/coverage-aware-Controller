# AppWorld Controller Agent

An event-triggered supervisor for tool-using ReAct agents in [AppWorld](https://github.com/StonyBrookNLP/appworld). The Controller reviews proposed actions and either approves them or provides guidance before the Agent generates a new action.

The project supports baseline/Controller comparisons, checkpoint replay, official evaluation, token accounting, and a local trajectory reviewer.

## Design

```text
Agent proposal -> Trigger -> Controller -> approve -> AppWorld execution
                                |
                                +-> block + guidance -> new Agent proposal
```

- Triggers: `before_finish`, `before_write`, and `no_progress`.
- Decisions: `PROCEED`, `VERIFY_TARGET_SET`, `LOOKUP_API_SPEC`, `REPLAN`, and `FIX_EXECUTION`.
- Input: captured current-task messages, proposed output and code, trigger information, and previous decisions. Static few-shot examples are excluded; incomplete context is flagged.
- Controller errors and exhausted intervention budgets fail open. Approved actions execute once; blocked actions do not reach AppWorld.
- Evaluation uses the official AppWorld evaluator in a separate process.

See [architecture](docs/controller_architecture.md) and [interface contracts](docs/development_contract.md).

## Setup

Use Python 3.11+ for this repository. Install AppWorld and its official data separately using its upstream instructions. From this repository:

```cmd
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
```

Default checkout layout:

```text
workspace/
  appworld/                  Official checkout, data and execution environment
  appworld-controller-agent/ This repository
```

Set `APPWORLD_ROOT` if AppWorld is elsewhere. `CONTROLLER_ARTIFACT_ROOT` optionally overrides the default `artifacts/experiments` result directory.

`.env.example` lists supported variables. Supply credentials through the environment or AppWorld's `experiments/.env`, which the runtime loads. A repository-root `.env` is not automatically loaded. Never commit credentials or private trajectories.

```cmd
python scripts\experiment.py doctor
python -m pytest
```

`doctor` checks the interpreter, AppWorld import, task lists and credential availability without model calls. Real execution uses AppWorld's `.venv` when present, otherwise the current interpreter. Tests use mocks and make no real model calls.

If Windows reports long temporary paths, run `python -m pytest --basetemp=.pytest_cache/test_tmp` to keep test files under the repository. Browser tests additionally use Playwright and Chrome when installed; otherwise they are skipped.

## Run

`scripts/experiment.py` is the recommended entry point. Use `--help` or `run --help` for options. Run commands preview their work unless `--execute` is supplied.

Choose a task from AppWorld's `data/datasets/train.txt` or `dev.txt`. For example:

```cmd
python scripts\experiment.py run --split train --task-id 3c13f5a_2 --experiment-id train_example --branch baseline --agent-model qwen3-coder-flash-2025-07-28
python scripts\experiment.py run --split train --task-id 3c13f5a_2 --experiment-id train_example --branch controller --agent-model qwen3-coder-flash-2025-07-28 --controller-model deepseek-v4-pro --prompt prompts\train_dev\train_dev_v1.txt
```

Append `--execute` to each command to run it. Model IDs must be supported by your provider and AppWorld registry. The included prompts are development candidates, not a claim of validated performance.

Keep the task, Agent model and seed identical across conditions. Defaults are seed `20260722`, four Controller calls and four blocks per task. `--trigger-profile before_finish` restricts review to completion; `all` enables all three triggers. Execution safeguards and output limits remain explicit in the runtime/study configuration. Report budget termination separately from task completion. Existing output directories are protected from overwrite.

For a shared-checkpoint pair, use a post-observation checkpoint from an active train/dev baseline produced by this CLI:

```cmd
python scripts\experiment.py run checkpoint --split train --task-id 3c13f5a_2 --experiment-id checkpoint_example --checkpoint-id <CHECKPOINT_ID> --source-task-dir <BASELINE_TASK_DIR> --agent-model qwen3-coder-flash-2025-07-28 --controller-model deepseek-v4-pro --prompt prompts\train_dev\train_dev_v1.txt
```

Each continuation restores an independent state copy. This command supports one checkpoint pair per experiment and performs official evaluation when executed.

## Evaluate and review

```cmd
python scripts\experiment.py evaluate --experiment-id train_example --task-id 3c13f5a_2 --branch baseline --execute
python scripts\experiment.py evaluate --experiment-id train_example --task-id 3c13f5a_2 --branch controller --execute
python scripts\experiment.py analyze --experiment-id train_example
python scripts\experiment.py list
python scripts\experiment.py review --serve
```

Open **http://127.0.0.1:8765** to select experiments and inspect actions, observations and Controller calls. Annotations save to each run's `review_annotations.json`. Keep the terminal open; Ctrl+C stops the reviewer. Use `--port` to change its port.

Analysis writes `analysis/summary.json`, paired task reports and `analysis/data_quality.json`. Missing or ambiguous pairs are reported rather than silently counted. Agent and Controller tokens are separate; unavailable provider usage remains missing.

## Layout

```text
controller_agent/
  controller/    Backend, decision parsing, input rendering and guidance
  triggers/      Rule-based detection and budget checks
  state/         Captured context and ControllerInput
  integration/   ReAct interception and AppWorld runner
  branching/     Checkpoint capture, replay and continuation logs
  evaluation/    Official evaluator subprocess
  results/       Result readers, token ledgers and local reviewer
  tests/         Offline regression and integration tests
scripts/         CLI and execution/analysis drivers
configs/         Study settings and fixed task selections
prompts/         Train/dev prompt versions
docs/            Architecture, contracts and research protocol
artifacts/       Local results; excluded from Git
```

Results are indexed under `artifacts/experiments/<experiment_id>/tasks/<task_id>/<branch>/`. Run manifests identify raw AppWorld logs and checkpoints. Preserve those references when moving experiments. Prefix state, branch continuation and evaluator results retain separate provenance.

Some internal drivers retain names containing `smoke`; supported workflows still import them. See the [script inventory](docs/script_inventory.md) before removing them.

## Research protocol

Develop prompts and rules on train/dev. `configs/studies/train_dev_v1.json` fixes eight train and eight reserved dev tasks; it is not automatically loaded by `run`. Freeze prompt hash, code, models, seed, selection and analysis before final evaluation. Historical test data and prompts are archived outside the active repositories.

End-to-end comparisons measure condition differences. Shared-checkpoint experiments investigate individual interventions. These are different forms of evidence. See [research protocol](docs/research_protocol.md).
