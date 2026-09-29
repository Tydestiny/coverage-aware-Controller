# AppWorld Controller Agent

This repository contains the Agent + event-triggered Controller implementation. The official AppWorld checkout and dataset remain in the sibling `appworld` repository. Historical G0/G1/G2 prompts, test trajectories, checkpoints and reports are isolated in `D:\TY\SJTU\CIS\controller_study_archive_2026`; see [migration map](docs/archive_migration.md). New work is **train/dev only** until a separate final-test protocol is frozen.

For runtime internals, see [Controller architecture](docs/controller_architecture.md).

The initial [train_dev_v1 prompt](prompts/train_dev/train_dev_v1.txt) is ready for train development. See its [version record and dry-run command](prompts/train_dev/README.md) and the [experiment plan](docs/research_protocol.md). Its model behavior has not yet been evaluated. No historical prompt is an implicit fallback.

## Fixed Development Tasks

`configs/studies/train_dev_v1.json` fixes eight train tasks and eight reserved dev tasks using official metadata, seed `20260722`, and disjoint task families. This is a prospective development reservation, not a claim that nobody previously encountered these tasks. The manifest is a reference configuration, not automatically loaded by `run`.

```cmd
python scripts\experiment.py prepare
python scripts\experiment.py run --split train --task-id 3c13f5a_2 --experiment-id train_dev_v1_smoke --branch baseline --agent-model qwen3-coder-flash-2025-07-28
```

`prepare` refuses to overwrite a different existing selection. The second command is a dry-run. Use the explicit train_dev_v1 prompt path for the first Controller trials. Initial runs now attach the existing dense checkpoint hook; the run manifest points to checkpoints alongside raw AppWorld task logs. The seed is forwarded to both Agent and environment. Existing execution safeguards remain explicit in the study configuration: 10,000 Agent/environment/extra turns, a repeated-code/output stop, and per-call output limits. Any such termination must be reported, not called normal completion.

## Where Things Live

```text
appworld-controller-agent/
  controller_agent/
    controller/      Controller backend, decisions, guidance and input rendering
    triggers/        before_finish, before_write, no_progress
    state/           Model-visible ControllerInput construction
    integration/     Existing AppWorld Agent and runner integration
    branching/       Checkpoint capture, restore, continuation logging
    evaluation/      Official evaluator subprocess wrapper
    results/         Token and experiment result readers
    tests/           Offline unit and integration tests
  scripts/
    experiment.py    Only recommended human-facing train/dev CLI
    run_live_controller_initial_task.py    Guarded underlying run
    run_checkpoint_dual_branch_smoke.py    Guarded paired checkpoint run
    run_official_evaluation.py             Official offline evaluation
    analyze_branch_experiment.py           Paired-result analysis
  artifacts/experiments/  New local runs (Git ignored)
```

`APPWORLD_ROOT` defaults to `..\appworld`. `CONTROLLER_ARTIFACT_ROOT` defaults to `artifacts\experiments`. Set either environment variable to override; never set it to the historical archive. `.env.example` lists the environment variables, but does not contain credentials. `doctor` reports whether provider keys are present in the current process and whether `<appworld>\experiments\.env` exists; it never prints key values. The underlying task runner reads that AppWorld `.env` at execution time.

`controller_agent/` contains reusable runtime components, not user commands. Most other scripts are compatibility entry points for older experiments; several import each other, so they are not safe to delete by filename. Use `scripts\experiment.py` for new train/dev work. See [research protocol](docs/research_protocol.md) for the fixed research question and evaluation boundary.
The [script inventory](docs/script_inventory.md) distinguishes supported commands from retained implementation dependencies.

## Start Here (Windows CMD)

```cmd
cd /d D:\TY\SJTU\CIS\appworld-controller-agent
python scripts\experiment.py doctor
python -m pytest
python scripts\experiment.py list --archive-root D:\TY\SJTU\CIS\controller_study_archive_2026
```

`doctor` does not call an API. It tests an actual AppWorld import using `<appworld>\.venv\Scripts\python.exe`, which is also used for task execution. The controller repo's old `.venv` currently points to a removed interpreter, so use a working Python 3.11+ for `python -m pytest`. The test suite uses mocks and does not run AppWorld tasks.

## One Train/Dev Task

Pick an ID from `D:\TY\SJTU\CIS\appworld\data\datasets\dev.txt` or `train.txt`. The default is **dry-run**, which checks the official task list and prints paths without creating a result or calling a model:

```cmd
python scripts\experiment.py run --split dev --task-id <DEV_TASK_ID> --experiment-id dev_probe_v1 --branch baseline --agent-model qwen3-coder-flash-2025-07-28
```

Run only after reviewing the printed command and providing the required credentials in the environment:

```cmd
python scripts\experiment.py run --split dev --task-id <DEV_TASK_ID> --experiment-id dev_probe_v1 --branch baseline --agent-model qwen3-coder-flash-2025-07-28 --execute
```

A Controller run requires an **independently developed train/dev prompt file**. No historical prompt is a default:

```cmd
python scripts\experiment.py run --split dev --task-id <DEV_TASK_ID> --experiment-id dev_probe_v1 --branch controller --agent-model qwen3-coder-flash-2025-07-28 --controller-model deepseek-v4-pro --prompt prompts\train_dev\candidate_v1.txt
python scripts\experiment.py run --split dev --task-id <DEV_TASK_ID> --experiment-id dev_probe_v1 --branch controller --agent-model qwen3-coder-flash-2025-07-28 --controller-model deepseek-v4-pro --prompt prompts\train_dev\candidate_v1.txt --execute
```

The run records a SHA-256 prompt hash. Results are under `artifacts\experiments\<experiment-id>\tasks\<task-id>\<branch>`. Reusing the same run directory is blocked; choose a new experiment ID rather than overwriting a result. Raw AppWorld task logs are written under the AppWorld checkout with the experiment name recorded in each run manifest.

Use `--trigger-profile before_finish` to restrict Controller opportunities to task completion, or the default `all` for `before_finish`, `before_write`, and `no_progress`. `--max-controller-calls` and `--max-controller-blocks` default to 4 and are recorded with the run. Trigger detection is logged for every executable candidate in `trigger_events.jsonl`, including positive detections bypassed after a budget is exhausted. Model, seed, and Controller endpoint can be selected from the same command.

## One Shared Checkpoint Pair

Use a post-observation checkpoint from a **train/dev baseline produced by this CLI**. The source task directory must match its active baseline manifest under `artifacts\experiments`; archived or unrelated task outputs are rejected. This command runs a no-Controller continuation and a Controller continuation from independent state copies. It is a dry-run unless `--execute` is supplied:

```cmd
python scripts\experiment.py run checkpoint --split dev --task-id <DEV_TASK_ID> --experiment-id dev_checkpoint_v1 --agent-model qwen3-coder-flash-2025-07-28 --controller-model deepseek-v4-pro --prompt prompts\train_dev\candidate_v1.txt --checkpoint-id <CHECKPOINT_ID> --source-task-dir <BASELINE_TASK_DIR>
```

Review the printed paths, then rerun the same command with `--execute`. The paired runner performs official offline evaluation and writes both branches under `artifacts\experiments\dev_checkpoint_v1\tasks\<DEV_TASK_ID>`. A nonexistent or mismatched checkpoint is rejected; it is never silently replaced by a nearby one. Use a new experiment ID for each pair. This entry point intentionally supports one checkpoint pair per experiment; existing batch collectors remain separate internal tools.

## Evaluate, Analyze, Find

Evaluation uses AppWorld's official evaluator in a separate process and is not an LLM call. First preview, then execute:

```cmd
python scripts\experiment.py evaluate --experiment-id dev_probe_v1 --task-id <DEV_TASK_ID> --branch baseline
python scripts\experiment.py evaluate --experiment-id dev_probe_v1 --task-id <DEV_TASK_ID> --branch baseline --execute
python scripts\experiment.py analyze --experiment-id dev_probe_v1
python scripts\experiment.py list
```

`analyze` writes `analysis\summary.json` and `analysis\data_quality.json`. It pairs only available baseline and Controller runs; a one-branch probe will be flagged as missing a pair. The official evaluator result is saved next to the run summary as `evaluator_result.json`.

The AppWorld checkout remains a separate upstream Git repository and hosts raw runtime logs. The Controller repo's `artifacts\experiments` is the user-facing experiment index, not a second copy of the full raw trajectory. Preserve manifest links when archiving a run.

## Research Boundary

Develop prompts and adjustable settings using **train/dev only**. Reserve dev tasks for a predeclared runtime/quality gate. Do not feed G2, archived test logs, evaluator hidden data, or historical oracle guidance into prompt development. Before any final test run, freeze the prompt hash, code, models, seed, sampling, and analysis method. See the single maintained [experiment plan](docs/research_protocol.md).
