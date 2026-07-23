# Coverage-Aware Controller

This repository contains the controller runtime for AppWorld:

```text
AppWorld simplified ReAct Agent + Coverage-Aware Controller + dense checkpoints
```

The repository used to be named `diagnosis-aware-compaction`, but the active
project is now `coverage-aware-controller`. The old AFTraj diagnosis-aware
compaction code is preserved on:

```cmd
obsolete/diagnosis-aware-compaction-legacy
```

## Scope

The active code implements a thin controller wrapper around a simplified ReAct-style AppWorld agent. The first controller trigger is `before_finish`: when the base agent attempts to call `supervisor.complete_task`, the wrapper can allow the finish action or return an explicit interception decision.

This project does not vendor AppWorld. Keep the official AppWorld repository and raw trajectories outside Git, and connect this package to AppWorld through an `ExecutionInterceptionAdapter`.

Recommended local workspace:

```text
workspace/
  appworld/                    # official AppWorld repository and task outputs
  coverage-aware-controller/   # this repository
```

On the current machine these are:

```text
D:\TY\SJTU\CIS\appworld
D:\TY\SJTU\CIS\coverage-aware-controller
```

## Layout

```text
controller_agent/
  controller/       Controller decision adapter and guidance formatting.
  triggers/         Trigger interfaces and before-finish trigger.
  state/            Dataclasses and visible-state construction.
  integration/      Controlled ReAct wrapper and interception adapter boundary.
  logging/          Structured controller event logger.
  branching/        Runtime checkpoint capture and static replay prototypes.
  tests/            Mock-based unit tests. No real API calls.
docs/
  architecture.md   Main loop, interfaces, and dependency notes.
  appworld_integration_notes.md
scripts/
  run_controller_api_smoke.py  Single-task controlled AppWorld runner.
  run_train_part.cmd           CMD template for per-member task collection.
task_ids/
  train_parts/                 Five train task-id splits for collection.
```

## Development

Run tests from CMD:

```cmd
cd /d D:\TY\SJTU\CIS\coverage-aware-controller
python -m pytest
```

No real AppWorld task or model API call is launched by the unit tests.

## Trajectory Collection

Each member can run one task-id part file. The model and Agent configuration are
fixed in `scripts/run_controller_api_smoke.py`:

- model: `qwen3-coder-flash-2025-07-28`
- endpoint: `https://dashscope.aliyuncs.com/compatible-mode/v1`
- agent: `simplified_react_code_agent`
- temperature: `0.0`
- seed: `100`
- thinking: disabled

Required local secret:

```text
<appworld>\experiments\.env
```

with:

```text
DASHSCOPE_API_KEY=...
```

Run a split from CMD:

```cmd
cd /d D:\TY\SJTU\CIS\coverage-aware-controller
scripts\run_train_part.cmd task_ids\train_parts\train_task_ids_part_1.txt
```

Outputs are task-local under AppWorld:

```text
<appworld>\experiments\outputs\controlled_react_checkpoint_collection\tasks\<task_id>\
  logs\
  evaluation\
  controller\
  checkpoints\
```

The `checkpoints\manifest.json` file indexes all `pre_action`,
`post_observation`, and `terminal` records for that task.

## Current Contract

When controller mode is disabled, `ControlledReActAgent` delegates directly to the wrapped base agent.

When controller mode is enabled:

1. The wrapped agent generates its normal next execution input.
2. `ControlledReActAgent` builds a `TriggerContext` from task id, step index, visible history, proposed execution, last execution outputs, and controller runtime counters.
3. `BeforeFinishTrigger.detect(context)` checks whether the proposed execution attempts `supervisor.complete_task`.
4. `StateBuilder` builds a `ControllerInput` from visible task/history/output only.
5. `Controller.decide` returns a `ControllerResponse` containing a `ControllerDecision` and optional usage metadata.
6. `ControllerDecision.action` remains one of:
   `PROCEED`, `VERIFY_TARGET_SET`, `LOOKUP_API_SPEC`, `REPLAN`, `BLOCK_RISKY_ACTION`.
7. `inspect_proposed_execution(...)` returns an `InterceptionResult`.
8. The integration adapter decides how to apply that result to the concrete runtime.

The core layer does not implement AppWorld-specific blocking semantics. In the current offline adapter, a blocked branch returns `integration not configured` and keeps the proposed execution unchanged. A production AppWorld adapter must define the exact runtime behavior before real tasks are run.

The minimal AppWorld integration spike adds `AppWorldInterceptionAdapter` and `ControlledAppWorldRunner`. The runner wraps the stock AppWorld step loop so a blocked `complete_task` is not sent to `world.batch_execute`; instead, marked guidance is injected and the agent advances to the next generation turn.

## Current Boundaries

- Trigger layer: detects whether a controller call should happen.
- State layer: defines contracts and builds visible controller inputs.
- Controller layer: validates backend output and fail-opens on invalid JSON or exceptions.
- Integration layer: adapts `InterceptionResult` to a concrete runtime.
- Logging layer: records controller events, usage metadata, and runtime counters without API keys.

Only `before_finish` is implemented. New triggers, a real controller model backend, controller prompts, and AppWorld-specific interception behavior should be implemented after these interfaces are treated as stable.

## AppWorld Smoke Path

The current smoke integration is deterministic and model-free:

- `DeterministicSmokeController` blocks the first `before_finish` and allows the second.
- `AppWorldInterceptionAdapter` injects `[CONTROLLER_GUIDANCE]` as a `user` role message because the simplified ReAct formatter only supports `system`, `assistant`, and `user`.
- `ControlledAppWorldRunner` is the minimal runner wrapper that skips environment execution for blocked finish attempts.

The inspected AppWorld call chain is documented in [docs/appworld_integration_notes.md](docs/appworld_integration_notes.md).

A real single-task smoke test should use a cached or mock model output source first. Do not run a live API-backed AppWorld task until the production adapter behavior is reviewed.
