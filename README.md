# Coverage-Aware Controller Agent

This repository now focuses on one project: a `simple_react + controller` execution method for AppWorld.

The old AFTraj diagnosis-aware compaction code is preserved on:

```cmd
obsolete/diagnosis-aware-compaction-legacy
```

## Scope

The active code implements a thin controller wrapper around a simplified ReAct-style AppWorld agent. The first controller trigger is `before_finish`: when the base agent attempts to call `supervisor.complete_task`, the wrapper can allow the finish action or block it and inject temporary guidance for the next turn.

This project does not vendor AppWorld. Keep the official AppWorld repository and raw trajectories outside Git, and connect this package to AppWorld through integration code and experiment configs.

## Layout

```text
controller_agent/
  controller/       Controller decision adapter and guidance formatting.
  triggers/         Trigger interfaces and before-finish trigger.
  state/            Dataclasses and visible-state construction.
  integration/      Controlled ReAct wrapper.
  logging/          Structured controller event logger.
  tests/            Mock-based unit tests. No real API calls.
docs/
  architecture.md   Main loop, interfaces, and dependency notes.
```

## Development

Run tests from CMD:

```cmd
cd /d D:\TY\SJTU\CIS\diagnosis-aware-compaction
python -m pytest
```

No real AppWorld task or model API call is launched by the unit tests.

## Current Contract

When controller mode is disabled, `ControlledReActAgent` delegates directly to the wrapped base agent.

When controller mode is enabled:

1. The wrapped agent generates its normal next execution input.
2. `BeforeFinishTrigger` checks whether the agent is attempting `supervisor.complete_task`.
3. `StateBuilder` builds a controller input from visible task/history/output only.
4. `Controller.decide` returns one of:
   `PROCEED`, `VERIFY_TARGET_SET`, `LOOKUP_API_SPEC`, `REPLAN`, `BLOCK_RISKY_ACTION`.
5. `PROCEED` returns the original execution input.
6. Any other action blocks the finish call and appends temporary guidance for the next turn.
