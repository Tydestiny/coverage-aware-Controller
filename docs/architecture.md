# Coverage-Aware Controller Architecture

## Main Project

The repository is now centered on a single execution method:

```text
AppWorld simplified ReAct agent + Coverage-Aware Controller
```

The official AppWorld project is treated as an external dependency and dataset/runtime provider. This repository contains the controller framework and tests, not the full AppWorld source tree or experiment outputs.

## Main Loop

```text
Base simplified ReAct agent
  -> next_execution_inputs_usage_and_status(last_execution_outputs)
  -> ControlledReActAgent inspects generated execution code
  -> BeforeFinishTrigger detects supervisor.complete_task(...)
  -> StateBuilder builds ControllerInput from visible state
  -> Controller.decide(...)
     PROCEED: return original execution input
     other: block finish call and add temporary guidance
  -> AppWorld executes returned code
  -> next agent turn
```

When `enabled=False`, the wrapper returns the base agent result unchanged and does not call the controller or logger.

## Interfaces

- `TriggerResult`: trigger decision, reason, and metadata.
- `ControllerInput`: task, trigger reason, recent history, current agent output, known entities, completed operations, and recent errors.
- `ControllerDecision`: action, reason, instruction, and confidence.
- `Action`: `PROCEED`, `VERIFY_TARGET_SET`, `LOOKUP_API_SPEC`, `REPLAN`, `BLOCK_RISKY_ACTION`.

The public interfaces are:

- `TriggerDetector.detect(...)`
- `StateBuilder.build(...)`
- `Controller.decide(...)`
- `GuidanceBuilder.build(...)`
- `ExperimentLogger.log_controller_event(...)`

## Dependency Direction

```text
state.schemas <- triggers
state.schemas <- state.builder
state.schemas <- controller
state.schemas <- logging
triggers + state + controller + guidance + logging -> integration
```

The framework deliberately avoids importing official AppWorld modules in unit tests. Tests use mocks for the base agent, execution input, status, and controller backend.

## Safety Rules

- The first implementation only intercepts finish attempts.
- Ordinary tool/code steps are not modified.
- Invalid controller JSON and controller exceptions fail open to `PROCEED`.
- A blocking decision only suppresses the current finish call and gives next-turn guidance.
- Controller logs must not contain API keys.
