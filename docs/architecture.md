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
  -> ControlledReActAgent builds TriggerContext
  -> TriggerDetector.detect(context)
  -> if no trigger: return original result
  -> StateBuilder builds ControllerInput from visible state
  -> Controller.decide(...) returns ControllerResponse
  -> inspect_proposed_execution(...) returns InterceptionResult
  -> ExecutionInterceptionAdapter.apply(...)
     allow: return original result
     block: runtime-specific handling
  -> next agent turn
```

When `enabled=False`, the wrapper returns the base agent result unchanged and does not call the controller or logger.

The core controller loop does not mutate execution code and does not use empty execution inputs as a blocking signal.

## Interfaces

- `TriggerResult`: trigger decision, reason, and metadata.
- `TriggerContext`: task id, step index, task text, proposed execution, last execution outputs, recent history, and runtime counters.
- `ControllerInput`: task, trigger reason, recent history, current agent output, known entities, completed operations, and recent errors.
- `ControllerDecision`: action, reason, instruction, and confidence.
- `ControllerResponse`: decision plus optional model name, input/output tokens, latency, cost, and raw response.
- `ControllerRuntimeState`: total calls, total blocks, consecutive blocks, last trigger step, and last action.
- `ControllerConfig`: enabled flag, max calls, max consecutive blocks, and fail-open setting.
- `InterceptionResult`: runtime-neutral allow/block result, guidance, trigger, controller response, and block reason.
- `Action`: `PROCEED`, `VERIFY_TARGET_SET`, `LOOKUP_API_SPEC`, `REPLAN`, `BLOCK_RISKY_ACTION`.

The public interfaces are:

- `TriggerDetector.detect(context)`
- `StateBuilder.build(...)`
- `Controller.decide(...)`
- `GuidanceBuilder.build(...)`
- `ControlledReActAgent.inspect_proposed_execution(...)`
- `ExecutionInterceptionAdapter.apply(original_result, interception, agent)`
- `ExperimentLogger.log_controller_event(...)`

`Controller.decide` is backward compatible with legacy backend mappings such as:

```json
{"action": "PROCEED", "reason": "verified", "confidence": 1.0}
```

It wraps those mappings into `ControllerResponse`. Invalid JSON and backend exceptions return a fail-open `PROCEED` response with reason `controller_failure_fallback`.

## Dependency Direction

```text
state.schemas <- triggers
state.schemas <- state.builder
state.schemas <- controller
state.schemas <- logging
state.schemas <- integration.base
triggers + state + controller + guidance + logging + integration.base -> integration.controlled_react_agent
```

The framework deliberately avoids importing official AppWorld modules in unit tests. Tests use mocks for the base agent, execution input, status, and controller backend.

## Integration Boundary

The core layer is runtime-neutral. It only returns `InterceptionResult`.

`ExecutionInterceptionAdapter` owns the concrete runtime behavior. The current `OfflineMockAdapter` is intentionally conservative:

- `allow_execution=True`: returns the original agent result.
- `allow_execution=False`: returns an explicit `integration not configured` result.
- It never clears execution code or attempts to simulate AppWorld blocking.

A production AppWorld adapter still needs to define how a blocked finish attempt is represented to the official runtime without accidentally executing partial or empty code.

## Safety Rules

- The first implementation only intercepts finish attempts.
- Ordinary tool/code steps are not modified.
- Invalid controller JSON and controller exceptions fail open to `PROCEED`.
- A blocking decision is represented as `InterceptionResult`; the core layer does not implement AppWorld-specific suppression.
- Controller logs must not contain API keys.
- New triggers, real controller prompts, real model backends, and AppWorld runtime adapters should be implemented after these contracts are reviewed and frozen.
