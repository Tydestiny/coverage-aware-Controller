# Coverage-Aware Controller Development Contract

Core changes require explicit review and offline regression tests. Historical real API smoke tests do not validate later changes; the current input revision has offline coverage only.

## Stable Public Interfaces

The following files are stable core interfaces:

- `controller_agent/state/schemas.py`
- `controller_agent/integration/controlled_react_agent.py`
- `controller_agent/integration/appworld_adapter.py`
- `controller_agent/integration/base.py`
- `controller_agent/triggers/base.py`

Public contracts:

- `TriggerDetector.detect(context: TriggerContext) -> TriggerResult`
- `StateBuilder.build(task, trigger, agent_context_messages, proposed_agent_output, proposed_execution, previous_controller_decisions, context_complete, context_metadata=None) -> ControllerInput`
- `Controller.decide(controller_input: ControllerInput) -> ControllerResponse`
- `GuidanceBuilder.build(decision: ControllerDecision) -> str`
- `ControlledReActAgent.inspect_proposed_execution(...) -> InterceptionResult`
- `ExecutionInterceptionAdapter.apply(original_result, interception, agent) -> Any`
- `ExperimentLogger.log_controller_event(...) -> None`
- `ControlledAppWorldRunner.run_initialized_world(world) -> None`

## Module Inputs And Outputs

`triggers/`

- Input: `TriggerContext`
- Output: `TriggerResult`
- Constraint: inspect only model-visible runtime context and counters.

`state/`

- Input: task text, trigger result, complete captured current-task messages, proposed output/code, prior decisions, context provenance.
- Output: `ControllerInput`
- Constraint: do not include evaluator state, ground truth answer, hidden database state, or future observations.

`controller/`

- Input: `ControllerInput`
- Output: `ControllerResponse`
- Constraint: invalid JSON, backend errors, and malformed decisions fail open to `PROCEED`.

`integration/controlled_react_agent.py`

- Input: wrapped Agent result and visible execution outputs.
- Output: original Agent result or adapter-applied result.
- Constraint: never clear proposed execution code to represent a block.

`integration/appworld_adapter.py`

- Input: `InterceptionResult`, wrapped AppWorld-style Agent, initialized AppWorld world.
- Output: AppWorld runner events and runtime behavior.
- Constraint: blocked finish attempts skip `world.batch_execute`, inject marked guidance once, and advance to the next Agent generation.

`logging/`

- Input: trigger context, controller input, controller response, runtime state.
- Output: JSON-serializable controller events.
- Constraint: never log API keys, authorization headers, evaluator internals, hidden state, or raw secrets.

## Allowed Extension Points

- Add new `TriggerDetector` implementations under `controller_agent/triggers/`.
- Add state extraction helpers under `controller_agent/state/` if they only use visible Agent context.
- Add real LLM Controller backends by passing a backend callable into `Controller`.
- Add prompt templates or backend clients under `controller_agent/controller/`.
- Add additional runtime adapters by implementing `ExecutionInterceptionAdapter`.
- Add experiment scripts that consume `ControlledAppWorldRunner`.
- Add log sinks that consume events from `ExperimentLogger`.

## Pull Request Checklist

- Does the change keep the frozen core main loop intact?
- Does it avoid passing evaluator outputs, ground truth answers, hidden AppWorld state, or database internals to the Controller?
- Does it preserve fail-open behavior for Controller exceptions and malformed responses?
- Does it avoid clearing, truncating, or mutating proposed execution code as a blocking mechanism?
- Does it avoid logging API keys or authorization headers?
- Does it include mock tests that do not call real APIs?
- If it touches AppWorld runtime behavior, does it include an integration test or smoke command?
- Does `python -m pytest` pass?

## Forbidden Inputs To Controller

The Controller must not receive:

- Official evaluator result or score.
- Ground truth answer.
- Hidden benchmark state.
- Direct database snapshots unavailable to the Agent.
- Future observations or future tool outputs.
- API keys, auth headers, or provider secrets.

The Controller may receive:

- Original task text.
- Captured current-task Agent messages.
- Current proposed execution code.
- Previous visible tool outputs.
- Prior decisions linked to actually executed actions and observations.
- Runtime counters such as controller calls and blocks.

## Ownership Guidance

- Trigger work can proceed independently once it implements `TriggerDetector`.
- Controller backend work can proceed independently once it returns `ControllerResponse` or a compatible mapping.
- State extraction work can proceed independently if it uses only `ControllerInput`-safe sources.
- Logging and reporting work can proceed independently if it consumes existing event dictionaries.
- AppWorld adapter changes require extra review because they affect execution lifecycle and task completion semantics.
