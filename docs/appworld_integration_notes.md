# AppWorld Integration Notes

This note records the local AppWorld call chain inspected for the minimal Coverage-Aware Controller integration spike.

Local source inspected:

```text
<APPWORLD_ROOT>
```

## Simplified ReAct Step Loop

The base loop is implemented in:

```text
experiments/code/simplified/agent.py
```

`Agent.solve_task(task_id)`:

1. Creates an `AppWorld(task_id=task_id)` context.
2. Calls `self.initialize(world)`.
3. For each step up to `max_steps`:
   - increments `self.step_number`;
   - calls `self.next_execution_inputs_usage_and_status(execution_outputs)`;
   - breaks on `status.failed`;
   - executes every returned `ExecutionIO.content` through `world.batch_execute(...)`;
   - converts environment outputs back into `ExecutionIO`;
   - records usage;
   - breaks if `world.task_completed()` or usage is exceeded.

The relevant handoff is:

```python
execution_outputs_ = world.batch_execute([input_.content for input_ in execution_inputs])
```

There is no hook between `next_execution_inputs_usage_and_status(...)` and `world.batch_execute(...)` in the stock loop.

## ReAct Message Format

The simplified ReAct agent is implemented in:

```text
experiments/code/simplified/react_code_agent.py
```

Observed message shape:

```python
{"role": "user" | "assistant" | "system", "content": str}
```

Initialization renders the prompt template and converts `USER:`, `ASSISTANT:`, and `SYSTEM:` text blocks to that list format. Environment outputs are appended as user messages:

```python
{"role": "user", "content": "Output:\n```\n...\n```\n\n"}
```

Assistant model outputs are appended as assistant messages after code extraction.

The `messages_to_text(...)` method only accepts `system`, `assistant`, and `user`. For this spike, Controller guidance uses role `user` because it is the only existing role that can carry runtime feedback into the next generation turn without changing AppWorld model formatting. The content is explicitly marked with:

```text
[CONTROLLER_GUIDANCE]
```

This marker distinguishes guidance from the original task instruction and from environment output.

## Execution and Empty Code

`AppWorld.execute(code)` is implemented in:

```text
src/appworld/environment.py
```

It strips code and parses it. Empty code is not a no-op; it records an interaction and returns:

```text
No code available to execute.
```

Therefore blocking must not be represented by setting execution code to an empty string.

## complete_task Termination

The supervisor API is implemented in:

```text
src/appworld/apps/supervisor/apis.py
```

`complete_task(...)` sets the active task `answer` and `status`, then saves it. `AppWorld.task_completed()` checks the active task status through `supervisor.show_active_task`; if the status is one of the task status literals, the stock loop stops.

## Integration Implication

The stock `Agent.solve_task` loop cannot skip one proposed execution and continue to the next agent generation turn. It always sends returned execution inputs to `world.batch_execute`.

For this spike, the repository implements a minimal runner wrapper instead of modifying AppWorld source:

```text
controller_agent/integration/appworld_adapter.py
```

The wrapper consumes `AppWorldInterceptionApplied` results:

- `allow_execution=True`: execute the original result normally.
- `allow_execution=False`: do not call `world.batch_execute`; inject marked guidance into `agent.messages`; continue to the next generation turn with no fabricated environment output.

This keeps the original `complete_task` code intact in the blocked result, avoids empty-code execution, and avoids patching AppWorld core files.
