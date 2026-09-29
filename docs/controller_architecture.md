# Controller Architecture

## Scope

This document describes the current AppWorld Coverage-Aware Controller runtime. The Controller observes an Agent proposal, decides whether the proposal may execute, and may provide one temporary guidance message. The Agent remains the only component that executes AppWorld APIs or calls `apis.supervisor.complete_task(...)`.

The main integration entry points are:

- `controller_agent/integration/controlled_react_agent.py`
- `controller_agent/integration/appworld_adapter.py`
- `controller_agent/controller/live_llm_backend.py`
- `controller_agent/state/schemas.py`

## Runtime Flow

```text
AppWorld simplified ReAct Agent generates assistant output
  -> proposed Python execution is parsed, but not yet executed
  -> CompositeTriggerDetectorWithBudget.detect(TriggerContext)
  -> no trigger: execute original proposal once
  -> trigger: StateBuilder constructs ControllerInput
  -> Controller / LiveLLMBackend returns validated JSON decision
  -> PROCEED: execute original proposal once
  -> non-PROCEED: block original proposal, inject guidance, request a new Agent turn
  -> AppWorld executes only Agent-generated code and returns observation
  -> observation and Controller history are recorded for later turns
```

The wrapper is transparent when the Controller is disabled: the wrapped simplified ReAct Agent proceeds through its original execution path.

## Trigger Layer

The runtime calls only `CompositeTriggerDetectorWithBudget.detect(context)`. The current experiment enables these deterministic triggers in priority order:

1. `before_finish`: detects a proposed `supervisor.complete_task(...)` call.
2. `before_write`: detects a proposed high-risk mutating API call, such as create, update, delete, send, pay, add, remove, or follow.
3. `no_progress`: detects repeated actions or repeated errors from recent executed action/observation pairs.

A trigger determines when the Controller is consulted. It does not require intervention. Trigger metadata is passed unchanged into `ControllerInput` and the audit log.

Every executable candidate is recorded in `trigger_events.jsonl`, including detector-positive candidates bypassed by the trigger budget. `trigger_budget_admitted` does not imply a Controller call: the separate Controller call or block guard may still fail open. Actual Controller decisions remain in `controller_events.jsonl`.

## Controller Input

`ControllerInput` is the complete causal input for one Controller decision:

```python
ControllerInput(
    task: str,
    trigger_reason: str,
    trigger_metadata: dict[str, Any],
    agent_context_messages: list[dict[str, Any]],
    proposed_agent_output: str,
    proposed_execution: str,
    previous_controller_decisions: list[dict[str, Any]],
    context_complete: bool,
    context_metadata: dict[str, Any] = {},
)
```

### Visible Context Boundary

`agent_context_messages` is captured immediately before the Agent model call, then deep-copied. It contains the full current-task runtime message sequence that the Agent sees:

- current task instruction;
- Agent assistant outputs;
- AppWorld observations;
- previously injected `[CONTROLLER_GUIDANCE]` messages;
- results of actions taken after prior guidance.

The Agent's static system/instruction preamble and AppWorld few-shot demonstration are structurally excluded from Controller input. The Agent itself still receives that preamble unchanged. No content-keyword filtering or history summarization is used.

`context_complete` is true only when the task boundary is verified, the captured task messages equal the Agent's stored task history, no truncation marker occurs, and one generation was captured. An unknown boundary preserves the unfiltered request and sets false. Cached pre-action input is marked unverified, not presented as a live capture.

`context_metadata` records source, task/step, message counts, request/context SHA-256 and incomplete reasons. Capture occurs at `language_model.generate`; the client may subsequently add cache controls or transform messages. `provider_payload_verified=false` makes this boundary explicit. Hashes identify content, not unique model calls. The optional field preserves existing constructors.

The model sees only scope, step, boundary status, incomplete reasons and proposal status from this metadata. Hashes, capture counts and client provenance stay in the audit records to avoid irrelevant input tokens.

`previous_controller_decisions` retains each original proposal and its status: `blocked`, `allowed_pending_execution`, or `executed`. `execution_result` belongs only to an executed original proposal. Later actions and observations go in `subsequent_executions`, with their actual step numbers. No automatic claim is made that guidance was followed or the problem was solved. The Runner records feedback immediately after execution, including the terminal action; next-turn delivery remains supported for direct wrapper users.

### Serialized Prompt Order

The user prompt uses this fixed order:

1. `task`
2. `trigger_reason`
3. `trigger_metadata`
4. `context_complete`
5. `context_metadata`
6. `agent_context_messages`
7. `previous_controller_decisions`
8. `proposed_agent_output`
9. `proposed_execution`

The order is implemented by `controller_agent/controller/input_renderer.py` and rendered by `render_controller_input`.

The Controller never receives evaluator output, ground truth, future trajectory events, or hidden AppWorld state.

## Actions

The validated action taxonomy is defined by `Action` in `controller_agent/state/schemas.py`.

| Action | Effect |
| --- | --- |
| `PROCEED` | Execute the original proposed code once. Guidance must be empty. |
| `VERIFY_TARGET_SET` | Block the proposal and ask the Agent to verify targets, coverage, mapping, preservation, outputs, or final state before retrying. |
| `LOOKUP_API_SPEC` | Block the proposal and ask the Agent to inspect unknown API semantics that are required for a correct next step. |
| `FIX_EXECUTION` | Block the proposal and give a visible-evidence-grounded local repair for a valid strategy, such as token flow, API arguments, variable use, parsing, or control flow. |
| `REPLAN` | Block the proposal and redirect an incorrect task-level objective, evidence source, population, scope, computation, operation, destination, or overall strategy. |

There is no confidence field in the Controller contract. `BLOCK_RISKY_WRITE` is not an active action.

## System Prompt Boundary

There is no active default system prompt. Historical G0/G1/G2 prompts were archived outside the repository. A new Controller run must select a versioned train/dev prompt explicitly. The response contract is a JSON decision with `action`, `reason`, and `instruction`; prompt content is developed separately using train/dev only.

The Live LLM backend uses temperature `0`, limits Controller output to 512 tokens, validates the JSON against the action enum, and removes any returned `confidence` field before parsing or logging. For DashScope DeepSeek V4 models, it sends `enable_thinking=false`.

## Blocking and Guidance

For a non-`PROCEED` decision:

1. The original execution is not sent to `world.batch_execute`.
2. The original code is retained in audit events; it is not cleared or replaced with empty code.
3. `GuidanceBuilder` injects one marked user message:

```text
[CONTROLLER_GUIDANCE]
Action: <ACTION>
Advice: <instruction>
```

4. The Agent generates a new output on the next turn.

Guidance is deduplicated per task. A blocked proposal is recorded as a separate Agent attempt, so the resulting trajectory preserves why the trigger fired and what was blocked.

## Fail-Open and Limits

The outer loop fail-opens and executes the original proposal when trigger detection, state construction, Controller calling, response parsing, or schema validation fails. It also fail-opens once the configured Controller call or block limit is exhausted.

For current end-to-end experiments, the Controller limit is four calls and four blocks per task. Agent/environment limits are configured high to avoid normal trajectory truncation. A Controller failure must not turn into synthetic AppWorld execution code or a fabricated environment error.

## Audit Outputs

Each run retains structured evidence for later analysis:

- `continuation.jsonl` or `turn_events.jsonl`: Agent proposals, execution decisions, observations, and termination;
- `controller_calls.jsonl` or `controller_events.jsonl`: full ControllerInput, ControllerResponse, action, reason, guidance, blocking, and token usage when available;
- `trigger_events.jsonl`: one candidate per line with positive, admitted, and budget-bypassed flags;
- `token_usage.jsonl`: per-call Agent and Controller token ledger where collected;
- `result.json` and `run_summary.json`: run-level termination and token summaries;
- `evaluator_result.json`: result from the separate official AppWorld evaluator process;
- raw AppWorld logs for detailed audit.

The official evaluator is outside the Controller runtime and is never passed into ControllerInput.
