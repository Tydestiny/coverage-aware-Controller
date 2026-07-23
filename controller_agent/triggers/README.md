# triggers/

Trigger modules decide whether a proposed Agent action should be inspected by the Controller.

Stable entry point:

- `TriggerDetector.detect(context: TriggerContext) -> TriggerResult`

Current implementation:

- `BeforeFinishTrigger`: detects `supervisor.complete_task(...)` in proposed execution code.

Extension rules:

- New triggers must consume only `TriggerContext`.
- New triggers must not read evaluator results, hidden AppWorld state, or future outputs.
- Triggers should be deterministic and cheap.
- Trigger metadata must be JSON-serializable and free of secrets.
