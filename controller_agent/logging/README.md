# logging/

Logging modules record structured Controller events for tests, smoke runs, and later reports.

Stable entry point:

- `ExperimentLogger.log_controller_event(...)`

Current behavior:

- Stores JSON-serializable event dictionaries in memory.
- Optionally forwards copied events to a sink callable.
- Records trigger metadata, controller decision, usage metadata, and runtime counters.

Extension rules:

- Log sinks may write JSONL, CSV, or experiment reports.
- Never log API keys, authorization headers, provider secrets, evaluator internals, or hidden AppWorld state.
- Keep event fields additive where possible so downstream reports do not break.
