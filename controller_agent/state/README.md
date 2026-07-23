# state/

State modules define the public data contracts and build the Controller-visible view.

Stable files:

- `schemas.py`: frozen dataclasses and action enum.
- `builder.py`: conservative visible-state extraction.

Stable outputs:

- `TriggerContext`
- `ControllerInput`
- `ControllerDecision`
- `ControllerResponse`
- `ControllerRuntimeState`
- `ControllerConfig`
- `InterceptionResult`

Extension rules:

- State extraction may use only task text, visible messages, proposed execution, visible tool outputs, and runtime counters.
- Do not include evaluator scores, ground truth answers, hidden database state, future observations, or API secrets.
- Keep dataclass additions backward compatible unless the team explicitly opens a new core version.
