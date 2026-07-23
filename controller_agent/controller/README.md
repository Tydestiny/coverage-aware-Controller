# controller/

Controller modules convert a visible `ControllerInput` into a validated `ControllerResponse`.

Stable entry point:

- `Controller.decide(controller_input)`

Current implementations:

- `Controller`: wraps an optional backend callable and validates output.
- `GuidanceBuilder`: converts non-`PROCEED` decisions into temporary guidance.
- `DeterministicSmokeController`: model-free smoke controller for integration tests only.

Extension rules:

- Real LLM controllers should be added as backend callables or small backend classes.
- Backends may return `ControllerResponse`, `ControllerDecision`, JSON text, or a legacy decision mapping.
- Backend failures must fail open through `Controller`.
- Do not call AppWorld evaluator APIs or read hidden task state here.
