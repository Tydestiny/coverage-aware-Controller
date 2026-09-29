# controller/

Controller modules convert a visible `ControllerInput` into a validated `ControllerResponse`.

Stable entry point:

- `Controller.decide(controller_input)`

Current implementations:

- `Controller`: wraps an optional backend callable and validates output.
- `GuidanceBuilder`: converts non-`PROCEED` decisions into Action/Advice guidance.
- `DeterministicSmokeController`: model-free smoke controller for integration tests only.
- `LiveLLMBackend`: uses a run-specific system prompt supplied explicitly by path.
- `input_renderer.py`: serializes the complete visible `ControllerInput` without a historical policy.

Usage (local only; do not push unless intentional):

```python
from controller_agent.controller.controller import Controller
from controller_agent.controller.live_llm_backend import LiveLLMBackend

controller = Controller(backend=LiveLLMBackend(system_prompt_path="path/to/new_train_dev_prompt.txt"))
```

Extension rules:

- Real LLM controllers should be added as backend callables or small backend classes.
- Backends may return `ControllerResponse`, `ControllerDecision`, JSON text, or a legacy decision mapping.
- Backend failures must fail open through `Controller`.
- Do not call AppWorld evaluator APIs or read hidden task state here.
