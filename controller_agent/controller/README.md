# controller/

Controller modules convert a visible `ControllerInput` into a validated `ControllerResponse`.

Stable entry point:

- `Controller.decide(controller_input)`

Current implementations:

- `Controller`: wraps an optional backend callable and validates output.
- `GuidanceBuilder`: converts non-`PROCEED` decisions into Action/Advice guidance.
- `DeterministicSmokeController`: model-free smoke controller for integration tests only.
- `PolicyG0Backend` (`policy_g0.py`): deterministic rule policy from Part3 branch experiments.
- `LiveLLMBackend` (`live_llm_backend.py`): Prompt-G0 LLM backend (OpenAI-compatible / DashScope).
- Prompt G0 templates under `prompts/` (`g0_system.txt`, `g0_user_template.txt`, `instruction_templates.json`).

Rule backend (no API):

```python
from controller_agent.controller.controller import Controller
from controller_agent.controller.policy_g0 import PolicyG0Backend

controller = Controller(backend=PolicyG0Backend())
```

Live LLM backend (needs `DASHSCOPE_API_KEY` or `OPENAI_API_KEY`):

```python
from controller_agent.controller.controller import Controller
from controller_agent.controller.live_llm_backend import LiveLLMBackend

controller = Controller(backend=LiveLLMBackend())
# optional: LiveLLMBackend(model_name="qwen3-coder-flash-2025-07-28")
```

`LiveLLMBackend.__call__` loads Prompt G0 → `call_model(...)` → returns JSON text for `Controller` to parse.

Extension rules:

- Real LLM controllers should be added as backend callables or small backend classes.
- Backends may return `ControllerResponse`, `ControllerDecision`, JSON text, or a legacy decision mapping.
- Backend failures must fail open through `Controller`.
- Do not call AppWorld evaluator APIs or read hidden task state here.
