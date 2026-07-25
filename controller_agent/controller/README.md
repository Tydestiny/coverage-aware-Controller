# controller/

Controller modules convert a visible `ControllerInput` into a validated `ControllerResponse`.

Stable entry point:

- `Controller.decide(controller_input)`

Current implementations:

- `Controller`: wraps an optional backend callable and validates output.
- `GuidanceBuilder`: converts non-`PROCEED` decisions into Action/Advice guidance.
- `DeterministicSmokeController`: model-free smoke controller for integration tests only.
- `PolicyG0Backend` (`policy_g0.py`): deterministic policy from Part3 branch experiments (local).
- Prompt G0 templates under `prompts/` (`g0_system.txt`, `g0_user_template.txt`, `instruction_templates.json`).
- Prompt G1 under `prompts/g1_system.txt`: diagnosis-first before-finish policy with self-contained minimal guidance.
- `LiveLLMBackend(system_prompt_path=...)`: selects G0 by default or another compatible fixed prompt such as G1.

Usage (local only; do not push unless intentional):

```python
from controller_agent.controller.controller import Controller
from controller_agent.controller.policy_g0 import PolicyG0Backend

controller = Controller(backend=PolicyG0Backend())
```

Prompt G1:

```python
from pathlib import Path

from controller_agent.controller.controller import Controller
from controller_agent.controller.live_llm_backend import LiveLLMBackend

prompt = Path("controller_agent/controller/prompts/g1_system.txt")
controller = Controller(backend=LiveLLMBackend(system_prompt_path=prompt))
```

Extension rules:

- Real LLM controllers should be added as backend callables or small backend classes.
- Backends may return `ControllerResponse`, `ControllerDecision`, JSON text, or a legacy decision mapping.
- Backend failures must fail open through `Controller`.
- Do not call AppWorld evaluator APIs or read hidden task state here.
