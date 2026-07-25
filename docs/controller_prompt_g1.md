# Controller Prompt G1（基于 latest simple-react-controller-agent）

## 目标

G1 在最新 `simple-react-controller-agent` 的 G0 架构上实现，不修改 Trigger、StateBuilder、Action 枚举、Controller 主循环、拦截逻辑或 guidance 注入格式。

当前阶段只处理 `before_finish`：Agent 准备调用 `supervisor.complete_task(...)` 时，Controller 判断完成声明是否得到可见证据支持。

## 与旧版 G1 的接口差异

旧版实验 G1 输出：

```json
{"action":"VERIFY_TARGET_SET","instruction":"..."}
```

最新架构要求：

```json
{"action":"VERIFY_TARGET_SET","reason":"...","instruction":"..."}
```

因此 latest-G1 保留 `reason` 作为审计字段。`instruction` 仍然必须自包含，因为实际注入 Agent 下一轮上下文的是 Action 和 instruction，而不是完整 Controller reason。

## 相对最新 G0 的主要变化

1. 在 Action 选择前静默识别 proposed behavior、required basis、visible support、concrete defect 和 minimal correction。
2. 对每个任务谓词区分 `SUPPORTED`、`CONTRADICTED`、`NOT_ESTABLISHED` 和 `STALE`。
3. 把任务关系、来源、范围、过滤、集合运算、分页、排名、量词和完成条件作为独立谓词，避免一个证据替代另一个证据。
4. `reason` 只用于简短、可审计的问题依据。
5. `instruction` 必须同时指出具体可见缺口和最小立即修正，通常不超过 50 个英文词。
6. 拒绝可复制到任意任务的宽泛建议，例如单独的 “re-check the task”。
7. 保持最新 G0 的 before-finish Action 边界：语义路线错误用 `REPLAN`，API 语义未知用 `LOOKUP_API_SPEC`，路线正确但覆盖不完整用 `VERIFY_TARGET_SET`，证据充分用 `PROCEED`；不使用 `BLOCK_RISKY_ACTION`。

## 文件

- `controller_agent/controller/prompts/g1_system.txt`
- `controller_agent/controller/live_llm_backend.py`
- `controller_agent/tests/test_controller_g1_prompt_contract.py`
- `controller_agent/tests/test_live_llm_backend.py`

## 使用

初始状态真实任务脚本：

```bash
python scripts/run_live_controller_initial_task.py \
  --controller-prompt g1 \
  --task-id <TASK_ID>
```

checkpoint smoke：

```bash
python scripts/run_live_controller_checkpoint_smoke.py \
  --controller-prompt g1 \
  --task-id <TASK_ID> \
  --checkpoint-id <CHECKPOINT_ID>
```

代码调用：

```python
from pathlib import Path

from controller_agent.controller.controller import Controller
from controller_agent.controller.live_llm_backend import LiveLLMBackend

prompt = Path("controller_agent/controller/prompts/g1_system.txt")
controller = Controller(backend=LiveLLMBackend(system_prompt_path=prompt))
```

G0 仍是默认值，旧脚本在不传 `--controller-prompt` 时行为不变。

两条运行脚本都会在配置文件中记录 Prompt 名称、绝对路径和 SHA-256，便于确认不同实验只改变了 System Prompt。

G1 不携带自己的模型配置。G0 和 G1 都显式使用最新 `LiveLLMBackend.DEFAULT_MODEL`，当前为 `qwen3-coder-flash-2025-07-28`。因此对照实验中的 Controller 模型保持一致，唯一主要变量是 System Prompt。
