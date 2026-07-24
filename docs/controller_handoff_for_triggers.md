# Controller ↔ Trigger 对接说明（本机 / Role 2）

面向 Trigger 同学：你们负责「何时调用」；Controller 负责「调用后选什么动作、给什么 Advice」。

> 改动目前在本机 `coverage-aware-Controller`，**默认不 push**。对接时以本文件 + 冻结 schemas 为准。

---

## 1. 调用约定

```text
Trigger 检出关键点
  → 组装 ControllerInput
  → Controller(backend=PolicyG0Backend()).decide(input)
  → 若 action != PROCEED：用 GuidanceBuilder 注入 Advice，并按 action 决定是否跳过 execute
```

Python：

```python
from controller_agent.controller.controller import Controller
from controller_agent.controller.policy_g0 import PolicyG0Backend
from controller_agent.controller.guidance import GuidanceBuilder
from controller_agent.state.schemas import Action

controller = Controller(backend=PolicyG0Backend())
response = controller.decide(controller_input)
decision = response.decision
guidance = GuidanceBuilder().build(decision)  # PROCEED → ""

# blocked execution when action is not PROCEED (team loop already does this for finish)
allow_execution = decision.action == Action.PROCEED
```

Fail-open：backend 异常 / 非法输出 → `Controller` 已回落为 `PROCEED`（保持原 Agent 行为）。

---

## 2. 输入：`ControllerInput`（你们填）

| 字段 | 要求 | 说明 |
|------|------|------|
| `task` | 必填 | 任务指令原文（不要塞 gold / evaluator） |
| `trigger_reason` | 必填 | 建议用稳定字符串（见下） |
| `current_agent_output` | 必填 | **本步拟执行代码**（即将 `batch_execute` 的内容） |
| `recent_history` | 建议 | 最近若干轮 messages（可截断） |
| `recent_errors` | 建议 | 最近失败输出片段 |
| `completed_operations` | 可选 | 已执行过的代码摘要 |
| `known_entities` | 可选 | 已知实体 id/名字 |

**禁止传入**：evaluator / ground truth / 未来轨迹 / 隐藏 DB / API key。

### 推荐 `trigger_reason`（便于 PolicyG0 路由）

| 你们的 trigger | 建议写入的 `trigger_reason` |
|----------------|-----------------------------|
| 首次/高风险写前 | `before_first_write` |
| `complete_task` 前 | `before_finish` |
| API/执行错误后 | `after_api_or_execution_error` |

也兼容文案里带 `write` / `complete` / `error` 的 reason；但 **优先用上表固定值**。

### 写前触发注意

请尽量在 **拟执行已含真实写 API**（`remove_` / `move_` / `create_` / `pay_` …）时再呼叫 Controller。  
纯 `show_api_doc` 步不要触发，否则策略会倾向 `PROCEED`，看起来像「没控」。

---

## 3. 输出：`ControllerDecision`

| 字段 | 用法 |
|------|------|
| `action` | `PROCEED` / `VERIFY_TARGET_SET` / `LOOKUP_API_SPEC` / `REPLAN` / `BLOCK_RISKY_ACTION` |
| `reason` | 日志用短因 |
| `instruction` | 注入给 Agent 的 Advice（`PROCEED` 时为空） |
| `confidence` | **暂忽略**（不参与是否拦截） |

注入格式（`GuidanceBuilder`）：

```text
Action: VERIFY_TARGET_SET
Advice: ...
Follow this advice in your next steps. ...
```

外层仍由 adapter 加 `[CONTROLLER_GUIDANCE]` 标记。

---

## 4. PolicyG0 先验（方便你们理解行为）

| `trigger_reason` 类 | 默认倾向 |
|---------------------|----------|
| 写前 | VERIFY；无集合证据 → BLOCK；缺字段/401/422 → LOOKUP |
| complete 前 | REPLAN/BLOCK；勿默认 LOOKUP |
| 错误后 | 已在修 docs → PROCEED；参数/鉴权错 → LOOKUP；目标替换 → REPLAN |

离线回放（vs 人工标注）：soft ≈ 83%（方向对齐即可，不要求与标注枚举 exact）。

---

## 5. 文件位置

| 路径 | 内容 |
|------|------|
| `controller_agent/controller/policy_g0.py` | 规则后端 |
| `controller_agent/controller/prompts/g0_system.txt` | 以后 LLM Controller 用（非必须） |
| `controller_agent/controller/guidance.py` | Advice 格式化 |
| `controller_agent/state/schemas.py` | **冻结**接口（勿擅自改） |
| `Desktop/branch_case_type_runs/` | 分支实验结论包 |

---

## 6. 联调检查清单

- [ ] 只在约定 trigger 点调用 `decide`
- [ ] `current_agent_output` 是拟执行代码，不是空字符串
- [ ] `PROCEED` 不注入、不跳过 execute
- [ ] 非 `PROCEED` 注入 guidance，且按主循环约定跳过/允许 execute
- [ ] 连续拦截有上限（建议由主循环/你们做 2–3 次后 fail-open，避免死循环）

有接口歧义时：以 `schemas.py` + 本文件为准；策略细节可再调 `policy_g0.py`，**尽量不改 schemas**。
