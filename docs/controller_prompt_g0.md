# Controller Prompt G0（基于 Part3 分支实验）

目标：把离线正样本规则写成 **可注入 Controller 的策略**，提高 pass_rate / 成功率。  
说明：分支实验里多为 **pass 提升、少见 success 翻盘**；本 G0 按“最稳先验”设计，不是保证大幅翻盘。

---

## 1. 核心策略（必须遵守）

| Checkpoint | 默认动作 | 条件动作 | 避免 |
|------------|----------|----------|------|
| 首次高风险写前 | **VERIFY** | 证据不足 → **BLOCK**；缺字段/API 错 → **LOOKUP** | 一律 LOOKUP；已成功时强拦 |
| 过早 complete 前 | **REPLAN** 或 **BLOCK** | 集合未核验 → **VERIFY** | 默认 LOOKUP |
| 报错后 | 已在修 → **PROCEED** | 401/422/参数错 → **LOOKUP**；目标替换 → **REPLAN/BLOCK** | 静默换目标 |

注入格式（与有效分支实验一致）：

```text
[CONTROLLER_GUIDANCE]
Action: VERIFY_TARGET_SET
Advice: <短指令，≤80词，可执行步骤>
Follow this advice in your next steps. ...
```

---

## 2. 已落地文件

| 路径 | 作用 |
|------|------|
| `coverage-aware-Controller/controller_agent/controller/prompts/g0_system.txt` | LLM Controller 系统提示 |
| `.../prompts/g0_user_template.txt` | 用户消息模板 |
| `.../prompts/instruction_templates.json` | 各动作可复用 Advice 模板 |
| `.../policy_g0.py` | **无需 LLM** 的确定性 G0 后端（推荐先上线） |
| `.../guidance.py` | 注入格式改为 Action/Advice |
| `controller/controller/rule_controller.py` | 桌面规则 Controller 已按 G0 加强写前/结束前默认 |

启用确定性后端：

```python
from controller_agent.controller.controller import Controller
from controller_agent.controller.policy_g0 import PolicyG0Backend

controller = Controller(backend=PolicyG0Backend())
```

单元测试：`controller_agent/tests/test_policy_g0.py`（已通过）。

---

## 3. Prompt G0（system，摘要）

完整文本见 `g0_system.txt`。要点：

1. 只输出一个 JSON：`action / reason / instruction / confidence`
2. 稀疏介入；已在正确路径 → `PROCEED`
3. 禁止泄露 gold / 隐藏状态
4. instruction 必须是祈使步骤清单
5. 先验规则严格按上表（来自 5 write + 6 complete 分支对比）

示例输出：

```json
{
  "action": "BLOCK_RISKY_ACTION",
  "reason": "irreversible remove without printed remove_set",
  "instruction": "Do not remove yet. List library/liked/downloaded, print remove_set, then remove only that set.",
  "confidence": 0.85
}
```

---

## 4. 可复用 Advice 模板（写入后给 Agent）

### 写前 VERIFY（默认）
> Do not perform irreversible writes yet. Print the exact target set required by the task (IDs/paths/recipients and the rule used). Only after the printed set is complete and unique, apply writes to that set only.

### 写前 BLOCK（证据不足）
> BLOCK the current write. First gather evidence, compute the exact target set, print counts/IDs, then rewrite so only that set is modified. Do not submit a natural-language answer instead of doing the work.

### 结束前 REPLAN（默认）
> Do not complete. Re-plan remaining steps in order, execute them for real, verify the resulting state, then complete_task with the correct answer format.

### 结束前 BLOCK（借口/半完成）
> Do not call complete_task yet. Remaining required work is unfinished or the answer is an excuse. Finish the real state changes, then complete_task() with the correct answer format (often answer=None for state tasks).

### LOOKUP（仅 API/字段不明）
> Call show_api_doc for the failing/uncertain API. Fix parameter names, types, and access_token requirements, then retry. Do not change the task goal.

---

## 5. 下一步怎么验证“是否极大提升”

1. **M0**：裸 ReAct  
2. **M3-G0**：`PolicyG0Backend`（或 LLM + 同一 system prompt）  
3. 套件：Dev10 或 part3 标注任务  
4. 看：success、mean pass_rate、干预次数、过拦率  

若只要快速上线，先用 **PolicyG0（规则）**；LLM 版用同一 `g0_system.txt`，解析失败 fail-open 到 PROCEED。
