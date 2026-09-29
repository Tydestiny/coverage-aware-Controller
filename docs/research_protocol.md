# 后续实验计划

研究问题：固定 Agent 和任务后，有限次事前审核能否提高成功率和需求完成度，增加多少调用与 token？

## 当前基础

- Prompt：`prompts/train_dev/train_dev_v1.txt`，尚未验证实际效果。
- 名单：`configs/studies/train_dev_v1.json`，8 个 train、8 个预留 dev，按任务族隔离；不宣称从未接触过这些任务。
- 模型：Coder Flash + DeepSeek；Controller 最多调用和阻断各 4 次，之后放行。
- 输入与运行契约见 [架构说明](controller_architecture.md)，操作入口见 README。

## 推进顺序

1. **工程核查**：核对输入来源、任务边界、执行反馈、checkpoint 恢复、token 和官方评价。缺失、异常、截断必须记录。
2. **train 开发**：先跑两个固定任务的 baseline/controller，再完成剩余任务。沿“原提案 → 判断 → 指导 → 实际行动 → 反馈”分析，区分输入、触发、判断、指导、执行问题；证据不足标为未能归因。
3. **单因素修改**：每次只改 Prompt、输入呈现、Trigger 或指导包装中的一项。保留版本、假设和退化案例。暂不训练神经 Trigger。
4. **dev 检验**：冻结候选后比较 baseline、三类 Trigger 和仅 before_finish。相同调用上限不代表相同实际计算量，报告实际调用和 token。
5. **局部分支**：预先选最多四个可恢复 dev checkpoint，比较自然继续与一次干预，每种条件重复两次。核对状态、消息、代码前缀和待执行提案，独立恢复。用于开发分析，不作为独立确认。
6. **正式评估**：链路可靠且有值得验证的信号后，冻结代码、Prompt 哈希、模型、任务、seed 和统计方法，再决定 test 规模。

## 评价标准

- 主指标：官方 exact success。其次为任务宏平均需求完成度、正负向变化、输入/输出 token、调用和执行次数。
- 判断质量：意见有可见依据、指导可执行、不重复已完成检查、合理动作能放行。人工抽样包含触发和未触发位置、正确和有问题的行动。
- Trigger 阳性率、实际调用率以可执行候选动作为分母；预算绕过率以阳性候选为分母。fail-open 不能当作模型主动 PROCEED。
- 同前缀比较报告双方成功、双方失败、仅干预成功、仅自然继续成功。重复运行按 checkpoint 和 task 汇总，不扩充独立任务数。
- 终局差异不能独自证明某条指导正确或有害；不清楚的成因保留未知。缺失 token 不能填零。

## 开发边界

只使用 train 开发，预留 dev 做门槛；根据 dev 修改后的结果归入开发证据。历史 G0/G1/G2、test 和 oracle 不作为新 Prompt 输入。新 Prompt 不会消除过去接触 test 的事实。

保留失败与负收益，不按结果换任务。Controller 不读取 evaluator、参考答案、未来轨迹或隐藏数据库状态。

设计参考：[ReAct](https://arxiv.org/abs/2210.03629) 的行动反馈关联、[Reflexion](https://arxiv.org/abs/2303.11366) 的简短建议、[CRITIC](https://arxiv.org/abs/2305.11738) 的反馈验证。它们不直接证明本方法有效。
