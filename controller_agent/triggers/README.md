# triggers/

Trigger modules decide whether a proposed Agent action should be inspected by the Controller.

Stable entry point:

- `TriggerDetector.detect(context: TriggerContext) -> TriggerResult`

Centralised configuration:

- All tunable knobs (lookback windows, keyword sets, per-task budgets, etc.) live in `config.py` as frozen dataclasses.  Detectors read from this module by default, so integration tests can override behaviour without editing trigger source files.  See `triggers/config.py` for the full list and defaults.

当前实现：

- `BeforeFinishTrigger`：检测 `proposed_execution` 中出现的 `supervisor.complete_task(...)` 调用。正则来源：`BeforeFinishConfig.pattern`。
- `BeforeWriteTrigger`：用 AST 检查候选代码中的 API 调用，按 `BeforeWriteConfig.high_risk_keywords` 匹配写操作；覆盖点赞、标记、清空、移动、付款审批和播放状态修改等。`read_only_api_names` 排除官方只读例外（例如查询评论、标签、下载收据），不匹配注释或字符串。参数预览仍由 `params_preview_limit` 控制。
- 这些规则识别审核机会，不判断操作是否错误或目标是否正确；动态 API 别名和运行时条件不做数据流推断。规则扩展后，实际 Controller 调用位置可能变化，旧实验结果不重新解释为新规则的结果。
- `NoProgressTrigger`：从 `recent_history` 中取最近 N 步（默认 5），按优先级检测：① **repeated_error**（同动作 + 同错误指纹出现 ≥ 2 次）；② **repeated_action**（同一动作在历史中出现 ≥ `repeated_action_threshold` 次，默认 3）。可配置是否排除 `apis.api_docs.*` 探索性调用，关键词与阈值都在 `NoProgressConfig` 中。
- `CompositeTriggerDetectorWithBudget`：以 `TriggerBudgetConfig` 为准，按 `detector_order` 顺序组合上述 triggers，并施加 `max_triggers_per_task` 与 `min_before_finish_triggers` 预算。默认顺序：`before_finish` → `before_write` → `no_progress`。

Extension rules:

- New triggers must consume only `TriggerContext`.
- New triggers must not read evaluator results, hidden AppWorld state, or future outputs.
- Triggers should be deterministic and cheap.
- Trigger metadata must be JSON-serializable and free of secrets.
- New tunables must be declared in `config.py` rather than hard-coded in the trigger module.
