# triggers/

Trigger modules decide whether a proposed Agent action should be inspected by the Controller.

Stable entry point:

- `TriggerDetector.detect(context: TriggerContext) -> TriggerResult`

Centralised configuration:

- All tunable knobs (lookback windows, keyword sets, per-task budgets, etc.) live in `config.py` as frozen dataclasses.  Detectors read from this module by default, so integration tests can override behaviour without editing trigger source files.  See `triggers/config.py` for the full list and defaults.

当前实现：

- `BeforeFinishTrigger`：检测 `proposed_execution` 中出现的 `supervisor.complete_task(...)` 调用。正则来源：`BeforeFinishConfig.pattern`。
- `BeforeWriteTrigger`：在 `proposed_execution` 中检测高风险写操作 API 关键词（默认：`delete`、`update`、`write`、`create`、`send`、`pay`、`purchase`、`follow`、`unfollow`、`cancel`、`remove`、`modify`、`set`、`add`、`block`、`unblock`），大小写无关，支持点号/下划线链式调用（如 `apis.orders.delete_order(...)`）。关键词列表与 `params_preview_limit` 来自 `BeforeWriteConfig`。
- `NoProgressTrigger`：从 `recent_history` 中取最近 N 步（默认 5），按优先级检测：① **repeated_error**（同动作 + 同错误指纹出现 ≥ 2 次）；② **repeated_action**（同一动作在历史中出现 ≥ `repeated_action_threshold` 次，默认 3）。可配置是否排除 `apis.api_docs.*` 探索性调用，关键词与阈值都在 `NoProgressConfig` 中。
- `CompositeTriggerDetectorWithBudget`：以 `TriggerBudgetConfig` 为准，按 `detector_order` 顺序组合上述 triggers，并施加 `max_triggers_per_task` 与 `min_before_finish_triggers` 预算。默认顺序：`before_finish` → `before_write` → `no_progress`。

Extension rules:

- New triggers must consume only `TriggerContext`.
- New triggers must not read evaluator results, hidden AppWorld state, or future outputs.
- Triggers should be deterministic and cheap.
- Trigger metadata must be JSON-serializable and free of secrets.
- New tunables must be declared in `config.py` rather than hard-coded in the trigger module.
