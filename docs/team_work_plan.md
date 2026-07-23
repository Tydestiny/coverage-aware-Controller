# Team Work Plan

Project: Coverage-Aware Controller Agent for AppWorld

Repository: `D:\TY\SJTU\CIS\coverage-aware-controller`

Team size: 5 people, including the project lead.

Current baseline: the core `simplified_react_code_agent + controller` lifecycle has passed mock tests and one real Qwen Coder Flash API smoke test with `DeterministicSmokeController`. The next work should build around the frozen interfaces, not rewrite the core loop.

## Global Research Constraints

- The online Controller capability must not clearly exceed the fixed Coder Flash Agent.
- Stronger models may be used only for offline analysis of positive-negative samples and prompt optimization.
- The Controller must never access evaluator outputs, ground truth answers, future trajectory steps, hidden AppWorld state, or database snapshots unavailable to the Agent.
- The online Controller must not directly output a full task solution or executable task code.
- Simple ReAct, Generic Reflection, and Structured Controller variants must use fair budgets.
- Checkpoint branching is for offline train/dev data generation, not for rollback-based formal evaluation.
- Formal evaluation must not gain unfair advantage through retrying from previous checkpoints.
- Failed experiments and negative results must be preserved.
- Group members must not modify frozen core interfaces without project lead review.

## Role 1: Project Lead And Integration Owner

Recommended branch: `lead/integration-protocol`

### Goal

Own the stable contracts, experiment protocol, PR review, and final integration decisions.

### Current Basis

- Frozen core interfaces in:
  - `controller_agent/state/schemas.py`
  - `controller_agent/integration/controlled_react_agent.py`
  - `controller_agent/integration/appworld_adapter.py`
  - `controller_agent/integration/base.py`
  - `controller_agent/triggers/base.py`
- Development contract in `docs/development_contract.md`.
- Real smoke evidence in `results/controller_integration_smoke_final/final_smoke_report.md`.

### Files To Modify

- `docs/`
- top-level `README.md`
- experiment protocol scripts under `scripts/`
- test orchestration files if needed

### Files Not To Modify Casually

The lead may modify frozen core only after documenting the breaking change and coordinating downstream owners. Other members should not.

### Input Interface

- PRs from all members.
- Test results from `python -m pytest`.
- Experiment reports under `results/`.

### Output Interface

- Approved interface changes.
- Updated experiment protocol.
- Release notes for each milestone.

### Minimum Deliverable

- Maintain a reviewed, passing main branch.
- Keep `docs/project_status_audit.md`, `docs/team_work_plan.md`, and `docs/module_dependency_map.md` current.

### Acceptance Criteria

- `python -m pytest` passes.
- No PR leaks evaluator, ground truth, future steps, hidden state, or API keys into Controller inputs.
- Frozen core changes are rare, reviewed, and documented.

### Test Requirement

- Full `python -m pytest` for every integration PR.

### Real API Permission

Allowed only for explicitly approved smoke or milestone runs. No unapproved batch runs.

### Dependencies

Depends on all other members for module deliverables.

## Role 2: Real Qwen Controller Backend And Prompt G0

Recommended branch: `feature/qwen-controller-backend-g0`

### Goal

Implement the first real LLM Controller backend behind the existing `Controller` adapter.

### Current Basis

- `controller_agent/controller/controller.py` already validates backend outputs.
- `ControllerResponse` already supports usage metadata.
- `DeterministicSmokeController` in `controller_agent/controller/smoke.py` shows the expected decision flow but is not a real policy.

### Files To Add Or Modify

- Add backend client modules under `controller_agent/controller/`.
- Add prompt template files under `controller_agent/controller/` or `docs/`.
- Add tests under `controller_agent/tests/`.
- Update controller README if needed.

### Files Not Allowed Without Review

- `controller_agent/state/schemas.py`
- `controller_agent/integration/controlled_react_agent.py`
- `controller_agent/integration/appworld_adapter.py`
- `controller_agent/triggers/base.py`

### Input Interface

- `ControllerInput`
- Action enum from `Action`
- Budget and runtime counters from caller-visible state only

### Output Interface

- `ControllerResponse`
- Valid JSON-compatible raw response logs with no API key

### Minimum Deliverable

- A mock-tested `QwenControllerBackend` that can be called through `Controller.decide`.
- A fixed Prompt G0 that returns one of the existing actions and a short instruction.
- A dry-run path that validates parsing without a live API.

### Acceptance Criteria

- Invalid model output fails open through existing `Controller`.
- Backend never receives evaluator, ground truth, future trajectory, hidden DB state, or secrets.
- Usage fields are populated when provider metadata is available; otherwise set to `None`.
- `python -m pytest` passes.

### Test Requirement

- Mock API tests for valid JSON, invalid JSON, timeout, provider error, and usage accounting.
- No real API calls in unit tests.

### Real API Permission

Allowed only after mock tests pass and the lead explicitly authorizes a single-task smoke test.

### Dependencies

- Depends on frozen `ControllerInput` and `ControllerResponse`.
- Later depends on Role 5 for online experiment reporting.

## Role 3: Trigger And Offline Checkpoint Selector Owner

Recommended branch: `feature/triggers-checkpoint-selector`

### Goal

Add offline checkpoint screening and future trigger features without changing the main loop.

### Current Basis

- `BeforeFinishTrigger` exists in `controller_agent/triggers/before_finish.py`.
- `TriggerDetector.detect(context)` is stable.
- Runtime checkpoint manifest format exists in `controller_agent/branching/checkpoint_storage.py`.

### Files To Add Or Modify

- Add trigger implementations under `controller_agent/triggers/`.
- Add checkpoint selector under `controller_agent/branching/checkpoint_selector.py`.
- Add selector tests under `controller_agent/branching/tests/`.
- Update trigger docs.

### Files Not Allowed Without Review

- `controller_agent/state/schemas.py`
- `controller_agent/integration/controlled_react_agent.py`
- `controller_agent/integration/appworld_adapter.py`

### Input Interface

- `TriggerContext` for online triggers.
- Runtime checkpoint `manifest.json`, `pre_action/*.json`, and `post_observation/*.json` for offline selection.

### Output Interface

- `TriggerResult`
- Candidate checkpoint list with reasons and deterministic ordering.

### Minimum Deliverable

- A deterministic offline selector that reads checkpoint manifests and selects candidate checkpoints by visible criteria such as context length, write-risk markers, recent errors, and before-finish position.

### Acceptance Criteria

- Does not inspect evaluator, ground truth, hidden state, or future records.
- Selection is deterministic with a fixed seed if sampling is used.
- Works on the mock runtime checkpoint demo.
- `python -m pytest` passes.

### Test Requirement

- Tests for deterministic selection, leakage prevention, empty manifest, missing files, and stable sort order.

### Real API Permission

Not allowed. This role works offline.

### Dependencies

- Depends on Role 4 for real dense checkpoint artifacts.
- Produces input for Role 5 Branch Runner.

## Role 4: Checkpoint Collection And Restore Owner

Recommended branch: `feature/runtime-checkpoint-restore`

### Goal

Turn the current checkpoint prototypes into a reliable real AppWorld checkpoint source for branch experiments.

### Current Basis

- Static replay prototype:
  - `controller_agent/branching/replay_checkpoint_builder.py`
  - `controller_agent/branching/trajectory_loader.py`
  - `results/static_checkpoint_replay/minimal_replay_result.json`
- Runtime capture prototype:
  - `controller_agent/branching/checkpoint_capture_hook.py`
  - `controller_agent/branching/runtime_checkpoint_manager.py`
  - `controller_agent/branching/checkpoint_storage.py`
  - `results/checkpoints/mock_runtime_collection_demo/scripted-task/manifest.json`

### Files To Add Or Modify

- `controller_agent/branching/`
- `controller_agent/branching/tests/`
- `docs/runtime_checkpoint_collection.md`
- `docs/static_trajectory_checkpoint_notes.md`
- Optional new script under `scripts/` for checkpoint collection dry runs.

### Files Not Allowed Without Review

- `controller_agent/integration/controlled_react_agent.py`
- `controller_agent/state/schemas.py`
- `controller_agent/triggers/base.py`

Changes to `controller_agent/integration/appworld_adapter.py` require lead review because it is frozen core.

### Input Interface

- `ControlledAppWorldRunner(checkpoint_capture_hook=...)`
- AppWorld world object with `save_state` and `load_state` if available.
- Agent messages and visible execution outputs.

### Output Interface

- `results/checkpoints/<run_id>/<task_id>/manifest.json`
- `pre_action/*.json`
- `post_observation/*.json`
- `terminal.json`
- Restore validation report.

### Minimum Deliverable

- A real single-task dense checkpoint collection run using cached or explicitly authorized model output.
- A restore validator that loads one saved checkpoint and confirms a read-only probe is stable.

### Acceptance Criteria

- Every environment execution has a post-observation checkpoint.
- Every Agent turn has a pre-action record.
- Blocked `complete_task` has no post-observation checkpoint.
- Terminal records correctly identify whether `complete_task` entered `world.batch_execute`.
- No evaluator or ground truth fields are stored.
- Namespace status is honestly recorded as `replay_required` unless a safe snapshot method is proven.
- `python -m pytest` passes.

### Test Requirement

- Existing mock runtime tests must remain green.
- Add restore tests with mock AppWorld state ids.
- No real model call in unit tests.

### Real API Permission

No batch API runs. A single smoke collection run is allowed only after lead approval.

### Dependencies

- Produces artifacts for Role 3 selector and Role 5 Branch Runner.

## Role 5: Branch Runner, Pair Builder, And Experiment Statistics Owner

Recommended branch: `feature/branch-runner-pair-builder`

### Goal

Build the offline experiment pipeline from selected checkpoints to branch outcomes, positive-negative pairs, and prompt optimization inputs.

### Current Basis

- No Branch Runner exists yet.
- `ControlledAppWorldRunner` provides budget semantics.
- Checkpoint storage and static replay produce candidate state metadata.
- Official evaluator should be used through AppWorld, not LLM Judge.

### Files To Add Or Modify

- Add `controller_agent/branching/branch_runner.py`.
- Add `controller_agent/branching/pair_builder.py`.
- Add `controller_agent/branching/statistics.py` or a dedicated evaluation module.
- Add scripts under `scripts/`.
- Add tests under `controller_agent/branching/tests/`.
- Add result report docs under `docs/`.

### Files Not Allowed Without Review

- Frozen core files.
- Controller backend prompt files owned by Role 2.

### Input Interface

- Selected checkpoints from Role 3.
- Restorable checkpoint artifacts from Role 4.
- Candidate controller strategies or prompts.

### Output Interface

- Branch rollout records.
- Official evaluator results.
- Per-checkpoint action comparison.
- Positive-negative pairs for offline prompt optimization.
- Cost and token accounting tables.

### Minimum Deliverable

- A mock Branch Runner that can run multiple branches from the same checkpoint object without real API.
- A pair builder that labels better and worse Controller decisions based on success, evaluator score, tokens, cost, and steps.

### Acceptance Criteria

- Same checkpoint produces isolated branch outputs.
- Branch results preserve failures and negative outcomes.
- No evaluator result is fed back into online Controller context.
- Budget accounting is fair across Simple ReAct, Generic Reflection, and Structured Controller variants.
- `python -m pytest` passes.

### Test Requirement

- Mock tests for branch isolation, budget exhaustion, evaluator result parsing, tie handling, and pair generation.

### Real API Permission

Not allowed until Milestone 3 approval. Initial work is offline/mock only.

### Dependencies

- Depends on Role 3 selected checkpoints.
- Depends on Role 4 restore artifacts.
- Depends on Role 2 for real Controller strategies in later milestones.

## Milestones

### Milestone 1: Module Interfaces And Mock Chain

Prerequisites:

- Current frozen core remains unchanged.
- `python -m pytest` passes.

Participants:

- All members.

Deliverables:

- Real Controller backend skeleton with mock tests.
- Offline checkpoint selector MVP.
- Restore validator MVP.
- Mock Branch Runner MVP.
- Updated docs and ownership map.

Pass standard:

- All modules consume existing public interfaces.
- No real API calls in tests.
- No leakage of evaluator, ground truth, hidden state, or future trajectory.

Parallel work:

- Role 2 backend skeleton.
- Role 3 selector.
- Role 4 restore validator.
- Role 5 mock Branch Runner.

Must wait:

- Any frozen core change waits for lead review.

### Milestone 2: Baseline And Dense Checkpoint Collection

Prerequisites:

- Role 4 checkpoint capture and restore validator pass mock tests.
- Lead approves a small real AppWorld collection run.

Participants:

- Role 1, Role 3, Role 4.

Deliverables:

- One or more dense checkpoint baseline runs.
- Restore validation report.
- Selector report over real checkpoint manifests.

Pass standard:

- Every environment execution is checkpointed.
- Manifests can locate checkpoints by step and task.
- Restore/load probe is stable or clearly marked replay-required.

Parallel work:

- Role 2 continues prompt/backend hardening with mocks.
- Role 5 prepares branch result schema.

Must wait:

- Real branch execution waits for validated checkpoint artifacts.

### Milestone 3: Single Checkpoint Multi-Branch Experiment

Prerequisites:

- At least one validated checkpoint from Milestone 2.
- Branch Runner mock tests pass.

Participants:

- Role 1, Role 4, Role 5.

Deliverables:

- One checkpoint, multiple offline branches.
- Official evaluator results for each branch.
- Fair budget accounting table.
- Failure-preserving branch logs.

Pass standard:

- Branches start from the same checkpoint.
- No rollback retry advantage in formal scoring.
- Official evaluator is used.
- Failures are retained.

Parallel work:

- Role 3 refines selector using only train/dev artifacts.
- Role 2 prepares Controller prompt G0 for controlled online test.

Must wait:

- Positive-negative pair generation waits for branch outcome schema.

### Milestone 4: Real LLM Controller Small-Scale Experiment

Prerequisites:

- Real backend has mock tests and one approved smoke.
- Branch Runner has validated branch isolation.
- Selector has deterministic candidate checkpoints.

Participants:

- Role 1, Role 2, Role 3, Role 5.

Deliverables:

- Small controlled experiment comparing Simple ReAct, Generic Reflection, and Structured Controller.
- Token, cost, evaluator, and failure analysis.
- Controller behavior audit.

Pass standard:

- Online Controller does not exceed the capability budget of Coder Flash.
- Controller does not output full code solutions.
- Controller receives only allowed visible fields.
- Fair budget accounting is documented.

Parallel work:

- Role 4 continues restore robustness work.

Must wait:

- Prompt optimization waits for positive-negative data.

### Milestone 5: ACON-Inspired Prompt Optimization And Formal Comparison

Prerequisites:

- Stable positive-negative pair dataset.
- Small-scale Controller experiment has analyzable failures.
- Lead approves offline strong-model use.

Participants:

- Role 1, Role 2, Role 5.

Deliverables:

- Offline prompt optimization workflow.
- Prompt G1/G2 candidates.
- Formal comparison protocol.
- Final report with negative results retained.

Pass standard:

- Strong model is used only offline.
- Train/dev/test task split prevents leakage.
- Formal evaluation does not use checkpoint rollback as an advantage.
- Report states whether the Controller improves outcome, cost, or both.

Parallel work:

- Role 3 can expand selector diagnostics.
- Role 4 can improve checkpoint restore reliability.

Must wait:

- Formal evaluation waits for frozen prompt and approved protocol.

## This Week's Recommended Tasks

1. Role 2: implement mock-tested Qwen Controller backend and Prompt G0 parser without live API calls.
2. Role 3: implement deterministic offline checkpoint selector over the existing mock manifest format.
3. Role 4: clean up the duplicate checkpoint hook protocol naming and add restore validation design notes.
4. Role 5: define branch result schema and mock Branch Runner tests.
5. Role 1: review all PRs for interface compatibility and leakage constraints.

## Git Collaboration Rules

- Each role works on its recommended feature branch.
- PRs must include:
  - changed files
  - tests run
  - leakage check statement
  - whether any real API was called
  - result artifacts if a smoke test was approved
- Frozen core files require explicit lead approval.
- Do not remove failed experiment outputs unless the lead approves archival.
- Do not commit API keys, `.env`, raw AppWorld data, or provider logs containing secrets.
