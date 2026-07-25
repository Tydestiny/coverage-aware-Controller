# Real AppWorld G0/G1 Controller Trajectory Analysis

## Executive conclusion

The real `before_finish -> StateBuilder -> Controller -> block -> guidance -> Agent continuation -> evaluator` path now runs end to end on AppWorld task `07b42fd_1`.

The experiment does **not** show that G1 is better than G0. Both prompts detected that the original completion should not be accepted, blocked it, injected guidance, and changed the Agent's subsequent behavior. However, both prompts diagnosed repaired historical API errors instead of the decisive semantic defect. The Agent continued to treat `search_artists(query="classical")` as if it were an explicit genre filter, followed additional non-classical artists, and supplied a natural-language answer for a state-changing task whose expected answer was null. Both final runs failed the official evaluator with the same two failed requirements as the original baseline.

This negative result is useful. It shows that the integration works, but the current Controller input and prompt do not yet support reliable temporal and predicate-level diagnosis.

## Research question tested

Given the same real Agent state immediately before an incorrect `complete_task` call, does replacing the latest G0 system prompt with G1 improve:

1. diagnosis of the concrete task defect;
2. specificity and correctness of the injected guidance;
3. Agent recovery behavior; and
4. official AppWorld task success?

The experiment changes only the Controller system prompt. Both conditions use the latest branch's Controller and Agent model:

`qwen3-coder-flash-2025-07-28`

## Real task and checkpoint

- Task: `07b42fd_1`
- Instruction: follow all classical Spotify artists with at least 22 followers.
- Source trajectory: `qwen3_coder_flash_train_part_4_checkpoint_collection`
- Restored post-observation checkpoint: `07b42fd_1_post_0008`
- Replayed pre-action record: `07b42fd_1_pre_0009`
- Original proposed action: a `complete_task` call claiming that five named artists had been followed.

This is a useful case because the original trajectory contains several distinct issues:

1. an early `page_limit=50` call failed with HTTP 422;
2. an early field lookup failed with `KeyError`;
3. both errors were subsequently repaired;
4. five `follow_artist` calls then visibly succeeded;
5. only one search page was inspected in the original trajectory;
6. keyword search was treated as proof of the `genre == classical` predicate;
7. a natural-language answer was passed to a state-changing task.

The official baseline evaluator failed on the answer and genre requirements. It passed three of five tests.

## Formal experimental controls

The final G0 and G1 runs used identical values except for the system prompt file.

| Variable | Value |
|---|---|
| Agent model | `qwen3-coder-flash-2025-07-28` |
| Controller model | `qwen3-coder-flash-2025-07-28` |
| Temperature | `0.0` |
| Random seed | `100` |
| Trigger | `before_finish` only |
| Controller calls | at most 2 |
| Controller blocks | at most 2 |
| Agent turns | at most 10 |
| Environment executions | at most 10 |
| Extra Agent turns after first block | at most 8 |
| Cost limit | CNY 1.0 |
| Initial environment state | same restored checkpoint |
| Initial proposed completion | same recorded pre-action |

Prompt identities:

- G0 SHA-256: `cc899228e4ffe7a8684c3d2b777483752b445f80dcc66d5d7f277407e563cc67`
- G1 SHA-256: `856f771af93e8ed9adf441d80674ccfa9bf23dccd426d757a9b10f1162295edc`

## Experimental harness corrections

Several earlier runs were useful for debugging but are not counted as the final comparison.

### Recorded execution failures during namespace replay

The checkpoint contained a historical cell that failed because `page_limit=50` exceeded the API limit. The old restore function treated any replayed cell failure as an infrastructure failure and stopped, even though the original Agent continued and recovered. Replay now continues through recorded failed cells, preserving variables created before an exception and allowing later recovery cells to rebuild the namespace.

### Current-task extraction

StateBuilder previously selected an Agent sentence containing “completed the task” as the task instruction. It now extracts `Task:` only from user messages and no longer consumes the following line when the task line is empty.

### Faithful pre-action reconstruction

The first cached runs restored the proposed completion but omitted the immediately preceding observation showing five successful follow operations. This produced an invalid Controller view dominated by old failures. The final runs restore the pre-action record's complete visible message list, including the last observation and proposed completion.

### Privacy-preserving executable context

Credentials, email addresses, and phone numbers are not sent to the external model. Literal credentials remain redacted. Required profile fields are stored only in local REPL variables, while the external message history contains variable references rather than non-executable placeholders. This prevents privacy redaction from causing artificial authentication failures.

### AppWorld time-freezer restoration

`AppWorld.load_state()` closes the world's original time freezer but does not install a replacement. The harness reinstalls normal time state after loading, preventing evaluator and context-cleanup failures.

The repository test suite passes after these corrections: 99 tests passed.

## What the Controller could see

At the first final Controller call, both G0 and G1 could see:

- the exact task instruction;
- the proposed `complete_task` code;
- the old `page_limit=50` failure;
- the old incorrect-field failure;
- the later corrected use of the `name` field;
- one sample artist containing `genre: classical`;
- the five selected artist names and follower counts;
- the final observation saying all five follow operations succeeded.

They could **not** see direct support for several required predicates:

- no visible API specification established that `query="classical"` meant `genre="classical"`;
- only one displayed artist had a visible genre value;
- no genre value was displayed for the other selected artists;
- the original trajectory did not establish complete pagination;
- StateBuilder discarded the surrounding AppWorld completion instructions when extracting the single-line task, so the Controller did not receive the rule that a state-changing task should use `answer=None`.

This visibility boundary matters. The genre defect was diagnosable as an unsupported API-semantic assumption, but the exact answer-format defect was not recoverable from the supplied `ControllerInput`.

## G0 trajectory

### First Controller decision

G0 returned `REPLAN`. It focused on the historical HTTP 422 and `KeyError`, instructing the Agent to reduce `page_limit`, verify field names, paginate, and retry.

This was temporally incorrect. Both errors had already been repaired before the proposed completion, and the visible final observation showed five successful follow operations. The real unresolved issues were keyword-search semantics, genre verification, full coverage, and completion answer format.

### Agent response to first guidance

The Agent accepted the guidance and changed its plan:

1. paginated two keyword-search pages and collected 36 results;
2. filtered them only by follower count;
3. attempted to follow seven selected artists;
4. received “already following” errors for the original five;
5. successfully followed two additional artists.

The guidance therefore had a real causal effect, but the effect was harmful: it expanded the same semantically unsupported keyword-search route and added more incorrect writes.

### Second Controller decision

G0 again returned `REPLAN`, this time asking the Agent to enumerate currently followed artists and compute which targets remained unfollowed. It still accepted the seven-item target set as “classical” without direct genre evidence.

The Agent performed the membership check and then attempted a third completion. Because the two-call Controller budget was exhausted, the trigger failed open and the completion executed.

### G0 result

- Controller calls: 2
- Controller blocks: 2
- Environment executions after restore: 6
- Agent tokens after restore: 58,327
- Controller tokens: 8,026 input and 248 output
- Agent cost after restore: approximately CNY 0.00897
- Termination: task completed
- Official evaluator: failure, 3/5 tests passed

The evaluator still rejected:

1. the non-null natural-language answer; and
2. the non-classical genre among added follows.

## G1 trajectory

### First Controller decision

G1 also returned `REPLAN` and again treated the old `page_limit=50` failure as the reason the completion was unsupported. It ignored the later successful search, corrected field handling, and successful follow observation.

This is important because G1 explicitly asks the model to classify predicates as `SUPPORTED`, `CONTRADICTED`, `NOT_ESTABLISHED`, or `STALE`. The observed behavior shows that including the scaffold in the system prompt did not make the model perform temporally correct evidence reconciliation.

### Agent response to first guidance

The Agent:

1. paginated the same keyword-search route;
2. filtered by follower count but not explicit genre;
3. attempted seven follow operations;
4. added two new artists; and
5. proposed another natural-language completion answer.

### Second Controller decision

G1 returned another `REPLAN`, but its diagnosis was internally wrong. The proposed completion already said that two artists were newly followed and five were already followed. G1 claimed this statement was contradicted by evidence that five were already followed and two were newly added, then instructed the Agent to state essentially the same fact.

The guidance changed wording but did not correct the task. On the next completion trigger, the Controller budget was exhausted and the system failed open.

### G1 result

- Controller calls: 2
- Controller blocks: 2
- Environment executions after restore: 4
- Agent tokens after restore: 38,174
- Controller tokens: 8,596 input and 209 output
- Agent cost after restore: approximately CNY 0.00579
- Termination: task completed
- Official evaluator: failure, 3/5 tests passed

The same evaluator requirements failed: answer format and genre correctness.

G1 used fewer recovery executions and Agent tokens than G0, but this is not an accuracy improvement. It terminated earlier through Controller budget exhaustion while leaving the task incorrect.

## Comparative result

| Measure | Baseline | G0 | G1 |
|---|---:|---:|---:|
| Official success | No | No | No |
| Evaluator tests passed | 3/5 | 3/5 | 3/5 |
| First bad completion blocked | No Controller | Yes | Yes |
| Guidance injected | No | Yes | Yes |
| Agent behavior changed | N/A | Yes | Yes |
| Explicit genre predicate enforced | No | No | No |
| Correct `answer=None` completion | No | No | No |
| Additional unsupported writes during recovery | No | Yes | Yes |
| Controller diagnosis temporally correct | N/A | No | No |

The experiment demonstrates integration efficacy but no task-level improvement.

## Root-cause analysis

### 1. Flat error accumulation destroys temporal meaning

`recent_errors` is a task-level list of extracted error strings. It has no step identifier, no link to the action that caused the error, and no resolution state. An early 422 remains equally salient after a later successful retry. Both prompts anchored on these stale errors.

This is not merely a wording problem. A prompt cannot reliably reconstruct temporal supersession from a flat error bag, especially when the same input separately contains later successes.

### 2. Recent successful writes are raw text, not structured operations

`completed_operations` contains only action-observation pairs before the recent window. In this checkpoint it was empty. The successful follow operations existed only inside raw recent history, while the old errors were duplicated in a dedicated `recent_errors` field. The input representation therefore gave failures greater structural prominence than successes.

### 3. Required predicates are not represented explicitly

The task requires at least:

- entity type = artist;
- genre = classical;
- follower count >= 22;
- quantifier = all;
- write effect = followed;
- completion answer = null.

The Controller receives one task string and unstructured history. G1 asks the model to derive predicates silently, but there is no machine-checkable mapping from each predicate to evidence. The model treated topical keyword relevance as if it proved genre membership.

### 4. API-source semantics were unsupported but not escalated

The visible state did not contain the `search_artists` API specification. Under G1's own policy, uncertainty over whether keyword search represents the requested genre relation should have selected `LOOKUP_API_SPEC`. Instead, G1 treated the old parameter error as decisive and never challenged `query="classical"` as a proxy.

### 5. Completion-format rules were removed from ControllerInput

StateBuilder extracts the current `Task:` line and discards the containing user message from task history. That message also contains AppWorld's rule that state-changing tasks should call `complete_task` without an answer. Consequently, neither prompt had the information needed to verify the answer-format predicate.

### 6. Prompt-only supervision shares the Agent's semantic bias

Agent and Controller use the same model. Both repeatedly made the same substitution: keyword match implies genre membership. A second prompt invocation is not structurally independent evidence when the model, input representation, and latent assumptions are shared.

### 7. Block-budget fail-open can convert repeated bad guidance into execution

After two blocks, later `before_finish` triggers fail open. This is a reasonable liveness safeguard, but it means the Controller must produce actionable correction within a small number of calls. Rephrasing an already-correct count consumed G1's final intervention without addressing the semantic defect.

## What is validated

The experiment validates the following engineering properties:

1. a real `before_finish` trigger fires on a recorded AppWorld completion;
2. StateBuilder can construct a Controller input from a restored trajectory;
3. the latest Controller model returns parseable decisions for both prompts;
4. non-`PROCEED` decisions block `complete_task` before environment execution;
5. guidance is injected into the Agent's next context;
6. the Agent reads and reacts to that guidance;
7. continuation writes are executed in the restored environment;
8. the official evaluator runs on the final state.

It does not validate diagnosis accuracy or task recovery.

## Recommended next experiment

Do not add more general prose to G1 yet. First isolate the state-representation problem with a small ablation on the same real checkpoint.

### State representation ablation

Keep the model and G1 prompt fixed, but compare:

1. current flat `recent_errors`;
2. step-indexed action-observation records;
3. step-indexed records with an explicit `resolved_by_step` link;
4. a predicate-evidence table derived only from visible task and API facts.

The expected Controller decision for this case should identify that keyword search does not visibly establish the genre predicate. Without visible API semantics, `LOOKUP_API_SPEC` is the most defensible first action. Once the API schema is visible, the next action should require explicit genre filtering, complete pagination, and verification of prior writes.

### Prompt ablation after state correction

Then compare:

1. G0;
2. G1 without the predicate-status scaffold;
3. G1 with the scaffold;
4. G1 with one targeted temporal example showing an old error superseded by a later success.

This order separates whether G1 fails because the prompt is weak or because the input representation makes the right judgment unnecessarily difficult.

### Evaluation scope

One trajectory proves neither general benefit nor general failure. The next batch should stratify real checkpoints by defect type:

- stale repaired error;
- incomplete pagination;
- wrong semantic relation or proxy source;
- missing filter;
- answer-format mismatch;
- incorrect prior write requiring audit.

Report Controller diagnosis correctness, guidance compliance, task success delta, harmful intervention rate, added writes, Controller calls, tokens, and latency separately.

## Artifact locations

- G0 configuration and prompt identity: `results/controller_latest_real/07b42fd_1/g0/config.json`
- G0 Controller decisions: `results/controller_latest_real/07b42fd_1/g0/controller_events.jsonl`
- G0 turn trajectory: `results/controller_latest_real/07b42fd_1/g0/turn_events.jsonl`
- G0 evaluator: `results/controller_latest_real/07b42fd_1/g0/evaluator_result.json`
- G1 configuration and prompt identity: `results/controller_latest_real/07b42fd_1/g1/config.json`
- G1 Controller decisions: `results/controller_latest_real/07b42fd_1/g1/controller_events.jsonl`
- G1 turn trajectory: `results/controller_latest_real/07b42fd_1/g1/turn_events.jsonl`
- G1 evaluator: `results/controller_latest_real/07b42fd_1/g1/evaluator_result.json`
- Original baseline evaluator: AppWorld task output `collection/evaluator_result.json`

## Final assessment

The current G1 prompt is operationally compatible with the latest Controller backend, but this real trajectory provides no evidence that it improves correctness over G0. Its intended predicate-level and temporal reasoning did not materialize in the model's decisions.

The strongest next research direction exposed by this experiment is not “make the Controller prompt longer.” It is to design a temporally grounded Controller state representation in which errors, repairs, predicate evidence, and writes have explicit provenance and supersession relationships. Prompt quality should then be evaluated on top of that representation.
