# AgentDisruptBench — Finalized Benchmark Design

## What Is This?

AgentDisruptBench is a benchmark for evaluating AI agent **resilience under runtime disruptions with side effects**. Unlike existing benchmarks that test planning or tool use in isolation, AgentDisruptBench injects controlled, dynamic faults into tool calls at runtime — forcing agents to recover, compensate, roll back, or gracefully degrade.

> [!IMPORTANT]
> **Core thesis:** A finalized plan from a successful run should NOT guarantee success on a subsequent run, because disruption profiles change the runtime landscape.

---

## 1. What We Test (Evaluation Dimensions)

### 1.1 Complexity

| Dimension | Target | Current State | Status |
|---|---|---|---|
| Total tools | 25–30 | 30 | ✅ |
| Tools per domain | 5–6 | 6–8 per domain | ✅ |
| Side-effect tools per domain | 2 | StateManager with in-memory mutable state | ✅ |
| Total tasks | 80+ | 80 (20 × 4 domains, D1–D5) | ✅ |
| Disruption types | 20 | 20 in 4 categories | ✅ |
| Profiles | 9+ | 9 built-in + YAML custom | ✅ |

### 1.2 Planning

Based on chaos engineering's *"steady-state hypothesis"* and PlanCraft's dependency graphs:

| Capability | What It Tests | How We Test It |
|---|---|---|
| **Greedy planning** | Following best-next-step reaches solution | D1–D2 tasks: linear tool chains |
| **Multi-turn branching** | Must choose the right branch, not just greedy | D3–D5: conditional logic ("book only if weather favorable") |
| **Local replanning** | Recovery from each error type within a step | Engine injects per-call disruptions; `recovery_actions` per task |
| **Long-horizon traps** | Correct early action looks wrong; "safe" action → catastrophe later | **NEW**: Adversarial task scenarios needed |
| **Impossible tasks** | Agent must recognize & give up (per PlanCraft) | **NEW**: Unsolvable tasks where no valid plan exists |

### 1.3 Failure Handling (Primary Differentiator)

Drawing from distributed systems fault taxonomy (infrastructure, network, dependency, state corruption, software faults) and mapping to agent-tool interactions:

| Failure Category | Can the Agent… | Disruption Types | Recovery Pattern |
|---|---|---|---|
| **Slow responses** | Detect via timeout, wait or retry | `TIMEOUT`, `LATENCY` | Retry with backoff |
| **Transient errors** | Recover with smart retry | `INTERMITTENT`, `FLAPPING`, `HTTP_5xx` | Exponential backoff + jitter |
| **Rate limits / Quota** | Back off or use alternative | `HTTP_429`, `QUOTA_EXHAUSTED` | Backoff, alternative tool path |
| **Auth failures** | Re-authenticate or escalate | `HTTP_401`, `HTTP_403`, `AUTH_EXPIRY` | Token refresh, escalation |
| **Corrupted responses** | Detect & discard bad data | `WRONG_DATA`, `TYPE_MISMATCH`, `SCHEMA_DRIFT` | Validation + re-request |
| **Partial responses** | Work with incomplete data | `TRUNCATED`, `MISSING_FIELDS`, `NULL_RESPONSE` | Request full data / degrade |
| **Malformed output** | Parse or request clean data | `MALFORMED_JSON` | Re-request |
| **Cascading failures** | Isolate blast radius | `CASCADING` | Circuit breaker pattern |
| **Permanent failures** | Find alternative if exists | All permanent types | Alternative tool path |
| **No alternative exists** | Fail gracefully, communicate | — | `acknowledged_failure` metric |
| **Handover available** | Hand off without crashing | — | **NEW**: Explicit handover tasks |
| **Unprompted failures** | Handle errors not in prompt | All — injected transparently | Agent must infer from error |

### 1.4 Validation & Measurement

#### Outcome Metrics (What happened?)

| Metric | Description | Source |
|---|---|---|
| `task_success` | `partial_score >= 0.8` or exact answer match | ours |
| `partial_score` | Weighted rubric satisfaction (goal-sat score) | ours |
| `acknowledged_failure` | Agent communicated failure to user | ours |
| `attempted_alternative` | Agent tried different tool after failure | ours |

#### Resilience Metrics (How well did it recover?)

| Metric | Description | Source |
|---|---|---|
| `recovery_rate` | recovered_failures / total_failures | ours |
| `mean_steps_to_recovery` | Avg tool calls between failure and recovery | ours |
| `retry_efficiency` | successful_retries / total_retries | ours |
| `resilience_ratio` | success_disrupted / success_clean | ours |
| `max_cascade_depth` | Consecutive cascade failures | ours |

#### Cost Metrics (What was the price?)

| Metric | Description | Source |
|---|---|---|
| `total_tool_calls` / `extra_tool_calls` | Overhead from disruptions | ours |
| `total_latency_ms` / `extra_latency_ms` | Time overhead | ours |
| Token usage | Total tokens consumed | Runner-level tracking |

#### NEW Metrics to Add (from research)

| Metric | Description | Inspired By |
|---|---|---|
| `compensation_count` | Rollback/undo actions attempted after side effects | SagaLLM, WorkBench gap |
| `compensation_success_rate` | Successful compensations / total attempts | SagaLLM |
| `side_effect_score` | Unintended state changes left unresolved | WorkBench |
| `idempotency_violations` | Duplicate actions from retries (double-booking, etc.) | Distributed systems |
| `planning_time_ratio` | Time on initial planning vs inter-step reasoning | User requirement |
| `recovery_strategy_classification` | Was recovery smart or lucky? (categorize: retry, alternative, escalate, give-up) | User requirement |
| R(k,ε,λ) reliability surface | Consistency × Robustness × Fault tolerance | ReliabilityBench |
| `tool_hallucination_rate` | Phantom actions, phantom outputs, reality drift | Tool hallucination research |

---

## 2. Controlled Disruption Profiles — The Killer Feature

> [!CAUTION]
> This is the **primary differentiator** from all existing benchmarks.

### Why It Matters

Every other benchmark has **static, deterministic disruptions**:
- REALM-Bench: disruptions hardcoded in text prompts
- WorkBench: deterministic sandbox — same inputs → same outputs
- PlanCraft: missing items baked into task generation
- BrowseComp-Plus: fixed offline corpus eliminates dynamic failures
- Finance-Agent: harness auto-retries HTTP 429 errors, hiding them from agent

### How We're Different

- **Same task, different profiles**: Run `travel_018` under `clean`, `mild_production`, `hostile_environment` → same task, different runtime chaos
- **Probability-based injection**: Disruptions fire per `probability` config, seeded for reproducibility
- **Pre-solved plans are useless**: Change the seed → different disruptions fire → memorized plan fails
- **Controlled chaos budget**: Exact disruption counts per profile are known, enabling fair comparison

### Current Profiles (9)

| Profile | Disruptions | Use Case |
|---|---|---|
| `clean` | None | Baseline measurement |
| `mild_production` | Latency, 429, truncation | Typical production API noise |
| `moderate_production` | Timeout, 429, 500, malformed, missing fields | Degraded production |
| `hostile_environment` | 8 disruption types at high probability | Worst-case stress test |
| `auth_pressure` | 401 + auth expiry after N calls | Token management testing |
| `quota_pressure` | 429 + quota exhaustion after N calls | Rate limit strategy |
| `data_corruption` | 6 data-quality disruption types | Response validation testing |
| `cascading_failure` | Cascade from payment → downstream | Blast radius isolation |
| `flapping_services` | 50% flapping + intermittent every 3rd call | Retry strategy testing |

---

## 3. What Existing Benchmarks Miss

### Detailed Gap Analysis

| Benchmark | Tools | Tasks | Disruptions | Side Effects | Recovery Testing | Dynamic Profiles |
|---|---|---|---|---|---|---|
| **REALM-Bench** | ❌ | ✅ | In-prompt only | ❌ | ❌ | ❌ |
| **Tau2-Bench** | ✅ | ✅ | State consistency | Partial | ❌ | ❌ |
| **WorkBench** | 26 | 690 | Soft "not found" | ✅ Flagged | ❌ Never tested | ❌ |
| **Finance-Agent** | 4 | ✅ | Hidden by harness | ❌ | ❌ | ❌ |
| **BrowseComp-Plus** | ❌ | ✅ | Hard negatives | ❌ | ❌ | ❌ |
| **PlanCraft** | ✅ | ✅ | Impossible tasks | ❌ | ❌ | ❌ |
| **AppWorld** | ✅ Many | ✅ | ❌ | ❌ in eval | ❌ | ❌ |
| **Hell or High Water** | SQL only | ✅ | ❌ | ❌ | ❌ | ❌ |
| **ReliabilityBench** | ✅ | ✅ | Chaos engineering | ❌ | Partial | Partial (λ-levels) |
| **AgentDisruptBench** | **30** | **80** | **20 types, runtime** | **✅** | **✅** | **✅ Per-profile** |

### The Key Gap We Fill

```text
WorkBench  →  Shows agents CAUSE side effects (55% in analytics domain)
Finance-Agent  →  Shows agents NEED error handling (but harness hides it)
ReliabilityBench  →  Shows agents BREAK under chaos (but no side-effect recovery)
SagaLLM  →  Shows Saga PATTERN works (but static disruptions, pre-baked plan succeeds on re-run)

AgentDisruptBench  →  DETECT + RECOVER + COMPENSATE under DYNAMIC runtime chaos
```

---

## 4. New Dimensions from Research

### 4.1 Microsoft AgentRx's 9-Category Failure Taxonomy

AgentRx (Mar 2026) introduced a systematic debugging framework with a 9-category failure taxonomy and 23.6% improvement in failure localization. We should align our evaluation traces with these categories for **root-cause attribution**:

| Category | Description | Our Coverage |
|---|---|---|
| Tool execution error | Tool fails to execute | ✅ HTTP errors, timeout |
| Wrong tool selection | Agent picks wrong tool | ⚠️ Can detect via trace |
| Missing tool call | Agent skipped required tool | ✅ Measured via rubric |
| Wrong parameters | Correct tool, wrong inputs | ⚠️ Trace captures inputs |
| Incorrect reasoning | Logical error in plan | ⚠️ Needs LLM-as-judge |
| Hallucinated tool call | Phantom action/output | **Future: P2** — need to add |
| Context loss | Agent forgets prior state | **Future: P1** — need long-horizon tasks |
| Loop detection | Agent repeats actions | ✅ **Implemented** — `loop_count` metric |
| Premature termination | Agent gives up too early | ✅ `acknowledged_failure` |

### 4.2 Tool Hallucination Types

From tool-call verification research — agents don't just fail, they *hallucinate* tool interactions:

| Type | Description | How to Test |
|---|---|---|
| **Phantom action** | Claims tool was called when it wasn't | Compare agent claim vs trace log |
| **Phantom output** | Invents tool results | Compare reported result vs actual `TraceCollector` data |
| **Schema drift** | Uses wrong parameter names | Engine already has `SCHEMA_DRIFT` disruption |
| **Reality drift** | Draws wrong conclusions from correct output | Needs LLM-as-judge evaluation |

### 4.3 Idempotency Testing

From distributed systems and production AI patterns — retries must not cause duplicate side effects:

| Scenario | Expected Behavior | Test Method |
|---|---|---|
| `book_flight` fails, agent retries, first call actually succeeded | Agent should check before re-booking | Needs stateful sandbox |
| `transfer_funds` times out (actually executed), agent retries | Agent should check balance first | Needs idempotency keys |
| `place_order` + retry → double order | Agent should detect duplicate | Needs order state tracking |

### 4.4 Circuit Breaker & Bulkhead Patterns

From production AI resilience patterns — agents should exhibit distributed systems resilience behaviors:

| Pattern | What an Agent Should Do | How to Test |
|---|---|---|
| **Circuit breaker** | Stop calling a tool after N consecutive failures | `flapping_services` or `quota_pressure` profile |
| **Bulkhead** | Isolate failing domain, continue with others | Cross-domain tasks + cascading profile |
| **Exponential backoff** | Increase delay between retries | Measure `mean_steps_to_recovery` trend |

### 4.5 ReliabilityBench's Reliability Surface R(k,ε,λ)

The R(k,ε,λ) model provides a mathematically rigorous way to evaluate agents across three axes simultaneously:

- **k** (consistency): pass rate over k repeated runs → we can add multi-seed evaluation
- **ε** (robustness): performance under semantic task perturbations → we can add task rewording variants
- **λ** (fault tolerance): performance under increasing disruption intensity → **already our core strength** via profiles

We should adopt this as a scoring framework to produce a **reliability surface** per agent.

### 4.6 Action Metamorphic Relations

From ReliabilityBench — define correctness by **end-state equivalence** rather than text similarity. This means:
- Two different agent traces that achieve the same final state are both "correct"
- Allows flexible plan execution paths
- Better evaluates agents that find creative solutions under disruption

---

## 5. DeepMind Paper Findings (Toward a Science of Scaling Agent Systems)

| Finding | Implication | Our Response |
|---|---|---|
| **17× side-effect multiplication** in MAS | Error propagation is catastrophic | `max_cascade_depth` metric ✅ |
| **Capability saturation** at 55% single-agent | MAS doesn't help if single agent can't do 55% | Validate single-agent baseline first |
| **Communication overhead** in context passing | More agents = more context load | `expected_tool_call_depth` provides baseline |
| **Diminishing returns on token cost** | More tokens ≠ better results | `extra_tool_calls`, `extra_latency_ms` ✅ |
| **R² = see abstract** | Mathematical model for scaling | Adopt reliability surface model |

---

## 6. Current Architecture Inventory

### Tools (30 total)

| Domain | Read-Only Tools | Side-Effect Tools (in **bold**) | Count |
|---|---|---|---|
| Retail | search_products, check_inventory, get_order_status, get_customer_profile | **place_order**, **process_refund**, **apply_coupon**, **update_cart** | 8 |
| Travel | search_flights, get_flight_details, search_hotels, check_hotel_availability, get_weather, currency_convert | **book_flight**, **cancel_booking** | 8 |
| Finance | get_account_balance, get_transaction_history, get_exchange_rate, validate_card, check_credit_limit | **transfer_funds** | 6 |
| DevOps | get_service_health, get_logs, get_metrics, run_tests | **deploy_service**, **rollback_deployment**, **create_incident**, **resolve_incident** | 8 |

### Disruption Types (20)

| Category | Types |
|---|---|
| **Timing** (2) | `timeout`, `latency` |
| **HTTP Status** (6) | `http_429`, `http_401`, `http_403`, `http_500`, `http_502`, `http_503` |
| **Response Content** (7) | `malformed_json`, `truncated`, `null_response`, `missing_fields`, `type_mismatch`, `schema_drift`, `wrong_data` |
| **Behavioral** (5) | `intermittent`, `flapping`, `quota_exhausted`, `auth_expiry`, `cascading` |

---

## 7. Gaps to Close — Priority Ranked

### P0 — Must Have for v1.0

| # | Gap | Why Critical | Source |
|---|---|---|---|
| 1 | **Stateful sandbox** — Mutable state layer so side-effect tools actually modify DB | Without this, we can't measure compensation or side-effect damage | WorkBench findings (55% side effects), user requirement |
| 2 | **Compensation metrics** — Track rollback/undo attempts and success rate | Core thesis: "can agent fix what it broke?" | SagaLLM, unique differentiator |
| 3 | **Idempotency violation detection** — Detect duplicate side effects from retries | Most dangerous failure mode in production (double-charges, double-bookings) | Distributed systems, production AI patterns |
| 4 | **Loop detection metric** — Count repeated identical tool calls | Agents frequently enter infinite loops under disruption | AgentRx, PlanCraft |

### P1 — Should Have for v1.0

| # | Gap | Why Valuable | Source |
|---|---|---|---|
| 5 | **Long-horizon adversarial tasks** — Tasks where greedy actions cause later catastrophe | Tests genuine planning depth, not just reactive recovery | User requirement, PlanCraft |
| 6 | **Impossible tasks** — Unsolvable scenarios where agent must recognize and give up | Tests premature termination vs. pathological people-pleasing | PlanCraft findings |
| 7 | **Recovery strategy classification** — Categorize: smart retry vs lucky retry vs alternative path vs escalation | "When the agent recovers, was it the right strategy?" | User requirement |
| 8 | **R(k,ε,λ) reliability surface** — Multi-seed, multi-perturbation, multi-fault evaluation | Mathematically rigorous comparison framework | ReliabilityBench |

### P2 — Nice to Have

| # | Gap | Why Useful | Source |
|---|---|---|---|
| 9 | **Planning vs reasoning time split** | Diagnostic: where does the agent spend its budget? | User requirement |
| 10 | **Handover testing** | Explicit tasks where correct action = hand off to human | User requirement |
| 11 | **Tool hallucination detection** | Track phantom actions/outputs vs `TraceCollector` reality | Tool hallucination research |
| 12 | **Action metamorphic relations** | End-state equivalence instead of text-match evaluation | ReliabilityBench |
| 13 | **Multi-agent support** | Extend to MAS topologies for error multiplication testing | DeepMind paper |
| 14 | **Token budget constraints** | Fixed-budget tasks to test efficiency under pressure | DeepMind paper |
| 15 | **AgentRx-aligned failure taxonomy** | 9-category root-cause attribution in traces | Microsoft AgentRx |

---

## 8. Summary

> **AgentDisruptBench is the first benchmark that combines controlled runtime disruption injection with side-effect recovery evaluation.** Existing benchmarks either test planning without failures (REALM-Bench), test failures without side effects (ReliabilityBench), hide failures from agents (Finance-Agent), or flag side effects without testing recovery (WorkBench). AgentDisruptBench closes all three loops:
>
> 1. **Inject** → controlled disruptions at runtime via probability-gated profiles
> 2. **Measure** → recovery, compensation, and collateral damage via trace analysis
> 3. **Vary** → same task + different profiles makes pre-solved plans worthless
>
> The benchmark draws from chaos engineering (fault injection patterns), distributed systems (circuit breaker, idempotency, saga compensation), and the latest agent research (AgentRx failure taxonomy, ReliabilityBench reliability surfaces, tool hallucination detection).
