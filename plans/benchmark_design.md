# AgentDisruptBench — Benchmark Design Document

> **Last updated:** 2026-03-25 — Reflects codebase after YAML config + OpenAI integration.

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
| Total tasks | 80+ | **100** (25 × 4 domains, D1–D5) | ✅ |
| Task types | 3 | **84 standard + 8 adversarial + 8 impossible** | ✅ |
| Disruption types | 20 | 20 in 4 categories | ✅ |
| Profiles | 9+ | 9 built-in + YAML custom | ✅ |
| LLM Providers | 2+ | **Gemini + OpenAI** via shared LLM factory | ✅ |
| Runners | 3+ | **6** (simple, openai, langchain, autogen, crewai, rac) | ✅ |

### 1.2 Planning

Based on chaos engineering's *"steady-state hypothesis"* and PlanCraft's dependency graphs:

| Capability | What It Tests | How We Test It | Status |
|---|---|---|---|
| **Greedy planning** | Following best-next-step reaches solution | D1–D2 tasks: linear tool chains | ✅ |
| **Multi-turn branching** | Must choose the right branch, not just greedy | D3–D5: conditional logic ("book only if weather favorable") | ✅ |
| **Local replanning** | Recovery from each error type within a step | Engine injects per-call disruptions; `recovery_actions` per task | ✅ |
| **Long-horizon traps** | Correct early action looks wrong; "safe" action → catastrophe later | **Adversarial tasks** (8 tasks across domains) | ✅ |
| **Impossible tasks** | Agent must recognize & give up (per PlanCraft) | **Impossible tasks** (8 tasks) — agent must not call forbidden tools | ✅ |

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
| **Handover available** | Hand off without crashing | — | `handover_detected` metric |
| **Unprompted failures** | Handle errors not in prompt | All — injected transparently | Agent must infer from error |

### 1.4 Validation & Measurement

#### Outcome Metrics

| Metric | Description | Status |
|---|---|---|
| `task_success` | `partial_score >= 0.8` or exact answer match | ✅ |
| `partial_score` | Weighted rubric satisfaction (goal-sat score) | ✅ |
| `acknowledged_failure` | Agent communicated failure to user | ✅ |
| `attempted_alternative` | Agent tried different tool after failure | ✅ |

#### Resilience Metrics

| Metric | Description | Status |
|---|---|---|
| `recovery_rate` | recovered_failures / total_failures | ✅ |
| `mean_steps_to_recovery` | Avg tool calls between failure and recovery | ✅ |
| `retry_efficiency` | successful_retries / total_retries | ✅ |
| `resilience_ratio` | success_disrupted / success_clean | ✅ |
| `max_cascade_depth` | Consecutive cascade failures | ✅ |

#### Cost Metrics

| Metric | Description | Status |
|---|---|---|
| `total_tool_calls` / `extra_tool_calls` | Overhead from disruptions | ✅ |
| `total_latency_ms` / `extra_latency_ms` | Time overhead | ✅ |
| Token usage | Total tokens consumed | ✅ Runner-level |

#### State & Compensation Metrics (P0 — Implemented)

| Metric | Description | Status |
|---|---|---|
| `compensation_count` | Rollback/undo actions via entity-level pairing | ✅ |
| `compensation_success_rate` | Successful compensations / total attempts | ✅ |
| `side_effect_score` | Unintended state changes left unresolved | ✅ |
| `idempotency_violations` | Duplicate actions from retries (double-booking, etc.) | ✅ |
| `loop_count` | Repeated identical tool calls | ✅ |

#### Recovery Strategy Metrics (P1 — Implemented)

| Metric | Description | Status |
|---|---|---|
| `recovery_strategies` | List of classified strategies per recovery event | ✅ |
| `dominant_strategy` | Most frequent strategy (RETRY, ALTERNATIVE, ESCALATION, GIVEUP) | ✅ |
| `graceful_giveup` | Agent correctly refused impossible tasks | ✅ |

#### Planning & Diagnostic Metrics (P2 — Implemented)

| Metric | Description | Status |
|---|---|---|
| `planning_time_ratio` | Time on initial planning vs execution | ✅ |
| `handover_detected` | Agent suggested human handoff | ✅ |
| `tool_hallucination_rate` | Phantom actions/outputs vs TraceCollector reality | ✅ |
| `state_equivalent_success` | End-state equivalence instead of text match | ✅ |
| `budget_exceeded` | Fixed budget overrun detection | ✅ |
| `failure_categories` | AgentRx-aligned 9-category root-cause attribution | ✅ |

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
| **AgentDisruptBench** | **30** | **100** | **20 types, runtime** | **✅** | **✅** | **✅ Per-profile** |

---

## 4. Research-Informed Dimensions

### 4.1 Microsoft AgentRx's 9-Category Failure Taxonomy

| Category | Description | Our Coverage |
|---|---|---|
| Tool execution error | Tool fails to execute | ✅ HTTP errors, timeout |
| Wrong tool selection | Agent picks wrong tool | ✅ Detected via trace |
| Missing tool call | Agent skipped required tool | ✅ Measured via rubric |
| Wrong parameters | Correct tool, wrong inputs | ✅ Trace captures inputs |
| Incorrect reasoning | Logical error in plan | ⚠️ Needs LLM-as-judge |
| Hallucinated tool call | Phantom action/output | ✅ `tool_hallucination_rate` |
| Context loss | Agent forgets prior state | ⚠️ Need long-horizon tasks |
| Loop detection | Agent repeats actions | ✅ `loop_count` metric |
| Premature termination | Agent gives up too early | ✅ `acknowledged_failure` |

### 4.2 Tool Hallucination Types

| Type | Description | Status |
|---|---|---|
| **Phantom action** | Claims tool was called when it wasn't | ✅ Compare agent claim vs trace log |
| **Phantom output** | Invents tool results | ✅ Compare reported result vs `TraceCollector` data |
| **Schema drift** | Uses wrong parameter names | ✅ Engine has `SCHEMA_DRIFT` disruption |
| **Reality drift** | Draws wrong conclusions from correct output | ⚠️ Needs LLM-as-judge evaluation |

### 4.3 ReliabilityBench R(k,ε,λ) Reliability Surface

- **k** (consistency): pass rate over k repeated runs → ✅ Multi-seed evaluation via `--seeds`
- **ε** (robustness): performance under semantic task perturbations → ⚠️ Future: task rewording variants
- **λ** (fault tolerance): performance under increasing disruption intensity → ✅ Core strength via profiles

### 4.4 DeepMind Paper Findings (Toward a Science of Scaling Agent Systems)

| Finding | Implication | Our Response |
|---|---|---|
| **17× side-effect multiplication** in MAS | Error propagation is catastrophic | `max_cascade_depth` metric ✅ |
| **Capability saturation** at 55% single-agent | MAS doesn't help if single agent can't do 55% | Multi-runner comparison ✅ |
| **Communication overhead** in context passing | More agents = more context load | `expected_tool_call_depth` provides baseline ✅ |
| **Diminishing returns on token cost** | More tokens ≠ better results | `extra_tool_calls`, `extra_latency_ms` ✅ |

---

## 5. Current Architecture

### 5.1 Tools (30 total)

| Domain | Read-Only Tools | Side-Effect Tools (in **bold**) | Count |
|---|---|---|---|
| Retail | search_products, check_inventory, get_order_status, get_customer_profile | **place_order**, **process_refund**, **apply_coupon**, **update_cart** | 8 |
| Travel | search_flights, get_flight_details, search_hotels, check_hotel_availability, get_weather, currency_convert | **book_flight**, **cancel_booking** | 8 |
| Finance | get_account_balance, get_transaction_history, get_exchange_rate, validate_card, check_credit_limit | **transfer_funds** | 6 |
| DevOps | get_service_health, get_logs, get_metrics, run_tests | **deploy_service**, **rollback_deployment**, **create_incident**, **resolve_incident** | 8 |

### 5.2 Disruption Types (20)

| Category | Types |
|---|---|
| **Timing** (2) | `timeout`, `latency` |
| **HTTP Status** (6) | `http_429`, `http_401`, `http_403`, `http_500`, `http_502`, `http_503` |
| **Response Content** (7) | `malformed_json`, `truncated`, `null_response`, `missing_fields`, `type_mismatch`, `schema_drift`, `wrong_data` |
| **Behavioral** (5) | `intermittent`, `flapping`, `quota_exhausted`, `auth_expiry`, `cascading` |

### 5.3 Evaluation Harness

| Component | File | Purpose |
|---|---|---|
| **BaseRunner** | `evaluation/base_runner.py` | Abstract runner with `RunnerConfig` dataclass |
| **LLM Factory** | `evaluation/llm_factory.py` | Shared LLM creation (Gemini + OpenAI), provider auto-detection |
| **Config Loader** | `evaluation/config_loader.py` | Pydantic-style YAML config with `LLMConfig` + `BenchmarkYAMLConfig` |
| **Run Logger** | `evaluation/run_logger.py` | Shared structured JSONL event logger for `show_run.py` |
| **Run Benchmark** | `evaluation/run_benchmark.py` | Main CLI entry point with YAML + CLI merge strategy |
| **Show Run** | `evaluation/show_run.py` | Rich CLI renderer for run log visualization |

#### Runners

| Runner | File | Framework | LLM Providers |
|---|---|---|---|
| `simple` | `runners/simple_runner.py` | Direct tool calls (no LLM) | None |
| `openai` | `runners/openai_runner.py` | OpenAI API | OpenAI |
| `langchain` | `runners/langchain_runner.py` | LangChain + LangGraph ReAct | Gemini, OpenAI |
| `rac` | `runners/rac_runner.py` | RAC (Recover-Analyze-Compensate) | Gemini, OpenAI |
| `autogen` | `runners/autogen_runner.py` | AutoGen | Gemini, OpenAI |
| `crewai` | `runners/crewai_runner.py` | CrewAI | Gemini, OpenAI |

### 5.4 Configuration System

```
config/
├── benchmark.yaml        # Top-level benchmark settings (profiles, seeds, domains)
└── llm/
    ├── gemini-2.5-flash.yaml   # Gemini Flash preset
    ├── gpt-4o.yaml             # GPT-4o preset
    ├── gpt-4o-mini.yaml        # GPT-4o Mini preset
    └── gpt-5-mini.yaml         # GPT-5 Mini preset
```

**Usage pattern** (inspired by pentest-evo):
```bash
# YAML-based (recommended)
python -m evaluation.run_benchmark \
  --config config/benchmark.yaml \
  --llm-config config/llm/gpt-5-mini.yaml

# CLI overrides
python -m evaluation.run_benchmark \
  --runner rac --model gpt-5-mini \
  --profiles clean hostile_environment \
  --max-difficulty 3

# View results
python evaluation/show_run.py
python evaluation/show_run.py --run-id <run_id>
```

### 5.5 Logging & Reporting

| Output | Location | Format | Viewer |
|---|---|---|---|
| Run event logs | `logs/<run_id>/run_log.jsonl` | Structured JSONL | `show_run.py` |
| Benchmark report | `results/report.md` | Markdown | Any viewer |
| Results data | `results/results.json` | JSON | Programmatic |
| Summary stats | `results/summary.json` | JSON | Programmatic |
| Per-task logs | `results/task_logs/` | JSON per task | Programmatic |

One JSONL run log is emitted per **(profile × seed)** combination, so each is independently viewable via `show_run.py`.

---

## 6. Remaining Gaps — Priority Ranked

### Completed (formerly P0–P2)

All items from the original P0, P1, and P2 gap lists have been implemented:

- ✅ **Stateful sandbox** — `StateManager` with in-memory mutable state
- ✅ **Compensation metrics** — Entity-level pairing in `MetricsCalculator`
- ✅ **Idempotency violation detection** — `idempotency_violations` metric
- ✅ **Loop detection** — `loop_count` metric
- ✅ **Adversarial tasks** — 8 long-horizon adversarial scenarios
- ✅ **Impossible tasks** — 8 unsolvable scenarios with forbidden tool guards
- ✅ **Recovery strategy classification** — RETRY, ALTERNATIVE, ESCALATION, GIVEUP
- ✅ **Planning time ratio** — `planning_time_ratio` metric
- ✅ **Handover testing** — `handover_detected` metric
- ✅ **Tool hallucination detection** — `tool_hallucination_rate` metric
- ✅ **AgentRx-aligned failure taxonomy** — 9-category `failure_categories` dict

### Future Work

| # | Gap | Why Useful | Source |
|---|---|---|---|
| 1 | **R(k,ε,λ) reliability surface** | Multi-seed, multi-perturbation scoring | ReliabilityBench |
| 2 | **Task rewording variants** (ε axis) | Semantic robustness testing | ReliabilityBench |
| 3 | **Action metamorphic relations** | End-state equivalence evaluation | ReliabilityBench |
| 4 | **LLM-as-judge evaluation** | Reality drift / incorrect reasoning detection | AgentRx |
| 5 | **Multi-agent support** | Extend to MAS topologies for error multiplication testing | DeepMind paper |
| 6 | **Token budget constraints** | Fixed-budget tasks to test efficiency under pressure | DeepMind paper |

---

## 7. Summary

> **AgentDisruptBench is the first benchmark that combines controlled runtime disruption injection with side-effect recovery evaluation.** Existing benchmarks either test planning without failures (REALM-Bench), test failures without side effects (ReliabilityBench), hide failures from agents (Finance-Agent), or flag side effects without testing recovery (WorkBench). AgentDisruptBench closes all three loops:
>
> 1. **Inject** → controlled disruptions at runtime via probability-gated profiles
> 2. **Measure** → recovery, compensation, and collateral damage via trace analysis
> 3. **Vary** → same task + different profiles makes pre-solved plans worthless
>
> The benchmark draws from chaos engineering (fault injection patterns), distributed systems (circuit breaker, idempotency, saga compensation), and the latest agent research (AgentRx failure taxonomy, ReliabilityBench reliability surfaces, tool hallucination detection).
>
> **v1.0 ships with:** 100 tasks, 30 tools, 20 disruption types, 9 profiles, 6 framework runners, YAML-based configuration, Gemini + OpenAI support, structured JSONL event logging, and a comprehensive metrics suite covering outcome, resilience, cost, compensation, recovery strategy, and diagnostic dimensions.
