# Benchmarking External Agents with AgentDisruptBench

This example demonstrates how to benchmark **any** agent framework against AgentDisruptBench **without modifying the agent's source code**.

## Architecture

```
 Your Agent              ADB REST Server (:8080)            Simulated Tools
 ─────────              ──────────────────────             ────────────────
    │                         │                                  │
    │  POST /api/tools/X     │                                  │
    ├───────────────────────→│                                  │
    │                         │  ToolProxy + DisruptionEngine   │
    │                         │  (injects failures silently)    │
    │                         ├─────────────────────────────────→│
    │                         │←─────────────────────────────────┤
    │  Response (or failure)  │                                  │
    │←───────────────────────┤                                  │
    │                         │                                  │
```

Your agent **never knows** disruptions are being injected. It simply calls HTTP endpoints that look like real tool APIs. The ADB server sits in front, silently injecting timeouts, 500s, malformed responses, and other faults.

## Quick Start

```bash
# 1. Install dependencies
pip install agentdisruptbench[server] langgraph langchain-core httpx

# 2. Run the full benchmark (starts server, runs agent, scores results)
python examples/langgraph_fault_tolerance/run_external_benchmark.py

# 3. Run with specific profiles
python examples/langgraph_fault_tolerance/run_external_benchmark.py \
    --profiles clean hostile_environment \
    --seed 42
```

## Source Provenance

This example benchmarks the **LangGraph fault-tolerance cookbook** from LangChain's official examples:

```
https://github.com/langchain-ai/cookbooks/tree/main/python/langgraph/persistence/fault-tolerance
```

The original cookbook demonstrates LangGraph's `pending-writes` checkpoint mechanism for recovering from partial failures in parallel execution. Our `agent_http.py` preserves the original agent's exact retry/fallback architecture, replacing only the failure injection mechanism:

| Aspect | Original Cookbook | ADB Adapter (`agent_http.py`) |
|--------|-----------------|-------------------------------|
| Graph topology | StateGraph + conditional routing | ✅ Same |
| Retry logic | Progressive fallback (retry → skip → giveup) | ✅ Same |
| Checkpointing | SQLiteSaver / MemorySaver | ✅ Same (MemorySaver) |
| Failure source | `random.random() < 0.7` hardcoded in nodes | HTTP 500 / timeout from ADB proxy |
| Tool calls | Direct Python function calls | `httpx.post()` to ADB REST server |

The adapter makes the agent suitable for controlled, reproducible disruption benchmarking without altering its resilience logic.

## Files

| File | Purpose |
|------|---------|
| `agent_http.py` | LangGraph fault-tolerance agent adapted to call tools via HTTP |
| `run_external_benchmark.py` | Harness that orchestrates server + agent + scoring |
| `README.md` | This file |

## How the Benchmark Works

1. **Server starts** — The harness starts the ADB REST server (`server/app.py`) on a configurable port
2. **Profile configured** — Calls `POST /admin/setup_run` with the disruption profile (e.g., `hostile_environment`)
3. **Task begins** — Calls `POST /admin/start_task` to reset trace collection
4. **Agent runs** — The agent is launched as a subprocess with `ADB_SERVER_URL` set in environment. Agent calls tools via `POST /api/tools/{name}`
5. **Traces collected** — After the agent finishes, `POST /admin/end_task` returns all tool call traces and idempotency violations
6. **Scoring** — Traces are fed into `MetricsCalculator` to produce a `BenchmarkResult` with PRS scores

## Benchmarking Your Own Agent

To benchmark **your** agent, you only need to:

1. **Make your agent call tools via HTTP** — Instead of calling tools directly, `POST` to `{ADB_SERVER_URL}/api/tools/{tool_name}` with the tool's parameters as JSON body
2. **Read `ADB_SERVER_URL` from environment** — The harness sets this automatically
3. **Run the harness** with your agent command:

```bash
python examples/langgraph_fault_tolerance/run_external_benchmark.py \
    --agent-cmd "python path/to/your_agent.py" \
    --agent-id "my_custom_agent"
```

### Available Tools

The ADB server exposes all simulated tools from 4 domains:

- **Retail**: `search_products`, `check_inventory`, `place_order`, `get_order_status`, `process_refund`, `get_customer_profile`, `apply_coupon`, `update_cart`
- **Travel**: `search_flights`, `get_flight_details`, `book_flight`, `cancel_booking`, `search_hotels`, `check_hotel_availability`, `get_weather`, `currency_convert`
- **Finance**: `get_account_balance`, `transfer_funds`, `get_transaction_history`, `get_exchange_rate`, `validate_card`, `check_credit_limit`
- **DevOps**: `get_service_health`, `deploy_service`, `rollback_deployment`, `get_logs`, `get_metrics`, `run_tests`, `create_incident`, `resolve_incident`

### Disruption Profiles

| Profile | Description |
|---------|-------------|
| `clean` | No disruptions (baseline) |
| `mild_production` | 10-20% failure rate on select tools |
| `hostile_environment` | 50-80% failure rate across all tools |
| `cascading_failure` | Failures propagate across tool dependencies |

## Understanding Results

The harness produces scored metrics including:

- **Partial Score** — Weighted rubric-based task completion
- **Recovery Rate** — % of disruptions the agent recovered from
- **Retry Efficiency** — Successful retries / total retries
- **Resilience Ratio** — Disrupted success rate / clean success rate
- **Extra Tool Calls** — Cost of resilience (disrupted − clean calls)
- **Recovery Strategies** — RETRY, ALTERNATIVE, ESCALATION, GIVEUP

A **Disruption Degradation Curve** compares scores across profiles, showing how gracefully the agent degrades under increasing failure intensity.
