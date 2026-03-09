# AgentDisruptBench — Full Implementation Plan

A novel open-source benchmark framework for evaluating AI agent resilience under runtime disruptions. Targets NeurIPS Datasets & Benchmarks 2026.

## User Review Required

> [!IMPORTANT]
> This is a **massive** implementation (~44 files across Python + Go + Docker). I propose implementing in 9 phases with modular git commits per phase. Due to conversation context limits, I'll implement Phase 1–5 (Python SDK complete) first, verify with tests, then continue with Phases 6–8 (Go network layer + infra + docs) in follow-up turns.

> [!WARNING]
> Every source file will include a mandatory header block describing the file's purpose, author, and key definitions per your requirement. This convention will be documented so future contributors maintain it.

> [!NOTE]
> Credits will be given to tau-bench (Sierra Research), REALM-Bench, ReliabilityBench (arXiv:2601.06112), and all relevant prior work throughout README and ARCHITECTURE.md.

---

## Proposed Changes

### Phase 1: Python SDK Core

#### [NEW] [\_\_init\_\_.py](file:///Users/kaviru/Downloads/agent-disrupt-bench/AgentDisruptBench/python/agentdisruptbench/__init__.py)
Package initialization — version, public API exports.

#### [NEW] [engine.py](file:///Users/kaviru/Downloads/agent-disrupt-bench/AgentDisruptBench/python/agentdisruptbench/core/engine.py)
`DisruptionEngine` class implementing all 20 disruption types. Seeded RNG, thread-safe state, per-tool call counters for stateful disruptions.

#### [NEW] [trace.py](file:///Users/kaviru/Downloads/agent-disrupt-bench/AgentDisruptBench/python/agentdisruptbench/core/trace.py)
`TraceCollector` — thread-safe ToolCallTrace recording, JSONL serialization.

#### [NEW] [proxy.py](file:///Users/kaviru/Downloads/agent-disrupt-bench/AgentDisruptBench/python/agentdisruptbench/core/proxy.py)
`ToolProxy` + `ToolCallTrace` dataclass — wraps any callable through DisruptionEngine.

#### [NEW] [profiles.py](file:///Users/kaviru/Downloads/agent-disrupt-bench/AgentDisruptBench/python/agentdisruptbench/core/profiles.py)
9 built-in disruption profiles + YAML loading.

#### [NEW] [metrics.py](file:///Users/kaviru/Downloads/agent-disrupt-bench/AgentDisruptBench/python/agentdisruptbench/core/metrics.py)
`MetricsCalculator` + `BenchmarkResult` dataclass with all resilience metrics.

---

### Phase 2: Task System

#### [NEW] [schemas.py](file:///Users/kaviru/Downloads/agent-disrupt-bench/AgentDisruptBench/python/agentdisruptbench/tasks/schemas.py)
Pydantic v2: `Task`, `ToolSchema`, `GroundTruth` models.

#### [NEW] [mock_tools.py](file:///Users/kaviru/Downloads/agent-disrupt-bench/AgentDisruptBench/python/agentdisruptbench/tools/mock_tools.py)
Deterministic mock tools for 4 domains (Retail, Travel, Finance, DevOps) — ~30 tool functions with Faker-seeded data.

#### [NEW] [registry.py (tools)](file:///Users/kaviru/Downloads/agent-disrupt-bench/AgentDisruptBench/python/agentdisruptbench/tools/registry.py)
`ToolRegistry`: maps tool names to callables.

#### [NEW] [registry.py (tasks)](file:///Users/kaviru/Downloads/agent-disrupt-bench/AgentDisruptBench/python/agentdisruptbench/tasks/registry.py)
`TaskRegistry`: load/filter/iterate tasks from YAML.

#### [NEW] [generator.py](file:///Users/kaviru/Downloads/agent-disrupt-bench/AgentDisruptBench/python/agentdisruptbench/tasks/generator.py)
`SyntheticTaskGenerator` — LLM-powered task generation with tau-bench/REALM-bench mining.

#### [NEW] Built-in task YAML files (4 files)
`retail.yaml`, `travel.yaml`, `finance.yaml`, `devops.yaml` — 20 tasks each.

---

### Phase 3: Framework Adapters

#### [NEW] [base.py](file:///Users/kaviru/Downloads/agent-disrupt-bench/AgentDisruptBench/python/agentdisruptbench/adapters/base.py)
`BaseAdapter` ABC.

#### [NEW] [langchain.py](file:///Users/kaviru/Downloads/agent-disrupt-bench/AgentDisruptBench/python/agentdisruptbench/adapters/langchain.py)
LangChain/LangGraph adapter — `DisruptedLangChainTool` with Pydantic v2 `PrivateAttr`.

#### [NEW] [openai.py](file:///Users/kaviru/Downloads/agent-disrupt-bench/AgentDisruptBench/python/agentdisruptbench/adapters/openai.py)
OpenAI function calling adapter — dispatch + tool message builder.

#### [NEW] [autogen.py](file:///Users/kaviru/Downloads/agent-disrupt-bench/AgentDisruptBench/python/agentdisruptbench/adapters/autogen.py)
AutoGen 0.2/0.4 dual-version support.

#### [NEW] [crewai.py](file:///Users/kaviru/Downloads/agent-disrupt-bench/AgentDisruptBench/python/agentdisruptbench/adapters/crewai.py)
CrewAI adapter with cache bypass.

---

### Phase 4: Benchmark Harness

#### [NEW] [evaluator.py](file:///Users/kaviru/Downloads/agent-disrupt-bench/AgentDisruptBench/python/agentdisruptbench/harness/evaluator.py)
Outcome evaluation against ground truth.

#### [NEW] [runner.py](file:///Users/kaviru/Downloads/agent-disrupt-bench/AgentDisruptBench/python/agentdisruptbench/harness/runner.py)
`BenchmarkRunner` orchestrating task × profile runs.

#### [NEW] [reporter.py](file:///Users/kaviru/Downloads/agent-disrupt-bench/AgentDisruptBench/python/agentdisruptbench/harness/reporter.py)
JSON, Markdown, CSV report generation.

---

### Phase 5: Config, Init, and Tests

#### [NEW] [pyproject.toml](file:///Users/kaviru/Downloads/agent-disrupt-bench/AgentDisruptBench/pyproject.toml)
Modern Python packaging: build system, dependencies, optional extras for frameworks.

#### [NEW] Tests (4 files)
`test_engine.py`, `test_proxy.py`, `test_adapters.py`, `test_metrics.py` with full coverage per spec.

---

### Phase 6: Go Network Layer

#### [NEW] Go types, engine, profiles, ext_proc server
Types, disruption engine mirroring Python, ext_proc gRPC handler, interceptor, and trace store.

#### [NEW] [go.mod](file:///Users/kaviru/Downloads/agent-disrupt-bench/AgentDisruptBench/network/go.mod)
Go module definition.

---

### Phase 7: Infrastructure

#### [NEW] [envoy.yaml](file:///Users/kaviru/Downloads/agent-disrupt-bench/AgentDisruptBench/envoy/envoy.yaml)
Envoy config with ext_proc filter.

#### [NEW] Docker Compose files + schema.sql
Complete benchmark environment.

---

### Phase 8: Examples + Documentation

#### [NEW] Example scripts (4 files)
LangChain, OpenAI, network agent examples with Dockerfiles.

#### [MODIFY] [README.md](file:///Users/kaviru/Downloads/agent-disrupt-bench/AgentDisruptBench/README.md)
Publication-quality README with badges, architecture diagrams, benchmarks, citation, credits.

#### [NEW] [ARCHITECTURE.md](file:///Users/kaviru/Downloads/agent-disrupt-bench/AgentDisruptBench/ARCHITECTURE.md)
Detailed technical architecture document.

---

## Development Conventions

1. **Mandatory file header** — Every code file starts with a docstring/comment block:
   - File name, purpose, key classes/functions defined
   - Author, license
   - Date created and last modified
2. **Python venv** — All Python commands use `.venv`
3. **Implementation plans** → saved to `plans/` folder
4. **Modular git commits** — one commit per phase

---

## Verification Plan

### Automated Tests (Phase 5)

```bash
cd AgentDisruptBench
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest python/tests/ -v
```

Expected: All tests pass covering engine (20 disruption types, targeting, stateful disruptions, determinism), proxy (traces, latency), adapters (wrap/unwrap, ImportError), metrics (recovery, resilience ratio, rubric).

### LangChain Example Run

```bash
cd AgentDisruptBench
source .venv/bin/activate
python examples/python/langchain_react_agent.py --profile mild_production --tasks 5
```

Expected: Prints results table with columns: Task ID, Profile, Success, Partial Score, Recovery Rate, Extra Tool Calls.

### Go Build

```bash
cd AgentDisruptBench/network
go build ./...
go test ./...
```

### Docker Compose Validation

```bash
cd AgentDisruptBench/deploy
docker compose -f docker-compose.yml -f docker-compose.agent.yml config
```

### Manual Verification
- Review each file has the mandatory header block
- Confirm README reads as a research paper landing page
- Confirm credits to tau-bench, REALM-Bench, ReliabilityBench are present
