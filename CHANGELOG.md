# Changelog

All notable changes to AgentDisruptBench will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.1.0] — 2026-04-03

### Added
- **Core Framework**
  - DisruptionEngine with 20 disruption types across 4 categories (Timing, HTTP, Response Content, Behavioral)
  - ToolProxy transparent wrapper with trace collection
  - TraceCollector with JSONL I/O for reproducible analysis
  - 9 built-in disruption profiles (`clean` → `hostile_environment`)
  - Seeded random for deterministic experiment replay

- **Task System**
  - 100 benchmark tasks across 4 domains (Retail, Travel, Finance, DevOps)
  - 80 standard tasks, 8 adversarial trap tasks, 8 impossible tasks, 4 handover tasks
  - Pydantic v2 schemas for Task, ToolSchema, and GroundTruth
  - YAML-based task registry

- **Mock Tools**
  - 30 deterministic mock tools across 4 domains
  - Hash-based reproducibility (same inputs → same outputs)
  - Internally consistent data (product IDs, customer IDs, order IDs)

- **Metrics & Evaluation**
  - MetricsCalculator with 30+ metrics across P0/P1/P2 tiers
  - R(k,ε,λ) reliability surface computation
  - Recovery strategy classification (RETRY, ALTERNATIVE, ESCALATION, GIVEUP, LUCKY)
  - AgentRx-aligned 9-category failure taxonomy
  - Stateful sandbox with compensation detection, idempotency violation detection, side-effect scoring

- **Framework Adapters**
  - LangChain / LangGraph adapter
  - OpenAI Function Calling adapter
  - AutoGen adapter
  - CrewAI adapter

- **Evaluation Runners**
  - SimpleRunner (no-LLM baseline)
  - OpenAIRunner (native function calling)
  - LangChainRunner (ReAct agent)
  - RACRunner (compensation-aware agent)
  - AutoGenRunner (two-agent pattern)
  - CrewAIRunner (Crew + Agent)

- **CLI & Reporting**
  - Unified CLI entry point (`python -m evaluation.run_benchmark`)
  - Markdown + JSON + CSV report generation
  - Per-task detailed logging
  - Rich CLI viewer for run logs

- **Infrastructure**
  - GitHub Actions CI pipeline
  - Ruff linting and formatting
  - mypy type checking configuration
  - pytest test suite
