---
language:
- en
license: mit
task_categories:
- text-generation
tags:
- agents
- evaluation-methodology
- tool-calling
- resilience
- disruption
- fault-injection
- LLM
- evaluation
pretty_name: AgentDisruptBench
size_categories:
- n<1K
---

# AgentDisruptBench

An evaluation methodology for measuring AI agent resilience under runtime tool-call disruptions.

## Overview

AgentDisruptBench provides a systematic methodology, backed by **100 base tasks and variants** across **4 domains**, a systematic **20-type disruption taxonomy**, and **9 disruption severity profiles**, to study and measure how well LLM-based agents handle real-world tool failures.

## Task Statistics

| Domain   | Standard | Adversarial | Impossible | Handover | Total |
|----------|:--------:|:-----------:|:----------:|:--------:|:-----:|
| Retail   | 20       | 2           | 2          | 1        | 25    |
| Travel   | 20       | 2           | 2          | 1        | 25    |
| Finance  | 20       | 2           | 2          | 1        | 25    |
| DevOps   | 20       | 2           | 2          | 1        | 25    |
| **Total**| **80**   | **8**       | **8**      | **4**    | **100** |

## Disruption Taxonomy (20 Types)

| Category | Types |
|----------|-------|
| Timing | timeout, latency |
| HTTP Status | http_429, http_401, http_403, http_500, http_502, http_503 |
| Response Content | malformed_json, truncated, null_response, missing_fields, type_mismatch, schema_drift, wrong_data |
| Behavioral | intermittent, flapping, quota_exhausted, auth_expiry, cascading |

## Key Metric: R(k, ε, λ) Reliability Surface

- **k-consistency**: Pass rate across repeated seeds (same task, same profile)
- **ε-robustness**: Pass rate across task-wording variants
- **λ-fault-tolerance**: Pass rate across disruption profiles

## Files

- `tasks/retail.yaml` — 20 retail domain tasks
- `tasks/travel.yaml` — 20 travel domain tasks
- `tasks/finance.yaml` — 20 finance domain tasks
- `tasks/devops.yaml` — 20 DevOps domain tasks
- `tasks/adversarial.yaml` — 8 adversarial trap tasks
- `tasks/impossible.yaml` — 8 impossible tasks
- `tasks/handover.yaml` — 4 handover tasks
- `tasks/variants.yaml` — 18 ε-robustness task variants
- `profiles/` — 9 disruption profile definitions

## Usage

```bash
pip install agentdisruptbench
```

```python
from agentdisruptbench import TaskRegistry, DisruptionEngine
registry = TaskRegistry.from_builtin()
tasks = registry.filter(domain="retail", max_difficulty=3)
```

## Citation

```bibtex
@inproceedings{agentdisruptbench2026,
  title={AgentDisruptBench: An Evaluation Methodology for AI Agent Resilience Under Runtime Tool-Call Disruptions},
  author={AgentDisruptBench Contributors},
  year={2026}
}
```

## License

MIT
