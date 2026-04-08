# Contributing to AgentDisruptBench

Thank you for your interest in contributing to AgentDisruptBench! This guide will help you get started.

---

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Contributions](#making-contributions)
  - [Adding New Tasks](#adding-new-tasks)
  - [Adding New Disruption Types](#adding-new-disruption-types)
  - [Adding New Domains](#adding-new-domains)
  - [Adding New Framework Adapters](#adding-new-framework-adapters)
  - [Improving Metrics](#improving-metrics)
- [Code Style](#code-style)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)

---

## Code of Conduct

Please be respectful and constructive in all interactions. We follow the [Contributor Covenant](https://www.contributor-covenant.org/version/2/1/code_of_conduct/).

---

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/<your-username>/AgentDisruptBench.git`
3. Create a feature branch: `git checkout -b feature/your-feature`

---

## Development Setup

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install in development mode with all extras
pip install -e ".[dev,all]"

# Verify installation
python -c "import agentdisruptbench; print(agentdisruptbench.__version__)"

# Run tests
pytest tests/ -v
```

---

## Making Contributions

### Adding New Tasks

Tasks are defined in YAML files under `python/agentdisruptbench/tasks/builtin/`.

Each task must include:
```yaml
- task_id: <domain>_<number>       # e.g., retail_021
  title: "Short descriptive title"
  description: |
    Full natural language prompt given to the agent.
    Be specific about what the agent should do.
  difficulty: 3                     # 1-5
  required_tools: [tool_a, tool_b]
  expected_tool_call_depth: 2
  ground_truth:
    expected_outcome: "Description of success"
    required_tool_calls: [tool_a, tool_b]
    forbidden_tool_calls: []        # Optional
    correct_final_answer: null      # Or exact string
    evaluation_rubric:
      criterion_a: 0.50
      criterion_b: 0.50            # Must sum ≈ 1.0
    disruption_sensitive_tools: [tool_b]
    recovery_actions: [retry_tool_b]
```

**Guidelines:**
- Rubric weights must sum to approximately 1.0
- Use existing simulated tools from `tools/simulated_tools.py`
- Test that the task is solvable under `clean` profile
- Verify ground truth against actual simulated tool outputs

### Adding New Disruption Types

1. Add the type to `DisruptionType` enum in `python/agentdisruptbench/core/engine.py`
2. Create a handler function `_h_<name>()` following the existing pattern
3. Register it in the `_HANDLERS` dispatch table
4. Add it to the failure taxonomy in `python/agentdisruptbench/core/metrics.py`
5. Update relevant disruption profiles in `python/agentdisruptbench/core/profiles.py`
6. Write tests in `tests/test_engine.py`

### Adding New Domains

1. Create a new tool class in `python/agentdisruptbench/tools/simulated_tools.py` (e.g., `HealthcareTools`)
2. Register the tools in `get_all_tools()`
3. Create a YAML task file in `python/agentdisruptbench/tasks/builtin/<domain>.yaml`
4. Add the domain to the compensation pairs in `python/agentdisruptbench/core/state.py` (if applicable)
5. Update `DATASHEET.md` with the new domain information

### Adding New Framework Adapters

1. Create a new adapter in `python/agentdisruptbench/adapters/<framework>.py`
2. Extend `BaseAdapter` from `python/agentdisruptbench/adapters/base.py`
3. Create a runner in `python/agentdisruptbench/evaluation/runners/<framework>_runner.py`
4. Register the runner in `python/agentdisruptbench/evaluation/run_benchmark.py`
5. Add the framework to optional dependencies in `pyproject.toml`

### Improving Metrics

Any metric changes must:
- Update the `BenchmarkResult` dataclass if adding new fields
- Update `MetricsCalculator.compute()` to populate new fields
- Add tests for new metric computation
- Update the paper's metric definitions table (if applicable)

---

## Code Style

We use the following conventions:

- **Formatter**: [Ruff](https://github.com/astral-sh/ruff) (line-length: 120)
- **Type hints**: Required for all public APIs
- **File headers**: Every `.py` file must include the standard header block:
  ```python
  """
  AgentDisruptBench — <Module Name>
  ==================================

  File:        <filename>.py
  Purpose:     <description>
  Author:      AgentDisruptBench Contributors
  License:     MIT
  Created:     <date>
  Modified:    <date>
  """
  ```

Run formatting and linting:
```bash
ruff check .
ruff format .
mypy python/agentdisruptbench/
```

---

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=python/agentdisruptbench --cov-report=term-missing

# Run specific test file
pytest tests/test_engine.py -v

# Run specific test
pytest tests/test_engine.py::test_timeout_disruption -v
```

**Requirements:**
- All new features must have tests
- Tests should be deterministic (use fixed seeds)
- Coverage should remain above 80%

---

## Pull Request Process

1. Ensure all tests pass: `pytest tests/ -v`
2. Ensure code is formatted: `ruff format . && ruff check .`
3. Update documentation if behaviour changed
4. Add a changelog entry to `CHANGELOG.md`
5. Submit the PR with a clear description of changes
6. Link any related issues

### PR Title Convention

```
feat: add healthcare domain with 25 tasks
fix: correct compensation detection for nested entities
docs: update DATASHEET.md with new domain info
test: add coverage for cascading failure handler
refactor: simplify metrics calculator rubric evaluation
```

---

## Questions?

Open a [GitHub Issue](https://github.com/Kavirubc/AgentDisruptBench/issues) for any questions about contributing.
