# Gemini Adapter + Evaluation Runner System

## Goal

1. **Gemini Adapter** — Support Google Gemini function calling via the `google-genai` SDK.
2. **Evaluation Folder** — Self-contained runner scripts (like REALM-Bench and tau2-bench) so users can plug in any framework and evaluate agents with one command.

---

## Proposed Changes

### Adapters

#### [NEW] [gemini.py](file:///Users/kaviru/Downloads/agent-disrupt-bench/AgentDisruptBench/python/agentdisruptbench/adapters/gemini.py)
- `GeminiAdapter(BaseAdapter)` for the `google-genai` SDK
- Intercepts at dispatch time (like the OpenAI adapter)
- `dispatch()` — executes a single tool call from Gemini's `function_call` response
- `build_tool_parts()` — builds `types.Part` tool result parts for the next turn
- Guarded import of `google.genai`

---

### Evaluation System

#### [NEW] [evaluation/__init__.py](file:///Users/kaviru/Downloads/agent-disrupt-bench/AgentDisruptBench/evaluation/__init__.py)
- Package init

#### [NEW] [evaluation/base_runner.py](file:///Users/kaviru/Downloads/agent-disrupt-bench/AgentDisruptBench/evaluation/base_runner.py)
- `BaseAgentRunner(ABC)` — abstract runner contract: `setup()`, `run_task(task, tools) → str`, `teardown()`
- Tracks timing, token usage, memory stats

#### [NEW] [evaluation/run_benchmark.py](file:///Users/kaviru/Downloads/agent-disrupt-bench/AgentDisruptBench/evaluation/run_benchmark.py)
- CLI entry point: `python -m evaluation.run_benchmark`
- Parses args (framework, profiles, domains, model, API key)
- Instantiates the chosen runner, wires up `BenchmarkRunner`, generates report

#### [NEW] [evaluation/runners/](file:///Users/kaviru/Downloads/agent-disrupt-bench/AgentDisruptBench/evaluation/runners/)
Self-contained runner implementations:

| File | Framework | LLM API Used |
|------|-----------|-------------|
| `openai_runner.py` | OpenAI function calling | `openai` SDK |
| `gemini_runner.py` | Gemini function calling | `google-genai` SDK |
| `langchain_runner.py` | LangChain ReAct agent | `langchain-openai` / `langchain-google-genai` |
| `autogen_runner.py` | AutoGen ConversableAgent | `pyautogen` |
| `crewai_runner.py` | CrewAI agent | `crewai` |
| `simple_runner.py` | Rule-based baseline | No LLM |

Each runner follows the same contract: `(task, tools) → str`.

---

### Config & Docs Updates

#### [MODIFY] [pyproject.toml](file:///Users/kaviru/Downloads/agent-disrupt-bench/AgentDisruptBench/pyproject.toml)
- Add `gemini = ["google-genai>=1.0"]` to optional deps

#### [MODIFY] [__init__.py](file:///Users/kaviru/Downloads/agent-disrupt-bench/AgentDisruptBench/python/agentdisruptbench/__init__.py)
- No change (adapter import is lazy/optional)

#### [MODIFY] [.env.example](file:///Users/kaviru/Downloads/agent-disrupt-bench/AgentDisruptBench/.env.example)
- Add `GOOGLE_API_KEY` variable

#### [MODIFY] [README.md](file:///Users/kaviru/Downloads/agent-disrupt-bench/AgentDisruptBench/README.md)
- Add Gemini adapter example, evaluation usage section

---

## Verification Plan

### Automated Tests
- `pytest tests/` — existing 36 tests still pass
- New smoke test for Gemini adapter (mocked, no real API call)

### Manual
- Run `python -m evaluation.run_benchmark --runner simple --profiles clean --max-difficulty 1`
