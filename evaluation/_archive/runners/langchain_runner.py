"""
AgentDisruptBench — LangChain ReAct Agent Runner
==================================================

File:        langchain_runner.py
Purpose:     Full LLM-powered agent runner using LangChain's create_agent
             API. Supports both OpenAI and Google Gemini models out of the
             box. Uses LangChain v1's unified create_agent which builds a
             ReAct-style agent graph under the hood.

Author:      AgentDisruptBench Contributors
License:     MIT
Created:     2026-03-09
Modified:    2026-03-09

Key Classes:
    LangChainRunner : ReAct agent using LangChain v1's create_agent.

Dependencies:
    For Gemini:  pip install langchain langchain-google-genai
    For OpenAI:  pip install langchain langchain-openai

Supported Models:
    - Gemini:  gemini-2.5-flash, gemini-2.5-pro, gemini-2.0-flash, etc.
    - OpenAI:  gpt-4o, gpt-4o-mini, gpt-3.5-turbo, etc.

Usage:
    python -m evaluation.run_benchmark --runner langchain --model gemini-2.5-flash

Convention:
    Every source file MUST include a header block like this one.
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any

from agentdisruptbench.tasks.schemas import Task

from evaluation.base_runner import (
    BaseAgentRunner, RunnerConfig,
    _RESET, _BOLD, _DIM, _GREEN, _RED, _CYAN,
)
from evaluation.llm_factory import create_langchain_llm, detect_provider

logger = logging.getLogger("agentdisruptbench.evaluation.runners.langchain")


class LangChainRunner(BaseAgentRunner):
    """ReAct agent using LangChain v1's create_agent API.

    LangChain v1 (1.2+) provides ``create_agent()`` which builds a
    ReAct-style agent graph that calls tools in a loop until the model
    produces a final answer. This is the unified successor to the older
    ``create_react_agent`` / ``AgentExecutor`` pattern.

    Automatically detects the model provider:
    - ``gemini-*`` models → ``ChatGoogleGenerativeAI`` (needs GEMINI_API_KEY)
    - ``gpt-*`` models   → ``ChatOpenAI`` (needs OPENAI_API_KEY)

    Usage::

        # With Gemini 2.5 Flash
        runner = LangChainRunner(RunnerConfig(model="gemini-2.5-flash"))
        result = runner.run_task(task, tools)

        # With OpenAI
        runner = LangChainRunner(RunnerConfig(model="gpt-4o"))
        result = runner.run_task(task, tools)
    """

    def __init__(self, config: RunnerConfig | None = None) -> None:
        super().__init__(config or RunnerConfig(model="gemini-2.5-flash"))
        self._llm = None

    def setup(self) -> None:
        """Initialise the LangChain chat model (Gemini or OpenAI)."""
        self._llm = create_langchain_llm(self.config)
        provider = detect_provider(self.config.model)
        logger.info("langchain_runner_setup provider=%s model=%s", provider, self.config.model)
        super().setup()

    @staticmethod
    def _build_args_schema(tool_name: str, proxy_fn: Any) -> type | None:
        """Build a Pydantic model from the mock tool's function signature.

        StructuredTool.from_function infers its schema from the wrapped
        function's signature.  Since our wrapper uses ``**kwargs``, the
        inferred schema has a single ``kwargs`` field — which breaks
        argument forwarding.

        This method inspects the *underlying* mock tool function (accessed
        via ``proxy_fn._fn``) and dynamically creates a Pydantic BaseModel
        with the correct parameter names and types.
        """
        import inspect
        from pydantic import Field, create_model

        # Walk the proxy chain to find the real function with a meaningful signature:
        # ToolProxy._fn → may be a stateful_wrapper(**kwargs) or the raw static method
        real_fn = getattr(proxy_fn, '_fn', None) or proxy_fn

        # If real_fn only has **kwargs, try to find the original fn in its closure
        try:
            sig = inspect.signature(real_fn)
            params = [
                p for p in sig.parameters.values()
                if p.name not in ("self", "cls")
                and p.kind not in (
                    inspect.Parameter.VAR_KEYWORD,
                    inspect.Parameter.VAR_POSITIONAL,
                )
            ]
            if not params:
                # Look inside closure cells for a callable with real params
                closure = getattr(real_fn, '__closure__', None) or []
                for cell in closure:
                    try:
                        candidate = cell.cell_contents
                        if callable(candidate) and candidate is not real_fn:
                            csig = inspect.signature(candidate)
                            cparams = [
                                p for p in csig.parameters.values()
                                if p.name not in ("self", "cls")
                                and p.kind not in (
                                    inspect.Parameter.VAR_KEYWORD,
                                    inspect.Parameter.VAR_POSITIONAL,
                                )
                            ]
                            if cparams:
                                real_fn = candidate
                                sig = csig
                                break
                    except (ValueError, AttributeError):
                        continue
        except (ValueError, TypeError):
            return None

        fields: dict[str, Any] = {}
        for param_name, param in sig.parameters.items():
            if param_name in ("self", "cls"):
                continue

            # Determine type annotation (default to str)
            ann = param.annotation if param.annotation != inspect.Parameter.empty else str

            # Determine default
            if param.default != inspect.Parameter.empty:
                fields[param_name] = (ann, Field(default=param.default))
            else:
                fields[param_name] = (ann, ...)

        if not fields:
            return None

        # Create a Pydantic model with the tool name as class name
        model_name = "".join(part.capitalize() for part in tool_name.split("_")) + "Input"
        return create_model(model_name, **fields)

    def run_task(self, task: Task, tools: dict[str, Any]) -> str:
        """Run the LangChain agent for one task.

        Uses LangChain's create_agent which builds a ReAct
        graph under the hood:  LLM → tool call → result → LLM → ...
        """
        if self._llm is None:
            self.setup()

        try:
            from langchain_core.tools import StructuredTool
        except ImportError:
            raise ImportError(
                "LangChain runner requires langchain-core. "
                "Install with: pip install langchain-core"
            )

        # Create Tool wrappers — use StructuredTool with a proper Pydantic schema
        # so that the LLM (Gemini/OpenAI) can see the correct parameter names.
        verbose = self.config.verbose
        lc_tools = []
        for name, fn in tools.items():
            proxy_fn = fn  # capture in closure
            tool_name = name  # capture for closure

            def _make_tool_fn(captured_fn=proxy_fn, tname=tool_name):
                def tool_fn(**kwargs) -> str:
                    if verbose:
                        args_str = ", ".join(f"{k}={repr(v)[:40]}" for k, v in kwargs.items())
                        print(f"    {_CYAN}🔧 {_BOLD}{tname}{_RESET}{_CYAN}({args_str}){_RESET}")
                    try:
                        result = captured_fn(**kwargs)
                        if verbose:
                            preview = str(result)[:100].replace("\n", " ")
                            print(f"    {_GREEN}✓  → {_DIM}{preview}{_RESET}")
                        return json.dumps(result) if isinstance(result, dict) else str(result)
                    except Exception as exc:
                        if verbose:
                            print(f"    {_RED}✗  → {exc}{_RESET}")
                        return json.dumps({"error": str(exc), "status": "failed"})
                return tool_fn

            # Build Pydantic schema from the underlying mock tool's signature
            args_schema = self._build_args_schema(name, fn)

            tool = StructuredTool.from_function(
                func=_make_tool_fn(),
                name=name,
                description=f"Execute the {name} tool.",
                args_schema=args_schema,
            )
            lc_tools.append(tool)

        # Build system prompt
        system_prompt = (
            "You are a helpful assistant that completes tasks by calling tools.\n"
            "If a tool call fails or returns an error, retry it once.\n"
            "When you have enough information, provide a clear final answer "
            "summarising what you accomplished."
        )

        # Create the agent using LangChain's standard ReAct agent (v1)
        from langchain.agents import create_agent
        if verbose:
            print(f"    [DEBUG] Creating agent with {len(lc_tools)} tools")
            for t in lc_tools:
                print(f"    [DEBUG] Tool: {t.name} (args: {t.args_schema.schema() if t.args_schema else 'None'})")

        agent = create_agent(
            self._llm,
            lc_tools,
            system_prompt=system_prompt,
        )

        # Build the task input
        task_input = (
            f"Task: {task.description}\n\n"
            f"Available tools: {', '.join(tools.keys())}\n\n"
            "Please complete this task using the available tools."
        )

        # Run the agent
        try:
            if verbose:
                print(f"    [DEBUG] Invoking agent for task {task.task_id}")
            result = agent.invoke(
                {"messages": [{"role": "user", "content": task_input}]},
            )

            # Extract the final message content
            messages = result.get("messages", [])
            if messages:
                # Track tokens from all messages with usage_metadata
                for m in messages:
                    if hasattr(m, "usage_metadata") and m.usage_metadata:
                        usage = m.usage_metadata
                        self._prompt_tokens += usage.get("input_tokens", 0)
                        self._completion_tokens += usage.get("output_tokens", 0)
                    elif hasattr(m, "response_metadata") and m.response_metadata:
                        # Fallback for older LangChain versions or specific providers
                        resp_meta = m.response_metadata
                        token_usage = resp_meta.get("token_usage", {})
                        if token_usage:
                            self._prompt_tokens += token_usage.get("prompt_tokens", 0)
                            self._completion_tokens += token_usage.get("completion_tokens", 0)

                if verbose:
                    print(f"    [DEBUG] Agent returned {len(messages)} messages")
                    for i, m in enumerate(messages):
                        msg_type = getattr(m, "type", "unknown")
                        tcalls = getattr(m, "tool_calls", [])
                        if not tcalls and hasattr(m, "additional_kwargs"):
                            tcalls = m.additional_kwargs.get("tool_calls", [])
                        
                        content_preview = str(m.content)[:100].replace("\n", " ")
                        print(f"    [DEBUG] Msg {i} ({msg_type}): {content_preview}")
                        if tcalls:
                            print(f"    [DEBUG]   -> Tool Calls: {tcalls}")
                        if hasattr(m, "additional_kwargs") and m.additional_kwargs:
                            print(f"    [DEBUG]   -> Addl Kwargs: {m.additional_kwargs}")

                last_msg = messages[-1]
                raw_content = getattr(last_msg, "content", str(last_msg))

                # Normalize structured content (e.g., Gemini returns list of parts)
                if isinstance(raw_content, list):
                    parts = []
                    for item in raw_content:
                        if isinstance(item, dict):
                            parts.append(str(item.get("text", item)))
                        else:
                            parts.append(str(item))
                    content = " ".join(parts)
                else:
                    content = str(raw_content)

                self._task_api_calls += sum(
                    1 for m in messages if getattr(m, "type", "") == "ai"
                )
                return content or "[No response from agent]"

            return "[Agent produced no output]"

        except Exception as exc:
            logger.exception("langchain_agent_error task=%s", task.task_id)
            return f"[Agent error: {exc}]"

    def teardown(self) -> None:
        """Clean up."""
        self._llm = None
        super().teardown()
