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

import logging
from typing import Any

from agentdisruptbench.tasks.schemas import Task

from evaluation.base_runner import BaseAgentRunner, RunnerConfig
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

    def run_task(self, task: Task, tools: dict[str, Any]) -> str:
        """Run the LangChain agent for one task.

        Uses create_agent() from langchain.agents which builds a ReAct
        graph under the hood:  LLM → tool call → result → LLM → ...
        """
        if self._llm is None:
            self.setup()

        try:
            from langchain.agents import create_agent
            from langchain_core.tools import Tool
        except ImportError:
            raise ImportError(
                "LangChain runner requires langchain and langchain-core. "
                "Install with: pip install langchain langchain-core"
            )

        # Create Tool wrappers — use Tool() with a lambda wrapper instead
        # of StructuredTool.from_function(), because ToolProxy objects are
        # callable classes and Pydantic's validate_arguments can't introspect them.
        lc_tools = []
        for name, fn in tools.items():
            # Wrap in a lambda so LangChain sees a proper function
            proxy_fn = fn  # capture in closure
            tool = Tool(
                name=name,
                description=f"Execute the {name} tool. Input should be a JSON string of keyword arguments.",
                func=lambda input_str, _fn=proxy_fn: _fn(
                    **__import__("json").loads(input_str)
                    if input_str.strip().startswith("{")
                    else {}
                ),
            )
            lc_tools.append(tool)

        # Build system prompt
        system_prompt = (
            "You are a helpful assistant that completes tasks by calling tools.\n"
            "If a tool call fails or returns an error, retry it once.\n"
            "When you have enough information, provide a clear final answer "
            "summarising what you accomplished.\n"
            "When calling a tool, pass the input as a JSON object string."
        )

        # Create the agent using LangChain v1's create_agent
        agent = create_agent(
            model=self._llm,
            tools=lc_tools,
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
            result = agent.invoke(
                {"messages": [{"role": "user", "content": task_input}]},
            )

            # Extract the final message content
            messages = result.get("messages", [])
            if messages:
                last_msg = messages[-1]
                content = getattr(last_msg, "content", str(last_msg))
                self._total_api_calls += sum(
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
