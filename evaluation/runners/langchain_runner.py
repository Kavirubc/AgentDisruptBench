"""
AgentDisruptBench — LangChain ReAct Agent Runner
==================================================

File:        langchain_runner.py
Purpose:     Full LLM-powered agent runner using LangChain's tool-calling
             agent. Creates StructuredTool wrappers from the tools dict,
             builds a ReAct-style agent, and runs it in a loop.

Author:      AgentDisruptBench Contributors
License:     MIT
Created:     2026-03-09
Modified:    2026-03-09

Key Classes:
    LangChainRunner : Autonomous agent using LangChain + ChatOpenAI.

Dependencies:
    pip install langchain-core langchain-openai langgraph

Convention:
    Every source file MUST include a header block like this one.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from evaluation.base_runner import BaseAgentRunner, RunnerConfig
from agentdisruptbench.tasks.schemas import Task

logger = logging.getLogger("agentdisruptbench.evaluation.runners.langchain")


class LangChainRunner(BaseAgentRunner):
    """Autonomous agent using LangChain with tool calling.

    Creates StructuredTool instances from the tools dict and runs them
    through a LangChain agent with a chat model.

    Supports any LangChain-compatible chat model. Defaults to ChatOpenAI.

    Usage::

        runner = LangChainRunner(RunnerConfig(
            model="gpt-4o",
            api_key="sk-...",
        ))
        result = runner.run_task(task, tools)
    """

    def __init__(self, config: RunnerConfig | None = None) -> None:
        super().__init__(config or RunnerConfig(model="gpt-4o"))
        self._llm = None

    def setup(self) -> None:
        """Initialise the LangChain chat model."""
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise ImportError(
                "LangChain runner requires langchain-openai. "
                "Install with: pip install langchain-openai langgraph"
            )

        api_key = self.config.api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "API key required. Set OPENAI_API_KEY env var "
                "or pass api_key in RunnerConfig."
            )

        self._llm = ChatOpenAI(
            model=self.config.model,
            api_key=api_key,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        super().setup()

    def run_task(self, task: Task, tools: dict[str, Any]) -> str:
        """Run a LangChain agent for one task."""
        if self._llm is None:
            self.setup()

        try:
            from langchain_core.tools import StructuredTool
            from langgraph.prebuilt import create_react_agent
        except ImportError:
            raise ImportError(
                "LangChain runner requires langchain-core and langgraph. "
                "Install with: pip install langchain-core langgraph"
            )

        # Create StructuredTool wrappers
        lc_tools = []
        for name, fn in tools.items():
            tool = StructuredTool.from_function(
                func=fn,
                name=name,
                description=f"Execute the {name} tool",
            )
            lc_tools.append(tool)

        # Create ReAct agent
        agent = create_react_agent(self._llm, lc_tools)

        # Build prompt
        prompt = (
            f"Task: {task.description}\n\n"
            f"Available tools: {', '.join(tools.keys())}\n\n"
            "Please complete this task using the available tools. "
            "If a tool call fails, retry once. Provide a clear final answer."
        )

        # Run agent
        try:
            result = agent.invoke(
                {"messages": [("user", prompt)]},
                config={"recursion_limit": self.config.max_steps * 2},
            )

            # Extract final message
            messages = result.get("messages", [])
            if messages:
                last = messages[-1]
                content = getattr(last, "content", str(last))
                self._total_api_calls += len(messages) // 2  # rough estimate
                return content or "[No response]"

            return "[Agent produced no output]"

        except Exception as exc:
            logger.exception("langchain_agent_error task=%s", task.task_id)
            return f"[Agent error: {exc}]"

    def teardown(self) -> None:
        """Clean up."""
        self._llm = None
        super().teardown()
