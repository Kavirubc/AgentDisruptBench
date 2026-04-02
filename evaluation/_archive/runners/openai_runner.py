"""
AgentDisruptBench — OpenAI Function Calling Runner
====================================================

File:        openai_runner.py
Purpose:     Full LLM-powered agent runner using OpenAI's function calling
             API. Runs a complete ReAct-style loop: system prompt → LLM
             generates tool calls → dispatch through (possibly disrupted)
             tools → feed results back → repeat until done.

Author:      AgentDisruptBench Contributors
License:     MIT
Created:     2026-03-09
Modified:    2026-03-09

Key Classes:
    OpenAIRunner : Autonomous agent using OpenAI chat completions with tools.

Dependencies:
    pip install openai

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

logger = logging.getLogger("agentdisruptbench.evaluation.runners.openai")

_SYSTEM_PROMPT = """You are a helpful assistant solving tasks by calling tools.
You will be given a task description and a set of available tools.
Call the necessary tools to complete the task. When you have enough
information to give a final answer, respond with your conclusion.

Important:
- If a tool call fails, you may retry it once.
- If a tool returns an error, acknowledge it and try an alternative approach.
- Always provide a clear final answer summarising what you accomplished."""


def _build_tool_schemas(tools: dict[str, Any]) -> list[dict]:
    """Build OpenAI-compatible tool schemas from a tools dict."""
    schemas = []
    for name in tools:
        schemas.append({
            "type": "function",
            "function": {
                "name": name,
                "description": f"Execute the {name} tool",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": True,
                },
            },
        })
    return schemas


class OpenAIRunner(BaseAgentRunner):
    """Autonomous agent using OpenAI chat completions with function calling.

    Runs a loop of up to ``max_steps`` iterations:
    1. Send messages to OpenAI with tool schemas.
    2. If response contains tool_calls, dispatch each one.
    3. Append tool results and loop.
    4. When the model returns a text response (no tool_calls), return it.

    Usage::

        runner = OpenAIRunner(RunnerConfig(
            model="gpt-4o",
            api_key="sk-...",
        ))
        result = runner.run_task(task, tools)
    """

    def __init__(self, config: RunnerConfig | None = None) -> None:
        super().__init__(config or RunnerConfig(model="gpt-4o"))
        self._client = None

    def setup(self) -> None:
        """Initialise the OpenAI client."""
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "OpenAI runner requires the openai package. "
                "Install with: pip install openai"
            )

        api_key = self.config.api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY env var "
                "or pass api_key in RunnerConfig."
            )

        self._client = OpenAI(api_key=api_key)
        super().setup()

    def run_task(self, task: Task, tools: dict[str, Any]) -> str:
        """Run the full OpenAI agent loop for one task."""
        if self._client is None:
            self.setup()

        tool_schemas = _build_tool_schemas(tools)
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"Task: {task.description}\n\n"
                f"Available tools: {', '.join(tools.keys())}\n\n"
                "Please complete this task using the available tools."
            )},
        ]

        for step in range(self.config.max_steps):
            self._total_api_calls += 1

            response = self._client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                tools=tool_schemas if tool_schemas else None,
                tool_choice="auto",
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )

            msg = response.choices[0].message
            usage = response.usage
            if usage:
                self._total_tokens += usage.total_tokens

            # No tool calls → model is done
            if not msg.tool_calls:
                return msg.content or "[No response from model]"

            # Append assistant message with tool calls
            messages.append(msg)

            # Dispatch each tool call
            for tc in msg.tool_calls:
                func_name = tc.function.name
                try:
                    func_args = json.loads(tc.function.arguments)
                except (json.JSONDecodeError, TypeError):
                    func_args = {}

                fn = tools.get(func_name)
                if fn is None:
                    tool_result = json.dumps({"error": f"Unknown tool: {func_name}"})
                else:
                    try:
                        result = fn(**func_args)
                        tool_result = json.dumps(result, default=str)
                    except Exception as exc:
                        tool_result = json.dumps({"error": str(exc)})

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": tool_result,
                })

                if self.config.verbose:
                    print(f"  [tool] {func_name}({func_args}) → {tool_result[:200]}")

        return "[Agent reached max steps without completing the task]"

    def teardown(self) -> None:
        """Clean up OpenAI client."""
        self._client = None
        super().teardown()
