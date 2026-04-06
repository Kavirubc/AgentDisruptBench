"""
AgentDisruptBench — AutoGen Agent Runner
==========================================

File:        autogen_runner.py
Purpose:     Full LLM-powered agent runner using AutoGen's ConversableAgent.
             Creates a two-agent setup: an AssistantAgent with tools and a
             UserProxyAgent that executes tool calls.

Author:      AgentDisruptBench Contributors
License:     MIT
Created:     2026-03-09
Modified:    2026-03-09

Key Classes:
    AutoGenRunner : Autonomous agent using AutoGen's conversable agents.

Dependencies:
    pip install pyautogen

Convention:
    Every source file MUST include a header block like this one.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from agentdisruptbench.tasks.schemas import Task

from evaluation.base_runner import BaseAgentRunner, RunnerConfig

logger = logging.getLogger("agentdisruptbench.evaluation.runners.autogen")


class AutoGenRunner(BaseAgentRunner):
    """Autonomous agent using AutoGen's two-agent pattern.

    Sets up an AssistantAgent (LLM) and a UserProxyAgent (tool executor).
    The AssistantAgent decides which tools to call; the UserProxyAgent
    dispatches them through the (possibly disrupted) tool functions.

    Usage::

        runner = AutoGenRunner(RunnerConfig(
            model="gpt-4o",
            api_key="sk-...",
        ))
        result = runner.run_task(task, tools)
    """

    def __init__(self, config: RunnerConfig | None = None) -> None:
        super().__init__(config or RunnerConfig(model="gpt-4o"))

    def setup(self) -> None:
        """Validate AutoGen is importable."""
        try:
            import autogen  # noqa: F401
        except ImportError:
            raise ImportError("AutoGen runner requires pyautogen. Install with: pip install pyautogen")

        api_key = self.config.api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("API key required. Set OPENAI_API_KEY env var or pass api_key in RunnerConfig.")
        super().setup()

    def run_task(self, task: Task, tools: dict[str, Any]) -> str:
        """Run the AutoGen two-agent pattern for one task."""
        try:
            import autogen
        except ImportError:
            raise ImportError("AutoGen runner requires pyautogen.")

        api_key = self.config.api_key or os.environ.get("OPENAI_API_KEY")

        llm_config = {
            "config_list": [
                {
                    "model": self.config.model,
                    "api_key": api_key,
                }
            ],
            "temperature": self.config.temperature,
        }

        # Build function_map from tools
        function_map = {}
        tool_descriptions = []
        for name, fn in tools.items():
            function_map[name] = fn
            tool_descriptions.append(f"- {name}: Execute the {name} tool")

        tools_text = "\n".join(tool_descriptions)

        # Create agents
        assistant = autogen.AssistantAgent(
            name="assistant",
            system_message=(
                "You are a helpful assistant that completes tasks by calling tools.\n"
                f"Available tools:\n{tools_text}\n\n"
                "Call the tools needed to complete the task. "
                "If a tool fails, retry once. Provide a clear final answer.\n"
                "Reply TERMINATE when the task is complete."
            ),
            llm_config=llm_config,
        )

        user_proxy = autogen.UserProxyAgent(
            name="user_proxy",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=self.config.max_steps,
            code_execution_config=False,
            function_map=function_map,
        )

        # Initiate chat
        prompt = f"Task: {task.description}\n\nPlease complete this task using the available tools."

        try:
            user_proxy.initiate_chat(
                assistant,
                message=prompt,
            )

            # Extract last assistant message
            messages = assistant.chat_messages.get(user_proxy, [])
            for msg in reversed(messages):
                content = msg.get("content", "")
                if content and "TERMINATE" not in content:
                    self._total_api_calls += len(messages) // 2
                    return content

            return "[AutoGen agent completed without a final answer]"

        except Exception as exc:
            logger.exception("autogen_agent_error task=%s", task.task_id)
            return f"[Agent error: {exc}]"
