"""
AgentDisruptBench — CrewAI Agent Runner
=========================================

File:        crewai_runner.py
Purpose:     Full LLM-powered agent runner using CrewAI. Creates a Crew
             with a single Agent and Task, equipped with tools from the
             benchmark.

Author:      AgentDisruptBench Contributors
License:     MIT
Created:     2026-03-09
Modified:    2026-03-09

Key Classes:
    CrewAIRunner : Autonomous agent using CrewAI's Crew + Agent + Task.

Dependencies:
    pip install crewai

Convention:
    Every source file MUST include a header block like this one.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from evaluation.base_runner import BaseAgentRunner, RunnerConfig
from agentdisruptbench.tasks.schemas import Task

logger = logging.getLogger("agentdisruptbench.evaluation.runners.crewai")


class CrewAIRunner(BaseAgentRunner):
    """Autonomous agent using CrewAI's Crew → Agent → Task pattern.

    Creates a single-agent crew with the benchmark tools attached.
    The crewAI agent decides which tools to call and executes them.

    Usage::

        runner = CrewAIRunner(RunnerConfig(
            model="gpt-4o",
            api_key="sk-...",
        ))
        result = runner.run_task(task, tools)
    """

    def __init__(self, config: RunnerConfig | None = None) -> None:
        super().__init__(config or RunnerConfig(model="gpt-4o"))

    def setup(self) -> None:
        """Validate CrewAI is importable."""
        try:
            import crewai  # noqa: F401
        except ImportError:
            raise ImportError(
                "CrewAI runner requires crewai. "
                "Install with: pip install crewai"
            )

        api_key = self.config.api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "API key required. Set OPENAI_API_KEY env var "
                "or pass api_key in RunnerConfig."
            )
        os.environ["OPENAI_API_KEY"] = api_key
        super().setup()

    def run_task(self, task: Task, tools: dict[str, Any]) -> str:
        """Run a CrewAI crew for one task."""
        try:
            from crewai import Agent, Crew, Task as CrewTask
            from crewai.tools import BaseTool
        except ImportError:
            raise ImportError("CrewAI runner requires crewai.")

        # Create tool wrappers
        crewai_tools = []
        for name, fn in tools.items():
            tool_fn = fn  # capture in closure

            class DynamicTool(BaseTool):
                name: str = name
                description: str = f"Execute the {name} tool"

                def _run(self, **kwargs: Any) -> Any:
                    return tool_fn(**kwargs)

            crewai_tools.append(DynamicTool())

        # Create agent
        agent = Agent(
            role="Task Executor",
            goal=f"Complete the following task: {task.description}",
            backstory=(
                "You are a skilled assistant that uses available tools "
                "to complete tasks accurately. If a tool fails, retry once."
            ),
            tools=crewai_tools,
            verbose=self.config.verbose,
            max_iter=self.config.max_steps,
        )

        # Create crew task
        crew_task = CrewTask(
            description=(
                f"{task.description}\n\n"
                f"Available tools: {', '.join(tools.keys())}\n"
                "Use the tools to complete the task and provide a final answer."
            ),
            expected_output="A clear summary of the task completion.",
            agent=agent,
        )

        # Create and run crew
        crew = Crew(
            agents=[agent],
            tasks=[crew_task],
            verbose=self.config.verbose,
        )

        try:
            result = crew.kickoff()
            self._total_api_calls += 1
            return str(result)
        except Exception as exc:
            logger.exception("crewai_agent_error task=%s", task.task_id)
            return f"[Agent error: {exc}]"
