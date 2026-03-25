"""
AgentDisruptBench — Base Agent Runner
=======================================

File:        base_runner.py
Purpose:     Abstract base class for evaluation runners. All runners
             (OpenAI, LangChain, AutoGen, CrewAI, Simple) extend this.
             Defines the agent contract: (task, tools) → str.

Author:      AgentDisruptBench Contributors
License:     MIT
Created:     2026-03-09
Modified:    2026-03-09

Key Classes:
    BaseAgentRunner : ABC with setup / run_task / teardown lifecycle.
    RunnerConfig    : Configuration shared across all runners.

Design Notes:
    - Mirrors REALM-Bench's BaseFrameworkRunner pattern.
    - The run_task() contract matches BenchmarkRunner's agent_fn signature:
        (Task, dict[str, Any]) → str
    - Runners manage their own LLM client lifecycle via setup/teardown.

Convention:
    Every source file MUST include a header block like this one.
"""

from __future__ import annotations

import abc
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

from agentdisruptbench.tasks.schemas import Task

logger = logging.getLogger("agentdisruptbench.evaluation.base_runner")


@dataclass
class RunnerConfig:
    """Configuration for an evaluation runner.

    Attributes:
        model:       Model name/identifier (e.g. "gpt-4o", "gemini-2.0-flash").
        api_key:     API key (falls back to env vars if not set).
        temperature: Sampling temperature for the LLM.
        max_tokens:  Max tokens per LLM response (None = provider default).
        max_retries: Max retries on API errors (not disruption retries).
        max_steps:   Max agent loop iterations before forced stop.
        verbose:     Print agent reasoning to stdout.
    """

    model: str = "gpt-4o"
    api_key: str | None = None
    temperature: float = 0.0
    max_tokens: int | None = None
    max_retries: int = 3
    max_steps: int = 20
    verbose: bool = False


class BaseAgentRunner(abc.ABC):
    """Abstract base class for framework-specific evaluation runners.

    Lifecycle:
        1. ``setup()``    — Initialise LLM client, load resources.
        2. ``run_task()``  — Execute one task (called N times by BenchmarkRunner).
        3. ``teardown()``  — Clean up resources.

    The ``__call__`` method implements the ``agent_fn`` contract
    expected by ``BenchmarkRunner``: ``(Task, dict[str, Any]) → str``.

    Parameters:
        config: Runner configuration.
    """

    def __init__(self, config: RunnerConfig | None = None) -> None:
        self.config = config or RunnerConfig()
        self._is_setup = False
        self._total_tokens = 0
        self._total_api_calls = 0
        self._total_time = 0.0

    def setup(self) -> None:
        """Initialise LLM client and resources. Override in subclasses."""
        self._is_setup = True
        logger.info("runner_setup class=%s model=%s", type(self).__name__, self.config.model)

    def teardown(self) -> None:
        """Clean up resources. Override in subclasses."""
        self._is_setup = False
        logger.info(
            "runner_teardown class=%s tokens=%d api_calls=%d",
            type(self).__name__, self._total_tokens, self._total_api_calls,
        )

    @abc.abstractmethod
    def run_task(self, task: Task, tools: dict[str, Any]) -> str:
        """Execute a single task with the given (possibly disrupted) tools.

        This is the core agent logic. The runner should:
        1. Build a system prompt from the task description.
        2. Convert the tools dict into framework-native tool declarations.
        3. Run the agent loop (LLM → tool call → result → LLM → ...).
        4. Return the agent's final textual output.

        Args:
            task:  Task definition with description, required_tools, etc.
            tools: Dict of tool_name → callable (may be ToolProxy instances).

        Returns:
            The agent's final output as a string.
        """
        ...

    def __call__(self, task: Task, tools: dict[str, Any]) -> str:
        """Entry point matching ``BenchmarkRunner``'s agent_fn contract.

        Handles setup-on-first-call and timing.
        """
        if not self._is_setup:
            self.setup()

        start = time.monotonic()
        result = self.run_task(task, tools)
        elapsed = time.monotonic() - start
        self._total_time += elapsed

        if self.config.verbose:
            print(f"[{type(self).__name__}] task={task.task_id} time={elapsed:.1f}s")

        return result

    @property
    def stats(self) -> dict[str, Any]:
        """Runner statistics."""
        return {
            "runner": type(self).__name__,
            "model": self.config.model,
            "total_tokens": self._total_tokens,
            "total_api_calls": self._total_api_calls,
            "total_time": round(self._total_time, 2),
        }
