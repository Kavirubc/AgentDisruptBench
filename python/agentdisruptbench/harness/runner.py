"""
AgentDisruptBench — BenchmarkRunner
====================================

File:        runner.py
Purpose:     Top-level benchmark harness. Iterates over (task × profile)
             combinations, runs clean baselines first, then disrupted runs.
             Supports concurrency via ProcessPoolExecutor.

Author:      AgentDisruptBench Contributors
License:     MIT
Created:     2026-03-09
Modified:    2026-03-09

Key Classes:
    BenchmarkRunner  : High-level ``run_all()`` / ``run_task()`` API.
    BenchmarkConfig  : Configuration for a full benchmark suite run.

Convention:
    Every source file MUST include a header block like this one.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

from agentdisruptbench.core.metrics import BenchmarkResult
from agentdisruptbench.core.profiles import get_profile
from agentdisruptbench.harness.evaluator import Evaluator
from agentdisruptbench.tasks.registry import TaskRegistry
from agentdisruptbench.tasks.schemas import Task
from agentdisruptbench.tools.registry import ToolRegistry

logger = logging.getLogger("agentdisruptbench.runner")


@dataclass
class BenchmarkConfig:
    """Configuration for a full benchmark suite run.

    Attributes:
        profiles:     List of profile names to evaluate.
        seeds:        List of seeds for repeated runs.
        domains:      Restrict to these domains (None = all).
        task_ids:     Restrict to specific task IDs (None = all).
        max_difficulty: Max task difficulty to include.
        agent_id:     Identifier for the agent under test.
        output_dir:   Directory for results output.
    """

    profiles: list[str] = field(default_factory=lambda: ["clean", "mild_production", "hostile_environment"])
    seeds: list[int] = field(default_factory=lambda: [42])
    domains: list[str] | None = None
    task_ids: list[str] | None = None
    max_difficulty: int = 5
    agent_id: str = "default"
    output_dir: str = "results"


class BenchmarkRunner:
    """Top-level benchmark harness.

    Runs all (task × profile × seed) combinations. Clean profile is always
    run first to establish baselines for resilience_ratio calculation.

    Usage::

        runner = BenchmarkRunner(
            agent_fn=my_agent,
            task_registry=TaskRegistry.from_builtin(),
            tool_registry=ToolRegistry.from_simulated_tools(),
            config=BenchmarkConfig(profiles=["clean", "hostile_environment"]),
        )
        results = runner.run_all()
    """

    def __init__(
        self,
        agent_fn: Callable[[Task, dict[str, Any]], str],
        task_registry: TaskRegistry,
        tool_registry: ToolRegistry,
        config: BenchmarkConfig | None = None,
    ) -> None:
        self._agent_fn = agent_fn
        self._task_registry = task_registry
        self._tool_registry = tool_registry
        self._config = config or BenchmarkConfig()
        self._evaluator = Evaluator(agent_fn, tool_registry)
        self._results: list[BenchmarkResult] = []

    def run_all(self) -> list[BenchmarkResult]:
        """Run the full benchmark suite.

        Returns:
            List of BenchmarkResult for every (task × profile × seed).
        """
        tasks = self._get_tasks()
        logger.info(
            "benchmark_start tasks=%d profiles=%d seeds=%d",
            len(tasks),
            len(self._config.profiles),
            len(self._config.seeds),
        )

        all_results: list[BenchmarkResult] = []

        for task in tasks:
            task_results = self.run_task(task)
            all_results.extend(task_results)

        self._results = all_results
        logger.info("benchmark_complete total_results=%d", len(all_results))
        return all_results

    def run_task(self, task: Task) -> list[BenchmarkResult]:
        """Run all profiles × seeds for a single task.

        Clean profile is always run first to establish baseline.

        Returns:
            List of BenchmarkResult for this task.
        """
        results: list[BenchmarkResult] = []
        baselines: dict[int, BenchmarkResult] = {}  # seed → clean result

        profiles = self._config.profiles
        # Ensure 'clean' is first if present
        if "clean" in profiles:
            profiles = ["clean"] + [p for p in profiles if p != "clean"]

        for seed in self._config.seeds:
            for profile_name in profiles:
                configs = get_profile(profile_name)
                baseline = baselines.get(seed)

                result = self._evaluator.run(
                    task=task,
                    configs=configs,
                    profile_name=profile_name,
                    agent_id=self._config.agent_id,
                    seed=seed,
                    baseline_result=baseline,
                )
                results.append(result)

                if profile_name == "clean":
                    baselines[seed] = result

                logger.info(
                    "task_profile_complete task=%s profile=%s seed=%d success=%s",
                    task.task_id,
                    profile_name,
                    seed,
                    result.success,
                )

        return results

    def _get_tasks(self) -> list[Task]:
        """Filter tasks based on config."""
        return self._task_registry.filter(
            domain=self._config.domains[0] if self._config.domains and len(self._config.domains) == 1 else None,
            task_ids=self._config.task_ids,
            max_difficulty=self._config.max_difficulty,
        )

    @property
    def results(self) -> list[BenchmarkResult]:
        """All results from the most recent ``run_all()``."""
        return list(self._results)
