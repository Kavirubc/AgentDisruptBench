"""
AgentDisruptBench — Evaluator
==============================

File:        evaluator.py
Purpose:     Orchestrates a single (task, profile) evaluation run.
             Sets up the engine, wraps tools, executes the agent function,
             collects traces, and produces BenchmarkResult via MetricsCalculator.

Author:      AgentDisruptBench Contributors
License:     MIT
Created:     2026-03-09
Modified:    2026-03-18

Key Classes:
    Evaluator : Encapsulates one run. Called by the BenchmarkRunner.

Convention:
    Every source file MUST include a header block like this one.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable

from agentdisruptbench.core.engine import DisruptionConfig, DisruptionEngine
from agentdisruptbench.core.metrics import BenchmarkResult, MetricsCalculator
from agentdisruptbench.core.proxy import ToolProxy
from agentdisruptbench.core.state import StateManager
from agentdisruptbench.core.trace import TraceCollector
from agentdisruptbench.tasks.schemas import Task
from agentdisruptbench.tools.registry import ToolRegistry
from agentdisruptbench.tools.stateful import wrap_tool_with_state

logger = logging.getLogger("agentdisruptbench.evaluator")


class Evaluator:
    """Orchestrate one (task, profile) evaluation run.

    1. Instantiate DisruptionEngine from the profile configs.
    2. Create StateManager for tracking side effects.
    3. Wrap all required tools via stateful wrapper + ToolProxy.
    4. Execute the agent function (user-supplied callable).
    5. Compute metrics via MetricsCalculator (including state metrics).

    Parameters:
        agent_fn:   A callable ``(task, tools) → str`` that runs the agent.
                    ``tools`` is a dict of ``name → ToolProxy``.
                    Must return the agent's final textual output.
        tool_registry: Pre-populated tool registry (mock or real).
        metrics:    MetricsCalculator instance.
    """

    def __init__(
        self,
        agent_fn: Callable[[Task, dict[str, Any]], str],
        tool_registry: ToolRegistry,
        metrics: MetricsCalculator | None = None,
    ) -> None:
        self._agent_fn = agent_fn
        self._tool_registry = tool_registry
        self._metrics = metrics or MetricsCalculator()

    def run(
        self,
        task: Task,
        configs: list[DisruptionConfig],
        profile_name: str = "unknown",
        agent_id: str = "default",
        seed: int = 42,
        baseline_result: BenchmarkResult | None = None,
    ) -> BenchmarkResult:
        """Execute one evaluation run.

        Args:
            task:            Task definition to evaluate.
            configs:         Disruption configs for this profile.
            profile_name:    Name of the disruption profile.
            agent_id:        Identifier for the agent under test.
            seed:            Random seed for deterministic replay.
            baseline_result: Clean-profile result (None if this IS the clean run).

        Returns:
            Fully populated BenchmarkResult.
        """
        # -- Setup --
        engine = DisruptionEngine(configs=configs, seed=seed)
        trace_collector = TraceCollector()
        state_manager = StateManager()

        # Wrap required tools (stateful wrapper + disruption proxy)
        proxied_tools: dict[str, Any] = {}
        for tool_name in task.required_tools:
            if self._tool_registry.has(tool_name):
                fn = self._tool_registry.get(tool_name)
                # Wrap with stateful layer first (records side effects)
                fn = wrap_tool_with_state(tool_name, fn, state_manager)
                # Then wrap with disruption proxy
                proxied_tools[tool_name] = ToolProxy(
                    name=tool_name, fn=fn,
                    engine=engine, trace_collector=trace_collector,
                )
            else:
                logger.warning("tool_not_found name=%s task=%s", tool_name, task.task_id)

        # -- Snapshot state before execution --
        before_snapshot = state_manager.snapshot()

        # -- Execute agent --
        start = time.monotonic()
        agent_output = ""
        try:
            agent_output = self._agent_fn(task, proxied_tools)
        except Exception:
            logger.exception("agent_failed task=%s", task.task_id)
            agent_output = "[AGENT ERROR] — agent raised an unhandled exception"
        duration = time.monotonic() - start

        # -- Snapshot state after execution and compute diff --
        after_snapshot = state_manager.snapshot()
        state_diff = StateManager.diff(before_snapshot, after_snapshot)

        # -- Compute metrics (now including state metrics) --
        traces = trace_collector.get_traces()
        result = self._metrics.compute(
            task=task,
            traces=traces,
            agent_output=agent_output or "",
            baseline_result=baseline_result,
            agent_id=agent_id,
            profile_name=profile_name,
            seed=seed,
            duration_seconds=duration,
            state_actions=state_manager.get_actions(),
            state_diff=state_diff,
            idempotency_violations=len(state_manager.get_idempotency_violations()),
        )

        logger.info(
            "run_complete task=%s profile=%s success=%s partial=%.2f "
            "side_effects=%d compensations=%d loops=%d duration=%.1fs",
            task.task_id, profile_name, result.success,
            result.partial_score, result.idempotency_violations,
            result.compensation_count, result.loop_count, duration,
        )
        return result
