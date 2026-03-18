"""
AgentDisruptBench — Metrics Calculator
=======================================

File:        metrics.py
Purpose:     Computes all benchmark metrics from raw traces and agent output.
             Defines BenchmarkResult, the canonical output of a single
             (task, profile) evaluation run.

Author:      AgentDisruptBench Contributors
License:     MIT
Created:     2026-03-09
Modified:    2026-03-18

Key Definitions:
    BenchmarkResult    : Dataclass holding all metrics for one run.
    MetricsCalculator  : Stateless calculator that produces BenchmarkResult
                         from traces, agent output, and ground-truth.

Metrics (normative definitions — §9 of spec):
    task_success        : partial_score >= 0.8 or exact match on correct_final_answer.
    partial_score       : Weighted sum of satisfied evaluation rubric criteria.
    resilience_ratio    : success_rate_disrupted / success_rate_clean.
    recovery_rate       : recovered_failures / total_failures (1.0 if none).
    retry_efficiency    : successful_retries / total_retries (1.0 if none).
    mean_steps_to_recovery : Avg tool calls between failure and recovery.
    extra_tool_calls    : total_disrupted − total_clean (can be negative).

Convention:
    Every source file MUST include a header block like this one.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from agentdisruptbench.core.state import COMPENSATION_PAIRS
from agentdisruptbench.core.trace import ToolCallTrace
from agentdisruptbench.tasks.schemas import Task

logger = logging.getLogger("agentdisruptbench.metrics")


# ---------------------------------------------------------------------------
# BenchmarkResult
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkResult:
    """All metrics for a single (task, profile) evaluation run.

    Attributes — Identity:
        task_id, agent_id, profile_name, seed.

    Attributes — Outcome:
        success, partial_score, agent_output.

    Attributes — Resilience:
        resilience_ratio, recovery_rate, mean_steps_to_recovery,
        retry_efficiency.

    Attributes — Graceful Degradation:
        acknowledged_failure, attempted_alternative.

    Attributes — Cost of Resilience:
        total_tool_calls, extra_tool_calls, total_latency_ms, extra_latency_ms.

    Attributes — Disruption Statistics:
        disruptions_encountered, disruptions_recovered,
        disruption_types_seen, max_cascade_depth.

    Attributes — Raw:
        traces, duration_seconds.
    """

    # Identity
    task_id: str
    agent_id: str
    profile_name: str
    seed: int

    # Task outcome
    success: bool
    partial_score: float
    agent_output: str

    # Resilience metrics
    resilience_ratio: float | None
    recovery_rate: float
    mean_steps_to_recovery: float
    retry_efficiency: float

    # Graceful degradation
    acknowledged_failure: bool
    attempted_alternative: bool

    # Cost of resilience
    total_tool_calls: int
    extra_tool_calls: int | None
    total_latency_ms: float
    extra_latency_ms: float | None

    # Disruption statistics
    disruptions_encountered: int
    disruptions_recovered: int
    disruption_types_seen: list[str]
    max_cascade_depth: int

    # State & compensation metrics (P0)
    compensation_count: int = 0
    compensation_success_rate: float = 0.0
    side_effect_score: float = 0.0
    idempotency_violations: int = 0
    loop_count: int = 0

    # Raw data
    traces: list[ToolCallTrace] = field(default_factory=list)
    duration_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Metrics Calculator
# ---------------------------------------------------------------------------


class MetricsCalculator:
    """Stateless calculator that produces :class:`BenchmarkResult`.

    Examines traces and agent output against ground truth to compute all
    benchmark metrics as defined in §9 of the AgentDisruptBench spec.
    """

    @staticmethod
    def _normalize_output(agent_output: Any) -> str:
        """Coerce agent_output to str.

        Some LLMs (e.g. Gemini) may return structured content as a list
        of parts or dicts. We join them into a single string.
        """
        if isinstance(agent_output, str):
            return agent_output
        if isinstance(agent_output, list):
            parts = []
            for item in agent_output:
                if isinstance(item, dict):
                    parts.append(str(item.get("text", item)))
                else:
                    parts.append(str(item))
            return " ".join(parts)
        return str(agent_output) if agent_output is not None else ""

    def compute(
        self,
        task: Task,
        traces: list[ToolCallTrace],
        agent_output: str,
        baseline_result: BenchmarkResult | None,
        agent_id: str,
        profile_name: str,
        seed: int,
        duration_seconds: float,
        state_diff: dict | None = None,
        idempotency_violations: int = 0,
    ) -> BenchmarkResult:
        """Compute metrics for a single (task, profile) run.

        Args:
            task:             The task definition with ground truth.
            traces:           All ToolCallTrace records from this run.
            agent_output:     The agent's final textual response.
            baseline_result:  The clean-profile BenchmarkResult (None if this
                              is the clean run itself).
            agent_id:         Identifier for the agent under test.
            profile_name:     Name of the disruption profile used.
            seed:             Random seed used.
            duration_seconds: Wall-clock seconds the run took.

        Returns:
            Fully populated :class:`BenchmarkResult`.
        """

        # -- Normalize output (Gemini may return list content) ----------------
        agent_output = self._normalize_output(agent_output)

        # -- Partial score (rubric evaluation) --------------------------------
        partial_score = self._evaluate_rubric(task, traces, agent_output)
        success = self._is_success(task, partial_score, agent_output)

        # -- Recovery analysis ------------------------------------------------
        (
            recovery_rate,
            mean_steps,
            retry_efficiency,
            disruptions_encountered,
            disruptions_recovered,
        ) = self._compute_recovery(traces)

        # -- Disruption statistics --------------------------------------------
        disruption_types_seen = list({
            t.disruption_fired for t in traces if t.disruption_fired
        })
        max_cascade_depth = self._cascade_depth(traces)

        # -- Cost of resilience -----------------------------------------------
        total_tool_calls = len(traces)
        total_latency_ms = sum(t.observed_latency_ms for t in traces)
        extra_tool_calls: int | None = None
        extra_latency_ms: float | None = None
        resilience_ratio: float | None = None

        if baseline_result is not None:
            extra_tool_calls = total_tool_calls - baseline_result.total_tool_calls
            extra_latency_ms = total_latency_ms - baseline_result.total_latency_ms
            if baseline_result.success:
                resilience_ratio = (1.0 if success else 0.0)
            else:
                resilience_ratio = None

        # -- Graceful degradation ---------------------------------------------
        acknowledged_failure = self._check_acknowledged(agent_output)
        attempted_alternative = self._check_alternative(traces, task)

        # -- Compensation & state metrics (P0) ---------------------------------
        comp_count, comp_success = self._compute_compensation(traces)
        side_effect_score = self._compute_side_effect_score(state_diff or {})
        loop_count = self._compute_loops(traces)

        return BenchmarkResult(
            task_id=task.task_id,
            agent_id=agent_id,
            profile_name=profile_name,
            seed=seed,
            success=success,
            partial_score=partial_score,
            agent_output=agent_output,
            resilience_ratio=resilience_ratio,
            recovery_rate=recovery_rate,
            mean_steps_to_recovery=mean_steps,
            retry_efficiency=retry_efficiency,
            acknowledged_failure=acknowledged_failure,
            attempted_alternative=attempted_alternative,
            total_tool_calls=total_tool_calls,
            extra_tool_calls=extra_tool_calls,
            total_latency_ms=total_latency_ms,
            extra_latency_ms=extra_latency_ms,
            disruptions_encountered=disruptions_encountered,
            disruptions_recovered=disruptions_recovered,
            disruption_types_seen=disruption_types_seen,
            max_cascade_depth=max_cascade_depth,
            compensation_count=comp_count,
            compensation_success_rate=comp_success,
            side_effect_score=side_effect_score,
            idempotency_violations=idempotency_violations,
            loop_count=loop_count,
            traces=traces,
            duration_seconds=duration_seconds,
        )

    # ------------------------------------------------------------------
    # Partial score — rubric-based evaluation
    # ------------------------------------------------------------------

    def _evaluate_rubric(
        self, task: Task, traces: list[ToolCallTrace], agent_output: str
    ) -> float:
        """Evaluate agent output against ground-truth rubric.

        Uses string-matching on agent output and trace inspection to
        determine which criteria are met.  Each criterion is weighted
        per ``evaluation_rubric``.
        """
        gt = task.ground_truth
        rubric = gt.evaluation_rubric
        if not rubric:
            return 1.0 if gt.correct_final_answer is None else 0.0

        score = 0.0
        output_lower = agent_output.lower()
        tool_names_called = {t.tool_name for t in traces}

        for criterion, weight in rubric.items():
            criterion_lower = criterion.lower().replace("_", " ")
            met = False

            # Check if criterion maps to a required tool call
            for req_tool in gt.required_tool_calls:
                if req_tool.lower().replace("_", " ") in criterion_lower:
                    if req_tool in tool_names_called:
                        met = True
                        break

            # Check if criterion keywords appear in agent output
            if not met:
                keywords = criterion_lower.split()
                if any(kw in output_lower for kw in keywords if len(kw) > 3):
                    met = True

            if met:
                score += weight

        return min(1.0, score)

    # ------------------------------------------------------------------
    # Success determination
    # ------------------------------------------------------------------

    def _is_success(
        self, task: Task, partial_score: float, agent_output: str
    ) -> bool:
        """Determine if the task was successful.

        Success = exact match on ``correct_final_answer`` or
        ``partial_score >= 0.8``.
        """
        gt = task.ground_truth
        if gt.correct_final_answer is not None:
            expected = str(gt.correct_final_answer).strip().lower()
            if expected in agent_output.strip().lower():
                return True
        return partial_score >= 0.8

    # ------------------------------------------------------------------
    # Recovery analysis
    # ------------------------------------------------------------------

    def _compute_recovery(
        self, traces: list[ToolCallTrace]
    ) -> tuple[float, float, float, int, int]:
        """Compute recovery_rate, mean_steps_to_recovery, retry_efficiency.

        A recovery: tool call fails (disruption), then a subsequent call
        to the same tool succeeds within the same run.

        Returns:
            (recovery_rate, mean_steps, retry_efficiency,
             disruptions_encountered, disruptions_recovered)
        """
        failed_tools: dict[str, list[int]] = {}   # tool -> [index of failure]
        for i, t in enumerate(traces):
            if t.disruption_fired and not t.observed_success:
                failed_tools.setdefault(t.tool_name, []).append(i)

        disruptions_encountered = sum(len(v) for v in failed_tools.values())
        if disruptions_encountered == 0:
            return 1.0, 0.0, 1.0, 0, 0

        recovered = 0
        steps_to_recovery: list[int] = []
        total_retries = 0
        successful_retries = 0

        for tool_name, fail_indices in failed_tools.items():
            for fail_idx in fail_indices:
                # Look for next successful call to same tool
                for j in range(fail_idx + 1, len(traces)):
                    if traces[j].tool_name == tool_name:
                        total_retries += 1
                        if traces[j].observed_success:
                            recovered += 1
                            successful_retries += 1
                            steps_to_recovery.append(j - fail_idx)
                            break

        recovery_rate = recovered / disruptions_encountered
        mean_steps = (
            sum(steps_to_recovery) / len(steps_to_recovery)
            if steps_to_recovery
            else 0.0
        )
        retry_eff = (
            successful_retries / total_retries if total_retries > 0 else 1.0
        )

        return (
            recovery_rate,
            mean_steps,
            retry_eff,
            disruptions_encountered,
            recovered,
        )

    # ------------------------------------------------------------------
    # Cascade depth
    # ------------------------------------------------------------------

    def _cascade_depth(self, traces: list[ToolCallTrace]) -> int:
        """Count max consecutive cascade failures."""
        max_depth = 0
        current_depth = 0
        for t in traces:
            if t.disruption_fired == "cascading":
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            else:
                current_depth = 0
        return max_depth

    # ------------------------------------------------------------------
    # Graceful degradation checks
    # ------------------------------------------------------------------

    def _check_acknowledged(self, agent_output: str) -> bool:
        """Check if the agent communicated tool failure to the user."""
        keywords = [
            "unable to", "failed to", "error", "could not",
            "unavailable", "timed out", "sorry", "cannot",
            "not able to", "issue", "problem",
        ]
        lower = agent_output.lower()
        return any(kw in lower for kw in keywords)

    def _check_alternative(
        self, traces: list[ToolCallTrace], task: Task
    ) -> bool:
        """Check if agent tried a different tool after a failure."""
        for i, t in enumerate(traces):
            if t.disruption_fired and not t.observed_success:
                # Did the agent call a *different* tool after this?
                for j in range(i + 1, min(i + 4, len(traces))):
                    if traces[j].tool_name != t.tool_name:
                        return True
        return False

    # ------------------------------------------------------------------
    # Compensation detection (P0)
    # ------------------------------------------------------------------

    def _compute_compensation(
        self,
        traces: list[ToolCallTrace],
    ) -> tuple[int, float]:
        """Detect compensation patterns in traces.

        A compensation is when a side-effect tool call is followed by its
        compensating tool call (e.g. book_flight → cancel_booking).

        Returns:
            (compensation_count, compensation_success_rate)
        """
        # Build set of side-effect tools that were called
        side_effect_calls: list[tuple[int, str]] = []
        for i, t in enumerate(traces):
            if t.tool_name in COMPENSATION_PAIRS and t.observed_success:
                comp_tool = COMPENSATION_PAIRS[t.tool_name]
                if comp_tool is not None:
                    side_effect_calls.append((i, t.tool_name))

        if not side_effect_calls:
            return 0, 0.0

        compensated = 0
        for idx, tool_name in side_effect_calls:
            comp_tool = COMPENSATION_PAIRS[tool_name]
            # Check if compensating tool was called after the side-effect
            for t in traces[idx + 1:]:
                if t.tool_name == comp_tool and t.observed_success:
                    compensated += 1
                    break

        total = len(side_effect_calls)
        return compensated, (compensated / total if total > 0 else 0.0)

    # ------------------------------------------------------------------
    # Side-effect score (P0)
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_side_effect_score(state_diff: dict) -> float:
        """Compute unresolved side-effect score from state diff.

        0.0 = no unresolved state changes (clean run or all compensated).
        1.0 = maximum unresolved side effects.

        Normalization: ``min(1.0, total_changes / 5.0)`` — 5+ unresolved
        changes saturate at 1.0.  Can be refined with severity weighting.
        """
        if not state_diff:
            return 0.0

        total_changes = 0
        for coll_changes in state_diff.values():
            total_changes += len(coll_changes)

        # Normalize: 0 changes = 0.0, 5+ changes = 1.0
        return min(1.0, total_changes / 5.0)

    # ------------------------------------------------------------------
    # Loop detection (P0)
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_loops(
        traces: list[ToolCallTrace], min_repeat: int = 3
    ) -> int:
        """Count loops: N (>= min_repeat) consecutive identical calls.

        An identical call = same tool_name AND same inputs.

        Returns:
            Number of distinct loops detected.
        """
        if len(traces) < min_repeat:
            return 0

        loops = 0
        streak = 1

        for i in range(1, len(traces)):
            same_tool = traces[i].tool_name == traces[i - 1].tool_name
            same_inputs = traces[i].inputs == traces[i - 1].inputs
            if same_tool and same_inputs:
                streak += 1
            else:
                if streak >= min_repeat:
                    loops += 1
                streak = 1

        # Check final streak
        if streak >= min_repeat:
            loops += 1

        return loops
