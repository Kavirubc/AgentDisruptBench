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
# Alternative-tool substitution map
# ---------------------------------------------------------------------------

# Maps a tool name to the set of tools that are known substitutes for it.
# Only tools listed here will be classified as ALTERNATIVE recoveries.
# Extend this mapping as the benchmark grows to cover new tool families.
ALTERNATIVE_TOOL_MAP: dict[str, set[str]] = {
    "search_flights": {"search_alternative_flights", "search_flights_v2"},
    "book_flight": {"book_alternative_flight", "reserve_flight"},
    "search_hotels": {"search_alternative_hotels", "search_hotels_v2"},
    "book_hotel": {"book_alternative_hotel", "reserve_hotel"},
    "transfer_funds": {"wire_transfer", "send_payment"},
    "get_stock_price": {"get_stock_quote", "fetch_market_data"},
    "deploy_service": {"deploy_service_v2", "rollout_service"},
}


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

    # P1 metrics
    graceful_giveup: bool = False
    recovery_strategies: list[str] = field(default_factory=list)
    dominant_strategy: str = ""

    # P2 metrics
    planning_time_ratio: float = 0.0
    handover_detected: bool = False
    tool_hallucination_rate: float = 0.0
    state_equivalent_success: bool = False
    budget_exceeded: bool = False
    token_usage: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    failure_categories: dict[str, int] = field(default_factory=dict)

    # Task metadata (authoritative fields for downstream slicing)
    task_domain: str = ""
    task_difficulty: int = 0
    task_description: str = ""
    runner_name: str = ""

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
        run_start_time: float | None = None,
        state_diff: dict | None = None,
        idempotency_violations: int = 0,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        runner_name: str = "",
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
        success = self._is_success(task, partial_score, agent_output, traces)

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

        # -- P1: Impossible task giveup ----------------------------------------
        graceful_giveup = (
            task.task_type == "impossible" and success
        )

        # -- P1: Recovery strategy classification ------------------------------
        recovery_strategies = self._classify_recovery(traces, agent_output)
        if recovery_strategies:
            from collections import Counter
            _strategy_counter = Counter(recovery_strategies)
            dominant_strategy = min(
                _strategy_counter,
                key=lambda s: (-_strategy_counter[s], s),
            )
        else:
            dominant_strategy = ""

        # -- P2: Planning time ratio -------------------------------------------
        planning_time_ratio = self._compute_planning_ratio(
            traces, duration_seconds, run_start_time=run_start_time
        )

        # -- P2: Handover detection --------------------------------------------
        handover_detected = self._check_handover(agent_output)

        # -- P2: Tool hallucination detection ----------------------------------
        tool_hallucination_rate = self._compute_hallucination_rate(
            traces, agent_output, expected_tools=set(task.required_tools)
        )

        # -- P2: Failure taxonomy ----------------------------------------------
        failure_categories = self._classify_failures(traces)

        return BenchmarkResult(
            task_id=task.task_id,
            agent_id=agent_id,
            profile_name=profile_name,
            seed=seed,
            success=success,
            partial_score=partial_score,
            agent_output=agent_output,
            task_domain=task.domain,
            task_difficulty=task.difficulty,
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
            graceful_giveup=graceful_giveup,
            recovery_strategies=recovery_strategies,
            dominant_strategy=dominant_strategy,
            planning_time_ratio=planning_time_ratio,
            handover_detected=handover_detected,
            tool_hallucination_rate=tool_hallucination_rate,
            failure_categories=failure_categories,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            token_usage=prompt_tokens + completion_tokens,
            runner_name=runner_name,
            task_description=task.description,
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
        self, task: Task, partial_score: float, agent_output: str,
        traces: list[ToolCallTrace] | None = None,
    ) -> bool:
        """Determine if the task was successful.

        For **standard** / **adversarial** tasks:
            Success = exact match on ``correct_final_answer`` or
            ``partial_score >= 0.8``.

        For **impossible** tasks:
            Success = agent recognized impossibility (acknowledged failure)
            AND did not call any forbidden tools.
        """
        gt = task.ground_truth

        if task.task_type == "impossible":
            # Agent must NOT have called forbidden tools
            if traces:
                called = {t.tool_name for t in traces}
                for ft in gt.forbidden_tool_calls:
                    if ft in called:
                        return False
            # Agent must have acknowledged the impossibility
            return self._check_acknowledged(agent_output)

        # Standard / adversarial
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
    # P1-7: Recovery strategy classification
    # ------------------------------------------------------------------

    def _classify_recovery(
        self, traces: list[ToolCallTrace], agent_output: str
    ) -> list[str]:
        """Classify each recovery event into a strategy category.

        Categories:
            RETRY       — Same tool called again → succeeds.
            ALTERNATIVE — Different tool called after failure → succeeds.
            ESCALATION  — Agent output mentions handover/escalation.
            GIVEUP      — Agent gives up after failure.
            LUCKY       — Recovery happened but strategy unclear.

        Returns:
            List of strategy strings, one per recovery event.
        """
        strategies: list[str] = []

        for i, t in enumerate(traces):
            if not (t.disruption_fired and not t.observed_success):
                continue

            recovered = False
            for j in range(i + 1, len(traces)):
                nxt = traces[j]
                if nxt.tool_name == t.tool_name and nxt.observed_success:
                    # Same tool retried and succeeded
                    strategies.append("RETRY")
                    recovered = True
                    break
                if (
                    nxt.tool_name != t.tool_name
                    and nxt.observed_success
                    and nxt.tool_name in ALTERNATIVE_TOOL_MAP.get(t.tool_name, set())
                ):
                    # A known substitute tool succeeded
                    strategies.append("ALTERNATIVE")
                    recovered = True
                    break

            if not recovered:
                # Check if agent mentioned escalation / handover
                if self._check_handover(agent_output):
                    strategies.append("ESCALATION")
                elif self._check_acknowledged(agent_output):
                    strategies.append("GIVEUP")
                # Otherwise emit no strategy — unrecovered failures without
                # an explicit response pattern are not classified as LUCKY.

        return strategies

    # ------------------------------------------------------------------
    # P2-9: Planning time ratio
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_planning_ratio(
        traces: list[ToolCallTrace],
        duration_seconds: float,
        run_start_time: float | None = None,
    ) -> float:
        """Estimate the fraction of run time spent before the first tool call.

        If *run_start_time* (Unix epoch seconds) is provided, planning time is
        measured as the wall-clock gap between run start and the first trace's
        timestamp.  Otherwise it is approximated as
        ``(total_duration - sum_of_tool_latencies) / total_duration``.

        Returns 1.0 when there are no traces (the entire run was pre-tool time).
        Returns 0.0 when duration_seconds is zero or negative.
        """
        if duration_seconds <= 0:
            return 0.0
        if not traces:
            return 1.0  # No tool calls — entire run is planning time

        if run_start_time is not None:
            first_call_time = traces[0].timestamp
            planning_time = max(0.0, first_call_time - run_start_time)
        else:
            # Approximation: non-tool latency as proxy for pre-first-call time
            tool_time_s = sum(t.observed_latency_ms for t in traces) / 1000.0
            planning_time = max(0.0, duration_seconds - tool_time_s)

        return min(1.0, planning_time / duration_seconds)

    # ------------------------------------------------------------------
    # P2-10: Handover detection
    # ------------------------------------------------------------------

    @staticmethod
    def _check_handover(agent_output: str) -> bool:
        """Check if agent suggested handing off to a human."""
        keywords = [
            "hand over", "handover", "hand off", "handoff",
            "escalate", "human agent", "human support",
            "contact support", "manual intervention",
            "speak to a representative", "customer service",
        ]
        lower = agent_output.lower()
        return any(kw in lower for kw in keywords)

    # ------------------------------------------------------------------
    # P2-11: Tool hallucination detection
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_hallucination_rate(
        traces: list[ToolCallTrace],
        agent_output: str,
        expected_tools: set[str] | None = None,
    ) -> float:
        """Detect tool hallucinations by comparing output claims vs traces.

        A hallucination occurs when the agent claims to have performed an
        action (e.g. "I booked a flight") but either no tool trace exists for
        that action or the tool call failed.  This handles the most suspicious
        case: the agent claims success with zero tool calls.

        Args:
            traces:         Tool call traces from the run (may be empty).
            agent_output:   Agent's final textual response.
            expected_tools: Set of tool names the task involves (used for
                            hallucination detection when traces are absent).
                            Falls back to tools seen in traces if None.

        Returns:
            Ratio of hallucinated claims to total claims (0.0–1.0).
        """
        # Build set of tools that actually succeeded
        successful_tools = {t.tool_name for t in traces if t.observed_success}

        # Expand with ALTERNATIVE_TOOL_MAP equivalents so that a
        # substitute-tool recovery (e.g. reserve_flight succeeding when
        # book_flight failed) is not counted as a hallucination.
        successful_or_equivalent: set[str] = set(successful_tools)
        for primary, alternatives in ALTERNATIVE_TOOL_MAP.items():
            tool_family = {primary, *alternatives}
            if successful_tools & tool_family:
                successful_or_equivalent.update(tool_family)

        all_tool_names = expected_tools or {t.tool_name for t in traces}
        # Also expand relevance check with alternatives
        all_tool_names_expanded: set[str] = set(all_tool_names)
        for primary, alternatives in ALTERNATIVE_TOOL_MAP.items():
            tool_family = {primary, *alternatives}
            if all_tool_names & tool_family:
                all_tool_names_expanded.update(tool_family)

        # Common action verbs that map to specific tools
        _ACTION_VERBS = {
            "booked": "book_flight",
            "cancelled": "cancel_booking",
            "canceled": "cancel_booking",
            "transferred": "transfer_funds",
            "ordered": "place_order",
            "deployed": "deploy_service",
            "refunded": "process_refund",
        }

        output_lower = agent_output.lower()
        hallucinations = 0
        total_claims = 0

        for verb, tool in _ACTION_VERBS.items():
            if verb in output_lower:
                # Only score this claim if the tool is relevant to the run
                if tool in all_tool_names_expanded or not all_tool_names:
                    total_claims += 1
                    # Hallucination: claimed action but tool never succeeded
                    # (including via known substitutes)
                    if tool not in successful_or_equivalent:
                        hallucinations += 1

        return hallucinations / total_claims if total_claims > 0 else 0.0

    # ------------------------------------------------------------------
    # P2-15: Failure taxonomy (AgentRx-aligned)
    # ------------------------------------------------------------------

    @staticmethod
    def _classify_failures(traces: list[ToolCallTrace]) -> dict[str, int]:
        """Classify trace failures into AgentRx-aligned categories.

        9 categories:
            TIMEOUT, RATE_LIMIT, AUTH_FAILURE, SERVER_ERROR,
            MALFORMED_RESPONSE, DATA_ERROR, CASCADING,
            INTERMITTENT, QUOTA_EXHAUSTED
        """
        counts: dict[str, int] = {}

        _DISRUPTION_TO_CATEGORY = {
            "timeout": "TIMEOUT",
            "latency": "TIMEOUT",
            "http_429": "RATE_LIMIT",
            "http_401": "AUTH_FAILURE",
            "http_403": "AUTH_FAILURE",
            "auth_expiry": "AUTH_FAILURE",
            "http_500": "SERVER_ERROR",
            "http_502": "SERVER_ERROR",
            "http_503": "SERVER_ERROR",
            "malformed_json": "MALFORMED_RESPONSE",
            "truncated": "MALFORMED_RESPONSE",
            "null_response": "MALFORMED_RESPONSE",
            "missing_fields": "DATA_ERROR",
            "type_mismatch": "DATA_ERROR",
            "schema_drift": "DATA_ERROR",
            "wrong_data": "DATA_ERROR",
            "cascading": "CASCADING",
            "intermittent": "INTERMITTENT",
            "flapping": "INTERMITTENT",
            "quota_exhausted": "QUOTA_EXHAUSTED",
        }

        for t in traces:
            if t.disruption_fired and not t.observed_success:
                cat = _DISRUPTION_TO_CATEGORY.get(
                    t.disruption_fired, "UNKNOWN"
                )
                counts[cat] = counts.get(cat, 0) + 1

        return counts

    # ------------------------------------------------------------------
    # Compensation detection (P0)
    # ------------------------------------------------------------------

    def _compute_compensation(
        self,
        traces: list[ToolCallTrace],
    ) -> tuple[int, float]:
        """Detect compensation patterns in traces via entity-level pairing.

        A compensation is when a side-effect tool's entity is later
        addressed by its compensating tool (e.g. book_flight(BKG-001) →
        cancel_booking(BKG-001)).

        Uses ``real_success`` to catch hidden successful side effects
        (where the tool actually succeeded but the agent was shown a
        disrupted response).  Pairing is one-to-one: each compensating
        trace can satisfy at most one side-effect.

        Returns:
            (compensation_count, compensation_success_rate)
        """
        from agentdisruptbench.tools.stateful import _TOOL_STATE_MAP

        # Build list of (index, tool_name, entity_id) for real side-effect calls
        side_effect_calls: list[tuple[int, str, str]] = []
        for i, t in enumerate(traces):
            if t.tool_name in COMPENSATION_PAIRS and t.real_success:
                comp_tool = COMPENSATION_PAIRS[t.tool_name]
                if comp_tool is not None:
                    # Extract entity_id from inputs or real_result
                    eid = self._extract_entity_id(t, _TOOL_STATE_MAP)
                    side_effect_calls.append((i, t.tool_name, eid))

        if not side_effect_calls:
            return 0, 0.0

        # One-to-one matching: track consumed compensating traces
        consumed: set[int] = set()
        compensated = 0
        for idx, tool_name, entity_id in side_effect_calls:
            comp_tool = COMPENSATION_PAIRS[tool_name]
            for j, t in enumerate(traces[idx + 1:], start=idx + 1):
                if j in consumed:
                    continue
                if t.tool_name == comp_tool and t.real_success:
                    comp_eid = self._extract_entity_id(t, _TOOL_STATE_MAP)
                    # Match by entity if both have identifiable entities
                    if entity_id != "unknown" and comp_eid != "unknown":
                        if entity_id != comp_eid:
                            continue
                    consumed.add(j)
                    compensated += 1
                    break

        total = len(side_effect_calls)
        return compensated, (compensated / total if total > 0 else 0.0)

    @staticmethod
    def _extract_entity_id(
        trace: ToolCallTrace,
        tool_state_map: dict[str, tuple[str, str]],
    ) -> str:
        """Extract entity ID from a trace's inputs or real_result."""
        if trace.tool_name not in tool_state_map:
            return "unknown"
        _, id_field = tool_state_map[trace.tool_name]

        # Try inputs first (canonical for cancel/update tools)
        eid = trace.inputs.get(id_field) or trace.inputs.get("id")
        if eid is not None:
            return str(eid)

        # Then try real_result
        if isinstance(trace.real_result, dict):
            eid = trace.real_result.get(id_field) or trace.real_result.get("id")
            if eid is not None:
                return str(eid)

        return "unknown"

    # ------------------------------------------------------------------
    # Side-effect score (P0)
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_side_effect_score(state_diff: dict) -> float:
        """Compute unresolved side-effect score from state diff.

        0.0 = no unresolved state changes (clean run or all compensated).
        1.0 = maximum unresolved side effects.

        Resolved changes (deletions, or modifications where the status
        indicates cancellation/resolution/refund) are excluded so that
        compensated flows don't inflate the score.

        Normalization: ``min(1.0, unresolved / 5.0)`` — 5+ unresolved
        changes saturate at 1.0.
        """
        if not state_diff:
            return 0.0

        _RESOLVED_STATUSES = {
            "cancelled", "canceled", "resolved", "refunded",
            "rolled_back", "compensated", "reversed",
        }

        unresolved = 0
        for coll_changes in state_diff.values():
            for change in coll_changes:
                # Deletions are resolved (entity was removed/cleaned up)
                if change.get("type") == "deleted":
                    continue
                # Modifications where status indicates resolution
                if change.get("type") == "modified":
                    after = change.get("after", {})
                    status = str(after.get("status", "")).lower()
                    if status in _RESOLVED_STATUSES:
                        continue
                unresolved += 1

        # Normalize: 0 unresolved = 0.0, 5+ = 1.0
        return min(1.0, unresolved / 5.0)

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
