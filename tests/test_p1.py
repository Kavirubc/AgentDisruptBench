"""
AgentDisruptBench — Unit Tests: P1/P2 Features
================================================

File:        test_p1.py
Purpose:     Tests for P1 (adversarial, impossible, recovery classification,
             reliability) and P2 (planning ratio, handover, hallucination,
             failure taxonomy) features.

Author:      AgentDisruptBench Contributors
License:     MIT
Created:     2026-03-19
Modified:    2026-03-19

Convention:
    Every source file MUST include a header block like this one.
"""

from agentdisruptbench.core.metrics import BenchmarkResult, MetricsCalculator
from agentdisruptbench.core.trace import ToolCallTrace
from agentdisruptbench.tasks.registry import TaskRegistry
from agentdisruptbench.tasks.schemas import GroundTruth, Task

# ===================================================================
# Helpers
# ===================================================================


def _make_trace(
    tool_name: str,
    disruption: str | None = None,
    observed_success: bool = True,
    real_success: bool = True,
    latency: float = 10.0,
    inputs: dict | None = None,
) -> ToolCallTrace:
    return ToolCallTrace(
        call_id="t",
        tool_name=tool_name,
        inputs=inputs or {},
        real_result={"ok": True},
        observed_result={"ok": True} if observed_success else {"error": "fail"},
        real_success=real_success,
        observed_success=observed_success,
        disruption_fired=disruption,
        real_latency_ms=latency,
        observed_latency_ms=latency,
        error=None if observed_success else "fail",
        timestamp=0.0,
        call_number=1,
    )


def _standard_task(task_type: str = "standard") -> Task:
    return Task(
        task_id="test_001",
        title="Test",
        description="Test task",
        domain="retail",
        difficulty=1,
        task_type=task_type,
        required_tools=["search_products"],
        expected_tool_call_depth=1,
        ground_truth=GroundTruth(
            expected_outcome="Test",
            required_tool_calls=["search_products"],
            evaluation_rubric={"searched": 1.0},
        ),
    )


def _impossible_task() -> Task:
    return Task(
        task_id="imp_001",
        title="Impossible",
        description="Impossible task",
        domain="retail",
        difficulty=3,
        task_type="impossible",
        required_tools=["search_products", "place_order"],
        expected_tool_call_depth=1,
        ground_truth=GroundTruth(
            expected_outcome="Agent recognizes impossibility",
            required_tool_calls=["search_products"],
            forbidden_tool_calls=["place_order"],
            evaluation_rubric={"recognized": 1.0},
            impossibility_reason="Product does not exist",
        ),
    )


# ===================================================================
# P1-5 & P1-6: Adversarial / Impossible Tasks
# ===================================================================


class TestTaskTypes:
    """Task type schema and loading tests."""

    def test_adversarial_tasks_load(self):
        """Adversarial YAML tasks load correctly."""
        reg = TaskRegistry.from_builtin()
        adv = [t for t in reg.all_tasks() if t.task_type == "adversarial"]
        assert len(adv) == 8
        assert all(t.ground_truth.trap_description for t in adv)

    def test_impossible_tasks_load(self):
        """Impossible YAML tasks load correctly."""
        reg = TaskRegistry.from_builtin()
        imp = [t for t in reg.all_tasks() if t.task_type == "impossible"]
        assert len(imp) == 8
        assert all(t.ground_truth.impossibility_reason for t in imp)
        assert all(t.ground_truth.forbidden_tool_calls for t in imp)

    def test_handover_tasks_load(self):
        """Handover YAML tasks load correctly."""
        reg = TaskRegistry.from_builtin()
        ho = [t for t in reg.all_tasks() if t.task_id.startswith("handover_")]
        assert len(ho) == 4

    def test_impossible_success_when_agent_recognizes(self):
        """Impossible task succeeds if agent recognizes impossibility."""
        calc = MetricsCalculator()
        task = _impossible_task()
        traces = [_make_trace("search_products")]
        result = calc.compute(
            task=task,
            traces=traces,
            agent_output="I'm sorry, this product is unavailable.",
            baseline_result=None,
            agent_id="a",
            profile_name="clean",
            seed=42,
            duration_seconds=1.0,
        )
        assert result.success is True
        assert result.graceful_giveup is True

    def test_impossible_fails_if_forbidden_tool_called(self):
        """Impossible task fails if forbidden tool is called."""
        calc = MetricsCalculator()
        task = _impossible_task()
        traces = [
            _make_trace("search_products"),
            _make_trace("place_order"),
        ]
        result = calc.compute(
            task=task,
            traces=traces,
            agent_output="I placed the order for you.",
            baseline_result=None,
            agent_id="a",
            profile_name="clean",
            seed=42,
            duration_seconds=1.0,
        )
        assert result.success is False
        assert result.graceful_giveup is False


# ===================================================================
# P1-7: Recovery Strategy Classification
# ===================================================================


class TestRecoveryClassification:
    """Tests for recovery strategy classification."""

    def test_retry_strategy(self):
        """Same tool retry → RETRY."""
        calc = MetricsCalculator()
        traces = [
            _make_trace("search_products", disruption="timeout", observed_success=False),
            _make_trace("search_products"),  # retry succeeds
        ]
        strategies = calc._classify_recovery(traces, "done")
        assert strategies == ["RETRY"]

    def test_alternative_strategy(self):
        """A known substitute tool succeeds → ALTERNATIVE."""
        calc = MetricsCalculator()
        traces = [
            _make_trace("search_flights", disruption="http_500", observed_success=False),
            _make_trace("search_alternative_flights"),  # known substitute
        ]
        strategies = calc._classify_recovery(traces, "Used alternative flight search")
        assert strategies == ["ALTERNATIVE"]

    def test_non_substitute_tool_not_alternative(self):
        """An unrelated different tool success does NOT produce ALTERNATIVE."""
        calc = MetricsCalculator()
        traces = [
            _make_trace("search_flights", disruption="http_500", observed_success=False),
            _make_trace("get_weather"),  # unrelated — not a known substitute
        ]
        strategies = calc._classify_recovery(traces, "Got weather instead")
        assert strategies == []

    def test_escalation_strategy(self):
        """Agent mentions escalation → ESCALATION."""
        calc = MetricsCalculator()
        traces = [
            _make_trace("search_products", disruption="timeout", observed_success=False),
        ]
        strategies = calc._classify_recovery(traces, "Please contact support for assistance")
        assert strategies == ["ESCALATION"]

    def test_giveup_strategy(self):
        """Agent acknowledges failure → GIVEUP."""
        calc = MetricsCalculator()
        traces = [
            _make_trace("search_products", disruption="timeout", observed_success=False),
        ]
        strategies = calc._classify_recovery(traces, "Sorry, I was unable to complete this")
        assert strategies == ["GIVEUP"]

    def test_dominant_strategy(self):
        """dominant_strategy reflects most common."""
        calc = MetricsCalculator()
        task = _standard_task()
        traces = [
            _make_trace("search_products", disruption="timeout", observed_success=False),
            _make_trace("search_products"),
            _make_trace("search_products", disruption="http_500", observed_success=False),
            _make_trace("search_products"),
        ]
        result = calc.compute(
            task=task,
            traces=traces,
            agent_output="Here are the products",
            baseline_result=None,
            agent_id="a",
            profile_name="p",
            seed=42,
            duration_seconds=1.0,
        )
        assert result.dominant_strategy == "RETRY"


# ===================================================================
# P2 Features
# ===================================================================


class TestP2Features:
    """Tests for P2 metric helpers."""

    def test_planning_ratio_no_traces(self):
        # No tool calls means the entire run was pre-tool (planning) time
        ratio = MetricsCalculator._compute_planning_ratio([], 10.0)
        assert ratio == 1.0

    def test_planning_ratio_zero_duration(self):
        ratio = MetricsCalculator._compute_planning_ratio([], 0.0)
        assert ratio == 0.0

    def test_planning_ratio_some_tool_time(self):
        traces = [_make_trace("x", latency=500.0)]  # 500ms = 0.5s
        ratio = MetricsCalculator._compute_planning_ratio(traces, 2.0)
        # Planning = 2.0 - 0.5 = 1.5s → ratio = 1.5/2.0 = 0.75
        assert ratio == 0.75

    def test_handover_detected(self):
        assert MetricsCalculator._check_handover("Please escalate to a human agent")
        assert MetricsCalculator._check_handover("Contact support for help")
        assert not MetricsCalculator._check_handover("Here are the results")

    def test_hallucination_detection(self):
        traces = [
            _make_trace("book_flight", observed_success=False),
        ]
        rate = MetricsCalculator._compute_hallucination_rate(
            traces,
            "I booked your flight successfully",
            expected_tools={"book_flight"},
        )
        assert rate == 1.0  # Agent claimed "booked" but book_flight failed

    def test_hallucination_no_traces(self):
        # Agent claims action with no traces at all — should detect hallucination
        rate = MetricsCalculator._compute_hallucination_rate(
            [],
            "I booked your flight successfully",
            expected_tools={"book_flight"},
        )
        assert rate == 1.0  # No tool calls, but agent claimed booking

    def test_no_hallucination(self):
        traces = [_make_trace("book_flight")]
        rate = MetricsCalculator._compute_hallucination_rate(
            traces,
            "I booked your flight successfully",
            expected_tools={"book_flight"},
        )
        assert rate == 0.0

    def test_failure_taxonomy(self):
        # Only disruptions where observed_success=False should be counted
        traces = [
            _make_trace("a", disruption="timeout", observed_success=False),
            _make_trace("b", disruption="http_429", observed_success=False),
            _make_trace("c", disruption="http_500", observed_success=False),
            _make_trace("d", disruption="malformed_json", observed_success=False),
            # This disruption fired but the call still succeeded — should NOT count
            _make_trace("e", disruption="latency", observed_success=True),
        ]
        cats = MetricsCalculator._classify_failures(traces)
        assert cats["TIMEOUT"] == 1
        assert cats["RATE_LIMIT"] == 1
        assert cats["SERVER_ERROR"] == 1
        assert cats["MALFORMED_RESPONSE"] == 1
        # Soft-fault with success should not appear
        assert cats.get("TIMEOUT", 0) == 1  # still only 1, not 2

    def test_failure_taxonomy_empty(self):
        cats = MetricsCalculator._classify_failures([])
        assert cats == {}
