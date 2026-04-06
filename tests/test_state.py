"""
AgentDisruptBench — Unit Tests: State & P0 Metrics
====================================================

File:        test_state.py
Purpose:     Tests for StateManager, compensation detection, idempotency
             violation tracking, loop detection, and end-to-end stateful
             evaluation pipeline.

Author:      AgentDisruptBench Contributors
License:     MIT
Created:     2026-03-18
Modified:    2026-03-18

Convention:
    Every source file MUST include a header block like this one.
"""

from __future__ import annotations

from typing import Any

from agentdisruptbench.core.metrics import MetricsCalculator
from agentdisruptbench.core.state import StateManager
from agentdisruptbench.core.trace import ToolCallTrace
from agentdisruptbench.tools.simulated_tools import RetailTools
from agentdisruptbench.tools.stateful import SIDE_EFFECT_TOOLS, wrap_tool_with_state

# ===================================================================
# StateManager Tests
# ===================================================================


class TestStateManager:
    """Tests for the StateManager class."""

    def test_write_and_read(self):
        """Write to a collection, verify it's queryable."""
        sm = StateManager()
        sm.write(
            tool_name="place_order",
            collection="orders",
            entity_id="ORD-001",
            data={"product": "widget", "quantity": 2, "status": "confirmed"},
        )

        result = sm.read("orders", "ORD-001")
        assert result is not None
        assert result["product"] == "widget"
        assert result["quantity"] == 2

    def test_read_all_collection(self):
        """Read entire collection."""
        sm = StateManager()
        sm.write("place_order", "orders", "ORD-001", {"total": 10})
        sm.write("place_order", "orders", "ORD-002", {"total": 20})

        all_orders = sm.read("orders")
        assert len(all_orders) == 2
        assert "ORD-001" in all_orders
        assert "ORD-002" in all_orders

    def test_read_nonexistent(self):
        """Reading a nonexistent entity returns None."""
        sm = StateManager()
        assert sm.read("orders", "ORD-999") is None

    def test_snapshot_and_diff(self):
        """Snapshot before/after writes, verify diff captures changes."""
        sm = StateManager()
        before = sm.snapshot()

        sm.write("book_flight", "bookings", "BK-001", {"flight": "FLT-abc"})
        sm.write("place_order", "orders", "ORD-001", {"total": 42})

        after = sm.snapshot()
        diff = StateManager.diff(before, after)

        assert "bookings" in diff
        assert "orders" in diff
        assert len(diff["bookings"]) == 1
        assert diff["bookings"][0]["type"] == "created"
        assert diff["bookings"][0]["entity_id"] == "BK-001"

    def test_diff_modification(self):
        """Diff detects modifications to existing entities."""
        sm = StateManager()
        sm.write("deploy_service", "deployments", "DEP-001", {"version": "1.0"})
        before = sm.snapshot()

        sm.write("deploy_service", "deployments", "DEP-001", {"version": "2.0"}, operation="update")
        after = sm.snapshot()
        diff = StateManager.diff(before, after)

        assert "deployments" in diff
        assert diff["deployments"][0]["type"] == "modified"

    def test_diff_deletion(self):
        """Diff detects deletions."""
        sm = StateManager()
        sm.write("create_incident", "incidents", "INC-001", {"severity": "high"})
        before = sm.snapshot()

        sm.write("resolve_incident", "incidents", "INC-001", {}, operation="delete")
        after = sm.snapshot()
        diff = StateManager.diff(before, after)

        assert "incidents" in diff
        assert diff["incidents"][0]["type"] == "deleted"

    def test_diff_no_changes(self):
        """Diff with no changes returns empty dict."""
        sm = StateManager()
        sm.write("place_order", "orders", "ORD-001", {"total": 10})
        snap = sm.snapshot()
        diff = StateManager.diff(snap, snap)
        assert diff == {}

    def test_idempotency_detection(self):
        """Write same action twice, verify violation detected."""
        sm = StateManager()
        sm.write("book_flight", "bookings", "BK-001", {"flight": "FLT-abc"})
        sm.write("book_flight", "bookings", "BK-001", {"flight": "FLT-abc"})

        violations = sm.get_idempotency_violations()
        assert len(violations) == 1
        assert violations[0] == ("book_flight", "BK-001")

    def test_no_idempotency_violation_for_updates(self):
        """Update operations should not trigger idempotency violations."""
        sm = StateManager()
        sm.write("deploy_service", "deployments", "DEP-001", {"v": "1.0"})
        sm.write("rollback_deployment", "deployments", "DEP-001", {"v": "rollback"}, operation="update")

        violations = sm.get_idempotency_violations()
        assert len(violations) == 0

    def test_reset(self):
        """Reset clears all state."""
        sm = StateManager()
        sm.write("place_order", "orders", "ORD-001", {"total": 10})
        sm.write("place_order", "orders", "ORD-001", {"total": 10})  # Trigger violation
        assert len(sm.get_actions()) > 0
        assert len(sm.get_idempotency_violations()) > 0

        sm.reset()
        assert sm.read("orders", "ORD-001") is None
        assert len(sm.get_actions()) == 0
        assert len(sm.get_idempotency_violations()) == 0

    def test_action_log(self):
        """Action log records all writes with correct metadata."""
        sm = StateManager()
        sm.write("book_flight", "bookings", "BK-001", {"flight": "FLT-abc"})

        actions = sm.get_actions()
        assert len(actions) == 1
        assert actions[0].tool_name == "book_flight"
        assert actions[0].collection == "bookings"
        assert actions[0].entity_id == "BK-001"
        assert actions[0].compensating_tool == "cancel_booking"


# ===================================================================
# Stateful Tool Wrapper Tests
# ===================================================================


class TestStatefulWrapper:
    """Tests for the stateful tool wrapper."""

    def test_backwards_compat_no_state(self):
        """Simulated tools without state_manager still work identically."""
        # No state manager → original function returned unchanged
        wrapped = wrap_tool_with_state("search_products", RetailTools.search_products, None)
        assert wrapped is RetailTools.search_products

        result = wrapped(query="widget", max_results=3)
        assert "products" in result
        assert len(result["products"]) == 3

    def test_stateful_side_effect_tool(self):
        """Side-effect tools record state when StateManager is provided."""
        sm = StateManager()
        wrapped = wrap_tool_with_state("place_order", RetailTools.place_order, sm)

        result = wrapped(customer_id="CUST-001", product_id="PRD-abc", quantity=1)
        assert "order_id" in result

        # Check state was recorded
        actions = sm.get_actions()
        assert len(actions) == 1
        assert actions[0].tool_name == "place_order"
        assert actions[0].collection == "orders"

    def test_non_side_effect_tool_passthrough(self):
        """Non-side-effect tools pass through unchanged even with StateManager."""
        sm = StateManager()
        wrapped = wrap_tool_with_state("search_products", RetailTools.search_products, sm)
        # search_products is not in _TOOL_STATE_MAP → should be the original fn
        assert wrapped is RetailTools.search_products

    def test_side_effect_tools_set(self):
        """SIDE_EFFECT_TOOLS contains expected tools."""
        assert "place_order" in SIDE_EFFECT_TOOLS
        assert "book_flight" in SIDE_EFFECT_TOOLS
        assert "transfer_funds" in SIDE_EFFECT_TOOLS
        assert "deploy_service" in SIDE_EFFECT_TOOLS


# ===================================================================
# Compensation Detection Tests
# ===================================================================


class TestCompensationDetection:
    """Tests for compensation detection in MetricsCalculator."""

    @staticmethod
    def _make_trace(tool_name: str, success: bool = True, inputs: dict | None = None) -> ToolCallTrace:
        """Helper to create a minimal trace."""
        return ToolCallTrace(
            call_id="test",
            tool_name=tool_name,
            inputs=inputs or {},
            real_result={"ok": True},
            observed_result={"ok": True} if success else None,
            real_success=True,
            observed_success=success,
            disruption_fired=None if success else "http_500",
            real_latency_ms=10.0,
            observed_latency_ms=10.0,
            error=None if success else "fail",
            timestamp=0.0,
            call_number=1,
        )

    def test_compensation_detected(self):
        """book_flight then cancel_booking → compensation_count=1."""
        calc = MetricsCalculator()
        traces = [
            self._make_trace("book_flight"),
            self._make_trace("cancel_booking"),
        ]
        count, rate = calc._compute_compensation(traces)
        assert count == 1
        assert rate == 1.0

    def test_no_compensation(self):
        """book_flight without cancel → compensation_count=0."""
        calc = MetricsCalculator()
        traces = [
            self._make_trace("book_flight"),
            self._make_trace("search_flights"),
        ]
        count, rate = calc._compute_compensation(traces)
        assert count == 0
        assert rate == 0.0

    def test_multiple_compensations(self):
        """Multiple side-effect tools with some compensated."""
        calc = MetricsCalculator()
        traces = [
            self._make_trace("book_flight"),
            self._make_trace("place_order"),
            self._make_trace("cancel_booking"),
            # place_order NOT compensated (no process_refund)
        ]
        count, rate = calc._compute_compensation(traces)
        assert count == 1
        assert rate == 0.5  # 1 out of 2 compensated


# ===================================================================
# Loop Detection Tests
# ===================================================================


class TestLoopDetection:
    """Tests for loop detection in MetricsCalculator."""

    @staticmethod
    def _make_trace(tool_name: str, inputs: dict | None = None, success: bool = True) -> ToolCallTrace:
        return ToolCallTrace(
            call_id="test",
            tool_name=tool_name,
            inputs=inputs or {"q": "widget"},
            real_result={"ok": True},
            observed_result={"ok": True} if success else None,
            real_success=True,
            observed_success=success,
            disruption_fired=None,
            real_latency_ms=10.0,
            observed_latency_ms=10.0,
            error=None,
            timestamp=0.0,
            call_number=1,
        )

    def test_loop_detected(self):
        """3+ identical consecutive calls → loop_count=1."""
        traces = [self._make_trace("search_products")] * 4
        assert MetricsCalculator._compute_loops(traces) == 1

    def test_no_loop_different_tools(self):
        """Different tools → no loops."""
        traces = [
            self._make_trace("search_products"),
            self._make_trace("check_inventory"),
            self._make_trace("search_products"),
        ]
        assert MetricsCalculator._compute_loops(traces) == 0

    def test_no_loop_different_inputs(self):
        """Same tool, different inputs → no loops."""
        traces = [
            self._make_trace("search_products", {"q": "a"}),
            self._make_trace("search_products", {"q": "b"}),
            self._make_trace("search_products", {"q": "c"}),
        ]
        assert MetricsCalculator._compute_loops(traces) == 0

    def test_multiple_loops(self):
        """Two separate loops."""
        traces = [self._make_trace("search_products", {"q": "a"})] * 3 + [
            self._make_trace("check_inventory", {"id": "x"})
        ] * 4
        assert MetricsCalculator._compute_loops(traces) == 2

    def test_below_threshold(self):
        """Only 2 consecutive → not a loop."""
        traces = [self._make_trace("search_products")] * 2
        assert MetricsCalculator._compute_loops(traces) == 0


# ===================================================================
# Side-Effect Score Tests
# ===================================================================


class TestSideEffectScore:
    """Tests for side-effect scoring."""

    def test_no_diff(self):
        """Empty diff → 0.0 score."""
        assert MetricsCalculator._compute_side_effect_score({}) == 0.0

    def test_some_changes(self):
        """A few state changes → moderate score."""
        diff = {
            "orders": [{"entity_id": "ORD-001", "type": "created"}],
            "bookings": [{"entity_id": "BK-001", "type": "created"}],
        }
        score = MetricsCalculator._compute_side_effect_score(diff)
        assert 0.0 < score < 1.0

    def test_many_changes(self):
        """5+ changes → score = 1.0."""
        diff = {
            "orders": [{"entity_id": f"ORD-{i}", "type": "created"} for i in range(5)],
        }
        score = MetricsCalculator._compute_side_effect_score(diff)
        assert score == 1.0

    def test_resolved_changes_excluded(self):
        """Deleted and cancelled changes should not count."""
        diff = {
            "bookings": [
                {"entity_id": "BK-001", "type": "deleted"},  # resolved
                {"entity_id": "BK-002", "type": "modified", "after": {"status": "cancelled"}},  # resolved
                {"entity_id": "BK-003", "type": "created"},  # unresolved
            ],
        }
        score = MetricsCalculator._compute_side_effect_score(diff)
        # Only 1 unresolved out of 3 → 1/5 = 0.2
        assert score == 0.2


# ===================================================================
# Entity-Level Compensation Tests
# ===================================================================


class TestEntityCompensation:
    """Tests for entity-level one-to-one compensation pairing."""

    @staticmethod
    def _make_trace(
        tool_name: str,
        inputs: dict | None = None,
        real_result: dict | None = None,
    ) -> ToolCallTrace:
        return ToolCallTrace(
            call_id="test",
            tool_name=tool_name,
            inputs=inputs or {},
            real_result=real_result or {"ok": True},
            observed_result={"ok": True},
            real_success=True,
            observed_success=True,
            disruption_fired=None,
            real_latency_ms=10.0,
            observed_latency_ms=10.0,
            error=None,
            timestamp=0.0,
            call_number=1,
        )

    def test_entity_matched_compensation(self):
        """Compensation matched by entity ID."""
        calc = MetricsCalculator()
        traces = [
            self._make_trace("book_flight", real_result={"booking_id": "BKG-001"}),
            self._make_trace("cancel_booking", inputs={"booking_id": "BKG-001"}),
        ]
        count, rate = calc._compute_compensation(traces)
        assert count == 1
        assert rate == 1.0

    def test_entity_mismatched_not_compensated(self):
        """Cancel for different entity shouldn't compensate the booking."""
        calc = MetricsCalculator()
        traces = [
            self._make_trace("book_flight", real_result={"booking_id": "BKG-001"}),
            self._make_trace("cancel_booking", inputs={"booking_id": "BKG-999"}),
        ]
        count, rate = calc._compute_compensation(traces)
        assert count == 0
        assert rate == 0.0

    def test_one_to_one_pairing(self):
        """One cancel_booking can only compensate one book_flight."""
        calc = MetricsCalculator()
        traces = [
            self._make_trace("book_flight", real_result={"booking_id": "BKG-001"}),
            self._make_trace("book_flight", real_result={"booking_id": "BKG-001"}),
            self._make_trace("cancel_booking", inputs={"booking_id": "BKG-001"}),
        ]
        count, rate = calc._compute_compensation(traces)
        # Only 1 cancel for 2 bookings → 1 compensated, rate = 0.5
        assert count == 1
        assert rate == 0.5


# ===================================================================
# End-to-End Evaluation with State
# ===================================================================


class TestEvaluatorWithState:
    """Integration test verifying the full pipeline with state management."""

    def test_stateful_evaluation_pipeline(self):
        """Run evaluator with side-effect tools, verify new metrics are populated."""
        from agentdisruptbench import ToolRegistry
        from agentdisruptbench.harness.evaluator import Evaluator
        from agentdisruptbench.tasks.schemas import GroundTruth, Task

        def booking_agent(task: Task, tools: dict[str, Any]) -> str:
            """Agent that books a flight and then cancels it."""
            results = []
            booking_id = None
            if "search_flights" in tools:
                try:
                    r = tools["search_flights"](origin="SFO", destination="JFK", date="2026-04-15")
                    results.append(f"Found {len(r.get('flights', []))} flights")
                except Exception as e:
                    results.append(f"Search failed: {e}")

            if "book_flight" in tools:
                try:
                    r = tools["book_flight"](flight_id="FLT-abc", passenger_name="Test User")
                    booking_id = r.get("booking_id")
                    results.append(f"Booked: {booking_id or 'unknown'}")
                except Exception as e:
                    results.append(f"Booking failed: {e}")

            if "cancel_booking" in tools:
                try:
                    r = tools["cancel_booking"](booking_id=booking_id or "BK-fallback")
                    results.append(f"Cancelled: {r.get('booking_id', booking_id)}")
                except Exception as e:
                    results.append(f"Cancel failed: {e}")

            return " | ".join(results)

        tool_registry = ToolRegistry.from_simulated_tools()
        evaluator = Evaluator(agent_fn=booking_agent, tool_registry=tool_registry)

        # Create a minimal task that uses travel tools
        task = Task(
            task_id="test_stateful_001",
            title="Book and Cancel Flight",
            domain="travel",
            difficulty=1,
            description="Book and cancel a flight",
            required_tools=["search_flights", "book_flight", "cancel_booking"],
            expected_tool_call_depth=3,
            ground_truth=GroundTruth(
                correct_final_answer=None,
                expected_outcome="Agent should search, book, and cancel a flight",
                required_tool_calls=["search_flights", "book_flight", "cancel_booking"],
                evaluation_rubric={"flight_search": 0.3, "booking": 0.3, "cancellation": 0.4},
                recovery_actions=[],
            ),
        )

        result = evaluator.run(
            task=task,
            configs=[],  # clean profile
            profile_name="clean",
            agent_id="test",
            seed=42,
        )

        # Verify new metrics are populated
        assert isinstance(result.compensation_count, int)
        assert isinstance(result.compensation_success_rate, float)
        assert isinstance(result.side_effect_score, float)
        assert isinstance(result.idempotency_violations, int)
        assert isinstance(result.loop_count, int)

        # The agent booked and cancelled → at least 1 compensation should be detected
        assert result.compensation_count >= 1
        assert result.loop_count == 0
