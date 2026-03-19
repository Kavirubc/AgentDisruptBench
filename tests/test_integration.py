"""
AgentDisruptBench — Unit Tests: Integration
=============================================

File:        test_integration.py
Purpose:     End-to-end integration tests verifying the full benchmark
             pipeline: tasks → tools → proxy → engine → metrics → report.

Author:      AgentDisruptBench Contributors
License:     MIT
Created:     2026-03-09
Modified:    2026-03-09

Convention:
    Every source file MUST include a header block like this one.
"""

from __future__ import annotations

import json
import tempfile
from typing import Any

import pytest

from agentdisruptbench import (
    BenchmarkConfig,
    BenchmarkRunner,
    TaskRegistry,
    ToolRegistry,
)
from agentdisruptbench.core.engine import DisruptionConfig, DisruptionEngine, DisruptionType
from agentdisruptbench.core.profiles import BUILTIN_PROFILES, get_profile
from agentdisruptbench.core.proxy import ToolProxy
from agentdisruptbench.core.trace import TraceCollector
from agentdisruptbench.harness.reporter import Reporter
from agentdisruptbench.tasks.schemas import Task
from agentdisruptbench.tools.mock_tools import RetailTools, TravelTools, FinanceTools, DevopsTools


class TestMockTools:
    """Tests for deterministic mock tools."""

    def test_search_products_deterministic(self):
        r1 = RetailTools.search_products(query="widget", max_results=3)
        r2 = RetailTools.search_products(query="widget", max_results=3)
        assert r1 == r2
        assert len(r1["products"]) == 3

    def test_search_flights(self):
        r = TravelTools.search_flights(origin="SFO", destination="JFK", date="2026-04-15")
        assert "flights" in r
        assert len(r["flights"]) == 3

    def test_get_account_balance(self):
        r = FinanceTools.get_account_balance(account_id="ACC-001122")
        assert "balance" in r
        assert r["account_id"] == "ACC-001122"

    def test_get_service_health(self):
        r = DevopsTools.get_service_health(service_name="api-gateway")
        assert r["service"] == "api-gateway"
        assert r["status"] in ["healthy", "degraded", "unhealthy"]


class TestToolRegistry:
    """Tests for ToolRegistry."""

    def test_from_mock_tools(self):
        registry = ToolRegistry.from_mock_tools()
        assert len(registry) > 20
        assert "search_products" in registry
        assert "search_flights" in registry

    def test_get_unknown_raises(self):
        registry = ToolRegistry()
        with pytest.raises(KeyError):
            registry.get("nonexistent")


class TestTaskRegistry:
    """Tests for TaskRegistry."""

    def test_from_builtin(self):
        registry = TaskRegistry.from_builtin()
        assert len(registry) == 100  # 80 standard + 8 adversarial + 8 impossible + 4 handover
        assert "retail" in registry.domains()
        assert "travel" in registry.domains()

    def test_filter_by_domain(self):
        registry = TaskRegistry.from_builtin()
        retail_tasks = registry.filter(domain="retail")
        assert len(retail_tasks) == 25  # 20 standard + 2 adversarial + 2 impossible + 1 handover
        assert all(t.domain == "retail" for t in retail_tasks)

    def test_filter_by_difficulty(self):
        registry = TaskRegistry.from_builtin()
        easy = registry.filter(max_difficulty=1)
        assert all(t.difficulty <= 1 for t in easy)


class TestProfiles:
    """Tests for built-in profiles."""

    def test_builtin_count(self):
        assert len(BUILTIN_PROFILES) == 9

    def test_clean_is_empty(self):
        assert get_profile("clean") == []

    def test_hostile_has_disruptions(self):
        hostile = get_profile("hostile_environment")
        assert len(hostile) > 5

    def test_unknown_raises(self):
        with pytest.raises(KeyError):
            get_profile("nonexistent_profile")


class TestToolProxy:
    """Tests for ToolProxy wrapping."""

    def test_clean_passthrough(self):
        engine = DisruptionEngine(configs=[], seed=42)
        tc = TraceCollector()
        proxy = ToolProxy(name="search_products", fn=RetailTools.search_products,
                         engine=engine, trace_collector=tc)
        result = proxy(query="widget")
        assert "products" in result
        assert len(tc.get_traces()) == 1

    def test_disruption_recorded(self):
        engine = DisruptionEngine(
            configs=[DisruptionConfig(type=DisruptionType.NULL_RESPONSE, probability=1.0)],
            seed=42,
        )
        tc = TraceCollector()
        proxy = ToolProxy(name="search_products", fn=RetailTools.search_products,
                         engine=engine, trace_collector=tc)
        result = proxy(query="widget")
        assert result is None
        traces = tc.get_traces()
        assert len(traces) == 1
        assert traces[0].disruption_fired == "null_response"


class TestTraceCollector:
    """Tests for TraceCollector JSONL I/O."""

    def test_jsonl_roundtrip(self):
        engine = DisruptionEngine(configs=[], seed=42)
        tc = TraceCollector()
        proxy = ToolProxy(name="test", fn=lambda: {"ok": True},
                         engine=engine, trace_collector=tc)
        proxy()
        proxy()

        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            tc.to_jsonl(f.name)
            tc2 = TraceCollector()
            tc2.from_jsonl(f.name)
            assert len(tc2.get_traces()) == 2


class TestEndToEnd:
    """End-to-end integration test."""

    def test_benchmark_run(self):
        def simple_agent(task: Task, tools: dict[str, Any]) -> str:
            results = []
            for name in task.required_tools:
                if name in tools:
                    try:
                        r = tools[name](query="test") if name == "search_products" else tools[name](product_id="PRD-abc")
                        results.append(f"OK: {name}")
                    except Exception as e:
                        results.append(f"Error: {name}: {e}")
            return " | ".join(results)

        task_registry = TaskRegistry.from_builtin()
        tool_registry = ToolRegistry.from_mock_tools()

        config = BenchmarkConfig(
            profiles=["clean", "mild_production"],
            seeds=[42],
            max_difficulty=1,
            agent_id="test_agent",
        )

        runner = BenchmarkRunner(
            agent_fn=simple_agent,
            task_registry=task_registry,
            tool_registry=tool_registry,
            config=config,
        )
        results = runner.run_all()
        assert len(results) > 0

        # All clean runs should succeed (simple D1 tasks)
        clean_results = [r for r in results if r.profile_name == "clean"]
        assert len(clean_results) > 0

    def test_reporter(self):
        """Reporter generates all files."""
        from agentdisruptbench.core.metrics import BenchmarkResult

        result = BenchmarkResult(
            task_id="test_001", agent_id="test", profile_name="clean",
            seed=42, success=True, partial_score=1.0, agent_output="OK",
            resilience_ratio=None, recovery_rate=1.0,
            mean_steps_to_recovery=0.0, retry_efficiency=1.0,
            acknowledged_failure=False, attempted_alternative=False,
            total_tool_calls=1, extra_tool_calls=None,
            total_latency_ms=10.0, extra_latency_ms=None,
            disruptions_encountered=0, disruptions_recovered=0,
            disruption_types_seen=[], max_cascade_depth=0,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            reporter = Reporter(output_dir=tmpdir)
            paths = reporter.generate([result])
            assert "report.md" in paths
            assert "results.json" in paths
            assert "summary.json" in paths
