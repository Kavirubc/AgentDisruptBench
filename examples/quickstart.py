#!/usr/bin/env python3
"""
AgentDisruptBench — Quick Start Example
=========================================

File:        quickstart.py
Purpose:     Demonstrates basic usage of AgentDisruptBench:
             1. Load built-in tasks and tools
             2. Define a simple rule-based agent
             3. Run the benchmark with clean and hostile profiles
             4. Generate a Markdown report

Author:      AgentDisruptBench Contributors
License:     MIT
Created:     2026-03-09
Modified:    2026-03-09

Usage:
    cd AgentDisruptBench
    python -m venv .venv && source .venv/bin/activate
    pip install -e .
    python examples/quickstart.py

Convention:
    Every source file MUST include a header block like this one.
"""

from __future__ import annotations

import json
import sys
from typing import Any

# AgentDisruptBench imports
from agentdisruptbench import (
    BenchmarkConfig,
    BenchmarkRunner,
    TaskRegistry,
    ToolRegistry,
)
from agentdisruptbench.harness.reporter import Reporter
from agentdisruptbench.tasks.schemas import Task


def simple_agent(task: Task, tools: dict[str, Any]) -> str:
    """A minimal rule-based agent that calls tools in order.

    This agent simply iterates through the required tools listed in the
    task definition and calls each one with placeholder arguments. It
    demonstrates the benchmark contract: ``(task, tools) → str``.
    """
    results = []

    for tool_name in task.required_tools:
        if tool_name not in tools:
            results.append(f"Tool {tool_name} not available")
            continue

        tool_fn = tools[tool_name]

        # Build simple kwargs based on tool name
        kwargs = _default_kwargs(tool_name)

        try:
            result = tool_fn(**kwargs)
            results.append(f"[{tool_name}] SUCCESS: {json.dumps(result, default=str)[:200]}")
        except Exception as exc:
            results.append(f"[{tool_name}] ERROR: {exc}")

            # Simple retry strategy
            try:
                result = tool_fn(**kwargs)
                results.append(f"[{tool_name}] RETRY SUCCESS: {json.dumps(result, default=str)[:200]}")
            except Exception as retry_exc:
                results.append(f"[{tool_name}] RETRY FAILED: {retry_exc}")

    return "\n".join(results)


def _default_kwargs(tool_name: str) -> dict[str, Any]:
    """Generate sensible default kwargs for each simulated tool."""
    defaults: dict[str, dict] = {
        "search_products": {"query": "blue widget", "max_results": 3},
        "check_inventory": {"product_id": "PRD-abc123"},
        "place_order": {"customer_id": "C-8821", "product_id": "PRD-abc123", "quantity": 1},
        "get_order_status": {"order_id": "ORD-778899"},
        "process_refund": {"order_id": "ORD-778899", "reason": "defective"},
        "get_customer_profile": {"customer_id": "C-8821"},
        "apply_coupon": {"cart_id": "CART-001", "coupon_code": "SAVE20"},
        "update_cart": {"cart_id": "CART-001", "product_id": "PRD-abc123", "quantity": 1},
        "search_flights": {"origin": "SFO", "destination": "JFK", "date": "2026-04-15"},
        "get_flight_details": {"flight_id": "FLT-abc123"},
        "book_flight": {"flight_id": "FLT-abc123", "passenger_name": "John Smith"},
        "cancel_booking": {"booking_id": "BKG-abc123"},
        "search_hotels": {"location": "Paris", "check_in": "2026-05-15", "check_out": "2026-05-20"},
        "check_hotel_availability": {"hotel_id": "HTL-abc123", "check_in": "2026-05-15", "check_out": "2026-05-20"},
        "get_weather": {"location": "London", "date": "2026-04-20"},
        "currency_convert": {"amount": 1000.0, "from_currency": "USD", "to_currency": "EUR"},
        "get_account_balance": {"account_id": "ACC-001122"},
        "transfer_funds": {"from_account": "ACC-001122", "to_account": "ACC-003344", "amount": 500.0},
        "get_transaction_history": {"account_id": "ACC-001122"},
        "get_exchange_rate": {"base_currency": "USD", "target_currency": "EUR"},
        "validate_card": {"card_number": "4111111111111234", "expiry": "12/27", "cvv": "123"},
        "check_credit_limit": {"account_id": "ACC-005566"},
        "get_service_health": {"service_name": "api-gateway"},
        "deploy_service": {"service_name": "api-gateway", "version": "v2.1.0"},
        "rollback_deployment": {"deployment_id": "DEP-abc123"},
        "get_logs": {"service_name": "api-gateway", "severity": "error"},
        "get_metrics": {"service_name": "api-gateway", "metric_type": "cpu"},
        "run_tests": {"service_name": "api-gateway", "test_suite": "unit"},
        "create_incident": {"title": "High latency", "severity": "P2", "service_name": "api-gateway"},
        "resolve_incident": {"incident_id": "INC-abc123", "resolution": "Deployed hotfix"},
    }
    return defaults.get(tool_name, {})


def main():
    """Run the quickstart example."""
    print("=" * 60)
    print(" AgentDisruptBench — Quick Start Example")
    print("=" * 60)

    # 1. Load built-in resources
    print("\n[1/4] Loading tasks and tools...")
    task_registry = TaskRegistry.from_builtin()
    tool_registry = ToolRegistry.from_simulated_tools()
    print(f"  → {len(task_registry)} tasks loaded")
    print(f"  → {len(tool_registry)} tools available")
    print(f"  → Domains: {task_registry.domains()}")

    # 2. Configure benchmark
    print("\n[2/4] Configuring benchmark...")
    config = BenchmarkConfig(
        profiles=["clean", "mild_production", "hostile_environment"],
        seeds=[42],
        max_difficulty=2,  # Keep it short for the demo
        agent_id="simple_rule_agent",
    )
    print(f"  → Profiles: {config.profiles}")
    print(f"  → Max difficulty: {config.max_difficulty}")

    # 3. Run benchmark
    print("\n[3/4] Running benchmark...")
    runner = BenchmarkRunner(
        agent_fn=simple_agent,
        task_registry=task_registry,
        tool_registry=tool_registry,
        config=config,
    )
    results = runner.run_all()
    print(f"  → {len(results)} runs completed")

    # 4. Generate report
    print("\n[4/4] Generating report...")
    reporter = Reporter(output_dir="results")
    paths = reporter.generate(results)
    for name, path in paths.items():
        print(f"  → {name}: {path}")

    # Summary
    print("\n" + "=" * 60)
    success_count = sum(1 for r in results if r.success)
    print(f"  Total runs:    {len(results)}")
    print(f"  Successful:    {success_count}/{len(results)}")
    print(f"  Success rate:  {success_count / max(len(results), 1) * 100:.1f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
