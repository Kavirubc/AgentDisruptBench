"""
AgentDisruptBench — Simple Rule-Based Runner
==============================================

File:        simple_runner.py
Purpose:     A no-LLM baseline runner. Calls each required tool in order
             with sensible defaults, implements a naive retry-once strategy.
             Useful for sanity-checking the benchmark pipeline and as a
             baseline comparison against LLM-powered agents.

Author:      AgentDisruptBench Contributors
License:     MIT
Created:     2026-03-09
Modified:    2026-03-09

Key Classes:
    SimpleRunner : Rule-based agent, no LLM dependency.

Convention:
    Every source file MUST include a header block like this one.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from evaluation.base_runner import BaseAgentRunner, RunnerConfig
from agentdisruptbench.tasks.schemas import Task

logger = logging.getLogger("agentdisruptbench.evaluation.runners.simple")

# Default kwargs per tool — matches mock tool signatures
_TOOL_DEFAULTS: dict[str, dict[str, Any]] = {
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


class SimpleRunner(BaseAgentRunner):
    """Rule-based agent that calls tools sequentially with retry-once.

    No LLM required. Uses hardcoded default arguments per tool.
    Implements a simple retry-once strategy on failures.

    Usage::

        runner = SimpleRunner()
        result = runner.run_task(task, tools)
    """

    def __init__(self, config: RunnerConfig | None = None) -> None:
        super().__init__(config or RunnerConfig(model="none"))

    def run_task(self, task: Task, tools: dict[str, Any]) -> str:
        """Call each required tool in order with retry-once on failure."""
        results: list[str] = []

        for tool_name in task.required_tools:
            if tool_name not in tools:
                results.append(f"[{tool_name}] SKIP — tool not available")
                continue

            tool_fn = tools[tool_name]
            kwargs = _TOOL_DEFAULTS.get(tool_name, {})

            # Attempt 1
            try:
                result = tool_fn(**kwargs)
                results.append(
                    f"[{tool_name}] OK: {json.dumps(result, default=str)[:300]}"
                )
                continue
            except Exception as exc:
                results.append(f"[{tool_name}] FAIL: {exc}")

            # Retry once
            try:
                result = tool_fn(**kwargs)
                results.append(
                    f"[{tool_name}] RETRY OK: {json.dumps(result, default=str)[:300]}"
                )
            except Exception as exc:
                results.append(f"[{tool_name}] RETRY FAIL: {exc}")

        return "\n".join(results)
