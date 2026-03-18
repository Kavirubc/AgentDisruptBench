"""
AgentDisruptBench — Stateful Tool Wrapper
==========================================

File:        stateful.py
Purpose:     Wraps side-effect mock tools so their results are also
             recorded in the StateManager. Non-side-effect tools pass
             through unchanged. Fully backwards compatible — if no
             StateManager is provided, the wrapper is a no-op passthrough.

Author:      AgentDisruptBench Contributors
License:     MIT
Created:     2026-03-18
Modified:    2026-03-18

Key Functions:
    wrap_tool_with_state : Returns a wrapped callable that records
                           side effects in the StateManager.
    SIDE_EFFECT_TOOLS    : Set of tool names that cause state mutations.

Convention:
    Every source file MUST include a header block like this one.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from agentdisruptbench.core.state import COMPENSATION_PAIRS, StateManager

logger = logging.getLogger("agentdisruptbench.stateful")


#: Tools that mutate state (must be tracked).
SIDE_EFFECT_TOOLS: set[str] = {
    name for name, comp in COMPENSATION_PAIRS.items()
    if comp is not None or name in {
        "cancel_booking", "process_refund", "rollback_deployment",
        "resolve_incident", "apply_coupon",
    }
}

#: Mapping of tool_name → (collection, id_field) for state tracking.
_TOOL_STATE_MAP: dict[str, tuple[str, str]] = {
    "place_order": ("orders", "order_id"),
    "process_refund": ("refunds", "refund_id"),
    "book_flight": ("bookings", "booking_id"),
    "cancel_booking": ("bookings", "booking_id"),
    "transfer_funds": ("transfers", "transfer_id"),
    "deploy_service": ("deployments", "deployment_id"),
    "rollback_deployment": ("deployments", "deployment_id"),
    "create_incident": ("incidents", "incident_id"),
    "resolve_incident": ("incidents", "incident_id"),
    "update_cart": ("carts", "cart_id"),
    "apply_coupon": ("carts", "cart_id"),
}


def wrap_tool_with_state(
    tool_name: str,
    fn: Callable,
    state_manager: StateManager | None,
) -> Callable:
    """Wrap a tool callable to also record state mutations.

    For non-side-effect tools or when state_manager is None, returns
    the original callable unchanged.

    Args:
        tool_name:     Logical tool name.
        fn:            Original tool callable.
        state_manager: StateManager instance (None = no-op).

    Returns:
        Wrapped callable that records results in the StateManager.
    """
    if state_manager is None or tool_name not in _TOOL_STATE_MAP:
        return fn

    collection, id_field = _TOOL_STATE_MAP[tool_name]

    # Determine operation type
    _COMPENSATION_OPS = {
        "cancel_booking", "process_refund", "rollback_deployment",
        "resolve_incident",
    }
    is_compensation = tool_name in _COMPENSATION_OPS

    def stateful_wrapper(**kwargs: Any) -> Any:
        # Call the original tool
        result = fn(**kwargs)

        # Extract entity ID from result
        entity_id = "unknown"
        if isinstance(result, dict):
            entity_id = str(result.get(id_field, result.get("id", "unknown")))

        # Determine operation
        operation = "update" if is_compensation else "create"

        # Record in state
        state_manager.write(
            tool_name=tool_name,
            collection=collection,
            entity_id=entity_id,
            data=result if isinstance(result, dict) else {"result": result},
            operation=operation,
        )

        logger.debug(
            "stateful_write tool=%s collection=%s entity=%s op=%s",
            tool_name, collection, entity_id, operation,
        )
        return result

    return stateful_wrapper
