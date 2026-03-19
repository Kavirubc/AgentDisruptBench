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
SIDE_EFFECT_TOOLS: set[str] = set(COMPENSATION_PAIRS)

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

#: Per-tool operation semantics for state tracking.
#: Tools not in this map default to "create".
_TOOL_OPERATION: dict[str, str] = {
    "cancel_booking": "update",       # status change on existing entity
    "process_refund": "create",       # new entity in refunds collection
    "rollback_deployment": "update",  # status change on existing entity
    "resolve_incident": "update",     # status change on existing entity
    "update_cart": "update",          # modifies existing cart
    "apply_coupon": "update",         # modifies existing cart
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
    operation = _TOOL_OPERATION.get(tool_name, "create")

    def stateful_wrapper(**kwargs: Any) -> Any:
        # Call the original tool
        result = fn(**kwargs)

        # Extract entity ID: prefer kwargs (canonical for cancel/update),
        # then result dict, then fall back to "unknown"
        entity_id_value = kwargs.get(id_field) or kwargs.get("id")
        if entity_id_value is None and isinstance(result, dict):
            entity_id_value = result.get(id_field) or result.get("id")
        entity_id = str(entity_id_value or "unknown")

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
