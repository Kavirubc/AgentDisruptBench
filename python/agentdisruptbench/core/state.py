"""
AgentDisruptBench — StateManager
=================================

File:        state.py
Purpose:     In-memory mutable state layer for side-effect tools. Tracks
             all write operations, enables snapshot/diff for evaluation,
             and detects idempotency violations (duplicate writes).

Author:      AgentDisruptBench Contributors
License:     MIT
Created:     2026-03-18
Modified:    2026-03-18

Key Classes:
    StateAction     : Record of a single state-mutating operation.
    StateManager    : Thread-safe in-memory DB with snapshot, diff,
                      and idempotency violation tracking.

Convention:
    Every source file MUST include a header block like this one.
"""

from __future__ import annotations

import copy
import logging
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("agentdisruptbench.state")


# ---------------------------------------------------------------------------
# Compensation pair registry
# ---------------------------------------------------------------------------

#: Maps side-effect tool → its compensating tool name.
#: Self-compensation pairs (e.g. transfer_funds → transfer_funds) are excluded
#: to avoid overcounting; those require input-level analysis.
COMPENSATION_PAIRS: dict[str, str | None] = {
    "book_flight": "cancel_booking",
    "cancel_booking": None,
    "place_order": "process_refund",
    "process_refund": None,
    "transfer_funds": None,  # self-compensation requires input analysis
    "deploy_service": "rollback_deployment",
    "rollback_deployment": None,
    "create_incident": "resolve_incident",
    "resolve_incident": None,
    "update_cart": None,  # self-compensation requires input analysis
    "apply_coupon": None,
}


# ---------------------------------------------------------------------------
# StateAction — record of one mutation
# ---------------------------------------------------------------------------

@dataclass
class StateAction:
    """Record of a single state-mutating operation.

    Attributes:
        action_id:          Unique identifier for this action.
        tool_name:          Name of the tool that performed the mutation.
        collection:         Which state collection was mutated.
        entity_id:          The key/ID of the entity created or modified.
        operation:          One of 'create', 'update', 'delete'.
        data:               The data written.
        compensating_tool:  Name of the tool that can undo this action (or None).
        timestamp:          Monotonic time of the action.
    """

    action_id: str
    tool_name: str
    collection: str
    entity_id: str
    operation: str
    data: dict[str, Any]
    compensating_tool: str | None
    timestamp: float


# ---------------------------------------------------------------------------
# StateManager
# ---------------------------------------------------------------------------

class StateManager:
    """In-memory mutable state layer for benchmark evaluation.

    Maintains per-collection dictionaries (bookings, orders, transfers,
    deployments, incidents, carts) that side-effect tools can read/write.

    Thread-safe via threading.Lock.

    Key features:
        - ``write()`` — record a state mutation with automatic idempotency check.
        - ``read()`` — query entities from a collection.
        - ``snapshot()`` — deep-copy the full state for before/after comparison.
        - ``diff()`` — compare two snapshots and return changes.
        - ``get_actions()`` — get the full action log for trace analysis.
        - ``get_idempotency_violations()`` — return duplicate write attempts.
        - ``reset()`` — clear all state between tasks.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._collections: dict[str, dict[str, dict[str, Any]]] = {
            "bookings": {},
            "orders": {},
            "transfers": {},
            "deployments": {},
            "incidents": {},
            "carts": {},
            "refunds": {},
        }
        self._actions: list[StateAction] = []
        self._action_keys: set[str] = set()  # (tool_name:entity_id) dedup
        self._idempotency_violations: list[tuple[str, str]] = []  # (tool, entity_id)

    # -- write / read -------------------------------------------------------

    def write(
        self,
        tool_name: str,
        collection: str,
        entity_id: str,
        data: dict[str, Any],
        operation: str = "create",
        action_id: str | None = None,
    ) -> StateAction:
        """Record a state mutation.

        Idempotency behaviour is **detection-only**: duplicate ``create``
        operations are logged as violations via
        :meth:`get_idempotency_violations` but the write still proceeds
        so that the benchmark can observe the agent's behaviour.

        Args:
            tool_name:   Name of the tool performing the write.
            collection:  Target collection (e.g. 'bookings', 'orders').
            entity_id:   Unique key for the entity.
            data:        Data payload to store.
            operation:   One of 'create', 'update', 'delete'.
            action_id:   Optional explicit action ID; auto-generated if None.

        Returns:
            The StateAction record that was created.
        """

        with self._lock:
            # Idempotency check
            dedup_key = f"{tool_name}:{entity_id}"
            if dedup_key in self._action_keys and operation == "create":
                self._idempotency_violations.append((tool_name, entity_id))
                logger.warning(
                    "idempotency_violation tool=%s entity=%s", tool_name, entity_id
                )

            self._action_keys.add(dedup_key)

            # Ensure collection exists
            if collection not in self._collections:
                self._collections[collection] = {}

            # Apply mutation
            if operation == "delete":
                self._collections[collection].pop(entity_id, None)
            else:
                self._collections[collection][entity_id] = copy.deepcopy(data)

            # Record action
            action = StateAction(
                action_id=action_id or str(uuid.uuid4()),
                tool_name=tool_name,
                collection=collection,
                entity_id=entity_id,
                operation=operation,
                data=copy.deepcopy(data),
                compensating_tool=COMPENSATION_PAIRS.get(tool_name),
                timestamp=time.monotonic(),
            )
            self._actions.append(action)

            logger.debug(
                "state_write tool=%s collection=%s entity=%s op=%s",
                tool_name, collection, entity_id, operation,
            )
            return action

    def read(self, collection: str, entity_id: str | None = None) -> Any:
        """Read from a collection.

        Args:
            collection: Collection name.
            entity_id:  If provided, returns the specific entity. Otherwise
                        returns the whole collection dict.

        Returns:
            The entity dict, the collection dict, or None if not found.
        """
        with self._lock:
            coll = self._collections.get(collection, {})
            if entity_id is not None:
                return copy.deepcopy(coll.get(entity_id))
            return copy.deepcopy(coll)

    # -- snapshot / diff ----------------------------------------------------

    def snapshot(self) -> dict[str, dict[str, dict[str, Any]]]:
        """Deep-copy the entire state for before/after comparison."""
        with self._lock:
            return copy.deepcopy(self._collections)

    @staticmethod
    def diff(
        before: dict[str, dict[str, dict[str, Any]]],
        after: dict[str, dict[str, dict[str, Any]]],
    ) -> dict[str, list[dict[str, Any]]]:
        """Compare two snapshots and return changes.

        Returns:
            Dict mapping collection name → list of change records.
            Each change record has 'entity_id', 'type' ('created', 'modified',
            'deleted'), and optionally 'before' / 'after' data.
        """
        changes: dict[str, list[dict[str, Any]]] = {}

        all_collections = set(before.keys()) | set(after.keys())
        for coll in all_collections:
            coll_before = before.get(coll, {})
            coll_after = after.get(coll, {})
            coll_changes: list[dict[str, Any]] = []

            all_ids = set(coll_before.keys()) | set(coll_after.keys())
            for eid in all_ids:
                in_before = eid in coll_before
                in_after = eid in coll_after

                if not in_before and in_after:
                    coll_changes.append({
                        "entity_id": eid, "type": "created",
                        "after": coll_after[eid],
                    })
                elif in_before and not in_after:
                    coll_changes.append({
                        "entity_id": eid, "type": "deleted",
                        "before": coll_before[eid],
                    })
                elif coll_before[eid] != coll_after[eid]:
                    coll_changes.append({
                        "entity_id": eid, "type": "modified",
                        "before": coll_before[eid], "after": coll_after[eid],
                    })

            if coll_changes:
                changes[coll] = coll_changes

        return changes

    # -- queries ------------------------------------------------------------

    def get_actions(self) -> list[StateAction]:
        """Return the full action log."""
        with self._lock:
            return list(self._actions)

    def get_idempotency_violations(self) -> list[tuple[str, str]]:
        """Return list of (tool_name, entity_id) idempotency violations."""
        with self._lock:
            return list(self._idempotency_violations)

    # -- lifecycle ----------------------------------------------------------

    def reset(self) -> None:
        """Clear all state. Call between tasks."""
        with self._lock:
            for coll in self._collections.values():
                coll.clear()
            self._actions.clear()
            self._action_keys.clear()
            self._idempotency_violations.clear()
            logger.debug("state_reset")
