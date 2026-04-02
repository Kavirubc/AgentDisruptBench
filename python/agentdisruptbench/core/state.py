"""
AgentDisruptBench — StateManager
=================================

File:        state.py
Purpose:     Persistent state layer for side-effect tools. Tracks
             all write operations, enables snapshot/diff for evaluation,
             and detects idempotency violations (duplicate writes).
             Backed by SQLite for ground truths and robust state tracking.

Author:      AgentDisruptBench Contributors
License:     MIT
Created:     2026-03-18
Modified:    2026-04-02

Key Classes:
    StateAction     : Record of a single state-mutating operation.
    StateManager    : Thread-safe SQLite DB with snapshot, diff,
                      and idempotency violation tracking.

Convention:
    Every source file MUST include a header block like this one.
"""

from __future__ import annotations

import copy
import json
import logging
import sqlite3
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
    """Persistent state layer for benchmark evaluation.

    Maintains collections (bookings, orders, transfers, deployments, incidents, carts).
    Backed by SQLite to support ground truths, persistence, and state tracking
    across different architectures. Thread-safe.

    Key features:
        - ``write()`` — record a state mutation with automatic idempotency check.
        - ``read()`` — query entities from a collection.
        - ``snapshot()`` — retrieve the full state for before/after comparison.
        - ``diff()`` — compare two snapshots and return changes.
        - ``get_actions()`` — get the full action log for trace analysis.
        - ``get_idempotency_violations()`` — return duplicate write attempts.
        - ``reset()`` — clear all state.
    """

    def __init__(self, db_path: str | None = None) -> None:
        self._lock = threading.Lock()
        self._db_path = db_path or ":memory:"
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._init_db()

    def _init_db(self) -> None:
        with self._lock:
            with self._conn:
                self._conn.executescript('''
                    CREATE TABLE IF NOT EXISTS state_collections (
                        collection_name TEXT,
                        entity_id TEXT,
                        data TEXT,
                        PRIMARY KEY (collection_name, entity_id)
                    );
                    CREATE TABLE IF NOT EXISTS state_actions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        action_id TEXT,
                        tool_name TEXT,
                        collection_name TEXT,
                        entity_id TEXT,
                        operation TEXT,
                        data TEXT,
                        compensating_tool TEXT,
                        timestamp REAL
                    );
                    CREATE TABLE IF NOT EXISTS action_keys (
                        dedup_key TEXT PRIMARY KEY
                    );
                    CREATE TABLE IF NOT EXISTS idempotency_violations (
                        tool_name TEXT,
                        entity_id TEXT
                    );
                ''')

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
        operations are logged as violations but the write still proceeds.

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
        dedup_key = f"{tool_name}:{entity_id}"
        actual_action_id = action_id or str(uuid.uuid4())
        data_json = json.dumps(data)
        comp_tool = COMPENSATION_PAIRS.get(tool_name)
        ts = time.monotonic()

        with self._lock:
            cur = self._conn.cursor()
            try:
                # 1. Idempotency Check
                cur.execute("SELECT 1 FROM action_keys WHERE dedup_key = ?", (dedup_key,))
                exists = cur.fetchone() is not None
                
                if exists and operation == "create":
                    cur.execute(
                        "INSERT INTO idempotency_violations (tool_name, entity_id) VALUES (?, ?)", 
                        (tool_name, entity_id)
                    )
                    logger.warning("idempotency_violation tool=%s entity=%s", tool_name, entity_id)

                if not exists:
                    cur.execute("INSERT INTO action_keys (dedup_key) VALUES (?)", (dedup_key,))

                # 2. Apply mutation
                if operation == "delete":
                    cur.execute(
                        "DELETE FROM state_collections WHERE collection_name = ? AND entity_id = ?",
                        (collection, entity_id)
                    )
                else:
                    cur.execute(
                        """
                        INSERT INTO state_collections (collection_name, entity_id, data) 
                        VALUES (?, ?, ?) 
                        ON CONFLICT(collection_name, entity_id) DO UPDATE SET data=excluded.data
                        """,
                        (collection, entity_id, data_json)
                    )

                # 3. Record action
                cur.execute(
                    """
                    INSERT INTO state_actions (action_id, tool_name, collection_name, entity_id, 
                                               operation, data, compensating_tool, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (actual_action_id, tool_name, collection, entity_id, operation, data_json, comp_tool, ts)
                )

                self._conn.commit()

            except Exception as e:
                self._conn.rollback()
                logger.error("Failed to write to DB: %s", e)
                raise
            finally:
                cur.close()

        logger.debug(
            "state_write tool=%s collection=%s entity=%s op=%s",
            tool_name, collection, entity_id, operation,
        )

        return StateAction(
            action_id=actual_action_id,
            tool_name=tool_name,
            collection=collection,
            entity_id=entity_id,
            operation=operation,
            data=copy.deepcopy(data),
            compensating_tool=comp_tool,
            timestamp=ts,
        )

    def read(self, collection: str, entity_id: str | None = None) -> Any:
        """Read from a collection."""
        with self._lock:
            cur = self._conn.cursor()
            try:
                if entity_id is not None:
                    cur.execute(
                        "SELECT data FROM state_collections WHERE collection_name = ? AND entity_id = ?",
                        (collection, entity_id)
                    )
                    row = cur.fetchone()
                    return json.loads(row[0]) if row else None
                else:
                    cur.execute(
                        "SELECT entity_id, data FROM state_collections WHERE collection_name = ?",
                        (collection,)
                    )
                    rows = cur.fetchall()
                    return {row[0]: json.loads(row[1]) for row in rows}
            finally:
                cur.close()

    # -- snapshot / diff ----------------------------------------------------

    def snapshot(self) -> dict[str, dict[str, dict[str, Any]]]:
        """Retrieve the entire state for before/after comparison."""
        with self._lock:
            cur = self._conn.cursor()
            try:
                cur.execute("SELECT collection_name, entity_id, data FROM state_collections")
                rows = cur.fetchall()

                snapshot_data: dict[str, dict[str, dict[str, Any]]] = {
                    "bookings": {}, "orders": {}, "transfers": {},
                    "deployments": {}, "incidents": {}, "carts": {}, "refunds": {}
                }

                for collection, eid, data_str in rows:
                    if collection not in snapshot_data:
                        snapshot_data[collection] = {}
                    snapshot_data[collection][eid] = json.loads(data_str)

                return snapshot_data
            finally:
                cur.close()

    @staticmethod
    def diff(
        before: dict[str, dict[str, dict[str, Any]]],
        after: dict[str, dict[str, dict[str, Any]]],
    ) -> dict[str, list[dict[str, Any]]]:
        """Compare two snapshots and return changes."""
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
            cur = self._conn.cursor()
            try:
                cur.execute(
                    """
                    SELECT action_id, tool_name, collection_name, entity_id,
                           operation, data, compensating_tool, timestamp
                    FROM state_actions ORDER BY id ASC
                    """
                )
                rows = cur.fetchall()
                return [
                    StateAction(
                        action_id=row[0],
                        tool_name=row[1],
                        collection=row[2],
                        entity_id=row[3],
                        operation=row[4],
                        data=json.loads(row[5]),
                        compensating_tool=row[6],
                        timestamp=row[7],
                    ) for row in rows
                ]
            finally:
                cur.close()

    def get_idempotency_violations(self) -> list[tuple[str, str]]:
        """Return list of (tool_name, entity_id) idempotency violations."""
        with self._lock:
            cur = self._conn.cursor()
            try:
                cur.execute("SELECT tool_name, entity_id FROM idempotency_violations")
                return [(row[0], row[1]) for row in cur.fetchall()]
            finally:
                cur.close()

    # -- lifecycle ----------------------------------------------------------

    def reset(self) -> None:
        """Clear all state. Call between tasks."""
        with self._lock:
            with self._conn:
                self._conn.executescript('''
                    DELETE FROM state_collections;
                    DELETE FROM state_actions;
                    DELETE FROM action_keys;
                    DELETE FROM idempotency_violations;
                ''')
            logger.debug("state_reset")
