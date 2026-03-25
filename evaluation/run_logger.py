"""
AgentDisruptBench — Structured Run Logger
==========================================

File:        run_logger.py
Purpose:     Writes structured JSONL event logs for benchmark runs.
             Used by run_benchmark.py and run_base_quick.py to produce
             logs that can be visualised with show_run.py.

Author:      AgentDisruptBench Contributors
License:     MIT
Created:     2026-03-25
Modified:    2026-03-25

Key Classes:
    RunLogger : Writes timestamped JSONL events for one benchmark run.

Convention:
    Every source file MUST include a header block like this one.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class RunLogger:
    """Writes structured JSONL events for a single benchmark run.

    Events are written to ``<output_dir>/<run_id>/run_log.jsonl``
    and can be rendered with ``python evaluation/show_run.py``.

    Usage::

        log = RunLogger(output_dir="logs")
        log.emit("run_started", {"model": "gpt-5-mini", ...})
        log.emit("task_started", {"task_id": "retail_001", ...})
        log.emit("tool_call", {"tool_name": "search_products", ...})
        log.emit("task_completed", {"task_id": "retail_001", ...})
        log.emit("run_completed", {"total_tasks": 1, ...})
        log.close()
    """

    def __init__(self, output_dir: str = "logs") -> None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        short_id = uuid.uuid4().hex[:6]
        self.run_id = f"{ts}_{short_id}"
        self.run_dir = Path(output_dir) / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._log_path = self.run_dir / "run_log.jsonl"
        self._f = open(self._log_path, "w")

    def emit(self, event_type: str, payload: dict[str, Any]) -> None:
        """Append a structured event to the log.

        Args:
            event_type: One of: run_started, tasks_selected, task_started,
                        tool_call, rac_event, task_completed, run_completed.
            payload:    Event-specific data dict.
        """
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "payload": payload,
        }
        self._f.write(json.dumps(record, default=str) + "\n")
        self._f.flush()

    def close(self) -> None:
        """Close the log file."""
        self._f.close()
