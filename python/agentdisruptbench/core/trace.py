"""
AgentDisruptBench — TraceCollector
===================================

File:        trace.py
Purpose:     Thread-safe collection of ToolCallTrace records for a single
             benchmark run. Supports JSONL serialisation for persistent
             storage and later analysis.

Author:      AgentDisruptBench Contributors
License:     MIT
Created:     2026-03-09
Modified:    2026-03-09

Key Classes:
    ToolCallTrace  : Dataclass capturing one proxied tool call.
    TraceCollector : Thread-safe container for ToolCallTrace records with
                     JSONL I/O and per-tool filtering.

Convention:
    Every source file MUST include a header block like this one.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import asdict, dataclass
from typing import Any

logger = logging.getLogger("agentdisruptbench.trace")


@dataclass
class ToolCallTrace:
    """A single proxied tool-call record.

    Attributes:
        call_id:             UUID string identifying this call.
        tool_name:           Logical name of the tool invoked.
        inputs:              Arguments passed to the tool.
        real_result:         The actual (un-disrupted) return value.
        observed_result:     The value the agent actually received.
        real_success:        Whether the real tool call succeeded.
        observed_success:    Whether the agent saw success.
        disruption_fired:    Name of the disruption type, or ``None``.
        real_latency_ms:     Wall-clock ms for the real tool execution.
        observed_latency_ms: Wall-clock ms including any injected delay.
        error:               Error message string if observed failure.
        timestamp:           Unix epoch timestamp of the call.
        call_number:         Sequential call count for this tool within a run.
    """

    call_id: str
    tool_name: str
    inputs: dict
    real_result: Any
    observed_result: Any
    real_success: bool
    observed_success: bool
    disruption_fired: str | None
    real_latency_ms: float
    observed_latency_ms: float
    error: str | None
    timestamp: float
    call_number: int


class TraceCollector:
    """Thread-safe collection of ToolCallTrace records.

    Used by :class:`ToolProxy` to accumulate traces during a benchmark run.
    Supports JSONL export/import and per-tool filtering.
    """

    def __init__(self) -> None:
        self._traces: list[ToolCallTrace] = []
        self._lock = threading.Lock()

    # -- recording ---------------------------------------------------------

    def record(self, trace: ToolCallTrace) -> None:
        """Append a trace entry (thread-safe)."""
        with self._lock:
            self._traces.append(trace)
            logger.debug(
                "trace_recorded tool=%s call_id=%s disruption=%s",
                trace.tool_name,
                trace.call_id,
                trace.disruption_fired,
            )

    # -- querying ----------------------------------------------------------

    def get_traces(self) -> list[ToolCallTrace]:
        """Return a shallow copy of all traces."""
        with self._lock:
            return list(self._traces)

    def get_traces_for_tool(self, tool_name: str) -> list[ToolCallTrace]:
        """Return traces for a specific tool name."""
        with self._lock:
            return [t for t in self._traces if t.tool_name == tool_name]

    def clear(self) -> None:
        """Remove all traces."""
        with self._lock:
            self._traces.clear()

    # -- JSONL I/O ---------------------------------------------------------

    def to_jsonl(self, path: str) -> None:
        """Write traces as JSONL (one JSON object per line).

        Raises:
            PermissionError: If the path is not writable.
            FileNotFoundError: If the parent directory does not exist.
        """
        traces = self.get_traces()
        with open(path, "w", encoding="utf-8") as fh:
            for t in traces:
                line = json.dumps(asdict(t), default=str)
                fh.write(line + "\n")
        logger.info("traces_written path=%s count=%d", path, len(traces))

    def from_jsonl(self, path: str) -> None:
        """Load traces from a JSONL file, appending to current collection.

        Raises:
            FileNotFoundError: If the path does not exist.
            PermissionError: If the path is not readable.
        """
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                trace = ToolCallTrace(**data)
                self.record(trace)
        logger.info("traces_loaded path=%s", path)
