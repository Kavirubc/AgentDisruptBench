"""
AgentDisruptBench — ToolProxy
==============================

File:        proxy.py
Purpose:     Wraps any callable tool function so that every invocation is:
             1. Executed against the real tool.
             2. Passed through the DisruptionEngine.
             3. Recorded as a ToolCallTrace.
             4. Returned to the caller (possibly disrupted).

Author:      AgentDisruptBench Contributors
License:     MIT
Created:     2026-03-09
Modified:    2026-03-09

Key Classes:
    ToolProxy : Callable wrapper that transparently injects disruptions.

Convention:
    Every source file MUST include a header block like this one.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Callable

from agentdisruptbench.core.engine import DisruptionEngine, DisruptionType
from agentdisruptbench.core.trace import ToolCallTrace, TraceCollector

logger = logging.getLogger("agentdisruptbench.proxy")


class ToolProxy:
    """Wraps a callable. On each call:

    1. Executes the real tool and records real result + latency.
    2. Passes result through :class:`DisruptionEngine`.
    3. Records a :class:`ToolCallTrace` entry.
    4. Returns disrupted result or raises disrupted exception.

    For ``latency`` disruptions the proxy injects ``time.sleep``
    proportional to ``delay_ms`` from the matching config.

    Parameters:
        name:            Logical tool name (e.g. ``"search_products"``).
        fn:              The real tool callable.
        engine:          Active :class:`DisruptionEngine`.
        trace_collector: Where to record traces.
    """

    def __init__(
        self,
        name: str,
        fn: Callable,
        engine: DisruptionEngine,
        trace_collector: TraceCollector,
    ) -> None:
        self.name = name
        self._fn = fn
        self._engine = engine
        self._trace_collector = trace_collector
        self._call_count = 0

    def __call__(self, **kwargs: Any) -> Any:
        """Invoke the wrapped tool, applying disruptions."""
        self._call_count += 1
        call_id = str(uuid.uuid4())
        ts = time.time()

        # --- 1. Execute the real tool ---
        real_start = time.monotonic()
        real_result: Any = None
        real_success = True
        real_error: str | None = None
        try:
            real_result = self._fn(**kwargs)
        except Exception as exc:
            real_success = False
            real_error = str(exc)
        real_latency_ms = (time.monotonic() - real_start) * 1000

        # --- 2. Run through disruption engine ---
        disruption_type: DisruptionType | None = None
        observed_result = real_result
        observed_success = real_success
        observed_error = real_error

        try:
            observed_result, observed_success, observed_error, disruption_type = self._engine.apply(
                tool_name=self.name,
                tool_input=kwargs,
                original_result=real_result,
                original_success=real_success,
                original_error=real_error,
            )
        except TimeoutError as te:
            observed_result = None
            observed_success = False
            observed_error = str(te)
            disruption_type = DisruptionType.TIMEOUT

        # Latency injection
        if disruption_type == DisruptionType.LATENCY:
            for cfg in self._engine._configs:
                if cfg.type == DisruptionType.LATENCY:
                    time.sleep(cfg.delay_ms / 1000.0)
                    break

        observed_latency_ms = (time.monotonic() - real_start) * 1000

        # --- 3. Record trace ---
        trace = ToolCallTrace(
            call_id=call_id,
            tool_name=self.name,
            inputs=kwargs,
            real_result=real_result,
            observed_result=observed_result,
            real_success=real_success,
            observed_success=observed_success,
            disruption_fired=disruption_type.value if disruption_type else None,
            real_latency_ms=real_latency_ms,
            observed_latency_ms=observed_latency_ms,
            error=observed_error,
            timestamp=ts,
            call_number=self._call_count,
        )
        self._trace_collector.record(trace)

        # --- 4. Return or raise ---
        if not observed_success and disruption_type == DisruptionType.TIMEOUT:
            raise TimeoutError(observed_error)

        if not observed_success and observed_error:
            raise RuntimeError(observed_error)

        return observed_result
