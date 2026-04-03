"""
AgentDisruptBench — AutoGen Adapter
=====================================

File:        autogen.py
Purpose:     Supports AutoGen 0.2 (function_map patching) and AutoGen 0.4
             (FunctionTool wrapping). Detects version at runtime.

Author:      AgentDisruptBench Contributors
License:     MIT
Created:     2026-03-09
Modified:    2026-03-09

Key Classes:
    AutoGenAdapter : Dual-version adapter for AutoGen 0.2 and 0.4.

Convention:
    Every source file MUST include a header block like this one.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from agentdisruptbench.adapters.base import BaseAdapter
from agentdisruptbench.core.engine import DisruptionEngine
from agentdisruptbench.core.proxy import ToolProxy
from agentdisruptbench.core.trace import TraceCollector

logger = logging.getLogger("agentdisruptbench.adapters.autogen")


def _get_autogen_version() -> str:
    """Detect installed AutoGen version."""
    try:
        import autogen

        return getattr(autogen, "__version__", "0.2.0")
    except ImportError:
        raise ImportError("AutoGen adapter requires autogen or pyautogen. Install with: pip install pyautogen")


class AutoGenAdapter(BaseAdapter):
    """Adapter for AutoGen 0.2 and 0.4.

    - **0.2**: Patches ``UserProxyAgent.function_map`` by replacing each
      callable with a ToolProxy-wrapped version.
    - **0.4**: Subclasses ``FunctionTool``, overriding ``run`` and ``run_json``.
    """

    def __init__(self, engine: DisruptionEngine, trace_collector: TraceCollector) -> None:
        super().__init__(engine, trace_collector)
        self._version = _get_autogen_version()
        self._originals: dict[str, Callable] = {}

    def wrap_tools(self, tools: Any) -> Any:
        """Wrap tools based on detected AutoGen version.

        For 0.2: ``tools`` is a dict (function_map).
        For 0.4: ``tools`` is a list of FunctionTool instances.
        """
        major = self._version.split(".")[0]
        if int(major) >= 1 or self._version.startswith("0.4"):
            return self._wrap_v04(tools)
        return self._wrap_v02(tools)

    def unwrap_tools(self, tools: Any) -> Any:
        """Restore original tools."""
        major = self._version.split(".")[0]
        if int(major) >= 1 or self._version.startswith("0.4"):
            return self._unwrap_v04(tools)
        return self._unwrap_v02(tools)

    # -- AutoGen 0.2 -------------------------------------------------------

    def _wrap_v02(self, function_map: dict[str, Callable]) -> dict[str, Callable]:
        """Patch function_map dict by wrapping each callable."""
        wrapped = {}
        for name, fn in function_map.items():
            self._originals[name] = fn
            proxy = ToolProxy(name=name, fn=fn, engine=self._engine, trace_collector=self._trace_collector)
            wrapped[name] = proxy
            logger.debug("autogen_v02_wrapped name=%s", name)
        return wrapped

    def _unwrap_v02(self, function_map: dict[str, Callable]) -> dict[str, Callable]:
        """Restore original function_map."""
        restored = {}
        for name, fn in function_map.items():
            restored[name] = self._originals.get(name, fn)
        return restored

    # -- AutoGen 0.4 -------------------------------------------------------

    def _wrap_v04(self, tools: list) -> list:
        """Wrap FunctionTool instances for AutoGen 0.4."""
        try:
            from autogen import FunctionTool
        except ImportError:
            logger.warning("autogen_v04_import_failed")
            return tools

        wrapped = []
        for tool in tools:
            if not isinstance(tool, FunctionTool):
                wrapped.append(tool)
                continue

            original_fn = tool._func
            self._originals[tool.name] = original_fn
            proxy = ToolProxy(
                name=tool.name,
                fn=original_fn,
                engine=self._engine,
                trace_collector=self._trace_collector,
            )

            class DisruptedFunctionTool(FunctionTool):
                """FunctionTool with disruption injection."""

                def __init__(self, proxy, original):
                    self._proxy = proxy
                    self._original = original
                    self.name = original.name
                    self.description = original.description

                def run(self, *args, **kwargs):
                    return self._proxy(**kwargs)

                def run_json(self, *args, **kwargs):
                    return self._proxy(**kwargs)

            disrupted = DisruptedFunctionTool(proxy, tool)
            wrapped.append(disrupted)
            logger.debug("autogen_v04_wrapped name=%s", tool.name)
        return wrapped

    def _unwrap_v04(self, tools: list) -> list:
        """Restore original FunctionTool instances."""
        restored = []
        for tool in tools:
            if hasattr(tool, "_original"):
                restored.append(tool._original)
            else:
                restored.append(tool)
        return restored
