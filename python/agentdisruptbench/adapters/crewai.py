"""
AgentDisruptBench — CrewAI Adapter
====================================

File:        crewai.py
Purpose:     Wraps CrewAI BaseTool instances. Disables caching on all
             wrapped tools to prevent disruption bypass.

Author:      AgentDisruptBench Contributors
License:     MIT
Created:     2026-03-09
Modified:    2026-03-09

Key Classes:
    CrewAIAdapter : Creates DisruptedCrewAITool wrappers.

Design Notes:
    CrewAI tools extend BaseTool with _run(). Tool caching is disabled
    (cache_function = None) on all wrapped tools to ensure disruptions
    are applied on every call and not bypassed by cached results.

Convention:
    Every source file MUST include a header block like this one.
"""

from __future__ import annotations

import logging
from typing import Any

from agentdisruptbench.adapters.base import BaseAdapter
from agentdisruptbench.core.engine import DisruptionEngine
from agentdisruptbench.core.proxy import ToolProxy
from agentdisruptbench.core.trace import TraceCollector

logger = logging.getLogger("agentdisruptbench.adapters.crewai")


def _import_crewai():
    """Guard import of crewai."""
    try:
        from crewai.tools import BaseTool  # noqa: F401
        return True
    except ImportError:
        raise ImportError(
            "CrewAI adapter requires crewai. "
            "Install with: pip install crewai"
        )


class CrewAIAdapter(BaseAdapter):
    """Adapter for CrewAI tools.

    Wraps each CrewAI BaseTool by subclassing and routing ``_run`` through
    :class:`ToolProxy`.  **Caching is disabled** (``cache_function = None``)
    on all wrapped tools to prevent disruption bypass.
    """

    def __init__(self, engine: DisruptionEngine, trace_collector: TraceCollector) -> None:
        super().__init__(engine, trace_collector)
        _import_crewai()

    def wrap_tools(self, tools: list) -> list:
        """Wrap CrewAI BaseTool instances.

        Args:
            tools: List of CrewAI BaseTool objects.

        Returns:
            List of DisruptedCrewAITool instances.
        """
        from crewai.tools import BaseTool

        wrapped = []
        for tool in tools:
            if not isinstance(tool, BaseTool):
                wrapped.append(tool)
                continue

            proxy = ToolProxy(
                name=tool.name, fn=tool._run,
                engine=self._engine, trace_collector=self._trace_collector,
            )
            original_tool = tool

            class DisruptedCrewAITool(BaseTool):
                """CrewAI tool with disruption injection.

                Caching is explicitly disabled (cache_function = None) to
                ensure disruptions fire on every invocation.
                """
                name: str = original_tool.name
                description: str = original_tool.description
                cache_function: Any = None  # Disable caching

                def _run(self, **kwargs: Any) -> Any:
                    return self._proxy(**kwargs)

            disrupted = DisruptedCrewAITool()
            disrupted._proxy = proxy
            disrupted._original_tool = original_tool
            wrapped.append(disrupted)
            logger.debug("crewai_wrapped name=%s cache=disabled", tool.name)

        return wrapped

    def unwrap_tools(self, tools: list) -> list:
        """Restore original CrewAI tools."""
        return [
            getattr(t, "_original_tool", t)
            for t in tools
        ]
