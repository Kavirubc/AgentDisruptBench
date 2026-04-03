"""
AgentDisruptBench — LangChain / LangGraph Adapter
===================================================

File:        langchain.py
Purpose:     Wraps LangChain BaseTool instances so that _run and _arun
             delegate through ToolProxy for disruption injection.
             Works with both LangChain agents and LangGraph ToolNode.

Author:      AgentDisruptBench Contributors
License:     MIT
Created:     2026-03-09
Modified:    2026-03-09

Key Classes:
    LangChainAdapter       : BaseAdapter subclass for LangChain/LangGraph.
    DisruptedLangChainTool : BaseTool subclass that delegates through ToolProxy.

Design Notes:
    - The _proxy attribute is stored as a Pydantic v2 PrivateAttr to avoid
      serialisation issues.
    - LangGraph's ToolNode calls tool.invoke(input) → _run(). Same wrapping.
    - Framework imports are guarded — raises ImportError with install instructions.

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

logger = logging.getLogger("agentdisruptbench.adapters.langchain")


def _import_langchain():
    """Guard import of langchain_core."""
    try:
        from langchain_core.tools import BaseTool  # noqa: F401
        from pydantic import PrivateAttr  # noqa: F401

        return True
    except ImportError:
        raise ImportError("LangChain adapter requires langchain-core. Install with: pip install langchain-core")


class LangChainAdapter(BaseAdapter):
    """Adapter for LangChain and LangGraph tools.

    Wraps each ``BaseTool`` in a ``DisruptedLangChainTool`` that routes
    ``_run`` calls through :class:`ToolProxy`.

    Usage::

        adapter = LangChainAdapter(engine, trace_collector)
        wrapped = adapter.wrap_tools(tools)
        # Use wrapped tools with agent / ToolNode
        originals = adapter.unwrap_tools(wrapped)
    """

    def __init__(
        self,
        engine: DisruptionEngine,
        trace_collector: TraceCollector,
    ) -> None:
        super().__init__(engine, trace_collector)
        _import_langchain()

    def wrap_tools(self, tools: list) -> list:
        """Wrap LangChain BaseTool instances.

        Args:
            tools: List of BaseTool instances.

        Returns:
            List of DisruptedLangChainTool instances.
        """
        from langchain_core.tools import BaseTool
        from pydantic import PrivateAttr

        wrapped = []
        for tool in tools:
            if not isinstance(tool, BaseTool):
                logger.warning("skip_non_basetool type=%s", type(tool).__name__)
                wrapped.append(tool)
                continue

            proxy = ToolProxy(
                name=tool.name,
                fn=tool._run,
                engine=self._engine,
                trace_collector=self._trace_collector,
            )

            # Dynamically create a subclass
            original_tool = tool

            class DisruptedLangChainTool(BaseTool):
                """LangChain tool with disruption injection via ToolProxy."""

                name: str = original_tool.name
                description: str = original_tool.description
                _proxy: Any = PrivateAttr(default=None)
                _original_tool: Any = PrivateAttr(default=None)

                def _run(self, *args: Any, **kwargs: Any) -> Any:
                    return self._proxy(**kwargs)

                async def _arun(self, *args: Any, **kwargs: Any) -> Any:
                    return self._proxy(**kwargs)

            disrupted = DisruptedLangChainTool()
            disrupted._proxy = proxy
            disrupted._original_tool = original_tool
            wrapped.append(disrupted)
            logger.debug("tool_wrapped name=%s", tool.name)

        return wrapped

    def unwrap_tools(self, tools: list) -> list:
        """Restore original BaseTool instances.

        Args:
            tools: List of wrapped tools from ``wrap_tools``.

        Returns:
            List of original BaseTool instances.
        """
        originals = []
        for tool in tools:
            if hasattr(tool, "_original_tool") and tool._original_tool is not None:
                originals.append(tool._original_tool)
            else:
                originals.append(tool)
        return originals
