"""
AgentDisruptBench — Base Adapter
=================================

File:        base.py
Purpose:     Abstract base class for framework adapters. All adapters
             (LangChain, OpenAI, AutoGen, CrewAI) extend this class.

Author:      AgentDisruptBench Contributors
License:     MIT
Created:     2026-03-09
Modified:    2026-03-09

Key Classes:
    BaseAdapter : ABC defining the wrap_tools / unwrap_tools contract.

Convention:
    Every source file MUST include a header block like this one.
"""

from __future__ import annotations

import abc
from typing import Any

from agentdisruptbench.core.engine import DisruptionEngine
from agentdisruptbench.core.trace import TraceCollector


class BaseAdapter(abc.ABC):
    """Abstract base class for framework-specific tool adapters.

    Adapters wrap framework-native tools so that every invocation passes
    through the :class:`DisruptionEngine`.  They also provide an
    ``unwrap_tools`` method to restore originals.

    Parameters:
        engine:          Active disruption engine.
        trace_collector: Where to record call traces.
    """

    def __init__(
        self,
        engine: DisruptionEngine,
        trace_collector: TraceCollector,
    ) -> None:
        self._engine = engine
        self._trace_collector = trace_collector

    @abc.abstractmethod
    def wrap_tools(self, tools: Any) -> Any:
        """Wrap framework-native tools with disruption proxies.

        Args:
            tools: Framework-specific tool list or registry.

        Returns:
            The same type with disrupted wrappers applied.
        """
        ...

    @abc.abstractmethod
    def unwrap_tools(self, tools: Any) -> Any:
        """Restore original (un-disrupted) tools.

        Args:
            tools: The wrapped tools returned by ``wrap_tools``.

        Returns:
            The original tools.
        """
        ...
