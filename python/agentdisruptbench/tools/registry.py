"""
AgentDisruptBench — Tool Registry
==================================

File:        registry.py
Purpose:     Maps logical tool names to callable implementations.  Provides
             a central registry used by the BenchmarkRunner and adapters
             to resolve tool names to functions.

Author:      AgentDisruptBench Contributors
License:     MIT
Created:     2026-03-09
Modified:    2026-03-09

Key Classes:
    ToolRegistry : dict-like registry mapping name → callable.
                   Supports domain filtering and tool listing.

Convention:
    Every source file MUST include a header block like this one.
"""

from __future__ import annotations

import logging
from typing import Callable

from agentdisruptbench.tools.mock_tools import get_all_tools

logger = logging.getLogger("agentdisruptbench.tool_registry")


class ToolRegistry:
    """Registry mapping tool names to callable implementations.

    Usage::

        registry = ToolRegistry.from_mock_tools()
        fn = registry.get("search_products")
        result = fn(query="blue widget")

    Can also be populated manually for custom tools::

        registry = ToolRegistry()
        registry.register("my_tool", my_tool_function)
    """

    def __init__(self) -> None:
        self._tools: dict[str, Callable] = {}

    def register(self, name: str, fn: Callable) -> None:
        """Register a tool by name."""
        self._tools[name] = fn
        logger.debug("tool_registered name=%s", name)

    def get(self, name: str) -> Callable:
        """Get a tool callable by name.

        Raises:
            KeyError: If the tool is not registered.
        """
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' not found. Available: {list(self._tools.keys())}")
        return self._tools[name]

    def list_tools(self) -> list[str]:
        """List all registered tool names."""
        return list(self._tools.keys())

    def has(self, name: str) -> bool:
        """Check if a tool is registered."""
        return name in self._tools

    def as_dict(self) -> dict[str, Callable]:
        """Return a copy of the internal tool mapping."""
        return dict(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    @classmethod
    def from_mock_tools(cls) -> ToolRegistry:
        """Create a registry pre-populated with all built-in mock tools.

        Includes tools from: RetailTools, TravelTools, FinanceTools, DevopsTools.
        """
        registry = cls()
        for name, fn in get_all_tools().items():
            registry.register(name, fn)
        logger.info("mock_tool_registry_created tool_count=%d", len(registry))
        return registry
