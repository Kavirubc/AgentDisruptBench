"""
AgentDisruptBench — OpenAI Function Calling Adapter
====================================================

File:        openai.py
Purpose:     Intercepts OpenAI function calling at dispatch time. Does not
             modify tool schemas — applies disruptions when tool calls are
             executed and result messages are built.

Author:      AgentDisruptBench Contributors
License:     MIT
Created:     2026-03-09
Modified:    2026-03-09

Key Classes:
    OpenAIAdapter : Dispatch + tool-message builder for OpenAI function calling.

Convention:
    Every source file MUST include a header block like this one.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable

from agentdisruptbench.adapters.base import BaseAdapter
from agentdisruptbench.core.engine import DisruptionEngine
from agentdisruptbench.core.proxy import ToolProxy
from agentdisruptbench.core.trace import TraceCollector

logger = logging.getLogger("agentdisruptbench.adapters.openai")


class OpenAIAdapter(BaseAdapter):
    """Adapter for OpenAI function calling workflow.

    Does not modify tool schemas sent to the API.  Instead, intercepts
    at dispatch time when tool calls are executed locally.

    Usage::

        adapter = OpenAIAdapter(engine, trace_collector)
        messages = adapter.build_tool_messages(tool_calls, tool_registry)
    """

    def __init__(
        self,
        engine: DisruptionEngine,
        trace_collector: TraceCollector,
    ) -> None:
        super().__init__(engine, trace_collector)
        self._proxies: dict[str, ToolProxy] = {}

    def wrap_tools(self, tools: dict[str, Callable]) -> dict[str, ToolProxy]:
        """Wrap a tool registry dict with ToolProxy instances.

        Args:
            tools: Mapping of tool_name → callable.

        Returns:
            Mapping of tool_name → ToolProxy.
        """
        wrapped = {}
        for name, fn in tools.items():
            proxy = ToolProxy(
                name=name,
                fn=fn,
                engine=self._engine,
                trace_collector=self._trace_collector,
            )
            wrapped[name] = proxy
            self._proxies[name] = proxy
        return wrapped

    def unwrap_tools(self, tools: dict[str, Any]) -> dict[str, Any]:
        """Return original functions (not used in typical OpenAI workflow)."""
        return {
            name: (proxy._fn if isinstance(proxy, ToolProxy) else proxy)
            for name, proxy in tools.items()
        }

    def dispatch(
        self,
        tool_call: Any,
        tool_registry: dict[str, Callable],
    ) -> str:
        """Execute a single tool call with disruption injection.

        Args:
            tool_call: An OpenAI tool call object with ``function.name``
                       and ``function.arguments``.
            tool_registry: Dict of name → callable (or name → ToolProxy).

        Returns:
            String suitable for the tool role message content.
        """
        func_name = tool_call.function.name
        try:
            func_args = json.loads(tool_call.function.arguments)
        except (json.JSONDecodeError, AttributeError):
            func_args = {}

        fn = tool_registry.get(func_name)
        if fn is None:
            return json.dumps({"error": f"Unknown tool: {func_name}"})

        # If already a proxy, call directly; otherwise create ad-hoc proxy
        if isinstance(fn, ToolProxy):
            proxy = fn
        else:
            proxy = ToolProxy(
                name=func_name,
                fn=fn,
                engine=self._engine,
                trace_collector=self._trace_collector,
            )

        try:
            result = proxy(**func_args)
            return json.dumps(result, default=str)
        except Exception as exc:
            return json.dumps({"error": str(exc)})

    def build_tool_messages(
        self,
        tool_calls: list,
        tool_registry: dict[str, Callable],
    ) -> list[dict]:
        """Build tool result messages for the next chat completion call.

        Handles multiple parallel tool calls correctly.

        Args:
            tool_calls: List of tool call objects from the assistant message.
            tool_registry: Dict of name → callable.

        Returns:
            List of dicts with ``role="tool"`` messages.
        """
        messages = []
        for tc in tool_calls:
            content = self.dispatch(tc, tool_registry)
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": content,
            })
        return messages
