"""
AgentDisruptBench — MCP Server
===============================

File:        mcp_server.py
Purpose:     Provides a Model Context Protocol (MCP) server interface to the
             mock tools, allowing seamless integration with Claude Desktop, Cursor,
             and next-gen agentic frameworks.
"""

from __future__ import annotations

import logging
from typing import Callable

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    raise ImportError("Please install with `pip install agentdisruptbench[server]` (requires mcp package)") from None

from agentdisruptbench.core.engine import DisruptionEngine
from agentdisruptbench.core.profiles import get_profile
from agentdisruptbench.core.proxy import ToolProxy
from agentdisruptbench.core.state import StateManager
from agentdisruptbench.core.trace import TraceCollector
from agentdisruptbench.tools.registry import ToolRegistry
from agentdisruptbench.tools.stateful import wrap_tool_with_state

logger = logging.getLogger("agentdisruptbench.server.mcp")


class MCPBenchmarkServer:
    def __init__(self):
        self.mcp = FastMCP("AgentDisrupt-Bench MCP Server")
        self.engine = None
        self.state_manager = StateManager()
        self.trace_collector = TraceCollector()
        self.registry = ToolRegistry.from_mock_tools()

        self.setup_run("clean", 42)
        self._register_tools()

    def setup_run(self, profile: str, seed: int):
        """Initialize the benchmark disruptions."""
        configs = get_profile(profile)
        self.engine = DisruptionEngine(configs=configs, seed=seed)
        self.state_manager.reset()
        self.trace_collector = TraceCollector()
        logger.info(f"MCP Server initialized with profile={profile}, seed={seed}")

    def _register_tools(self):
        """Dynamically register all benchmark tools to the MCP server."""
        for tool_name, tool_fn in self.registry.as_dict().items():
            # Wrap the tool for state and disruptions
            def make_tool_wrapper(name: str, fn: Callable) -> Callable:
                # We use *args, **kwargs to accept all inputs cleanly
                def wrapper(*args, **kwargs):
                    if not self.engine:
                        self.setup_run("clean", 42)

                    state_wrapped_fn = wrap_tool_with_state(name, fn, self.state_manager)
                    proxy = ToolProxy(
                        name=name, fn=state_wrapped_fn, engine=self.engine, trace_collector=self.trace_collector
                    )

                    try:
                        return proxy(*args, **kwargs)
                    except Exception as e:
                        return f"Error executing tool: {str(e)}"

                # Preserve the original signature and docstring for MCP inspection
                import inspect

                wrapper.__signature__ = inspect.signature(fn)
                wrapper.__name__ = name
                wrapper.__doc__ = fn.__doc__
                return wrapper

            wrapped_tool = make_tool_wrapper(tool_name, tool_fn)
            # Add to FastMCP
            self.mcp.add_tool(wrapped_tool)

    def run_stdio(self):
        """Run the MCP server over standard I/O (often used by Claude Desktop)."""
        logger.info("Starting MCP Server over STDIO...")
        self.mcp.run(transport="stdio")


def main():
    import sys

    logging.basicConfig(level=logging.INFO, stream=sys.stderr)

    # Simple CLI argument matching for setup
    profile = "clean"
    seed = 42
    if "--profile" in sys.argv:
        profile = sys.argv[sys.argv.index("--profile") + 1]
    if "--seed" in sys.argv:
        seed = int(sys.argv[sys.argv.index("--seed") + 1])

    server = MCPBenchmarkServer()
    server.setup_run(profile, seed)
    server.run_stdio()


if __name__ == "__main__":
    main()
