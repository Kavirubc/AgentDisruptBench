"""
AgentDisruptBench — FastAPI Server
==================================

File:        app.py
Purpose:     Provides a REST interface to all mock tools, injecting disruptions
             via middleware and managing state in SQLite. This enables
             language-agnostic agents to interact with the environment map.
"""

from __future__ import annotations

import inspect
import logging
from dataclasses import asdict
from typing import Any, Callable, Optional

from pydantic import BaseModel, Field, create_model

try:
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse
except ImportError:
    raise ImportError("Please install with `pip install agentdisruptbench[server]`") from None

from agentdisruptbench.core.engine import DisruptionEngine
from agentdisruptbench.core.profiles import get_profile
from agentdisruptbench.core.proxy import ToolProxy
from agentdisruptbench.core.state import StateManager
from agentdisruptbench.core.trace import TraceCollector
from agentdisruptbench.tools.registry import ToolRegistry
from agentdisruptbench.tools.stateful import wrap_tool_with_state

logger = logging.getLogger("agentdisruptbench.server.app")

app = FastAPI(
    title="AgentDisrupt-Bench API",
    description="Standardized REST Sandbox for Agent Disruption Benchmarking. Exposes mocked tools as HTTP endpoints and injects disruptions at runtime.",
    version="1.0.0",
)


# Global test state for the server
class ServerState:
    def __init__(self):
        self.reset()

    def reset(self):
        self.engine: Optional[DisruptionEngine] = None
        self.state_manager: Optional[StateManager] = StateManager()
        self.trace_collector: TraceCollector = TraceCollector()
        self.active_profile: str = "clean"
        self.seed: int = 42
        self.baseline_snapshot = None

    def setup_run(self, profile: str, seed: int):
        self.active_profile = profile
        self.seed = seed
        configs = get_profile(profile)
        self.engine = DisruptionEngine(configs=configs, seed=seed)
        self.state_manager.reset()  # Reset SQLite DB
        self.trace_collector = TraceCollector()


server_state = ServerState()
registry = ToolRegistry.from_mock_tools()

# --- Admin Endpoints ---


class SetupRunRequest(BaseModel):
    profile: str = "clean"
    seed: int = 42


@app.post("/admin/setup_run", tags=["Admin"])
async def setup_run(req: SetupRunRequest):
    """Initialize a new benchmark run, resetting the database and setting the disruption profile."""
    server_state.setup_run(profile=req.profile, seed=req.seed)
    return {"message": f"Run initialized with profile={req.profile}, seed={req.seed}"}


@app.post("/admin/start_task", tags=["Admin"])
async def start_task(task_id: str):
    """Mark the beginning of a task and create a baseline state snapshot."""
    # The traces are cleared out or marked for this task ID.
    server_state.trace_collector = TraceCollector()
    server_state.baseline_snapshot = server_state.state_manager.snapshot()
    return {"message": "Task started", "task_id": task_id}


@app.post("/admin/end_task", tags=["Admin"])
async def end_task():
    """End the task and retrieve all recorded traces and idempotency violations."""
    traces = server_state.trace_collector.get_traces()
    dict_traces = [asdict(t) for t in traces]
    violations = server_state.state_manager.get_idempotency_violations()
    # Snapshot at the end is up to the client evaluation engine via DB.
    return {"traces": dict_traces, "idempotency_violations": [{"tool": v[0], "entity_id": v[1]} for v in violations]}


# --- Dynamic Tool Endpoints ---


def _build_pydantic_schema_from_fn(name: str, fn: Callable) -> type:
    sig = inspect.signature(fn)
    fields = {}
    for param_name, param in sig.parameters.items():
        if param_name in ("self", "cls", "kwargs", "args"):
            continue
        ann = param.annotation if param.annotation != inspect.Parameter.empty else str
        if param.default != inspect.Parameter.empty:
            fields[param_name] = (ann, Field(default=param.default))
        else:
            fields[param_name] = (ann, ...)

    if not fields:
        # Create an empty model
        return create_model(f"{name.capitalize()}Request")
    return create_model(f"{name.capitalize()}Request", **fields)


def create_tool_endpoint(tool_name: str, fn: Callable):
    RequestModel = _build_pydantic_schema_from_fn(tool_name, fn)
    RequestModel.model_rebuild()

    # We use 'Any' in the actual definition to avoid Pydantic ForwardRef issues,
    # but manually apply the __annotations__ so FastAPI generates the OpenAPI schema.
    async def endpoint_wrapper(request_body: Any):
        if not server_state.engine:
            server_state.setup_run("clean", 42)

        kwargs = request_body.model_dump() if hasattr(request_body, "model_dump") else request_body

        # 1. State wrapper (records side effects to SQLite DB before proxying)
        state_wrapped_fn = wrap_tool_with_state(tool_name, fn, server_state.state_manager)

        # 2. Disruption Proxy (injects disruption)
        proxy = ToolProxy(
            name=tool_name,
            fn=state_wrapped_fn,
            engine=server_state.engine,
            trace_collector=server_state.trace_collector,
        )

        try:
            result = proxy(**kwargs)
            return result if isinstance(result, dict) else {"result": result}
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": str(e), "status": "failed"})

    endpoint_wrapper.__annotations__["request_body"] = RequestModel
    return endpoint_wrapper


# Register all tools as endpoints dynamically
for tool_name, tool_fn in registry.as_dict().items():
    endpoint_func = create_tool_endpoint(tool_name, tool_fn)

    # Use FastAPI router to add dynamically created endpoint
    app.add_api_route(
        path=f"/api/tools/{tool_name}",
        endpoint=endpoint_func,
        methods=["POST"],
        summary=f"Tool: {tool_name}",
        description=tool_fn.__doc__ or f"Mock endpoint for {tool_name}",
        tags=["Tools"],
    )
