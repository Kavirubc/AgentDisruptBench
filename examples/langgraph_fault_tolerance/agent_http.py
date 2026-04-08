"""
LangGraph Fault-Tolerance Agent — HTTP Adapter for AgentDisruptBench
=====================================================================

File:        agent_http.py
Purpose:     Adapts the LangGraph fault-tolerance cookbook pattern for
             benchmarking under AgentDisruptBench's disruption proxy.

Source:      https://github.com/langchain-ai/cookbooks/tree/main/python/
             langgraph/persistence/fault-tolerance

What this preserves from the original cookbook:
    - StateGraph architecture with conditional routing
    - Retry logic with progressive fallback (retry → skip → giveup)
    - Checkpoint-based state persistence via MemorySaver
    - Parallel-safe state merging patterns

What this changes from the original:
    - REMOVED: random.random() hardcoded failure injection in nodes
    - REPLACED: Internal Python function calls → HTTP POST to ADB server
    - The ADB server injects real disruptions (timeouts, 500s, malformed
      responses) transparently — the agent sees failures identically to
      how it would see real API failures in production.

Usage:
    ADB_SERVER_URL=http://localhost:8080 python agent_http.py
"""


from __future__ import annotations

import os
import sys
import json
import time
from typing import Annotated, Optional

import httpx
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from typing_extensions import TypedDict

# ── Configuration ──────────────────────────────────────────────────────────

ADB_SERVER_URL = os.environ.get("ADB_SERVER_URL", "http://localhost:8080")
HTTP_TIMEOUT = 30.0


# ── HTTP Tool Client ──────────────────────────────────────────────────────

def call_adb_tool(tool_name: str, **kwargs) -> dict:
    """Call an AgentDisruptBench simulated tool via HTTP.

    The ADB server wraps every tool with ToolProxy + DisruptionEngine,
    so this call may fail with injected disruptions (timeouts, 500s,
    malformed responses, etc.) depending on the active profile.
    """
    url = f"{ADB_SERVER_URL}/api/tools/{tool_name}"
    with httpx.Client(timeout=HTTP_TIMEOUT) as client:
        resp = client.post(url, json=kwargs)
        resp.raise_for_status()
        return resp.json()


# ── Graph State ────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    service_name: str
    deploy_version: str
    health_status: Optional[dict]
    deploy_result: Optional[dict]
    test_result: Optional[dict]
    retry_count: int
    max_retries: int
    fallback_used: bool
    workflow_complete: bool
    workflow_outcome: str


# ── Graph Nodes ────────────────────────────────────────────────────────────

def check_health(state: AgentState) -> dict:
    """Step 1: Check service health via ADB server."""
    service = state["service_name"]
    retry_count = state.get("retry_count", 0)
    print(f"[HEALTH] Checking health of {service} (attempt {retry_count + 1})")

    try:
        result = call_adb_tool("get_service_health", service_name=service)
        print(f"[HEALTH] ✓ Status: {result.get('status', 'unknown')}")
        return {
            "health_status": result,
            "messages": [AIMessage(content=f"Health check: {result.get('status', 'unknown')}")],
        }
    except Exception as e:
        print(f"[HEALTH] ✗ Failed: {e}")
        return {
            "health_status": None,
            "messages": [AIMessage(content=f"Health check failed: {e}")],
        }


def deploy_service(state: AgentState) -> dict:
    """Step 2: Deploy the new version via ADB server."""
    service = state["service_name"]
    version = state["deploy_version"]
    print(f"[DEPLOY] Deploying {service} {version}")

    try:
        result = call_adb_tool(
            "deploy_service",
            service_name=service,
            version=version,
            environment="staging",
        )
        print(f"[DEPLOY] ✓ Deployment ID: {result.get('deployment_id', 'unknown')}")
        return {
            "deploy_result": result,
            "messages": [AIMessage(content=f"Deployed {version}: {result.get('status', 'unknown')}")],
        }
    except Exception as e:
        print(f"[DEPLOY] ✗ Failed: {e}")
        return {
            "deploy_result": None,
            "messages": [AIMessage(content=f"Deployment failed: {e}")],
        }


def run_tests(state: AgentState) -> dict:
    """Step 3: Run tests to verify the deployment via ADB server."""
    service = state["service_name"]
    print(f"[TESTS] Running integration tests for {service}")

    try:
        result = call_adb_tool(
            "run_tests",
            service_name=service,
            test_suite="integration",
        )
        passed = result.get("passed", 0)
        total = result.get("total_tests", 0)
        print(f"[TESTS] ✓ {passed}/{total} tests passed")
        return {
            "test_result": result,
            "messages": [AIMessage(content=f"Tests: {passed}/{total} passed")],
        }
    except Exception as e:
        print(f"[TESTS] ✗ Failed: {e}")
        return {
            "test_result": None,
            "messages": [AIMessage(content=f"Test run failed: {e}")],
        }


def handle_retry(state: AgentState) -> dict:
    """Retry logic with progressive fallback (mirrors LangGraph cookbook pattern)."""
    retry_count = state.get("retry_count", 0) + 1
    max_retries = state.get("max_retries", 3)
    print(f"[RETRY] Handling failure (attempt {retry_count}/{max_retries})")

    if retry_count <= max_retries:
        # Direct retry
        print(f"[RETRY] Strategy: direct retry")
        return {"retry_count": retry_count}
    elif retry_count <= max_retries + 2:
        # Fallback A: skip the failing step and proceed
        print(f"[RETRY] Strategy: fallback — skipping failed step")
        return {"retry_count": retry_count, "fallback_used": True}
    else:
        # Fallback B: give up gracefully
        print(f"[RETRY] Strategy: graceful giveup")
        return {
            "retry_count": retry_count,
            "fallback_used": True,
            "workflow_complete": True,
            "workflow_outcome": "FAILED_GRACEFULLY",
            "messages": [AIMessage(
                content="⚠️ Unable to complete the deployment workflow after multiple "
                        "retries. The service may need manual intervention."
            )],
        }


def finalize(state: AgentState) -> dict:
    """Generate final summary."""
    health = state.get("health_status")
    deploy = state.get("deploy_result")
    tests = state.get("test_result")
    fallback = state.get("fallback_used", False)

    parts = []
    if health:
        parts.append(f"Health: {health.get('status', 'unknown')}")
    if deploy:
        parts.append(f"Deploy: {deploy.get('status', 'unknown')}")
    if tests:
        parts.append(f"Tests: {tests.get('passed', '?')}/{tests.get('total_tests', '?')}")
    if fallback:
        parts.append("(completed with fallback strategy)")

    summary = " | ".join(parts) if parts else "Workflow completed with no data"
    outcome = "SUCCESS" if (deploy and tests) else "PARTIAL"

    print(f"[DONE] {summary}")
    return {
        "workflow_complete": True,
        "workflow_outcome": outcome,
        "messages": [AIMessage(content=f"✅ Workflow complete: {summary}")],
    }


# ── Routing ────────────────────────────────────────────────────────────────

def route_after_health(state: AgentState) -> str:
    """If health check succeeded → deploy. Otherwise → retry."""
    if state.get("health_status") is not None:
        return "deploy_service"
    return "handle_retry"


def route_after_deploy(state: AgentState) -> str:
    """If deployment succeeded → test. Otherwise → retry."""
    if state.get("deploy_result") is not None:
        return "run_tests"
    return "handle_retry"


def route_after_tests(state: AgentState) -> str:
    """If tests passed → finalize. Otherwise → retry."""
    if state.get("test_result") is not None:
        return "finalize"
    return "handle_retry"


def route_after_retry(state: AgentState) -> str:
    """After retry logic: if workflow is over, finalize. Otherwise re-enter from health."""
    if state.get("workflow_complete"):
        return "finalize"
    # Re-enter the workflow from the beginning (health check)
    return "check_health"


# ── Build Graph ────────────────────────────────────────────────────────────

builder = StateGraph(AgentState)

builder.add_node("check_health", check_health)
builder.add_node("deploy_service", deploy_service)
builder.add_node("run_tests", run_tests)
builder.add_node("handle_retry", handle_retry)
builder.add_node("finalize", finalize)

builder.add_edge(START, "check_health")
builder.add_conditional_edges("check_health", route_after_health, {
    "deploy_service": "deploy_service",
    "handle_retry": "handle_retry",
})
builder.add_conditional_edges("deploy_service", route_after_deploy, {
    "run_tests": "run_tests",
    "handle_retry": "handle_retry",
})
builder.add_conditional_edges("run_tests", route_after_tests, {
    "finalize": "finalize",
    "handle_retry": "handle_retry",
})
builder.add_conditional_edges("handle_retry", route_after_retry, {
    "check_health": "check_health",
    "finalize": "finalize",
})
builder.add_edge("finalize", END)

graph = builder.compile(checkpointer=MemorySaver())


# ── Entry Point ────────────────────────────────────────────────────────────

def main():
    """Run the agent. Called as a subprocess by the ADB external harness."""
    import uuid

    # Read task parameters from env or use defaults
    service = os.environ.get("ADB_SERVICE_NAME", "api-gateway")
    version = os.environ.get("ADB_DEPLOY_VERSION", "v2.2.0")

    thread_id = f"adb-{uuid.uuid4().hex[:8]}"
    config = {"configurable": {"thread_id": thread_id}}

    init_state = {
        "messages": [HumanMessage(
            content=f"Deploy {version} of {service}. Check health first, then deploy and verify with tests."
        )],
        "service_name": service,
        "deploy_version": version,
        "health_status": None,
        "deploy_result": None,
        "test_result": None,
        "retry_count": 0,
        "max_retries": 3,
        "fallback_used": False,
        "workflow_complete": False,
        "workflow_outcome": "",
    }

    print(f"\n{'='*60}")
    print(f"  LangGraph Fault-Tolerance Agent")
    print(f"  Server: {ADB_SERVER_URL}")
    print(f"  Task: Deploy {version} of {service}")
    print(f"{'='*60}\n")

    try:
        result = graph.invoke(init_state, config)
        outcome = result.get("workflow_outcome", "UNKNOWN")
        if result.get("messages"):
            final_msg = result["messages"][-1].content
        else:
            final_msg = "No output"

        print(f"\n{'─'*60}")
        print(f"  Outcome: {outcome}")
        print(f"  Output:  {final_msg}")
        print(f"{'─'*60}\n")

        # Write the output to stdout for the harness to capture
        sys.stdout.flush()

    except Exception as e:
        print(f"\n[FATAL] Agent crashed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
