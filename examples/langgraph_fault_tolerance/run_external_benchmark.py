#!/usr/bin/env python3
"""
AgentDisruptBench — External Agent Benchmark Runner
=====================================================

File:        run_external_benchmark.py
Purpose:     Generalizable harness for benchmarking ANY external agent against
             AgentDisruptBench. Starts the ADB REST server, configures a
             disruption profile, launches the agent as a subprocess, collects
             traces via admin endpoints, and produces scored BenchmarkResults.

Usage:
    # Run the LangGraph agent against the hostile_environment profile
    python run_external_benchmark.py --profile hostile_environment

    # Run against multiple profiles for comparison
    python run_external_benchmark.py --profiles clean mild_production hostile_environment

    # Use a custom agent command
    python run_external_benchmark.py --agent-cmd "python my_agent.py" --profile hostile_environment

Convention:
    Every source file MUST include a header block like this one.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import httpx

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agentdisruptbench.core.metrics import BenchmarkResult, MetricsCalculator
from agentdisruptbench.core.trace import ToolCallTrace
from agentdisruptbench.tasks.schemas import Task, GroundTruth


# ── Configuration ──────────────────────────────────────────────────────────

DEFAULT_PORT = 8080
DEFAULT_AGENT_CMD = f"python {Path(__file__).parent / 'agent_http.py'}"
HEALTH_CHECK_RETRIES = 30
HEALTH_CHECK_INTERVAL = 1.0
AGENT_TIMEOUT_SECONDS = 120


# ── Task Definition ───────────────────────────────────────────────────────
# The LangGraph agent performs a health-check → deploy → test workflow,
# which maps to existing ADB DevOps simulated tools.

DEVOPS_TASK = Task(
    task_id="external_devops_deploy",
    title="Health check, deploy, and verify",
    description=(
        "Check the health of api-gateway. If healthy, deploy v2.2.0 to "
        "staging. Run integration tests to verify the deployment."
    ),
    domain="devops",
    difficulty=3,
    task_type="standard",
    required_tools=["get_service_health", "deploy_service", "run_tests"],
    expected_tool_call_depth=3,
    ground_truth=GroundTruth(
        expected_outcome="Health-conditional deployment with test verification",
        required_tool_calls=["get_service_health", "deploy_service", "run_tests"],
        evaluation_rubric={
            "health_checked": 0.25,
            "deploy_succeeded": 0.35,
            "tests_verified": 0.25,
            "workflow_complete": 0.15,
        },
        disruption_sensitive_tools=["deploy_service", "run_tests"],
        recovery_actions=["retry_deploy", "skip_deploy", "retry_tests"],
    ),
)


# ── Server Management ─────────────────────────────────────────────────────

class ADBServerManager:
    """Start, configure, and stop the ADB REST server."""

    def __init__(self, port: int = DEFAULT_PORT):
        self.port = port
        self.base_url = f"http://localhost:{port}"
        self._process: subprocess.Popen | None = None
        self._stderr_file = None

    def start(self) -> None:
        """Start the ADB REST server as a subprocess."""
        import tempfile
        print(f"  Starting ADB server on port {self.port}...")

        # Use temp file for stderr to avoid pipe blocking
        self._stderr_file = tempfile.TemporaryFile(mode="w+")
        self._process = subprocess.Popen(
            [
                sys.executable, "-m", "uvicorn",
                "agentdisruptbench.server.app:app",
                "--host", "0.0.0.0",
                "--port", str(self.port),
                "--log-level", "warning",
            ],
            cwd=str(PROJECT_ROOT),
            env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT / "python")},
            stdout=subprocess.DEVNULL,
            stderr=self._stderr_file,
        )

        # Wait for server to be ready
        for i in range(HEALTH_CHECK_RETRIES):
            # Check if process died early
            if self._process.poll() is not None:
                self._stderr_file.seek(0)
                stderr = self._stderr_file.read()[:500]
                code = self._process.returncode
                self._process = None
                raise RuntimeError(
                    f"ADB server exited immediately (code={code})\n  stderr: {stderr}"
                )
            try:
                resp = httpx.get(f"{self.base_url}/openapi.json", timeout=2.0)
                if resp.status_code in (200, 404):
                    print(f"  ✓ ADB server ready at {self.base_url}")
                    return
            except (httpx.ConnectError, httpx.ReadTimeout):
                pass
            time.sleep(HEALTH_CHECK_INTERVAL)

        # Timeout — read server stderr for diagnostics
        self._stderr_file.seek(0)
        stderr = self._stderr_file.read()[:500]
        self.stop()
        raise RuntimeError(
            f"ADB server failed to start on port {self.port} after {HEALTH_CHECK_RETRIES}s\n"
            f"  stderr: {stderr[:500]}"
        )

    def setup_run(self, profile: str, seed: int) -> None:
        """Configure the disruption profile via admin API."""
        resp = httpx.post(
            f"{self.base_url}/admin/setup_run",
            json={"profile": profile, "seed": seed},
            timeout=5.0,
        )
        resp.raise_for_status()
        print(f"  ✓ Profile configured: {profile} (seed={seed})")

    def start_task(self, task_id: str) -> None:
        """Signal the start of a task evaluation."""
        resp = httpx.post(
            f"{self.base_url}/admin/start_task",
            params={"task_id": task_id},
            timeout=5.0,
        )
        resp.raise_for_status()

    def end_task(self) -> dict:
        """End the task and collect traces + violations."""
        resp = httpx.post(
            f"{self.base_url}/admin/end_task",
            timeout=10.0,
        )
        resp.raise_for_status()
        return resp.json()

    def stop(self) -> None:
        """Stop the ADB server."""
        if self._process:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
            self._process = None
            print("  ✓ ADB server stopped")


# ── Trace Deserialization ─────────────────────────────────────────────────

def deserialize_traces(raw_traces: list[dict]) -> list[ToolCallTrace]:
    """Convert the JSON trace dicts from the server into ToolCallTrace objects."""
    traces = []
    for t in raw_traces:
        traces.append(ToolCallTrace(
            call_id=t["call_id"],
            tool_name=t["tool_name"],
            inputs=t.get("inputs", {}),
            real_result=t.get("real_result"),
            observed_result=t.get("observed_result"),
            real_success=t.get("real_success", True),
            observed_success=t.get("observed_success", True),
            disruption_fired=t.get("disruption_fired"),
            real_latency_ms=t.get("real_latency_ms", 0.0),
            observed_latency_ms=t.get("observed_latency_ms", 0.0),
            error=t.get("error"),
            timestamp=t.get("timestamp", 0.0),
            call_number=t.get("call_number", 0),
        ))
    return traces


# ── Agent Execution ───────────────────────────────────────────────────────

def run_agent(agent_cmd: str, server_url: str, timeout: int = AGENT_TIMEOUT_SECONDS) -> str:
    """Launch the external agent as a subprocess.

    Returns the agent's stdout output (its final response).
    """
    env = {
        **os.environ,
        "ADB_SERVER_URL": server_url,
        "PYTHONPATH": str(PROJECT_ROOT / "python"),
    }

    print(f"\n  Launching agent: {agent_cmd}")
    result = subprocess.run(
        agent_cmd.split(),
        cwd=str(PROJECT_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )

    # Print agent output (for visibility)
    if result.stdout:
        for line in result.stdout.strip().split("\n"):
            print(f"    │ {line}")

    if result.stderr:
        for line in result.stderr.strip().split("\n")[-5:]:
            print(f"    │ [stderr] {line}")

    return result.stdout


# ── Single Profile Run ────────────────────────────────────────────────────

def run_single_profile(
    server: ADBServerManager,
    task: Task,
    profile: str,
    seed: int,
    agent_cmd: str,
    agent_id: str,
    baseline: BenchmarkResult | None = None,
) -> BenchmarkResult:
    """Run one (task × profile × seed) evaluation."""

    calculator = MetricsCalculator()

    # Configure server
    server.setup_run(profile, seed)
    server.start_task(task.task_id)

    # Run agent
    start = time.monotonic()
    agent_output = run_agent(agent_cmd, server.base_url)
    duration = time.monotonic() - start

    # Collect traces
    end_data = server.end_task()
    raw_traces = end_data.get("traces", [])
    violations = end_data.get("idempotency_violations", [])

    traces = deserialize_traces(raw_traces)

    # Compute metrics
    result = calculator.compute(
        task=task,
        traces=traces,
        agent_output=agent_output,
        baseline_result=baseline,
        agent_id=agent_id,
        profile_name=profile,
        seed=seed,
        duration_seconds=duration,
        idempotency_violations=len(violations),
    )

    return result


# ── Result Printing ───────────────────────────────────────────────────────

def print_result(r: BenchmarkResult) -> None:
    """Print a single BenchmarkResult in a readable format."""
    print(f"\n  ┌─────────────────────────────────────────────┐")
    print(f"  │ Task:     {r.task_id:<34}│")
    print(f"  │ Profile:  {r.profile_name:<34}│")
    print(f"  │ Seed:     {r.seed:<34}│")
    print(f"  ├─────────────────────────────────────────────┤")
    print(f"  │ Success:          {'✅ YES' if r.success else '❌ NO':<28}│")
    print(f"  │ Partial Score:    {r.partial_score:<28.4f}│")
    print(f"  │ Recovery Rate:    {r.recovery_rate:<28.4f}│")
    print(f"  │ Retry Efficiency: {r.retry_efficiency:<28.4f}│")
    print(f"  ├─────────────────────────────────────────────┤")
    print(f"  │ Tool Calls:       {r.total_tool_calls:<28}│")
    print(f"  │ Disruptions Hit:  {r.disruptions_encountered:<28}│")
    print(f"  │ Disruptions Rcvd: {r.disruptions_recovered:<28}│")
    print(f"  │ Duration (s):     {r.duration_seconds:<28.2f}│")
    if r.extra_tool_calls is not None:
        print(f"  │ Extra Calls:      {r.extra_tool_calls:<28}│")
    if r.resilience_ratio is not None:
        print(f"  │ Resilience Ratio: {r.resilience_ratio:<28.4f}│")
    if r.disruption_types_seen:
        types_str = ", ".join(r.disruption_types_seen[:3])
        print(f"  │ Disruption Types: {types_str:<28}│")
    if r.recovery_strategies:
        strat_str = ", ".join(r.recovery_strategies[:3])
        print(f"  │ Strategies:       {strat_str:<28}│")
    print(f"  │ Fallback Used:    {'Yes' if r.acknowledged_failure else 'No':<28}│")
    print(f"  └─────────────────────────────────────────────┘")


def print_comparison(results: list[BenchmarkResult]) -> None:
    """Print a comparison table across profiles."""
    print(f"\n{'='*65}")
    print(f"  DISRUPTION DEGRADATION CURVE")
    print(f"{'='*65}")
    print(f"  {'Profile':<25} {'Success':>8} {'PScore':>8} {'Recovery':>10} {'Calls':>6}")
    print(f"  {'─'*25} {'─'*8} {'─'*8} {'─'*10} {'─'*6}")
    for r in results:
        success_str = "✅" if r.success else "❌"
        print(f"  {r.profile_name:<25} {success_str:>8} {r.partial_score:>8.3f} {r.recovery_rate:>10.3f} {r.total_tool_calls:>6}")
    print(f"{'='*65}")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog="run_external_benchmark",
        description="AgentDisruptBench — External Agent Benchmark Runner",
    )
    parser.add_argument(
        "--profiles", "-p", nargs="+",
        default=["clean", "mild_production", "hostile_environment"],
        help="Disruption profiles to evaluate",
    )
    parser.add_argument("--seed", "-s", type=int, default=42, help="Random seed")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="ADB server port")
    parser.add_argument(
        "--agent-cmd", default=DEFAULT_AGENT_CMD,
        help="Command to launch the external agent",
    )
    parser.add_argument("--agent-id", default="langgraph_fault_tolerance", help="Agent identifier")
    parser.add_argument("--timeout", type=int, default=AGENT_TIMEOUT_SECONDS, help="Agent timeout (seconds)")
    args = parser.parse_args()

    print("=" * 65)
    print("  AgentDisruptBench — External Agent Benchmark")
    print("=" * 65)
    print(f"  Agent:    {args.agent_id}")
    print(f"  Profiles: {args.profiles}")
    print(f"  Seed:     {args.seed}")
    print(f"  Port:     {args.port}")

    server = ADBServerManager(port=args.port)

    try:
        # Start the ADB server
        print(f"\n[1/3] Starting ADB server...")
        server.start()

        # Run across all profiles
        print(f"\n[2/3] Running benchmark...")
        results: list[BenchmarkResult] = []
        baseline: BenchmarkResult | None = None

        # Ensure clean runs first
        profiles = args.profiles
        if "clean" in profiles:
            profiles = ["clean"] + [p for p in profiles if p != "clean"]

        for profile in profiles:
            print(f"\n{'─'*65}")
            print(f"  Profile: {profile}")
            print(f"{'─'*65}")

            result = run_single_profile(
                server=server,
                task=DEVOPS_TASK,
                profile=profile,
                seed=args.seed,
                agent_cmd=args.agent_cmd,
                agent_id=args.agent_id,
                baseline=baseline,
            )

            if profile == "clean":
                baseline = result

            print_result(result)
            results.append(result)

        # Comparison
        print(f"\n[3/3] Results...")
        if len(results) > 1:
            print_comparison(results)

    except KeyboardInterrupt:
        print("\n\n  Benchmark interrupted.")
    except Exception as e:
        print(f"\n  ❌ Benchmark failed: {e}")
        raise
    finally:
        server.stop()


if __name__ == "__main__":
    main()
