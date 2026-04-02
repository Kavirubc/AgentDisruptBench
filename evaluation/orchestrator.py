"""
AgentDisruptBench — Sandbox Orchestrator
========================================

File:        orchestrator.py
Purpose:     Coordinates the benchmark execution by iterating over tasks,
             invoking the Sandbox Server API (to start/end tasks and manage state),
             and running the Reference Clients.
"""

import logging
import httpx
import time
from typing import List, Dict, Any, Type
from agentdisruptbench.core.registry import TaskRegistry
from agentdisruptbench.core.metrics import compute_metrics, MetricResult
from agentdisruptbench.harness.reporter import Reporter
from agentdisruptbench.core.trace import ToolTrace

logger = logging.getLogger("agentdisruptbench.orchestrator")


class SandboxOrchestrator:
    def __init__(self, server_url: str = "http://localhost:8080"):
        self.server_url = server_url.rstrip("/")
        self.registry = TaskRegistry()
        self.client = httpx.Client(timeout=30.0)
        self.reporter = Reporter()

    def check_server(self):
        try:
            resp = self.client.get(f"{self.server_url}/openapi.json")
            if resp.status_code == 200:
                logger.info(f"Connected to Sandbox Server at {self.server_url}")
                return True
        except httpx.RequestError:
            pass
        logger.error(f"Could not connect to server at {self.server_url}. Did you run 'adb serve'?")
        return False

    def setup_run(self, profile: str, seed: int):
        logger.info(f"Setting up sandbox run (profile={profile}, seed={seed})")
        resp = self.client.post(
            f"{self.server_url}/admin/setup_run",
            json={"profile": profile, "seed": seed}
        )
        resp.raise_for_status()

    def start_task(self, task_id: str):
        resp = self.client.post(
            f"{self.server_url}/admin/start_task", 
            params={"task_id": task_id}
        )
        resp.raise_for_status()

    def end_task(self) -> Dict[str, Any]:
        resp = self.client.post(f"{self.server_url}/admin/end_task")
        resp.raise_for_status()
        return resp.json()

    def run_evaluation(self, agent_client_func: Callable, profile: str, num_tasks: int = 10, seed: int = 42) -> str:
        """Run the evaluation loop using the specified agent client function."""
        if not self.check_server():
            return ""

        self.setup_run(profile, seed)
        tasks = self.registry.get_all_tasks()[:num_tasks]
        
        self.reporter.start_run(
            runner_name=agent_client_func.__name__,
            target_profile=profile
        )

        for i, task in enumerate(tasks):
            logger.info(f"Executing Task {i+1}/{num_tasks}: {task.task_id}")
            self.start_task(task.task_id)
            
            # Execute Agent Reference Client
            start_time = time.time()
            try:
                # The agent strictly takes the task.instruction to solve it
                # Optionally pass the server URL so the client knows where the OpenAPI spec is
                success = agent_client_func(task.instruction, self.server_url)
                error_msg = ""
            except Exception as e:
                success = False
                error_msg = str(e)
            duration = time.time() - start_time
            
            # Fetch server-side metrics
            end_data = self.end_task()
            traces_dicts = end_data.get("traces", [])
            
            # Reconstruct ToolTrace objects for existing compute_metrics
            traces = [ToolTrace(**t) for t in traces_dicts]
            violations = end_data.get("idempotency_violations", [])
            
            # We skip actual diff analysis here and rely on the traces for compute_metrics 
            # to remain backwards compatible with the current Reporter, though true state-diff
            # pass/fail is possible.
            
            metrics = compute_metrics(traces)
            # Factor in success from agent and violations from server
            if len(violations) > 0:
                metrics.total_idempotency_violations += len(violations)
                success = False # strict penalty

            self.reporter.add_result(
                task=task,
                is_success=success,
                error_msg=error_msg,
                duration_sec=duration,
                metrics=metrics,
                traces=traces
            )

        return self.reporter.finish_run()
