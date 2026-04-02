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
from typing import List, Dict, Any, Type, Callable
from dotenv import load_dotenv
load_dotenv()

from agentdisruptbench.tasks.registry import TaskRegistry
from agentdisruptbench.core.metrics import MetricsCalculator, BenchmarkResult
from agentdisruptbench.harness.reporter import Reporter
from agentdisruptbench.core.trace import ToolCallTrace

logger = logging.getLogger("agentdisruptbench.orchestrator")


class SandboxOrchestrator:
    def __init__(self, server_url: str = "http://localhost:8080"):
        self.server_url = server_url.rstrip("/")
        self.registry = TaskRegistry.from_builtin()
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

    def run_evaluation(self, agent_client_func: Callable, profile: str, num_tasks: int = 10, seed: int = 42, domain: str | None = None, task_id: str | None = None) -> str:
        """Run the evaluation loop using the specified agent client function."""
        if not self.check_server():
            return ""

        self.setup_run(profile, seed)
        if domain or task_id:
            tasks = self.registry.filter(
                domain=domain, 
                task_ids=[task_id] if task_id else None
            )[:num_tasks]
        else:
            tasks = self.registry.all_tasks()[:num_tasks]
        
        results_list = []

        for i, task in enumerate(tasks):
            logger.info(f"Executing Task {i+1}/{num_tasks}: {task.task_id}")
            self.start_task(task.task_id)
            
            # Execute Agent Reference Client
            start_time = time.time()
            try:
                # The agent strictly takes the task.description to solve it
                agent_output = agent_client_func(task.description, self.server_url)
                if agent_output:
                    success_flag = True
                else:
                    success_flag = False
                error_msg = ""
            except Exception as e:
                success_flag = False
                agent_output = ""
                error_msg = str(e)
                logger.error(f"Agent Client Error on task {task.task_id}: {error_msg}")
            duration = time.time() - start_time
            
            # Fetch server-side metrics
            end_data = self.end_task()
            traces_dicts = end_data.get("traces", [])
            
            # Reconstruct ToolTrace objects for existing compute_metrics
            traces = [ToolCallTrace(**t) for t in traces_dicts]
            violations = end_data.get("idempotency_violations", [])
            
            # We skip actual diff analysis here and rely on the traces for compute_metrics 
            # to remain backwards compatible with the current Reporter, though true state-diff
            # pass/fail is possible.
            
            metrics_calc = MetricsCalculator()
            # Calculate base metrics - we stub state_diff for now since server tracks it internally but returns violations.
            # Real evaluation should pull the before/after snapshot from server to calculate the diff.
            metrics = metrics_calc.compute(
                task=task,
                traces=traces,
                agent_output=agent_output,
                baseline_result=None,
                agent_id=agent_client_func.__name__,
                profile_name=profile,
                seed=seed,
                duration_seconds=duration,
                idempotency_violations=len(violations)
            )
            
            if len(violations) > 0:
                metrics.success = False # strict penalty
                
            results_list.append(metrics)

        report_paths = self.reporter.generate(results_list)
        return report_paths.get("report.md", "")
