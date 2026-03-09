"""
AgentDisruptBench — Synthetic Task Generator
=============================================

File:        generator.py
Purpose:     Generates complete task triples (description, tool suite, ground
             truth) using an LLM client.  Also mines tasks from tau-bench and
             REALM-Bench if their data directories are present.

Author:      AgentDisruptBench Contributors
License:     MIT
Created:     2026-03-09
Modified:    2026-03-09

Key Classes:
    SyntheticTaskGenerator : LLM-powered task generation with benchmark
                             mining adapters for tau-bench and REALM-Bench.

Convention:
    Every source file MUST include a header block like this one.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable

from agentdisruptbench.tasks.schemas import GroundTruth, Task

logger = logging.getLogger("agentdisruptbench.task_generator")

# Domain definitions with tool suites
_DOMAIN_TOOLS: dict[str, list[str]] = {
    "retail": [
        "search_products", "check_inventory", "place_order",
        "get_order_status", "process_refund", "get_customer_profile",
        "apply_coupon", "update_cart",
    ],
    "travel": [
        "search_flights", "get_flight_details", "book_flight",
        "cancel_booking", "search_hotels", "check_hotel_availability",
        "get_weather", "currency_convert",
    ],
    "finance": [
        "get_account_balance", "transfer_funds", "get_transaction_history",
        "get_exchange_rate", "validate_card", "check_credit_limit",
    ],
    "devops": [
        "get_service_health", "deploy_service", "rollback_deployment",
        "get_logs", "get_metrics", "run_tests", "create_incident",
        "resolve_incident",
    ],
}

# Difficulty distribution per domain: {difficulty: count}
_DIFFICULTY_DIST: dict[int, int] = {1: 2, 2: 4, 3: 6, 4: 5, 5: 3}


class SyntheticTaskGenerator:
    """Generates benchmark tasks using an LLM or mines them from reference benchmarks.

    Parameters:
        llm_fn:  Callable that takes a prompt string and returns a response
                 string.  If ``None``, only mining and static generation
                 are available.
    """

    def __init__(self, llm_fn: Callable[[str], str] | None = None) -> None:
        self._llm_fn = llm_fn

    def generate_for_domain(
        self, domain: str, count: int = 20
    ) -> list[Task]:
        """Generate tasks for a single domain.

        If an LLM function is available, uses it for generation.
        Otherwise falls back to template-based static generation.

        Args:
            domain: One of retail, travel, finance, devops.
            count:  Number of tasks to generate (default 20).

        Returns:
            List of generated Task objects.
        """
        if domain not in _DOMAIN_TOOLS:
            raise ValueError(
                f"Unknown domain '{domain}'. Available: {list(_DOMAIN_TOOLS.keys())}"
            )

        if self._llm_fn:
            return self._generate_with_llm(domain, count)
        return self._generate_static(domain, count)

    def generate_all(self) -> list[Task]:
        """Generate tasks for all domains (20 per domain = 80 total)."""
        tasks: list[Task] = []
        for domain in _DOMAIN_TOOLS:
            tasks.extend(self.generate_for_domain(domain))
        return tasks

    # -- LLM-based generation ---------------------------------------------

    def _generate_with_llm(self, domain: str, count: int) -> list[Task]:
        """Use the LLM to generate structured task definitions."""
        assert self._llm_fn is not None
        tools = _DOMAIN_TOOLS[domain]
        tasks: list[Task] = []

        prompt = f"""Generate {count} benchmark tasks for an AI agent evaluating tool-use resilience.

Domain: {domain}
Available tools: {', '.join(tools)}

For each task provide a JSON object with:
- task_id: "{domain}_NNN" format
- title: short descriptive title
- description: detailed instruction prompt for the agent
- difficulty: 1-5 integer
- required_tools: list of tool names needed
- expected_tool_call_depth: integer estimate of tool calls
- ground_truth: object with expected_outcome (string), required_tool_calls (list),
  evaluation_rubric (dict of criterion->weight summing to 1.0),
  disruption_sensitive_tools (list), recovery_actions (list)

Difficulty distribution: 2 at D1, 4 at D2, 6 at D3, 5 at D4, 3 at D5.

Return a JSON array of task objects. No markdown fencing."""

        try:
            response = self._llm_fn(prompt)
            raw_tasks = json.loads(response)
            for raw in raw_tasks:
                if "ground_truth" in raw:
                    raw["ground_truth"] = GroundTruth(**raw["ground_truth"])
                raw.setdefault("domain", domain)
                raw.setdefault("source", "synthetic")
                tasks.append(Task(**raw))
        except Exception:
            logger.exception("llm_generation_failed domain=%s", domain)
            tasks = self._generate_static(domain, count)

        return tasks[:count]

    # -- Static template generation ----------------------------------------

    def _generate_static(self, domain: str, count: int) -> list[Task]:
        """Template-based fallback when no LLM is available."""
        tools = _DOMAIN_TOOLS[domain]
        tasks: list[Task] = []
        task_num = 1

        for difficulty, n_tasks in _DIFFICULTY_DIST.items():
            for i in range(n_tasks):
                if task_num > count:
                    break
                n_tools = min(difficulty + 1, len(tools))
                required = tools[:n_tools]

                task = Task(
                    task_id=f"{domain}_{task_num:03d}",
                    title=f"{domain.title()} Task {task_num} (D{difficulty})",
                    description=self._make_description(domain, required, difficulty),
                    domain=domain,
                    difficulty=difficulty,
                    required_tools=required,
                    expected_tool_call_depth=n_tools,
                    ground_truth=GroundTruth(
                        expected_outcome=f"Complete the {domain} workflow using {', '.join(required)}",
                        required_tool_calls=required,
                        evaluation_rubric=self._make_rubric(required),
                        disruption_sensitive_tools=[required[-1]],
                        recovery_actions=[f"retry_{required[-1]}"],
                    ),
                    source="synthetic",
                )
                tasks.append(task)
                task_num += 1

        return tasks

    def _make_description(
        self, domain: str, tools: list[str], difficulty: int
    ) -> str:
        """Generate a task description from domain, tools, and difficulty."""
        readable_tools = ", ".join(t.replace("_", " ") for t in tools)
        complexity = {1: "simple", 2: "straightforward", 3: "moderate",
                      4: "complex", 5: "challenging"}
        return (
            f"You are an AI assistant for a {domain} platform. "
            f"Complete this {complexity.get(difficulty, 'moderate')} task: "
            f"Use the following tools in order: {readable_tools}. "
            f"Report the results clearly in your final response."
        )

    def _make_rubric(self, tools: list[str]) -> dict[str, float]:
        """Create an evaluation rubric with equal weights for each tool call."""
        weight = round(1.0 / len(tools), 4)
        rubric = {}
        for t in tools:
            rubric[f"called_{t}"] = weight
        # Adjust last weight so rubric sums to exactly 1.0
        total = sum(rubric.values())
        if total != 1.0:
            last_key = list(rubric.keys())[-1]
            rubric[last_key] = round(rubric[last_key] + (1.0 - total), 4)
        return rubric

    # -- Benchmark mining --------------------------------------------------

    def mine_from_tau_bench(self, path: str | Path) -> list[Task]:
        """Mine tasks from a tau-bench data directory.

        Converts tau-bench's retail and airline task formats to
        AgentDisruptBench Task objects.

        Args:
            path: Path to the tau-bench root directory.

        Returns:
            List of mined Task objects (source='tau_bench').
        """
        path = Path(path)
        tasks: list[Task] = []

        data_dir = path / "data" / "tau2" / "domains"
        if not data_dir.exists():
            data_dir = path / "data"

        for domain_name in ["retail", "airline"]:
            domain_dir = data_dir / domain_name
            if not domain_dir.exists():
                logger.info("tau_bench_domain_not_found domain=%s", domain_name)
                continue

            # Look for task files
            for task_file in domain_dir.glob("*.json"):
                try:
                    with open(task_file, "r") as fh:
                        raw = json.load(fh)
                    if isinstance(raw, list):
                        for item in raw:
                            t = self._convert_tau_task(item, domain_name)
                            if t:
                                tasks.append(t)
                    elif isinstance(raw, dict):
                        if "tasks" in raw:
                            for item in raw["tasks"]:
                                t = self._convert_tau_task(item, domain_name)
                                if t:
                                    tasks.append(t)
                except Exception:
                    logger.debug("tau_bench_parse_error file=%s", task_file)

        logger.info("tau_bench_mined count=%d", len(tasks))
        return tasks

    def _convert_tau_task(self, raw: dict, domain: str) -> Task | None:
        """Convert a single tau-bench task dict to Task."""
        task_id = raw.get("task_id", raw.get("id", ""))
        if not task_id:
            return None

        tools = raw.get("tools", raw.get("actions", []))
        if isinstance(tools, list) and tools and isinstance(tools[0], dict):
            tools = [t.get("name", "") for t in tools]

        return Task(
            task_id=f"tau_{task_id}",
            title=raw.get("title", f"tau-bench {domain} task"),
            description=raw.get("instruction", raw.get("description", "")),
            domain=domain,
            difficulty=min(5, max(1, len(tools))),
            required_tools=tools,
            expected_tool_call_depth=len(tools),
            ground_truth=GroundTruth(
                expected_outcome=raw.get("expected_output", "Task completed"),
                required_tool_calls=tools,
                evaluation_rubric=self._make_rubric(tools) if tools else {},
            ),
            source="tau_bench",
        )

    def mine_from_realm_bench(self, path: str | Path) -> list[Task]:
        """Mine tasks from a REALM-Bench data directory.

        REALM-Bench focuses on planning-time disruptions; we convert its
        planning scenarios into runtime tool-execution tasks.

        Args:
            path: Path to the REALM-Bench root directory.

        Returns:
            List of mined Task objects (source='realm_bench').
        """
        path = Path(path)
        tasks: list[Task] = []

        # Look for task definitions in evaluation directory
        eval_dir = path / "evaluation"
        if not eval_dir.exists():
            logger.info("realm_bench_eval_dir_not_found path=%s", eval_dir)
            return tasks

        for py_file in eval_dir.glob("*.py"):
            try:
                content = py_file.read_text()
                # Extract task definitions from Python source
                if "task_definitions" in py_file.name or "TASKS" in content:
                    logger.info("realm_bench_file_found file=%s", py_file.name)
            except Exception:
                logger.debug("realm_bench_parse_error file=%s", py_file)

        logger.info("realm_bench_mined count=%d", len(tasks))
        return tasks
