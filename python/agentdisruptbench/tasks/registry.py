"""
AgentDisruptBench — Task Registry
==================================

File:        registry.py
Purpose:     Loads, filters, and iterates task definitions from YAML files.
             Supports built-in task sets and custom YAML directories.

Author:      AgentDisruptBench Contributors
License:     MIT
Created:     2026-03-09
Modified:    2026-03-09

Key Classes:
    TaskRegistry : Load tasks from YAML, filter by domain/difficulty, iterate.

Convention:
    Every source file MUST include a header block like this one.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from agentdisruptbench.tasks.schemas import GroundTruth, Task

logger = logging.getLogger("agentdisruptbench.task_registry")

# Path to built-in YAML task files shipped with the package
_BUILTIN_DIR = Path(__file__).parent / "builtin"


class TaskRegistry:
    """Load, filter, and iterate benchmark task definitions.

    Usage::

        registry = TaskRegistry.from_builtin()
        retail_tasks = registry.filter(domain="retail", max_difficulty=3)
        for task in retail_tasks:
            ...

    Tasks can also be loaded from custom YAML directories::

        registry = TaskRegistry.from_directory("/path/to/tasks")
    """

    def __init__(self) -> None:
        self._tasks: dict[str, Task] = {}

    # -- loading -----------------------------------------------------------

    def add_task(self, task: Task) -> None:
        """Register a single task."""
        self._tasks[task.task_id] = task

    def load_yaml(self, path: str | Path) -> int:
        """Load tasks from a single YAML file.

        Returns:
            Number of tasks loaded from the file.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        path = Path(path)
        with open(path, "r", encoding="utf-8") as fh:
            raw: dict[str, Any] = yaml.safe_load(fh)

        if not raw or "tasks" not in raw:
            logger.warning("no_tasks_key path=%s", path)
            return 0

        domain = raw.get("domain", "unknown")
        count = 0
        for task_data in raw["tasks"]:
            # Inject domain if not set in task
            task_data.setdefault("domain", domain)
            # Parse ground_truth sub-model
            if "ground_truth" in task_data:
                task_data["ground_truth"] = GroundTruth(**task_data["ground_truth"])
            task = Task(**task_data)
            self._tasks[task.task_id] = task
            count += 1

        logger.info("tasks_loaded path=%s count=%d domain=%s", path, count, domain)
        return count

    def load_directory(self, directory: str | Path) -> int:
        """Load all ``*.yaml`` files in a directory.

        Returns:
            Total number of tasks loaded.
        """
        directory = Path(directory)
        total = 0
        for yaml_file in sorted(directory.glob("*.yaml")):
            total += self.load_yaml(yaml_file)
        return total

    # -- querying ----------------------------------------------------------

    def get(self, task_id: str) -> Task:
        """Get a task by ID.

        Raises:
            KeyError: If the task ID is not found.
        """
        if task_id not in self._tasks:
            raise KeyError(f"Task '{task_id}' not found.")
        return self._tasks[task_id]

    def all_tasks(self) -> list[Task]:
        """Return all registered tasks."""
        return list(self._tasks.values())

    def filter(
        self,
        domain: str | None = None,
        min_difficulty: int = 1,
        max_difficulty: int = 5,
        source: str | None = None,
        limit: int | None = None,
    ) -> list[Task]:
        """Filter tasks by domain, difficulty range, and source.

        Args:
            domain:         Filter to this domain (None = all).
            min_difficulty: Minimum difficulty (inclusive).
            max_difficulty: Maximum difficulty (inclusive).
            source:         Filter by source string (None = all).
            limit:          Max number of results (None = unlimited).

        Returns:
            List of matching Task objects.
        """
        results: list[Task] = []
        for t in self._tasks.values():
            if domain and t.domain != domain:
                continue
            if t.difficulty < min_difficulty or t.difficulty > max_difficulty:
                continue
            if source and t.source != source:
                continue
            results.append(t)
            if limit and len(results) >= limit:
                break
        return results

    def domains(self) -> list[str]:
        """List all unique domains in the registry."""
        return sorted({t.domain for t in self._tasks.values()})

    def __len__(self) -> int:
        return len(self._tasks)

    def __iter__(self):
        return iter(self._tasks.values())

    # -- factory methods ---------------------------------------------------

    @classmethod
    def from_builtin(cls) -> TaskRegistry:
        """Create a registry with all built-in task YAML files."""
        registry = cls()
        if _BUILTIN_DIR.exists():
            registry.load_directory(_BUILTIN_DIR)
        else:
            logger.warning("builtin_dir_missing path=%s", _BUILTIN_DIR)
        return registry

    @classmethod
    def from_directory(cls, directory: str | Path) -> TaskRegistry:
        """Create a registry from a custom directory of YAML files."""
        registry = cls()
        registry.load_directory(directory)
        return registry
