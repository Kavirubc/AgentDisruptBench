"""
AgentDisruptBench — Tests for compare_runs
============================================

File:        test_compare_runs.py
Purpose:     Tests for the run comparison tool: JSONL parsing, summary
             extraction, discovery, and win/loss calculation.

Author:      AgentDisruptBench Contributors
License:     MIT
Created:     2026-03-25
Modified:    2026-03-25

Convention:
    Every source file MUST include a header block like this one.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Ensure project root is on path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from evaluation.compare_runs import (
    RunSummary,
    TaskResult,
    discover_runs,
    load_run_summary,
)

# ─── FIXTURES ─────────────────────────────────────────────────────────────────


def _write_run_log(run_dir: Path, events: list[dict]) -> None:
    """Write events to a run_log.jsonl file."""
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "run_log.jsonl", "w") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")


def _make_events(
    run_id: str = "test_run",
    model: str = "test-model",
    runner: str = "simple",
    profile: str = "clean",
    tasks: list[dict] | None = None,
) -> list[dict]:
    """Build a minimal set of JSONL events."""
    events = [
        {
            "timestamp": "2026-03-25T00:00:00+00:00",
            "event_type": "run_started",
            "payload": {
                "run_id": run_id,
                "runner": runner,
                "model": model,
                "profile": profile,
                "domain": "all",
                "seed": 42,
                "min_difficulty": 1,
                "max_difficulty": 5,
            },
        },
    ]

    task_list = tasks or [
        {
            "task_id": "task_001",
            "title": "Test Task",
            "difficulty": 1,
            "success": True,
            "partial_score": 0.8,
            "total_tool_calls": 3,
        },
        {
            "task_id": "task_002",
            "title": "Test Task 2",
            "difficulty": 2,
            "success": False,
            "partial_score": 0.4,
            "total_tool_calls": 5,
        },
    ]

    events.append(
        {
            "timestamp": "2026-03-25T00:00:00+00:00",
            "event_type": "tasks_selected",
            "payload": {
                "count": len(task_list),
                "tasks": [
                    {
                        "id": t["task_id"],
                        "title": t.get("title", ""),
                        "difficulty": t.get("difficulty", 1),
                        "tools": [],
                        "depth": 1,
                    }
                    for t in task_list
                ],
            },
        }
    )

    successful = 0
    for t in task_list:
        events.append(
            {
                "timestamp": "2026-03-25T00:00:01+00:00",
                "event_type": "task_started",
                "payload": {
                    "task_id": t["task_id"],
                    "title": t.get("title", ""),
                    "difficulty": t.get("difficulty", 1),
                    "task_type": "standard",
                    "required_tools": [],
                    "expected_depth": 1,
                    "profile": profile,
                },
            }
        )
        events.append(
            {
                "timestamp": "2026-03-25T00:00:02+00:00",
                "event_type": "task_completed",
                "payload": {
                    "task_id": t["task_id"],
                    "success": t.get("success", False),
                    "partial_score": t.get("partial_score", 0.0),
                    "recovery_rate": t.get("recovery_rate", 1.0),
                    "total_tool_calls": t.get("total_tool_calls", 0),
                    "disruptions_encountered": t.get("disruptions_encountered", 0),
                    "duration_seconds": t.get("duration_seconds", 1.0),
                    "recovery_strategies": [],
                    "dominant_strategy": "",
                    "tool_hallucination_rate": 0.0,
                },
            }
        )
        if t.get("success", False):
            successful += 1

    total = len(task_list)
    avg_score = sum(t.get("partial_score", 0.0) for t in task_list) / max(total, 1)
    events.append(
        {
            "timestamp": "2026-03-25T00:00:03+00:00",
            "event_type": "run_completed",
            "payload": {
                "total_tasks": total,
                "successful": successful,
                "success_rate": round(successful / max(total, 1), 4),
                "avg_partial_score": round(avg_score, 4),
                "total_duration_seconds": 2.0,
            },
        }
    )

    return events


# ─── TESTS: PARSING ──────────────────────────────────────────────────────────


class TestLoadRunSummary:
    """Tests for load_run_summary()."""

    def test_basic_parsing(self, tmp_path: Path) -> None:
        """Should parse run_started, task_completed, run_completed events."""
        events = _make_events(model="gpt-4o", runner="rac", profile="hostile_environment")
        run_dir = tmp_path / "test_run"
        _write_run_log(run_dir, events)

        summary = load_run_summary(run_dir)

        assert summary.model == "gpt-4o"
        assert summary.runner == "rac"
        assert summary.profile == "hostile_environment"
        assert summary.seed == 42
        assert summary.total_tasks == 2
        assert summary.successful == 1
        assert len(summary.tasks) == 2

    def test_task_scores(self, tmp_path: Path) -> None:
        """Should correctly parse per-task scores."""
        events = _make_events(
            model="test-model",
            tasks=[
                {"task_id": "t1", "success": True, "partial_score": 1.0, "total_tool_calls": 2},
                {"task_id": "t2", "success": False, "partial_score": 0.3, "total_tool_calls": 4},
            ],
        )
        _write_run_log(tmp_path / "run", events)
        summary = load_run_summary(tmp_path / "run")

        assert summary.tasks[0].task_id == "t1"
        assert summary.tasks[0].success is True
        assert summary.tasks[0].partial_score == 1.0
        assert summary.tasks[1].task_id == "t2"
        assert summary.tasks[1].success is False

    def test_missing_run_log(self, tmp_path: Path) -> None:
        """Should return empty summary if no run_log.jsonl exists."""
        run_dir = tmp_path / "empty_run"
        run_dir.mkdir()
        summary = load_run_summary(run_dir)
        assert summary.model == "?"
        assert summary.total_tasks == 0

    def test_backfills_task_metadata(self, tmp_path: Path) -> None:
        """Should backfill title and difficulty from task_started events."""
        events = _make_events(
            tasks=[
                {
                    "task_id": "retail_001",
                    "title": "Search products",
                    "difficulty": 3,
                    "success": True,
                    "partial_score": 0.9,
                },
            ],
        )
        _write_run_log(tmp_path / "run", events)
        summary = load_run_summary(tmp_path / "run")

        assert summary.tasks[0].title == "Search products"
        assert summary.tasks[0].difficulty == 3


# ─── TESTS: DISCOVERY ────────────────────────────────────────────────────────


class TestDiscoverRuns:
    """Tests for discover_runs()."""

    def test_by_run_ids(self, tmp_path: Path) -> None:
        """Should find runs by explicit IDs."""
        _write_run_log(tmp_path / "run_a", _make_events(run_id="run_a"))
        _write_run_log(tmp_path / "run_b", _make_events(run_id="run_b"))
        _write_run_log(tmp_path / "run_c", _make_events(run_id="run_c"))

        dirs = discover_runs(
            logs_dir=str(tmp_path),
            run_ids=["run_a", "run_c"],
        )
        assert len(dirs) == 2
        names = {d.name for d in dirs}
        assert names == {"run_a", "run_c"}

    def test_latest_n(self, tmp_path: Path) -> None:
        """Should return the N most recent runs."""
        for i in range(5):
            _write_run_log(
                tmp_path / f"run_{i:02d}",
                _make_events(run_id=f"run_{i:02d}"),
            )

        dirs = discover_runs(logs_dir=str(tmp_path), latest=2)
        assert len(dirs) == 2

    def test_filter_by_profile(self, tmp_path: Path) -> None:
        """Should filter runs matching a specific profile."""
        _write_run_log(
            tmp_path / "run_clean",
            _make_events(run_id="run_clean", profile="clean"),
        )
        _write_run_log(
            tmp_path / "run_hostile",
            _make_events(run_id="run_hostile", profile="hostile_environment"),
        )

        dirs = discover_runs(
            logs_dir=str(tmp_path),
            profile="hostile_environment",
        )
        assert len(dirs) == 1
        assert dirs[0].name == "run_hostile"

    def test_nonexistent_dir(self, tmp_path: Path) -> None:
        """Should return empty list for missing directory."""
        dirs = discover_runs(logs_dir=str(tmp_path / "nonexistent"))
        assert dirs == []


# ─── TESTS: DATA STRUCTURES ──────────────────────────────────────────────────


class TestRunSummary:
    """Tests for RunSummary and TaskResult dataclasses."""

    def test_default_values(self) -> None:
        """Should have sensible defaults."""
        s = RunSummary(run_id="test")
        assert s.model == "?"
        assert s.tasks == []
        assert s.total_tasks == 0

    def test_task_result_defaults(self) -> None:
        """TaskResult should have sensible defaults."""
        t = TaskResult(task_id="task_001")
        assert t.success is False
        assert t.partial_score == 0.0
        assert t.dominant_strategy == ""
