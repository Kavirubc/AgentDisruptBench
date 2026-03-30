#!/usr/bin/env python3
"""
AgentDisruptBench — Run Comparison Tool
=========================================

File:        compare_runs.py
Purpose:     Rich CLI tool to compare 2+ benchmark runs side-by-side.
             Renders metadata, aggregate scores, per-task breakdowns,
             and win/loss matrices for model/runner comparison.

Author:      AgentDisruptBench Contributors
License:     MIT
Created:     2026-03-25
Modified:    2026-03-25

Usage:
    python evaluation/compare_runs.py --run-ids <id1> <id2>
    python evaluation/compare_runs.py --latest 3
    python evaluation/compare_runs.py --profile hostile_environment

Convention:
    Every source file MUST include a header block like this one.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

try:
    import typer
    from rich import box
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
except ImportError:
    print("This script requires 'rich' and 'typer'.")
    print("Install with: pip install rich typer")
    raise SystemExit(1)

app = typer.Typer(
    help="Compare 2+ AgentDisruptBench runs side-by-side."
)
console = Console()


# ─── DATA STRUCTURES ─────────────────────────────────────────────────────────


@dataclass
class TaskResult:
    """Metrics for a single task within a run."""

    task_id: str
    title: str = ""
    difficulty: int = 0
    success: bool = False
    partial_score: float = 0.0
    recovery_rate: float = 0.0
    total_tool_calls: int = 0
    disruptions_encountered: int = 0
    duration_seconds: float = 0.0
    dominant_strategy: str = ""
    tool_hallucination_rate: float = 0.0


@dataclass
class RunSummary:
    """Normalized summary of a benchmark run parsed from JSONL."""

    run_id: str
    model: str = "?"
    runner: str = "?"
    profile: str = "?"
    domain: str = "?"
    seed: int = 0
    total_tasks: int = 0
    successful: int = 0
    success_rate: float = 0.0
    avg_partial_score: float = 0.0
    total_duration: float = 0.0
    tasks: list[TaskResult] = field(default_factory=list)


# ─── PARSING ──────────────────────────────────────────────────────────────────


def load_run_summary(run_dir: Path) -> RunSummary:
    """Parse a run_log.jsonl into a RunSummary."""
    jsonl_path = run_dir / "run_log.jsonl"
    if not jsonl_path.exists():
        return RunSummary(run_id=run_dir.name)

    events: list[dict[str, Any]] = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                pass

    summary = RunSummary(run_id=run_dir.name)

    for e in events:
        et = e["event_type"]
        p = e["payload"]

        if et == "run_started":
            summary.model = p.get("model", "?")
            summary.runner = p.get("runner", "?")
            summary.profile = p.get("profile", "?")
            summary.domain = p.get("domain", "?")
            summary.seed = p.get("seed", 0)

        elif et == "run_completed":
            summary.total_tasks = p.get("total_tasks", 0)
            summary.successful = p.get("successful", 0)
            summary.success_rate = p.get("success_rate", 0.0)
            summary.avg_partial_score = p.get(
                "avg_partial_score", 0.0
            )
            summary.total_duration = p.get(
                "total_duration_seconds", 0.0
            )

        elif et == "task_completed":
            summary.tasks.append(TaskResult(
                task_id=p.get("task_id", "?"),
                success=p.get("success", False),
                partial_score=p.get("partial_score", 0.0),
                recovery_rate=p.get("recovery_rate", 0.0),
                total_tool_calls=p.get("total_tool_calls", 0),
                disruptions_encountered=p.get(
                    "disruptions_encountered", 0
                ),
                duration_seconds=p.get("duration_seconds", 0.0),
                dominant_strategy=p.get("dominant_strategy", ""),
                tool_hallucination_rate=p.get(
                    "tool_hallucination_rate", 0.0
                ),
            ))

        elif et == "task_started":
            # Enrich the last task with title/difficulty when
            # task_completed arrives next — store pending metadata.
            # We'll match by task_id when rendering.
            pass

    # Backfill title/difficulty from task_started events
    task_meta: dict[str, dict] = {}
    for e in events:
        if e["event_type"] == "task_started":
            p = e["payload"]
            task_meta[p.get("task_id", "")] = p

    for tr in summary.tasks:
        meta = task_meta.get(tr.task_id, {})
        tr.title = meta.get("title", tr.task_id)
        tr.difficulty = meta.get("difficulty", 0)

    return summary


def discover_runs(
    logs_dir: str = "logs",
    run_ids: list[str] | None = None,
    latest: int | None = None,
    profile: str | None = None,
) -> list[Path]:
    """Find run directories matching filters."""
    logs_root = Path(logs_dir)
    if not logs_root.exists():
        return []

    if run_ids:
        dirs = []
        for rid in run_ids:
            d = logs_root / rid
            if d.exists() and (d / "run_log.jsonl").exists():
                dirs.append(d)
        return dirs

    # Discover all run dirs
    candidates = sorted(
        logs_root.glob("*/run_log.jsonl"),
        key=lambda p: p.stat().st_mtime,
    )
    dirs = [c.parent for c in candidates]

    if profile:
        filtered = []
        for d in dirs:
            s = load_run_summary(d)
            if s.profile == profile:
                filtered.append(d)
        dirs = filtered

    if latest:
        dirs = dirs[-latest:]

    return dirs


# ─── RENDERERS ────────────────────────────────────────────────────────────────


# ─── HELPERS ───────────────────────────────────────────────────────────────────


def get_run_label(summary: RunSummary, summaries: list[RunSummary]) -> str:
    """Generate a unique, descriptive label for a run in a set."""
    # Count how many runs share this model
    same_model = [s for s in summaries if s.model == summary.model]
    if len(same_model) == 1:
        return summary.model

    # Multiple runs of same model - check profiles
    same_profile = [s for s in same_model if s.profile == summary.profile]
    if len(same_profile) == 1:
        return f"{summary.model}\n({summary.profile})"

    # Multiple runs of same model/profile - add run ID suffix
    return f"{summary.model}\n({summary.profile} / {summary.run_id[-6:]})"


def _score_color(score: float) -> str:
    if score >= 0.8:
        return "bold green"
    if score >= 0.5:
        return "bold yellow"
    if score > 0:
        return "bold red"
    return "dim red"


def _delta_str(val: float, fmt: str = ".2f", higher_is_better: bool = True) -> str:
    """Format a delta value with color."""
    if abs(val) < 0.001:
        return "[dim]—[/dim]"
    sign = "+" if val > 0 else ""
    color = "green" if (val > 0) == higher_is_better else "red"
    return f"[{color}]{sign}{val:{fmt}}[/{color}]"


def render_metadata(summaries: list[RunSummary]) -> None:
    """Render run metadata side-by-side."""
    table = Table(
        title="Run Metadata",
        box=box.ROUNDED,
        show_header=True,
        padding=(0, 1),
    )
    table.add_column("", style="bold dim", width=12)
    for s in summaries:
        label = get_run_label(s, summaries)
        table.add_column(label, min_width=18, justify="center")

    rows = [
        ("Run ID", [s.run_id[-12:] for s in summaries]),
        ("Model", [f"[bold cyan]{s.model}[/bold cyan]" for s in summaries]),
        ("Runner", [s.runner for s in summaries]),
        ("Profile", [f"[magenta]{s.profile}[/magenta]" for s in summaries]),
        ("Domain", [s.domain for s in summaries]),
        ("Seed", [str(s.seed) for s in summaries]),
    ]

    for label, values in rows:
        table.add_row(label, *values)

    console.print(table)


def render_aggregate(summaries: list[RunSummary]) -> None:
    """Render aggregate scores comparison."""
    table = Table(
        title="Aggregate Scores",
        box=box.ROUNDED,
        show_header=True,
        padding=(0, 1),
    )
    table.add_column("Metric", style="bold", width=18)
    for s in summaries:
        label = get_run_label(s, summaries).replace("\n", " ")
        table.add_column(label, min_width=14, justify="center")

    # Find best values for highlighting
    success_rates = [s.success_rate for s in summaries]
    avg_scores = [s.avg_partial_score for s in summaries]
    durations = [s.total_duration for s in summaries]

    best_sr = max(success_rates) if success_rates else 0.0
    best_as = max(avg_scores) if avg_scores else 0.0
    best_dur = min(d for d in durations if d > 0) if any(d > 0 for d in durations) else 0.0

    def _highlight(val: float, best: float, fmt: str, higher: bool = True) -> str:
        sc = _score_color(val) if higher else "white"
        is_best = abs(val - best) < 0.001 and len(summaries) > 1
        marker = " 🏆" if is_best else ""
        return f"[{sc}]{val:{fmt}}{marker}[/{sc}]"

    table.add_row(
        "Tasks",
        *[str(s.total_tasks) for s in summaries],
    )
    table.add_row(
        "Successful",
        *[f"{s.successful}/{s.total_tasks}" for s in summaries],
    )
    table.add_row(
        "Success Rate",
        *[_highlight(s.success_rate, best_sr, ".0%") for s in summaries],
    )
    table.add_row(
        "Avg Score",
        *[_highlight(s.avg_partial_score, best_as, ".2f") for s in summaries],
    )

    # Recovery rate (avg across tasks)
    for s in summaries:
        if s.tasks:
            s._avg_recovery = sum(t.recovery_rate for t in s.tasks) / len(s.tasks)
        else:
            s._avg_recovery = 0.0

    recovery_rates = [s._avg_recovery for s in summaries]
    best_rr = max(recovery_rates) if recovery_rates else 0.0
    table.add_row(
        "Avg Recovery",
        *[_highlight(s._avg_recovery, best_rr, ".0%") for s in summaries],
    )

    table.add_row(
        "Duration",
        *[
            _highlight(s.total_duration, best_dur, ".1f", higher=False)
            if s.total_duration > 0 else "[dim]—[/dim]"
            for s in summaries
        ],
    )

    # Resilience Index (Hostile / Clean ratio)
    # Only calculate if we have exactly 2 runs of the same model where one is 'clean' and one is 'hostile_environment'
    if len(summaries) == 2 and summaries[0].model == summaries[1].model:
        r1, r2 = summaries[0], summaries[1]
        resilience = None
        if r1.profile == "clean" and r2.profile == "hostile_environment" and r1.success_rate > 0:
            resilience = r2.success_rate / r1.success_rate
        elif r2.profile == "clean" and r1.profile == "hostile_environment" and r2.success_rate > 0:
            resilience = r1.success_rate / r2.success_rate

        if resilience is not None:
            r_color = "green" if resilience >= 0.9 else "yellow" if resilience >= 0.7 else "red"
            vals = [f"[{r_color}]{resilience:.1%}[/{r_color}]" for _ in summaries]
            table.add_row("Resilience Index", *vals)

    console.print(table)


def render_per_task(summaries: list[RunSummary]) -> None:
    """Render per-task comparison."""
    # Collect all task IDs across all runs
    all_task_ids: list[str] = []
    seen: set[str] = set()
    for s in summaries:
        for t in s.tasks:
            if t.task_id not in seen:
                all_task_ids.append(t.task_id)
                seen.add(t.task_id)

    if not all_task_ids:
        return

    table = Table(
        title="Per-Task Breakdown",
        box=box.SIMPLE_HEAD,
        show_header=True,
        padding=(0, 1),
    )
    table.add_column("Task", style="cyan", no_wrap=True, width=16)
    table.add_column("D", justify="center", width=2)

    for s in summaries:
        label = get_run_label(s, summaries).replace("\n", " ")
        table.add_column(
            f"{label}\nScore",
            justify="center", width=12,
        )
        table.add_column(
            f"{label}\nTools",
            justify="center", width=6,
        )

    for tid in all_task_ids:
        row: list[str] = [tid]
        difficulty = "?"

        for s in summaries:
            task = next((t for t in s.tasks if t.task_id == tid), None)
            if task and difficulty == "?":
                difficulty = str(task.difficulty)

            if task:
                icon = "✅" if task.success else "❌"
                sc = _score_color(task.partial_score)
                row.append(
                    f"{icon} [{sc}]{task.partial_score:.2f}[/{sc}]"
                )
                row.append(str(task.total_tool_calls))
            else:
                row.append("[dim]—[/dim]")
                row.append("[dim]—[/dim]")

        row.insert(1, f"D{difficulty}")
        table.add_row(*row)

    console.print(table)


def render_win_loss(summaries: list[RunSummary]) -> None:
    """Render win/loss matrix for pairwise comparison."""
    if len(summaries) < 2:
        return

    # Collect all task IDs
    all_task_ids: set[str] = set()
    for s in summaries:
        for t in s.tasks:
            all_task_ids.add(t.task_id)

    # Count wins for each run
    run_labels = [get_run_label(s, summaries).replace("\n", " ") for s in summaries]
    wins: dict[int, int] = {i: 0 for i in range(len(summaries))}
    ties = 0
    total = 0

    for tid in all_task_ids:
        scores: list[float] = []
        for s in summaries:
            task = next(
                (t for t in s.tasks if t.task_id == tid), None
            )
            scores.append(task.partial_score if task else 0.0)

        if len(scores) < 2:
            continue

        total += 1
        best_score = max(scores)
        winners = [i for i, sc in enumerate(scores) if abs(sc - best_score) < 0.001]

        if len(winners) == len(scores):
            ties += 1
        else:
            for w in winners:
                wins[w] += 1

    # Render
    table = Table(
        title="Win/Loss Summary",
        box=box.ROUNDED,
        padding=(0, 2),
    )
    table.add_column("Run", style="bold cyan")
    table.add_column("Wins", justify="center", style="bold green")
    table.add_column("Win %", justify="center")

    for i, s in enumerate(summaries):
        w = wins.get(i, 0)
        pct = w / max(total, 1) * 100
        table.add_row(
            run_labels[i],
            str(w),
            f"{pct:.0f}%",
        )

    table.add_row(
        "[dim]Ties[/dim]",
        f"[dim]{ties}[/dim]",
        f"[dim]{ties / max(total, 1) * 100:.0f}%[/dim]",
    )

    console.print(table)


def render_cost_efficiency(summaries: list[RunSummary]) -> None:
    """Render cost efficiency comparison."""
    table = Table(
        title="Cost Efficiency",
        box=box.ROUNDED,
        padding=(0, 1),
    )
    table.add_column("Metric", style="bold", width=22)
    for s in summaries:
        label = get_run_label(s, summaries).replace("\n", " ")
        table.add_column(label, min_width=14, justify="center")

    # Total tool calls
    total_calls = [
        sum(t.total_tool_calls for t in s.tasks) for s in summaries
    ]
    table.add_row(
        "Total Tool Calls",
        *[str(c) for c in total_calls],
    )

    # Avg tool calls per task
    table.add_row(
        "Avg Calls/Task",
        *[
            f"{c / max(len(s.tasks), 1):.1f}"
            for c, s in zip(total_calls, summaries)
        ],
    )

    # Calls per successful task
    table.add_row(
        "Calls/Success",
        *[
            f"{c / max(s.successful, 1):.1f}"
            if s.successful > 0 else "[dim]—[/dim]"
            for c, s in zip(total_calls, summaries)
        ],
    )

    # Avg duration per task
    table.add_row(
        "Avg Time/Task",
        *[
            f"{s.total_duration / max(len(s.tasks), 1):.1f}s"
            for s in summaries
        ],
    )

    # Total disruptions
    total_disruptions = [
        sum(t.disruptions_encountered for t in s.tasks)
        for s in summaries
    ]
    table.add_row(
        "Total Disruptions",
        *[str(d) for d in total_disruptions],
    )

    console.print(table)


# ─── MAIN RENDERER ────────────────────────────────────────────────────────────


def render_comparison(summaries: list[RunSummary]) -> None:
    """Render the full comparison."""
    n = len(summaries)
    models = ", ".join(s.model for s in summaries)
    console.rule(
        f"[bold cyan]Comparing {n} Runs: {models}[/bold cyan]",
        style="cyan",
    )
    console.print()

    render_metadata(summaries)
    console.print()
    render_aggregate(summaries)
    console.print()
    render_per_task(summaries)
    console.print()
    render_win_loss(summaries)
    console.print()
    render_cost_efficiency(summaries)
    console.print()
    console.rule(style="dim")


# ─── CLI ──────────────────────────────────────────────────────────────────────


@app.command()
def main(
    run_ids: Optional[list[str]] = typer.Argument(
        None,
        help="Run IDs to compare (space-separated)",
    ),
    latest: Optional[int] = typer.Option(
        None, "--latest", "-n",
        help="Compare the N most recent runs",
    ),
    profile: Optional[str] = typer.Option(
        None, "--profile", "-p",
        help="Filter runs by disruption profile",
    ),
    logs_dir: str = typer.Option(
        "logs", "--logs-dir", "-d",
        help="Root directory containing run folders",
    ),
) -> None:
    """Compare 2+ AgentDisruptBench runs side-by-side.

    Pass run IDs as positional arguments, or use --latest / --profile
    to auto-discover runs.

    Examples:

        compare_runs.py <id1> <id2>

        compare_runs.py --latest 3

        compare_runs.py --profile hostile_environment
    """
    dirs = discover_runs(
        logs_dir=logs_dir,
        run_ids=run_ids or None,
        latest=latest,
        profile=profile,
    )

    if len(dirs) < 2:
        console.print(
            "[red]Need at least 2 runs to compare. "
            f"Found {len(dirs)}.[/red]"
        )
        console.print(
            "[dim]Pass run IDs as arguments, or use "
            "--latest / --profile to select runs.[/dim]"
        )
        raise typer.Exit(1)

    summaries = [load_run_summary(d) for d in dirs]
    render_comparison(summaries)


if __name__ == "__main__":
    app()

