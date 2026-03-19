#!/usr/bin/env python3
"""
AgentDisruptBench — Run Log Renderer
======================================

File:        show_run.py
Purpose:     Rich CLI renderer for AgentDisruptBench run logs.
             Renders the full step-by-step narrative of a benchmark run:
             task descriptions, tool call timeline, disruptions fired,
             RAC compensation/recovery events, and final metrics.

Author:      AgentDisruptBench Contributors
License:     MIT
Created:     2026-03-19
Modified:    2026-03-19

Usage:
    python evaluation/show_run.py                              # latest run
    python evaluation/show_run.py --run-id 20260319_045500_a3f
    python evaluation/show_run.py -d results                   # different logs dir

Convention:
    Every source file MUST include a header block like this one.
"""

import json
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

app = typer.Typer(help="Show step-by-step analysis for an AgentDisruptBench run.")
console = Console()


# ─── CLI HELPERS ──────────────────────────────────────────────────────────────


def resolve_run_dir(run_id: Optional[str], logs_dir: str = "logs") -> Path:
    logs_root = Path(logs_dir)
    if run_id:
        run_dir = logs_root / run_id
        if not run_dir.exists():
            console.print(f"[red]Run directory not found: {run_dir}[/red]")
            raise typer.Exit(1)
        return run_dir
    # Default: find the most recently modified run_log.jsonl
    candidates = list(logs_root.glob("*/run_log.jsonl"))
    if not candidates:
        console.print(f"[red]No run_log.jsonl files found under {logs_root}/[/red]")
        raise typer.Exit(1)
    return max(candidates, key=lambda p: p.stat().st_mtime).parent


def load_events(run_dir: Path) -> list[dict[str, Any]]:
    jsonl_path = run_dir / "run_log.jsonl"
    if not jsonl_path.exists():
        console.print(f"[red]No run_log.jsonl found in {run_dir}[/red]")
        raise typer.Exit(1)
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
    return events


# ─── STYLE HELPERS ────────────────────────────────────────────────────────────


def success_icon(val: bool) -> str:
    return "✅" if val else "❌"


def score_color(score: float) -> str:
    if score >= 0.8:
        return "bold green"
    elif score >= 0.5:
        return "bold yellow"
    elif score > 0:
        return "bold red"
    return "dim red"


def disruption_color(dtype: Optional[str]) -> str:
    if dtype is None:
        return "green"
    if "timeout" in (dtype or "").lower():
        return "bold yellow"
    if "429" in (dtype or ""):
        return "bold red"
    if "corrupt" in (dtype or "").lower():
        return "bold magenta"
    return "bold yellow"


def strategy_style(strat: str) -> str:
    styles = {
        "RETRY": "bold cyan",
        "ALTERNATIVE": "bold blue",
        "ESCALATION": "bold magenta",
        "WORKAROUND": "bold yellow",
        "SKIP": "dim yellow",
        "GIVEUP": "bold red",
        "LUCKY": "bold green",
    }
    return styles.get(strat.upper(), "white") if strat else "dim"


# ─── RENDERER ─────────────────────────────────────────────────────────────────


def render_run(events: list[dict[str, Any]], run_dir: Path) -> None:
    # Bucket events by type
    run_started = None
    tasks_selected = None
    task_blocks: list[dict] = []  # Each has: started, tool_calls, rac_events, completed
    run_completed = None

    current_block: dict[str, Any] | None = None

    for e in events:
        et = e["event_type"]
        payload = e["payload"]

        if et == "run_started":
            run_started = payload
        elif et == "tasks_selected":
            tasks_selected = payload
        elif et == "task_started":
            # Save any in-progress block before starting a new one
            if current_block is not None:
                task_blocks.append(current_block)
            current_block = {
                "started": payload,
                "tool_calls": [],
                "rac_events": [],
                "completed": None,
            }
        elif et == "tool_call" and current_block is not None:
            current_block["tool_calls"].append(payload)
        elif et == "rac_event" and current_block is not None:
            current_block["rac_events"].append(payload)
        elif et == "task_completed":
            if current_block is not None:
                current_block["completed"] = payload
                task_blocks.append(current_block)
                current_block = None
        elif et == "run_completed":
            run_completed = payload

    # Capture any block that was in-progress when the run was interrupted
    if current_block is not None:
        task_blocks.append(current_block)

    # ── RUN HEADER ────────────────────────────────────────────────────────────
    run_id = (run_started or {}).get("run_id", run_dir.name)
    console.rule(f"[bold cyan]AgentDisruptBench Run: {run_id}[/bold cyan]", style="cyan")

    if run_started:
        meta = Table(show_header=False, box=box.SIMPLE, padding=(0, 1))
        meta.add_column("Key", style="bold dim")
        meta.add_column("Value")
        meta.add_row("Model", f"[bold]{run_started.get('model', '?')}[/bold]")
        meta.add_row("Profile", f"[bold magenta]{run_started.get('profile', '?')}[/bold magenta]")
        meta.add_row("Domain", run_started.get("domain", "?"))
        meta.add_row("Difficulty", f"{run_started.get('min_difficulty', '?')}–{run_started.get('max_difficulty', '?')}")
        meta.add_row("Seed", str(run_started.get("seed", "?")))
        if run_completed:
            meta.add_row("Tasks", str(run_completed.get("total_tasks", "?")))
            meta.add_row("Duration", f"{run_completed.get('total_duration_seconds', '?')}s")
        console.print(meta)

    # ── TASK LISTING ──────────────────────────────────────────────────────────
    if tasks_selected:
        task_table = Table(show_header=True, box=box.SIMPLE_HEAD, padding=(0, 1))
        task_table.add_column("#", style="bold dim", width=3)
        task_table.add_column("Task ID", style="cyan", no_wrap=True)
        task_table.add_column("Diff", justify="center", width=4)
        task_table.add_column("Title", style="white")
        task_table.add_column("Tools", style="dim")
        task_table.add_column("Depth", justify="center", width=5)
        for i, t in enumerate(tasks_selected.get("tasks", []), 1):
            task_table.add_row(
                str(i),
                t["id"],
                f"D{t['difficulty']}",
                t["title"],
                ", ".join(t["tools"]),
                str(t.get("depth", "?")),
            )
        console.print(Panel(task_table, title="[bold]Selected Tasks[/bold]", border_style="blue"))

    # ── PER-TASK BREAKDOWN ────────────────────────────────────────────────────
    for block_idx, block in enumerate(task_blocks):
        started = block["started"]
        tool_calls = block["tool_calls"]
        rac_events = block["rac_events"]
        completed = block["completed"] or {}

        task_id = started["task_id"]
        success = completed.get("success", False)
        score = completed.get("partial_score", 0)

        # Task header
        status_icon = success_icon(success)
        sc = score_color(score)
        console.rule(
            f"[bold yellow]Task {block_idx + 1}: {task_id}[/bold yellow]  "
            f"{status_icon}  [{sc}]{score:.0%}[/{sc}]",
            style="yellow",
        )

        # Task info
        info = Table(show_header=False, box=box.SIMPLE, padding=(0, 1))
        info.add_column("Key", style="bold dim")
        info.add_column("Value")
        info.add_row("Title", started.get("title", ""))
        info.add_row("Difficulty", f"D{started.get('difficulty', '?')}")
        info.add_row("Type", started.get("task_type", "standard"))
        info.add_row("Tools", ", ".join(started.get("required_tools", [])))
        info.add_row("Expected depth", str(started.get("expected_depth", "?")))
        console.print(info)

        # ── TOOL CALL TIMELINE ────────────────────────────────────────────────
        if tool_calls:
            timeline = Table(show_header=True, box=box.SIMPLE_HEAD, padding=(0, 1))
            timeline.add_column("#", style="bold dim", width=3)
            timeline.add_column("Tool", style="cyan", no_wrap=True)
            timeline.add_column("Status", justify="center", width=8)
            timeline.add_column("Disruption", style="yellow")
            timeline.add_column("Latency", justify="right", width=8)

            for j, tc in enumerate(tool_calls, 1):
                tool_name = tc.get("tool_name", "?")
                ok = tc.get("success", True)
                dtype = tc.get("disruption_type")
                latency = tc.get("latency_ms")

                status = "[green]OK[/green]" if ok else "[red]FAIL[/red]"
                dc = disruption_color(dtype)
                disruption_str = f"[{dc}]{dtype}[/{dc}]" if dtype else "[dim]—[/dim]"
                lat_str = f"{latency:.0f}ms" if latency else "—"

                timeline.add_row(str(j), tool_name, status, disruption_str, lat_str)

            console.print(Panel(timeline, title="[bold]Tool Call Timeline[/bold]", border_style="green"))

        # ── RAC COMPENSATION EVENTS ───────────────────────────────────────────
        if rac_events:
            rac_table = Table(show_header=True, box=box.SIMPLE_HEAD, padding=(0, 1))
            rac_table.add_column("#", style="bold dim", width=3)
            rac_table.add_column("Level", width=7)
            rac_table.add_column("Event", style="white")

            for j, re in enumerate(rac_events, 1):
                level = re.get("level", "INFO")
                msg = re.get("message", "")
                level_style = {
                    "DEBUG": "dim",
                    "INFO": "cyan",
                    "WARNING": "bold yellow",
                    "ERROR": "bold red",
                }.get(level, "white")
                rac_table.add_row(str(j), f"[{level_style}]{level}[/{level_style}]", msg)

            console.print(Panel(
                rac_table,
                title="[bold magenta]RAC Compensation Events[/bold magenta]",
                border_style="magenta",
            ))

        # ── METRICS ───────────────────────────────────────────────────────────
        metrics = Table(show_header=False, box=box.SIMPLE, padding=(0, 1))
        metrics.add_column("Metric", style="bold dim")
        metrics.add_column("Value")

        sc_style = score_color(score)
        metrics.add_row("Success", f"{status_icon}  {success}")
        metrics.add_row("Partial Score", f"[{sc_style}]{score:.2f}[/{sc_style}]")
        metrics.add_row("Recovery Rate", f"{completed.get('recovery_rate', 0):.0%}")
        metrics.add_row("Tool Calls", str(completed.get("total_tool_calls", 0)))
        metrics.add_row("Disruptions", str(completed.get("disruptions_encountered", 0)))
        metrics.add_row("Duration", f"{completed.get('duration_seconds', 0):.1f}s")

        strategies = completed.get("recovery_strategies", [])
        dominant = completed.get("dominant_strategy") or "—"
        if strategies:
            strat_parts = []
            for s in strategies:
                ss = strategy_style(s)
                strat_parts.append(f"[{ss}]{s}[/{ss}]")
            metrics.add_row("Strategies", ", ".join(strat_parts))
            ds = strategy_style(dominant)
            metrics.add_row("Dominant", f"[{ds}]{dominant}[/{ds}]")

        # P1/P2 metrics (emitted by both quick runners)
        if "graceful_giveup" in completed:
            metrics.add_row("Graceful Giveup", str(completed["graceful_giveup"]))
        if "compensation_count" in completed:
            comp_rate = completed.get("compensation_success_rate", 0)
            metrics.add_row(
                "Compensations",
                f"{completed['compensation_count']} ({comp_rate:.0%} success)",
            )
        if completed.get("side_effect_score") is not None:
            metrics.add_row("Side-Effect Score", str(completed["side_effect_score"]))
        if completed.get("idempotency_violations") is not None:
            metrics.add_row(
                "Idempotency Violations",
                str(completed["idempotency_violations"]),
            )
        if "loop_count" in completed:
            metrics.add_row("Loop Count", str(completed["loop_count"]))
        if "planning_time_ratio" in completed:
            metrics.add_row(
                "Planning Ratio",
                f"{completed['planning_time_ratio']:.0%}",
            )
        if "handover_detected" in completed:
            metrics.add_row("Handover Detected", str(completed["handover_detected"]))
        if "tool_hallucination_rate" in completed:
            metrics.add_row(
                "Hallucination Rate",
                f"{completed['tool_hallucination_rate']:.0%}",
            )
        if completed.get("failure_categories"):
            metrics.add_row(
                "Failure Categories",
                ", ".join(
                    f"{k}:{v}"
                    for k, v in sorted(completed["failure_categories"].items())
                ),
            )

        console.print(Panel(metrics, title="[bold]Metrics[/bold]", border_style="yellow"))

        # ── AGENT OUTPUT ──────────────────────────────────────────────────────
        output = completed.get("agent_output", "")
        if output:
            # Normalise list-of-dicts (Gemini format) to plain string
            if isinstance(output, list):
                parts = []
                for item in output:
                    if isinstance(item, dict):
                        parts.append(item.get("text", str(item)))
                    else:
                        parts.append(str(item))
                output = "\n".join(parts)
            output = str(output)
            # Truncate long outputs
            display_output = output[:600] + ("..." if len(output) > 600 else "")
            console.print(Panel(
                Text(display_output, style="white"),
                title="[bold]Agent Output[/bold]",
                border_style="dim",
            ))

    # ── RUN SUMMARY ───────────────────────────────────────────────────────────
    if run_completed:
        console.rule("[bold green]Run Summary[/bold green]", style="green")

        summary = Table(show_header=False, box=box.HEAVY_HEAD, padding=(0, 2))
        summary.add_column("Metric", style="bold")
        summary.add_column("Value", justify="right")

        total = run_completed.get("total_tasks", 0)
        success = run_completed.get("successful", 0)
        rate = run_completed.get("success_rate", 0)
        avg_score = run_completed.get("avg_partial_score", 0)
        duration = run_completed.get("total_duration_seconds", 0)

        rate_style = score_color(rate)
        score_style_val = score_color(avg_score)

        summary.add_row("Total Tasks", str(total))
        summary.add_row("Successful", f"{success}/{total}")
        summary.add_row("Success Rate", f"[{rate_style}]{rate:.0%}[/{rate_style}]")
        summary.add_row("Avg Partial Score", f"[{score_style_val}]{avg_score:.2f}[/{score_style_val}]")
        summary.add_row("Total Duration", f"{duration:.1f}s")

        console.print(Panel(summary, title="[bold green]Final Results[/bold green]", border_style="green"))

        # Per-task summary table
        if task_blocks:
            task_summary = Table(show_header=True, box=box.SIMPLE_HEAD, padding=(0, 1))
            task_summary.add_column("Task", style="cyan")
            task_summary.add_column("D", justify="center", width=3)
            task_summary.add_column("Status", justify="center", width=6)
            task_summary.add_column("Score", justify="right", width=6)
            task_summary.add_column("Tools", justify="right", width=5)
            task_summary.add_column("Disrupts", justify="right", width=8)
            task_summary.add_column("Strategy", width=12)
            task_summary.add_column("Time", justify="right", width=6)

            for block in task_blocks:
                c = block["completed"] or {}
                s = block["started"]
                ok = c.get("success", False)
                sc = c.get("partial_score", 0)
                sc_s = score_color(sc)
                dom = c.get("dominant_strategy") or "—"
                ds = strategy_style(dom) if dom != "—" else "dim"

                task_summary.add_row(
                    s.get("task_id", "?"),
                    f"D{s.get('difficulty', '?')}",
                    success_icon(ok),
                    f"[{sc_s}]{sc:.2f}[/{sc_s}]",
                    str(c.get("total_tool_calls", 0)),
                    str(c.get("disruptions_encountered", 0)),
                    f"[{ds}]{dom}[/{ds}]",
                    f"{c.get('duration_seconds', 0):.1f}s",
                )

            console.print(task_summary)

    console.rule(style="dim")


# ─── ENTRY POINT ──────────────────────────────────────────────────────────────


@app.command()
def main(
    run_id: Optional[str] = typer.Option(None, "--run-id", "-r", help="Run ID to display"),
    logs_dir: str = typer.Option("logs", "--logs-dir", "-d", help="Root directory containing run folders"),
) -> None:
    """Render the full step-by-step narrative for an AgentDisruptBench run."""
    run_dir = resolve_run_dir(run_id, logs_dir)
    console.print(f"[dim]Loading from: {run_dir}[/dim]\n")
    events = load_events(run_dir)
    if not events:
        console.print("[yellow]No events found in this run.[/yellow]")
        raise typer.Exit(0)
    render_run(events, run_dir)


if __name__ == "__main__":
    app()
