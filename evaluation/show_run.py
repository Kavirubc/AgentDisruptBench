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


def resolve_run_dir(run_id: Optional[str], logs_dir: str = "runs") -> Path:
    logs_root = Path(logs_dir)
    if run_id:
        return logs_root / run_id
    
    # Default is finding the most recently modified directory in runs/
    candidates = [p for p in logs_root.iterdir() if p.is_dir()]
    if not candidates:
        console.print(f"[red]No run directories found under {logs_root}/[/red]")
        raise typer.Exit(1)
    
    return max(candidates, key=lambda p: p.stat().st_mtime)


def load_events(run_dir: Path) -> dict[str, Any]:
    # New task_logs JSON format parsing
    task_logs_dir = run_dir / "task_logs"
    if not task_logs_dir.exists():
        console.print(f"[red]No task_logs directory found in {run_dir}.[/red]")
        raise typer.Exit(1)
        
    runs = []
    for log_file in sorted(task_logs_dir.glob("*.json")):
        try:
            with open(log_file) as f:
                runs.append(json.load(f))
        except Exception as e:
            pass
            
    summary_path = run_dir / "summary.json"
    summary = {}
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
            
    return {"runs": runs, "summary": summary}


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


def render_run(data: dict[str, Any], run_dir: Path) -> None:
    runs = data.get("runs", [])
    summary = data.get("summary", {})
    
    if not runs:
        console.print("[yellow]No tasks found to render.[/yellow]")
        return
        
    import sys
    import os
    from pathlib import Path
    try:
        project_root = str(Path(__file__).parent.parent.absolute())
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        from agentdisruptbench import TaskRegistry
        registry = TaskRegistry.from_builtin()
        task_map = {
            t.task_id: {
                "difficulty": t.difficulty,
                "ground_truth": getattr(t.ground_truth, "expected_outcome", "?") if t.ground_truth else "?"
            } 
            for t in registry.all_tasks()
        }
    except Exception:
        task_map = {}

    # ── RUN HEADER ────────────────────────────────────────────────────────────
    run_id = run_dir.name
    console.rule(f"[bold cyan]AgentDisruptBench Run: {run_id}[/bold cyan]", style="cyan")

    first_run = runs[0]
    meta = Table(show_header=False, box=box.SIMPLE, padding=(0, 1))
    meta.add_column("Key", style="bold dim")
    meta.add_column("Value")
    meta.add_row("Agent", f"[bold]{first_run.get('agent_id', '?')}[/bold]")
    meta.add_row("Profile", f"[bold magenta]{first_run.get('profile_name', '?')}[/bold magenta]")
    meta.add_row("Seed", str(first_run.get("seed", "?")))
    meta.add_row("Tasks", str(len(runs)))
    console.print(meta)

    # ── PER-TASK BREAKDOWN ────────────────────────────────────────────────────
    for block_idx, r in enumerate(runs):
        tool_calls = r.get("traces", [])
        rac_events = [] # Not stored in new format yet
        completed = r

        task_id = r.get("task_id", "")
        success = r.get("success", False)
        score = r.get("partial_score", 0)

        # Task header
        status_icon = success_icon(success)
        sc = score_color(score)
        console.rule(
            f"[bold yellow]Task {block_idx + 1}: {task_id}[/bold yellow]  "
            f"{status_icon}  [{sc}]{score:.0%}[/{sc}]",
            style="yellow",
        )

        # Task info placeholder
        info = Table(show_header=False, box=box.SIMPLE, padding=(0, 1))
        info.add_column("Key", style="bold dim")
        info.add_column("Value")
        
        task_info = task_map.get(task_id, {})
        gt_outcome = task_info.get("ground_truth", "—")
        info.add_row("Ground Truth", f"[bold dim]{gt_outcome}[/bold dim]")
        info.add_row("Duration", f"{r.get('duration_seconds', 0):.2f}s")
        err = r.get("error_msg") or "None"
        info.add_row("Agent Error", f"[red]{err}[/red]")
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
                dtype = tc.get("disruption_fired") or tc.get("disruption_type")
                latency = tc.get("latency_ms") or tc.get("observed_latency_ms")

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
            console.print(Panel(
                Text(output, style="white"),
                title="[bold]Agent Output[/bold]",
                border_style="dim",
            ))

    # ── RUN SUMMARY ───────────────────────────────────────────────────────────
    if summary:
        console.rule("[bold green]Run Summary[/bold green]", style="green")

        sum_table = Table(show_header=False, box=box.HEAVY_HEAD, padding=(0, 2))
        sum_table.add_column("Metric", style="bold")
        sum_table.add_column("Value", justify="right")

        total = summary.get("total_runs", 0)
        profile_stats = list(summary.get("profiles", {}).values())
        if profile_stats:
            p = profile_stats[0]
            success_rate = p.get("success_rate", 0)
            avg_score = p.get("avg_partial_score", 0)

            rate_style = score_color(success_rate)
            score_style_val = score_color(avg_score)

            sum_table.add_row("Total Tasks", str(total))
            sum_table.add_row("Success Rate", f"[{rate_style}]{success_rate:.0%}[/{rate_style}]")
            sum_table.add_row("Avg Partial Score", f"[{score_style_val}]{avg_score:.2f}[/{score_style_val}]")
            console.print(Panel(sum_table, title="[bold green]Final Results[/bold green]", border_style="green"))

        # Per-task summary table
        if runs:
            task_summary = Table(show_header=True, box=box.SIMPLE_HEAD, padding=(0, 1))
            task_summary.add_column("Task", style="cyan")
            task_summary.add_column("Diff", justify="center", width=4)
            task_summary.add_column("Status", justify="center", width=6)
            task_summary.add_column("Score", justify="right", width=6)
            task_summary.add_column("Tools", justify="right", width=5)
            task_summary.add_column("Disr.", justify="right", width=5)
            task_summary.add_column("Recv.", justify="right", width=6)
            task_summary.add_column("Strategy", justify="left", width=12)
            task_summary.add_column("Time", justify="right", width=6)
            task_summary.add_column("Ground Truth", justify="left", style="dim", width=40)

            for r in runs:
                ok = r.get("success", False)
                sc = r.get("partial_score", 0)
                sc_s = score_color(sc)
                
                tid = r.get("task_id", "?")
                task_info = task_map.get(tid, {})
                diff = str(r.get("difficulty") or task_info.get("difficulty", "?"))
                
                gt = task_info.get("ground_truth", "—")
                gt_display = gt[:37] + "..." if len(gt) > 40 else gt

                dist = str(r.get("disruptions_encountered", 0))
                recv = r.get("recovery_rate", 0)
                strat = r.get("dominant_strategy") or "—"
                strat_str = f"[{strategy_style(strat)}]{strat}[/{strategy_style(strat)}]" if strat != "—" else "[dim]—[/dim]"

                task_summary.add_row(
                    tid,
                    f"D{diff}",
                    success_icon(ok),
                    f"[{sc_s}]{sc:.2f}[/{sc_s}]",
                    str(r.get("total_tool_calls", 0)),
                    dist,
                    f"{recv:.0%}",
                    strat_str,
                    f"{r.get('duration_seconds', 0):.1f}s",
                    gt_display,
                )

            console.print(task_summary)

    console.rule(style="dim")


# ─── ENTRY POINT ──────────────────────────────────────────────────────────────


@app.command()
def main(
    run_id: Optional[str] = typer.Option(None, "--run-id", "-r", help="Run ID to display"),
    logs_dir: str = typer.Option("runs", "--logs-dir", "-d", help="Root directory containing run folders"),
) -> None:
    """Render the full step-by-step narrative for an AgentDisruptBench run."""
    run_dir = resolve_run_dir(run_id, logs_dir)
    console.print(f"[dim]Loading from: {run_dir}[/dim]\n")
    data = load_events(run_dir)
    if not data["runs"]:
        console.print("[yellow]No events found in this run.[/yellow]")
        raise typer.Exit(0)
    render_run(data, run_dir)


if __name__ == "__main__":
    app()
