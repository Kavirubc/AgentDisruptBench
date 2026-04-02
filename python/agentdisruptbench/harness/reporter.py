"""
AgentDisruptBench — Reporter
==============================

File:        reporter.py
Purpose:     Generates Markdown and JSON reports from BenchmarkResult lists.
             Outputs per-task, per-profile, and aggregate summaries with
             tables, statistics, and scoring breakdowns.

Author:      AgentDisruptBench Contributors
License:     MIT
Created:     2026-03-09
Modified:    2026-03-09

Key Classes:
    Reporter : Generates Markdown and JSON report files from results.

Convention:
    Every source file MUST include a header block like this one.
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agentdisruptbench.core.metrics import BenchmarkResult

logger = logging.getLogger("agentdisruptbench.reporter")


class Reporter:
    """Generates benchmark reports from a list of BenchmarkResult.

    Produces:
    - ``report.md``  : Markdown report with tables and summaries.
    - ``results.json``: Machine-readable JSON with all metrics.
    - ``results.csv`` : Detailed flat-file report for spreadsheet analysis.
    - ``summary.json``: Aggregate statistics per profile.
    """

    def __init__(self, output_dir: str | None = None) -> None:
        if output_dir is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            self._output_dir = Path(f"runs/{ts}")
        else:
            self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def generate(self, results: list[BenchmarkResult]) -> dict[str, str]:
        """Generate all report files.

        Args:
            results: List of BenchmarkResult from a benchmark run.

        Returns:
            Dict mapping filename → absolute path of generated files.
        """
        paths: dict[str, str] = {}

        paths["report.md"] = self._write_markdown(results)
        paths["results.json"] = self._write_json(results)
        paths["results.csv"] = self._write_csv(results)
        paths["summary.json"] = self._write_summary(results)
        paths["task_logs"] = self._write_task_logs(results)

        logger.info("reports_generated output_dir=%s", self._output_dir)
        return paths

    # -- Markdown report ---------------------------------------------------

    def _write_markdown(self, results: list[BenchmarkResult]) -> str:
        """Generate the Markdown report."""
        path = self._output_dir / "report.md"
        lines: list[str] = []
        ts = datetime.now(timezone.utc).isoformat()

        lines.append("# AgentDisruptBench — Benchmark Report")
        lines.append(f"\nGenerated: {ts}\n")

        # Aggregate stats
        profiles = sorted({r.profile_name for r in results})
        lines.append("## Summary by Profile\n")
        lines.append("| Profile | Tasks | Success Rate | Avg Partial | Recovery Rate | Avg Extra Calls |")
        lines.append("|---------|-------|-------------|-------------|---------------|-----------------|")

        for profile in profiles:
            pr = [r for r in results if r.profile_name == profile]
            n = len(pr)
            success_pct = sum(1 for r in pr if r.success) / max(n, 1) * 100
            avg_partial = sum(r.partial_score for r in pr) / max(n, 1)
            avg_recovery = sum(r.recovery_rate for r in pr) / max(n, 1)
            avg_extra = sum((r.extra_tool_calls or 0) for r in pr) / max(n, 1)
            lines.append(
                f"| {profile} | {n} | {success_pct:.1f}% | {avg_partial:.3f} | "
                f"{avg_recovery:.3f} | {avg_extra:.1f} |"
            )

        # Per-domain breakdown
        domains = sorted({r.task_id.split("_")[0] for r in results})
        lines.append("\n## Results by Domain\n")
        for domain in domains:
            dr = [r for r in results if r.task_id.startswith(domain)]
            lines.append(f"### {domain.title()}\n")
            lines.append("| Task | Profile | Success | Partial | Recovery | Disruptions | Extra Calls |")
            lines.append("|------|---------|---------|---------|----------|-------------|-------------|")
            for r in sorted(dr, key=lambda x: (x.task_id, x.profile_name)):
                lines.append(
                    f"| {r.task_id} | {r.profile_name} | "
                    f"{'✅' if r.success else '❌'} | {r.partial_score:.2f} | "
                    f"{r.recovery_rate:.2f} | {r.disruptions_encountered} | "
                    f"{r.extra_tool_calls or 'N/A'} |"
                )

        # Disruption type distribution
        lines.append("\n## Disruption Types Encountered\n")
        type_counts: dict[str, int] = {}
        for r in results:
            for dt in r.disruption_types_seen:
                type_counts[dt] = type_counts.get(dt, 0) + 1
        if type_counts:
            lines.append("| Type | Count |")
            lines.append("|------|-------|")
            for dt, count in sorted(type_counts.items(), key=lambda x: -x[1]):
                lines.append(f"| {dt} | {count} |")
        else:
            lines.append("No disruptions encountered.\n")

        # Footer
        lines.append("\n---")
        lines.append("*Report generated by [AgentDisruptBench](https://github.com/AgentDisruptBench)*")

        content = "\n".join(lines)
        path.write_text(content, encoding="utf-8")
        logger.info("markdown_report_written path=%s", path)
        return str(path)

    # -- JSON results ------------------------------------------------------

    def _write_json(self, results: list[BenchmarkResult]) -> str:
        """Write full results as JSON."""
        path = self._output_dir / "results.json"

        data = []
        for r in results:
            d = asdict(r)
            # Remove raw traces from JSON output (too large)
            d.pop("traces", None)
            data.append(d)

        path.write_text(
            json.dumps(data, indent=2, default=str), encoding="utf-8"
        )
        logger.info("json_results_written path=%s count=%d", path, len(data))
        return str(path)

    # -- CSV results -------------------------------------------------------

    def _write_csv(self, results: list[BenchmarkResult]) -> str:
        """Write detailed results as a flat CSV for analysis."""
        path = self._output_dir / "results.csv"

        headers = [
            "Task ID", "Domain", "Difficulty", "Profile", "Seed",
            "Success", "Partial Score", "Duration (s)", 
            "Prompt Tokens", "Completion Tokens", "Total Tokens",
            "Runner", "Environment",
            "Recovery Rate", "Disruptions Encountered", "Disruptions Recovered",
            "Extra Tool Calls", "Total Tool Calls",
            "State Score", "Compensations", "Idempotency Violations",
            "Loop Count", "Hallucination Rate", "Handover Detected",
            "Task Description"
        ]

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            for r in results:
                writer.writerow({
                    "Task ID": r.task_id,
                    "Domain": r.task_domain,
                    "Difficulty": r.task_difficulty,
                    "Profile": r.profile_name,
                    "Seed": r.seed,
                    "Success": r.success,
                    "Partial Score": round(r.partial_score, 4),
                    "Duration (s)": round(r.duration_seconds, 2),
                    "Prompt Tokens": r.prompt_tokens,
                    "Completion Tokens": r.completion_tokens,
                    "Total Tokens": r.token_usage,
                    "Runner": r.runner_name or "N/A",
                    "Environment": r.agent_id,
                    "Recovery Rate": round(r.recovery_rate, 4),
                    "Disruptions Encountered": r.disruptions_encountered,
                    "Disruptions Recovered": r.disruptions_recovered,
                    "Extra Tool Calls": r.extra_tool_calls if r.extra_tool_calls is not None else 0,
                    "Total Tool Calls": r.total_tool_calls,
                    "State Score": round(r.side_effect_score, 4),
                    "Compensations": r.compensation_count,
                    "Idempotency Violations": r.idempotency_violations,
                    "Loop Count": r.loop_count,
                    "Hallucination Rate": round(r.tool_hallucination_rate, 4),
                    "Handover Detected": r.handover_detected,
                    "Task Description": r.task_description
                })

        logger.info("csv_results_written path=%s count=%d", path, len(results))
        return str(path)

    # -- Summary JSON ------------------------------------------------------

    def _write_summary(self, results: list[BenchmarkResult]) -> str:
        """Write aggregate summary statistics."""
        path = self._output_dir / "summary.json"

        profiles = sorted({r.profile_name for r in results})
        summary: dict[str, Any] = {
            "generated": datetime.now(timezone.utc).isoformat(),
            "total_runs": len(results),
            "profiles": {},
        }

        for profile in profiles:
            pr = [r for r in results if r.profile_name == profile]
            n = len(pr)
            summary["profiles"][profile] = {
                "tasks_evaluated": n,
                "success_rate": sum(1 for r in pr if r.success) / max(n, 1),
                "avg_partial_score": sum(r.partial_score for r in pr) / max(n, 1),
                "avg_recovery_rate": sum(r.recovery_rate for r in pr) / max(n, 1),
                "avg_retry_efficiency": sum(r.retry_efficiency for r in pr) / max(n, 1),
                "total_disruptions": sum(r.disruptions_encountered for r in pr),
                "total_recovered": sum(r.disruptions_recovered for r in pr),
                "avg_extra_tool_calls": sum((r.extra_tool_calls or 0) for r in pr) / max(n, 1),
                "avg_duration_seconds": sum(r.duration_seconds for r in pr) / max(n, 1),
            }

        path.write_text(
            json.dumps(summary, indent=2, default=str), encoding="utf-8"
        )
        logger.info("summary_written path=%s", path)
        return str(path)

    # -- Per-task log files ------------------------------------------------

    def _write_task_logs(self, results: list[BenchmarkResult]) -> str:
        """Write per-task detailed log files with traces and agent output.

        Creates a ``task_logs/`` subdirectory with one JSON file per run.
        Each file contains the full traces, agent output, and all metrics.
        """
        log_dir = self._output_dir / "task_logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        for r in results:
            filename = f"{r.task_id}_{r.profile_name}_{r.seed}.json"
            log_path = log_dir / filename

            # Build trace dicts
            trace_dicts = []
            for t in r.traces:
                trace_dicts.append({
                    "call_id": t.call_id,
                    "call_number": t.call_number,
                    "tool_name": t.tool_name,
                    "inputs": t.inputs,
                    "real_result": str(t.real_result)[:500],
                    "observed_result": str(t.observed_result)[:500],
                    "real_success": t.real_success,
                    "observed_success": t.observed_success,
                    "disruption_fired": t.disruption_fired,
                    "real_latency_ms": t.real_latency_ms,
                    "observed_latency_ms": t.observed_latency_ms,
                    "error": t.error,
                    "timestamp": t.timestamp,
                })

            log_data = {
                "task_id": r.task_id,
                "agent_id": r.agent_id,
                "profile_name": r.profile_name,
                "seed": r.seed,
                "success": r.success,
                "partial_score": r.partial_score,
                "agent_output": r.agent_output,
                "recovery_rate": r.recovery_rate,
                "mean_steps_to_recovery": r.mean_steps_to_recovery,
                "retry_efficiency": r.retry_efficiency,
                "disruptions_encountered": r.disruptions_encountered,
                "disruptions_recovered": r.disruptions_recovered,
                "disruption_types_seen": r.disruption_types_seen,
                "total_tool_calls": r.total_tool_calls,
                "extra_tool_calls": r.extra_tool_calls,
                "duration_seconds": r.duration_seconds,
                "acknowledged_failure": r.acknowledged_failure,
                "attempted_alternative": r.attempted_alternative,
                "traces": trace_dicts,
            }

            log_path.write_text(
                json.dumps(log_data, indent=2, default=str), encoding="utf-8"
            )

        logger.info("task_logs_written dir=%s count=%d", log_dir, len(results))
        return str(log_dir)
