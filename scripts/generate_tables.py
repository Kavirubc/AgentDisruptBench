#!/usr/bin/env python3
"""
AgentDisruptBench — Paper Table Generator
==========================================

Generates LaTeX tables from evaluation run results for the NeurIPS paper.

Usage:
    python3 scripts/generate_tables.py --results-dir runs/baselines_<timestamp>
    python3 scripts/generate_tables.py --results-dir runs/baselines_<timestamp> --output paper/tables/
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


def load_results(results_dir: Path) -> list[dict]:
    """Load all results.json files from a baselines run directory."""
    all_results = []
    for results_file in results_dir.rglob("results.json"):
        with open(results_file) as f:
            data = json.load(f)
            if isinstance(data, list):
                all_results.extend(data)
            else:
                all_results.append(data)
    return all_results


def table_overall_success(results: list[dict]) -> str:
    """Table 1: Overall success rate by model × profile."""
    # Group by (runner_name/model, profile)
    groups: dict[str, dict[str, list[bool]]] = defaultdict(lambda: defaultdict(list))
    for r in results:
        model = r.get("runner_name", "unknown") or r.get("agent_id", "unknown")
        profile = r.get("profile_name", "unknown")
        groups[model][profile].append(r.get("success", False))

    profiles = sorted({r.get("profile_name", "") for r in results})
    models = sorted(groups.keys())

    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Overall task success rate (\%) by model and disruption profile.}",
        r"\label{tab:overall-success}",
        r"\begin{tabular}{l" + "c" * len(profiles) + "}",
        r"\toprule",
        r"\textbf{Model} & " + " & ".join(f"\\textbf{{{p}}}" for p in profiles) + r" \\",
        r"\midrule",
    ]

    for model in models:
        cells = []
        for profile in profiles:
            successes = groups[model][profile]
            if successes:
                rate = sum(successes) / len(successes) * 100
                cells.append(f"{rate:.1f}")
            else:
                cells.append("—")
        lines.append(f"{model} & " + " & ".join(cells) + r" \\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    return "\n".join(lines)


def table_per_domain(results: list[dict]) -> str:
    """Table 2: Success rate by domain × profile."""
    groups: dict[str, dict[str, list[bool]]] = defaultdict(lambda: defaultdict(list))
    for r in results:
        domain = r.get("task_domain", "unknown")
        profile = r.get("profile_name", "unknown")
        groups[domain][profile].append(r.get("success", False))

    profiles = sorted({r.get("profile_name", "") for r in results})
    domains = sorted(groups.keys())

    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Success rate (\%) by domain and disruption profile (averaged across models).}",
        r"\label{tab:per-domain}",
        r"\begin{tabular}{l" + "c" * len(profiles) + "}",
        r"\toprule",
        r"\textbf{Domain} & " + " & ".join(f"\\textbf{{{p}}}" for p in profiles) + r" \\",
        r"\midrule",
    ]

    for domain in domains:
        cells = []
        for profile in profiles:
            successes = groups[domain][profile]
            if successes:
                rate = sum(successes) / len(successes) * 100
                cells.append(f"{rate:.1f}")
            else:
                cells.append("—")
        lines.append(f"{domain.capitalize()} & " + " & ".join(cells) + r" \\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    return "\n".join(lines)


def table_recovery_metrics(results: list[dict]) -> str:
    """Table 3: Recovery metrics by model."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        model = r.get("runner_name", "unknown") or r.get("agent_id", "unknown")
        groups[model].append(r)

    models = sorted(groups.keys())
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Recovery and resilience metrics by model (disrupted profiles only).}",
        r"\label{tab:recovery}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"\textbf{Model} & \textbf{Recovery} & \textbf{Retry} & \textbf{Steps to} & \textbf{Side-Effect} & \textbf{Comp.} & \textbf{Dominant} \\",
        r" & \textbf{Rate} & \textbf{Efficiency} & \textbf{Recovery} & \textbf{Score} & \textbf{Count} & \textbf{Strategy} \\",
        r"\midrule",
    ]

    for model in models:
        rs = [r for r in groups[model] if r.get("profile_name") != "clean"]
        if not rs:
            continue
        avg_recovery = sum(r.get("recovery_rate", 0) for r in rs) / len(rs)
        avg_retry = sum(r.get("retry_efficiency", 0) for r in rs) / len(rs)
        avg_steps = sum(r.get("mean_steps_to_recovery", 0) for r in rs) / len(rs)
        avg_side = sum(r.get("side_effect_score", 0) for r in rs) / len(rs)
        avg_comp = sum(r.get("compensation_count", 0) for r in rs) / len(rs)

        # Dominant strategy across all results
        from collections import Counter
        all_strats: list[str] = []
        for r in rs:
            all_strats.extend(r.get("recovery_strategies", []))
        dominant = Counter(all_strats).most_common(1)[0][0] if all_strats else "—"

        lines.append(
            f"{model} & {avg_recovery:.2f} & {avg_retry:.2f} & {avg_steps:.1f} & "
            f"{avg_side:.2f} & {avg_comp:.1f} & {dominant} \\\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate LaTeX tables for NeurIPS paper")
    parser.add_argument("--results-dir", required=True, help="Path to baselines results directory")
    parser.add_argument("--output", default="paper/tables/", help="Output directory for .tex files")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = load_results(results_dir)
    if not results:
        print(f"❌ No results found in {results_dir}")
        sys.exit(1)

    print(f"📊 Loaded {len(results)} results from {results_dir}")

    # Generate tables
    tables = {
        "table_overall_success.tex": table_overall_success(results),
        "table_per_domain.tex": table_per_domain(results),
        "table_recovery_metrics.tex": table_recovery_metrics(results),
    }

    for filename, content in tables.items():
        path = output_dir / filename
        path.write_text(content)
        print(f"  ✅ {path}")

    print(f"\n✅ Generated {len(tables)} tables in {output_dir}")


if __name__ == "__main__":
    main()
