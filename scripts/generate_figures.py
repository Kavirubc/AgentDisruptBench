#!/usr/bin/env python3
"""
AgentDisruptBench — Paper Figure Generator
===========================================

Generates publication-quality figures from evaluation results for the NeurIPS paper.

Usage:
    python3 scripts/generate_figures.py --results-dir runs/baselines_<timestamp>
    python3 scripts/generate_figures.py --results-dir runs/baselines_<timestamp> --output paper/figures/

Produces:
    - fig_heatmap_model_disruption.pdf    : Heatmap of success rate by model × disruption type
    - fig_boxplot_domain_scores.pdf       : Box plots of score distributions per domain
    - fig_difficulty_degradation.pdf      : Line plot showing score degradation by difficulty level
    - fig_recovery_strategy_dist.pdf      : Stacked bar chart of recovery strategy distribution
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

MODEL_COLORS = ["#2E86AB", "#A23B72", "#F18F01", "#2CA58D", "#E84855", "#8B5CF6"]


def load_results(results_dir: Path) -> list[dict]:
    """Load all results.json files from a baselines run directory."""
    all_results = []
    for results_file in results_dir.rglob("results.json"):
        try:
            with open(results_file, encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    all_results.extend(data)
                else:
                    all_results.append(data)
        except (json.JSONDecodeError, OSError) as e:
            print(f"  ⚠️  Failed to load {results_file}: {e}")
    return all_results


def fig_heatmap(results: list[dict], output_dir: Path):
    """Heatmap: model × disruption type success rate."""
    # Gather unique models and disruption types
    models = sorted({r.get("runner_name", "unknown") for r in results})
    all_disruption_types = set()
    for r in results:
        for d in r.get("disruption_types_seen", []):
            all_disruption_types.add(d)
    disruption_types = sorted(all_disruption_types) if all_disruption_types else ["clean"]

    if len(disruption_types) < 2:
        print("  ⚠️  Not enough disruption data for heatmap, skipping")
        return

    # Build matrix
    model_idx = {m: i for i, m in enumerate(models)}
    disruption_idx = {d: i for i, d in enumerate(disruption_types)}
    matrix = np.zeros((len(models), len(disruption_types)))
    counts = np.zeros_like(matrix)
    for r in results:
        mi = model_idx.get(r.get("runner_name", "unknown"))
        if mi is None:
            continue
        for d in r.get("disruption_types_seen", []):
            di = disruption_idx.get(d)
            if di is not None:
                counts[mi, di] += 1
                if r.get("success", False):
                    matrix[mi, di] += 1

    with np.errstate(divide="ignore", invalid="ignore"):
        rates = np.where(counts > 0, matrix / counts * 100, np.nan)

    fig, ax = plt.subplots(figsize=(max(8, len(disruption_types) * 0.8), max(4, len(models) * 0.6)))
    im = ax.imshow(rates, cmap="RdYlGn", aspect="auto", vmin=0, vmax=100)

    ax.set_xticks(range(len(disruption_types)))
    ax.set_xticklabels(disruption_types, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=9)
    ax.set_xlabel("Disruption Type")
    ax.set_ylabel("Model")
    ax.set_title("Success Rate by Model × Disruption Type (%)")

    plt.colorbar(im, ax=ax, label="Success Rate (%)")
    fig.tight_layout()
    fig.savefig(output_dir / "fig_heatmap_model_disruption.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  ✅ fig_heatmap_model_disruption.pdf")


def fig_boxplot_domains(results: list[dict], output_dir: Path):
    """Box plots: score distribution per domain."""
    domain_scores: dict[str, list[float]] = defaultdict(list)
    for r in results:
        domain = r.get("task_domain", r.get("domain", "unknown"))
        score = r.get("partial_score", 0.0)
        domain_scores[domain].append(score)

    domains = sorted(domain_scores.keys())
    data = [domain_scores[d] for d in domains]

    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot(data, labels=[d.capitalize() for d in domains], patch_artist=True)
    for i, patch in enumerate(bp["boxes"]):
        color = MODEL_COLORS[i % len(MODEL_COLORS)]
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel("Partial Score")
    ax.set_title("Score Distribution by Domain")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "fig_boxplot_domain_scores.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  ✅ fig_boxplot_domain_scores.pdf")


def fig_difficulty_degradation(results: list[dict], output_dir: Path):
    """Line plot: score degradation by difficulty level per profile."""
    groups: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in results:
        profile = r.get("profile_name", "unknown")
        diff = r.get("task_difficulty", r.get("difficulty", 1))
        score = r.get("partial_score", 0.0)
        groups[profile][diff].append(score)

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (profile, diff_scores) in enumerate(sorted(groups.items())):
        difficulties = sorted(diff_scores.keys())
        means = [np.mean(diff_scores[d]) for d in difficulties]
        stds = [np.std(diff_scores[d]) for d in difficulties]
        color = MODEL_COLORS[i % len(MODEL_COLORS)]
        ax.errorbar(difficulties, means, yerr=stds, label=profile,
                    marker="o", capsize=3, color=color, linewidth=2)

    all_difficulties = sorted({d for diff_scores in groups.values() for d in diff_scores.keys()})
    ax.set_xlabel("Task Difficulty")
    ax.set_ylabel("Mean Partial Score")
    ax.set_title("Score Degradation by Difficulty Level")
    ax.set_xticks(all_difficulties if all_difficulties else range(1, 6))
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "fig_difficulty_degradation.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  ✅ fig_difficulty_degradation.pdf")


def fig_recovery_distribution(results: list[dict], output_dir: Path):
    """Stacked bar chart: recovery strategy distribution per model."""
    known_order = ["RETRY", "ALTERNATIVE", "ESCALATION", "GIVEUP", "LUCKY"]
    all_strategies = set()
    for r in results:
        all_strategies.update(r.get("recovery_strategies", []))
    strategy_options = [s for s in known_order if s in all_strategies]
    strategy_options += sorted(all_strategies - set(known_order))
    models = sorted({r.get("runner_name", "unknown") for r in results})

    model_strats: dict[str, Counter] = {m: Counter() for m in models}
    for r in results:
        model = r.get("runner_name", "unknown")
        for s in r.get("recovery_strategies", []):
            model_strats[model][s] += 1

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(models))
    width = 0.15
    for i, strat in enumerate(strategy_options):
        vals = [model_strats[m].get(strat, 0) for m in models]
        ax.bar(x + i * width, vals, width, label=strat, color=MODEL_COLORS[i % len(MODEL_COLORS)])

    ax.set_xlabel("Model")
    ax.set_ylabel("Count")
    ax.set_title("Recovery Strategy Distribution by Model")
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "fig_recovery_strategy_dist.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  ✅ fig_recovery_strategy_dist.pdf")


def fig_model_comparison_bar(results: list[dict], output_dir: Path):
    """Grouped bar chart: model comparison across key metrics."""
    models = sorted({r.get("runner_name", "unknown") for r in results})
    metrics = ["success", "partial_score", "recovery_rate"]
    metric_labels = ["Success Rate", "Partial Score", "Recovery Rate"]

    model_metrics: dict[str, dict[str, float]] = {}
    for m in models:
        m_results = [r for r in results if r.get("runner_name") == m]
        model_metrics[m] = {
            "success": np.mean([r.get("success", False) for r in m_results]),
            "partial_score": np.mean([r.get("partial_score", 0) for r in m_results]),
            "recovery_rate": np.mean([r.get("recovery_rate", 0) for r in m_results]),
        }

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(models))
    width = 0.25
    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        vals = [model_metrics[m][metric] for m in models]
        ax.bar(x + i * width, vals, width, label=label, color=MODEL_COLORS[i])

    ax.set_ylabel("Score")
    ax.set_title("Model Comparison: Key Metrics")
    ax.set_xticks(x + width)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "fig_model_comparison.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  ✅ fig_model_comparison.pdf")


def main():
    parser = argparse.ArgumentParser(description="Generate NeurIPS paper figures")
    parser.add_argument("--results-dir", required=True, help="Path to baselines results directory")
    parser.add_argument("--output", default="paper/figures/", help="Output directory for PDFs")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = load_results(results_dir)
    if not results:
        print(f"❌ No results found in {results_dir}")
        sys.exit(1)

    print(f"📊 Loaded {len(results)} results from {results_dir}")
    print(f"📁 Output: {output_dir}\n")

    fig_heatmap(results, output_dir)
    fig_boxplot_domains(results, output_dir)
    fig_difficulty_degradation(results, output_dir)
    fig_recovery_distribution(results, output_dir)
    fig_model_comparison_bar(results, output_dir)

    print(f"\n✅ All figures saved to {output_dir}")


if __name__ == "__main__":
    main()
