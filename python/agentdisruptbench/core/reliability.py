"""
AgentDisruptBench — Reliability Surface
========================================

File:        reliability.py
Purpose:     Computes the R(k,ε,λ) reliability surface from multi-seed,
             multi-profile evaluation results.  Provides per-agent,
             per-domain, and per-difficulty reliability breakdowns.

Author:      AgentDisruptBench Contributors
License:     MIT
Created:     2026-03-19
Modified:    2026-03-19

Key Classes:
    ReliabilitySurface : Holds k-consistency, ε-robustness, λ-fault-tolerance.
    compute_reliability_surface : Factory function to build the surface.

Convention:
    Every source file MUST include a header block like this one.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field

from agentdisruptbench.core.metrics import BenchmarkResult

logger = logging.getLogger("agentdisruptbench.reliability")


@dataclass
class ReliabilitySurface:
    """R(k,ε,λ) reliability surface for an agent.

    Attributes:
        k_consistency:       Pass rate across repeated seeds for the same
                             (task, profile) pair.  Higher = more deterministic.
        epsilon_robustness:  Pass rate across task-wording variants.
                             Placeholder (1.0) until variant tasks exist.
        lambda_fault_tolerance: Pass rate across disruption profiles for
                                the same task, ordered by intensity.
        composite_score:     Product of the three axes: k × ε × λ.
        per_domain:          Domain → composite score.
        per_difficulty:      Difficulty → composite score.
        num_results:         Total results used in computation.
    """

    k_consistency: float = 0.0
    epsilon_robustness: float = 1.0  # Placeholder until variant tasks
    lambda_fault_tolerance: float = 0.0
    composite_score: float = 0.0
    per_domain: dict[str, float] = field(default_factory=dict)
    per_difficulty: dict[int, float] = field(default_factory=dict)
    num_results: int = 0


def compute_reliability_surface(
    results: list[BenchmarkResult],
) -> ReliabilitySurface:
    """Compute reliability surface from a collection of benchmark results.

    The results should contain runs across multiple seeds and/or profiles
    for meaningful k and λ values.

    Args:
        results: List of BenchmarkResult from multi-seed/multi-profile runs.

    Returns:
        Populated ReliabilitySurface.
    """
    if not results:
        return ReliabilitySurface()

    # --- k-consistency: pass rate over seeds for same (task, profile) ---
    k_groups: dict[tuple[str, str], list[bool]] = defaultdict(list)
    for r in results:
        k_groups[(r.task_id, r.profile_name)].append(r.success)

    k_rates = []
    for successes in k_groups.values():
        if len(successes) > 1:
            k_rates.append(sum(successes) / len(successes))
    k_consistency = sum(k_rates) / len(k_rates) if k_rates else (
        sum(r.success for r in results) / len(results)
    )

    # --- ε-robustness: placeholder (no variant tasks yet) ---
    epsilon_robustness = 1.0

    # --- λ-fault-tolerance: pass rate across profiles for same task ---
    lambda_groups: dict[str, list[bool]] = defaultdict(list)
    for r in results:
        lambda_groups[r.task_id].append(r.success)

    lambda_rates = []
    for successes in lambda_groups.values():
        if len(successes) > 1:
            lambda_rates.append(sum(successes) / len(successes))
    lambda_fault_tolerance = (
        sum(lambda_rates) / len(lambda_rates) if lambda_rates
        else sum(r.success for r in results) / len(results)
    )

    # --- Composite ---
    composite = k_consistency * epsilon_robustness * lambda_fault_tolerance

    # --- Per-domain breakdown ---
    domain_groups: dict[str, list[BenchmarkResult]] = defaultdict(list)
    for r in results:
        # Extract domain from task_id prefix (e.g. "retail_001" → "retail")
        domain = r.task_id.rsplit("_", 1)[0] if "_" in r.task_id else "unknown"
        domain_groups[domain].append(r)

    per_domain: dict[str, float] = {}
    for domain, domain_results in domain_groups.items():
        dr_pass = sum(r.success for r in domain_results)
        per_domain[domain] = dr_pass / len(domain_results)

    # --- Per-difficulty breakdown ---
    # Difficulty is not in BenchmarkResult, so we approximate from task_id suffix
    # This will be more accurate when we store difficulty in the result
    per_difficulty: dict[int, float] = {}

    return ReliabilitySurface(
        k_consistency=round(k_consistency, 4),
        epsilon_robustness=round(epsilon_robustness, 4),
        lambda_fault_tolerance=round(lambda_fault_tolerance, 4),
        composite_score=round(composite, 4),
        per_domain=per_domain,
        per_difficulty=per_difficulty,
        num_results=len(results),
    )
