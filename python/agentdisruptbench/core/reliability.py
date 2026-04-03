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
        per_domain:          Domain → composite score (k × ε × λ per domain).
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


def _compute_k_and_lambda(
    results: list[BenchmarkResult],
) -> tuple[float, float]:
    """Compute k-consistency and lambda-fault-tolerance for a result set.

    Returns:
        (k_consistency, lambda_fault_tolerance) both in [0.0, 1.0].
    """
    overall_rate = sum(r.success for r in results) / len(results)

    # k: pass rate across repeated seeds for the same (task, profile) pair
    k_groups: dict[tuple[str, str], list[bool]] = defaultdict(list)
    for r in results:
        k_groups[(r.task_id, r.profile_name)].append(r.success)

    k_rates = [sum(s) / len(s) for s in k_groups.values() if len(s) > 1]
    k_consistency = sum(k_rates) / len(k_rates) if k_rates else overall_rate

    # λ: pass rate across *unique* profiles per task; dedupe repeated seeds
    profile_groups: dict[tuple[str, str], list[bool]] = defaultdict(list)
    for r in results:
        profile_groups[(r.task_id, r.profile_name)].append(r.success)

    task_profile_rates: dict[str, list[float]] = defaultdict(list)
    for (task_id, _), successes in profile_groups.items():
        task_profile_rates[task_id].append(sum(successes) / len(successes))

    lambda_rates = [sum(rates) / len(rates) for rates in task_profile_rates.values() if rates]
    lambda_fault_tolerance = sum(lambda_rates) / len(lambda_rates) if lambda_rates else overall_rate

    return k_consistency, lambda_fault_tolerance


def _base_task_id(task_id: str) -> str:
    """Strip variant suffix (e.g. _v1, _v2) to get the family base ID.

    Examples:
        "retail_001"    → "retail_001"
        "retail_001_v1" → "retail_001"
        "retail_001_v2" → "retail_001"
    """
    import re
    return re.sub(r"_v\d+$", "", task_id)


def _compute_epsilon(results: list[BenchmarkResult]) -> float:
    """Compute ε-robustness: pass rate consistency across task-wording variants.

    Groups results by variant family (base task ID) and profile, then
    measures how consistently an agent passes across different phrasings.

    Returns 1.0 if no variant families exist (backwards compatible), otherwise
    returns the average intra-family pass-rate consistency.
    """
    # Group results by (base_task_id, profile) → list of successes
    family_groups: dict[tuple[str, str], list[bool]] = defaultdict(list)
    for r in results:
        base = _base_task_id(r.task_id)
        family_groups[(base, r.profile_name)].append(r.success)

    # Only consider families with multiple variants (i.e., base + at least 1 _vN)
    multi_variant_rates: list[float] = []
    for (base, _), successes in family_groups.items():
        if len(successes) > 1:
            multi_variant_rates.append(sum(successes) / len(successes))

    if not multi_variant_rates:
        return 1.0  # No variants → placeholder (backward compatible)

    return sum(multi_variant_rates) / len(multi_variant_rates)


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

    k_consistency, lambda_fault_tolerance = _compute_k_and_lambda(results)

    # ε-robustness: pass rate across task-wording variants.
    # Variant families are identified by stripping the _v1/_v2/... suffix.
    # E.g., retail_001, retail_001_v1, retail_001_v2 form one family.
    epsilon_robustness = _compute_epsilon(results)

    composite = k_consistency * epsilon_robustness * lambda_fault_tolerance

    # --- Per-domain breakdown ---
    # Use the authoritative task_domain field when available; fall back to
    # parsing task_id for results produced before this field existed.
    _TASK_TYPE_PREFIXES = {"adversarial", "impossible", "handover"}

    domain_groups: dict[str, list[BenchmarkResult]] = defaultdict(list)
    for r in results:
        if r.task_domain:
            domain = r.task_domain
        else:
            parts = r.task_id.split("_")
            if parts[0] in _TASK_TYPE_PREFIXES and len(parts) >= 3:
                domain = parts[1]
            elif len(parts) >= 2:
                domain = parts[0]
            else:
                domain = "unknown"
        domain_groups[domain].append(r)

    per_domain: dict[str, float] = {}
    for domain, dr in domain_groups.items():
        dk, dl = _compute_k_and_lambda(dr)
        per_domain[domain] = round(dk * epsilon_robustness * dl, 4)

    # --- Per-difficulty breakdown ---
    # Use the authoritative task_difficulty field when available.
    difficulty_groups: dict[int, list[BenchmarkResult]] = defaultdict(list)
    for r in results:
        if r.task_difficulty:
            difficulty_groups[r.task_difficulty].append(r)

    per_difficulty: dict[int, float] = {}
    for diff, dr in difficulty_groups.items():
        dk, dl = _compute_k_and_lambda(dr)
        per_difficulty[diff] = round(dk * epsilon_robustness * dl, 4)

    return ReliabilitySurface(
        k_consistency=round(k_consistency, 4),
        epsilon_robustness=round(epsilon_robustness, 4),
        lambda_fault_tolerance=round(lambda_fault_tolerance, 4),
        composite_score=round(composite, 4),
        per_domain=per_domain,
        per_difficulty=per_difficulty,
        num_results=len(results),
    )
