"""
AgentDisruptBench — Disruption Profiles
========================================

File:        profiles.py
Purpose:     Built-in disruption profiles (9 profiles) and a YAML loader for
             custom profiles. Profiles map a name to an ordered list of
             DisruptionConfig entries that the engine evaluates per tool call.

Author:      AgentDisruptBench Contributors
License:     MIT
Created:     2026-03-09
Modified:    2026-03-09

Key Definitions:
    BUILTIN_PROFILES : dict[str, list[DisruptionConfig]]  All 9 built-in
                       profiles defined in the AgentDisruptBench spec.
    load_profiles    : Load profiles from a YAML file on disk.
    get_profile      : Retrieve a profile by name (built-in or custom).

Convention:
    Every source file MUST include a header block like this one.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from agentdisruptbench.core.engine import DisruptionConfig, DisruptionType

logger = logging.getLogger("agentdisruptbench.profiles")


# ---------------------------------------------------------------------------
# Built-in profiles — spec §5
# ---------------------------------------------------------------------------

BUILTIN_PROFILES: dict[str, list[DisruptionConfig]] = {
    "clean": [],

    "mild_production": [
        DisruptionConfig(type=DisruptionType.LATENCY, probability=0.10, delay_ms=800),
        DisruptionConfig(type=DisruptionType.HTTP_429, probability=0.03),
        DisruptionConfig(type=DisruptionType.TRUNCATED, probability=0.02, truncation_pct=0.75),
    ],

    "moderate_production": [
        DisruptionConfig(type=DisruptionType.TIMEOUT, probability=0.07, delay_ms=6000),
        DisruptionConfig(type=DisruptionType.HTTP_429, probability=0.08),
        DisruptionConfig(type=DisruptionType.HTTP_500, probability=0.05),
        DisruptionConfig(type=DisruptionType.MALFORMED_JSON, probability=0.04),
        DisruptionConfig(type=DisruptionType.MISSING_FIELDS, probability=0.06),
    ],

    "hostile_environment": [
        DisruptionConfig(type=DisruptionType.TIMEOUT, probability=0.15, delay_ms=10000),
        DisruptionConfig(type=DisruptionType.HTTP_429, probability=0.12),
        DisruptionConfig(type=DisruptionType.HTTP_500, probability=0.10),
        DisruptionConfig(type=DisruptionType.MALFORMED_JSON, probability=0.10),
        DisruptionConfig(type=DisruptionType.SCHEMA_DRIFT, probability=0.08),
        DisruptionConfig(type=DisruptionType.MISSING_FIELDS, probability=0.10),
        DisruptionConfig(type=DisruptionType.WRONG_DATA, probability=0.08),
        DisruptionConfig(type=DisruptionType.NULL_RESPONSE, probability=0.05),
    ],

    "auth_pressure": [
        DisruptionConfig(type=DisruptionType.HTTP_401, probability=0.10),
        DisruptionConfig(type=DisruptionType.AUTH_EXPIRY, fail_after_n_calls=4),
    ],

    "quota_pressure": [
        DisruptionConfig(type=DisruptionType.HTTP_429, probability=0.15),
        DisruptionConfig(type=DisruptionType.QUOTA_EXHAUSTED, fail_after_n_calls=6),
    ],

    "data_corruption": [
        DisruptionConfig(type=DisruptionType.WRONG_DATA, probability=0.15),
        DisruptionConfig(type=DisruptionType.MISSING_FIELDS, probability=0.15),
        DisruptionConfig(type=DisruptionType.SCHEMA_DRIFT, probability=0.10),
        DisruptionConfig(type=DisruptionType.TYPE_MISMATCH, probability=0.08),
        DisruptionConfig(type=DisruptionType.NULL_RESPONSE, probability=0.05),
        DisruptionConfig(type=DisruptionType.TRUNCATED, probability=0.10, truncation_pct=0.40),
    ],

    "cascading_failure": [
        DisruptionConfig(
            type=DisruptionType.CASCADING,
            probability=1.0,
            target_tools=["payment_process"],
            cascade_targets=["refund_issue", "order_status", "invoice_generate"],
        ),
    ],

    "flapping_services": [
        DisruptionConfig(type=DisruptionType.FLAPPING, probability=0.50),
        DisruptionConfig(type=DisruptionType.INTERMITTENT, fail_every_n=3),
    ],
}


# ---------------------------------------------------------------------------
# YAML loader
# ---------------------------------------------------------------------------


def load_profiles(path: str | Path) -> dict[str, list[DisruptionConfig]]:
    """Load disruption profiles from a YAML file.

    YAML format::

        profiles:
          my_profile:
            description: "Human-readable description"
            disruptions:
              - type: timeout
                probability: 0.10
                delay_ms: 5000

    Args:
        path: Path to the YAML profiles file.

    Returns:
        Dictionary mapping profile name → list of DisruptionConfig.

    Raises:
        FileNotFoundError: If the YAML file does not exist.
        PermissionError: If the file is not readable.
    """
    path = Path(path)
    with open(path, "r", encoding="utf-8") as fh:
        raw: dict[str, Any] = yaml.safe_load(fh)

    if not raw or "profiles" not in raw:
        logger.warning("no_profiles_key path=%s", path)
        return {}

    profiles: dict[str, list[DisruptionConfig]] = {}
    for name, body in raw["profiles"].items():
        disruptions = body.get("disruptions", [])
        configs: list[DisruptionConfig] = []
        for d in disruptions:
            configs.append(DisruptionConfig(**d))
        profiles[name] = configs
        logger.info(
            "profile_loaded name=%s disruption_count=%d", name, len(configs)
        )

    return profiles


def get_profile(name: str, custom_profiles: dict[str, list[DisruptionConfig]] | None = None) -> list[DisruptionConfig]:
    """Retrieve a profile by name.

    Checks custom profiles first, then built-in profiles.

    Args:
        name: Profile name.
        custom_profiles: Optional externally-loaded profiles.

    Returns:
        List of DisruptionConfig for the profile.

    Raises:
        KeyError: If the profile name is not found.
    """
    if custom_profiles and name in custom_profiles:
        return custom_profiles[name]
    if name in BUILTIN_PROFILES:
        return BUILTIN_PROFILES[name]
    available = list(BUILTIN_PROFILES.keys())
    if custom_profiles:
        available += list(custom_profiles.keys())
    raise KeyError(
        f"Unknown profile '{name}'. Available: {available}"
    )
