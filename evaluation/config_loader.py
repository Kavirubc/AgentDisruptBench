"""
AgentDisruptBench — Configuration Loader
==========================================

File:        config_loader.py
Purpose:     Load benchmark and LLM configurations from YAML files.
             Provides Pydantic-validated models for both benchmark-level
             settings and per-LLM provider settings. Inspired by the
             pentest-evo config/llm/*.yaml pattern.

Author:      AgentDisruptBench Contributors
License:     MIT
Created:     2026-03-25
Modified:    2026-03-25

Key Classes:
    LLMConfig            : Pydantic model for config/llm/*.yaml files.
    BenchmarkYAMLConfig  : Pydantic model for config/benchmark.yaml.

Key Functions:
    load_llm_config       : Parse an LLM YAML file into an LLMConfig.
    load_benchmark_config : Parse a benchmark YAML file into a BenchmarkYAMLConfig.

Convention:
    Every source file MUST include a header block like this one.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from evaluation.base_runner import RunnerConfig

logger = logging.getLogger("agentdisruptbench.evaluation.config_loader")

# Maps LLM provider names → default runner names.
# When a user specifies only an LLM config, we can infer the runner.
PROVIDER_RUNNER_MAP: dict[str, str] = {
    "openai": "langchain",
    "gemini": "langchain",
}


@dataclass
class LLMConfig:
    """Validated configuration for an LLM provider.

    Loaded from a YAML file (e.g. ``config/llm/gpt-4o.yaml``).

    Attributes:
        provider:     LLM provider name (``"openai"`` or ``"gemini"``).
        model:        Model identifier (e.g. ``"gpt-4o"``, ``"gemini-2.5-flash"``).
        temperature:  Sampling temperature.
        max_tokens:   Max tokens per LLM response (None = provider default).
        max_retries:  Max retries on API errors.
        max_steps:    Max agent loop iterations.
        api_key:      Optional API key (falls back to env vars).
        extra:        Provider-specific extra parameters.
    """

    provider: str = "openai"
    model: str = "gpt-4o"
    temperature: float = 0.0
    max_tokens: int | None = None
    max_retries: int = 3
    max_steps: int = 20
    api_key: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_runner_config(self) -> RunnerConfig:
        """Convert to a RunnerConfig for use with BaseAgentRunner."""
        return RunnerConfig(
            model=self.model,
            api_key=self.api_key,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            max_retries=self.max_retries,
            max_steps=self.max_steps,
        )

    def infer_runner(self) -> str:
        """Infer the best runner name from the provider.

        Returns:
            Runner name string (e.g. ``"langchain"``).
        """
        return PROVIDER_RUNNER_MAP.get(self.provider, "langchain")


@dataclass
class BenchmarkYAMLConfig:
    """Validated configuration for a benchmark run.

    Loaded from a YAML file (e.g. ``config/benchmark.yaml``).

    Attributes:
        runner:         Runner name (``"simple"``, ``"langchain"``, ``"rac"``).
        profiles:       Disruption profiles to evaluate.
        domains:        Restrict to these domains (None = all).
        max_difficulty: Max task difficulty (1–5).
        seeds:          Random seeds for reproducibility.
        output_dir:     Directory for report output.
        verbose:        Print agent reasoning to stdout.
        agent_id:       Optional agent identifier for reports.
    """

    runner: str | None = None
    profiles: list[str] = field(default_factory=lambda: ["clean", "mild_production", "hostile_environment"])
    domains: list[str] | None = None
    tasks: list[str] | None = None
    max_difficulty: int = 5
    seeds: list[int] = field(default_factory=lambda: [42])
    output_dir: str = "results"
    verbose: bool = False
    agent_id: str | None = None


# -- Known fields for each config type (used for validation) --------

_LLM_FIELDS = {
    "provider",
    "model",
    "temperature",
    "max_tokens",
    "max_retries",
    "max_steps",
    "api_key",
}

_BENCHMARK_FIELDS = {
    "runner",
    "profiles",
    "domains",
    "tasks",
    "max_difficulty",
    "seeds",
    "output_dir",
    "verbose",
    "agent_id",
}


def load_llm_config(path: str | Path) -> LLMConfig:
    """Load and validate an LLM configuration from a YAML file.

    Any keys not in the standard LLMConfig fields are stored in ``extra``
    for provider-specific pass-through (e.g. ``thinking_budget`` for Gemini,
    ``reasoning_effort`` for OpenAI).

    Args:
        path: Path to the YAML file.

    Returns:
        Validated LLMConfig instance.

    Raises:
        FileNotFoundError: If the path does not exist.
        ValueError: If required fields are missing or invalid.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"LLM config file not found: {path}")

    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    if not isinstance(data, dict):
        raise ValueError(f"LLM config must be a YAML mapping, got: {type(data).__name__}")

    if "provider" not in data:
        raise ValueError(f"LLM config missing required field 'provider' in: {path}")

    if "model" not in data:
        raise ValueError(f"LLM config missing required field 'model' in: {path}")

    # Separate known fields from provider-specific extras
    known_kwargs: dict[str, Any] = {}
    extra_kwargs: dict[str, Any] = {}

    for key, value in data.items():
        if key in _LLM_FIELDS:
            known_kwargs[key] = value
        else:
            extra_kwargs[key] = value

    config = LLMConfig(**known_kwargs, extra=extra_kwargs)

    logger.info(
        "llm_config_loaded path=%s provider=%s model=%s",
        path,
        config.provider,
        config.model,
    )
    return config


def load_benchmark_config(path: str | Path) -> BenchmarkYAMLConfig:
    """Load and validate a benchmark configuration from a YAML file.

    Args:
        path: Path to the YAML file.

    Returns:
        Validated BenchmarkYAMLConfig instance.

    Raises:
        FileNotFoundError: If the path does not exist.
        ValueError: If the YAML is not a valid mapping.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Benchmark config file not found: {path}")

    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    if not isinstance(data, dict):
        raise ValueError(f"Benchmark config must be a YAML mapping, got: {type(data).__name__}")

    # Filter to known fields only (ignore unknown keys gracefully)
    known_kwargs: dict[str, Any] = {}
    for key, value in data.items():
        if key in _BENCHMARK_FIELDS:
            known_kwargs[key] = value
        else:
            logger.warning("benchmark_config_unknown_field key=%s path=%s", key, path)

    config = BenchmarkYAMLConfig(**known_kwargs)

    logger.info(
        "benchmark_config_loaded path=%s runner=%s profiles=%s",
        path,
        config.runner,
        config.profiles,
    )
    return config
