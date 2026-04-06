"""
AgentDisruptBench — Tests for run_multi_benchmark
====================================================

File:        test_run_multi_benchmark.py
Purpose:     Tests for the multi-model benchmark runner: provider
             grouping and config parsing.

Author:      AgentDisruptBench Contributors
License:     MIT
Created:     2026-03-25
Modified:    2026-03-25

Convention:
    Every source file MUST include a header block like this one.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import yaml

# Ensure project root is on path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from evaluation.run_multi_benchmark import group_by_provider

# ─── FIXTURES ─────────────────────────────────────────────────────────────────


def _write_llm_yaml(path: Path, model: str, provider: str) -> str:
    """Write a minimal LLM config YAML and return its path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "provider": provider,
        "model": model,
        "temperature": 0.0,
    }
    with open(path, "w") as f:
        yaml.dump(data, f)
    return str(path)


# ─── TESTS: PROVIDER GROUPING ────────────────────────────────────────────────


class TestGroupByProvider:
    """Tests for group_by_provider()."""

    def test_single_provider(self, tmp_path: Path) -> None:
        """All OpenAI models should be in one group."""
        paths = [
            _write_llm_yaml(tmp_path / "gpt4o.yaml", "gpt-4o", "openai"),
            _write_llm_yaml(tmp_path / "gpt4omini.yaml", "gpt-4o-mini", "openai"),
        ]
        groups = group_by_provider(paths)
        assert len(groups) == 1
        assert "openai" in groups
        assert len(groups["openai"]) == 2

    def test_multiple_providers(self, tmp_path: Path) -> None:
        """Gemini + OpenAI should be in separate groups."""
        paths = [
            _write_llm_yaml(tmp_path / "gemini.yaml", "gemini-2.5-flash", "gemini"),
            _write_llm_yaml(tmp_path / "gpt4o.yaml", "gpt-4o", "openai"),
        ]
        groups = group_by_provider(paths)
        assert len(groups) == 2
        assert "gemini" in groups
        assert "openai" in groups
        assert len(groups["gemini"]) == 1
        assert len(groups["openai"]) == 1

    def test_three_providers(self, tmp_path: Path) -> None:
        """Three different providers should get three groups."""
        paths = [
            _write_llm_yaml(tmp_path / "g1.yaml", "gemini-2.5-flash", "gemini"),
            _write_llm_yaml(tmp_path / "o1.yaml", "gpt-4o", "openai"),
            _write_llm_yaml(tmp_path / "o2.yaml", "gpt-5-mini", "openai"),
        ]
        groups = group_by_provider(paths)
        assert len(groups) == 2
        assert len(groups["openai"]) == 2
        assert len(groups["gemini"]) == 1

    def test_preserves_order(self, tmp_path: Path) -> None:
        """Config paths should be in insertion order within groups."""
        p1 = _write_llm_yaml(tmp_path / "a.yaml", "gpt-4o", "openai")
        p2 = _write_llm_yaml(tmp_path / "b.yaml", "gpt-4o-mini", "openai")
        p3 = _write_llm_yaml(tmp_path / "c.yaml", "gpt-5-mini", "openai")

        groups = group_by_provider([p1, p2, p3])
        assert groups["openai"] == [p1, p2, p3]
