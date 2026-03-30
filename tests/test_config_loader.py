"""
AgentDisruptBench — Unit Tests: Config Loader
===============================================

File:        test_config_loader.py
Purpose:     Tests for YAML configuration loading and validation.
             Covers LLM config, benchmark config, validation errors,
             and provider-specific extras pass-through.

Author:      AgentDisruptBench Contributors
License:     MIT
Created:     2026-03-25
Modified:    2026-03-25

Convention:
    Every source file MUST include a header block like this one.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from evaluation.config_loader import (
    BenchmarkYAMLConfig,
    LLMConfig,
    load_benchmark_config,
    load_llm_config,
)


class TestLLMConfig:
    """Tests for LLMConfig dataclass and to_runner_config()."""

    def test_defaults(self):
        cfg = LLMConfig()
        assert cfg.provider == "openai"
        assert cfg.model == "gpt-4o"
        assert cfg.temperature == 0.0

    def test_to_runner_config(self):
        cfg = LLMConfig(
            provider="openai", model="gpt-4o-mini",
            temperature=0.5, max_tokens=8192,
        )
        rc = cfg.to_runner_config()
        assert rc.model == "gpt-4o-mini"
        assert rc.temperature == 0.5
        assert rc.max_tokens == 8192

    def test_to_runner_config_no_max_tokens(self):
        cfg = LLMConfig(provider="openai", model="gpt-4o")
        rc = cfg.to_runner_config()
        assert rc.max_tokens is None

    def test_infer_runner_openai(self):
        cfg = LLMConfig(provider="openai")
        assert cfg.infer_runner() == "openai"

    def test_infer_runner_gemini(self):
        cfg = LLMConfig(provider="gemini")
        assert cfg.infer_runner() == "langchain"

    def test_infer_runner_unknown_fallback(self):
        cfg = LLMConfig(provider="anthropic")
        assert cfg.infer_runner() == "openai"


class TestBenchmarkYAMLConfig:
    """Tests for BenchmarkYAMLConfig dataclass."""

    def test_defaults(self):
        cfg = BenchmarkYAMLConfig()
        assert cfg.runner == "simple"
        assert cfg.max_difficulty == 5
        assert cfg.domains is None


class TestLoadLLMConfig:
    """Tests for load_llm_config()."""

    def test_load_gemini_yaml(self):
        yaml_content = (
            "provider: gemini\n"
            "model: gemini-2.5-flash\n"
            "temperature: 0.0\n"
            "max_tokens: 4096\n"
        )
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            cfg = load_llm_config(f.name)

        assert cfg.provider == "gemini"
        assert cfg.model == "gemini-2.5-flash"
        assert cfg.temperature == 0.0
        assert cfg.max_tokens == 4096
        assert cfg.extra == {}

    def test_load_openai_yaml(self):
        yaml_content = (
            "provider: openai\n"
            "model: gpt-4o\n"
            "temperature: 0.2\n"
        )
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            cfg = load_llm_config(f.name)

        assert cfg.provider == "openai"
        assert cfg.model == "gpt-4o"
        assert cfg.temperature == 0.2

    def test_extra_fields_captured(self):
        """Provider-specific fields (e.g. thinking_budget) go to extra."""
        yaml_content = (
            "provider: gemini\n"
            "model: gemini-2.5-flash\n"
            "thinking_budget: 0\n"
            "reasoning_effort: minimal\n"
        )
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            cfg = load_llm_config(f.name)

        assert cfg.extra["thinking_budget"] == 0
        assert cfg.extra["reasoning_effort"] == "minimal"

    def test_missing_provider_raises(self):
        yaml_content = "model: gpt-4o\n"
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            with pytest.raises(ValueError, match="provider"):
                load_llm_config(f.name)

    def test_missing_model_raises(self):
        yaml_content = "provider: openai\n"
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            with pytest.raises(ValueError, match="model"):
                load_llm_config(f.name)

    def test_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            load_llm_config("/nonexistent/path.yaml")

    def test_invalid_yaml_raises(self):
        yaml_content = "just a plain string\n"
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            with pytest.raises(ValueError, match="YAML mapping"):
                load_llm_config(f.name)

    def test_load_builtin_gemini_config(self):
        """Load the actual shipped config/llm/gemini-2.5-flash.yaml."""
        config_path = Path(__file__).parent.parent / "config" / "llm" / "gemini-2.5-flash.yaml"
        if config_path.exists():
            cfg = load_llm_config(config_path)
            assert cfg.provider == "gemini"
            assert cfg.model == "gemini-2.5-flash"

    def test_load_builtin_openai_config(self):
        """Load the actual shipped config/llm/gpt-4o.yaml."""
        config_path = Path(__file__).parent.parent / "config" / "llm" / "gpt-4o.yaml"
        if config_path.exists():
            cfg = load_llm_config(config_path)
            assert cfg.provider == "openai"
            assert cfg.model == "gpt-4o"


class TestLoadBenchmarkConfig:
    """Tests for load_benchmark_config()."""

    def test_load_full_config(self):
        yaml_content = (
            "runner: rac\n"
            "profiles: [clean, hostile_environment]\n"
            "domains: [retail]\n"
            "max_difficulty: 3\n"
            "seeds: [42, 99]\n"
            "output_dir: my_results\n"
            "verbose: true\n"
        )
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            cfg = load_benchmark_config(f.name)

        assert cfg.runner == "rac"
        assert cfg.profiles == ["clean", "hostile_environment"]
        assert cfg.domains == ["retail"]
        assert cfg.max_difficulty == 3
        assert cfg.seeds == [42, 99]
        assert cfg.output_dir == "my_results"
        assert cfg.verbose is True

    def test_partial_config_uses_defaults(self):
        yaml_content = "runner: openai\n"
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            cfg = load_benchmark_config(f.name)

        assert cfg.runner == "openai"
        assert cfg.max_difficulty == 5  # default
        assert cfg.seeds == [42]  # default

    def test_unknown_fields_ignored(self):
        yaml_content = (
            "runner: simple\n"
            "unknown_field: some_value\n"
        )
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            cfg = load_benchmark_config(f.name)

        assert cfg.runner == "simple"

    def test_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            load_benchmark_config("/nonexistent/path.yaml")

    def test_invalid_yaml_raises(self):
        yaml_content = "42\n"
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            with pytest.raises(ValueError, match="YAML mapping"):
                load_benchmark_config(f.name)

    def test_load_builtin_benchmark_config(self):
        """Load the actual shipped config/benchmark.yaml."""
        config_path = Path(__file__).parent.parent / "config" / "benchmark.yaml"
        if config_path.exists():
            cfg = load_benchmark_config(config_path)
            assert cfg.runner in ("simple", "openai", "langchain", "rac")
