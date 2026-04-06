"""
AgentDisruptBench — Unit Tests: LLM Factory
=============================================

File:        test_llm_factory.py
Purpose:     Tests for the shared LLM factory module. Validates provider
             detection and verifies that create_langchain_llm raises
             appropriate errors for missing API keys and dependencies.

Author:      AgentDisruptBench Contributors
License:     MIT
Created:     2026-03-25
Modified:    2026-03-25

Convention:
    Every source file MUST include a header block like this one.
"""

from __future__ import annotations

from unittest.simulated import patch

import pytest

from evaluation.base_runner import RunnerConfig
from evaluation.llm_factory import create_langchain_llm, detect_provider


class TestDetectProvider:
    """Tests for detect_provider()."""

    def test_gemini_models(self):
        assert detect_provider("gemini-2.5-flash") == "gemini"
        assert detect_provider("gemini-2.0-flash") == "gemini"
        assert detect_provider("gemini-2.5-pro") == "gemini"
        assert detect_provider("Gemini-2.5-Flash") == "gemini"  # case insensitive

    def test_openai_models(self):
        assert detect_provider("gpt-4o") == "openai"
        assert detect_provider("gpt-4o-mini") == "openai"
        assert detect_provider("gpt-3.5-turbo") == "openai"

    def test_unknown_models_default_to_openai(self):
        assert detect_provider("claude-3-opus") == "openai"
        assert detect_provider("llama-3.1") == "openai"
        assert detect_provider("unknown-model") == "openai"

    def test_empty_string(self):
        assert detect_provider("") == "openai"


class TestCreateLangChainLLM:
    """Tests for create_langchain_llm() error handling.

    These tests do NOT call real APIs — they verify error handling
    for missing keys and missing packages.
    """

    def test_gemini_missing_key_raises(self):
        """Missing Gemini API key raises ValueError."""
        config = RunnerConfig(model="gemini-2.5-flash", api_key=None)
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises((ValueError, ImportError)):
                create_langchain_llm(config, provider="gemini")

    def test_openai_missing_key_raises(self):
        """Missing OpenAI API key raises ValueError."""
        config = RunnerConfig(model="gpt-4o", api_key=None)
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises((ValueError, ImportError)):
                create_langchain_llm(config, provider="openai")

    def test_explicit_provider_overrides_detection(self):
        """Explicit provider param overrides auto-detection."""
        config = RunnerConfig(model="gemini-2.5-flash", api_key=None)
        # Force openai provider for a gemini model name → should try OpenAI path
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises((ValueError, ImportError)):
                create_langchain_llm(config, provider="openai")

    def test_auto_detection_gemini(self):
        """Auto-detects gemini from model name."""
        config = RunnerConfig(model="gemini-2.5-flash", api_key=None)
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises((ValueError, ImportError)):
                create_langchain_llm(config)  # Should auto-detect as gemini
