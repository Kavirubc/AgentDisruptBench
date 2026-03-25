"""
AgentDisruptBench — LLM Factory
=================================

File:        llm_factory.py
Purpose:     Shared LLM creation logic for LangChain-based runners.
             Eliminates the duplicated _create_llm() / _is_gemini_model()
             code in langchain_runner.py and rac_runner.py.

Author:      AgentDisruptBench Contributors
License:     MIT
Created:     2026-03-25
Modified:    2026-03-25

Key Functions:
    detect_provider      : Infer provider from a model name string.
    create_langchain_llm : Create the appropriate LangChain ChatModel.

Convention:
    Every source file MUST include a header block like this one.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from evaluation.base_runner import RunnerConfig

logger = logging.getLogger("agentdisruptbench.evaluation.llm_factory")


def detect_provider(model: str) -> str:
    """Infer the LLM provider from a model name.

    Args:
        model: Model identifier string (e.g. ``"gemini-2.5-flash"``, ``"gpt-4o"``).

    Returns:
        Provider string: ``"gemini"`` or ``"openai"`` (default fallback).
    """
    if model.lower().startswith("gemini"):
        return "gemini"
    return "openai"


def create_langchain_llm(config: RunnerConfig, provider: str | None = None) -> Any:
    """Create a LangChain ChatModel for the given provider.

    Supports Gemini (via ``langchain-google-genai``) and OpenAI
    (via ``langchain-openai``).  If *provider* is not specified,
    it is auto-detected from ``config.model``.

    Args:
        config:   Runner configuration with model, api_key, temperature, etc.
        provider: Optional explicit provider name (``"gemini"`` or ``"openai"``).

    Returns:
        A LangChain ChatModel instance.

    Raises:
        ImportError: If the required LangChain integration is not installed.
        ValueError: If the API key is missing.
    """
    if provider is None:
        provider = detect_provider(config.model)

    if provider == "gemini":
        return _create_gemini_llm(config)
    return _create_openai_llm(config)


def _create_gemini_llm(config: RunnerConfig) -> Any:
    """Create a ChatGoogleGenerativeAI instance.

    Raises:
        ImportError: If langchain-google-genai is not installed.
        ValueError: If no Gemini API key is found.
    """
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError:
        raise ImportError(
            "Gemini models require langchain-google-genai. "
            "Install with: pip install langchain-google-genai"
        )

    api_key = (
        config.api_key
        or os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
    )
    if not api_key:
        raise ValueError(
            "Gemini API key required. Set GEMINI_API_KEY or GOOGLE_API_KEY "
            "env var, or pass api_key in RunnerConfig."
        )

    logger.info("creating_gemini_llm model=%s", config.model)
    return ChatGoogleGenerativeAI(
        model=config.model,
        google_api_key=api_key,
        temperature=config.temperature,
        max_output_tokens=config.max_tokens,
    )


def _create_openai_llm(config: RunnerConfig) -> Any:
    """Create a ChatOpenAI instance.

    Raises:
        ImportError: If langchain-openai is not installed.
        ValueError: If no OpenAI API key is found.
    """
    try:
        from langchain_openai import ChatOpenAI
    except ImportError:
        raise ImportError(
            "OpenAI models require langchain-openai. "
            "Install with: pip install langchain-openai"
        )

    api_key = config.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OpenAI API key required. Set OPENAI_API_KEY env var "
            "or pass api_key in RunnerConfig."
        )

    logger.info("creating_openai_llm model=%s", config.model)
    return ChatOpenAI(
        model=config.model,
        api_key=api_key,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
    )
