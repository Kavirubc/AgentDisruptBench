"""
AgentDisruptBench — RAC (React Agent Compensation) Runner
===========================================================

File:        rac_runner.py
Purpose:     Evaluation runner using Kaviru's RAC framework (react-agent-
             compensation). Wraps AgentDisruptBench ToolProxy callables as
             LangChain tools, passes them through RAC's CompensationMiddleware
             so that compensation/rollback is automatic, then runs the full
             LangGraph ReAct loop.

Author:      AgentDisruptBench Contributors
License:     MIT
Created:     2026-03-19
Modified:    2026-03-19

Key Classes:
    RACRunner : BaseAgentRunner subclass using RAC's compensated agent.

Dependencies:
    pip install react-agent-compensation[langchain]
    pip install langchain-google-genai  # for Gemini
    # or
    pip install langchain-openai         # for OpenAI

Usage:
    python -m evaluation.run_benchmark --runner rac --model gemini-2.0-flash

Convention:
    Every source file MUST include a header block like this one.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from evaluation.base_runner import BaseAgentRunner, RunnerConfig
from agentdisruptbench.tasks.schemas import Task

logger = logging.getLogger("agentdisruptbench.evaluation.runners.rac")

# Maps AgentDisruptBench side-effect tools → their compensation tools.
# Uses the same mappings that AgentDisruptBench's StateManager tracks.
_BENCH_COMPENSATION_PAIRS: dict[str, str] = {
    "book_flight": "cancel_booking",
    "place_order": "process_refund",
    "deploy_service": "rollback_deployment",
    "create_incident": "resolve_incident",
}


def _is_gemini_model(model: str) -> bool:
    return model.lower().startswith("gemini")


def _create_llm(config: RunnerConfig):
    """Create the LangChain chat model (Gemini or OpenAI)."""
    if _is_gemini_model(config.model):
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
                "Gemini API key required. Set GEMINI_API_KEY or GOOGLE_API_KEY."
            )

        return ChatGoogleGenerativeAI(
            model=config.model,
            google_api_key=api_key,
            temperature=config.temperature,
            max_output_tokens=config.max_tokens,
        )
    else:
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise ImportError(
                "OpenAI models require langchain-openai. "
                "Install with: pip install langchain-openai"
            )

        api_key = config.api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY.")

        return ChatOpenAI(
            model=config.model,
            api_key=api_key,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )


class RACRunner(BaseAgentRunner):
    """ReAct agent using RAC (react-agent-compensation) framework.

    This runner wraps AgentDisruptBench's ToolProxy callables as LangChain
    tools and creates a RAC compensated agent via ``create_compensated_agent``.

    RAC provides:
    - Automatic compensation/rollback on tool failures
    - Retry strategies with configurable backoff
    - Strategic context preservation (failure history for LLM)
    - Goal-aware recovery guidance

    The runner measures how RAC's compensation layer interacts with
    AgentDisruptBench's disruption injection to test agent resilience.

    Usage::

        runner = RACRunner(RunnerConfig(model="gemini-2.0-flash"))
        result = runner.run_task(task, tools)
    """

    def __init__(self, config: RunnerConfig | None = None) -> None:
        super().__init__(config or RunnerConfig(model="gemini-2.0-flash"))
        self._llm = None

    def setup(self) -> None:
        """Initialise the LLM for RAC agent creation."""
        self._llm = _create_llm(self.config)
        provider = "Gemini" if _is_gemini_model(self.config.model) else "OpenAI"
        logger.info("rac_runner_setup provider=%s model=%s", provider, self.config.model)
        super().setup()

    def run_task(self, task: Task, tools: dict[str, Any]) -> str:
        """Run a RAC compensated agent for one benchmark task.

        Steps:
        1. Convert ToolProxy callables → LangChain Tool objects.
        2. Build compensation mapping from the tools present in this task.
        3. Create a RAC compensated agent.
        4. Invoke with the task prompt.
        5. Extract final message content.
        6. Log RAC's transaction log for analysis.
        """
        if self._llm is None:
            self.setup()

        try:
            from langchain_core.tools import StructuredTool
            from react_agent_compensation.langchain_adaptor import (
                create_compensated_agent,
                get_compensation_middleware,
            )
            from react_agent_compensation.core import RetryPolicy
        except ImportError as e:
            raise ImportError(
                "RAC runner requires react-agent-compensation[langchain]. "
                "Install with: pip install react-agent-compensation[langchain]"
            ) from e

        # Step 1: Convert ToolProxy callables → LangChain StructuredTool
        lc_tools = []
        for name, fn in tools.items():
            proxy_fn = fn  # capture in closure

            def _make_tool_fn(captured_fn=proxy_fn):
                """Create a tool function that accepts **kwargs."""
                def tool_fn(**kwargs) -> str:
                    try:
                        result = captured_fn(**kwargs)
                        return json.dumps(result) if isinstance(result, dict) else str(result)
                    except Exception as exc:
                        return json.dumps({"error": str(exc), "status": "failed"})
                return tool_fn

            tool = StructuredTool.from_function(
                func=_make_tool_fn(),
                name=name,
                description=f"Execute the {name} tool.",
            )
            lc_tools.append(tool)

        # Step 2: Build compensation mapping for tools in this task.
        #
        # IMPORTANT: RAC only tracks and retries tools that appear in the
        # compensation_mapping.  Tools NOT in the mapping get zero retry
        # or recovery coverage — errors pass straight through to the LLM.
        #
        # Strategy:
        #   - Side-effect tools → their real compensator (e.g. book_flight → cancel_booking)
        #   - Read-only tools  → mapped to themselves as a no-op sentinel,
        #     which makes RAC consider them "compensatable" so it will
        #     record, detect errors, and retry them using the retry_policy.
        comp_mapping: dict[str, str] = {}
        tool_names = set(tools.keys())

        for name in tool_names:
            if name in _BENCH_COMPENSATION_PAIRS:
                compensator = _BENCH_COMPENSATION_PAIRS[name]
                if compensator and compensator in tool_names:
                    comp_mapping[name] = compensator
                else:
                    # Side-effect tool whose compensator isn't in this task
                    comp_mapping[name] = name
            else:
                # Read-only tool — map to self so RAC tracks + retries it
                comp_mapping[name] = name

        # Step 3: Create RAC compensated agent
        retry_policy = RetryPolicy(
            max_retries=2,
            initial_delay=0.1,
            backoff_multiplier=1.5,
        )

        system_prompt = (
            "You are a resilient assistant completing tasks by calling tools.\n"
            "If a tool call fails, the compensation framework will handle "
            "retries and rollbacks automatically.\n"
            "Focus on completing the task and providing a clear final answer "
            "summarising what you accomplished.\n"
            "If you determine a task is impossible, explain why clearly."
        )

        agent = create_compensated_agent(
            model=self._llm,
            tools=lc_tools,
            compensation_mapping=comp_mapping,
            retry_policy=retry_policy,
            auto_rollback=True,
            auto_recover=True,
            goals=["complete_task_successfully", "minimize_side_effects"],
        )

        # Step 4: Build task input and invoke
        task_input = (
            f"Task: {task.description}\n\n"
            f"Available tools: {', '.join(tools.keys())}\n\n"
            "Please complete this task using the available tools."
        )

        try:
            result = agent.invoke(
                {"messages": [{"role": "user", "content": task_input}]},
            )

            # Step 5: Extract final message content
            messages = result.get("messages", [])
            if messages:
                last_msg = messages[-1]
                content = getattr(last_msg, "content", str(last_msg))
                self._total_api_calls += sum(
                    1 for m in messages if getattr(m, "type", "") == "ai"
                )

                # Step 6: Log RAC transaction log for analysis
                middleware = get_compensation_middleware(agent)
                if middleware:
                    log_snapshot = middleware.transaction_log.snapshot()
                    if log_snapshot:
                        logger.info(
                            "rac_transaction_log task=%s entries=%d",
                            task.task_id, len(log_snapshot),
                        )
                        for rid, record in log_snapshot.items():
                            logger.debug(
                                "  action=%s status=%s compensator=%s",
                                record.action, record.status, record.compensator,
                            )

                return content or "[No response from agent]"

            return "[Agent produced no output]"

        except Exception as exc:
            logger.exception("rac_agent_error task=%s", task.task_id)
            return f"[Agent error: {exc}]"

    def teardown(self) -> None:
        """Clean up."""
        self._llm = None
        super().teardown()
