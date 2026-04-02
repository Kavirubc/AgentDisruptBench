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
from typing import Any

from agentdisruptbench.tasks.schemas import Task

from evaluation.base_runner import (
    BaseAgentRunner, RunnerConfig,
    _RESET, _BOLD, _DIM, _GREEN, _RED, _YELLOW, _CYAN, _MAGENTA,
)
from evaluation.llm_factory import create_langchain_llm, detect_provider

logger = logging.getLogger("agentdisruptbench.evaluation.runners.rac")


# Maps AgentDisruptBench side-effect tools → their compensation tools.
# Uses the same mappings that AgentDisruptBench's StateManager tracks.
_BENCH_COMPENSATION_PAIRS: dict[str, str] = {
    "book_flight": "cancel_booking",
    "place_order": "process_refund",
    "deploy_service": "rollback_deployment",
    "create_incident": "resolve_incident",
}


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
        self._llm = create_langchain_llm(self.config)
        provider = detect_provider(self.config.model)
        logger.info("rac_runner_setup provider=%s model=%s", provider, self.config.model)
        super().setup()

    @staticmethod
    def _build_args_schema(tool_name: str, proxy_fn: Any) -> type | None:
        """Build a Pydantic model from the mock tool's function signature.

        StructuredTool.from_function infers its schema from the wrapped
        function's signature.  Since our wrapper uses ``**kwargs``, the
        inferred schema has a single ``kwargs`` field — which breaks
        argument forwarding.

        This method inspects the *underlying* mock tool function (accessed
        via ``proxy_fn._fn``) and dynamically creates a Pydantic BaseModel
        with the correct parameter names and types.
        """
        import inspect

        from pydantic import Field, create_model

        # Walk the proxy chain to find the real function with a meaningful signature:
        #   ToolProxy._fn → may be a stateful_wrapper(**kwargs) or the raw static method
        #   stateful_wrapper closure → contains the real mock tool function
        # We keep unwrapping until we find a function with named parameters.
        real_fn = getattr(proxy_fn, '_fn', None) or proxy_fn

        # If real_fn only has **kwargs, try to find the original fn in its closure
        try:
            sig = inspect.signature(real_fn)
            params = [
                p for p in sig.parameters.values()
                if p.name not in ("self", "cls")
                and p.kind not in (
                    inspect.Parameter.VAR_KEYWORD,
                    inspect.Parameter.VAR_POSITIONAL,
                )
            ]
            if not params:
                # Look inside closure cells for a callable with real params
                closure = getattr(real_fn, '__closure__', None) or []
                for cell in closure:
                    try:
                        candidate = cell.cell_contents
                        if callable(candidate) and candidate is not real_fn:
                            csig = inspect.signature(candidate)
                            cparams = [
                                p for p in csig.parameters.values()
                                if p.name not in ("self", "cls")
                                and p.kind not in (
                                    inspect.Parameter.VAR_KEYWORD,
                                    inspect.Parameter.VAR_POSITIONAL,
                                )
                            ]
                            if cparams:
                                real_fn = candidate
                                sig = csig
                                break
                    except (ValueError, AttributeError):
                        continue
        except (ValueError, TypeError):
            return None

        fields: dict[str, Any] = {}
        for param_name, param in sig.parameters.items():
            if param_name in ("self", "cls"):
                continue

            # Determine type annotation (default to str)
            ann = param.annotation if param.annotation != inspect.Parameter.empty else str

            # Determine default
            if param.default != inspect.Parameter.empty:
                fields[param_name] = (ann, Field(default=param.default))
            else:
                fields[param_name] = (ann, ...)

        if not fields:
            return None

        # Create a Pydantic model with the tool name as class name
        model_name = "".join(part.capitalize() for part in tool_name.split("_")) + "Input"
        return create_model(model_name, **fields)

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
            from react_agent_compensation.core import RetryPolicy
            from react_agent_compensation.langchain_adaptor import (
                create_compensated_agent,
                get_compensation_middleware,
            )
        except ImportError as e:
            raise ImportError(
                "RAC runner requires react-agent-compensation[langchain]. "
                "Install with: pip install react-agent-compensation[langchain]"
            ) from e

        # Step 1: Convert ToolProxy callables → LangChain StructuredTool
        #
        # We must build a proper Pydantic args_schema for each tool so that
        # LangChain (and the LLM) sees the correct parameter names/types.
        # Without this, StructuredTool.from_function infers a schema from
        # the wrapper's **kwargs signature, which produces a single "kwargs"
        # field — breaking argument forwarding.
        verbose = self.config.verbose
        lc_tools = []
        for name, fn in tools.items():
            proxy_fn = fn  # capture in closure

            def _make_tool_fn(captured_fn=proxy_fn, tool_name=name):
                """Create a tool function that accepts **kwargs."""
                def tool_fn(**kwargs) -> str:
                    if verbose:
                        args_str = ", ".join(
                            f"{k}={repr(v)[:40]}" for k, v in kwargs.items()
                        )
                        print(f"    {_CYAN}🔧 {_BOLD}{tool_name}{_RESET}{_CYAN}({args_str}){_RESET}")
                    try:
                        result = captured_fn(**kwargs)
                        if verbose:
                            preview = str(result)[:100].replace("\n", " ")
                            print(f"    {_GREEN}✓  → {_DIM}{preview}{_RESET}")
                        return json.dumps(result) if isinstance(result, dict) else str(result)
                    except Exception as exc:
                        if verbose:
                            print(f"    {_RED}✗  → {exc}{_RESET}")
                        return json.dumps({"error": str(exc), "status": "failed"})
                return tool_fn

            # Build Pydantic schema from the underlying mock tool's signature
            args_schema = self._build_args_schema(name, fn)

            tool = StructuredTool.from_function(
                func=_make_tool_fn(),
                name=name,
                description=f"Execute the {name} tool.",
                args_schema=args_schema,
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

            # Step 5: Extract tokens and final message content
            messages = result.get("messages", [])
            if messages:
                # Track tokens from all messages with usage_metadata
                for m in messages:
                    if hasattr(m, "usage_metadata") and m.usage_metadata:
                        usage = m.usage_metadata
                        self._prompt_tokens += usage.get("input_tokens", 0)
                        self._completion_tokens += usage.get("output_tokens", 0)
                    elif hasattr(m, "response_metadata") and m.response_metadata:
                        # Fallback for older LangChain versions or specific providers
                        resp_meta = m.response_metadata
                        token_usage = resp_meta.get("token_usage", {})
                        if token_usage:
                            self._prompt_tokens += token_usage.get("prompt_tokens", 0)
                            self._completion_tokens += token_usage.get("completion_tokens", 0)

                last_msg = messages[-1]
                content = getattr(last_msg, "content", str(last_msg))

                # Normalize structured content (e.g., Gemini returns list of parts)
                if isinstance(content, list):
                    parts = []
                    for item in content:
                        if isinstance(item, dict):
                            parts.append(str(item.get("text", item)))
                        else:
                            parts.append(str(item))
                    content = " ".join(parts)
                else:
                    content = str(content)

                self._task_api_calls += sum(
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
