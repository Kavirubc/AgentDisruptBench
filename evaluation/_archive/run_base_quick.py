#!/usr/bin/env python3
"""
AgentDisruptBench — Quick Base LangChain Runner
=================================================

File:        run_base_quick.py
Purpose:     Quick smoke-run script for the vanilla LangChain ReAct agent
             (no RAC compensation) for comparison against the RAC runner.

Author:      AgentDisruptBench Contributors
License:     MIT
Created:     2026-03-19
Modified:    2026-03-19

Usage:
    python evaluation/run_base_quick.py --model gemini-flash-latest
    python evaluation/run_base_quick.py --model gemini-flash-latest --task-ids travel_019 retail_020

Convention:
    Every source file MUST include a header block like this one.
"""

import hashlib
import inspect
import json
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Auto-load .env
try:
    from dotenv import load_dotenv

    load_dotenv(os.path.join(project_root, ".env"))
except ImportError:
    pass

import argparse
import logging

from agentdisruptbench import MetricsCalculator, TaskRegistry, ToolRegistry
from agentdisruptbench.core.engine import DisruptionEngine
from agentdisruptbench.core.profiles import get_profile
from agentdisruptbench.core.proxy import ToolProxy
from agentdisruptbench.core.trace import TraceCollector

# ──────────────────────────────────────────────────────────────────────────
# Structured Run Logger
# ──────────────────────────────────────────────────────────────────────────


class RunLogger:
    """Writes structured JSONL events for a single benchmark run."""

    def __init__(self, output_dir: str = "runs"):
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        short_id = uuid.uuid4().hex[:6]
        self.run_id = f"{ts}_{short_id}"
        self.run_dir = Path(output_dir) / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._log_path = self.run_dir / "run_log.jsonl"
        self._f = open(self._log_path, "w")

    def emit(self, event_type: str, payload: dict):
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "payload": payload,
        }
        self._f.write(json.dumps(record, default=str) + "\n")
        self._f.flush()

    def close(self):
        self._f.close()


# ──────────────────────────────────────────────────────────────────────────
# Helper: build Pydantic schema from tool proxy
# ──────────────────────────────────────────────────────────────────────────


def build_args_schema(tool_name, proxy_fn):
    """Build a Pydantic model from the mock tool's function signature."""
    from pydantic import Field, create_model

    real_fn = getattr(proxy_fn, "_fn", None)
    if real_fn is None:
        return None

    try:
        sig = inspect.signature(real_fn)
    except (ValueError, TypeError):
        return None

    fields = {}
    for param_name, param in sig.parameters.items():
        if param_name in ("self", "cls"):
            continue
        ann = param.annotation if param.annotation != inspect.Parameter.empty else str
        if param.default != inspect.Parameter.empty:
            fields[param_name] = (ann, Field(default=param.default))
        else:
            fields[param_name] = (ann, ...)

    if not fields:
        return None

    model_name = "".join(part.capitalize() for part in tool_name.split("_")) + "Input"
    return create_model(model_name, **fields)


# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Quick BASE LangChain runner (no RAC)")
    parser.add_argument("--model", default="gemini-flash-latest", help="LLM model")
    parser.add_argument("--profile", default="mild_production", help="Disruption profile")
    parser.add_argument("--domain", default="travel", help="Task domain")
    parser.add_argument("--task-ids", nargs="+", default=None, help="Specific task IDs")
    parser.add_argument("--max-tasks", type=int, default=3, help="Max tasks to run")
    parser.add_argument("--max-difficulty", type=int, default=5, help="Max difficulty")
    parser.add_argument("--min-difficulty", type=int, default=1, help="Min difficulty")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", default="runs", help="Log output directory")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(levelname)s:%(name)s:%(message)s")
    for noisy in ("httpcore", "httpx", "urllib3", "google_genai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    run_log = RunLogger(output_dir=args.output_dir)

    print("=" * 60)
    print(" AgentDisruptBench — Base LangChain Runner (NO RAC)")
    print("=" * 60)
    print(f"  Run ID:     {run_log.run_id}")
    print(f"  Model:      {args.model}")
    print(f"  Profile:    {args.profile}")
    print(f"  Domain:     {args.domain}")
    print(f"  Difficulty: {args.min_difficulty}–{args.max_difficulty}")
    print(f"  Seed:       {args.seed}")
    print(f"  Logs:       {run_log.run_dir}")
    print()

    run_log.emit(
        "run_started",
        {
            "run_id": run_log.run_id,
            "runner": "base_langchain",
            "model": args.model,
            "profile": args.profile,
            "domain": args.domain,
            "seed": args.seed,
            "min_difficulty": args.min_difficulty,
            "max_difficulty": args.max_difficulty,
        },
    )

    # Load tasks and tools
    task_registry = TaskRegistry.from_builtin()
    tool_registry = ToolRegistry.from_mock_tools()

    # Filter tasks
    if args.task_ids:
        tasks = [t for t in task_registry.all_tasks() if t.task_id in args.task_ids]
    else:
        tasks = task_registry.filter(
            domain=args.domain,
            max_difficulty=args.max_difficulty,
        )
        tasks = [t for t in tasks if t.difficulty >= args.min_difficulty]
        tasks = tasks[: args.max_tasks]

    if not tasks:
        print("❌ No tasks found matching filters")
        sys.exit(1)

    print(f"Found {len(tasks)} task(s) to run:\n")
    for t in tasks:
        print(f"  [{t.task_id}] D{t.difficulty} — {t.title}")
        print(f"    Tools: {', '.join(t.required_tools)}")
        print(f"    Call depth: {t.expected_tool_call_depth}")
    print()

    run_log.emit(
        "tasks_selected",
        {
            "count": len(tasks),
            "tasks": [
                {
                    "id": t.task_id,
                    "title": t.title,
                    "difficulty": t.difficulty,
                    "tools": t.required_tools,
                    "depth": t.expected_tool_call_depth,
                }
                for t in tasks
            ],
        },
    )

    # ── Create LLM ──
    from evaluation.base_runner import RunnerConfig
    from evaluation.runners.rac_runner import _create_llm

    runner_config = RunnerConfig(model=args.model, verbose=args.verbose)
    llm = _create_llm(runner_config)

    # Load disruption profile
    profile = get_profile(args.profile)

    results = []
    calc = MetricsCalculator()

    for i, task in enumerate(tasks, 1):
        print(f"\n{'─' * 60}")
        print(f"Task {i}/{len(tasks)}: {task.task_id}")
        print(f"  Title:       {task.title}")
        print(f"  Difficulty:  {task.difficulty}")
        print(f"  Type:        {task.task_type}")
        print(f"  Tools:       {', '.join(task.required_tools)}")
        print(f"  Call depth:  {task.expected_tool_call_depth}")
        print(f"  Profile:     {args.profile}")
        print(f"{'─' * 60}")

        # Derive a stable per-task seed so each task gets a unique but
        # reproducible RNG stream even when the same global seed is used.
        # Use hashlib (not hash()) to avoid PYTHONHASHSEED randomization.
        _task_id_hash = int.from_bytes(
            hashlib.blake2s(task.task_id.encode("utf-8"), digest_size=4).digest(),
            "big",
        )
        per_task_seed = (args.seed ^ _task_id_hash) & 0x7FFFFFFF

        run_log.emit(
            "task_started",
            {
                "task_id": task.task_id,
                "title": task.title,
                "difficulty": task.difficulty,
                "task_type": task.task_type,
                "required_tools": task.required_tools,
                "expected_depth": task.expected_tool_call_depth,
                "profile": args.profile,
                "disruption_seed": per_task_seed,
            },
        )

        # Create disruption engine with a per-task seed
        engine = DisruptionEngine(configs=profile, seed=per_task_seed)
        trace_collector = TraceCollector()

        # Create tool proxies
        proxied_tools = {}
        for tool_name in task.required_tools:
            raw_fn = tool_registry.get(tool_name)
            proxy = ToolProxy(
                name=tool_name,
                fn=raw_fn,
                engine=engine,
                trace_collector=trace_collector,
            )
            proxied_tools[tool_name] = proxy

        # ── Convert to LangChain StructuredTool (same schema approach as RAC) ──
        from langchain_core.tools import StructuredTool

        lc_tools = []
        for name, fn in proxied_tools.items():
            proxy_fn = fn

            def _make_tool_fn(captured_fn=proxy_fn):
                def tool_fn(**kwargs) -> str:
                    try:
                        result = captured_fn(**kwargs)
                        return json.dumps(result) if isinstance(result, dict) else str(result)
                    except Exception as exc:
                        return json.dumps({"error": str(exc), "status": "failed"})

                return tool_fn

            args_schema = build_args_schema(name, fn)
            tool = StructuredTool.from_function(
                func=_make_tool_fn(),
                name=name,
                description=f"Execute the {name} tool.",
                args_schema=args_schema,
            )
            lc_tools.append(tool)

        # ── Create vanilla ReAct agent (NO RAC) ──
        from langgraph.prebuilt import create_react_agent

        system_prompt = (
            "You are a helpful assistant that completes tasks by calling tools.\n"
            "If a tool call fails or returns an error, you may retry it.\n"
            "When you have enough information, provide a clear final answer "
            "summarising what you accomplished."
        )

        agent = create_react_agent(llm, tools=lc_tools, prompt=system_prompt)

        # Build task input
        task_input = (
            f"Task: {task.description}\n\n"
            f"Available tools: {', '.join(proxied_tools.keys())}\n\n"
            "Please complete this task using the available tools."
        )

        # Run
        start = time.time()
        try:
            result = agent.invoke(
                {"messages": [{"role": "user", "content": task_input}]},
            )
            messages = result.get("messages", [])
            if messages:
                last_msg = messages[-1]
                raw_content = getattr(last_msg, "content", str(last_msg))
                # Normalize structured content (e.g., Gemini returns list of parts)
                if isinstance(raw_content, list):
                    parts = []
                    for item in raw_content:
                        if isinstance(item, dict):
                            parts.append(str(item.get("text", item)))
                        else:
                            parts.append(str(item))
                    agent_output = " ".join(parts)
                elif raw_content:
                    agent_output = str(raw_content)
                else:
                    agent_output = "[No response from agent]"
            else:
                agent_output = "[Agent produced no output]"
        except Exception as exc:
            agent_output = f"[Agent error: {exc}]"
        elapsed = time.time() - start

        # Gather traces
        traces = trace_collector.get_traces()

        for trace in traces:
            trace_dict = {
                "call_id": trace.call_id,
                "tool_name": trace.tool_name,
                "success": trace.observed_success,
                "disruption_type": str(trace.disruption_fired) if trace.disruption_fired else None,
                "latency_ms": round(trace.observed_latency_ms, 1),
                "error": trace.error,
            }
            run_log.emit("tool_call", trace_dict)

        # Compute metrics
        metric_result = calc.compute(
            task=task,
            traces=traces,
            agent_output=agent_output,
            baseline_result=None,
            agent_id=f"base_{args.model}",
            profile_name=args.profile,
            seed=args.seed,
            duration_seconds=elapsed,
            run_start_time=start,
        )
        results.append(metric_result)

        run_log.emit(
            "task_completed",
            {
                "task_id": task.task_id,
                "success": metric_result.success,
                "partial_score": round(metric_result.partial_score, 4),
                "recovery_rate": round(metric_result.recovery_rate, 4),
                "total_tool_calls": metric_result.total_tool_calls,
                "disruptions_encountered": metric_result.disruptions_encountered,
                "duration_seconds": round(elapsed, 2),
                "recovery_strategies": metric_result.recovery_strategies,
                "dominant_strategy": metric_result.dominant_strategy,
                # P1/P2 metrics
                "graceful_giveup": metric_result.graceful_giveup,
                "compensation_count": metric_result.compensation_count,
                "compensation_success_rate": round(metric_result.compensation_success_rate, 4),
                # State-safety metrics require state snapshots; not measured in quick-run
                "side_effect_score": None,
                "idempotency_violations": None,
                "loop_count": metric_result.loop_count,
                "planning_time_ratio": round(metric_result.planning_time_ratio, 4),
                "handover_detected": metric_result.handover_detected,
                "tool_hallucination_rate": round(metric_result.tool_hallucination_rate, 4),
                "failure_categories": metric_result.failure_categories,
                "agent_output": agent_output[:500],
            },
        )

        status = "✅" if metric_result.success else "❌"
        print(f"\n  {status} Success:          {metric_result.success}")
        print(f"  📊 Partial score:   {metric_result.partial_score:.2f}")
        print(f"  🔄 Recovery rate:   {metric_result.recovery_rate:.2%}")
        print(f"  🛠️  Tool calls:      {metric_result.total_tool_calls}")
        print(f"  💥 Disruptions:     {metric_result.disruptions_encountered}")
        print(f"  ⏱️  Duration:        {elapsed:.1f}s")
        if metric_result.recovery_strategies:
            print(f"  🧠 Strategies:      {metric_result.recovery_strategies}")
        if metric_result.dominant_strategy:
            print(f"  👆 Dominant:        {metric_result.dominant_strategy}")
        print("\n  Agent output (first 200 chars):")
        print(f"  {str(agent_output)[:200]}...")

    # Summary
    print(f"\n{'=' * 60}")
    print(" Summary")
    print(f"{'=' * 60}")
    success_count = sum(1 for r in results if r.success)
    total_time = sum(r.duration_seconds for r in results)
    avg_partial = sum(r.partial_score for r in results) / max(len(results), 1)
    print(f"  Total:        {len(results)}")
    print(f"  Successful:   {success_count}/{len(results)}")
    print(f"  Success rate: {success_count / max(len(results), 1) * 100:.1f}%")
    print(f"  Avg partial:  {avg_partial:.2f}")
    print(f"  Total time:   {total_time:.1f}s")
    print(f"  Logs saved:   {run_log.run_dir}")
    print(f"{'=' * 60}")

    run_log.emit(
        "run_completed",
        {
            "total_tasks": len(results),
            "successful": success_count,
            "success_rate": round(success_count / max(len(results), 1), 4),
            "avg_partial_score": round(avg_partial, 4),
            "total_duration_seconds": round(total_time, 2),
        },
    )

    print(f"\n  💡 To analyze: python evaluation/show_run.py --run-id {run_log.run_id}\n")
    run_log.close()


if __name__ == "__main__":
    main()
