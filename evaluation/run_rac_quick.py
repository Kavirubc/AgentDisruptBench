#!/usr/bin/env python3
"""
AgentDisruptBench — Quick RAC Runner
======================================

File:        run_rac_quick.py
Purpose:     Quick smoke-run script for the RAC (React-Agent-Compensation)
             evaluation runner with structured JSONL logging.

Author:      AgentDisruptBench Contributors
License:     MIT
Created:     2026-03-19
Modified:    2026-03-19

Usage:
    # Ensure GEMINI_API_KEY or GOOGLE_API_KEY is set
    python evaluation/run_rac_quick.py

    # With OpenAI
    python evaluation/run_rac_quick.py --model gpt-4o

    # Analyze the run afterwards:
    python evaluation/show_run.py           # latest run
    python evaluation/show_run.py --run-id <timestamp>

Convention:
    Every source file MUST include a header block like this one.
"""

import os
import sys
import time
import json
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

from agentdisruptbench import TaskRegistry, ToolRegistry, MetricsCalculator
from agentdisruptbench.core.profiles import get_profile
from agentdisruptbench.core.engine import DisruptionEngine
from agentdisruptbench.core.proxy import ToolProxy
from agentdisruptbench.core.trace import TraceCollector
from evaluation.base_runner import RunnerConfig
from evaluation.runners.rac_runner import RACRunner


# ──────────────────────────────────────────────────────────────────────────
# Structured Run Logger — emits JSONL events to logs/<run_id>/run_log.jsonl
# ──────────────────────────────────────────────────────────────────────────

class RunLogger:
    """Writes structured JSONL events for a single benchmark run."""

    def __init__(self, output_dir: str = "logs"):
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
# Custom logging handler to capture RAC compensation events
# ──────────────────────────────────────────────────────────────────────────

class RACEventCapture(logging.Handler):
    """Captures RAC compensation/recovery log messages as structured events."""

    def __init__(self, run_logger: RunLogger):
        super().__init__()
        self.run_logger = run_logger

    def emit(self, record):
        msg = record.getMessage()
        # Only capture interesting RAC events
        if "[COMPENSATION]" in msg or "Retrying" in msg:
            self.run_logger.emit("rac_event", {
                "logger": record.name,
                "level": record.levelname,
                "message": msg,
            })


# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Quick RAC runner test")
    parser.add_argument("--model", default="gemini-2.0-flash", help="LLM model")
    parser.add_argument("--profile", default="mild_production", help="Disruption profile")
    parser.add_argument("--domain", default="travel", help="Task domain")
    parser.add_argument("--task-ids", nargs="+", default=None, help="Specific task IDs to run")
    parser.add_argument("--max-tasks", type=int, default=3, help="Max tasks to run")
    parser.add_argument("--max-difficulty", type=int, default=5, help="Max difficulty")
    parser.add_argument("--min-difficulty", type=int, default=1, help="Min difficulty")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", default="logs", help="Log output directory")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(levelname)s:%(name)s:%(message)s")

    # Quieten noisy loggers
    for noisy in ("httpcore", "httpx", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # Start run logger
    run_log = RunLogger(output_dir=args.output_dir)

    # Attach RAC event capture handler
    rac_handler = RACEventCapture(run_log)
    rac_handler.setLevel(logging.DEBUG)
    for logger_name in ("react_agent_compensation", "react_agent_compensation.langchain_adaptor.agent",
                         "react_agent_compensation.core.recovery_manager"):
        logging.getLogger(logger_name).addHandler(rac_handler)

    print("=" * 60)
    print(" AgentDisruptBench — Quick RAC Runner")
    print("=" * 60)
    print(f"  Run ID:     {run_log.run_id}")
    print(f"  Model:      {args.model}")
    print(f"  Profile:    {args.profile}")
    print(f"  Domain:     {args.domain}")
    print(f"  Difficulty: {args.min_difficulty}–{args.max_difficulty}")
    print(f"  Seed:       {args.seed}")
    print(f"  Logs:       {run_log.run_dir}")
    print()

    # Emit run_started event
    run_log.emit("run_started", {
        "run_id": run_log.run_id,
        "model": args.model,
        "profile": args.profile,
        "domain": args.domain,
        "seed": args.seed,
        "min_difficulty": args.min_difficulty,
        "max_difficulty": args.max_difficulty,
    })

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
        # Filter by min difficulty
        tasks = [t for t in tasks if t.difficulty >= args.min_difficulty]
        tasks = tasks[:args.max_tasks]

    if not tasks:
        print(f"❌ No tasks found matching filters")
        sys.exit(1)

    print(f"Found {len(tasks)} task(s) to run:\n")
    for t in tasks:
        print(f"  [{t.task_id}] D{t.difficulty} — {t.title}")
        print(f"    Tools: {', '.join(t.required_tools)}")
        print(f"    Call depth: {t.expected_tool_call_depth}")
    print()

    run_log.emit("tasks_selected", {
        "count": len(tasks),
        "tasks": [{"id": t.task_id, "title": t.title, "difficulty": t.difficulty,
                    "tools": t.required_tools, "depth": t.expected_tool_call_depth} for t in tasks],
    })

    # Create runner
    runner_config = RunnerConfig(model=args.model, verbose=args.verbose)
    runner = RACRunner(runner_config)
    runner.setup()

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
        per_task_seed = (args.seed ^ hash(task.task_id)) & 0x7FFFFFFF

        run_log.emit("task_started", {
            "task_id": task.task_id,
            "title": task.title,
            "difficulty": task.difficulty,
            "task_type": task.task_type,
            "required_tools": task.required_tools,
            "expected_depth": task.expected_tool_call_depth,
            "profile": args.profile,
            "disruption_seed": per_task_seed,
        })

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

        # Run the task
        start = time.time()
        try:
            agent_output = runner.run_task(task, proxied_tools)
        except Exception as exc:
            agent_output = f"[Runner error: {exc}]"
        elapsed = time.time() - start

        # Gather traces
        traces = trace_collector.get_traces()

        # Log every tool call trace
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
        result = calc.compute(
            task=task,
            traces=traces,
            agent_output=agent_output,
            baseline_result=None,
            agent_id=f"rac_{args.model}",
            profile_name=args.profile,
            seed=args.seed,
            duration_seconds=elapsed,
        )
        results.append(result)

        # Log task result (emit full metrics so show_run.py sees everything)
        run_log.emit("task_completed", {
            "task_id": task.task_id,
            "success": result.success,
            "partial_score": round(result.partial_score, 4),
            "recovery_rate": round(result.recovery_rate, 4),
            "total_tool_calls": result.total_tool_calls,
            "disruptions_encountered": result.disruptions_encountered,
            "duration_seconds": round(elapsed, 2),
            "recovery_strategies": result.recovery_strategies,
            "dominant_strategy": result.dominant_strategy,
            # P1/P2 metrics
            "graceful_giveup": result.graceful_giveup,
            "compensation_count": result.compensation_count,
            "compensation_success_rate": round(result.compensation_success_rate, 4),
            "side_effect_score": round(result.side_effect_score, 4),
            "idempotency_violations": result.idempotency_violations,
            "loop_count": result.loop_count,
            "planning_time_ratio": round(result.planning_time_ratio, 4),
            "handover_detected": result.handover_detected,
            "tool_hallucination_rate": round(result.tool_hallucination_rate, 4),
            "failure_categories": result.failure_categories,
            "agent_output": agent_output[:500],
        })

        # Print results
        status = "✅" if result.success else "❌"
        print(f"\n  {status} Success:          {result.success}")
        print(f"  📊 Partial score:   {result.partial_score:.2f}")
        print(f"  🔄 Recovery rate:   {result.recovery_rate:.2%}")
        print(f"  🛠️  Tool calls:      {result.total_tool_calls}")
        print(f"  💥 Disruptions:     {result.disruptions_encountered}")
        print(f"  ⏱️  Duration:        {elapsed:.1f}s")
        if result.recovery_strategies:
            print(f"  🧠 Strategies:      {result.recovery_strategies}")
        if result.dominant_strategy:
            print(f"  👆 Dominant:        {result.dominant_strategy}")
        print(f"\n  Agent output (first 200 chars):")
        print(f"  {agent_output[:200]}...")

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

    # Emit run_completed
    run_log.emit("run_completed", {
        "total_tasks": len(results),
        "successful": success_count,
        "success_rate": round(success_count / max(len(results), 1), 4),
        "avg_partial_score": round(avg_partial, 4),
        "total_duration_seconds": round(total_time, 2),
    })

    print(f"\n  💡 To analyze: python evaluation/show_run.py --run-id {run_log.run_id}\n")

    run_log.close()
    runner.teardown()


if __name__ == "__main__":
    main()
