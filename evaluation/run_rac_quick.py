#!/usr/bin/env python3
"""
Quick RAC Runner Test — Run a few tasks with the RAC framework.

Usage:
    # Ensure GEMINI_API_KEY or GOOGLE_API_KEY is set
    python evaluation/run_rac_quick.py

    # With OpenAI
    python evaluation/run_rac_quick.py --model gpt-4o
"""

import os
import sys
import time

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


def main():
    parser = argparse.ArgumentParser(description="Quick RAC runner test")
    parser.add_argument("--model", default="gemini-2.0-flash", help="LLM model")
    parser.add_argument("--profile", default="mild_production", help="Disruption profile")
    parser.add_argument("--domain", default="travel", help="Task domain")
    parser.add_argument("--max-tasks", type=int, default=3, help="Max tasks to run")
    parser.add_argument("--max-difficulty", type=int, default=2, help="Max difficulty")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print(" AgentDisruptBench — Quick RAC Runner")
    print("=" * 60)
    print(f"  Model:      {args.model}")
    print(f"  Profile:    {args.profile}")
    print(f"  Domain:     {args.domain}")
    print(f"  Max tasks:  {args.max_tasks}")
    print(f"  Difficulty: ≤{args.max_difficulty}")
    print(f"  Seed:       {args.seed}")
    print()

    # Load tasks and tools
    task_registry = TaskRegistry.from_builtin()
    tool_registry = ToolRegistry.from_mock_tools()

    # Filter tasks
    tasks = task_registry.filter(
        domain=args.domain,
        max_difficulty=args.max_difficulty,
    )[:args.max_tasks]

    if not tasks:
        print(f"❌ No tasks found for domain={args.domain}, max_difficulty={args.max_difficulty}")
        sys.exit(1)

    print(f"Found {len(tasks)} task(s) to run:\n")
    for t in tasks:
        print(f"  [{t.task_id}] D{t.difficulty} — {t.title}")
    print()

    # Create runner
    runner_config = RunnerConfig(
        model=args.model,
        verbose=args.verbose,
    )
    runner = RACRunner(runner_config)
    runner.setup()

    # Load disruption profile
    profile = get_profile(args.profile)

    results = []
    calc = MetricsCalculator()

    for i, task in enumerate(tasks, 1):
        print(f"\n{'─' * 60}")
        print(f"Task {i}/{len(tasks)}: {task.task_id}")
        print(f"  Title:      {task.title}")
        print(f"  Difficulty: {task.difficulty}")
        print(f"  Type:       {task.task_type}")
        print(f"  Profile:    {args.profile}")
        print(f"{'─' * 60}")

        # Create disruption engine for this run
        engine = DisruptionEngine(disruptions=profile, seed=args.seed)
        trace_collector = TraceCollector()

        # Create tool proxies (with disruption injection)
        proxied_tools = {}
        for tool_name in task.required_tools:
            raw_fn = tool_registry.get(tool_name)
            proxy = ToolProxy(
                tool_name=tool_name,
                real_fn=raw_fn,
                engine=engine,
                collector=trace_collector,
            )
            proxied_tools[tool_name] = proxy

        # Run the task
        start = time.time()
        try:
            agent_output = runner.run_task(task, proxied_tools)
        except Exception as exc:
            agent_output = f"[Runner error: {exc}]"
        elapsed = time.time() - start

        # Compute metrics
        traces = trace_collector.traces
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

        # Print results
        print(f"\n  ✅ Success:          {result.success}")
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
    print(f"  Total:        {len(results)}")
    print(f"  Successful:   {success_count}/{len(results)}")
    print(f"  Success rate: {success_count / max(len(results), 1) * 100:.1f}%")
    avg_partial = sum(r.partial_score for r in results) / max(len(results), 1)
    print(f"  Avg partial:  {avg_partial:.2f}")
    print(f"{'=' * 60}")

    runner.teardown()


if __name__ == "__main__":
    main()
