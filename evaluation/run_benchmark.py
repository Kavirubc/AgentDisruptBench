#!/usr/bin/env python3
"""
AgentDisruptBench — Benchmark CLI
====================================

File:        run_benchmark.py
Purpose:     Command-line entry point for running the benchmark with any
             registered runner. Users can specify framework, profiles,
             domains, difficulty, and model via CLI args.

Author:      AgentDisruptBench Contributors
License:     MIT
Created:     2026-03-09
Modified:    2026-03-09

Usage:
    # Simple baseline (no LLM needed)
    python -m evaluation.run_benchmark --runner simple --profiles clean mild_production

    # OpenAI GPT-4o
    python -m evaluation.run_benchmark --runner openai --model gpt-4o --profiles clean hostile_environment

    # LangChain with a specific model
    python -m evaluation.run_benchmark --runner langchain --model gpt-4o-mini --max-difficulty 3

    # AutoGen
    python -m evaluation.run_benchmark --runner autogen --model gpt-4o --domains retail travel

    # CrewAI
    python -m evaluation.run_benchmark --runner crewai --model gpt-4o --profiles clean

Convention:
    Every source file MUST include a header block like this one.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

# Ensure project root is on path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from agentdisruptbench import (
    BenchmarkConfig,
    BenchmarkRunner,
    TaskRegistry,
    ToolRegistry,
)
from agentdisruptbench.harness.reporter import Reporter
from evaluation.base_runner import RunnerConfig


# Registry of available runners
RUNNER_REGISTRY = {
    "simple": "evaluation.runners.simple_runner:SimpleRunner",
    "openai": "evaluation.runners.openai_runner:OpenAIRunner",
    "langchain": "evaluation.runners.langchain_runner:LangChainRunner",
    "autogen": "evaluation.runners.autogen_runner:AutoGenRunner",
    "crewai": "evaluation.runners.crewai_runner:CrewAIRunner",
}


def _load_runner(name: str, runner_config: RunnerConfig):
    """Dynamically load a runner class by registry name."""
    if name not in RUNNER_REGISTRY:
        print(f"❌ Unknown runner: {name}")
        print(f"   Available runners: {', '.join(RUNNER_REGISTRY.keys())}")
        sys.exit(1)

    module_path, class_name = RUNNER_REGISTRY[name].rsplit(":", 1)

    try:
        import importlib
        module = importlib.import_module(module_path)
        runner_cls = getattr(module, class_name)
        return runner_cls(runner_config)
    except ImportError as exc:
        print(f"❌ Failed to import runner '{name}': {exc}")
        print(f"   Install the required dependencies for this runner.")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        prog="run_benchmark",
        description="AgentDisruptBench — Run the benchmark with any framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Simple baseline (no LLM needed)
  python -m evaluation.run_benchmark --runner simple

  # OpenAI GPT-4o on retail tasks
  python -m evaluation.run_benchmark --runner openai --model gpt-4o --domains retail

  # Full hostile environment test
  python -m evaluation.run_benchmark --runner openai --profiles clean hostile_environment --max-difficulty 3
        """,
    )

    # Runner selection
    parser.add_argument(
        "--runner", "-r",
        choices=list(RUNNER_REGISTRY.keys()),
        default="simple",
        help="Framework runner to use (default: simple)",
    )

    # LLM configuration
    parser.add_argument("--model", "-m", default="gpt-4o", help="LLM model name")
    parser.add_argument("--api-key", default=None, help="API key (or set env var)")
    parser.add_argument("--temperature", type=float, default=0.0, help="LLM temperature")
    parser.add_argument("--max-tokens", type=int, default=4096, help="Max tokens per response")
    parser.add_argument("--max-steps", type=int, default=20, help="Max agent loop steps")

    # Benchmark configuration
    parser.add_argument(
        "--profiles", "-p",
        nargs="+",
        default=["clean", "mild_production", "hostile_environment"],
        help="Disruption profiles to evaluate",
    )
    parser.add_argument(
        "--domains", "-d",
        nargs="+",
        default=None,
        help="Filter to specific domains (default: all)",
    )
    parser.add_argument(
        "--max-difficulty",
        type=int, default=5,
        help="Max task difficulty level (1-5)",
    )
    parser.add_argument(
        "--seeds", "-s",
        nargs="+", type=int, default=[42],
        help="Random seeds for reproducibility",
    )

    # Output
    parser.add_argument("--output-dir", "-o", default="results", help="Output directory")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print agent reasoning")
    parser.add_argument("--agent-id", default=None, help="Agent identifier for reports")

    args = parser.parse_args()

    # Banner
    print("=" * 60)
    print(" AgentDisruptBench — Benchmark Runner")
    print("=" * 60)

    # Build runner config
    runner_config = RunnerConfig(
        model=args.model,
        api_key=args.api_key,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_steps=args.max_steps,
        verbose=args.verbose,
    )

    # Load runner
    print(f"\n[1/5] Loading runner: {args.runner}")
    runner = _load_runner(args.runner, runner_config)
    runner.setup()
    agent_id = args.agent_id or f"{args.runner}_{args.model}"
    print(f"  → Runner: {type(runner).__name__}")
    print(f"  → Model:  {args.model}")

    # Load tasks and tools
    print("\n[2/5] Loading tasks and tools...")
    task_registry = TaskRegistry.from_builtin()
    tool_registry = ToolRegistry.from_mock_tools()
    print(f"  → {len(task_registry)} tasks loaded")
    print(f"  → {len(tool_registry)} tools available")

    # Configure benchmark
    print("\n[3/5] Configuring benchmark...")
    config = BenchmarkConfig(
        profiles=args.profiles,
        seeds=args.seeds,
        domains=args.domains,
        max_difficulty=args.max_difficulty,
        agent_id=agent_id,
        output_dir=args.output_dir,
    )
    print(f"  → Profiles:       {config.profiles}")
    print(f"  → Domains:        {args.domains or 'all'}")
    print(f"  → Max difficulty:  {args.max_difficulty}")
    print(f"  → Seeds:           {args.seeds}")

    # Run benchmark
    print("\n[4/5] Running benchmark...")
    start = time.time()

    bench_runner = BenchmarkRunner(
        agent_fn=runner,
        task_registry=task_registry,
        tool_registry=tool_registry,
        config=config,
    )
    results = bench_runner.run_all()
    elapsed = time.time() - start

    print(f"  → {len(results)} runs completed in {elapsed:.1f}s")

    # Generate report
    print("\n[5/5] Generating report...")
    reporter = Reporter(output_dir=args.output_dir)
    paths = reporter.generate(results)
    for name, path in paths.items():
        print(f"  → {name}: {path}")

    # Summary
    print("\n" + "=" * 60)
    success_count = sum(1 for r in results if r.success)
    print(f"  Runner:       {args.runner} ({args.model})")
    print(f"  Total runs:   {len(results)}")
    print(f"  Successful:   {success_count}/{len(results)}")
    print(f"  Success rate: {success_count / max(len(results), 1) * 100:.1f}%")
    print(f"  Duration:     {elapsed:.1f}s")

    # Runner stats
    stats = runner.stats
    if stats.get("total_tokens"):
        print(f"  Tokens used:  {stats['total_tokens']}")
    if stats.get("total_api_calls"):
        print(f"  API calls:    {stats['total_api_calls']}")

    print("=" * 60)

    # Cleanup
    runner.teardown()


if __name__ == "__main__":
    main()
