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

    # YAML-based configuration (recommended)
    python -m evaluation.run_benchmark --config config/benchmark.yaml --llm-config config/llm/gpt-4o.yaml

    # YAML config with CLI overrides
    python -m evaluation.run_benchmark --config config/benchmark.yaml --llm-config config/llm/gemini-2.5-flash.yaml --max-difficulty 3

Convention:
    Every source file MUST include a header block like this one.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path

# Ensure project root is on path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Auto-load .env file if present
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(project_root, ".env"))
except ImportError:
    pass  # python-dotenv is optional

from agentdisruptbench import (
    BenchmarkConfig,
    BenchmarkRunner,
    TaskRegistry,
    ToolRegistry,
)
from agentdisruptbench.harness.reporter import Reporter

from evaluation.base_runner import RunnerConfig
from evaluation.config_loader import load_benchmark_config, load_llm_config
from evaluation.run_logger import RunLogger

# Registry of available runners
RUNNER_REGISTRY = {
    "simple": "evaluation.runners.simple_runner:SimpleRunner",
    "openai": "evaluation.runners.openai_runner:OpenAIRunner",
    "langchain": "evaluation.runners.langchain_runner:LangChainRunner",
    "autogen": "evaluation.runners.autogen_runner:AutoGenRunner",
    "crewai": "evaluation.runners.crewai_runner:CrewAIRunner",
    "rac": "evaluation.runners.rac_runner:RACRunner",
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
        print("   Install the required dependencies for this runner.")
        sys.exit(1)


def _make_run_dir_name(runner_name: str, model: str, task_ids: list[str] | None,
                       domains: list[str] | None) -> str:
    """Generate a descriptive run directory name.

    Format: YYYYMMDD_HHMM_{runner}_{model}_{scope}
    Examples:
        20260331_1258_langchain_gpt5mini_adversarial_retail
        20260331_1300_simple_gpt4o_retail
        20260331_1305_openai_gpt4o_all_100tasks
    """
    from datetime import datetime, timezone
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    # Sanitise model name: gpt-5-mini → gpt5mini
    safe_model = re.sub(r'[^a-zA-Z0-9]', '', model)

    # Build scope segment
    if task_ids and len(task_ids) <= 4:
        scope = "_".join(task_ids)
    elif task_ids:
        scope = f"{len(task_ids)}tasks"
    elif domains:
        scope = "_".join(domains)
    else:
        scope = "all"

    # Truncate scope to keep folder names readable
    if len(scope) > 60:
        scope = scope[:57] + "etc"

    return f"{ts}_{runner_name}_{safe_model}_{scope}"


def _emit_run_logs(results, runner_name, model, task_registry, config,
                   run_dir: Path | None = None):
    """Emit structured JSONL run logs for show_run.py.

    Creates one run log per (profile × seed) combination so each
    is independently viewable via ``python evaluation/show_run.py``.

    If *run_dir* is provided, all logs go into that directory.
    Otherwise falls back to creating timestamped folders under ``logs/``.
    """
    from itertools import groupby

    # Group results by (profile, seed)
    def keyfunc(r):
        return (r.profile_name, r.seed)

    sorted_results = sorted(results, key=keyfunc)

    run_ids = []
    for (profile, seed), group in groupby(sorted_results, key=keyfunc):
        group_results = list(group)
        run_log = RunLogger(run_dir=run_dir) if run_dir else RunLogger(output_dir="logs")
        run_ids.append(run_log.run_id)

        # Emit run_started
        run_log.emit("run_started", {
            "run_id": run_log.run_id,
            "runner": runner_name,
            "model": model,
            "profile": profile,
            "domain": config.domains[0] if config.domains else "all",
            "seed": seed,
            "min_difficulty": 1,
            "max_difficulty": config.max_difficulty,
        })

        # Emit tasks_selected
        task_list = []
        for r in group_results:
            # Find task metadata from registry
            matched = [
                t for t in task_registry.all_tasks()
                if t.task_id == r.task_id
            ]
            if matched:
                t = matched[0]
                task_list.append({
                    "id": t.task_id,
                    "title": t.title,
                    "difficulty": t.difficulty,
                    "tools": t.required_tools,
                    "depth": t.expected_tool_call_depth,
                })
            else:
                task_list.append({
                    "id": r.task_id,
                    "title": r.task_id,
                    "difficulty": r.task_difficulty,
                    "tools": [],
                    "depth": 0,
                })

        run_log.emit("tasks_selected", {
            "count": len(group_results),
            "tasks": task_list,
        })

        # Emit per-task events
        total_duration = 0.0
        for r in group_results:
            # Find task for metadata
            matched = [
                t for t in task_registry.all_tasks()
                if t.task_id == r.task_id
            ]
            task_meta = matched[0] if matched else None

            run_log.emit("task_started", {
                "task_id": r.task_id,
                "title": (
                    task_meta.title if task_meta
                    else r.task_id
                ),
                "difficulty": (
                    task_meta.difficulty if task_meta
                    else r.task_difficulty
                ),
                "task_type": (
                    task_meta.task_type if task_meta
                    else "standard"
                ),
                "required_tools": (
                    task_meta.required_tools if task_meta
                    else []
                ),
                "expected_depth": (
                    task_meta.expected_tool_call_depth
                    if task_meta else 0
                ),
                "profile": profile,
            })

            # Emit tool call traces
            for trace in r.traces:
                run_log.emit("tool_call", {
                    "call_id": trace.call_id,
                    "tool_name": trace.tool_name,
                    "success": trace.observed_success,
                    "disruption_type": (
                        str(trace.disruption_fired)
                        if trace.disruption_fired else None
                    ),
                    "latency_ms": round(
                        trace.observed_latency_ms, 1
                    ),
                    "error": trace.error,
                })

            # Emit task_completed
            run_log.emit("task_completed", {
                "task_id": r.task_id,
                "success": r.success,
                "partial_score": round(r.partial_score, 4),
                "recovery_rate": round(r.recovery_rate, 4),
                "total_tool_calls": r.total_tool_calls,
                "disruptions_encountered": r.disruptions_encountered,
                "duration_seconds": round(r.duration_seconds, 2),
                "recovery_strategies": r.recovery_strategies,
                "dominant_strategy": r.dominant_strategy,
                "graceful_giveup": r.graceful_giveup,
                "compensation_count": r.compensation_count,
                "compensation_success_rate": round(
                    r.compensation_success_rate, 4
                ),
                "side_effect_score": (
                    r.side_effect_score
                    if r.side_effect_score else None
                ),
                "idempotency_violations": (
                    r.idempotency_violations
                    if r.idempotency_violations else None
                ),
                "loop_count": r.loop_count,
                "planning_time_ratio": round(
                    r.planning_time_ratio, 4
                ),
                "handover_detected": r.handover_detected,
                "tool_hallucination_rate": round(
                    r.tool_hallucination_rate, 4
                ),
                "failure_categories": r.failure_categories,
                "agent_output": str(r.agent_output)[:500],
            })

            total_duration += r.duration_seconds

        # Emit run_completed
        success_count = sum(1 for r in group_results if r.success)
        avg_partial = (
            sum(r.partial_score for r in group_results)
            / max(len(group_results), 1)
        )
        run_log.emit("run_completed", {
            "total_tasks": len(group_results),
            "successful": success_count,
            "success_rate": round(
                success_count / max(len(group_results), 1), 4
            ),
            "avg_partial_score": round(avg_partial, 4),
            "total_duration_seconds": round(total_duration, 2),
        })

        run_log.close()

    return run_ids


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

  # Only run specific tasks
  python -m evaluation.run_benchmark -t adversarial_retail_001,adversarial_retail_002 --runner rac

  # YAML-based configuration (recommended)
  python -m evaluation.run_benchmark --config config/benchmark.yaml --llm-config config/llm/gpt-4o.yaml

  # YAML config with CLI overrides
  python -m evaluation.run_benchmark -c config/benchmark.yaml -l config/llm/gemini-2.5-flash.yaml --max-difficulty 3
        """,
    )

    # YAML configuration (recommended)
    parser.add_argument(
        "--config", "-c",
        default=None,
        help="Path to benchmark YAML config (e.g. config/benchmark.yaml)",
    )
    parser.add_argument(
        "--llm-config", "-l",
        default=None,
        help="Path to LLM YAML config (e.g. config/llm/gpt-4o.yaml)",
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
    parser.add_argument(
        "--max-tokens", type=int, default=None,
        help="Max tokens per response (default: provider limit)",
    )
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

    # Task selection
    parser.add_argument(
        "--tasks", "-t",
        nargs="+",
        default=None,
        help="Run specific task IDs (comma-separated or multiple args)",
    )
    parser.add_argument(
        "--task-dir",
        default=None,
        help="Directory of task YAML files (overrides built-in tasks)",
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

    # ── Step 0: Load YAML configs (if provided) ──────────────────
    # YAML values serve as defaults; CLI args override them.
    yaml_bench = None
    yaml_llm = None

    if args.config:
        print(f"\n[0a] Loading benchmark config: {args.config}")
        yaml_bench = load_benchmark_config(args.config)

    if args.llm_config:
        print(f"\n[0b] Loading LLM config: {args.llm_config}")
        yaml_llm = load_llm_config(args.llm_config)

    # ── Resolve runner name ──────────────────────────────────────
    # Priority: CLI --runner > benchmark.yaml runner > LLM config inferred > default
    cli_provided_runner = "--runner" in sys.argv or "-r" in sys.argv
    if cli_provided_runner:
        runner_name = args.runner
    elif yaml_bench and yaml_bench.runner:
        runner_name = yaml_bench.runner
    elif yaml_llm:
        runner_name = yaml_llm.infer_runner()
    else:
        runner_name = args.runner  # argparse default

    # ── Resolve model / runner config ────────────────────────────
    cli_provided_model = "--model" in sys.argv or "-m" in sys.argv
    if yaml_llm:
        runner_config = yaml_llm.to_runner_config()
        # CLI overrides for model
        if cli_provided_model:
            runner_config.model = args.model
        if args.api_key:
            runner_config.api_key = args.api_key
        if "--temperature" in sys.argv:
            runner_config.temperature = args.temperature
        if "--max-tokens" in sys.argv:
            runner_config.max_tokens = args.max_tokens
        if "--max-steps" in sys.argv:
            runner_config.max_steps = args.max_steps
        runner_config.verbose = args.verbose
    else:
        runner_config = RunnerConfig(
            model=args.model,
            api_key=args.api_key,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            max_steps=args.max_steps,
            verbose=args.verbose,
        )

    # ── Resolve benchmark config ─────────────────────────────────
    cli_provided_profiles = "--profiles" in sys.argv or "-p" in sys.argv
    cli_provided_domains = "--domains" in sys.argv or "-d" in sys.argv
    cli_provided_seeds = "--seeds" in sys.argv or "-s" in sys.argv
    cli_provided_difficulty = "--max-difficulty" in sys.argv
    cli_provided_output = "--output-dir" in sys.argv or "-o" in sys.argv
    cli_provided_tasks = "--tasks" in sys.argv or "-t" in sys.argv

    if cli_provided_tasks and args.tasks:
        cli_tasks = []
        for t in args.tasks:
            cli_tasks.extend([x.strip() for x in t.split(",") if x.strip()])
    else:
        cli_tasks = None

    profiles = (
        args.profiles if cli_provided_profiles
        else yaml_bench.profiles if yaml_bench
        else args.profiles
    )
    domains = (
        args.domains if cli_provided_domains
        else yaml_bench.domains if yaml_bench
        else args.domains
    )
    tasks = (
        cli_tasks if cli_provided_tasks
        else yaml_bench.tasks if yaml_bench and hasattr(yaml_bench, 'tasks')
        else cli_tasks
    )
    max_difficulty = (
        args.max_difficulty if cli_provided_difficulty
        else yaml_bench.max_difficulty if yaml_bench
        else args.max_difficulty
    )
    seeds = (
        args.seeds if cli_provided_seeds
        else yaml_bench.seeds if yaml_bench
        else args.seeds
    )
    output_dir = (
        args.output_dir if cli_provided_output
        else yaml_bench.output_dir if yaml_bench
        else args.output_dir
    )
    verbose = args.verbose or (yaml_bench.verbose if yaml_bench else False)
    runner_config.verbose = verbose

    # Load runner
    print(f"\n[1/5] Loading runner: {runner_name}")
    runner = _load_runner(runner_name, runner_config)
    runner.setup()
    agent_id = args.agent_id or f"{runner_name}_{runner_config.model}"
    print(f"  → Runner: {type(runner).__name__}")
    print(f"  → Model:  {runner_config.model}")

    # Load tasks and tools
    print("\n[2/5] Loading tasks and tools...")
    if args.task_dir:
        task_registry = TaskRegistry.from_directory(args.task_dir)
    else:
        task_registry = TaskRegistry.from_builtin()
    tool_registry = ToolRegistry.from_mock_tools()
    print(f"  → {len(task_registry)} tasks loaded")
    print(f"  → {len(tool_registry)} tools available")

    # Configure benchmark
    print("\n[3/5] Configuring benchmark...")
    config = BenchmarkConfig(
        profiles=profiles,
        seeds=seeds,
        domains=domains,
        task_ids=tasks,
        max_difficulty=max_difficulty,
        agent_id=agent_id,
        output_dir=output_dir,
    )
    print(f"  → Profiles:       {config.profiles}")
    print(f"  → Domains:        {domains or 'all'}")
    if tasks:
        print(f"  → Task IDs:       {len(tasks)} tasks specified")
    print(f"  → Max difficulty:  {max_difficulty}")
    print(f"  → Seeds:           {seeds}")

    # Create consolidated run directory
    run_dir_name = _make_run_dir_name(
        runner_name, runner_config.model, tasks, domains,
    )
    run_dir = Path("runs") / run_dir_name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n  📁 Run directory: {run_dir}")

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

    # Generate report — all outputs go into the consolidated run directory
    print("\n[5/5] Generating report...")
    reporter = Reporter(output_dir=str(run_dir))
    paths = reporter.generate(results)
    for name, path in paths.items():
        print(f"  → {name}: {path}")

    # Emit structured JSONL logs into the SAME run directory
    run_ids = _emit_run_logs(
        results, runner_name, runner_config.model,
        task_registry, config,
        run_dir=run_dir,
    )
    print(f"  → run_log: {run_dir}/run_log.jsonl")

    # Summary
    print("\n" + "=" * 60)
    success_count = sum(1 for r in results if r.success)
    print(f"  Runner:       {runner_name} ({runner_config.model})")
    print(f"  Total runs:   {len(results)}")
    print(f"  Successful:   {success_count}/{len(results)}")
    print(f"  Success rate: {success_count / max(len(results), 1) * 100:.1f}%")
    print(f"  Duration:     {elapsed:.1f}s")
    print(f"  Output:       {run_dir}")

    # Runner stats
    stats = runner.stats
    if stats.get("total_tokens"):
        print(f"  Tokens used:  {stats['total_tokens']}")
    if stats.get("total_api_calls"):
        print(f"  API calls:    {stats['total_api_calls']}")

    print("=" * 60)

    # Hint for viewing logs
    print(f"\n  💡 View latest: python evaluation/show_run.py")
    print(f"  💡 View specific: python evaluation/show_run.py --run-id {run_dir_name}\n")

    # Cleanup
    runner.teardown()


if __name__ == "__main__":
    main()
