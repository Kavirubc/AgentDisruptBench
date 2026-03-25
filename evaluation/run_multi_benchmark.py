#!/usr/bin/env python3
"""
AgentDisruptBench — Multi-Model Benchmark Runner
===================================================

File:        run_multi_benchmark.py
Purpose:     Run benchmarks across multiple LLM configs in parallel.
             Different providers (Gemini, OpenAI) run concurrently;
             same-provider models are serialized to avoid rate limits.

Author:      AgentDisruptBench Contributors
License:     MIT
Created:     2026-03-25
Modified:    2026-03-25

Usage:
    # Gemini + OpenAI in parallel
    python -m evaluation.run_multi_benchmark \
      --llm-configs config/llm/gemini-2.5-flash.yaml config/llm/gpt-5-mini.yaml

    # Multiple OpenAI models (serialized within provider)
    python -m evaluation.run_multi_benchmark \
      --llm-configs config/llm/gpt-4o.yaml config/llm/gpt-4o-mini.yaml

Convention:
    Every source file MUST include a header block like this one.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
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

from evaluation.config_loader import load_llm_config
from evaluation.llm_factory import detect_provider


# ─── PROVIDER GROUPING ───────────────────────────────────────────────────────


def group_by_provider(
    config_paths: list[str],
) -> dict[str, list[str]]:
    """Group LLM config paths by provider.

    Returns:
        Dict mapping provider name to list of config paths.
        e.g. {"gemini": ["config/llm/gemini-2.5-flash.yaml"],
              "openai": ["config/llm/gpt-4o.yaml", "config/llm/gpt-5-mini.yaml"]}
    """
    groups: dict[str, list[str]] = defaultdict(list)
    for path in config_paths:
        cfg = load_llm_config(path)
        provider = detect_provider(cfg.model)
        groups[provider].append(path)
    return dict(groups)


# ─── SINGLE RUN EXECUTOR ─────────────────────────────────────────────────────


def run_single_benchmark(
    llm_config_path: str,
    benchmark_config_path: str | None,
    extra_args: list[str],
    python_bin: str,
) -> dict:
    """Run a single benchmark as a subprocess.

    Returns:
        Dict with model, provider, config_path, returncode, output,
        duration, and any run_ids found in the output.
    """
    cfg = load_llm_config(llm_config_path)
    provider = detect_provider(cfg.model)

    cmd = [
        python_bin, "-m", "evaluation.run_benchmark",
        "--llm-config", llm_config_path,
    ]
    if benchmark_config_path:
        cmd.extend(["--config", benchmark_config_path])

    cmd.extend(extra_args)

    print(f"  🚀 [{provider}] Starting: {cfg.model}")

    start = time.time()
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=project_root,
    )
    elapsed = time.time() - start

    # Extract run IDs from output
    run_ids = []
    for line in result.stdout.splitlines():
        if "run_log: logs/" in line:
            # e.g. "  → run_log: logs/20260325_015751_b584c7/run_log.jsonl"
            parts = line.split("logs/")
            if len(parts) >= 2:
                rid = parts[1].split("/")[0]
                run_ids.append(rid)

    status = "✅" if result.returncode == 0 else "❌"
    print(
        f"  {status} [{provider}] {cfg.model} "
        f"completed in {elapsed:.1f}s"
    )

    if result.returncode != 0 and result.stderr:
        # Show last 5 lines of stderr for debugging
        err_lines = result.stderr.strip().splitlines()[-5:]
        for line in err_lines:
            print(f"     [stderr] {line}")

    return {
        "model": cfg.model,
        "provider": provider,
        "config_path": llm_config_path,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "duration": elapsed,
        "run_ids": run_ids,
    }


# ─── PROVIDER GROUP EXECUTOR ─────────────────────────────────────────────────


def run_provider_group(
    provider: str,
    config_paths: list[str],
    benchmark_config_path: str | None,
    extra_args: list[str],
    python_bin: str,
) -> list[dict]:
    """Run all models for a single provider sequentially.

    This prevents rate-limit conflicts for same-provider models.
    """
    results = []
    for config_path in config_paths:
        result = run_single_benchmark(
            llm_config_path=config_path,
            benchmark_config_path=benchmark_config_path,
            extra_args=extra_args,
            python_bin=python_bin,
        )
        results.append(result)
    return results


# ─── MAIN ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        prog="run_multi_benchmark",
        description=(
            "AgentDisruptBench — Run benchmarks across"
            " multiple models in parallel"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Gemini + OpenAI in parallel
  python -m evaluation.run_multi_benchmark \\
    --llm-configs config/llm/gemini-2.5-flash.yaml \\
                  config/llm/gpt-5-mini.yaml

  # With benchmark config
  python -m evaluation.run_multi_benchmark \\
    --config config/benchmark.yaml \\
    --llm-configs config/llm/gemini-2.5-flash.yaml \\
                  config/llm/gpt-5-mini.yaml

  # Override profiles and difficulty
  python -m evaluation.run_multi_benchmark \\
    --llm-configs config/llm/gpt-4o.yaml \\
                  config/llm/gpt-4o-mini.yaml \\
    --profiles clean hostile_environment \\
    --max-difficulty 3
        """,
    )

    parser.add_argument(
        "--llm-configs",
        nargs="+",
        required=True,
        help="Paths to LLM YAML configs to benchmark",
    )
    parser.add_argument(
        "--config", "-c",
        default=None,
        help="Path to benchmark YAML config",
    )

    # Pass-through args for each benchmark run
    parser.add_argument(
        "--profiles", "-p",
        nargs="+",
        default=None,
        help="Disruption profiles to evaluate",
    )
    parser.add_argument(
        "--domains", "-d",
        nargs="+",
        default=None,
        help="Filter to specific domains",
    )
    parser.add_argument(
        "--max-difficulty",
        type=int,
        default=None,
        help="Max task difficulty level",
    )
    parser.add_argument(
        "--seeds", "-s",
        nargs="+",
        type=int,
        default=None,
        help="Random seeds",
    )
    parser.add_argument(
        "--runner", "-r",
        default=None,
        help="Force a specific runner for all configs",
    )
    parser.add_argument(
        "--no-compare",
        action="store_true",
        help="Skip auto-comparison after runs complete",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
    )

    args = parser.parse_args()

    # Banner
    print("=" * 60)
    print(" AgentDisruptBench — Multi-Model Benchmark")
    print("=" * 60)

    # Group configs by provider
    groups = group_by_provider(args.llm_configs)
    print(f"\n  Configs:   {len(args.llm_configs)}")
    print(f"  Providers: {len(groups)}")
    for provider, paths in groups.items():
        models = []
        for p in paths:
            cfg = load_llm_config(p)
            models.append(cfg.model)
        print(
            f"    {provider}: {', '.join(models)}"
            f" ({'parallel' if len(groups) > 1 else 'sequential'})"
        )
    print()

    # Build extra args to pass through to run_benchmark
    extra_args: list[str] = []
    if args.profiles:
        extra_args.extend(["--profiles"] + args.profiles)
    if args.domains:
        extra_args.extend(["--domains"] + args.domains)
    if args.max_difficulty is not None:
        extra_args.extend(["--max-difficulty", str(args.max_difficulty)])
    if args.seeds:
        extra_args.extend(
            ["--seeds"] + [str(s) for s in args.seeds]
        )
    if args.runner:
        extra_args.extend(["--runner", args.runner])
    if args.verbose:
        extra_args.append("--verbose")

    # Detect python binary
    python_bin = sys.executable

    # Execute: parallel across providers, sequential within
    print("Starting benchmark runs...\n")
    start = time.time()
    all_results: list[dict] = []

    if len(groups) == 1:
        # Single provider — everything sequential
        provider, paths = next(iter(groups.items()))
        results = run_provider_group(
            provider, paths, args.config, extra_args, python_bin,
        )
        all_results.extend(results)
    else:
        # Multiple providers — parallel across providers
        with ThreadPoolExecutor(
            max_workers=len(groups),
        ) as executor:
            futures = {}
            for provider, paths in groups.items():
                fut = executor.submit(
                    run_provider_group,
                    provider, paths,
                    args.config, extra_args, python_bin,
                )
                futures[fut] = provider

            for fut in as_completed(futures):
                provider = futures[fut]
                try:
                    results = fut.result()
                    all_results.extend(results)
                except Exception as exc:
                    print(
                        f"  ❌ [{provider}] Failed: {exc}"
                    )

    elapsed = time.time() - start

    # Summary
    print(f"\n{'=' * 60}")
    print(" Multi-Benchmark Summary")
    print(f"{'=' * 60}")
    print(f"  Total time:    {elapsed:.1f}s")
    print(f"  Models run:    {len(all_results)}")
    successful = sum(
        1 for r in all_results if r["returncode"] == 0
    )
    print(f"  Successful:    {successful}/{len(all_results)}")
    print()

    for r in all_results:
        status = "✅" if r["returncode"] == 0 else "❌"
        print(
            f"  {status} {r['model']:20s} "
            f"({r['provider']:8s}) "
            f"{r['duration']:.1f}s"
            f"  runs: {', '.join(r['run_ids']) or 'none'}"
        )

    print(f"{'=' * 60}")

    # Auto-compare if we have run IDs from multiple models
    all_run_ids = []
    for r in all_results:
        all_run_ids.extend(r["run_ids"])

    if not args.no_compare and len(all_run_ids) >= 2:
        print("\n  📊 Auto-comparing runs...\n")
        compare_cmd = [
            python_bin, "-m", "evaluation.compare_runs",
        ] + all_run_ids
        subprocess.run(compare_cmd, cwd=project_root)
    elif all_run_ids:
        print(
            "\n  💡 Compare manually: "
            f"python evaluation/compare_runs.py "
            f"{' '.join(all_run_ids)}\n"
        )


if __name__ == "__main__":
    main()
