"""
AgentDisruptBench — Python SDK
================================

File:        __init__.py (top-level package)
Purpose:     Public API surface for the AgentDisruptBench Python SDK.
             Import convenience classes and functions from sub-packages.

Author:      AgentDisruptBench Contributors
License:     MIT
Created:     2026-03-09
Modified:    2026-03-09

Public API:
    DisruptionEngine     : Core engine applying 20 disruption types.
    DisruptionConfig     : Pydantic config for a single disruption rule.
    DisruptionType       : Enum of all 20 disruption types.
    ToolProxy            : Callable wrapper injecting disruptions.
    TraceCollector       : Thread-safe trace storage with JSONL I/O.
    ToolCallTrace        : Dataclass for one proxied tool call record.
    MetricsCalculator    : Computes resilience metrics from traces.
    BenchmarkResult      : Dataclass holding all metrics for one run.
    BenchmarkRunner      : Top-level harness iterating (task × profile × seed).
    BenchmarkConfig      : Configuration for benchmark suite runs.
    Reporter             : Generates Markdown and JSON reports.
    Evaluator            : Orchestrates a single (task, profile) run.
    TaskRegistry         : Loads and filters task definitions.
    ToolRegistry         : Maps tool names to callables.
    Task, GroundTruth    : Pydantic task schemas.
    BUILTIN_PROFILES     : dict of 9 built-in disruption profiles.
    get_profile          : Retrieve a profile by name.
    load_profiles        : Load profiles from YAML.

Convention:
    Every source file MUST include a header block like this one.
"""

__version__ = "0.1.0"
__author__ = "AgentDisruptBench Contributors"

# Core
from agentdisruptbench.core.engine import (
    DisruptionConfig,
    DisruptionEngine,
    DisruptionType,
)
from agentdisruptbench.core.metrics import BenchmarkResult, MetricsCalculator
from agentdisruptbench.core.profiles import (
    BUILTIN_PROFILES,
    get_profile,
    load_profiles,
)
from agentdisruptbench.core.proxy import ToolProxy
from agentdisruptbench.core.trace import ToolCallTrace, TraceCollector

# Harness
from agentdisruptbench.harness.evaluator import Evaluator
from agentdisruptbench.harness.reporter import Reporter
from agentdisruptbench.harness.runner import BenchmarkConfig, BenchmarkRunner
from agentdisruptbench.tasks.registry import TaskRegistry

# Tasks
from agentdisruptbench.tasks.schemas import GroundTruth, Task

# Tools
from agentdisruptbench.tools.registry import ToolRegistry

__all__ = [
    # Core
    "DisruptionEngine",
    "DisruptionConfig",
    "DisruptionType",
    "ToolProxy",
    "TraceCollector",
    "ToolCallTrace",
    "MetricsCalculator",
    "BenchmarkResult",
    # Profiles
    "BUILTIN_PROFILES",
    "get_profile",
    "load_profiles",
    # Harness
    "BenchmarkRunner",
    "BenchmarkConfig",
    "Evaluator",
    "Reporter",
    # Tasks
    "Task",
    "GroundTruth",
    "TaskRegistry",
    # Tools
    "ToolRegistry",
    # Meta
    "__version__",
]
