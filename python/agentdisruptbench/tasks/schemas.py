"""
AgentDisruptBench — Task Schemas
=================================

File:        schemas.py
Purpose:     Pydantic v2 data models for the task system:  Task, ToolSchema,
             and GroundTruth.

Author:      AgentDisruptBench Contributors
License:     MIT
Created:     2026-03-09
Modified:    2026-03-09

Key Models:
    ToolSchema  : Describes a tool's interface.
    GroundTruth : Expected outcomes, required calls, rubric.
    Task        : Complete task definition loaded from YAML.

Convention:
    Every source file MUST include a header block like this one.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ToolSchema(BaseModel):
    """Describes a single tool available to the agent.

    Attributes:
        name:        Unique tool identifier.
        description: Human-readable description.
        parameters:  JSON Schema for accepted parameters.
        returns:     JSON Schema for return value.
        domain:      Domain this tool belongs to.
        tags:        Freeform tags for filtering.
    """

    name: str
    description: str
    parameters: dict = Field(default_factory=dict)
    returns: dict = Field(default_factory=dict)
    domain: str
    tags: list[str] = Field(default_factory=list)


class GroundTruth(BaseModel):
    """Ground-truth data for evaluating a task outcome.

    Attributes:
        expected_outcome:           NL description of success.
        required_tool_calls:        Tool names that must be called.
        forbidden_tool_calls:       Tool names that must NOT be called.
        correct_final_answer:       Exact expected answer (or None).
        evaluation_rubric:          Criterion → weight (should sum ≈ 1.0).
        disruption_sensitive_tools: Tools where failure is most impactful.
        recovery_actions:           Expected recovery behaviours.
        trap_description:           For adversarial tasks: describes the trap
                                    the agent should avoid.
        impossibility_reason:       For impossible tasks: why the task has no
                                    valid solution.
    """

    expected_outcome: str
    required_tool_calls: list[str]
    forbidden_tool_calls: list[str] = Field(default_factory=list)
    correct_final_answer: Any | None = None
    evaluation_rubric: dict[str, float] = Field(default_factory=dict)
    disruption_sensitive_tools: list[str] = Field(default_factory=list)
    recovery_actions: list[str] = Field(default_factory=list)
    trap_description: str | None = None
    impossibility_reason: str | None = None


class Task(BaseModel):
    """Complete task definition for a benchmark run.

    Attributes:
        task_id:                  Unique identifier (e.g. "retail_001").
        title:                    Short human-readable title.
        description:              Full prompt given to the agent.
        domain:                   Domain category.
        difficulty:               1–5 difficulty rating.
        task_type:                One of 'standard', 'adversarial', 'impossible'.
        required_tools:           Tool names needed to solve the task.
        expected_tool_call_depth: Expected tool calls under clean conditions.
        ground_truth:             Evaluation ground truth.
        metadata:                 Arbitrary additional metadata.
        source:                   Origin (synthetic, tau_bench, realm_bench).
    """

    task_id: str
    title: str
    description: str
    domain: str
    difficulty: int = Field(ge=1, le=5)
    task_type: Literal["standard", "adversarial", "impossible"] = "standard"
    required_tools: list[str]
    expected_tool_call_depth: int
    ground_truth: GroundTruth
    metadata: dict = Field(default_factory=dict)
    source: str = "synthetic"
