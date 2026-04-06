"""
AgentDisruptBench — Unit Tests: Engine
=======================================

File:        test_engine.py
Purpose:     Tests for DisruptionEngine, all 20 disruption handlers,
             state management, and pickling support.

Author:      AgentDisruptBench Contributors
License:     MIT
Created:     2026-03-09
Modified:    2026-03-09

Convention:
    Every source file MUST include a header block like this one.
"""

from __future__ import annotations

import pickle

import pytest
from agentdisruptbench.core.engine import (
    DisruptionConfig,
    DisruptionEngine,
    DisruptionType,
)


class TestDisruptionType:
    """Tests for the DisruptionType enum."""

    def test_all_types_exist(self):
        assert len(DisruptionType) == 20

    def test_type_values(self):
        assert DisruptionType.TIMEOUT.value == "timeout"
        assert DisruptionType.CASCADING.value == "cascading"


class TestDisruptionConfig:
    """Tests for the DisruptionConfig model."""

    def test_default_values(self):
        cfg = DisruptionConfig(type=DisruptionType.TIMEOUT)
        assert cfg.probability == 1.0
        assert cfg.delay_ms == 5000
        assert cfg.target_tools is None

    def test_custom_values(self):
        cfg = DisruptionConfig(
            type=DisruptionType.HTTP_429,
            probability=0.5,
            target_tools=["search"],
        )
        assert cfg.probability == 0.5
        assert cfg.target_tools == ["search"]


class TestDisruptionEngine:
    """Tests for the DisruptionEngine."""

    def test_clean_passthrough(self):
        """No configs → no disruption."""
        engine = DisruptionEngine(configs=[])
        result, success, error, dtype = engine.apply("tool", {}, {"data": 1}, True, None)
        assert result == {"data": 1}
        assert success is True
        assert error is None
        assert dtype is None

    def test_timeout_raises(self):
        """Timeout config raises TimeoutError."""
        engine = DisruptionEngine(configs=[DisruptionConfig(type=DisruptionType.TIMEOUT, probability=1.0)])
        with pytest.raises(TimeoutError):
            engine.apply("tool", {}, {"data": 1}, True, None)

    def test_http_429(self):
        """HTTP 429 returns error body."""
        engine = DisruptionEngine(configs=[DisruptionConfig(type=DisruptionType.HTTP_429, probability=1.0)])
        result, success, error, dtype = engine.apply("tool", {}, {}, True, None)
        assert success is False
        assert dtype == DisruptionType.HTTP_429
        assert "429" in error

    def test_malformed_json(self):
        """Malformed JSON truncates the response."""
        engine = DisruptionEngine(configs=[DisruptionConfig(type=DisruptionType.MALFORMED_JSON, probability=1.0)])
        result, success, error, dtype = engine.apply("tool", {}, {"key": "value"}, True, None)
        assert success is False
        assert dtype == DisruptionType.MALFORMED_JSON

    def test_null_response(self):
        """Null response returns None."""
        engine = DisruptionEngine(configs=[DisruptionConfig(type=DisruptionType.NULL_RESPONSE, probability=1.0)])
        result, success, error, dtype = engine.apply("tool", {}, {"data": 1}, True, None)
        assert result is None
        assert dtype == DisruptionType.NULL_RESPONSE

    def test_missing_fields(self):
        """Missing fields removes keys from dict."""
        engine = DisruptionEngine(configs=[DisruptionConfig(type=DisruptionType.MISSING_FIELDS, probability=1.0)])
        original = {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5}
        result, success, error, dtype = engine.apply("tool", {}, original, True, None)
        assert dtype == DisruptionType.MISSING_FIELDS
        assert len(result) < len(original)

    def test_wrong_data(self):
        """Wrong data perturbs values."""
        engine = DisruptionEngine(configs=[DisruptionConfig(type=DisruptionType.WRONG_DATA, probability=1.0)])
        result, success, error, dtype = engine.apply("tool", {}, {"count": 10, "name": "test"}, True, None)
        assert dtype == DisruptionType.WRONG_DATA
        assert result != {"count": 10, "name": "test"}

    def test_target_filter(self):
        """Config with target_tools only fires for matching tools."""
        engine = DisruptionEngine(
            configs=[
                DisruptionConfig(
                    type=DisruptionType.HTTP_500,
                    probability=1.0,
                    target_tools=["targeted_tool"],
                )
            ]
        )
        # Non-targeted tool: no disruption
        _, success, _, dtype = engine.apply("other_tool", {}, {}, True, None)
        assert dtype is None

        # Targeted tool: disruption fires
        _, success, _, dtype = engine.apply("targeted_tool", {}, {}, True, None)
        assert dtype == DisruptionType.HTTP_500

    def test_quota_exhausted(self):
        """Quota exhaustion fires after N calls."""
        engine = DisruptionEngine(
            configs=[
                DisruptionConfig(
                    type=DisruptionType.QUOTA_EXHAUSTED,
                    fail_after_n_calls=2,
                )
            ]
        )
        # Calls 1-2: OK
        _, s1, _, d1 = engine.apply("tool", {}, {}, True, None)
        _, s2, _, d2 = engine.apply("tool", {}, {}, True, None)
        assert d1 is None
        assert d2 is None

        # Call 3: quota exhausted
        _, s3, _, d3 = engine.apply("tool", {}, {}, True, None)
        assert d3 == DisruptionType.QUOTA_EXHAUSTED

    def test_intermittent(self):
        """Intermittent fails every Nth call."""
        engine = DisruptionEngine(
            configs=[
                DisruptionConfig(
                    type=DisruptionType.INTERMITTENT,
                    fail_every_n=3,
                    probability=1.0,
                )
            ]
        )
        results = []
        for _ in range(6):
            _, _, _, dtype = engine.apply("tool", {}, {}, True, None)
            results.append(dtype)

        # Calls 3 and 6 should fire (every 3rd)
        assert results[2] == DisruptionType.INTERMITTENT
        assert results[5] == DisruptionType.INTERMITTENT
        assert results[0] is None
        assert results[1] is None

    def test_cascading(self):
        """Cascading marks downstream tools."""
        engine = DisruptionEngine(
            configs=[
                DisruptionConfig(
                    type=DisruptionType.CASCADING,
                    probability=1.0,
                    target_tools=["upstream"],
                    cascade_targets=["downstream"],
                )
            ]
        )
        # Trigger cascade on upstream
        _, _, _, dtype = engine.apply("upstream", {}, {}, True, None)
        assert dtype == DisruptionType.CASCADING

        # Now downstream is affected
        _, success, _, dtype = engine.apply("downstream", {}, {}, True, None)
        assert dtype == DisruptionType.CASCADING
        assert success is False

    def test_reset(self):
        """Reset clears all state."""
        engine = DisruptionEngine(
            configs=[
                DisruptionConfig(
                    type=DisruptionType.QUOTA_EXHAUSTED,
                    fail_after_n_calls=1,
                )
            ]
        )
        engine.apply("tool", {}, {}, True, None)
        engine.apply("tool", {}, {}, True, None)  # Should fire
        engine.reset()
        _, _, _, dtype = engine.apply("tool", {}, {}, True, None)
        assert dtype is None  # Reset cleared counters

    def test_pickle_roundtrip(self):
        """Engine survives pickle serialisation."""
        engine = DisruptionEngine(configs=[DisruptionConfig(type=DisruptionType.HTTP_500, probability=1.0)])
        pickled = pickle.dumps(engine)
        restored = pickle.loads(pickled)
        _, success, _, dtype = restored.apply("tool", {}, {}, True, None)
        assert dtype == DisruptionType.HTTP_500

    def test_deterministic_with_seed(self):
        """Same seed produces same disruption sequence."""
        configs = [DisruptionConfig(type=DisruptionType.WRONG_DATA, probability=0.5)]
        e1 = DisruptionEngine(configs=configs, seed=99)
        e2 = DisruptionEngine(configs=configs, seed=99)

        results1 = [e1.apply("t", {}, {"v": 1}, True, None) for _ in range(10)]
        results2 = [e2.apply("t", {}, {"v": 1}, True, None) for _ in range(10)]

        dtypes1 = [r[3] for r in results1]
        dtypes2 = [r[3] for r in results2]
        assert dtypes1 == dtypes2
