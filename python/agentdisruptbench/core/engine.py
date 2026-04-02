"""
AgentDisruptBench — DisruptionEngine
=====================================

File:        engine.py
Purpose:     Core disruption engine that applies configured faults to tool call
             results at runtime. Implements all 20 disruption types defined in
             the AgentDisruptBench taxonomy.

Author:      AgentDisruptBench Contributors
License:     MIT
Created:     2026-03-09
Modified:    2026-03-09

Key Classes / Definitions:
    DisruptionType   : StrEnum of all 20 disruption type identifiers.
    DisruptionConfig : Pydantic v2 model for a single disruption configuration.
    DisruptionEngine : Stateful engine that evaluates configs against tool calls.
                       Thread-safe, seeded for reproducibility, picklable.

Design Notes:
    - The engine iterates configs in order. The FIRST config that fires
      (passes probability + target check) terminates the loop.
    - Stateful disruptions (quota_exhausted, auth_expiry, intermittent,
      flapping, cascading) maintain per-tool call counters.
    - Thread-safe via threading.Lock for all state mutations.
    - Seeded random.Random instance ensures deterministic replay.

Convention:
    Every source file MUST include a header block like this one.
"""

from __future__ import annotations

import copy
import enum
import json
import logging
import random
import threading
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger("agentdisruptbench.engine")


# ---------------------------------------------------------------------------
# Disruption Type Enum
# ---------------------------------------------------------------------------


class DisruptionType(str, enum.Enum):
    """All 20 disruption types in the AgentDisruptBench taxonomy.

    Categories:
        TIMING           — timeout, latency
        HTTP STATUS      — http_429 … http_503
        RESPONSE CONTENT — malformed_json … wrong_data
        BEHAVIORAL       — intermittent … cascading
    """

    TIMEOUT = "timeout"
    LATENCY = "latency"
    HTTP_429 = "http_429"
    HTTP_401 = "http_401"
    HTTP_403 = "http_403"
    HTTP_500 = "http_500"
    HTTP_502 = "http_502"
    HTTP_503 = "http_503"
    MALFORMED_JSON = "malformed_json"
    TRUNCATED = "truncated"
    NULL_RESPONSE = "null_response"
    MISSING_FIELDS = "missing_fields"
    TYPE_MISMATCH = "type_mismatch"
    SCHEMA_DRIFT = "schema_drift"
    WRONG_DATA = "wrong_data"
    INTERMITTENT = "intermittent"
    FLAPPING = "flapping"
    QUOTA_EXHAUSTED = "quota_exhausted"
    AUTH_EXPIRY = "auth_expiry"
    CASCADING = "cascading"


# ---------------------------------------------------------------------------
# Disruption Config (Pydantic v2)
# ---------------------------------------------------------------------------


class DisruptionConfig(BaseModel):
    """Configuration for a single disruption rule.

    Attributes:
        type:               Which disruption type to apply.
        probability:        Probability (0.0–1.0) this disruption fires per
                            eligible call.  Default 1.0.
        target_tools:       If set, only fire for these tool names.
        delay_ms:           Delay in ms (timeout, latency).
        truncation_pct:     Fraction of body to keep (truncated).
        fail_after_n_calls: Threshold for quota_exhausted / auth_expiry.
        fail_every_n:       N-th call fails (intermittent).
        cascade_targets:    Downstream tools affected (cascading).
    """

    type: DisruptionType
    probability: float = Field(default=1.0, ge=0.0, le=1.0)
    target_tools: list[str] | None = None
    delay_ms: int = Field(default=5000)
    truncation_pct: float = Field(default=0.5, ge=0.0, le=1.0)
    fail_after_n_calls: int = Field(default=5)
    fail_every_n: int = Field(default=3)
    cascade_targets: list[str] | None = None


# ---------------------------------------------------------------------------
# HTTP error body templates
# ---------------------------------------------------------------------------

_HTTP_ERROR_BODIES: dict[DisruptionType, dict[str, Any]] = {
    DisruptionType.HTTP_429: {"error": {"code": 429, "message": "Too Many Requests", "retry_after": 30}},
    DisruptionType.HTTP_401: {"error": {"code": 401, "message": "Unauthorized — token expired or invalid"}},
    DisruptionType.HTTP_403: {"error": {"code": 403, "message": "Forbidden — insufficient permissions"}},
    DisruptionType.HTTP_500: {"error": {"code": 500, "message": "Internal Server Error"}},
    DisruptionType.HTTP_502: {"error": {"code": 502, "message": "Bad Gateway — upstream unreachable"}},
    DisruptionType.HTTP_503: {"error": {"code": 503, "message": "Service Unavailable — try again later"}},
}


# ---------------------------------------------------------------------------
# Disruption Engine
# ---------------------------------------------------------------------------


class DisruptionEngine:
    """Applies configured disruptions to tool call results.

    Stateful: tracks per-tool call counts for quota/auth expiry disruptions.
    Thread-safe: uses threading.Lock for all state mutations.
    Reproducible: seeded random for deterministic experiment replay.

    Parameters:
        configs: Ordered list of disruption configs.  First match wins.
        seed:    Random seed for deterministic results (default 42).
    """

    def __init__(self, configs: list[DisruptionConfig], seed: int = 42) -> None:
        self._configs: list[DisruptionConfig] = list(configs)
        self._seed: int = seed
        self._rng: random.Random = random.Random(seed)
        self._lock: threading.Lock = threading.Lock()

        # Per-tool state
        self._call_counts: dict[str, int] = {}
        self._cascade_triggered: set[str] = set()
        self._flap_state: dict[str, bool] = {}
        self._last_disruption: DisruptionType | None = None

    # -- public API --------------------------------------------------------

    def apply(
        self,
        tool_name: str,
        tool_input: dict,
        original_result: Any,
        original_success: bool,
        original_error: str | None,
    ) -> tuple[Any, bool, str | None, DisruptionType | None]:
        """Apply disruptions to a tool call result.

        Returns:
            ``(result, success, error, disruption_type)``
            If no disruption fires, returns originals with ``disruption_type=None``.

        Raises:
            TimeoutError: If a ``timeout`` disruption fires.
        """
        with self._lock:
            self._call_counts.setdefault(tool_name, 0)
            self._call_counts[tool_name] += 1
            call_number = self._call_counts[tool_name]

            # Cascade check first
            if tool_name in self._cascade_triggered:
                self._last_disruption = DisruptionType.CASCADING
                logger.debug(
                    "disruption_fired=cascading tool=%s reason=cascade_downstream",
                    tool_name,
                )
                error_body = json.dumps(
                    {
                        "error": {
                            "code": 503,
                            "message": "Service unavailable — upstream dependency failure (cascade)",
                        }
                    }
                )
                return None, False, error_body, DisruptionType.CASCADING

            # Iterate configs — first match wins
            for cfg in self._configs:
                fired = self._evaluate_config(cfg, tool_name, call_number, original_result)
                if fired is not None:
                    result, success, error, dtype = fired
                    self._last_disruption = dtype
                    return result, success, error, dtype

        self._last_disruption = None
        return original_result, original_success, original_error, None

    def reset(self) -> None:
        """Reset all stateful call counters.  Call between tasks."""
        with self._lock:
            self._call_counts.clear()
            self._cascade_triggered.clear()
            self._flap_state.clear()
            self._last_disruption = None
            logger.debug("engine_reset seed=%d", self._seed)

    @property
    def last_disruption(self) -> DisruptionType | None:
        """Disruption type that fired on the most recent ``apply()``."""
        return self._last_disruption

    # -- pickling (multiprocessing) ----------------------------------------

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        del state["_lock"]
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._lock = threading.Lock()

    # -- internal ----------------------------------------------------------

    def _evaluate_config(
        self,
        cfg: DisruptionConfig,
        tool_name: str,
        call_number: int,
        original_result: Any,
    ) -> tuple[Any, bool, str | None, DisruptionType] | None:
        """Evaluate a single config.  Returns disrupted result or ``None``."""

        # Target filter
        if cfg.target_tools and tool_name not in cfg.target_tools:
            return None

        # Stateful types do their own gating
        _STATEFUL = {
            DisruptionType.INTERMITTENT,
            DisruptionType.FLAPPING,
            DisruptionType.QUOTA_EXHAUSTED,
            DisruptionType.AUTH_EXPIRY,
        }
        if cfg.type not in _STATEFUL:
            if self._rng.random() > cfg.probability:
                return None

        handler = _HANDLERS.get(cfg.type)
        if handler is None:
            logger.warning("unknown_disruption_type=%s", cfg.type)
            return None
        return handler(self, cfg, tool_name, call_number, original_result)


# ---------------------------------------------------------------------------
# Individual disruption handlers
# ---------------------------------------------------------------------------


def _h_timeout(
    eng: DisruptionEngine,
    cfg: DisruptionConfig,
    tool: str,
    call_n: int,
    result: Any,
) -> tuple[Any, bool, str | None, DisruptionType]:
    logger.debug("disruption_fired=timeout tool=%s delay_ms=%d", tool, cfg.delay_ms)
    raise TimeoutError(
        json.dumps(
            {
                "error": {
                    "code": "TIMEOUT",
                    "message": f"Tool '{tool}' timed out after {cfg.delay_ms}ms",
                    "delay_ms": cfg.delay_ms,
                }
            }
        )
    )


def _h_latency(
    eng: DisruptionEngine,
    cfg: DisruptionConfig,
    tool: str,
    call_n: int,
    result: Any,
) -> tuple[Any, bool, str | None, DisruptionType]:
    """Return real result — actual sleep applied by ToolProxy.

    The proxy reads ``DisruptionType.LATENCY`` from the return tuple and
    injects ``cfg.delay_ms`` of sleep before returning to the caller.
    """
    logger.debug("disruption_fired=latency tool=%s delay_ms=%d", tool, cfg.delay_ms)
    return result, True, None, DisruptionType.LATENCY


def _h_http_error(
    eng: DisruptionEngine,
    cfg: DisruptionConfig,
    tool: str,
    call_n: int,
    result: Any,
) -> tuple[Any, bool, str | None, DisruptionType]:
    body = _HTTP_ERROR_BODIES.get(cfg.type, {"error": {"code": 500, "message": "Unknown error"}})
    logger.debug("disruption_fired=%s tool=%s", cfg.type.value, tool)
    return None, False, json.dumps(body), cfg.type


def _h_malformed_json(
    eng: DisruptionEngine,
    cfg: DisruptionConfig,
    tool: str,
    call_n: int,
    result: Any,
) -> tuple[Any, bool, str | None, DisruptionType]:
    try:
        raw = json.dumps(result)
    except (TypeError, ValueError):
        raw = str(result)
    cut = eng._rng.randint(1, max(1, len(raw) - 1)) if len(raw) > 2 else 1
    logger.debug("disruption_fired=malformed_json tool=%s", tool)
    return raw[:cut], False, "Malformed JSON response", DisruptionType.MALFORMED_JSON


def _h_truncated(
    eng: DisruptionEngine,
    cfg: DisruptionConfig,
    tool: str,
    call_n: int,
    result: Any,
) -> tuple[Any, bool, str | None, DisruptionType]:
    try:
        raw = json.dumps(result)
    except (TypeError, ValueError):
        raw = str(result)
    cut = max(1, int(len(raw) * cfg.truncation_pct))
    logger.debug("disruption_fired=truncated tool=%s pct=%.2f", tool, cfg.truncation_pct)
    return raw[:cut], False, "Truncated response", DisruptionType.TRUNCATED


def _h_null_response(
    eng: DisruptionEngine,
    cfg: DisruptionConfig,
    tool: str,
    call_n: int,
    result: Any,
) -> tuple[Any, bool, str | None, DisruptionType]:
    logger.debug("disruption_fired=null_response tool=%s", tool)
    return None, True, None, DisruptionType.NULL_RESPONSE


def _h_missing_fields(
    eng: DisruptionEngine,
    cfg: DisruptionConfig,
    tool: str,
    call_n: int,
    result: Any,
) -> tuple[Any, bool, str | None, DisruptionType] | None:
    if not isinstance(result, dict):
        return None
    out = copy.deepcopy(result)
    keys = list(out.keys())
    if not keys:
        return out, True, None, DisruptionType.MISSING_FIELDS
    n_rm = max(1, eng._rng.randint(int(len(keys) * 0.2), max(1, int(len(keys) * 0.6))))
    for k in eng._rng.sample(keys, min(n_rm, len(keys))):
        del out[k]
    logger.debug("disruption_fired=missing_fields tool=%s", tool)
    return out, True, None, DisruptionType.MISSING_FIELDS


def _h_type_mismatch(
    eng: DisruptionEngine,
    cfg: DisruptionConfig,
    tool: str,
    call_n: int,
    result: Any,
) -> tuple[Any, bool, str | None, DisruptionType] | None:
    if not isinstance(result, dict):
        return None
    out = copy.deepcopy(result)
    keys = list(out.keys())
    if not keys:
        return out, True, None, DisruptionType.TYPE_MISMATCH
    n = max(1, eng._rng.randint(1, max(1, len(keys) // 2)))
    for k in eng._rng.sample(keys, min(n, len(keys))):
        v = out[k]
        if isinstance(v, str):
            out[k] = eng._rng.randint(0, 9999)
        elif isinstance(v, (int, float)):
            out[k] = str(v) + "_corrupted"
        elif isinstance(v, list):
            out[k] = None
        elif isinstance(v, dict):
            out[k] = list(v.keys()) if v else []
        elif v is None:
            out[k] = "null_was_here"
        else:
            out[k] = None
    logger.debug("disruption_fired=type_mismatch tool=%s", tool)
    return out, True, None, DisruptionType.TYPE_MISMATCH


def _h_schema_drift(
    eng: DisruptionEngine,
    cfg: DisruptionConfig,
    tool: str,
    call_n: int,
    result: Any,
) -> tuple[Any, bool, str | None, DisruptionType] | None:
    if not isinstance(result, dict):
        return None
    out = copy.deepcopy(result)
    keys = list(out.keys())
    if not keys:
        return out, True, None, DisruptionType.SCHEMA_DRIFT
    n = max(1, eng._rng.randint(1, max(1, len(keys) // 2)))
    for k in eng._rng.sample(keys, min(n, len(keys))):
        suffix = eng._rng.choice(["_v2", "_v3", "_new", "_updated", "_beta"])
        out[k + suffix] = out.pop(k)
    logger.debug("disruption_fired=schema_drift tool=%s", tool)
    return out, True, None, DisruptionType.SCHEMA_DRIFT


def _h_wrong_data(
    eng: DisruptionEngine,
    cfg: DisruptionConfig,
    tool: str,
    call_n: int,
    result: Any,
) -> tuple[Any, bool, str | None, DisruptionType]:
    def _perturb(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: _perturb(v) for k, v in obj.items()}
        if isinstance(obj, list):
            p = [_perturb(i) for i in obj]
            eng._rng.shuffle(p)
            return p
        if isinstance(obj, bool):
            return not obj
        if isinstance(obj, int):
            return obj + eng._rng.randint(-1000, 1000)
        if isinstance(obj, float):
            return round(obj + eng._rng.uniform(-100.0, 100.0), 2)
        if isinstance(obj, str):
            mods = [
                lambda s: s.upper(),
                lambda s: s + " [MODIFIED]",
                lambda s: s[::-1] if len(s) < 50 else s[:25] + "..." + s[-25:],
                lambda s: f"WRONG_{s}",
            ]
            return eng._rng.choice(mods)(obj)
        return obj

    logger.debug("disruption_fired=wrong_data tool=%s", tool)
    return _perturb(copy.deepcopy(result)), True, None, DisruptionType.WRONG_DATA


def _h_intermittent(
    eng: DisruptionEngine,
    cfg: DisruptionConfig,
    tool: str,
    call_n: int,
    result: Any,
) -> tuple[Any, bool, str | None, DisruptionType] | None:
    if cfg.fail_every_n <= 0 or call_n % cfg.fail_every_n != 0:
        return None
    if eng._rng.random() > cfg.probability:
        return None
    logger.debug("disruption_fired=intermittent tool=%s call=%d", tool, call_n)
    return (
        None,
        False,
        json.dumps({"error": {"code": 500, "message": f"Intermittent failure on call #{call_n}"}}),
        DisruptionType.INTERMITTENT,
    )


def _h_flapping(
    eng: DisruptionEngine,
    cfg: DisruptionConfig,
    tool: str,
    call_n: int,
    result: Any,
) -> tuple[Any, bool, str | None, DisruptionType] | None:
    if eng._rng.random() > cfg.probability:
        return None
    last_fail = eng._flap_state.get(tool, False)
    eng._flap_state[tool] = not last_fail
    if not last_fail:
        logger.debug("disruption_fired=flapping tool=%s state=fail", tool)
        return (
            None,
            False,
            json.dumps({"error": {"code": 503, "message": "Service flapping — currently unavailable"}}),
            DisruptionType.FLAPPING,
        )
    return None


def _h_quota_exhausted(
    eng: DisruptionEngine,
    cfg: DisruptionConfig,
    tool: str,
    call_n: int,
    result: Any,
) -> tuple[Any, bool, str | None, DisruptionType] | None:
    if call_n <= cfg.fail_after_n_calls:
        return None
    logger.debug("disruption_fired=quota_exhausted tool=%s call=%d", tool, call_n)
    return (
        None,
        False,
        json.dumps(
            {
                "error": {
                    "code": 429,
                    "message": f"Quota exhausted after {cfg.fail_after_n_calls} calls",
                    "retry_after": 60,
                }
            }
        ),
        DisruptionType.QUOTA_EXHAUSTED,
    )


def _h_auth_expiry(
    eng: DisruptionEngine,
    cfg: DisruptionConfig,
    tool: str,
    call_n: int,
    result: Any,
) -> tuple[Any, bool, str | None, DisruptionType] | None:
    if call_n <= cfg.fail_after_n_calls:
        return None
    logger.debug("disruption_fired=auth_expiry tool=%s call=%d", tool, call_n)
    return (
        None,
        False,
        json.dumps({"error": {"code": 401, "message": "Authentication token expired"}}),
        DisruptionType.AUTH_EXPIRY,
    )


def _h_cascading(
    eng: DisruptionEngine,
    cfg: DisruptionConfig,
    tool: str,
    call_n: int,
    result: Any,
) -> tuple[Any, bool, str | None, DisruptionType] | None:
    if eng._rng.random() > cfg.probability:
        return None
    if cfg.cascade_targets:
        for t in cfg.cascade_targets:
            eng._cascade_triggered.add(t)
            logger.debug("disruption_cascade_propagated from=%s to=%s", tool, t)
    logger.debug("disruption_fired=cascading tool=%s", tool)
    return (
        None,
        False,
        json.dumps({"error": {"code": 500, "message": f"Service '{tool}' failure — cascading to dependents"}}),
        DisruptionType.CASCADING,
    )


# ---------------------------------------------------------------------------
# Handler dispatch table
# ---------------------------------------------------------------------------

_HANDLERS: dict[DisruptionType, Any] = {
    DisruptionType.TIMEOUT: _h_timeout,
    DisruptionType.LATENCY: _h_latency,
    DisruptionType.HTTP_429: _h_http_error,
    DisruptionType.HTTP_401: _h_http_error,
    DisruptionType.HTTP_403: _h_http_error,
    DisruptionType.HTTP_500: _h_http_error,
    DisruptionType.HTTP_502: _h_http_error,
    DisruptionType.HTTP_503: _h_http_error,
    DisruptionType.MALFORMED_JSON: _h_malformed_json,
    DisruptionType.TRUNCATED: _h_truncated,
    DisruptionType.NULL_RESPONSE: _h_null_response,
    DisruptionType.MISSING_FIELDS: _h_missing_fields,
    DisruptionType.TYPE_MISMATCH: _h_type_mismatch,
    DisruptionType.SCHEMA_DRIFT: _h_schema_drift,
    DisruptionType.WRONG_DATA: _h_wrong_data,
    DisruptionType.INTERMITTENT: _h_intermittent,
    DisruptionType.FLAPPING: _h_flapping,
    DisruptionType.QUOTA_EXHAUSTED: _h_quota_exhausted,
    DisruptionType.AUTH_EXPIRY: _h_auth_expiry,
    DisruptionType.CASCADING: _h_cascading,
}
