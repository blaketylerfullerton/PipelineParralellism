from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import Iterable, Sequence


@dataclass(frozen=True)
class ModelInputs:
    verifier_ms: float
    draft_ms: float
    alpha: float
    k: int
    latency_ms: float


@dataclass(frozen=True)
class ThroughputPoint:
    k: int
    latency_ms: float
    no_spec_tps: float
    vanilla_spec_tps: float
    pipelined_spec_tps: float
    expected_tokens_per_round: float

    @property
    def pipelined_gain_vs_no_spec(self) -> float:
        return relative_gain(self.pipelined_spec_tps, self.no_spec_tps)

    @property
    def pipelined_gain_vs_vanilla(self) -> float:
        return relative_gain(self.pipelined_spec_tps, self.vanilla_spec_tps)


@dataclass(frozen=True)
class Decision:
    outcome: str
    reason: str
    first_confirming_latency_ms: float | None
    best_latency_ms: float
    best_gain_vs_no_spec: float
    best_gain_vs_vanilla: float


def relative_gain(candidate: float, baseline: float) -> float:
    if baseline <= 0:
        raise ValueError("baseline must be positive")
    return candidate / baseline - 1.0


def predict_point(inputs: ModelInputs) -> ThroughputPoint:
    if inputs.verifier_ms <= 0:
        raise ValueError("verifier_ms must be positive")
    if inputs.draft_ms <= 0:
        raise ValueError("draft_ms must be positive")
    if not 0 <= inputs.alpha <= 1:
        raise ValueError("alpha must be in [0, 1]")
    if inputs.k <= 0:
        raise ValueError("k must be positive")
    if inputs.latency_ms < 0:
        raise ValueError("latency_ms must be non-negative")

    expected_tokens = 1.0 + inputs.alpha * inputs.k
    no_spec_round_ms = inputs.verifier_ms + inputs.latency_ms
    vanilla_round_ms = inputs.draft_ms * inputs.k + inputs.verifier_ms + inputs.latency_ms
    pipelined_round_ms = max(inputs.draft_ms * inputs.k, inputs.verifier_ms + inputs.latency_ms)

    return ThroughputPoint(
        k=inputs.k,
        latency_ms=inputs.latency_ms,
        no_spec_tps=1000.0 / no_spec_round_ms,
        vanilla_spec_tps=1000.0 * expected_tokens / vanilla_round_ms,
        pipelined_spec_tps=1000.0 * expected_tokens / pipelined_round_ms,
        expected_tokens_per_round=expected_tokens,
    )


def predict_grid(
    *,
    verifier_ms: float,
    draft_ms: float,
    alpha_by_k: dict[int, float],
    latencies_ms: Iterable[float],
) -> list[ThroughputPoint]:
    points: list[ThroughputPoint] = []
    for k in sorted(alpha_by_k):
        for latency_ms in latencies_ms:
            points.append(
                predict_point(
                    ModelInputs(
                        verifier_ms=verifier_ms,
                        draft_ms=draft_ms,
                        alpha=alpha_by_k[k],
                        k=k,
                        latency_ms=float(latency_ms),
                    )
                )
            )
    return points


def decide_direction(
    points: Sequence[ThroughputPoint],
    *,
    max_realistic_latency_ms: float = 500.0,
    confirm_gain: float = 0.20,
) -> Decision:
    if not points:
        raise ValueError("points must not be empty")

    realistic = [p for p in points if p.latency_ms <= max_realistic_latency_ms]
    if not realistic:
        raise ValueError("no points at or below max_realistic_latency_ms")

    best = max(realistic, key=lambda p: p.pipelined_gain_vs_no_spec)
    confirming = []
    for k in sorted({p.k for p in realistic}):
        by_k = sorted((p for p in realistic if p.k == k), key=lambda p: p.latency_ms)
        for point in by_k:
            suffix = [p for p in by_k if p.latency_ms >= point.latency_ms]
            if all(
                p.pipelined_gain_vs_no_spec >= confirm_gain
                and p.pipelined_gain_vs_vanilla >= confirm_gain
                for p in suffix
            ):
                confirming.append(point)
    if confirming:
        first = min(confirming, key=lambda p: p.latency_ms)
        return Decision(
            outcome="confirmed",
            reason=(
                "pipelined spec clears the configured gain threshold versus "
                "both no-spec and vanilla spec inside the realistic latency range"
            ),
            first_confirming_latency_ms=first.latency_ms,
            best_latency_ms=best.latency_ms,
            best_gain_vs_no_spec=best.pipelined_gain_vs_no_spec,
            best_gain_vs_vanilla=best.pipelined_gain_vs_vanilla,
        )

    any_win = any(p.pipelined_spec_tps > p.no_spec_tps for p in realistic)
    outcome = "ambiguous" if any_win else "killed"
    reason = (
        "pipelined spec wins somewhere, but not by the configured threshold"
        if any_win
        else "pipelined spec never beats no-spec inside the realistic latency range"
    )
    return Decision(
        outcome=outcome,
        reason=reason,
        first_confirming_latency_ms=None,
        best_latency_ms=best.latency_ms,
        best_gain_vs_no_spec=best.pipelined_gain_vs_no_spec,
        best_gain_vs_vanilla=best.pipelined_gain_vs_vanilla,
    )


def bootstrap_median_ci(
    values: Sequence[float],
    *,
    samples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    if not values:
        raise ValueError("values must not be empty")
    if samples <= 0:
        raise ValueError("samples must be positive")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")

    import random

    rng = random.Random(seed)
    n = len(values)
    boot = []
    for _ in range(samples):
        boot.append(median(values[rng.randrange(n)] for _ in range(n)))
    boot.sort()
    lower_q = (1.0 - confidence) / 2.0
    upper_q = 1.0 - lower_q
    lower = boot[min(samples - 1, max(0, int(lower_q * samples)))]
    upper = boot[min(samples - 1, max(0, int(upper_q * samples) - 1))]
    return median(values), lower, upper
