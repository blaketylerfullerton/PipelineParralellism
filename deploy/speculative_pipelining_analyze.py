#!/usr/bin/env python3
"""Summarize empirical speculative-pipelining runs.

Input CSV schema:
  config,r_ms,seed,tps[,draft_accept_pct]

The config names should be no-spec, vanilla-spec, and pipelined-spec. The
script writes median TPS and 95% bootstrap CIs per RTT cell, then applies the
experiment spec's kill/confirm/ambiguous criteria.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from speculative_pipelining import bootstrap_median_ci, relative_gain  # noqa: E402


CONFIGS = ("no-spec", "vanilla-spec", "pipelined-spec")
DEFAULT_MODEL_DECISION = ROOT / "logs" / "speculative-pipelining" / "model" / "decision.json"


def _load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        missing = {"config", "r_ms", "seed", "tps"} - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"{path} is missing columns: {', '.join(sorted(missing))}")
        for row in reader:
            if row["config"] not in CONFIGS:
                raise SystemExit(f"unknown config {row['config']!r}; expected one of {', '.join(CONFIGS)}")
            rows.append(
                {
                    "config": row["config"],
                    "r_ms": float(row["r_ms"]),
                    "seed": row["seed"],
                    "tps": float(row["tps"]),
                    "draft_accept_pct": float(row["draft_accept_pct"]) if row.get("draft_accept_pct") else None,
                }
            )
    return rows


def _summarize(rows: list[dict], *, bootstrap_samples: int, seed: int) -> list[dict]:
    grouped: dict[tuple[float, str], list[float]] = defaultdict(list)
    acceptance: dict[tuple[float, str], list[float]] = defaultdict(list)
    for row in rows:
        key = (row["r_ms"], row["config"])
        grouped[key].append(row["tps"])
        if row["draft_accept_pct"] is not None:
            acceptance[key].append(row["draft_accept_pct"])

    summaries = []
    for (r_ms, config), values in sorted(grouped.items()):
        med, lo, hi = bootstrap_median_ci(values, samples=bootstrap_samples, seed=seed)
        accept_values = acceptance.get((r_ms, config), [])
        summaries.append(
            {
                "r_ms": r_ms,
                "config": config,
                "runs": len(values),
                "median_tps": med,
                "ci95_low": lo,
                "ci95_high": hi,
                "median_draft_accept_pct": (
                    bootstrap_median_ci(accept_values, samples=bootstrap_samples, seed=seed)[0]
                    if accept_values
                    else None
                ),
            }
        )
    return summaries


def _load_model_threshold(path: Path) -> float | None:
    if not path.exists():
        return None
    with path.open() as f:
        payload = json.load(f)
    decision = payload.get("decision", {})
    if decision.get("outcome") != "confirmed":
        return None
    return decision.get("first_confirming_latency_ms")


def _threshold_matches(observed_r_ms: float, predicted_r_ms: float) -> bool:
    if observed_r_ms == predicted_r_ms == 0:
        return True
    if observed_r_ms <= 0 or predicted_r_ms <= 0:
        return False
    return max(observed_r_ms, predicted_r_ms) / min(observed_r_ms, predicted_r_ms) <= 2.0


def _decide(
    summaries: list[dict],
    *,
    confirm_gain: float,
    max_realistic_latency_ms: float,
    model_threshold_r_ms: float | None,
) -> dict:
    by_r: dict[float, dict[str, dict]] = defaultdict(dict)
    for item in summaries:
        by_r[item["r_ms"]][item["config"]] = item

    comparable = []
    for r_ms, configs in sorted(by_r.items()):
        if r_ms > max_realistic_latency_ms:
            continue
        if all(name in configs for name in CONFIGS):
            pipe = configs["pipelined-spec"]["median_tps"]
            no_spec = configs["no-spec"]["median_tps"]
            vanilla = configs["vanilla-spec"]["median_tps"]
            comparable.append(
                {
                    "r_ms": r_ms,
                    "gain_vs_no_spec": relative_gain(pipe, no_spec),
                    "gain_vs_vanilla": relative_gain(pipe, vanilla),
                    "pipelined_tps": pipe,
                    "no_spec_tps": no_spec,
                    "vanilla_spec_tps": vanilla,
                }
            )

    if not comparable:
        raise SystemExit("no RTT cells contain all three configs")

    best = max(comparable, key=lambda row: row["gain_vs_no_spec"])
    confirming = []
    for row in comparable:
        suffix = [candidate for candidate in comparable if candidate["r_ms"] >= row["r_ms"]]
        if all(
            candidate["gain_vs_no_spec"] >= confirm_gain
            and candidate["gain_vs_vanilla"] >= confirm_gain
            for candidate in suffix
        ):
            confirming.append(row)
    if confirming:
        first = min(confirming, key=lambda row: row["r_ms"])
        if model_threshold_r_ms is None:
            return {
                "outcome": "ambiguous",
                "reason": (
                    "pipelined spec clears the TPS threshold, but no confirmed "
                    "Stage 0 threshold is available for the required ~2x check"
                ),
                "first_confirming_r_ms": first["r_ms"],
                "predicted_confirming_r_ms": None,
                "prediction_match": None,
                "best": best,
            }
        prediction_match = _threshold_matches(first["r_ms"], model_threshold_r_ms)
        if not prediction_match:
            return {
                "outcome": "ambiguous",
                "reason": (
                    "pipelined spec clears the TPS threshold, but the observed "
                    "threshold does not match the Stage 0 prediction within ~2x"
                ),
                "first_confirming_r_ms": first["r_ms"],
                "predicted_confirming_r_ms": model_threshold_r_ms,
                "prediction_match": False,
                "best": best,
            }
        return {
            "outcome": "confirmed",
            "reason": (
                "pipelined spec beats both baselines by the configured threshold "
                "and matches the Stage 0 prediction within ~2x"
            ),
            "first_confirming_r_ms": first["r_ms"],
            "predicted_confirming_r_ms": model_threshold_r_ms,
            "prediction_match": True,
            "best": best,
        }
    if any(row["pipelined_tps"] > row["no_spec_tps"] for row in comparable):
        return {
            "outcome": "ambiguous",
            "reason": "pipelined spec wins somewhere, but below the configured threshold",
            "first_confirming_r_ms": None,
            "predicted_confirming_r_ms": model_threshold_r_ms,
            "prediction_match": None,
            "best": best,
        }
    return {
        "outcome": "killed",
        "reason": "pipelined spec never beats no-spec in the measured realistic RTT range",
        "first_confirming_r_ms": None,
        "predicted_confirming_r_ms": model_threshold_r_ms,
        "prediction_match": None,
        "best": best,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze speculative-pipelining empirical results")
    parser.add_argument("runs_csv", help="CSV with config,r_ms,seed,tps[,draft_accept_pct]")
    parser.add_argument("--out-dir", default=str(ROOT / "logs" / "speculative-pipelining" / "empirical"))
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--confirm-gain", type=float, default=0.20)
    parser.add_argument("--max-realistic-latency-ms", type=float, default=500.0)
    parser.add_argument(
        "--model-decision-json",
        default=str(DEFAULT_MODEL_DECISION),
        help="Stage 0 analytical decision.json used for the required ~2x threshold check",
    )
    args = parser.parse_args()

    rows = _load_rows(Path(args.runs_csv))
    summaries = _summarize(rows, bootstrap_samples=args.bootstrap_samples, seed=args.seed)
    model_threshold = _load_model_threshold(Path(args.model_decision_json))
    decision = _decide(
        summaries,
        confirm_gain=args.confirm_gain,
        max_realistic_latency_ms=args.max_realistic_latency_ms,
        model_threshold_r_ms=model_threshold,
    )

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "summary.csv").open("w", newline="") as f:
        fieldnames = ["r_ms", "config", "runs", "median_tps", "ci95_low", "ci95_high", "median_draft_accept_pct"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)
    (out_dir / "decision.json").write_text(
        json.dumps({"decision": decision, "summary": summaries}, indent=2) + "\n"
    )

    best = decision["best"]
    print(f"outcome: {decision['outcome']}")
    print(f"reason:  {decision['reason']}")
    if decision["first_confirming_r_ms"] is not None:
        print(f"R*:      {decision['first_confirming_r_ms']:.1f} ms")
    print(f"best R:  {best['r_ms']:.1f} ms")
    print(f"best gain vs no-spec: {best['gain_vs_no_spec'] * 100:.1f}%")
    print(f"best gain vs vanilla: {best['gain_vs_vanilla'] * 100:.1f}%")
    print(f"wrote:   {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
