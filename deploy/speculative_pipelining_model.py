#!/usr/bin/env python3
"""Analytical gate for the speculative-pipelining WAN experiment.

The experiment spec says to run this before touching the two-machine setup:
if the simple timing model predicts no crossover at realistic RTT, kill the
hypothesis on paper and avoid a week of orchestration work.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from speculative_pipelining import decide_direction, predict_grid  # noqa: E402


def _parse_float_list(text: str) -> list[float]:
    values = [float(part.strip()) for part in text.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one number")
    return values


def _parse_alpha_by_k(text: str) -> dict[int, float]:
    result: dict[int, float] = {}
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            key, value = item.split(":", 1)
            k = int(key)
            alpha = float(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError("expected K:ALPHA pairs, e.g. 2:0.70,4:0.55") from exc
        if k <= 0:
            raise argparse.ArgumentTypeError("k must be positive")
        if not 0 <= alpha <= 1:
            raise argparse.ArgumentTypeError("alpha must be in [0, 1]")
        result[k] = alpha
    if not result:
        raise argparse.ArgumentTypeError("expected at least one K:ALPHA pair")
    return result


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_svg(path: Path, rows: list[dict], *, title: str) -> None:
    """Write a dependency-free line chart for quick README/report drops."""
    width, height = 900, 520
    margin_left, margin_right, margin_top, margin_bottom = 70, 30, 42, 62
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    max_x = max(float(r["latency_ms"]) for r in rows) or 1.0
    max_y = max(
        float(r[key])
        for r in rows
        for key in ["no_spec_tps", "vanilla_spec_tps", "pipelined_spec_tps"]
    ) or 1.0

    def x(value: float) -> float:
        return margin_left + plot_w * value / max_x

    def y(value: float) -> float:
        return margin_top + plot_h * (1.0 - value / max_y)

    def polyline(k: int, key: str) -> str:
        points = [r for r in rows if int(r["k"]) == k]
        points.sort(key=lambda r: float(r["latency_ms"]))
        return " ".join(f"{x(float(r['latency_ms'])):.1f},{y(float(r[key])):.1f}" for r in points)

    colors = {
        "no_spec_tps": "#2f5d62",
        "vanilla_spec_tps": "#b85c38",
        "pipelined_spec_tps": "#6c63ff",
    }
    labels = {
        "no_spec_tps": "no-spec",
        "vanilla_spec_tps": "vanilla spec",
        "pipelined_spec_tps": "pipelined spec",
    }
    ks = sorted({int(r["k"]) for r in rows})

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfbf8"/>',
        f'<text x="{margin_left}" y="26" font-family="Arial, sans-serif" font-size="18" fill="#222">{title}</text>',
        f'<line x1="{margin_left}" y1="{margin_top + plot_h}" x2="{margin_left + plot_w}" y2="{margin_top + plot_h}" stroke="#444"/>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_h}" stroke="#444"/>',
        f'<text x="{width / 2}" y="{height - 18}" text-anchor="middle" font-family="Arial, sans-serif" font-size="13" fill="#333">one-way activation latency R (ms)</text>',
        f'<text x="18" y="{height / 2}" transform="rotate(-90 18 {height / 2})" text-anchor="middle" font-family="Arial, sans-serif" font-size="13" fill="#333">predicted TPS</text>',
    ]

    for tick in range(0, 6):
        tx = max_x * tick / 5
        ty = max_y * tick / 5
        parts.append(f'<line x1="{x(tx):.1f}" y1="{margin_top + plot_h}" x2="{x(tx):.1f}" y2="{margin_top + plot_h + 5}" stroke="#444"/>')
        parts.append(f'<text x="{x(tx):.1f}" y="{margin_top + plot_h + 22}" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="#444">{tx:.0f}</text>')
        parts.append(f'<line x1="{margin_left - 5}" y1="{y(ty):.1f}" x2="{margin_left}" y2="{y(ty):.1f}" stroke="#444"/>')
        parts.append(f'<text x="{margin_left - 10}" y="{y(ty) + 4:.1f}" text-anchor="end" font-family="Arial, sans-serif" font-size="11" fill="#444">{ty:.1f}</text>')

    dash_by_k = {k: dash for k, dash in zip(ks, ["", "8 5", "2 4", "12 5 2 5"])}
    for k in ks:
        for key in ["no_spec_tps", "vanilla_spec_tps", "pipelined_spec_tps"]:
            dash = f' stroke-dasharray="{dash_by_k[k]}"' if dash_by_k[k] else ""
            parts.append(
                f'<polyline points="{polyline(k, key)}" fill="none" stroke="{colors[key]}" '
                f'stroke-width="2.5"{dash}/>'
            )

    legend_x, legend_y = width - 240, 58
    for idx, key in enumerate(["no_spec_tps", "vanilla_spec_tps", "pipelined_spec_tps"]):
        ly = legend_y + idx * 22
        parts.append(f'<line x1="{legend_x}" y1="{ly}" x2="{legend_x + 34}" y2="{ly}" stroke="{colors[key]}" stroke-width="3"/>')
        parts.append(f'<text x="{legend_x + 44}" y="{ly + 4}" font-family="Arial, sans-serif" font-size="12" fill="#333">{labels[key]}</text>')
    if len(ks) > 1:
        parts.append(f'<text x="{legend_x}" y="{legend_y + 82}" font-family="Arial, sans-serif" font-size="11" fill="#555">line dashes distinguish k values: {", ".join(map(str, ks))}</text>')
    parts.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Predict whether speculative pipelining can beat no-spec over WAN")
    parser.add_argument("--verifier-ms", type=float, required=True, help="Measured verifier forward time T_v in ms")
    parser.add_argument("--draft-ms", type=float, required=True, help="Measured draft-model one-token time T_d in ms")
    parser.add_argument(
        "--alpha-by-k",
        type=_parse_alpha_by_k,
        required=True,
        help="Draft acceptance rates by k, e.g. 2:0.70,4:0.55,6:0.50",
    )
    parser.add_argument(
        "--latencies-ms",
        type=_parse_float_list,
        default=_parse_float_list("0,25,50,100,200,300,500"),
        help="One-way activation latencies R in ms",
    )
    parser.add_argument(
        "--rtts-ms",
        type=_parse_float_list,
        default=None,
        help="Optional RTTs in ms. If set, converts to one-way R=RTT/2 and ignores --latencies-ms.",
    )
    parser.add_argument("--max-realistic-latency-ms", type=float, default=500.0)
    parser.add_argument("--confirm-gain", type=float, default=0.20)
    parser.add_argument("--out-dir", default=str(ROOT / "logs" / "speculative-pipelining" / "model"))
    args = parser.parse_args()

    latencies = [value / 2.0 for value in args.rtts_ms] if args.rtts_ms is not None else args.latencies_ms
    points = predict_grid(
        verifier_ms=args.verifier_ms,
        draft_ms=args.draft_ms,
        alpha_by_k=args.alpha_by_k,
        latencies_ms=latencies,
    )
    decision = decide_direction(
        points,
        max_realistic_latency_ms=args.max_realistic_latency_ms,
        confirm_gain=args.confirm_gain,
    )

    rows = []
    for point in points:
        rows.append(
            {
                "k": point.k,
                "latency_ms": f"{point.latency_ms:.3f}",
                "expected_tokens_per_round": f"{point.expected_tokens_per_round:.6f}",
                "no_spec_tps": f"{point.no_spec_tps:.6f}",
                "vanilla_spec_tps": f"{point.vanilla_spec_tps:.6f}",
                "pipelined_spec_tps": f"{point.pipelined_spec_tps:.6f}",
                "pipelined_gain_vs_no_spec": f"{point.pipelined_gain_vs_no_spec:.6f}",
                "pipelined_gain_vs_vanilla": f"{point.pipelined_gain_vs_vanilla:.6f}",
            }
        )

    out_dir = Path(args.out_dir).resolve()
    _write_csv(out_dir / "prediction.csv", rows)
    _write_svg(out_dir / "prediction.svg", rows, title="Speculative pipelining analytical prediction")
    (out_dir / "decision.json").write_text(
        json.dumps(
            {
                "inputs": {
                    "verifier_ms": args.verifier_ms,
                    "draft_ms": args.draft_ms,
                    "alpha_by_k": args.alpha_by_k,
                    "latencies_ms": latencies,
                    "max_realistic_latency_ms": args.max_realistic_latency_ms,
                    "confirm_gain": args.confirm_gain,
                },
                "decision": decision.__dict__,
            },
            indent=2,
        )
        + "\n"
    )

    print(f"outcome: {decision.outcome}")
    print(f"reason:  {decision.reason}")
    if decision.first_confirming_latency_ms is not None:
        print(f"R*:      {decision.first_confirming_latency_ms:.1f} ms")
    print(f"best R:  {decision.best_latency_ms:.1f} ms")
    print(f"best gain vs no-spec:  {decision.best_gain_vs_no_spec * 100:.1f}%")
    print(f"best gain vs vanilla:  {decision.best_gain_vs_vanilla * 100:.1f}%")
    print(f"wrote:   {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
