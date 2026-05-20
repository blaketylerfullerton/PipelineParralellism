import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "deploy"))

from speculative_pipelining_analyze import _threshold_matches  # noqa: E402
from speculative_pipelining import (  # noqa: E402
    ModelInputs,
    bootstrap_median_ci,
    decide_direction,
    predict_grid,
    predict_point,
)


class SpeculativePipeliningModelTests(unittest.TestCase):
    def test_predict_point_matches_model_equations(self):
        point = predict_point(
            ModelInputs(verifier_ms=100.0, draft_ms=10.0, alpha=0.5, k=4, latency_ms=50.0)
        )

        self.assertAlmostEqual(point.expected_tokens_per_round, 3.0)
        self.assertAlmostEqual(point.no_spec_tps, 1000.0 / 150.0)
        self.assertAlmostEqual(point.vanilla_spec_tps, 3000.0 / 190.0)
        self.assertAlmostEqual(point.pipelined_spec_tps, 3000.0 / 150.0)

    def test_decision_confirms_when_pipeline_beats_both_baselines(self):
        points = predict_grid(
            verifier_ms=100.0,
            draft_ms=10.0,
            alpha_by_k={4: 0.5},
            latencies_ms=[0.0, 50.0],
        )

        decision = decide_direction(points, confirm_gain=0.20)

        self.assertEqual(decision.outcome, "confirmed")
        self.assertEqual(decision.first_confirming_latency_ms, 0.0)

    def test_decision_kills_when_pipeline_never_beats_no_spec(self):
        points = predict_grid(
            verifier_ms=100.0,
            draft_ms=10.0,
            alpha_by_k={4: 0.0},
            latencies_ms=[0.0, 50.0, 200.0],
        )

        decision = decide_direction(points, confirm_gain=0.20)

        self.assertEqual(decision.outcome, "killed")

    def test_decision_marks_small_win_as_ambiguous(self):
        points = predict_grid(
            verifier_ms=100.0,
            draft_ms=10.0,
            alpha_by_k={1: 0.05},
            latencies_ms=[0.0],
        )

        decision = decide_direction(points, confirm_gain=0.20)

        self.assertEqual(decision.outcome, "ambiguous")

    def test_decision_requires_threshold_to_hold_after_crossover(self):
        points = predict_grid(
            verifier_ms=100.0,
            draft_ms=10.0,
            alpha_by_k={4: 0.5},
            latencies_ms=[0.0, 50.0],
        )
        points = [
            point
            if point.latency_ms == 0.0
            else point.__class__(
                k=point.k,
                latency_ms=point.latency_ms,
                no_spec_tps=point.no_spec_tps,
                vanilla_spec_tps=point.vanilla_spec_tps,
                pipelined_spec_tps=point.no_spec_tps * 1.01,
                expected_tokens_per_round=point.expected_tokens_per_round,
            )
            for point in points
        ]

        decision = decide_direction(points, confirm_gain=0.20)

        self.assertEqual(decision.outcome, "ambiguous")

    def test_bootstrap_median_ci_contains_median(self):
        median, low, high = bootstrap_median_ci([1.0, 2.0, 100.0], samples=100, seed=1)

        self.assertEqual(median, 2.0)
        self.assertLessEqual(low, median)
        self.assertGreaterEqual(high, median)

    def test_threshold_match_allows_two_x_model_agreement(self):
        self.assertTrue(_threshold_matches(100.0, 200.0))
        self.assertTrue(_threshold_matches(0.0, 0.0))
        self.assertFalse(_threshold_matches(100.0, 250.0))
        self.assertFalse(_threshold_matches(0.0, 25.0))


if __name__ == "__main__":
    unittest.main()
