from __future__ import annotations

import math
import unittest

from fissionspec.metrics import (
    batch_fallback_probability,
    expected_collateral_hit_stalls,
    head_of_line_amplification,
    percentile,
)
from fissionspec.profiles import HardwareProfile, LatencyCurve


class TheoryTests(unittest.TestCase):
    def test_batch_fallback_is_one_minus_product(self) -> None:
        self.assertAlmostEqual(batch_fallback_probability([0.9, 0.8, 0.5]), 0.64)
        self.assertEqual(batch_fallback_probability([1.0, 1.0]), 0.0)
        self.assertEqual(batch_fallback_probability([0.0, 1.0]), 1.0)

    def test_homogeneous_fallback_grows_with_batch(self) -> None:
        probability = 0.93
        observed = [
            batch_fallback_probability([probability] * batch)
            for batch in (1, 2, 4, 8)
        ]
        self.assertEqual(observed, sorted(observed))
        self.assertAlmostEqual(observed[-1], 1.0 - probability**8)

    def test_batch_fallback_is_stable_for_rare_and_tiny_probabilities(self) -> None:
        rare_miss = 1e-12
        probability = 1.0 - rare_miss
        observed = batch_fallback_probability([probability] * 1_000)
        expected = -math.expm1(1_000 * math.log1p(probability - 1.0))
        self.assertAlmostEqual(observed, expected, places=20)
        self.assertEqual(batch_fallback_probability([1e-300, 1.0]), 1.0)

    def test_collateral_stalls_excludes_the_miss_itself(self) -> None:
        # For two p=.5 rows, either row is a collateral hit-and-stall with
        # probability .25, hence .5 expected collateral rows.
        self.assertAlmostEqual(expected_collateral_hit_stalls([0.5, 0.5]), 0.5)
        self.assertEqual(expected_collateral_hit_stalls([1.0, 1.0]), 0.0)

    def test_head_of_line_amplification_matches_expected_work_ratio(self) -> None:
        # Barrier work: 2 * .75 = 1.5. Fission work: E[misses] = 1.
        self.assertAlmostEqual(head_of_line_amplification([0.5, 0.5]), 1.5)
        self.assertAlmostEqual(head_of_line_amplification([0.9]), 1.0)
        self.assertEqual(head_of_line_amplification([1.0, 1.0]), 1.0)

    def test_probability_validation(self) -> None:
        with self.assertRaises(ValueError):
            batch_fallback_probability([])
        for invalid in (-0.1, 1.1, math.nan, math.inf):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                head_of_line_amplification([invalid])

    def test_percentile_interpolates_and_handles_empty_samples(self) -> None:
        self.assertEqual(percentile([], 0.99), 0.0)
        self.assertEqual(percentile([1.0], 0.5), 1.0)
        self.assertAlmostEqual(percentile([0.0, 10.0], 0.95), 9.5)
        with self.assertRaises(ValueError):
            percentile([1.0], 1.01)


class ProfileTests(unittest.TestCase):
    def test_latency_curve_interpolates_and_extrapolates(self) -> None:
        curve = LatencyCurve(((2, 4.0), (4, 6.0)))
        self.assertEqual(curve(1), 2.0)
        self.assertEqual(curve(3), 5.0)
        self.assertEqual(curve(6), 8.0)

    def test_profile_prices_verifier_slots_separately(self) -> None:
        profile = HardwareProfile.linear(
            target_overhead_ms=1.0,
            target_per_row_ms=0.5,
            verifier_slot_ms=0.1,
        )
        self.assertAlmostEqual(profile.target_latency_ms(2, 8), 2.8)
        self.assertGreater(
            profile.draft_latency_ms(2, recovery=True),
            profile.draft_latency_ms(2, recovery=False),
        )

    def test_curve_rejects_non_monotone_measurements(self) -> None:
        with self.assertRaises(ValueError):
            LatencyCurve(((1, 2.0), (2, 1.0)))
        with self.assertRaises(ValueError):
            LatencyCurve(((1, 1.0), (1, 2.0)))


if __name__ == "__main__":
    unittest.main()
