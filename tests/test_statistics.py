from __future__ import annotations

import unittest

from fissionspec.statistics import (
    bonferroni_metadata,
    bounded_mean_confidence_sequence,
    paired_cluster_bootstrap,
    paired_effect_size,
    paired_replication_plan,
    precision_stopping,
)


class PairedEffectTests(unittest.TestCase):
    def test_effects_preserve_pairing_and_orientation(self) -> None:
        higher = paired_effect_size(
            candidate=(2.0, 4.0, 6.0),
            baseline=(1.0, 2.0, 3.0),
            direction="higher",
        )
        self.assertEqual(higher.observations, 3)
        self.assertEqual(higher.raw_mean_difference, 2.0)
        self.assertEqual(higher.oriented_mean_improvement, 2.0)
        self.assertEqual(higher.relative_mean_difference, 1.0)
        self.assertAlmostEqual(higher.paired_standardized_improvement or 0.0, 2.0)
        self.assertEqual(higher.probability_of_improvement, 1.0)

        lower = paired_effect_size(
            candidate=(9.0, 8.0, 7.0),
            baseline=(10.0, 10.0, 10.0),
            direction="lower",
        )
        self.assertEqual(lower.raw_mean_difference, -2.0)
        self.assertEqual(lower.oriented_mean_improvement, 2.0)
        self.assertAlmostEqual(lower.paired_standardized_improvement or 0.0, 2.0)

    def test_zero_variance_standardized_effect_is_disclosed_as_undefined(self) -> None:
        effect = paired_effect_size(
            candidate=(2.0, 3.0, 4.0),
            baseline=(1.0, 2.0, 3.0),
            direction="higher",
        )
        self.assertIsNone(effect.paired_standardized_improvement)

    def test_pair_validation_rejects_mismatch_and_nonfinite_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "equal paired lengths"):
            paired_effect_size((1.0, 2.0), (1.0, 2.0, 3.0), direction="higher")
        with self.assertRaisesRegex(ValueError, "finite"):
            paired_effect_size((1.0, float("nan")), (1.0, 2.0), direction="higher")


class ClusterBootstrapTests(unittest.TestCase):
    def test_resampling_is_deterministic_order_invariant_and_cluster_weighted(self) -> None:
        clusters = {
            "seed-3": (2.0, 2.0),
            "seed-1": (1.0, 3.0),
            "seed-2": (-1.0, -3.0),
        }
        first = paired_cluster_bootstrap(
            clusters,
            confidence_level=0.95,
            resamples=500,
            seed="bootstrap-test",
        )
        repeated = paired_cluster_bootstrap(
            dict(reversed(tuple(clusters.items()))),
            confidence_level=0.95,
            resamples=500,
            seed="bootstrap-test",
        )
        self.assertEqual(first, repeated)
        self.assertAlmostEqual(first.point_estimate, 2.0 / 3.0)
        self.assertLessEqual(first.lower, first.point_estimate)
        self.assertGreaterEqual(first.upper, first.point_estimate)
        self.assertEqual(len(first.resample_fingerprint_sha256), 64)
        different_seed = paired_cluster_bootstrap(
            clusters,
            resamples=500,
            seed="different-bootstrap-test",
        )
        self.assertNotEqual(
            first.resample_fingerprint_sha256,
            different_seed.resample_fingerprint_sha256,
        )

    def test_bootstrap_rejects_pseudoreplication_and_too_few_draws(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least two clusters"):
            paired_cluster_bootstrap({"only-seed": (1.0,)})
        with self.assertRaisesRegex(ValueError, "at least 100"):
            paired_cluster_bootstrap({"a": (1.0,), "b": (2.0,)}, resamples=99)


class SequentialInferenceTests(unittest.TestCase):
    def test_confidence_sequence_is_simultaneous_and_contracts(self) -> None:
        sequence = bounded_mean_confidence_sequence(
            (0.5,) * 500,
            lower_bound=0.0,
            upper_bound=1.0,
            confidence_level=0.95,
        )
        self.assertEqual(len(sequence), 500)
        self.assertTrue(all(point.lower <= 0.5 <= point.upper for point in sequence))
        self.assertLess(sequence[-1].half_width, sequence[9].half_width)
        self.assertAlmostEqual(
            sum(point.alpha_spent_at_look for point in sequence),
            0.05 * 500 / 501,
        )

    def test_precision_stopping_uses_first_valid_look(self) -> None:
        result = precision_stopping(
            (0.5,) * 2_000,
            lower_bound=0.0,
            upper_bound=1.0,
            target_half_width=0.1,
            minimum_observations=20,
        )
        self.assertTrue(result.reached_precision)
        self.assertGreaterEqual(result.observations_used, 20)
        self.assertLess(result.observations_used, result.available_observations)
        self.assertLessEqual(result.final_interval.half_width, 0.1)
        insufficient = precision_stopping(
            (0.5,) * 3,
            lower_bound=0.0,
            upper_bound=1.0,
            target_half_width=0.1,
            minimum_observations=20,
        )
        self.assertFalse(insufficient.reached_precision)
        self.assertEqual(insufficient.observations_used, 3)
        with self.assertRaisesRegex(ValueError, "predeclared bounds"):
            bounded_mean_confidence_sequence(
                (2.0,),
                lower_bound=0.0,
                upper_bound=1.0,
            )


class PlanningTests(unittest.TestCase):
    def test_multiplicity_metadata_declares_simultaneous_level(self) -> None:
        metadata = bonferroni_metadata(
            ("h1", "h2", "h3", "h4"),
            family_id="primary-family",
            familywise_alpha=0.05,
        )
        self.assertEqual(metadata.per_hypothesis_alpha, 0.0125)
        self.assertEqual(metadata.simultaneous_per_hypothesis_confidence_level, 0.9875)
        self.assertFalse(metadata.confirmatory)
        with self.assertRaisesRegex(ValueError, "unique"):
            bonferroni_metadata(("duplicate", "duplicate"), family_id="bad")

    def test_replication_plan_accounts_for_power_and_multiplicity(self) -> None:
        unadjusted = paired_replication_plan(
            (-1.0, 0.0, 1.0, 2.0),
            minimum_detectable_standardized_effect=0.5,
            familywise_alpha=0.05,
            target_power=0.8,
            hypotheses=1,
        )
        adjusted = paired_replication_plan(
            (-1.0, 0.0, 1.0, 2.0),
            minimum_detectable_standardized_effect=0.5,
            familywise_alpha=0.05,
            target_power=0.8,
            hypotheses=20,
        )
        self.assertGreater(unadjusted.recommended_replications, 4)
        self.assertGreater(adjusted.recommended_replications, unadjusted.recommended_replications)
        self.assertEqual(
            unadjusted.additional_replications,
            unadjusted.recommended_replications - 4,
        )
        self.assertEqual(len(unadjusted.assumptions), 4)


if __name__ == "__main__":
    unittest.main()
