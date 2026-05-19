from __future__ import annotations

import itertools
import unittest
from fractions import Fraction

from fissionspec.theory import (
    RecoveryScenario,
    closed_cohort_completion_bound,
    expected_recovery_externality,
    horizon_two_complexity,
    linear_masked_padding_cost,
    padding_break_even,
)


class CorrelatedExternalityTests(unittest.TestCase):
    def test_correlated_random_recovery_decomposes_exactly(self) -> None:
        result = expected_recovery_externality(
            (
                RecoveryScenario(Fraction(1, 4), (None, None, None)),
                RecoveryScenario(Fraction(1, 4), (Fraction(2), None, None)),
                RecoveryScenario(Fraction(1, 4), (None, Fraction(5), None)),
                RecoveryScenario(
                    Fraction(1, 4),
                    (Fraction(2), Fraction(5), Fraction(3)),
                ),
            )
        )
        self.assertEqual(result.barrier_stalled_row_ms, Fraction(9))
        self.assertEqual(result.isolated_stalled_row_ms, Fraction(17, 4))
        self.assertEqual(result.collateral_hit_stall_ms, Fraction(7, 2))
        self.assertEqual(result.cross_miss_stall_ms, Fraction(5, 4))
        self.assertEqual(result.total_externality_ms, Fraction(19, 4))
        self.assertEqual(result.amplification, Fraction(36, 17))

    def test_perfectly_correlated_equal_recoveries_have_no_externality(self) -> None:
        result = expected_recovery_externality(
            (
                RecoveryScenario(Fraction(3, 4), (None, None)),
                RecoveryScenario(Fraction(1, 4), (Fraction(7), Fraction(7))),
            )
        )
        self.assertEqual(result.barrier_stalled_row_ms, result.isolated_stalled_row_ms)
        self.assertEqual(result.total_externality_ms, 0)
        self.assertEqual(result.amplification, 1)

    def test_iid_fixed_recovery_matches_closed_form_for_small_domain(self) -> None:
        for rows, hit_numerator in itertools.product(range(1, 6), range(0, 5)):
            with self.subTest(rows=rows, hit_numerator=hit_numerator):
                hit_probability = Fraction(hit_numerator, 4)
                atoms: list[RecoveryScenario] = []
                for hit_mask in itertools.product((False, True), repeat=rows):
                    probability = Fraction(1)
                    recovery: list[Fraction | None] = []
                    for hit in hit_mask:
                        probability *= hit_probability if hit else 1 - hit_probability
                        recovery.append(None if hit else Fraction(3))
                    if probability:
                        atoms.append(RecoveryScenario(probability, tuple(recovery)))
                result = expected_recovery_externality(atoms)
                expected_barrier = rows * 3 * (1 - hit_probability**rows)
                expected_isolated = rows * 3 * (1 - hit_probability)
                self.assertEqual(result.barrier_stalled_row_ms, expected_barrier)
                self.assertEqual(result.isolated_stalled_row_ms, expected_isolated)

    def test_distribution_validation_is_exact(self) -> None:
        with self.assertRaises(ValueError):
            expected_recovery_externality((RecoveryScenario(Fraction(1, 3), (None,)),))
        with self.assertRaises(ValueError):
            expected_recovery_externality(
                (
                    RecoveryScenario(Fraction(1, 2), (None,)),
                    RecoveryScenario(Fraction(1, 2), (None, None)),
                )
            )
        with self.assertRaises(ValueError):
            RecoveryScenario(Fraction(1), ())


class PaddingBreakEvenTests(unittest.TestCase):
    def test_linear_padding_boundary(self) -> None:
        cost = linear_masked_padding_cost(
            recovering_rows=2,
            verifier_width=4,
            masked_slot_cost_ms=Fraction(1, 2),
            row_cost_ms=1,
        )
        self.assertEqual(cost, 5)
        self.assertEqual(
            padding_break_even(
                incremental_target_cost_ms=cost,
                delay_saved_ms=(2, 2),
            ).preference,
            "fission",
        )
        self.assertEqual(
            padding_break_even(
                incremental_target_cost_ms=cost,
                delay_saved_ms=(2, 3),
            ).preference,
            "indifferent",
        )
        self.assertEqual(
            padding_break_even(
                incremental_target_cost_ms=cost,
                delay_saved_ms=(4, 3),
            ).preference,
            "padding",
        )

    def test_weighted_delay_credit(self) -> None:
        result = padding_break_even(
            incremental_target_cost_ms=8,
            delay_saved_ms=(2, 4),
            delay_weights=(1, 2),
        )
        self.assertEqual(result.weighted_delay_credit_ms, 10)
        self.assertEqual(result.padding_minus_fission_ms, -2)

    def test_invalid_linear_model_rejected(self) -> None:
        with self.assertRaises(ValueError):
            linear_masked_padding_cost(
                recovering_rows=0,
                verifier_width=4,
                masked_slot_cost_ms=1,
            )
        with self.assertRaises(ValueError):
            padding_break_even(incremental_target_cost_ms=1, delay_saved_ms=())


class ControllerBoundTests(unittest.TestCase):
    def test_closed_cohort_liveness_bound(self) -> None:
        bound = closed_cohort_completion_bound(
            rows=17,
            capacity=8,
            max_coalescing_wait_ms=2,
            max_target_launch_ms=5,
            max_recovery_ms=7,
        )
        self.assertEqual(bound.launches, 3)
        self.assertEqual(bound.completion_after_ready_ms, 24)

    def test_complexity_profile_lookup_count(self) -> None:
        result = horizon_two_complexity(current_rows=3, future_rows=18, capacity=8)
        self.assertEqual(result.forecast_rows, 21)
        self.assertEqual(result.latency_profile_lookups, 7)
        self.assertEqual(result.auxiliary_rows, 21)
        self.assertIn("log", result.time_complexity)

    def test_bound_inputs_reject_bools_and_nonpositive_sizes(self) -> None:
        with self.assertRaises(ValueError):
            closed_cohort_completion_bound(
                rows=True,
                capacity=1,
                max_coalescing_wait_ms=0,
                max_target_launch_ms=1,
            )
        with self.assertRaises(ValueError):
            horizon_two_complexity(current_rows=1, future_rows=-1, capacity=1)


if __name__ == "__main__":
    unittest.main()
