from __future__ import annotations

import unittest

from fissionspec.experiment_design import (
    PRIMARY_FAMILY_ID,
    PRIMARY_HYPOTHESIS_IDS,
    SEQUENTIAL_PROTOCOL_VERSION,
    DesignCell,
    ExperimentSpendCaps,
    GateStatus,
    SequentialGateConfig,
    SequentialGateDecision,
    calibration_refinement_plan,
    evaluate_sequential_family,
    evaluate_sequential_gate,
    paired_block_order,
    select_farthest_cells,
    sequential_gate_feasibility,
    sequential_gate_monte_carlo,
    symmetric_improvement,
)


class BoundedMetricTests(unittest.TestCase):
    def test_symmetric_improvement_is_bounded_and_oriented(self) -> None:
        self.assertAlmostEqual(
            symmetric_improvement(120, 100, direction="higher"),
            1 / 6,
        )
        self.assertAlmostEqual(
            symmetric_improvement(80, 100, direction="lower"),
            0.2,
        )
        self.assertEqual(symmetric_improvement(0, 0, direction="higher"), 0)
        self.assertEqual(symmetric_improvement(0, 4, direction="higher"), -1)
        self.assertEqual(symmetric_improvement(4, 0, direction="higher"), 1)
        with self.assertRaises(ValueError):
            symmetric_improvement(-1, 1, direction="higher")

    def test_block_order_alternates_abba_and_baab(self) -> None:
        self.assertEqual(
            paired_block_order(0),
            ("candidate", "baseline", "baseline", "candidate"),
        )
        self.assertEqual(
            paired_block_order(1),
            ("baseline", "candidate", "candidate", "baseline"),
        )
        self.assertEqual(paired_block_order(2), paired_block_order(0))


class SequentialGateTests(unittest.TestCase):
    @staticmethod
    def _evaluate(
        values: tuple[float, ...],
        config: SequentialGateConfig,
    ) -> SequentialGateDecision:
        return evaluate_sequential_gate(
            values,
            config,
            hypothesis_id=config.hypothesis_ids[0],
            observed_family_id=config.family_id,
            observed_hypothesis_ids=config.hypothesis_ids,
        )

    def test_only_predeclared_looks_are_evaluated(self) -> None:
        config = SequentialGateConfig(minimum_blocks=10, maximum_blocks=20, look_every=5)
        self.assertEqual(
            self._evaluate((0.1,) * 9, config).status,
            GateStatus.NOT_A_LOOK,
        )
        self.assertEqual(
            self._evaluate((0.1,) * 11, config).status,
            GateStatus.NOT_A_LOOK,
        )
        decision = self._evaluate((0.1,) * 10, config)
        self.assertIsNotNone(decision.interval)
        self.assertIsNotNone(decision.distribution_free_sensitivity)
        self.assertEqual(decision.protocol_version, SEQUENTIAL_PROTOCOL_VERSION)

    def test_efficacy_futility_and_max_are_deterministic(self) -> None:
        config = SequentialGateConfig(
            family_id="test-family",
            hypothesis_ids=("h1",),
            minimum_blocks=10,
            maximum_blocks=20,
            look_every=5,
            target_half_width=0.01,
            minimum_worthwhile_improvement=0.03,
        )
        efficacy = self._evaluate((0.1,) * 10, config)
        self.assertEqual(efficacy.status, GateStatus.EFFICACY)
        self.assertTrue(efficacy.terminal)

        futility = self._evaluate((0.0,) * 10, config)
        self.assertEqual(futility.status, GateStatus.FUTILITY)
        self.assertTrue(futility.terminal)

        maximum = self._evaluate(
            tuple(-0.5 if index % 2 else 0.5 for index in range(20)),
            config,
        )
        self.assertEqual(maximum.status, GateStatus.MAXIMUM_REACHED)
        self.assertTrue(maximum.terminal)

    def test_observations_cannot_exceed_cap_or_bounds(self) -> None:
        config = SequentialGateConfig(minimum_blocks=5, maximum_blocks=10, look_every=5)
        with self.assertRaises(ValueError):
            self._evaluate((0.0,) * 11, config)
        with self.assertRaises(ValueError):
            self._evaluate((2.0,) * 5, config)

    def test_all_nontrivial_terminal_boundaries_are_disjoint(self) -> None:
        positive_but_small = SequentialGateConfig(
            family_id="positive-small",
            hypothesis_ids=("h1",),
            minimum_blocks=10,
            maximum_blocks=20,
            look_every=10,
            target_half_width=0.01,
            minimum_worthwhile_improvement=0.03,
        )
        self.assertEqual(
            self._evaluate((0.01,) * 10, positive_but_small).status,
            GateStatus.POSITIVE_BELOW_MWI,
        )

        precise = SequentialGateConfig(
            family_id="precise",
            hypothesis_ids=("h1",),
            minimum_blocks=10,
            maximum_blocks=20,
            look_every=10,
            target_half_width=0.01,
            minimum_worthwhile_improvement=0.03,
        )
        self.assertEqual(
            self._evaluate((0.03,) * 10, precise).status,
            GateStatus.PRECISE_INCONCLUSIVE,
        )

        continuing = SequentialGateConfig(
            family_id="continuing",
            hypothesis_ids=("h1",),
            minimum_blocks=10,
            maximum_blocks=20,
            look_every=10,
            target_half_width=0.01,
            minimum_worthwhile_improvement=0.03,
        )
        balanced = tuple(-0.5 if index % 2 else 0.5 for index in range(10))
        decision = self._evaluate(balanced, continuing)
        self.assertEqual(decision.status, GateStatus.CONTINUE)
        self.assertFalse(decision.terminal)

    def test_invalid_gate_configurations_are_rejected(self) -> None:
        for kwargs in (
            {"minimum_blocks": 0},
            {"minimum_blocks": 10, "maximum_blocks": 5},
            {"minimum_blocks": 9, "look_every": 5},
            {"maximum_blocks": 11, "look_every": 5},
            {"familywise_alpha": 1.0},
            {"target_half_width": 0.0},
            {"minimum_worthwhile_improvement": 1.1},
            {"protocol_version": 1},
            {"hypothesis_ids": ("duplicate", "duplicate")},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                SequentialGateConfig(**kwargs)

    def test_default_family_is_exactly_registered_and_mismatch_fails_closed(self) -> None:
        config = SequentialGateConfig()
        self.assertEqual(config.family_id, PRIMARY_FAMILY_ID)
        self.assertEqual(config.hypothesis_ids, PRIMARY_HYPOTHESIS_IDS)
        self.assertEqual(len(config.hypothesis_ids), 24)
        self.assertAlmostEqual(config.per_interval_alpha, 0.05 / (24 * 9))
        with self.assertRaisesRegex(ValueError, "family_id"):
            evaluate_sequential_gate(
                (0.0,) * 10,
                config,
                hypothesis_id=config.hypothesis_ids[0],
                observed_family_id="wrong-family",
                observed_hypothesis_ids=config.hypothesis_ids,
            )
        with self.assertRaisesRegex(ValueError, "hypothesis_ids"):
            evaluate_sequential_gate(
                (0.0,) * 10,
                config,
                hypothesis_id=config.hypothesis_ids[0],
                observed_family_id=config.family_id,
                observed_hypothesis_ids=tuple(reversed(config.hypothesis_ids)),
            )

    def test_synchronized_family_gate_is_conjunctive_and_complete(self) -> None:
        config = SequentialGateConfig(
            family_id="two-endpoint-family",
            hypothesis_ids=("h1", "h2"),
            minimum_blocks=10,
            maximum_blocks=20,
            look_every=5,
        )
        efficacy = evaluate_sequential_family(
            {"h1": (0.1,) * 10, "h2": (0.2,) * 10},
            config,
            observed_family_id=config.family_id,
            observed_hypothesis_ids=config.hypothesis_ids,
        )
        self.assertEqual(efficacy.status, GateStatus.EFFICACY)
        self.assertTrue(efficacy.terminal)
        futility = evaluate_sequential_family(
            {"h1": (0.1,) * 10, "h2": (0.0,) * 10},
            config,
            observed_family_id=config.family_id,
            observed_hypothesis_ids=config.hypothesis_ids,
        )
        self.assertEqual(futility.status, GateStatus.FUTILITY)
        self.assertTrue(futility.terminal)
        with self.assertRaisesRegex(ValueError, "exactly cover"):
            evaluate_sequential_family(
                {"h1": (0.1,) * 10},
                config,
                observed_family_id=config.family_id,
                observed_hypothesis_ids=config.hypothesis_ids,
            )
        with self.assertRaisesRegex(ValueError, "same completed"):
            evaluate_sequential_family(
                {"h1": (0.1,) * 10, "h2": (0.1,) * 15},
                config,
                observed_family_id=config.family_id,
                observed_hypothesis_ids=config.hypothesis_ids,
            )

    def test_feasibility_quantifies_old_impossibility_and_new_variance_boundary(self) -> None:
        diagnostics = sequential_gate_feasibility()

        self.assertGreater(
            diagnostics.original_endpointwise_hoeffding_radius_at_maximum,
            diagnostics.target_half_width,
        )
        self.assertGreater(
            diagnostics.familywise_hoeffding_radius_at_maximum,
            diagnostics.target_half_width,
        )
        maximum_sd = dict(diagnostics.maximum_sd_for_target_half_width)[50]
        self.assertGreater(maximum_sd, 0.04)
        self.assertLess(maximum_sd, 0.06)

    def test_monte_carlo_is_deterministic_and_exposes_assumption_boundary(self) -> None:
        first = sequential_gate_monte_carlo(trials=100, seed="gate-mc-test")
        repeated = sequential_gate_monte_carlo(trials=100, seed="gate-mc-test")
        self.assertEqual(first, repeated)
        self.assertLessEqual(first.normal_null_familywise_noncoverage_rate, 0.10)
        self.assertLessEqual(first.normal_null_any_false_positive_rate, 0.10)
        by_name = {scenario.name: scenario for scenario in first.scenarios}
        self.assertGreaterEqual(
            by_name["null-normal"].primary_all_look_coverage_rate,
            0.95,
        )
        self.assertGreaterEqual(
            by_name["adversarial-skewed-null"].sensitivity_all_look_coverage_rate,
            0.95,
        )
        self.assertLess(
            by_name["adversarial-skewed-null"].primary_all_look_coverage_rate,
            0.90,
        )
        low_variance_rates = dict(by_name["worthwhile-low-variance"].stop_rates)
        self.assertGreater(low_variance_rates[GateStatus.EFFICACY.value], 0.8)
        high_variance_rates = dict(by_name["worthwhile-high-variance"].stop_rates)
        self.assertGreater(high_variance_rates[GateStatus.MAXIMUM_REACHED.value], 0.5)


class SpendCapTests(unittest.TestCase):
    def test_registered_replay_caps_are_exact_and_enforced(self) -> None:
        caps = ExperimentSpendCaps()
        self.assertEqual(caps.minimum_primary_replays, 240)
        self.assertEqual(caps.maximum_primary_replays, 1200)
        self.assertEqual(caps.maximum_unique_ablation_replays, 300)
        caps.validate_manifest_counts(
            primary_replays=1200,
            unique_ablation_replays=300,
            robustness_cells=12,
        )
        for counts in (
            {
                "primary_replays": 1201,
                "unique_ablation_replays": 300,
                "robustness_cells": 12,
            },
            {
                "primary_replays": 1200,
                "unique_ablation_replays": 301,
                "robustness_cells": 12,
            },
            {
                "primary_replays": 1200,
                "unique_ablation_replays": 300,
                "robustness_cells": 13,
            },
        ):
            with self.subTest(counts=counts), self.assertRaises(ValueError):
                caps.validate_manifest_counts(**counts)


class CalibrationDesignTests(unittest.TestCase):
    def test_anchor_error_adds_only_predeclared_intermediates(self) -> None:
        plan = calibration_refinement_plan(
            {
                (1, 4): 1.0,
                (8, 4): 1.2,
                (32, 4): 3.0,
                (1, 8): 2.0,
                (8, 8): 2.4516129032258065,
                (32, 8): 4.0,
            }
        )
        by_width = {item.width: item for item in plan}
        self.assertEqual(by_width[4].add_batch_rows, (2, 4, 16))
        self.assertEqual(by_width[8].add_batch_rows, ())
        with self.assertRaises(ValueError):
            calibration_refinement_plan({(1, 4): 1.0})

    def test_farthest_point_selection_is_stratified_and_deterministic(self) -> None:
        cells = tuple(
            DesignCell(
                cell_id=f"{region}-{index}",
                region=region,
                parameters=(float(index), float(index % 2)),
            )
            for region in ("boundary", "immediate", "refusion")
            for index in range(6)
        )
        expected = select_farthest_cells(
            cells,
            per_region={"immediate": 2, "boundary": 2, "refusion": 2},
        )
        repeated = select_farthest_cells(
            tuple(reversed(cells)),
            per_region={"refusion": 2, "boundary": 2, "immediate": 2},
        )
        self.assertEqual(expected, repeated)
        self.assertEqual(
            {
                region: sum(cell.region == region for cell in expected)
                for region in {
                    "boundary",
                    "immediate",
                    "refusion",
                }
            },
            {"boundary": 2, "immediate": 2, "refusion": 2},
        )
        for region in ("boundary", "immediate", "refusion"):
            identifiers = [cell.cell_id for cell in expected if cell.region == region]
            self.assertIn(f"{region}-0", identifiers)
            self.assertIn(f"{region}-5", identifiers)


if __name__ == "__main__":
    unittest.main()
