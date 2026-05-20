from __future__ import annotations

import unittest

from fissionspec.experiment_design import (
    DesignCell,
    ExperimentSpendCaps,
    GateStatus,
    SequentialGateConfig,
    calibration_refinement_plan,
    evaluate_sequential_gate,
    paired_block_order,
    select_farthest_cells,
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
    def test_only_predeclared_looks_are_evaluated(self) -> None:
        config = SequentialGateConfig(minimum_blocks=10, maximum_blocks=20, look_every=5)
        self.assertEqual(
            evaluate_sequential_gate((0.1,) * 9, config).status,
            GateStatus.NOT_A_LOOK,
        )
        self.assertEqual(
            evaluate_sequential_gate((0.1,) * 11, config).status,
            GateStatus.NOT_A_LOOK,
        )
        self.assertIsNotNone(evaluate_sequential_gate((0.1,) * 10, config).interval)

    def test_efficacy_futility_and_max_are_deterministic(self) -> None:
        # Relaxed confidence is used only to exercise the finite test boundary.
        config = SequentialGateConfig(
            minimum_blocks=100,
            maximum_blocks=200,
            look_every=50,
            confidence_level=0.5,
            target_half_width=0.01,
            minimum_worthwhile_improvement=0.03,
        )
        efficacy = evaluate_sequential_gate((0.9,) * 100, config)
        self.assertEqual(efficacy.status, GateStatus.EFFICACY)
        self.assertTrue(efficacy.terminal)

        futility = evaluate_sequential_gate((-0.9,) * 100, config)
        self.assertEqual(futility.status, GateStatus.FUTILITY)
        self.assertTrue(futility.terminal)

        maximum = evaluate_sequential_gate(
            tuple(-0.5 if index % 2 else 0.5 for index in range(200)),
            config,
        )
        self.assertEqual(maximum.status, GateStatus.MAXIMUM_REACHED)
        self.assertTrue(maximum.terminal)

    def test_observations_cannot_exceed_cap_or_bounds(self) -> None:
        config = SequentialGateConfig(minimum_blocks=5, maximum_blocks=10, look_every=5)
        with self.assertRaises(ValueError):
            evaluate_sequential_gate((0.0,) * 11, config)
        with self.assertRaises(ValueError):
            evaluate_sequential_gate((2.0,) * 5, config)

    def test_all_nontrivial_terminal_boundaries_are_disjoint(self) -> None:
        positive_but_small = SequentialGateConfig(
            minimum_blocks=1000,
            maximum_blocks=2000,
            look_every=1000,
            confidence_level=0.5,
            target_half_width=0.01,
            minimum_worthwhile_improvement=0.8,
        )
        self.assertEqual(
            evaluate_sequential_gate((0.5,) * 1000, positive_but_small).status,
            GateStatus.POSITIVE_BELOW_MWI,
        )

        precise = SequentialGateConfig(
            minimum_blocks=100,
            maximum_blocks=200,
            look_every=100,
            confidence_level=0.5,
            target_half_width=1.0,
            minimum_worthwhile_improvement=0.03,
        )
        balanced = tuple(-0.5 if index % 2 else 0.5 for index in range(100))
        self.assertEqual(
            evaluate_sequential_gate(balanced, precise).status,
            GateStatus.PRECISE_INCONCLUSIVE,
        )

        continuing = SequentialGateConfig(
            minimum_blocks=100,
            maximum_blocks=200,
            look_every=100,
            confidence_level=0.5,
            target_half_width=0.01,
            minimum_worthwhile_improvement=0.03,
        )
        decision = evaluate_sequential_gate(balanced, continuing)
        self.assertEqual(decision.status, GateStatus.CONTINUE)
        self.assertFalse(decision.terminal)

    def test_invalid_gate_configurations_are_rejected(self) -> None:
        for kwargs in (
            {"minimum_blocks": 0},
            {"minimum_blocks": 10, "maximum_blocks": 5},
            {"minimum_blocks": 9, "look_every": 5},
            {"maximum_blocks": 11, "look_every": 5},
            {"confidence_level": 1.0},
            {"target_half_width": 0.0},
            {"minimum_worthwhile_improvement": 1.1},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                SequentialGateConfig(**kwargs)


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
