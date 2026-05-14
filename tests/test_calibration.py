from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from fissionspec.calibration import (
    CalibrationError,
    TimingSample,
    fit_profile,
    load_samples_csv,
    write_profile_json,
)


class CalibrationTests(unittest.TestCase):
    def test_recovers_separable_target_slot_cost(self) -> None:
        samples = []
        for rows, base in ((1, 2.0), (4, 3.0), (8, 4.5)):
            for slots in (rows, rows * 4):
                samples.extend(
                    TimingSample("target", rows, base + 0.02 * slots + noise, slots)
                    for noise in (-0.001, 0.0, 0.001)
                )
            samples.append(TimingSample("draft", rows, 0.4 + rows * 0.05))
            samples.append(TimingSample("recovery", rows, 0.8 + rows * 0.09))

        fitted = fit_profile(samples, name="unit")
        self.assertTrue(fitted.slot_slope_identified)
        self.assertAlmostEqual(fitted.verifier_slot_ms, 0.02, places=8)
        self.assertAlmostEqual(fitted.raw_verifier_slot_ms, 0.02, places=8)
        self.assertFalse(fitted.slot_slope_clipped)
        self.assertEqual(tuple(value for _, value in fitted.target_curve), (2.0, 3.0, 4.5))
        self.assertLess(fitted.target_rmse_ms, 0.002)
        self.assertEqual(fitted.sample_count, len(samples))

    def test_replicate_medians_protect_theil_sen_slot_slope(self) -> None:
        samples = [TimingSample("target", 4, latency, 4) for latency in (2.08, 2.08, 90.0)]
        samples.extend(TimingSample("target", 4, latency, 16) for latency in (2.32, 2.32, 0.01))
        samples.extend(
            (
                TimingSample("draft", 4, 0.8),
                TimingSample("recovery", 4, 1.7),
            )
        )

        fitted = fit_profile(samples)
        self.assertAlmostEqual(fitted.verifier_slot_ms, 0.02)
        self.assertAlmostEqual(fitted.target_curve[0][1], 2.0)

    def test_pava_repairs_noisy_nonmonotone_curves(self) -> None:
        samples = [
            TimingSample("target", 1, 1.0, 1),
            TimingSample("target", 2, 1.1, 2),
            TimingSample("draft", 1, 0.8),
            TimingSample("draft", 2, 0.6),
            TimingSample("recovery", 1, 2.0),
            TimingSample("recovery", 2, 1.0),
            TimingSample("recovery", 2, 3.0),
        ]
        fitted = fit_profile(samples)
        draft_values = [value for _, value in fitted.draft_curve]
        recovery_values = [value for _, value in fitted.recovery_curve]
        self.assertEqual(draft_values, [0.7, 0.7])
        self.assertEqual(recovery_values, sorted(recovery_values))
        self.assertFalse(fitted.slot_slope_identified)
        self.assertEqual(fitted.verifier_slot_ms, 0.0)

    def test_pava_uses_replicate_count_as_weight(self) -> None:
        samples = [
            TimingSample("target", 1, 1.0, 1),
            TimingSample("target", 2, 2.0, 2),
            TimingSample("draft", 1, 10.0),
            TimingSample("draft", 2, 4.0),
            TimingSample("draft", 2, 4.0),
            TimingSample("draft", 2, 4.0),
            TimingSample("recovery", 1, 1.0),
            TimingSample("recovery", 2, 2.0),
        ]

        fitted = fit_profile(samples)
        self.assertEqual(fitted.draft_curve, ((1, 5.5), (2, 5.5)))

    def test_negative_slot_slope_is_clipped_and_disclosed(self) -> None:
        samples = [
            TimingSample("target", 1, 2.0, 1),
            TimingSample("target", 1, 1.9, 2),
            TimingSample("draft", 1, 0.5),
            TimingSample("recovery", 1, 1.0),
        ]

        fitted = fit_profile(samples)
        self.assertTrue(fitted.slot_slope_identified)
        self.assertTrue(fitted.slot_slope_clipped)
        self.assertAlmostEqual(fitted.raw_verifier_slot_ms, -0.1)
        self.assertEqual(fitted.verifier_slot_ms, 0.0)

    def test_rejects_nonpositive_target_baseline(self) -> None:
        samples = [
            TimingSample("target", 1, 1.0, 1),
            TimingSample("target", 1, 3.0, 2),
            TimingSample("draft", 1, 0.5),
            TimingSample("recovery", 1, 1.0),
        ]
        with self.assertRaisesRegex(CalibrationError, "baseline is non-positive"):
            fit_profile(samples)

    def test_csv_schema_and_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "timings.csv"
            path.write_text(
                "component,batch_rows,latency_ms,verifier_slots\n"
                "target,2,1.5,4\n"
                "draft,2,0.4,0\n"
                "recovery,2,0.8,0\n",
                encoding="utf-8",
            )
            loaded = load_samples_csv(path)
            self.assertEqual(len(loaded), 3)
            self.assertEqual(fit_profile(loaded).target_curve, ((2, 1.5),))

            path.write_text("component,batch_rows,latency_ms\ntarget,1,2\n")
            with self.assertRaises(CalibrationError):
                load_samples_csv(path)

            path.write_text(
                "component,batch_rows,latency_ms,verifier_slots,extra\ntarget,1,2,1,nope\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(CalibrationError, "exactly"):
                load_samples_csv(path)

            path.write_text(
                "component,batch_rows,latency_ms,verifier_slots\nunknown,1,2,1\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(CalibrationError, "CSV line 2"):
                load_samples_csv(path)

    def test_requires_every_component(self) -> None:
        with self.assertRaises(CalibrationError):
            fit_profile([TimingSample("target", 1, 1.0, 1)])
        with self.assertRaises(CalibrationError):
            TimingSample("target", 2, 1.0, 1)

    def test_strict_runtime_types(self) -> None:
        with self.assertRaises(CalibrationError):
            TimingSample("draft", 1.5, 1.0)  # type: ignore[arg-type]
        with self.assertRaises(CalibrationError):
            TimingSample("draft", 1, True)  # type: ignore[arg-type]
        with self.assertRaises(CalibrationError):
            TimingSample("draft", 1, 1.0, 1.5)  # type: ignore[arg-type]
        with self.assertRaisesRegex(CalibrationError, "must set verifier_slots"):
            TimingSample("draft", 1, 1.0, 1)

    def test_json_output_has_schema_and_diagnostics(self) -> None:
        profile = fit_profile(
            [
                TimingSample("target", 1, 1.0, 1),
                TimingSample("draft", 1, 0.5),
                TimingSample("recovery", 1, 0.8),
            ],
            name="  test-profile  ",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            write_profile_json(
                profile,
                path,
                provenance={"kind": "synthetic", "warning": "not GPU"},
            )
            parsed = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(parsed["schema_version"], 1)
        self.assertEqual(parsed["name"], "test-profile")
        self.assertEqual(parsed["provenance"]["kind"], "synthetic")
        self.assertIn("raw_verifier_slot_ms", parsed["fit"])

    def test_missing_provenance_is_never_claimed_as_measured(self) -> None:
        profile = fit_profile(
            [
                TimingSample("target", 1, 1.0, 1),
                TimingSample("draft", 1, 0.5),
                TimingSample("recovery", 1, 0.8),
            ]
        )
        document = profile.as_dict()
        provenance = document["provenance"]
        self.assertIsInstance(provenance, dict)
        self.assertEqual(provenance["kind"], "unverified-measurement")
        self.assertFalse(provenance["publication_ready"])


if __name__ == "__main__":
    unittest.main()
