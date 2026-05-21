"""GPU-free completion-study design, evidence, and golden-bundle tests."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from typing import ClassVar, cast

from experiments.run_cpu_completion_study import (
    FIDELITY_HARNESS,
    HEADLINE_POLICY,
    MAIN_HARNESS,
    MAIN_REFERENCE,
    SCHEDULER_HARNESS,
    SCHEDULER_REFERENCE,
    WARNING,
    StudyIntegrityError,
    StudyRun,
    design_cells,
    mode_config,
    run_study,
    verify_bundle,
)


class DesignTests(unittest.TestCase):
    def test_pb12_is_balanced_and_every_workload_has_both_splits(self) -> None:
        cells = design_cells()

        self.assertEqual(len(cells), 12)
        self.assertEqual(len({cell.cell_id for cell in cells}), 12)
        for column in range(11):
            self.assertEqual(sum(cell.factor_signs[column] for cell in cells), 0)
        by_workload: dict[str, set[str]] = {}
        for cell in cells:
            by_workload.setdefault(cell.workload_kind, set()).add(cell.split)
        self.assertEqual(len(by_workload), 6)
        self.assertTrue(all(splits == {"train", "validation"} for splits in by_workload.values()))

    def test_full_mode_predeclares_thirty_independent_paired_clusters(self) -> None:
        config = mode_config("full")

        self.assertEqual(len(config.seeds), 30)
        self.assertEqual(len(set(config.seeds)), 30)
        self.assertGreaterEqual(config.bootstrap_resamples, 20_000)
        self.assertGreaterEqual(config.sequential_monte_carlo_trials, 2_000)
        self.assertEqual(config.oracle_jobs, 6)


class BundleTests(unittest.TestCase):
    _temporary: ClassVar[tempfile.TemporaryDirectory[str]]
    bundle: ClassVar[Path]
    study_result: ClassVar[StudyRun]

    @classmethod
    def setUpClass(cls) -> None:
        cls._temporary = tempfile.TemporaryDirectory()
        cls.bundle = Path(cls._temporary.name) / "bundle"
        cls.study_result = run_study(mode="ci", output_dir=cls.bundle)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temporary.cleanup()

    def _csv(self, name: str) -> list[dict[str, str]]:
        with (self.bundle / name).open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))

    def _json(self, name: str) -> dict[str, object]:
        document = cast(object, json.loads((self.bundle / name).read_bytes()))
        self.assertIsInstance(document, dict)
        return cast(dict[str, object], document)

    def test_manifest_and_all_artifact_hashes_verify(self) -> None:
        self.assertEqual(
            verify_bundle(self.bundle),
            self.study_result.manifest_sha256,
        )
        manifest = self._json("manifest.json")
        self.assertEqual(manifest["measurement_warning"], WARNING)
        self.assertFalse(manifest["cross_harness_ranking_permitted"])

    def test_policy_table_preserves_non_comparable_harness_strata(self) -> None:
        rows = self._csv("metrics.csv")

        self.assertEqual(len(rows), 12 * 2 * 8)
        self.assertTrue(all(row["measurement_warning"] == WARNING for row in rows))
        by_harness: dict[str, set[str]] = {}
        references: dict[str, set[str]] = {}
        for row in rows:
            by_harness.setdefault(row["harness"], set()).add(row["policy"])
            references.setdefault(row["harness"], set()).add(row["comparison_reference"])
        self.assertEqual(
            by_harness[MAIN_HARNESS],
            {
                MAIN_REFERENCE,
                "immediate-fission",
                "fixed-coalesce",
                HEADLINE_POLICY,
            },
        )
        self.assertEqual(references[MAIN_HARNESS], {MAIN_REFERENCE})
        self.assertEqual(
            by_harness[SCHEDULER_HARNESS],
            {
                SCHEDULER_REFERENCE,
                "spectre-hybrid-abstraction",
                "exspec-sliding-pool-abstraction",
                "myopic-slack-aging-abstraction",
            },
        )
        self.assertEqual(
            references[SCHEDULER_HARNESS],
            {SCHEDULER_REFERENCE},
        )

    def test_fidelity_rows_and_intervals_cover_every_cell(self) -> None:
        rows = self._csv("fidelity_metrics.csv")
        uncertainty = self._json("uncertainty.json")

        self.assertEqual(len(rows), 12 * 2)
        self.assertEqual({row["harness"] for row in rows}, {FIDELITY_HARNESS})
        intervals = uncertainty["fidelity_cell_intervals"]
        assert isinstance(intervals, list)
        self.assertEqual(len(intervals), 12 * 3)
        self.assertTrue(all(entry["clusters"] == 2 for entry in intervals))
        for entry in intervals:
            interval = entry["interval_on_cluster_mean"]
            assert isinstance(interval, dict)
            self.assertEqual(
                interval["method"],
                "one-sample-percentile-cluster-mean-bootstrap",
            )
            self.assertEqual(
                interval["estimand"],
                "equally weighted mean of within-cluster observations",
            )

    def test_validation_headlines_have_paired_multiplicity_aware_intervals(self) -> None:
        uncertainty = self._json("uncertainty.json")
        comparisons = uncertainty["comparisons"]
        assert isinstance(comparisons, list)
        headline = [entry for entry in comparisons if entry["headline"]]

        self.assertEqual(len(headline), 6 * 3)
        self.assertTrue(all(entry["candidate"] == HEADLINE_POLICY for entry in headline))
        self.assertTrue(all(entry["reference"] == MAIN_REFERENCE for entry in headline))
        self.assertTrue(all(entry["clusters"] == 2 for entry in headline))
        self.assertFalse(uncertainty["cross_harness_ranking_permitted"])

    def test_every_metric_trace_hash_resolves_to_a_complete_trace_record(self) -> None:
        metric_hashes = {row["trace_payload_sha256"] for row in self._csv("metrics.csv")}
        metric_hashes.update(
            row["trace_payload_sha256"] for row in self._csv("fidelity_metrics.csv")
        )
        with gzip.open(self.bundle / "traces.jsonl.gz", "rt", encoding="utf-8") as handle:
            records = [json.loads(line) for line in handle]
        record_hashes = {record["payload_sha256"] for record in records}

        self.assertEqual(len(records), 12 * 2 * 9)
        self.assertEqual(metric_hashes, record_hashes)
        self.assertTrue(all(record["measurement_warning"] == WARNING for record in records))

    def test_exact_oracles_verify_and_adversarial_witnesses_hold(self) -> None:
        oracle = self._json("oracle.json")
        rows = oracle["rows"]
        assert isinstance(rows, list)
        self.assertEqual(len(rows), 6)
        self.assertTrue(all(row["verification"]["optimality_checked"] for row in rows))
        adversarial = self._json("adversarial.json")
        cases = adversarial["cases"]
        assert isinstance(cases, list)
        self.assertEqual(len(cases), 3)
        self.assertTrue(all(case["witness"] for case in cases))

    def test_sequential_diagnostic_is_pre_gpu_familywise_and_assumption_bounded(self) -> None:
        diagnostic = self._json("sequential_inference.json")
        self.assertFalse(diagnostic["uses_accelerator_observations"])
        feasibility = diagnostic["feasibility"]
        assert isinstance(feasibility, dict)
        self.assertEqual(feasibility["hypotheses"], 24)
        self.assertGreater(
            feasibility["original_endpointwise_hoeffding_radius_at_maximum"],
            feasibility["target_half_width"],
        )
        calibration = diagnostic["monte_carlo"]
        assert isinstance(calibration, dict)
        self.assertEqual(calibration["trials"], 100)
        scenarios = calibration["scenarios"]
        assert isinstance(scenarios, list)
        by_name = {entry["name"]: entry for entry in scenarios}
        self.assertLess(
            by_name["adversarial-skewed-null"]["primary_all_look_coverage_rate"],
            0.9,
        )
        self.assertGreaterEqual(
            by_name["adversarial-skewed-null"]["sensitivity_all_look_coverage_rate"],
            0.95,
        )

    def test_ci_bundle_is_byte_identical_on_rerun(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            rerun = Path(temporary) / "rerun"
            run_study(mode="ci", output_dir=rerun)
            first = {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in self.bundle.iterdir()
                if path.is_file()
            }
            second = {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in rerun.iterdir()
                if path.is_file()
            }

        self.assertEqual(first, second)

    def test_verifier_fails_closed_after_artifact_corruption(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            corrupted = Path(temporary) / "bundle"
            run_study(mode="ci", output_dir=corrupted)
            design = corrupted / "design.json"
            design.write_bytes(design.read_bytes() + b" ")

            with self.assertRaisesRegex(
                StudyIntegrityError,
                "byte count mismatch",
            ):
                verify_bundle(corrupted)


if __name__ == "__main__":
    unittest.main()
