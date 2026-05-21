from __future__ import annotations

import csv
import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from typing import ClassVar, cast

from experiments.run_mechanism_study import (
    DECODER_CONFIRMATORY_METRICS,
    FIDELITY_CONFIRMATORY_METRICS,
    WARNING,
    MechanismStudyError,
    MetricRow,
    ModeConfig,
    _apply_decoder,
    _apply_fidelity,
    all_interventions,
    analyze_rows,
    decoder_interventions,
    decoder_reference,
    design_document,
    environment_document,
    fidelity_interventions,
    fidelity_reference,
    generate_rows,
    mode_config,
    verify_bundle,
    write_bundle,
)


class MechanismStudyTests(unittest.TestCase):
    config: ClassVar[ModeConfig]
    rows: ClassVar[list[MetricRow]]

    @classmethod
    def setUpClass(cls) -> None:
        cls.config = mode_config("ci")
        cls.rows = generate_rows(cls.config)

    def test_every_contrast_changes_exactly_one_declared_field(self) -> None:
        for intervention in all_interventions():
            if intervention.stratum == "decoder-policy":
                reference = asdict(decoder_reference())
                alternate = asdict(_apply_decoder(intervention))
            else:
                reference = asdict(fidelity_reference())
                alternate = asdict(_apply_fidelity(intervention))
            changed = [field for field in reference if reference[field] != alternate[field]]
            self.assertEqual(changed, [intervention.changed_field])
            self.assertEqual(
                reference[intervention.changed_field],
                intervention.reference_value,
            )
            self.assertEqual(
                alternate[intervention.changed_field],
                intervention.intervention_value,
            )

    def test_rows_are_complete_paired_and_have_trace_hashes(self) -> None:
        expected = len(self.config.seeds) * (
            2 + len(decoder_interventions()) + len(fidelity_interventions())
        )
        self.assertEqual(len(self.rows), expected)
        keys = {(row["stratum"], row["setting_id"], row["seed"]) for row in self.rows}
        self.assertEqual(len(keys), expected)
        for row in self.rows:
            self.assertEqual(row["measurement_warning"], WARNING)
            self.assertRegex(str(row["trace_payload_sha256"]), r"^[0-9a-f]{64}$")
        for seed in self.config.seeds:
            self.assertIn(("decoder-policy", "decoder/reference", seed), keys)
            self.assertIn(
                ("one-round-fidelity", "fidelity/reference", seed),
                keys,
            )

    def test_analysis_declares_one_complete_family_and_retains_nulls(self) -> None:
        analysis = analyze_rows(self.rows, self.config)
        family = cast(dict[str, object], analysis["family"])
        comparisons = cast(list[dict[str, object]], analysis["comparisons"])
        expected_hypotheses = len(decoder_interventions()) * len(
            DECODER_CONFIRMATORY_METRICS
        ) + len(fidelity_interventions()) * len(FIDELITY_CONFIRMATORY_METRICS)
        self.assertEqual(expected_hypotheses, 48)
        self.assertEqual(family["hypotheses"], expected_hypotheses)
        self.assertEqual(len(comparisons), expected_hypotheses)
        self.assertAlmostEqual(
            cast(float, family["per_hypothesis_alpha"]),
            0.05 / expected_hypotheses,
        )
        for comparison in comparisons:
            interval = cast(dict[str, object], comparison["simultaneous_interval"])
            self.assertLessEqual(
                cast(float, interval["lower"]),
                cast(float, interval["upper"]),
            )
            self.assertIn("holm_adjusted_sign_pvalue", comparison)

    def test_ci_generation_is_deterministic_and_verifiable(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        with (
            tempfile.TemporaryDirectory() as left_directory,
            tempfile.TemporaryDirectory() as right_directory,
        ):
            left = Path(left_directory)
            right = Path(right_directory)
            write_bundle(left, self.config, repo_root=repo_root)
            write_bundle(right, self.config, repo_root=repo_root)
            for filename in ("design.json", "inference.json", "rows.csv", "SUMMARY.md"):
                self.assertEqual(
                    (left / filename).read_bytes(),
                    (right / filename).read_bytes(),
                )
            manifest = verify_bundle(
                left,
                expected_mode="ci",
                repo_root=repo_root,
            )
            self.assertEqual(manifest["mode"], "ci")

    def test_verifier_rejects_tampering_and_non_ofat_design(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            write_bundle(output, self.config, repo_root=repo_root)
            rows_path = output / "rows.csv"
            with rows_path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            rows[0]["trace_payload_sha256"] = "0" * 64
            with rows_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=rows[0])
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(MechanismStudyError, "hash mismatch"):
                verify_bundle(output, expected_mode="ci")

    def test_manifest_is_self_hashed_and_closed(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            write_bundle(output, self.config, repo_root=repo_root)
            manifest_path = output / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["rows"] += 1
            manifest_path.write_text(
                json.dumps(manifest, sort_keys=True),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(MechanismStudyError, "payload hash mismatch"):
                verify_bundle(output, expected_mode="ci")

    def test_design_has_no_configuration_selection_rule(self) -> None:
        design = design_document(self.config)
        self.assertFalse(design["configuration_search"])
        interventions = cast(list[dict[str, object]], design["interventions"])
        self.assertEqual(len(interventions), 16)
        self.assertTrue(
            all(item["changed_fields_audit"] == [item["changed_field"]] for item in interventions)
        )
        serialized = json.dumps(design)
        self.assertIn("not estimate CUDA-kernel", serialized)
        self.assertIn("miss/hit recovery probes", serialized)

    def test_environment_provenance_is_path_independent(self) -> None:
        environment = environment_document("full")
        contract = cast(dict[str, object], environment["reproduction_contract"])
        self.assertEqual(contract["archival_python_version"], "3.12.8")
        self.assertEqual(
            contract["command_template"],
            [
                "python",
                "experiments/run_mechanism_study.py",
                "--mode",
                "full",
                "--output-dir",
                "<OUTPUT_DIR>",
            ],
        )
        self.assertNotIn(str(Path.cwd()), json.dumps(environment))


if __name__ == "__main__":
    unittest.main()
