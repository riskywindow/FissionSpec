from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from fissionspec.cli import load_profile_json, load_workload_json, main


class CliEvidenceTests(unittest.TestCase):
    @staticmethod
    def _profile_document() -> dict[str, object]:
        return {
            "schema_version": 1,
            "name": "test-profile",
            "provenance": {"engine_commit": "abc123", "gpu": "test-gpu"},
            "fit": {"sample_count": 9, "method": "test"},
            "target_curve": [[1, 1.0], [2, 1.2]],
            "draft_curve": [[1, 0.2], [2, 0.3]],
            "recovery_curve": [[1, 0.4], [2, 0.6]],
            "verifier_slot_ms": 0.01,
        }

    def test_simulation_output_is_self_describing_and_warns(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            status = main(
                [
                    "simulate",
                    "--requests",
                    "1",
                    "--output-tokens",
                    "1",
                    "--indent",
                    "0",
                ]
            )

        document = json.loads(output.getvalue())
        self.assertEqual(status, 0)
        self.assertEqual(document["evidence_class"], "simulation-model")
        self.assertIn("NOT AN END-TO-END GPU", document["measurement_warning"])
        self.assertEqual(document["profile"]["source"], "built-in-synthetic")
        self.assertIn("fissionspec-horizon-2", document["results"])
        self.assertEqual(document["workload"]["requests"][0]["output_tokens"], 1)

    def test_json_loaders_reject_lossy_bool_and_fractional_coercions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workload_path = root / "workload.json"
            workload_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "requests": [
                            {
                                "request_id": "bad",
                                "arrival_ms": 0.0,
                                "output_tokens": 2,
                                "speculation_length": 2,
                                "cache_hit_probability": [True],
                                "token_acceptance_probability": 0.5,
                                "tbt_slo_ms": 10.0,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "finite real"):
                load_workload_json(workload_path)

            workload_document = json.loads(workload_path.read_text(encoding="utf-8"))
            workload_document["schema_version"] = 999
            workload_path.write_text(json.dumps(workload_document), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "schema_version"):
                load_workload_json(workload_path)

            profile_path = root / "profile.json"
            profile = self._profile_document()
            profile["target_curve"] = [[1.9, 1.0]]
            profile_path.write_text(json.dumps(profile), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "row counts must be integers"):
                load_profile_json(profile_path)

            profile["target_curve"] = [[True, 1.0]]
            profile_path.write_text(json.dumps(profile), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "row counts must be integers"):
                load_profile_json(profile_path)

            profile["schema_version"] = 1.0
            profile["target_curve"] = [[1, 1.0]]
            profile_path.write_text(json.dumps(profile), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "schema_version"):
                load_profile_json(profile_path)

    def test_custom_profile_provenance_fit_and_file_hash_survive_cli(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            profile_path = Path(directory) / "profile.json"
            workload_path = Path(directory) / "workload.json"
            profile_path.write_text(json.dumps(self._profile_document()), encoding="utf-8")
            workload_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "name": "custom-workload",
                        "description": "hash-preservation-test",
                        "requests": [
                            {
                                "request_id": "one",
                                "output_tokens": 1,
                                "speculation_length": 1,
                                "cache_hit_probability": 1.0,
                                "token_acceptance_probability": 1.0,
                                "tbt_slo_ms": 10.0,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            output = io.StringIO()
            with redirect_stdout(output):
                status = main(
                    [
                        "simulate",
                        "--profile-json",
                        str(profile_path),
                        "--workload-json",
                        str(workload_path),
                        "--indent",
                        "0",
                    ]
                )
            document = json.loads(output.getvalue())
            profile_output = document["profile"]
            self.assertEqual(status, 0)
            self.assertEqual(profile_output["provenance"]["engine_commit"], "abc123")
            self.assertEqual(profile_output["fit"]["sample_count"], 9)
            self.assertEqual(len(profile_output["source_document_sha256"]), 64)
            self.assertEqual(document["workload"]["schema_version"], 1)
            self.assertEqual(document["workload"]["description"], "hash-preservation-test")
            self.assertEqual(len(document["workload"]["source_document_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
