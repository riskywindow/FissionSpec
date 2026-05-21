from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from typing import ClassVar, cast

from experiments.run_oracle_campaign import (
    POLICY_NAMES,
    CampaignResult,
    OracleCampaignError,
    _active_jobs,
    _limits,
    campaign_specs,
    compile_problem,
    controller_settings,
    full_scenario_specs,
    mode_config,
    run_campaign,
    verify_bundle,
    write_bundle,
)
from fissionspec.general_oracle import (
    objective_gap,
    solve_general_oracle,
    work_conserving_edf,
)


class OracleCampaignDesignTests(unittest.TestCase):
    def test_full_matrix_expands_six_certificates_to_216_unique_inputs(self) -> None:
        specs = full_scenario_specs()
        self.assertEqual(len(specs), 216)
        self.assertEqual(len({spec.scenario_id for spec in specs}), 216)
        hashes = {compile_problem(spec).input_hash for spec in specs}
        self.assertEqual(len(hashes), 216)

    def test_pre_realized_failure_and_cancellation_have_explicit_boundaries(self) -> None:
        specs = full_scenario_specs()
        canceled = next(spec for spec in specs if spec.outcome_pattern == "canceled-tail")
        terminal = next(spec for spec in specs if spec.outcome_pattern == "terminal-failure-middle")
        retry = next(spec for spec in specs if spec.outcome_pattern == "retry-once")
        self.assertEqual(tuple(job.job_id for job in _active_jobs(canceled)), ("a", "b"))
        self.assertEqual(tuple(job.job_id for job in _active_jobs(terminal)), ("a", "c"))
        self.assertEqual(len(_active_jobs(retry)), 3)

    def test_exact_oracle_never_loses_to_edf_in_lexicographic_order(self) -> None:
        spec = campaign_specs("ci")[0]
        problem = compile_problem(spec)
        certificate = solve_general_oracle(
            problem,
            limits=_limits(),
        )
        edf = work_conserving_edf(problem)
        gap = objective_gap(edf.objective, certificate.objective)
        self.assertGreaterEqual(gap.deadline_violation_gap, 0)
        if gap.deadline_violation_gap == 0:
            self.assertGreaterEqual(gap.weighted_flow_gap, 0)

    def test_full_independent_proofs_preserve_strata_within_node_budget(self) -> None:
        from experiments.run_oracle_campaign import _proof_ids

        specs = full_scenario_specs()
        proof_ids = _proof_ids(specs, 24)
        proof_specs = tuple(spec for spec in specs if spec.scenario_id in proof_ids)
        self.assertEqual(len(proof_specs), 24)
        self.assertEqual(
            {spec.outcome_pattern for spec in proof_specs},
            {
                "head-miss",
                "tail-miss",
                "double-miss",
                "retry-once",
                "canceled-tail",
                "terminal-failure-middle",
            },
        )
        self.assertEqual(
            {spec.physical_mode for spec in proof_specs},
            {"packed-slots", "graph-bucket-slots"},
        )
        self.assertEqual(
            {spec.capacity_mode for spec in proof_specs},
            {"tight", "wide"},
        )
        self.assertFalse(
            any(
                len(_active_jobs(spec)) == 3 and spec.physical_mode == "packed-slots"
                for spec in proof_specs
            )
        )


class OracleCampaignExecutionTests(unittest.TestCase):
    result: ClassVar[CampaignResult]

    @classmethod
    def setUpClass(cls) -> None:
        cls.result = run_campaign(mode_config("ci"))

    def test_ci_campaign_covers_every_policy_and_retains_counterexamples(self) -> None:
        result = self.result
        expected = len(campaign_specs("ci")) * len(controller_settings()) * len(POLICY_NAMES)
        self.assertEqual(len(result.comparisons), expected)
        self.assertEqual(
            {str(row["policy"]) for row in result.comparisons},
            set(POLICY_NAMES),
        )
        self.assertTrue(result.counterexamples)
        self.assertEqual(
            result.coverage["unique_problem_hashes"],
            len(campaign_specs("ci")),
        )

    def test_dispatch_component_aliases_are_explicitly_validated(self) -> None:
        validations = cast(
            dict[str, object],
            self.result.coverage["metamorphic_and_adversarial_validations"],
        )
        self.assertEqual(
            validations["dispatch_component_alias_groups"],
            len(campaign_specs("ci")) * len(controller_settings()),
        )
        self.assertGreater(
            int(cast(int, validations["input_order_and_latency_map_order_invariance_cases"])),
            0,
        )

    def test_ci_bundle_is_byte_deterministic_and_tamper_evident(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        with (
            tempfile.TemporaryDirectory() as left_directory,
            tempfile.TemporaryDirectory() as right_directory,
        ):
            left = Path(left_directory)
            right = Path(right_directory)
            write_bundle(left, mode_config("ci"), repo_root=repo_root)
            write_bundle(right, mode_config("ci"), repo_root=repo_root)
            self.assertEqual(
                sorted(path.name for path in left.iterdir()),
                sorted(path.name for path in right.iterdir()),
            )
            for path in left.iterdir():
                self.assertEqual(path.read_bytes(), (right / path.name).read_bytes())
            self.assertNotIn(b"\r", (left / "comparisons.csv").read_bytes())
            self.assertFalse(
                any(
                    line.endswith((b" ", b"\t"))
                    for line in (left / "SUMMARY.md").read_bytes().splitlines()
                )
            )
            verify_bundle(left, expected_mode="ci", repo_root=repo_root)
            comparisons = left / "comparisons.csv"
            with comparisons.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            rows[0]["candidate_trace_sha256"] = "0" * 64
            with comparisons.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=rows[0])
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(OracleCampaignError, "hash mismatch"):
                verify_bundle(left, expected_mode="ci", repo_root=repo_root)
            manifest_path = left / "manifest.json"
            manifest = cast(
                dict[str, object],
                json.loads(manifest_path.read_text(encoding="utf-8")),
            )
            artifact_files = cast(
                dict[str, object],
                manifest["artifact_files"],
            )
            comparison_record = cast(
                dict[str, object],
                artifact_files["comparisons.csv"],
            )
            payload = comparisons.read_bytes()
            comparison_record["bytes"] = len(payload)
            comparison_record["sha256"] = hashlib.sha256(payload).hexdigest()
            manifest_path.write_text(
                json.dumps(
                    manifest,
                    indent=2,
                    sort_keys=True,
                    ensure_ascii=False,
                    allow_nan=False,
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(OracleCampaignError, "semantic artifact mismatch"):
                verify_bundle(left, expected_mode="ci", repo_root=repo_root)

            environment_path = right / "environment.json"
            environment = cast(
                dict[str, object],
                json.loads(environment_path.read_text(encoding="utf-8")),
            )
            observed_runtime = cast(
                dict[str, object],
                environment["observed_runtime"],
            )
            observed_runtime.update(
                {
                    "platform": "Linux-archival-verifier",
                    "machine": "x86_64",
                    "processor": "",
                    "logical_cpu_count": 8,
                }
            )
            environment_path.write_text(
                json.dumps(
                    environment,
                    indent=2,
                    sort_keys=True,
                    ensure_ascii=False,
                    allow_nan=False,
                )
                + "\n",
                encoding="utf-8",
            )
            right_manifest_path = right / "manifest.json"
            right_manifest = cast(
                dict[str, object],
                json.loads(right_manifest_path.read_text(encoding="utf-8")),
            )
            right_artifact_files = cast(
                dict[str, object],
                right_manifest["artifact_files"],
            )
            environment_record = cast(
                dict[str, object],
                right_artifact_files["environment.json"],
            )
            environment_payload = environment_path.read_bytes()
            environment_record["bytes"] = len(environment_payload)
            environment_record["sha256"] = hashlib.sha256(environment_payload).hexdigest()
            right_manifest_path.write_text(
                json.dumps(
                    right_manifest,
                    indent=2,
                    sort_keys=True,
                    ensure_ascii=False,
                    allow_nan=False,
                )
                + "\n",
                encoding="utf-8",
            )
            verify_bundle(right, expected_mode="ci", repo_root=repo_root)

            right_comparisons = right / "comparisons.csv"
            right_comparisons.unlink()
            right_comparisons.symlink_to(comparisons)
            with self.assertRaisesRegex(
                OracleCampaignError,
                "regular non-symlink file",
            ):
                verify_bundle(right, expected_mode="ci", repo_root=repo_root)

    def test_design_declares_exclusions_and_no_selection(self) -> None:
        design = self.result.design
        boundary = cast(dict[str, object], design["tractability_boundary"])
        excluded = cast(list[str], boundary["excluded"])
        self.assertIn("multi-round token generation", excluded)
        self.assertIn("SPECTRE padded-recovery execution cost", excluded)
        self.assertIn("Every predeclared", str(design["selection"]))
        self.assertNotIn(str(Path.cwd()), str(design))


if __name__ == "__main__":
    unittest.main()
