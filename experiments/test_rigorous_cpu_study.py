from __future__ import annotations

import csv
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from typing import cast

from experiments.run_rigorous_cpu_study import (
    CLAIM_BOUNDARY,
    SCHEMA_VERSION,
    WARNING,
    MetricSpec,
    PrecisionSpec,
    StudyConfig,
    StudyError,
    analyze_csv,
    main,
    write_document,
)

FIELDS = (
    "evidence_class",
    "measurement_warning",
    "workload",
    "regime",
    "seed",
    "policy",
    "throughput_tokens_per_s",
    "p95_tbt_ms",
)


def _rows() -> list[dict[str, str | float | int]]:
    rows: list[dict[str, str | float | int]] = []
    for seed, baseline_throughput, candidate_throughput, baseline_tbt, candidate_tbt in (
        (7, 100.0, 105.0, 10.0, 9.0),
        (17, 110.0, 112.0, 11.0, 10.0),
        (29, 120.0, 125.0, 12.0, 10.0),
        (41, 130.0, 129.0, 13.0, 12.0),
    ):
        common: dict[str, str | float | int] = {
            "evidence_class": "synthetic-model",
            "measurement_warning": "SYNTHETIC MODEL OUTPUT — NOT GPU MEASUREMENTS.",
            "workload": "bursty",
            "regime": "cache-high_acceptance-high",
            "seed": seed,
        }
        rows.append(
            {
                **common,
                "policy": "saguaro-barrier",
                "throughput_tokens_per_s": baseline_throughput,
                "p95_tbt_ms": baseline_tbt,
            }
        )
        rows.append(
            {
                **common,
                "policy": "fissionspec-horizon-2",
                "throughput_tokens_per_s": candidate_throughput,
                "p95_tbt_ms": candidate_tbt,
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, str | float | int]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _config() -> StudyConfig:
    return StudyConfig(
        candidate_policy="fissionspec-horizon-2",
        baseline_policy="saguaro-barrier",
        metrics=(
            MetricSpec("throughput_tokens_per_s", "higher"),
            MetricSpec("p95_tbt_ms", "lower"),
        ),
        resamples=300,
        bootstrap_seed="rigorous-study-test",
        familywise_alpha=0.05,
        target_power=0.8,
        minimum_detectable_standardized_effect=0.5,
        precision_specs=(
            PrecisionSpec(
                metric="throughput_tokens_per_s",
                lower_difference=-20.0,
                upper_difference=20.0,
                target_half_width=5.0,
                minimum_observations=2,
            ),
        ),
    )


class RigorousCPUStudyTests(unittest.TestCase):
    def test_analysis_is_paired_deterministic_and_explicitly_not_gpu(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "rows.csv"
            _write_csv(input_path, _rows())
            first = analyze_csv(input_path, _config())
            repeated = analyze_csv(input_path, _config())
        self.assertEqual(first, repeated)
        self.assertEqual(first["schema_version"], SCHEMA_VERSION)
        self.assertEqual(first["measurement_warning"], WARNING)
        self.assertEqual(first["claim_boundary"], CLAIM_BOUNDARY)
        self.assertEqual(first["inference_status"], "exploratory-not-confirmatory")

        input_metadata = cast(dict[str, object], first["input"])
        self.assertEqual(input_metadata["row_count"], 8)
        self.assertEqual(len(cast(str, input_metadata["sha256"])), 64)
        design = cast(dict[str, object], first["design"])
        multiplicity = cast(dict[str, object], design["multiplicity"])
        self.assertEqual(multiplicity["method"], "bonferroni-simultaneous-confidence-intervals")
        self.assertEqual(multiplicity["per_hypothesis_alpha"], 0.025)

        comparisons = cast(list[dict[str, object]], first["comparisons"])
        self.assertEqual(len(comparisons), 2)
        by_metric = {str(item["metric"]): item for item in comparisons}
        throughput_effect = cast(dict[str, object], by_metric["throughput_tokens_per_s"]["effect"])
        tbt_effect = cast(dict[str, object], by_metric["p95_tbt_ms"]["effect"])
        self.assertGreater(
            cast(float, throughput_effect["oriented_mean_improvement"]),
            0.0,
        )
        self.assertGreater(cast(float, tbt_effect["oriented_mean_improvement"]), 0.0)
        self.assertEqual(by_metric["throughput_tokens_per_s"]["independent_clusters"], 4)
        self.assertEqual(
            cast(dict[str, object], by_metric["throughput_tokens_per_s"]["precision_stopping"])[
                "status"
            ],
            "run",
        )
        self.assertEqual(
            cast(dict[str, object], by_metric["p95_tbt_ms"]["precision_stopping"])["status"],
            "not-run",
        )
        replication = cast(
            dict[str, object],
            by_metric["throughput_tokens_per_s"]["replication_plan"],
        )
        self.assertEqual(replication["hypotheses"], 2)
        self.assertGreater(cast(int, replication["recommended_replications"]), 4)

    def test_pairing_failures_and_bad_evidence_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "unpaired.csv"
            rows = _rows()
            rows.pop()
            _write_csv(input_path, rows)
            with self.assertRaisesRegex(StudyError, "unpaired seeds"):
                analyze_csv(input_path, _config())

            bad_path = Path(directory) / "bad-evidence.csv"
            bad_rows = _rows()
            bad_rows[0]["measurement_warning"] = "benchmark"
            _write_csv(bad_path, bad_rows)
            with self.assertRaisesRegex(StudyError, "not-GPU"):
                analyze_csv(bad_path, _config())

            duplicate_path = Path(directory) / "duplicate.csv"
            duplicate_rows = _rows()
            duplicate_rows.append(dict(duplicate_rows[0]))
            _write_csv(duplicate_path, duplicate_rows)
            with self.assertRaisesRegex(StudyError, "duplicate policy row"):
                analyze_csv(duplicate_path, _config())

    def test_writer_protects_outputs_and_cli_emits_evidence_schema(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "rows.csv"
            output_path = root / "study.json"
            _write_csv(input_path, _rows())
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                status = main(
                    [
                        "--input",
                        str(input_path),
                        "--output",
                        str(output_path),
                        "--candidate",
                        "fissionspec-horizon-2",
                        "--baseline",
                        "saguaro-barrier",
                        "--metric",
                        "throughput_tokens_per_s:higher",
                        "--resamples",
                        "100",
                    ]
                )
            self.assertEqual(status, 0)
            self.assertIn(WARNING, stdout.getvalue())
            document = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(document["measurement_warning"], WARNING)
            with self.assertRaises(FileExistsError):
                write_document(output_path, document)
            write_document(output_path, document, force=True)


if __name__ == "__main__":
    unittest.main()
