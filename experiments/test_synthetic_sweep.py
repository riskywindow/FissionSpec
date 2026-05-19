from __future__ import annotations

import csv
import hashlib
import io
import json
import tempfile
import unittest
import xml.etree.ElementTree as ElementTree
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from experiments.run_synthetic_sweep import (
    POLICIES,
    WARNING,
    SweepConfig,
    run_sweep,
    write_results,
)
from experiments.run_synthetic_sweep import (
    main as sweep_main,
)
from tools.render_synthetic_results import (
    load_aggregates,
    render_markdown,
    render_svg,
)


def tiny_config() -> SweepConfig:
    return SweepConfig(
        seeds=(11,),
        request_count=3,
        output_tokens=5,
        speculation_length=4,
        cache_hit_probabilities=(0.65, 0.95),
        token_acceptance_probabilities=(0.50, 0.90),
        tbt_slo_ms=15.0,
        max_batch_size=3,
        poisson_mean_ms=0.4,
        burst_size=2,
        burst_gap_ms=5.0,
        burst_width_ms=0.2,
        coalesce_ms=0.5,
        max_wait_ms=1.0,
    )


class SyntheticSweepTests(unittest.TestCase):
    def test_full_factorial_is_deterministic_and_complete(self) -> None:
        config = tiny_config()
        rows, fingerprints = run_sweep(config)
        repeated_rows, repeated_fingerprints = run_sweep(config)

        expected_rows = 3 * 4 * len(POLICIES)
        self.assertEqual(len(rows), expected_rows)
        self.assertEqual(rows, repeated_rows)
        self.assertEqual(fingerprints, repeated_fingerprints)
        self.assertEqual(len(fingerprints), 3)
        self.assertEqual(
            {row.policy for row in rows},
            set(POLICIES),
        )
        self.assertEqual(
            {
                (
                    row.configured_cache_hit_probability,
                    row.configured_token_acceptance_probability,
                )
                for row in rows
            },
            {(0.65, 0.50), (0.65, 0.90), (0.95, 0.50), (0.95, 0.90)},
        )
        self.assertTrue(
            all(
                row.observed_cache_hits + row.observed_cache_misses > 0
                and 0.0 <= row.observed_cache_hit_rate <= 1.0
                and 1.0 <= row.mean_verifier_tokens_per_round <= 4.0
                for row in rows
            )
        )
        self.assertTrue(
            all(
                len(str(fingerprint["cache_hit_sha256"])) == 64
                and len(str(fingerprint["token_acceptance_sha256"])) == 64
                for fingerprint in fingerprints
            )
        )

    def test_outputs_and_renderers_keep_synthetic_provenance(self) -> None:
        config = tiny_config()
        rows, fingerprints = run_sweep(config)
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            json_path, csv_path = write_results(output_dir, config, rows, fingerprints)
            first_json = json_path.read_bytes()
            first_csv = csv_path.read_bytes()
            write_results(output_dir, config, rows, fingerprints)
            self.assertEqual(json_path.read_bytes(), first_json)
            self.assertEqual(csv_path.read_bytes(), first_csv)

            document = json.loads(json_path.read_text(encoding="utf-8"))
            with csv_path.open(newline="", encoding="utf-8") as handle:
                csv_rows = list(csv.DictReader(handle))
            self.assertEqual(document["measurement_warning"], WARNING)
            self.assertEqual(document["profile"]["gpu_measurement"], False)
            self.assertIn("target_curve", document["profile"])
            self.assertIn("verifier_slot_ms", document["profile"])
            self.assertEqual(document["per_seed_rows"]["row_count"], len(csv_rows))
            self.assertEqual(
                document["per_seed_rows"]["sha256"],
                hashlib.sha256(first_csv).hexdigest(),
            )
            self.assertTrue(all(row["measurement_warning"] == WARNING for row in csv_rows))

            aggregates = load_aggregates(json_path)
            markdown = render_markdown(aggregates)
            svg = render_svg(aggregates)
            self.assertIn(WARNING, markdown)
            self.assertIn(WARNING, svg)
            self.assertIn("Cache p", markdown)
            ElementTree.fromstring(svg)

    def test_checked_in_golden_artifact_matches_current_schema(self) -> None:
        result_dir = Path(__file__).parent / "results"
        json_path = result_dir / "synthetic_sweep.json"
        csv_path = result_dir / "synthetic_sweep.csv"
        with tempfile.TemporaryDirectory() as directory:
            regenerated_dir = Path(directory)
            with (
                patch(
                    "sys.argv",
                    [
                        "run_synthetic_sweep.py",
                        "--output-dir",
                        str(regenerated_dir),
                    ],
                ),
                redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(sweep_main(), 0)
            regenerated_json = regenerated_dir / "synthetic_sweep.json"
            regenerated_csv = regenerated_dir / "synthetic_sweep.csv"
            self.assertEqual(json_path.read_bytes(), regenerated_json.read_bytes())
            self.assertEqual(csv_path.read_bytes(), regenerated_csv.read_bytes())

        document = json.loads(json_path.read_text(encoding="utf-8"))
        self.assertEqual(document["measurement_warning"], WARNING)
        self.assertEqual(document["config"]["tbt_slo_ms"], 10.0)
        self.assertEqual(len(document["method"]["implementation_sha256"]), 64)
        self.assertEqual(
            document["per_seed_rows"]["sha256"],
            hashlib.sha256(csv_path.read_bytes()).hexdigest(),
        )
        aggregates = load_aggregates(json_path)
        self.assertTrue(aggregates)
        self.assertIn("direct_hit_delay_ms", aggregates[0])
        self.assertIn("request_tbt_slo_attainment", aggregates[0])
        self.assertIn("tbt_request_goodput_tokens_per_s", aggregates[0])
        self.assertNotIn("hit_externality_ms", aggregates[0])
        self.assertEqual(
            (result_dir / "SYNTHETIC_RESULTS.md").read_text(encoding="utf-8"),
            render_markdown(aggregates),
        )
        self.assertEqual(
            (result_dir / "synthetic_factorial.svg").read_text(encoding="utf-8"),
            render_svg(aggregates),
        )

    def test_requires_independent_probability_axes(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least two distinct"):
            replace(tiny_config(), cache_hit_probabilities=(0.8,))


if __name__ == "__main__":
    unittest.main()
