from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import cast

from experiments.run_controller_overhead import (
    COMPLEXITY_WARNING,
    SIZES,
    WARNING,
    ControllerOverheadError,
    build_complexity_artifact,
    run_python_timing,
    verify_artifacts,
    write_artifacts,
)


class ControllerOverheadTests(unittest.TestCase):
    def test_complexity_counts_scale_and_call_production_policy(self) -> None:
        artifact = build_complexity_artifact()
        self.assertEqual(artifact["measurement_warning"], COMPLEXITY_WARNING)
        samples = cast(list[dict[str, object]], artifact["samples"])
        self.assertEqual(
            [sample["current_rows"] for sample in samples],
            list(SIZES),
        )
        for sample in samples:
            size = int(cast(int, sample["current_rows"]))
            python = cast(dict[str, object], sample["python_source_audit"])
            rust = cast(dict[str, object], sample["rust_source_audit"])
            self.assertEqual(python["global_sort_key_evaluations"], 2 * size)
            self.assertEqual(rust["current_item_visits"], 8 * size)
            self.assertEqual(rust["future_item_visits"], 7 * size)
            self.assertIn(sample["production_python_decision"], {"re-fuse", "dispatch-now"})

    def test_small_python_timing_retains_raw_repeats(self) -> None:
        timing = run_python_timing(target_row_visits=10_000, repeats=2)
        samples = cast(list[dict[str, object]], timing["samples"])
        self.assertEqual(len(samples), len(SIZES))
        for sample in samples:
            elapsed = cast(list[int], sample["elapsed_ns_repeats"])
            self.assertEqual(len(elapsed), 2)
            self.assertTrue(all(value > 0 for value in elapsed))
            self.assertGreater(cast(float, sample["median_ns_per_decision"]), 0.0)

    def test_structural_bundle_is_reproducible_and_tamper_evident(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        with (
            tempfile.TemporaryDirectory() as left_directory,
            tempfile.TemporaryDirectory() as right_directory,
        ):
            left = Path(left_directory)
            right = Path(right_directory)
            write_artifacts(left, repo_root=repo_root, structural_only=True)
            write_artifacts(right, repo_root=repo_root, structural_only=True)
            self.assertEqual(
                (left / "controller_complexity.json").read_bytes(),
                (right / "controller_complexity.json").read_bytes(),
            )
            verify_artifacts(left)
            complexity_path = left / "controller_complexity.json"
            document = json.loads(complexity_path.read_text(encoding="utf-8"))
            document["samples"][0]["rust_source_audit"]["batch_view_item_visits"] = -1
            complexity_path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(ControllerOverheadError, "hash mismatch"):
                verify_artifacts(left)

    def test_manifest_is_self_hashed_and_closed(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            write_artifacts(output, repo_root=repo_root, structural_only=True)
            manifest_path = output / "controller_overhead_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["measurement_warning"] = "tampered"
            manifest_path.write_text(
                json.dumps(manifest, sort_keys=True),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ControllerOverheadError, "payload hash mismatch"):
                verify_artifacts(output)

    def test_timing_bundle_is_labeled_and_not_compared_across_languages(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            write_artifacts(
                output,
                repo_root=repo_root,
                target_row_visits=10_000,
                repeats=1,
                include_rust=False,
            )
            timing = json.loads(
                (output / "controller_overhead_local.json").read_text(encoding="utf-8")
            )
            self.assertEqual(timing["measurement_warning"], WARNING)
            self.assertIn("no cross-language ratio", timing["claim_boundary"])
            self.assertFalse(timing["environment"]["timing_controls"]["process_affinity_pinned"])
            self.assertNotIn(str(repo_root), json.dumps(timing))


if __name__ == "__main__":
    unittest.main()
