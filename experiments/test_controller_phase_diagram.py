from __future__ import annotations

import json
import tempfile
import unittest
import xml.etree.ElementTree as ElementTree
from pathlib import Path
from typing import cast

from experiments.run_controller_phase_diagram import (
    CLAIM_BOUNDARY,
    SCHEMA_VERSION,
    WARNING,
    build_artifact,
    render_json,
    render_svg,
    write_artifact,
)


class ControllerPhaseDiagramTests(unittest.TestCase):
    def setUp(self) -> None:
        self.document = build_artifact()

    def _panel(self, eta_ms: float) -> dict[str, object]:
        panels = cast(list[dict[str, object]], self.document["panels"])
        for panel in panels:
            if panel["recovery_eta_from_now_ms"] == eta_ms:
                return panel
        self.fail(f"missing ETA panel {eta_ms}")

    def _boundary_decision(self, check_id: str) -> str:
        checks = cast(list[dict[str, object]], self.document["boundary_checks"])
        for check in checks:
            if check["id"] == check_id:
                return str(check["decision"])
        self.fail(f"missing boundary check {check_id}")

    def test_schema_and_provenance_are_explicit(self) -> None:
        self.assertEqual(
            set(self.document),
            {
                "schema_version",
                "artifact",
                "evidence_class",
                "measurement_warning",
                "claim_boundary",
                "comparison",
                "controller",
                "axes_and_constraints",
                "hardware_profile",
                "legend",
                "panels",
                "summary",
                "boundary_checks",
            },
        )
        self.assertEqual(self.document["schema_version"], SCHEMA_VERSION)
        self.assertEqual(self.document["measurement_warning"], WARNING)
        self.assertEqual(self.document["claim_boundary"], CLAIM_BOUNDARY)
        profile = cast(dict[str, object], self.document["hardware_profile"])
        self.assertFalse(profile["gpu_measurement"])
        self.assertEqual(
            profile["target_curve_rows_to_ms"],
            [[1, 2.1], [4, 2.8], [8, 3.8], [16, 5.9], [32, 10.5]],
        )
        self.assertEqual(profile["verifier_slot_ms"], 0.018)

    def test_near_eta_can_refuse_and_late_eta_dispatches(self) -> None:
        near_matrix = cast(
            list[list[str]],
            self._panel(0.05)["decision_matrix_current_by_future"],
        )
        late_matrix = cast(
            list[list[str]],
            self._panel(1.5)["decision_matrix_current_by_future"],
        )
        self.assertEqual(near_matrix[0][0], "re-fuse")
        self.assertEqual(late_matrix[0][0], "dispatch-now")
        self.assertGreater(
            sum(cell == "re-fuse" for row in near_matrix for cell in row),
            sum(cell == "re-fuse" for row in late_matrix for cell in row),
        )

    def test_max_wait_deadline_and_capacity_boundaries(self) -> None:
        self.assertEqual(self._boundary_decision("eta-past-max-wait"), "dispatch-now")
        self.assertEqual(
            self._boundary_decision("oldest-row-already-at-max-wait"),
            "dispatch-now",
        )
        self.assertEqual(
            self._boundary_decision("current-deadline-forces-dispatch"),
            "dispatch-now",
        )
        self.assertEqual(
            self._boundary_decision("future-deadline-favors-refusion"),
            "re-fuse",
        )
        self.assertEqual(self._boundary_decision("capacity-forces-dispatch"), "dispatch-now")
        past_horizon = cast(
            list[list[str]],
            self._panel(2.01)["decision_matrix_current_by_future"],
        )
        self.assertTrue(all(cell == "dispatch-now" for row in past_horizon for cell in row))

    def test_immediate_baseline_is_evaluated_at_now_in_every_boundary(self) -> None:
        checks = cast(list[dict[str, object]], self.document["boundary_checks"])
        for check in checks:
            context = cast(dict[str, object], check["context"])
            self.assertEqual(
                check["immediate_fission_dispatch_at_ms"],
                context["now_ms"],
            )

    def test_rendering_is_deterministic_and_matches_checked_in_golden(self) -> None:
        result_dir = Path(__file__).parent / "results"
        expected_json = render_json(self.document)
        expected_svg = render_svg(self.document)
        self.assertEqual(
            (result_dir / "controller_phase_diagram.json").read_text(encoding="utf-8"),
            expected_json,
        )
        self.assertEqual(
            (result_dir / "controller_phase_diagram.svg").read_text(encoding="utf-8"),
            expected_svg,
        )
        parsed = json.loads(expected_json)
        self.assertEqual(parsed["schema_version"], SCHEMA_VERSION)
        ElementTree.fromstring(expected_svg)
        self.assertIn(WARNING, expected_svg)
        self.assertIn("NOT THROUGHPUT EVIDENCE", expected_svg)

        with tempfile.TemporaryDirectory() as directory:
            json_path, svg_path = write_artifact(Path(directory))
            self.assertEqual(json_path.read_text(encoding="utf-8"), expected_json)
            self.assertEqual(svg_path.read_text(encoding="utf-8"), expected_svg)


if __name__ == "__main__":
    unittest.main()
