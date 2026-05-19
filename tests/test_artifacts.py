from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from fissionspec.artifacts import (
    SIMULATION_WARNING,
    ArtifactIntegrityError,
    canonical_json_bytes,
    implementation_sha256,
    load_trace_document,
    simulation_trace_document,
    verify_trace_document,
    write_trace_document,
)
from fissionspec.model import SimulationResult
from fissionspec.policies import ImmediateFissionPolicy
from fissionspec.profiles import HardwareProfile
from fissionspec.rng import CounterRNG
from fissionspec.simulator import simulate
from fissionspec.workload import Workload


def _result() -> SimulationResult:
    workload = Workload.homogeneous(
        5,
        arrival_interval_ms=0.3,
        output_tokens=9,
        cache_hit_probability=0.7,
        token_acceptance_probability=0.6,
        name="artifact-test",
    )
    return simulate(
        workload,
        HardwareProfile.linear(name="artifact-profile"),
        ImmediateFissionPolicy(),
        CounterRNG("artifact-seed"),
        max_batch_size=3,
    )


class ArtifactTests(unittest.TestCase):
    def test_full_trace_is_canonical_and_byte_reproducible(self) -> None:
        result = _result()
        first = simulation_trace_document(result)
        second = simulation_trace_document(result)
        self.assertEqual(canonical_json_bytes(first), canonical_json_bytes(second))
        self.assertEqual(verify_trace_document(first), first["payload_sha256"])
        self.assertEqual(first["measurement_warning"], SIMULATION_WARNING)
        trace = first["trace"]
        self.assertIsInstance(trace, dict)
        self.assertEqual(len(trace["requests"]), 5)
        self.assertGreater(len(trace["target_launches"]), 0)
        self.assertGreater(len(trace["draft_launches"]), 0)

    def test_any_payload_mutation_is_detected(self) -> None:
        document = simulation_trace_document(_result())
        tampered = copy.deepcopy(document)
        summary = tampered["summary"]
        self.assertIsInstance(summary, dict)
        summary["output_tokens"] = 999
        with self.assertRaises(ArtifactIntegrityError):
            verify_trace_document(tampered)

    def test_write_and_load_round_trip_strictly(self) -> None:
        document = simulation_trace_document(_result())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.json"
            write_trace_document(path, document)
            self.assertEqual(path.read_bytes(), canonical_json_bytes(document))
            self.assertEqual(
                canonical_json_bytes(load_trace_document(path)),
                canonical_json_bytes(document),
            )
            raw = json.loads(path.read_text(encoding="utf-8"))
            raw["payload_sha256"] = "0" * 64
            path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaises(ArtifactIntegrityError):
                load_trace_document(path)

    def test_implementation_hash_frames_names_and_content(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a").write_bytes(b"bc")
            (root / "ab").write_bytes(b"c")
            left = implementation_sha256(root, ("a", "ab"))
            right = implementation_sha256(root, ("ab", "a"))
            self.assertEqual(left, right)
            (root / "a").write_bytes(b"different")
            self.assertNotEqual(left, implementation_sha256(root, ("a", "ab")))
            with self.assertRaises(ValueError):
                implementation_sha256(root, ())


if __name__ == "__main__":
    unittest.main()
