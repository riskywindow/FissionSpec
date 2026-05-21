"""Tests for the optional installed-stack CPU transformer semantics gate."""

from __future__ import annotations

import copy
import importlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from typing import ClassVar

DEPENDENCIES_AVAILABLE = (
    importlib.util.find_spec("torch") is not None
    and importlib.util.find_spec("transformers") is not None
)
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = REPOSITORY_ROOT / "tools/run_cpu_transformer_semantics.py"
EVIDENCE_PATH = REPOSITORY_ROOT / "experiments/results/cpu_transformer_semantics/evidence.json"


class StaticNoDownloadContractTests(unittest.TestCase):
    def test_tool_has_no_pretrained_or_tokenizer_loader(self) -> None:
        source = TOOL_PATH.read_text(encoding="utf-8")

        self.assertNotIn(".from_pretrained(", source)
        self.assertNotIn("AutoTokenizer", source)
        self.assertIn('DEVICE_NAME: Final = "cpu"', source)
        self.assertIn('os.environ["HF_HUB_OFFLINE"] = "1"', source)
        self.assertIn('os.environ["CUDA_VISIBLE_DEVICES"] = ""', source)
        self.assertNotIn("\nimport torch", source)
        self.assertNotIn("\nimport transformers", source)


class CpuTransformerEvidenceVerifierTests(unittest.TestCase):
    temporary: ClassVar[tempfile.TemporaryDirectory[str]]
    module: ClassVar[ModuleType]
    first_path: ClassVar[Path]
    first_digest: ClassVar[str]
    evidence: ClassVar[dict[str, object]]

    @classmethod
    def setUpClass(cls) -> None:
        cls.module = importlib.import_module("tools.run_cpu_transformer_semantics")
        cls.temporary = tempfile.TemporaryDirectory()
        cls.first_path = EVIDENCE_PATH
        cls.first_digest = cls.module.verify_evidence(cls.first_path)
        document = json.loads(cls.first_path.read_bytes())
        if not isinstance(document, dict):
            raise AssertionError("evidence root must be an object")
        cls.evidence = document

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def _write_rehashed(self, document: dict[str, object], label: str) -> Path:
        payload = copy.deepcopy(document)
        payload.pop("payload_sha256", None)
        payload["payload_sha256"] = self.module._sha256_document(payload)
        path = Path(self.temporary.name) / f"tampered/{label}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.module._canonical_json_bytes(payload))
        return path

    def test_verify_only_is_stdlib_only_with_site_packages_disabled(self) -> None:
        environment = dict(os.environ)
        environment["PYTHONNOUSERSITE"] = "1"
        completed = subprocess.run(
            (
                sys.executable,
                "-S",
                str(TOOL_PATH),
                "--verify-only",
                str(self.first_path),
            ),
            cwd=REPOSITORY_ROOT,
            env=environment,
            capture_output=True,
            check=False,
            text=True,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn(self.first_digest, completed.stdout)

    def test_verifier_rejects_rehashed_claim_contract_mutations(self) -> None:
        def mutate(
            label: str,
            path: tuple[str, ...],
            value: object,
        ) -> tuple[str, dict[str, object]]:
            document = copy.deepcopy(self.evidence)
            target: dict[str, object] = document
            for component in path[:-1]:
                nested = target[component]
                assert isinstance(nested, dict)
                target = nested
            target[path[-1]] = value
            return label, document

        cases = [
            mutate("claim-boundary", ("claim_boundary",), "expanded claim"),
            mutate(
                "pretrained",
                ("offline_contract", "pretrained_weights_loaded"),
                True,
            ),
            mutate("tokenizer", ("offline_contract", "tokenizer_loaded"), True),
            mutate(
                "audited-ops",
                ("device_contract", "dispatch_operations_audited"),
                0,
            ),
            mutate(
                "mps-operations",
                ("device_contract", "mps_tensor_operations_observed"),
                1,
            ),
            mutate("device", ("device_contract", "requested_device"), "mps"),
            mutate("torch-version", ("runtime", "torch"), "0.0"),
            mutate("runtime-shape", ("runtime", "python"), "unknown"),
            mutate(
                "tolerance",
                ("equivalence", "absolute_tolerance"),
                1.0,
            ),
            mutate(
                "valid-path-difference",
                (
                    "equivalence",
                    "reinsert_vs_monolithic_full_forward_max_abs",
                ),
                1.0,
            ),
            mutate(
                "greedy-equivalence",
                ("equivalence", "greedy_tokens_exact"),
                False,
            ),
            mutate(
                "parked-cache",
                ("cache_ownership", "parked_cache_byte_identical"),
                False,
            ),
            mutate(
                "reinsert-order",
                ("cache_ownership", "reinsert_order"),
                ["request-a", "request-b", "request-c"],
            ),
            mutate(
                "negative-control-delta",
                (
                    "negative_controls",
                    "wrong_cache_request_association",
                    "max_abs_logit_delta",
                ),
                0.0,
            ),
        ]
        empty_controls = copy.deepcopy(self.evidence)
        empty_controls["negative_controls"] = {}
        cases.append(("empty-negative-controls", empty_controls))
        missing_runtime = copy.deepcopy(self.evidence)
        raw_runtime = missing_runtime["runtime"]
        assert isinstance(raw_runtime, dict)
        raw_runtime.pop("machine")
        cases.append(("missing-runtime-field", missing_runtime))
        extra_top_level = copy.deepcopy(self.evidence)
        extra_top_level["undeclared_claim"] = True
        cases.append(("extra-top-level-field", extra_top_level))
        missing_requests = copy.deepcopy(self.evidence)
        missing_requests["requests"] = []
        cases.append(("missing-requests", missing_requests))

        for label, document in cases:
            with self.subTest(label=label), self.assertRaises(self.module.SemanticsGateError):
                self.module.verify_evidence(self._write_rehashed(document, label))

    def test_verifier_rejects_rehashed_source_hash(self) -> None:
        document = copy.deepcopy(self.evidence)
        implementation = document["implementation"]
        assert isinstance(implementation, dict)
        implementation["sha256"] = "0" * 64
        with self.assertRaisesRegex(
            self.module.SemanticsGateError,
            "implementation hash",
        ):
            self.module.verify_evidence(self._write_rehashed(document, "source-hash"))

    def test_verifier_rejects_rehashed_within_bound_result_mutations(self) -> None:
        def mutate(
            label: str,
            path: tuple[str, ...],
            value: object,
        ) -> tuple[str, dict[str, object]]:
            document = copy.deepcopy(self.evidence)
            target: dict[str, object] = document
            for component in path[:-1]:
                nested = target[component]
                assert isinstance(nested, dict)
                target = nested
            target[path[-1]] = value
            return label, document

        cases = [
            mutate(
                "audited-operations-upward",
                ("device_contract", "dispatch_operations_audited"),
                4_161,
            ),
            mutate(
                "runtime-platform",
                ("runtime", "platform"),
                "other-valid-platform",
            ),
            mutate(
                "model-state-hash",
                ("determinism", "model_state_sha256"),
                "a" * 64,
            ),
            mutate(
                "valid-path-delta-downward",
                (
                    "equivalence",
                    "reinsert_vs_monolithic_full_forward_max_abs",
                ),
                0.0,
            ),
            mutate(
                "per-request-cache-delta-downward",
                ("equivalence", "per_request_cache_max_abs", "request-a"),
                0.0,
            ),
            mutate(
                "negative-control-delta-upward",
                (
                    "negative_controls",
                    "wrong_cache_request_association",
                    "max_abs_logit_delta",
                ),
                0.5,
            ),
        ]

        greedy_token = copy.deepcopy(self.evidence)
        greedy_requests = greedy_token["requests"]
        assert isinstance(greedy_requests, list)
        first_request = greedy_requests[0]
        assert isinstance(first_request, dict)
        first_request["first_greedy_token"] = 69
        cases.append(("greedy-token", greedy_token))

        prefill_fingerprint = copy.deepcopy(self.evidence)
        prefill_requests = prefill_fingerprint["requests"]
        assert isinstance(prefill_requests, list)
        prefill_first = prefill_requests[0]
        assert isinstance(prefill_first, dict)
        prefill_first["prefill_cache_sha256"] = "b" * 64
        cases.append(("prefill-cache-fingerprint", prefill_fingerprint))

        reinsert_fingerprint = copy.deepcopy(self.evidence)
        reinsert_requests = reinsert_fingerprint["requests"]
        assert isinstance(reinsert_requests, list)
        reinsert_first = reinsert_requests[0]
        assert isinstance(reinsert_first, dict)
        reinsert_first["reinserted_cache_sha256"] = "c" * 64
        cases.append(("reinsert-cache-fingerprint", reinsert_fingerprint))

        parked_fingerprint = copy.deepcopy(self.evidence)
        parked_ownership = parked_fingerprint["cache_ownership"]
        assert isinstance(parked_ownership, dict)
        parked_ownership["parked_cache_sha256_before_peer_advance"] = "d" * 64
        parked_ownership["parked_cache_sha256_after_peer_advance"] = "d" * 64
        cases.append(("parked-cache-fingerprint", parked_fingerprint))

        for label, document in cases:
            with self.subTest(label=label), self.assertRaises(self.module.SemanticsGateError):
                self.module.verify_evidence(self._write_rehashed(document, label))

    def test_verifier_rejects_nonregular_and_symlink_paths(self) -> None:
        temporary_root = Path(self.temporary.name)
        with self.assertRaisesRegex(
            self.module.SemanticsGateError,
            "regular non-symlink",
        ):
            self.module.verify_evidence(temporary_root)

        link_path = temporary_root / "evidence-link.json"
        link_path.symlink_to(self.first_path)
        with self.assertRaisesRegex(
            self.module.SemanticsGateError,
            "regular non-symlink",
        ):
            self.module.verify_evidence(link_path)

    def test_verifier_rejects_duplicate_keys_and_nonfinite_constants(self) -> None:
        original = self.first_path.read_text(encoding="utf-8")
        malformed = {
            "duplicate-key": original.replace(
                "{",
                '{"schema_version":1,',
                1,
            ),
            "nonfinite": original.replace(
                '"seed":20260723',
                '"seed":NaN',
                1,
            ),
        }
        for label, contents in malformed.items():
            path = Path(self.temporary.name) / f"malformed/{label}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(contents, encoding="utf-8")
            with (
                self.subTest(label=label),
                self.assertRaisesRegex(
                    self.module.SemanticsGateError,
                    "strict UTF-8 JSON",
                ),
            ):
                self.module.verify_evidence(path)


@unittest.skipUnless(
    DEPENDENCIES_AVAILABLE,
    "installed torch and transformers are required for the optional CPU stack gate",
)
class CpuTransformerSemanticsTests(unittest.TestCase):
    temporary: ClassVar[tempfile.TemporaryDirectory[str]]
    module: ClassVar[ModuleType]
    first_path: ClassVar[Path]
    first_digest: ClassVar[str]
    evidence: ClassVar[dict[str, object]]

    @classmethod
    def setUpClass(cls) -> None:
        cls.module = importlib.import_module("tools.run_cpu_transformer_semantics")
        cls.temporary = tempfile.TemporaryDirectory()
        cls.first_path = Path(cls.temporary.name) / "first/evidence.json"
        cls.first_digest, _ = cls.module.run_gate(cls.first_path)
        document = json.loads(cls.first_path.read_bytes())
        if not isinstance(document, dict):
            raise AssertionError("evidence root must be an object")
        cls.evidence = document

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def test_evidence_is_self_hashed_offline_and_cpu_only(self) -> None:
        self.assertEqual(
            self.module.verify_evidence(self.first_path),
            self.first_digest,
        )
        offline = self.evidence["offline_contract"]
        device = self.evidence["device_contract"]
        assert isinstance(offline, dict)
        assert isinstance(device, dict)
        self.assertFalse(offline["pretrained_weights_loaded"])
        self.assertFalse(offline["tokenizer_loaded"])
        self.assertFalse(offline["network_downloads"])
        self.assertEqual(device["requested_device"], "cpu")
        self.assertEqual(device["dispatch_devices_seen"], ["cpu"])
        self.assertEqual(device["model_parameter_devices"], ["cpu"])
        self.assertEqual(device["cache_devices"], ["cpu"])
        self.assertFalse(device["cuda_initialized_before"])
        self.assertFalse(device["cuda_initialized_after"])
        self.assertEqual(device["mps_tensor_operations_observed"], 0)
        self.assertEqual(device["non_cpu_tensor_observations"], 0)
        self.assertGreater(device["dispatch_operations_audited"], 1_000)

    def test_logits_tokens_cache_ownership_and_negative_controls_pass(self) -> None:
        equivalence = self.evidence["equivalence"]
        ownership = self.evidence["cache_ownership"]
        controls = self.evidence["negative_controls"]
        assert isinstance(equivalence, dict)
        assert isinstance(ownership, dict)
        assert isinstance(controls, dict)

        self.assertTrue(equivalence["greedy_tokens_exact"])
        tolerance = float(equivalence["absolute_tolerance"])
        for field in (
            "mixed_left_padded_prefill_vs_monolithic_max_abs",
            "active_rebatch_vs_individual_cache_max_abs",
            "parked_catchup_vs_individual_cache_max_abs",
            "reinsert_vs_individual_cache_max_abs",
            "reinsert_vs_monolithic_full_forward_max_abs",
        ):
            self.assertLessEqual(float(equivalence[field]), tolerance)
        cache_differences = equivalence["per_request_cache_max_abs"]
        assert isinstance(cache_differences, dict)
        self.assertTrue(all(float(value) <= tolerance for value in cache_differences.values()))
        self.assertTrue(ownership["parked_cache_byte_identical"])
        self.assertTrue(ownership["per_request_prefill_fingerprints_distinct"])
        self.assertTrue(ownership["per_request_cache_content_equivalent"])
        self.assertEqual(
            ownership["parked_cache_sha256_before_peer_advance"],
            ownership["parked_cache_sha256_after_peer_advance"],
        )
        self.assertEqual(
            ownership["reinsert_order"],
            ["request-c", "request-b", "request-a"],
        )
        for result in controls.values():
            assert isinstance(result, dict)
            self.assertTrue(result["detected"])
            self.assertGreater(float(result["max_abs_logit_delta"]), tolerance)

    def test_two_runs_are_byte_identical(self) -> None:
        repeated_path = Path(self.temporary.name) / "repeated/evidence.json"
        repeated_digest, _ = self.module.run_gate(repeated_path)

        self.assertEqual(repeated_digest, self.first_digest)
        self.assertEqual(repeated_path.read_bytes(), self.first_path.read_bytes())


if __name__ == "__main__":
    unittest.main()
