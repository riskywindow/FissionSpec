"""Tests for the pinned pretrained Qwen3 CPU semantics artifact."""

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
from typing import ClassVar, cast

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = REPOSITORY_ROOT / "tools/run_pretrained_cpu_semantics.py"
EVIDENCE_PATH = REPOSITORY_ROOT / "experiments/results/pretrained_cpu_semantics/evidence.json"
HEAVY_DEPENDENCIES_AVAILABLE = all(
    importlib.util.find_spec(name) is not None
    for name in (
        "torch",
        "transformers",
        "huggingface_hub",
        "safetensors",
        "tokenizers",
    )
)


class PretrainedCpuStaticContractTests(unittest.TestCase):
    def test_generator_is_offline_pinned_and_has_no_top_level_heavy_imports(
        self,
    ) -> None:
        source = TOOL_PATH.read_text(encoding="utf-8")

        self.assertIn(
            'MODEL_ID: Final = "Qwen/Qwen3-0.6B"',
            source,
        )
        self.assertIn(
            'MODEL_REVISION: Final = "c1899de289a04d12100db370d81485cdf75e47ca"',
            source,
        )
        self.assertIn("local_files_only=True", source)
        self.assertIn('os.environ["CUDA_VISIBLE_DEVICES"] = ""', source)
        self.assertIn('os.environ["FISSIONSPEC_MPS_DISABLED"] = "1"', source)
        self.assertIn('os.environ["HF_HUB_OFFLINE"] = "1"', source)
        self.assertIn('os.environ["TRANSFORMERS_OFFLINE"] = "1"', source)
        self.assertNotIn("\nimport torch", source)
        self.assertNotIn("\nimport transformers", source)
        self.assertNotIn("\nimport huggingface_hub", source)

    def test_repository_tracks_no_checkpoint_weight_file(self) -> None:
        completed = subprocess.run(
            ("git", "ls-files", "*.safetensors", "*.bin", "*.gguf"),
            cwd=REPOSITORY_ROOT,
            capture_output=True,
            check=True,
            text=True,
        )

        self.assertEqual(completed.stdout, "")


class PretrainedCpuEvidenceVerifierTests(unittest.TestCase):
    module: ClassVar[ModuleType]
    temporary: ClassVar[tempfile.TemporaryDirectory[str]]
    evidence: ClassVar[dict[str, object]]
    digest: ClassVar[str]

    @classmethod
    def setUpClass(cls) -> None:
        cls.module = importlib.import_module("tools.run_pretrained_cpu_semantics")
        cls.temporary = tempfile.TemporaryDirectory()
        cls.digest = cls.module.verify_evidence(EVIDENCE_PATH)
        document = json.loads(EVIDENCE_PATH.read_bytes())
        if not isinstance(document, dict):
            raise AssertionError("evidence root must be an object")
        cls.evidence = document

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def _write_rehashed(
        self,
        document: dict[str, object],
        label: str,
    ) -> Path:
        payload = copy.deepcopy(document)
        payload.pop("payload_sha256", None)
        payload["payload_sha256"] = self.module._sha256_document(payload)
        destination = Path(self.temporary.name) / f"tampered/{label}.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(self.module._canonical_json_bytes(payload))
        return destination

    def test_verify_only_works_without_site_packages(self) -> None:
        environment = dict(os.environ)
        environment["PYTHONNOUSERSITE"] = "1"
        completed = subprocess.run(
            (
                sys.executable,
                "-S",
                str(TOOL_PATH),
                "--verify-only",
                str(EVIDENCE_PATH),
            ),
            cwd=REPOSITORY_ROOT,
            env=environment,
            capture_output=True,
            check=False,
            text=True,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn(self.digest, completed.stdout)

    def test_verifier_rejects_rehashed_contract_mutations(self) -> None:
        def mutate(
            label: str,
            path: tuple[str, ...],
            value: object,
        ) -> tuple[str, dict[str, object]]:
            document = copy.deepcopy(self.evidence)
            target = document
            for component in path[:-1]:
                target = cast(dict[str, object], target[component])
            target[path[-1]] = value
            return label, document

        cases = [
            mutate(
                "revision",
                ("checkpoint", "revision"),
                "0" * 40,
            ),
            mutate(
                "weight-sha",
                (
                    "checkpoint",
                    "files",
                    "model.safetensors",
                    "sha256",
                ),
                "0" * 64,
            ),
            mutate(
                "network-access",
                ("offline_contract", "generation_network_access"),
                True,
            ),
            mutate(
                "mps-operation",
                ("device_contract", "mps_tensor_operations_observed"),
                1,
            ),
            mutate(
                "non-cpu",
                ("device_contract", "dispatch_devices_seen"),
                ["cpu", "mps"],
            ),
            mutate(
                "greedy-equivalence",
                ("equivalence", "greedy_tokens_exact"),
                False,
            ),
            mutate(
                "parked-mutation",
                ("cache_ownership", "parked_cache_byte_identical"),
                False,
            ),
            mutate(
                "negative-control",
                (
                    "negative_controls",
                    "wrong_cache_request_association",
                    "max_abs_logit_delta",
                ),
                0.0,
            ),
            mutate(
                "claim-boundary",
                ("claim_boundary",),
                "production GPU evidence",
            ),
        ]
        for label, document in cases:
            with (
                self.subTest(label=label),
                self.assertRaises(self.module.PretrainedSemanticsError),
            ):
                self.module.verify_evidence(self._write_rehashed(document, label))

    def test_verifier_rejects_rehashed_within_bound_result_mutations(
        self,
    ) -> None:
        cases: list[tuple[str, dict[str, object]]] = []

        equivalence = copy.deepcopy(self.evidence)
        equivalence_document = cast(
            dict[str, object],
            equivalence["equivalence"],
        )
        equivalence_document["reinsert_vs_monolithic_full_forward_max_abs"] = 0.0
        cases.append(("valid-delta", equivalence))

        request_token = copy.deepcopy(self.evidence)
        requests = cast(list[object], request_token["requests"])
        first_request = cast(dict[str, object], requests[0])
        first_request["third_greedy_token_after_reinsert_id"] = 3909
        cases.append(("valid-token", request_token))

        runtime = copy.deepcopy(self.evidence)
        runtime_document = cast(dict[str, object], runtime["runtime"])
        runtime_document["platform"] = "other-valid-platform"
        cases.append(("valid-runtime", runtime))

        fingerprint = copy.deepcopy(self.evidence)
        fingerprint_requests = cast(
            list[object],
            fingerprint["requests"],
        )
        fingerprint_request = cast(
            dict[str, object],
            fingerprint_requests[0],
        )
        fingerprint_request["reinserted_logits_sha256"] = "a" * 64
        cases.append(("valid-fingerprint", fingerprint))

        for label, document in cases:
            with (
                self.subTest(label=label),
                self.assertRaisesRegex(
                    self.module.PretrainedSemanticsError,
                    "frozen semantic result map",
                ),
            ):
                self.module.verify_evidence(self._write_rehashed(document, label))

    def test_verifier_rejects_rehashed_source_binding(self) -> None:
        document = copy.deepcopy(self.evidence)
        implementation = cast(
            dict[str, object],
            document["implementation"],
        )
        implementation["sha256"] = "0" * 64

        with self.assertRaisesRegex(
            self.module.PretrainedSemanticsError,
            "implementation hash",
        ):
            self.module.verify_evidence(self._write_rehashed(document, "source"))

    def test_verifier_rejects_noncanonical_duplicate_and_nonfinite_json(
        self,
    ) -> None:
        original = EVIDENCE_PATH.read_text(encoding="utf-8")
        malformed = {
            "noncanonical": json.dumps(
                json.loads(original),
                indent=2,
                sort_keys=True,
            ),
            "duplicate": original.replace(
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
                self.assertRaises(self.module.PretrainedSemanticsError),
            ):
                self.module.verify_evidence(path)

    def test_verifier_rejects_nonregular_and_symlink_inputs(self) -> None:
        temporary_root = Path(self.temporary.name)
        with self.assertRaisesRegex(
            self.module.PretrainedSemanticsError,
            "regular non-symlink",
        ):
            self.module.verify_evidence(temporary_root)

        link = temporary_root / "evidence-link.json"
        link.symlink_to(EVIDENCE_PATH)
        with self.assertRaisesRegex(
            self.module.PretrainedSemanticsError,
            "regular non-symlink",
        ):
            self.module.verify_evidence(link)

    def test_evidence_names_exact_checkpoint_and_cpu_contract(self) -> None:
        checkpoint = cast(
            dict[str, object],
            self.evidence["checkpoint"],
        )
        device = cast(
            dict[str, object],
            self.evidence["device_contract"],
        )
        offline = cast(
            dict[str, object],
            self.evidence["offline_contract"],
        )

        self.assertEqual(checkpoint["model_id"], "Qwen/Qwen3-0.6B")
        self.assertEqual(
            checkpoint["revision"],
            "c1899de289a04d12100db370d81485cdf75e47ca",
        )
        self.assertEqual(device["dispatch_devices_seen"], ["cpu"])
        self.assertEqual(device["non_cpu_tensor_observations"], 0)
        self.assertEqual(device["mps_tensor_operations_observed"], 0)
        self.assertFalse(offline["generation_network_access"])
        self.assertEqual(offline["network_attempts_observed"], 0)


@unittest.skipUnless(
    os.environ.get("FISSIONSPEC_RUN_PRETRAINED_CPU") == "1" and HEAVY_DEPENDENCIES_AVAILABLE,
    "set FISSIONSPEC_RUN_PRETRAINED_CPU=1 for cached heavy regeneration",
)
class PretrainedCpuOptionalRegenerationTests(unittest.TestCase):
    def test_cached_offline_regeneration_is_byte_identical(self) -> None:
        module = importlib.import_module("tools.run_pretrained_cpu_semantics")
        with tempfile.TemporaryDirectory() as directory:
            candidate = Path(directory) / "evidence.json"
            module.run_gate(candidate)

            self.assertEqual(
                candidate.read_bytes(),
                EVIDENCE_PATH.read_bytes(),
            )


if __name__ == "__main__":
    unittest.main()
