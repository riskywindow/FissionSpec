"""Adversarial tests for the zero-GPU campaign freezer."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from typing import cast

from fissionspec.spend_gate import CampaignLedger, CampaignPlan, CampaignStage
from tools.freeze_gpu_campaign import (
    OUTPUT_FILES,
    CampaignFreezeError,
    EvidenceSpec,
    _canonical_bytes,
    _evidence_records,
    _self_hashed,
    freeze_bundle,
    verify_bundle,
)

CODE_COMMIT = "a" * 40


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_bytes(root: Path, relative: str, payload: bytes) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _write_json(root: Path, relative: str, document: dict[str, object]) -> Path:
    return _write_bytes(root, relative, _canonical_bytes(document))


def _fake_repository(parent: Path) -> tuple[Path, tuple[EvidenceSpec, ...]]:
    root = parent / "repo"
    _write_bytes(root, "paper/gpu_preregistration.md", b"# Frozen protocol\n")
    _write_bytes(
        root,
        "src/fissionspec/workload_generators.py",
        b'"""Frozen generator source."""\n',
    )
    _write_bytes(root, "src/fissionspec/rng.py", b'"""Frozen RNG source."""\n')

    replay = b"TIMESTAMP,ContextTokens,GeneratedTokens\n0,128,32\n10,256,64\n"
    _write_bytes(root, "configs/traces/azure_llm_code_v3_1024.csv", replay)
    replay_manifest = _self_hashed(
        {
            "schema": "test.public-replay.v1",
            "derivative": {
                "sha256": _sha256(replay),
                "rows": 2,
            },
            "selection": {"arrival_span_ms": 10},
        }
    )
    _write_json(
        root,
        "configs/traces/azure_llm_code_v3_1024.manifest.json",
        replay_manifest,
    )

    _write_json(
        root,
        "evidence/alpha.json",
        _self_hashed(
            {
                "schema": "test.cpu-evidence.v1",
                "claim": "CPU only",
            }
        ),
    )
    _write_bytes(root, "evidence/beta.txt", b"independent CPU evidence\n")
    evidence = (
        EvidenceSpec("alpha-evidence", Path("evidence/alpha.json")),
        EvidenceSpec("beta-evidence", Path("evidence/beta.txt")),
    )
    return root, evidence


def _bundle_bytes(output_dir: Path) -> dict[str, bytes]:
    return {name: (output_dir / name).read_bytes() for name in OUTPUT_FILES}


def _rehash_spend_document(document: dict[str, object]) -> dict[str, object]:
    payload = dict(document)
    payload.pop("payload_sha256", None)
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return {**payload, "payload_sha256": _sha256(encoded)}


def _refresh_bundle_manifest(output_dir: Path, changed_name: str) -> None:
    manifest_path = output_dir / "bundle_manifest.json"
    manifest = cast(dict[str, object], json.loads(manifest_path.read_text()))
    files = cast(list[dict[str, object]], manifest["files"])
    changed = output_dir / changed_name
    for record in files:
        if record["path"] == changed_name:
            record["bytes"] = changed.stat().st_size
            record["sha256"] = _sha256(changed.read_bytes())
            break
    else:
        raise AssertionError(f"missing manifest record for {changed_name}")
    payload = dict(manifest)
    payload.pop("payload_sha256")
    _write_bytes(output_dir, "bundle_manifest.json", _canonical_bytes(_self_hashed(payload)))


class CampaignFreezeTests(unittest.TestCase):
    def test_freeze_is_deterministic_and_authorizes_zero_gpu_seconds(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root, evidence = _fake_repository(Path(temporary))
            first = root / "bundle-first"
            second = root / "bundle-second"

            first_manifest = freeze_bundle(
                root,
                first,
                code_commit=CODE_COMMIT,
                evidence=evidence,
                enforce_git=False,
            )
            before = _bundle_bytes(first)
            freeze_bundle(
                root,
                first,
                code_commit=CODE_COMMIT,
                evidence=evidence,
                enforce_git=False,
            )
            second_manifest = freeze_bundle(
                root,
                second,
                code_commit=CODE_COMMIT,
                evidence=evidence,
                enforce_git=False,
            )

            self.assertEqual(before, _bundle_bytes(first))
            self.assertEqual(before, _bundle_bytes(second))
            self.assertEqual(first_manifest, second_manifest)
            self.assertEqual(
                verify_bundle(
                    root,
                    first,
                    evidence=evidence,
                    enforce_git=False,
                ),
                first_manifest,
            )

            plan_document = json.loads((first / "campaign_plan.json").read_text())
            plan = CampaignPlan.from_document(plan_document)
            ledger_document = json.loads((first / "campaign_ledger_f0.json").read_text())
            ledger = CampaignLedger.from_document(ledger_document)
            self.assertEqual(ledger.plan, plan)
            self.assertEqual(ledger.spent_gpu_seconds, 0)
            self.assertEqual(ledger.currently_authorized_gpu_seconds, 0)
            self.assertEqual(ledger.next_stage, CampaignStage.ROW_OMISSION)
            self.assertEqual(ledger.records[0].used_gpu_seconds, 0)
            self.assertEqual(ledger.records[0].completed_replays, 0)

    def test_source_evidence_tampering_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root, evidence = _fake_repository(Path(temporary))
            output = root / "bundle"
            freeze_bundle(
                root,
                output,
                code_commit=CODE_COMMIT,
                evidence=evidence,
                enforce_git=False,
            )
            beta = root / "evidence/beta.txt"
            beta.write_bytes(beta.read_bytes() + b"tampered")

            with self.assertRaisesRegex(
                CampaignFreezeError,
                "CPU evidence index does not reproduce",
            ):
                verify_bundle(
                    root,
                    output,
                    evidence=evidence,
                    enforce_git=False,
                )

    def test_rehashed_derived_ledger_field_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root, evidence = _fake_repository(Path(temporary))
            output = root / "bundle"
            freeze_bundle(
                root,
                output,
                code_commit=CODE_COMMIT,
                evidence=evidence,
                enforce_git=False,
            )
            ledger_path = output / "campaign_ledger_f0.json"
            ledger = cast(dict[str, object], json.loads(ledger_path.read_text()))
            ledger["currently_authorized_gpu_seconds"] = 1
            ledger = _rehash_spend_document(ledger)
            ledger_path.write_bytes(_canonical_bytes(ledger))
            _refresh_bundle_manifest(output, "campaign_ledger_f0.json")

            with self.assertRaisesRegex(
                CampaignFreezeError,
                "F0 ledger verification failed",
            ):
                verify_bundle(
                    root,
                    output,
                    evidence=evidence,
                    enforce_git=False,
                )

    def test_closed_bundle_and_symlink_boundaries_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root, evidence = _fake_repository(Path(temporary))
            output = root / "bundle"
            freeze_bundle(
                root,
                output,
                code_commit=CODE_COMMIT,
                evidence=evidence,
                enforce_git=False,
            )

            extra = _write_bytes(output, "escape.txt", b"not declared\n")
            with self.assertRaisesRegex(CampaignFreezeError, "file set is not closed"):
                verify_bundle(
                    root,
                    output,
                    evidence=evidence,
                    enforce_git=False,
                )
            extra.unlink()

            linked_output = root / "linked-bundle"
            linked_output.symlink_to(output, target_is_directory=True)
            with self.assertRaisesRegex(CampaignFreezeError, "may not be a symlink"):
                verify_bundle(
                    root,
                    linked_output,
                    evidence=evidence,
                    enforce_git=False,
                )

            manifest_path = output / "bundle_manifest.json"
            manifest = cast(dict[str, object], json.loads(manifest_path.read_text()))
            files = cast(list[dict[str, object]], manifest["files"])
            files[0]["bytes"] = float(cast(int, files[0]["bytes"]))
            payload = dict(manifest)
            payload.pop("payload_sha256")
            manifest_path.write_bytes(_canonical_bytes(_self_hashed(payload)))
            with self.assertRaisesRegex(CampaignFreezeError, "positive integer"):
                verify_bundle(
                    root,
                    output,
                    evidence=evidence,
                    enforce_git=False,
                )

            with self.assertRaisesRegex(CampaignFreezeError, "inside the repository"):
                freeze_bundle(
                    root,
                    root.parent / "outside-bundle",
                    code_commit=CODE_COMMIT,
                    evidence=evidence,
                    enforce_git=False,
                )

    def test_evidence_registry_rejects_aliases_ordering_and_unsafe_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root, _ = _fake_repository(Path(temporary))
            alpha = EvidenceSpec("alpha", Path("evidence/alpha.json"))
            beta = EvidenceSpec("beta", Path("evidence/beta.txt"))

            with self.assertRaisesRegex(CampaignFreezeError, "sorted"):
                _evidence_records(root, (beta, alpha))
            with self.assertRaisesRegex(CampaignFreezeError, "artifact IDs"):
                _evidence_records(
                    root,
                    (
                        alpha,
                        EvidenceSpec("alpha", Path("evidence/beta.txt")),
                    ),
                )
            with self.assertRaisesRegex(CampaignFreezeError, "paths"):
                _evidence_records(
                    root,
                    (
                        alpha,
                        EvidenceSpec("beta", Path("evidence/alpha.json")),
                    ),
                )
            with self.assertRaisesRegex(ValueError, "lowercase ASCII slug"):
                EvidenceSpec("Alpha", Path("evidence/alpha.json"))
            with self.assertRaisesRegex(ValueError, "repository-relative"):
                EvidenceSpec("escape", Path("../outside"))

    def test_partial_output_is_not_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root, evidence = _fake_repository(Path(temporary))
            output = root / "bundle"
            marker = _write_bytes(output, "operator-note.txt", b"preserve me\n")

            with self.assertRaisesRegex(CampaignFreezeError, "non-empty"):
                freeze_bundle(
                    root,
                    output,
                    code_commit=CODE_COMMIT,
                    evidence=evidence,
                    enforce_git=False,
                )
            self.assertEqual(marker.read_bytes(), b"preserve me\n")


if __name__ == "__main__":
    unittest.main()
