"""Tests for deterministic public-trace freezing and offline verification."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import tempfile
import unittest
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast

from fissionspec.artifacts import canonical_json_bytes, sha256_document
from tools.freeze_public_replay import (
    SCHEMA,
    WARNING,
    PublicReplayError,
    SourceContract,
    freeze_replay,
    inspect_source,
    verify_frozen_replay,
)


def _source_bytes(*, rows: int = 24) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(("TIMESTAMP", "ContextTokens", "GeneratedTokens"))
    start = datetime(2024, 1, 1, tzinfo=UTC)
    for index in range(rows):
        timestamp = start + timedelta(seconds=index, microseconds=index * 17)
        writer.writerow(
            (
                timestamp.isoformat(sep=" ", timespec="microseconds"),
                100 + index,
                1 + index,
            )
        )
    return output.getvalue().encode("ascii")


def _contract(source: bytes, *, rows: int = 24) -> SourceContract:
    lines = source.decode("ascii").splitlines()
    return SourceContract(
        dataset_id="unit-test-source",
        url="https://example.invalid/source.csv",
        sha256=hashlib.sha256(source).hexdigest(),
        bytes=len(source),
        rows=rows,
        first_timestamp=lines[1].split(",")[0],
        last_timestamp=lines[-1].split(",")[0],
        license_spdx="CC-BY-4.0",
        license_url="https://example.invalid/license",
        attribution="Unit Test Author",
        citation_url="https://example.invalid/citation",
    )


class PublicReplayTests(unittest.TestCase):
    def test_freeze_is_deterministic_hash_linked_and_loadable(self) -> None:
        source_bytes = _source_bytes()
        contract = _contract(source_bytes)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.csv"
            first_trace = root / "first.csv"
            first_manifest = root / "first.manifest.json"
            second_trace = root / "second.csv"
            second_manifest = root / "second.manifest.json"
            source.write_bytes(source_bytes)

            first = freeze_replay(
                source,
                first_trace,
                first_manifest,
                contract=contract,
                selected_rows=5,
                digest_prefix_hex_chars=4,
            )
            second = freeze_replay(
                source,
                second_trace,
                second_manifest,
                contract=contract,
                selected_rows=5,
                digest_prefix_hex_chars=4,
            )

            self.assertEqual(first_trace.read_bytes(), second_trace.read_bytes())
            first_payload = dict(first)
            second_payload = dict(second)
            first_payload["derivative"] = dict(cast(dict[str, object], first["derivative"]))
            second_payload["derivative"] = dict(cast(dict[str, object], second["derivative"]))
            cast(dict[str, object], first_payload["derivative"])["file"] = "replay.csv"
            cast(dict[str, object], second_payload["derivative"])["file"] = "replay.csv"
            first_payload.pop("payload_sha256")
            second_payload.pop("payload_sha256")
            self.assertEqual(first_payload, second_payload)
            verified = verify_frozen_replay(first_trace, first_manifest)
            self.assertEqual(verified["schema"], SCHEMA)
            self.assertEqual(verified["measurement_warning"], WARNING)
            selection = cast(dict[str, object], verified["selection"])
            self.assertEqual(selection["rows"], 5)
            self.assertEqual(selection["split"], "validation")

    def test_source_contract_and_malformed_rows_fail_closed(self) -> None:
        source_bytes = _source_bytes()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.csv"
            source.write_bytes(source_bytes)
            observed = inspect_source(source)
            self.assertEqual(observed.rows, 24)

            wrong = replace(_contract(source_bytes), sha256="0" * 64)
            with self.assertRaises(PublicReplayError):
                freeze_replay(
                    source,
                    root / "trace.csv",
                    root / "manifest.json",
                    contract=wrong,
                    selected_rows=4,
                )

            source.write_bytes(
                b"TIMESTAMP,ContextTokens,GeneratedTokens\n"
                b"2024-01-01 00:00:00.000000+00:00,1,2\n"
                b"2023-01-01 00:00:00.000000+00:00,1,2\n"
            )
            with self.assertRaisesRegex(PublicReplayError, "nondecreasing"):
                inspect_source(source)

    def test_manifest_and_trace_tampering_are_rejected(self) -> None:
        source_bytes = _source_bytes()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.csv"
            trace = root / "trace.csv"
            manifest = root / "manifest.json"
            source.write_bytes(source_bytes)
            freeze_replay(
                source,
                trace,
                manifest,
                contract=_contract(source_bytes),
                selected_rows=4,
            )

            trace.write_bytes(trace.read_bytes() + b"\n")
            with self.assertRaisesRegex(PublicReplayError, "hash mismatch"):
                verify_frozen_replay(trace, manifest)

            freeze_replay(
                source,
                trace,
                manifest,
                contract=_contract(source_bytes),
                selected_rows=4,
                force=True,
            )
            document = json.loads(manifest.read_text(encoding="utf-8"))
            document["selection"]["rows"] = 999
            manifest.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(PublicReplayError, "payload hash mismatch"):
                verify_frozen_replay(trace, manifest)

            freeze_replay(
                source,
                trace,
                manifest,
                contract=_contract(source_bytes),
                selected_rows=4,
                force=True,
            )
            document = json.loads(manifest.read_text(encoding="utf-8"))
            document["source"]["dataset_id"] = "substituted-source"
            payload = dict(document)
            payload.pop("payload_sha256")
            document["payload_sha256"] = sha256_document(payload)
            manifest.write_bytes(canonical_json_bytes(document))
            with self.assertRaisesRegex(PublicReplayError, "registered contract"):
                verify_frozen_replay(
                    trace,
                    manifest,
                    expected_contract=_contract(source_bytes),
                )

            freeze_replay(
                source,
                trace,
                manifest,
                contract=_contract(source_bytes),
                selected_rows=4,
                force=True,
            )
            lines = trace.read_text(encoding="ascii").splitlines()
            fields = lines[2].split(",")
            fields[-1] = str(int(fields[-1]) + 7)
            lines[2] = ",".join(fields)
            trace.write_text("\n".join(lines) + "\n", encoding="ascii")
            document = json.loads(manifest.read_text(encoding="utf-8"))
            trace_bytes = trace.read_bytes()
            document["derivative"]["sha256"] = hashlib.sha256(trace_bytes).hexdigest()
            document["derivative"]["bytes"] = len(trace_bytes)
            payload = dict(document)
            payload.pop("payload_sha256")
            document["payload_sha256"] = sha256_document(payload)
            manifest.write_bytes(canonical_json_bytes(document))
            with self.assertRaisesRegex(PublicReplayError, "not consecutive"):
                verify_frozen_replay(trace, manifest)

    def test_overwrite_and_insufficient_tail_are_rejected(self) -> None:
        source_bytes = _source_bytes(rows=3)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.csv"
            trace = root / "trace.csv"
            manifest = root / "manifest.json"
            source.write_bytes(source_bytes)
            contract = _contract(source_bytes, rows=3)
            with self.assertRaisesRegex(PublicReplayError, "source ended"):
                freeze_replay(
                    source,
                    trace,
                    manifest,
                    contract=contract,
                    selected_rows=100,
                )
            freeze_replay(
                source,
                trace,
                manifest,
                contract=contract,
                selected_rows=1,
            )
            with self.assertRaisesRegex(PublicReplayError, "overwrite"):
                freeze_replay(
                    source,
                    trace,
                    manifest,
                    contract=contract,
                    selected_rows=1,
                )


if __name__ == "__main__":
    unittest.main()
