#!/usr/bin/env python3
"""Freeze a bounded, hash-derived replay from the Azure 2024 LLM trace.

The raw Azure trace is intentionally not vendored.  This tool verifies its
exact bytes, chooses a source window from the source digest rather than observed
performance, and emits a small canonical CSV plus a self-hashed provenance
manifest.  The checked-in derivative is workload input, never GPU evidence.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import sys
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final, cast

from fissionspec.artifacts import canonical_json_bytes, sha256_document
from fissionspec.workload_generators import load_trace_csv

SCHEMA: Final = "fissionspec.public-replay-manifest.v1"
WARNING: Final = (
    "PUBLIC WORKLOAD TRACE INPUT — NOT MODEL OUTPUT, NOT A GPU MEASUREMENT, "
    "AND NOT PERFORMANCE EVIDENCE."
)
DEFAULT_TRACE: Final = Path("configs/traces/azure_llm_code_v3_1024.csv")
DEFAULT_MANIFEST: Final = Path("configs/traces/azure_llm_code_v3_1024.manifest.json")
SOURCE_FIELDS: Final = frozenset(
    {
        "dataset_id",
        "url",
        "sha256",
        "bytes",
        "rows",
        "first_timestamp",
        "last_timestamp",
        "license_spdx",
        "license_url",
        "attribution",
        "citation_url",
    }
)
SELECTION_FIELDS: Final = frozenset(
    {
        "algorithm",
        "digest_prefix_hex_chars",
        "offset_seconds",
        "target_timestamp",
        "first_source_row_zero_based",
        "last_source_row_zero_based",
        "first_timestamp",
        "last_timestamp",
        "arrival_span_ms",
        "rows",
        "split",
    }
)
DERIVATIVE_FIELDS: Final = frozenset({"file", "sha256", "bytes", "rows", "format", "modifications"})
USE_FIELDS: Final = frozenset({"registered_anchor", "arrival_rescaling", "claim_boundary"})


class PublicReplayError(ValueError):
    """Raised when a source or frozen replay violates its closed contract."""


@dataclass(frozen=True, slots=True)
class SourceContract:
    """Immutable identity and attribution for one upstream trace."""

    dataset_id: str
    url: str
    sha256: str
    bytes: int
    rows: int
    first_timestamp: str
    last_timestamp: str
    license_spdx: str
    license_url: str
    attribution: str
    citation_url: str

    def __post_init__(self) -> None:
        if not self.dataset_id or not self.url:
            raise ValueError("source dataset ID and URL must not be empty")
        _validate_sha256(self.sha256, field="source sha256")
        if self.bytes <= 0 or self.rows <= 0:
            raise ValueError("source byte and row counts must be positive")
        _parse_utc_timestamp(self.first_timestamp)
        _parse_utc_timestamp(self.last_timestamp)
        if self.first_timestamp >= self.last_timestamp:
            raise ValueError("source timestamps must span positive time")
        if not all((self.license_spdx, self.license_url, self.attribution, self.citation_url)):
            raise ValueError("source license and attribution must be complete")


@dataclass(frozen=True, slots=True)
class SourceInspection:
    """Observed identity of a validated raw source."""

    sha256: str
    bytes: int
    rows: int
    first_timestamp: str
    last_timestamp: str


@dataclass(frozen=True, slots=True)
class Selection:
    """The deterministic source selection encoded in a derivative."""

    offset_seconds: int
    target_timestamp: str
    first_source_row: int
    last_source_row: int
    first_timestamp: str
    last_timestamp: str
    arrival_span_ms: float
    rows: int


def _validate_sha256(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise PublicReplayError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _parse_utc_timestamp(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise PublicReplayError(f"invalid ISO-8601 timestamp: {value!r}") from error
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise PublicReplayError("source timestamps must carry an explicit UTC offset")
    return parsed.astimezone(UTC)


AZURE_CODE_2024: Final = SourceContract(
    dataset_id="azure-llm-inference-2024-code",
    url=(
        "https://github.com/Azure/AzurePublicDataset/releases/download/"
        "dataset-llm-2024/AzureLLMInferenceTrace_code_1week.csv"
    ),
    sha256="71de5c55cbc35f8f1ed0b6b7806b4cd1e9764b0058469725a6aac98023a1448f",
    bytes=691_989_454,
    rows=16_803_695,
    first_timestamp="2024-05-10 00:00:00.009930+00:00",
    last_timestamp="2024-05-16 23:59:59.929501+00:00",
    license_spdx="CC-BY-4.0",
    license_url="https://github.com/Azure/AzurePublicDataset/blob/master/LICENSE",
    attribution=(
        "Microsoft Azure and Microsoft Research, Azure LLM Inference Trace 2024 (code service)"
    ),
    citation_url=(
        "https://github.com/Azure/AzurePublicDataset/blob/master/AzureLLMInferenceDataset2024.md"
    ),
)


def _positive_decimal(raw: bytes, *, field: str, row: int) -> int:
    if not raw or any(byte < 48 or byte > 57 for byte in raw):
        raise PublicReplayError(f"source row {row}: {field} must be a decimal integer")
    value = int(raw)
    if value <= 0:
        raise PublicReplayError(f"source row {row}: {field} must be positive")
    return value


def inspect_source(path: Path) -> SourceInspection:
    """Hash and structurally validate every row of a raw Azure-format source."""

    digest = hashlib.sha256()
    byte_count = 0
    row_count = 0
    first_timestamp: str | None = None
    last_timestamp: str | None = None
    previous_timestamp: bytes | None = None
    try:
        with path.open("rb") as source:
            header = source.readline()
            digest.update(header)
            byte_count += len(header)
            if header.rstrip(b"\r\n") != b"TIMESTAMP,ContextTokens,GeneratedTokens":
                raise PublicReplayError("source has an unexpected CSV header")
            for row_count, line in enumerate(source, start=1):
                digest.update(line)
                byte_count += len(line)
                stripped = line.rstrip(b"\r\n")
                fields = stripped.split(b",")
                if len(fields) != 3:
                    raise PublicReplayError(
                        f"source row {row_count}: expected exactly three columns"
                    )
                timestamp_raw, context_raw, generated_raw = fields
                try:
                    timestamp = timestamp_raw.decode("ascii")
                except UnicodeDecodeError as error:
                    raise PublicReplayError(
                        f"source row {row_count}: timestamp is not ASCII"
                    ) from error
                _parse_utc_timestamp(timestamp)
                if previous_timestamp is not None and timestamp_raw < previous_timestamp:
                    raise PublicReplayError(
                        f"source row {row_count}: timestamps are not nondecreasing"
                    )
                _positive_decimal(context_raw, field="ContextTokens", row=row_count)
                _positive_decimal(generated_raw, field="GeneratedTokens", row=row_count)
                if first_timestamp is None:
                    first_timestamp = timestamp
                last_timestamp = timestamp
                previous_timestamp = timestamp_raw
    except OSError as error:
        raise PublicReplayError(f"cannot read source trace: {path}") from error
    if row_count == 0 or first_timestamp is None or last_timestamp is None:
        raise PublicReplayError("source trace contains no data rows")
    return SourceInspection(
        sha256=digest.hexdigest(),
        bytes=byte_count,
        rows=row_count,
        first_timestamp=first_timestamp,
        last_timestamp=last_timestamp,
    )


def verify_source_contract(
    observed: SourceInspection,
    expected: SourceContract,
) -> None:
    """Reject any raw source other than the exact registered release asset."""

    comparisons = {
        "sha256": (observed.sha256, expected.sha256),
        "bytes": (observed.bytes, expected.bytes),
        "rows": (observed.rows, expected.rows),
        "first_timestamp": (observed.first_timestamp, expected.first_timestamp),
        "last_timestamp": (observed.last_timestamp, expected.last_timestamp),
    }
    mismatches = [
        f"{field}: observed={actual!r}, expected={registered!r}"
        for field, (actual, registered) in comparisons.items()
        if actual != registered
    ]
    if mismatches:
        raise PublicReplayError("source contract mismatch; " + "; ".join(mismatches))


def _selection_target(
    source: SourceInspection,
    *,
    digest_prefix_hex_chars: int,
) -> tuple[int, datetime]:
    if not 1 <= digest_prefix_hex_chars <= 64:
        raise PublicReplayError("digest prefix length must be in [1, 64]")
    first = _parse_utc_timestamp(source.first_timestamp)
    last = _parse_utc_timestamp(source.last_timestamp)
    span_seconds = math.floor((last - first).total_seconds())
    if span_seconds < 1:
        raise PublicReplayError("source trace must span at least one full second")
    offset = int(source.sha256[:digest_prefix_hex_chars], 16) % span_seconds
    return offset, first + timedelta(seconds=offset)


def select_rows(
    source_path: Path,
    source: SourceInspection,
    *,
    selected_rows: int,
    digest_prefix_hex_chars: int,
) -> tuple[bytes, Selection]:
    """Select the first N rows after a source-hash-derived absolute offset."""

    if isinstance(selected_rows, bool) or not isinstance(selected_rows, int) or selected_rows <= 0:
        raise PublicReplayError("selected_rows must be a positive integer")
    offset_seconds, target = _selection_target(
        source,
        digest_prefix_hex_chars=digest_prefix_hex_chars,
    )
    target_text = target.isoformat(sep=" ", timespec="microseconds")
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(
        (
            "request_id",
            "arrival_ms",
            "output_tokens",
            "split",
            "prompt_tokens",
            "source_timestamp",
            "source_row",
        )
    )
    first_time: datetime | None = None
    first_timestamp: str | None = None
    last_timestamp: str | None = None
    first_source_row: int | None = None
    last_source_row: int | None = None
    selected = 0
    try:
        with source_path.open("r", encoding="ascii", newline="") as source_file:
            reader = csv.reader(source_file)
            next(reader)
            for source_row, fields in enumerate(reader):
                if len(fields) != 3:
                    raise PublicReplayError(
                        f"source row {source_row}: expected exactly three columns"
                    )
                timestamp, context_tokens, generated_tokens = fields
                if timestamp < target_text:
                    continue
                timestamp_value = _parse_utc_timestamp(timestamp)
                if first_time is None:
                    first_time = timestamp_value
                    first_timestamp = timestamp
                    first_source_row = source_row
                arrival_ms = (timestamp_value - first_time).total_seconds() * 1_000.0
                writer.writerow(
                    (
                        f"azure-code-v3-{selected:04d}",
                        f"{arrival_ms:.3f}",
                        generated_tokens,
                        "validation",
                        context_tokens,
                        timestamp,
                        source_row,
                    )
                )
                last_timestamp = timestamp
                last_source_row = source_row
                selected += 1
                if selected == selected_rows:
                    break
    except (OSError, UnicodeDecodeError, StopIteration) as error:
        raise PublicReplayError(f"cannot select rows from source: {source_path}") from error
    if (
        selected != selected_rows
        or first_time is None
        or first_timestamp is None
        or last_timestamp is None
        or first_source_row is None
        or last_source_row is None
    ):
        raise PublicReplayError(
            f"source ended after {selected} selected rows; required {selected_rows}"
        )
    last_time = _parse_utc_timestamp(last_timestamp)
    return output.getvalue().encode("ascii"), Selection(
        offset_seconds=offset_seconds,
        target_timestamp=target_text,
        first_source_row=first_source_row,
        last_source_row=last_source_row,
        first_timestamp=first_timestamp,
        last_timestamp=last_timestamp,
        arrival_span_ms=(last_time - first_time).total_seconds() * 1_000.0,
        rows=selected,
    )


def _manifest_payload(
    *,
    contract: SourceContract,
    observed: SourceInspection,
    selection: Selection,
    digest_prefix_hex_chars: int,
    derivative_file: str,
    derivative_bytes: bytes,
) -> dict[str, object]:
    return {
        "schema": SCHEMA,
        "evidence_class": "public-workload-trace-input",
        "measurement_warning": WARNING,
        "source": {
            "dataset_id": contract.dataset_id,
            "url": contract.url,
            "sha256": observed.sha256,
            "bytes": observed.bytes,
            "rows": observed.rows,
            "first_timestamp": observed.first_timestamp,
            "last_timestamp": observed.last_timestamp,
            "license_spdx": contract.license_spdx,
            "license_url": contract.license_url,
            "attribution": contract.attribution,
            "citation_url": contract.citation_url,
        },
        "selection": {
            "algorithm": "source-sha256-offset-first-n-v1",
            "digest_prefix_hex_chars": digest_prefix_hex_chars,
            "offset_seconds": selection.offset_seconds,
            "target_timestamp": selection.target_timestamp,
            "first_source_row_zero_based": selection.first_source_row,
            "last_source_row_zero_based": selection.last_source_row,
            "first_timestamp": selection.first_timestamp,
            "last_timestamp": selection.last_timestamp,
            "arrival_span_ms": selection.arrival_span_ms,
            "rows": selection.rows,
            "split": "validation",
        },
        "derivative": {
            "file": derivative_file,
            "sha256": hashlib.sha256(derivative_bytes).hexdigest(),
            "bytes": len(derivative_bytes),
            "rows": selection.rows,
            "format": "fissionspec replay CSV v1",
            "modifications": (
                "Selected a consecutive digest-derived window; converted timestamps to "
                "relative milliseconds; renamed token-count columns; added deterministic "
                "request IDs, validation split labels, and source-row provenance."
            ),
        },
        "use": {
            "registered_anchor": "V3",
            "arrival_rescaling": (
                "Scale relative arrival_ms only after Stage 1 target-only saturation "
                "calibration to the preregistered 0.70 offered-load fraction."
            ),
            "claim_boundary": (
                "The trace supplies arrivals and request shapes only. It contains no "
                "prompts, model outputs, SSD outcomes, kernel timings, or GPU evidence."
            ),
        },
    }


def _atomic_write(path: Path, payload: bytes, *, force: bool) -> None:
    if path.exists() and not force:
        raise PublicReplayError(f"refusing to overwrite existing path without --force: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def freeze_replay(
    source_path: Path,
    trace_path: Path,
    manifest_path: Path,
    *,
    contract: SourceContract,
    selected_rows: int = 1_024,
    digest_prefix_hex_chars: int = 12,
    force: bool = False,
) -> dict[str, object]:
    """Create a deterministic replay and self-hashed provenance manifest."""

    if trace_path.resolve() == manifest_path.resolve():
        raise PublicReplayError("trace and manifest paths must differ")
    observed = inspect_source(source_path)
    verify_source_contract(observed, contract)
    derivative, selection = select_rows(
        source_path,
        observed,
        selected_rows=selected_rows,
        digest_prefix_hex_chars=digest_prefix_hex_chars,
    )
    payload = _manifest_payload(
        contract=contract,
        observed=observed,
        selection=selection,
        digest_prefix_hex_chars=digest_prefix_hex_chars,
        derivative_file=trace_path.name,
        derivative_bytes=derivative,
    )
    manifest: dict[str, object] = {
        **payload,
        "payload_sha256": sha256_document(payload),
    }
    if (trace_path.exists() or manifest_path.exists()) and not force:
        raise PublicReplayError("refusing partial overwrite without --force")
    _atomic_write(trace_path, derivative, force=force)
    try:
        _atomic_write(manifest_path, canonical_json_bytes(manifest), force=force)
    except Exception:
        if not force and trace_path.exists():
            trace_path.unlink()
        raise
    verify_frozen_replay(
        trace_path,
        manifest_path,
        expected_contract=contract,
        expected_selected_rows=selected_rows,
        expected_digest_prefix_hex_chars=digest_prefix_hex_chars,
    )
    return manifest


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise PublicReplayError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _load_manifest(path: Path) -> dict[str, object]:
    try:
        raw = json.loads(
            path.read_bytes(),
            object_pairs_hook=_strict_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-standard JSON constant: {value}")
            ),
        )
    except (OSError, UnicodeDecodeError, ValueError) as error:
        raise PublicReplayError(f"manifest is not strict UTF-8 JSON: {path}") from error
    if not isinstance(raw, dict):
        raise PublicReplayError("manifest root must be an object")
    return cast(dict[str, object], raw)


def _expected_source_mapping(contract: SourceContract) -> dict[str, object]:
    return {
        "dataset_id": contract.dataset_id,
        "url": contract.url,
        "sha256": contract.sha256,
        "bytes": contract.bytes,
        "rows": contract.rows,
        "first_timestamp": contract.first_timestamp,
        "last_timestamp": contract.last_timestamp,
        "license_spdx": contract.license_spdx,
        "license_url": contract.license_url,
        "attribution": contract.attribution,
        "citation_url": contract.citation_url,
    }


def _verify_derivative_rows(
    trace_bytes: bytes,
    selection: dict[str, object],
    *,
    rows: int,
) -> None:
    try:
        text = trace_bytes.decode("ascii")
    except UnicodeDecodeError as error:
        raise PublicReplayError("frozen replay is not ASCII CSV") from error
    reader = csv.reader(io.StringIO(text, newline=""))
    try:
        header = next(reader)
    except StopIteration as error:
        raise PublicReplayError("frozen replay is empty") from error
    expected_header = [
        "request_id",
        "arrival_ms",
        "output_tokens",
        "split",
        "prompt_tokens",
        "source_timestamp",
        "source_row",
    ]
    if header != expected_header:
        raise PublicReplayError("frozen replay has an unexpected CSV header")
    first_time: datetime | None = None
    previous_arrival = -1.0
    first_timestamp: str | None = None
    last_timestamp: str | None = None
    first_source_row: int | None = None
    last_source_row: int | None = None
    count = 0
    for count, fields in enumerate(reader, start=1):
        if len(fields) != len(expected_header):
            raise PublicReplayError(f"frozen replay row {count}: expected seven columns")
        (
            request_id,
            arrival_raw,
            output_raw,
            split,
            prompt_raw,
            timestamp,
            source_row_raw,
        ) = fields
        ordinal = count - 1
        if request_id != f"azure-code-v3-{ordinal:04d}":
            raise PublicReplayError(f"frozen replay row {count}: request ID mismatch")
        if split != "validation":
            raise PublicReplayError(f"frozen replay row {count}: split must be validation")
        timestamp_value = _parse_utc_timestamp(timestamp)
        if first_time is None:
            first_time = timestamp_value
            first_timestamp = timestamp
        expected_arrival = (timestamp_value - first_time).total_seconds() * 1_000.0
        if arrival_raw != f"{expected_arrival:.3f}":
            raise PublicReplayError(f"frozen replay row {count}: relative arrival mismatch")
        arrival = float(arrival_raw)
        if not math.isfinite(arrival) or arrival < previous_arrival:
            raise PublicReplayError(f"frozen replay row {count}: arrivals are not ordered")
        previous_arrival = arrival
        _positive_decimal(output_raw.encode("ascii"), field="output_tokens", row=count)
        _positive_decimal(prompt_raw.encode("ascii"), field="prompt_tokens", row=count)
        try:
            source_row = int(source_row_raw)
        except ValueError as error:
            raise PublicReplayError(
                f"frozen replay row {count}: source_row must be an integer"
            ) from error
        if source_row < 0:
            raise PublicReplayError(f"frozen replay row {count}: source_row is negative")
        if last_source_row is not None and source_row != last_source_row + 1:
            raise PublicReplayError(f"frozen replay row {count}: source rows are not consecutive")
        if first_source_row is None:
            first_source_row = source_row
        last_source_row = source_row
        last_timestamp = timestamp
    if count != rows or first_time is None:
        raise PublicReplayError("frozen replay row count disagrees with manifest")
    comparisons = {
        "first timestamp": (first_timestamp, selection.get("first_timestamp")),
        "last timestamp": (last_timestamp, selection.get("last_timestamp")),
        "first source row": (
            first_source_row,
            selection.get("first_source_row_zero_based"),
        ),
        "last source row": (
            last_source_row,
            selection.get("last_source_row_zero_based"),
        ),
    }
    mismatches = [
        field for field, (observed, expected) in comparisons.items() if observed != expected
    ]
    if mismatches:
        raise PublicReplayError(
            "frozen replay provenance disagrees with manifest: " + ", ".join(mismatches)
        )
    if not isinstance(selection.get("arrival_span_ms"), (int, float)) or isinstance(
        selection.get("arrival_span_ms"), bool
    ):
        raise PublicReplayError("manifest arrival_span_ms must be numeric")
    if not math.isclose(
        previous_arrival,
        float(cast(float, selection["arrival_span_ms"])),
        rel_tol=0.0,
        abs_tol=0.000_5,
    ):
        raise PublicReplayError("frozen replay span disagrees with manifest")


def verify_frozen_replay(
    trace_path: Path,
    manifest_path: Path,
    *,
    expected_contract: SourceContract | None = None,
    expected_selected_rows: int | None = None,
    expected_digest_prefix_hex_chars: int | None = None,
) -> dict[str, object]:
    """Verify manifest integrity, derivative bytes, and replay-schema loading."""

    manifest = _load_manifest(manifest_path)
    expected_keys = {
        "schema",
        "evidence_class",
        "measurement_warning",
        "source",
        "selection",
        "derivative",
        "use",
        "payload_sha256",
    }
    if set(manifest) != expected_keys:
        raise PublicReplayError("manifest fields do not match the closed schema")
    if manifest["schema"] != SCHEMA:
        raise PublicReplayError("unsupported public replay manifest schema")
    if manifest["evidence_class"] != "public-workload-trace-input":
        raise PublicReplayError("manifest has an invalid evidence class")
    if manifest["measurement_warning"] != WARNING:
        raise PublicReplayError("manifest evidence warning is missing")
    supplied_hash = _validate_sha256(
        manifest["payload_sha256"],
        field="manifest payload_sha256",
    )
    payload = dict(manifest)
    payload.pop("payload_sha256")
    if sha256_document(payload) != supplied_hash:
        raise PublicReplayError("manifest payload hash mismatch")
    raw_source = manifest["source"]
    raw_selection = manifest["selection"]
    raw_derivative = manifest["derivative"]
    raw_use = manifest["use"]
    if not all(
        isinstance(value, dict) for value in (raw_source, raw_selection, raw_derivative, raw_use)
    ):
        raise PublicReplayError("manifest nested sections must be objects")
    source = cast(dict[str, object], raw_source)
    derivative = cast(dict[str, object], raw_derivative)
    selection = cast(dict[str, object], raw_selection)
    use = cast(dict[str, object], raw_use)
    if set(source) != SOURCE_FIELDS:
        raise PublicReplayError("manifest source fields do not match the closed schema")
    if set(selection) != SELECTION_FIELDS:
        raise PublicReplayError("manifest selection fields do not match the closed schema")
    if set(derivative) != DERIVATIVE_FIELDS:
        raise PublicReplayError("manifest derivative fields do not match the closed schema")
    if set(use) != USE_FIELDS:
        raise PublicReplayError("manifest use fields do not match the closed schema")
    _validate_sha256(source.get("sha256"), field="source sha256")
    if expected_contract is not None and source != _expected_source_mapping(expected_contract):
        raise PublicReplayError("manifest source does not match the registered contract")
    if selection.get("algorithm") != "source-sha256-offset-first-n-v1":
        raise PublicReplayError("manifest has an unsupported selection algorithm")
    if selection.get("split") != "validation":
        raise PublicReplayError("manifest selection split must be validation")
    if expected_selected_rows is not None and selection.get("rows") != expected_selected_rows:
        raise PublicReplayError("manifest selection row count differs from registration")
    if (
        expected_digest_prefix_hex_chars is not None
        and selection.get("digest_prefix_hex_chars") != expected_digest_prefix_hex_chars
    ):
        raise PublicReplayError("manifest digest-prefix length differs from registration")
    if derivative.get("file") != trace_path.name:
        raise PublicReplayError("manifest derivative filename mismatch")
    try:
        trace_bytes = trace_path.read_bytes()
    except OSError as error:
        raise PublicReplayError(f"cannot read frozen replay: {trace_path}") from error
    if hashlib.sha256(trace_bytes).hexdigest() != _validate_sha256(
        derivative.get("sha256"),
        field="derivative sha256",
    ):
        raise PublicReplayError("frozen replay hash mismatch")
    if derivative.get("bytes") != len(trace_bytes):
        raise PublicReplayError("frozen replay byte count mismatch")
    rows = derivative.get("rows")
    if (
        isinstance(rows, bool)
        or not isinstance(rows, int)
        or rows <= 0
        or selection.get("rows") != rows
    ):
        raise PublicReplayError("manifest has inconsistent replay row counts")
    loaded = load_trace_csv(trace_path, split="validation", name="public-replay-verification")
    if loaded.source_rows != rows or loaded.selected_rows != rows:
        raise PublicReplayError("replay loader row count disagrees with manifest")
    if loaded.source_sha256 != derivative["sha256"]:
        raise PublicReplayError("replay loader hash disagrees with manifest")
    request_ids = [request.request_id for request in loaded.workload]
    if len(set(request_ids)) != len(request_ids):
        raise PublicReplayError("frozen replay request IDs are not unique")
    if loaded.workload.requests[0].arrival_ms != 0.0:
        raise PublicReplayError("frozen replay must begin at relative time zero")
    _verify_derivative_rows(trace_bytes, selection, rows=rows)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Freeze or verify FissionSpec's hash-selected Azure public replay."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("source", type=Path)
    freeze.add_argument("--trace", type=Path, default=DEFAULT_TRACE)
    freeze.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    freeze.add_argument("--force", action="store_true")
    verify = subparsers.add_parser("verify")
    verify.add_argument("--trace", type=Path, default=DEFAULT_TRACE)
    verify.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the public-replay freezer or offline verifier."""

    args = _parser().parse_args(argv)
    try:
        if args.command == "freeze":
            manifest = freeze_replay(
                args.source,
                args.trace,
                args.manifest,
                contract=AZURE_CODE_2024,
                force=args.force,
            )
        else:
            manifest = verify_frozen_replay(
                args.trace,
                args.manifest,
                expected_contract=AZURE_CODE_2024,
                expected_selected_rows=1_024,
                expected_digest_prefix_hex_chars=12,
            )
    except (PublicReplayError, OSError, ValueError) as error:
        print(f"public replay failed closed: {error}", file=sys.stderr)
        return 2
    print(
        canonical_json_bytes(
            {
                "manifest": str(args.manifest),
                "payload_sha256": manifest["payload_sha256"],
                "status": "verified",
                "trace": str(args.trace),
            }
        ).decode(),
        end="",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
