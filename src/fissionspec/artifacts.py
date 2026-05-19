"""Canonical, hash-linked evidence artifacts for CPU simulations."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Final

from .metrics import summarize
from .model import SimulationResult

TRACE_SCHEMA_VERSION: Final = 1
SIMULATION_WARNING: Final = "SIMULATION MODEL OUTPUT — NOT A GPU MEASUREMENT."


class ArtifactIntegrityError(ValueError):
    """Raised when a hash-linked artifact is malformed or was modified."""


def canonical_json_bytes(document: object) -> bytes:
    """Serialize strict JSON with one cross-platform canonical representation."""

    return (
        json.dumps(
            document,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def sha256_document(document: object) -> str:
    """Hash the canonical representation of a JSON-compatible document."""

    return hashlib.sha256(canonical_json_bytes(document)).hexdigest()


def implementation_sha256(root: str | Path, paths: Sequence[str | Path]) -> str:
    """Hash an ordered set of implementation files with framed relative names."""

    base = Path(root).resolve()
    normalized: list[tuple[str, Path]] = []
    for supplied in paths:
        path = (base / supplied).resolve()
        try:
            relative = path.relative_to(base).as_posix()
        except ValueError as error:
            raise ValueError("implementation paths must remain below root") from error
        if not path.is_file():
            raise ValueError(f"implementation path is not a file: {relative}")
        normalized.append((relative, path))
    if not normalized:
        raise ValueError("at least one implementation path is required")
    if len({relative for relative, _ in normalized}) != len(normalized):
        raise ValueError("implementation paths must be unique")

    digest = hashlib.sha256()
    digest.update(b"fissionspec/implementation/v1\0")
    for relative, path in sorted(normalized):
        name = relative.encode("utf-8")
        content = path.read_bytes()
        digest.update(len(name).to_bytes(8, "big"))
        digest.update(name)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def environment_manifest(
    *,
    implementation_digest: str | None = None,
) -> dict[str, object]:
    """Return non-semantic runtime provenance kept outside golden payloads."""

    return {
        "schema_version": 1,
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "cache_tag": sys.implementation.cache_tag,
        },
        "platform": {
            "machine": platform.machine(),
            "release": platform.release(),
            "system": platform.system(),
        },
        "implementation_sha256": implementation_digest,
    }


def _trace_payload(
    result: SimulationResult,
    *,
    implementation_digest: str | None,
    source_hashes: Mapping[str, str],
) -> dict[str, object]:
    profile = result.profile
    return {
        "schema_version": TRACE_SCHEMA_VERSION,
        "evidence_class": "simulation-model",
        "measurement_warning": SIMULATION_WARNING,
        "identity": {
            "policy": result.policy_name,
            "hardware_profile": result.hardware_name,
            "workload": result.workload_name,
            "rng_provenance": result.rng_provenance,
        },
        "provenance": {
            "implementation_sha256": implementation_digest,
            "source_sha256": dict(sorted(source_hashes.items())),
        },
        "configuration": {
            "profile": {
                "name": profile.name,
                "target_curve": profile.target_curve.points,
                "draft_curve": profile.draft_curve.points,
                "recovery_curve": profile.recovery_curve.points,
                "verifier_slot_ms": profile.verifier_slot_ms,
            },
            "workload": {
                "name": result.workload.name,
                "requests": [asdict(request) for request in result.workload],
            },
        },
        "summary": summarize(result).as_dict(),
        "trace": {
            "started_ms": result.started_ms,
            "finished_ms": result.finished_ms,
            "requests": [asdict(request) for request in result.requests],
            "target_launches": [asdict(launch) for launch in result.target_launches],
            "draft_launches": [asdict(launch) for launch in result.draft_launches],
        },
    }


def simulation_trace_document(
    result: SimulationResult,
    *,
    implementation_digest: str | None = None,
    source_hashes: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Build a full per-request/event trace with a self-verifying payload hash."""

    if implementation_digest is not None and (
        len(implementation_digest) != 64
        or any(character not in "0123456789abcdef" for character in implementation_digest)
    ):
        raise ValueError("implementation_digest must be a lowercase SHA-256 hex digest")
    hashes = {} if source_hashes is None else dict(source_hashes)
    for label, digest in hashes.items():
        if not label:
            raise ValueError("source hash labels must not be empty")
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError("source hashes must be lowercase SHA-256 hex digests")
    payload = _trace_payload(
        result,
        implementation_digest=implementation_digest,
        source_hashes=hashes,
    )
    return {
        **payload,
        "payload_sha256": sha256_document(payload),
    }


def verify_trace_document(document: Mapping[str, object]) -> str:
    """Validate a trace envelope and return its verified payload digest."""

    supplied = document.get("payload_sha256")
    if not isinstance(supplied, str):
        raise ArtifactIntegrityError("artifact is missing payload_sha256")
    payload = dict(document)
    payload.pop("payload_sha256", None)
    if payload.get("schema_version") != TRACE_SCHEMA_VERSION:
        raise ArtifactIntegrityError("unsupported trace schema_version")
    if payload.get("evidence_class") != "simulation-model":
        raise ArtifactIntegrityError("unexpected evidence_class")
    if payload.get("measurement_warning") != SIMULATION_WARNING:
        raise ArtifactIntegrityError("simulation evidence warning is missing")
    actual = sha256_document(payload)
    if supplied != actual:
        raise ArtifactIntegrityError("artifact payload hash mismatch")
    return actual


def write_trace_document(path: str | Path, document: Mapping[str, object]) -> None:
    """Verify and atomically write a canonical trace document."""

    verify_trace_document(document)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_bytes(canonical_json_bytes(document))
    temporary.replace(destination)


def load_trace_document(path: str | Path) -> dict[str, object]:
    """Read strict JSON, verify its payload link, and return the document."""

    try:
        raw = json.loads(
            Path(path).read_bytes(),
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-standard JSON numeric constant: {value}")
            ),
        )
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ArtifactIntegrityError("artifact is not valid UTF-8 JSON") from error
    if not isinstance(raw, dict):
        raise ArtifactIntegrityError("artifact root must be an object")
    verify_trace_document(raw)
    return raw


__all__ = [
    "SIMULATION_WARNING",
    "TRACE_SCHEMA_VERSION",
    "ArtifactIntegrityError",
    "canonical_json_bytes",
    "environment_manifest",
    "implementation_sha256",
    "load_trace_document",
    "sha256_document",
    "simulation_trace_document",
    "verify_trace_document",
    "write_trace_document",
]
