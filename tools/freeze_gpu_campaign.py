#!/usr/bin/env python3
"""Freeze or verify the zero-GPU handoff for the accelerator campaign.

The command never launches an accelerator job. It binds the exact CPU evidence
files, protocol, source commit, and three pre-calibration workload templates to
a canonical campaign plan, then records a zero-resource F0 pass. The resulting
ledger intentionally authorizes zero GPU seconds until an operator separately
seals the immediate F1 budget.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, TypeAlias, cast

from fissionspec.spend_gate import (
    CampaignLedger,
    CampaignPlan,
    CampaignStage,
    GateRecord,
    GateVerdict,
)

JsonObject: TypeAlias = dict[str, object]

WARNING: Final = "ZERO-GPU CAMPAIGN FREEZE — CONTAINS NO ACCELERATOR MEASUREMENT."
CLAIM_BOUNDARY: Final = (
    "This bundle proves that the declared CPU artifacts, workload templates, "
    "protocol, replay caps, and F0 decision were frozen together. It does not "
    "establish physical row omission, kernel speedup, production correctness, "
    "or any GPU performance result."
)
DEFAULT_OUTPUT = Path("configs/gpu_campaign")
PROTOCOL_PATH = Path("paper/gpu_preregistration.md")
ANCHOR_SOURCE_PATHS: Final = (
    Path("src/fissionspec/workload_generators.py"),
    Path("src/fissionspec/rng.py"),
    Path("configs/traces/azure_llm_code_v3_1024.csv"),
    Path("configs/traces/azure_llm_code_v3_1024.manifest.json"),
)


@dataclass(frozen=True, slots=True)
class EvidenceSpec:
    artifact_id: str
    path: Path

    def __post_init__(self) -> None:
        if (
            not self.artifact_id
            or self.artifact_id.strip("-") != self.artifact_id
            or any(
                character not in "abcdefghijklmnopqrstuvwxyz0123456789-"
                for character in self.artifact_id
            )
        ):
            raise ValueError("evidence artifact_id must be a lowercase ASCII slug")
        if self.path.is_absolute() or not self.path.parts or ".." in self.path.parts:
            raise ValueError("evidence path must be a repository-relative path")


DEFAULT_EVIDENCE: Final = (
    EvidenceSpec(
        "azure-public-replay",
        Path("configs/traces/azure_llm_code_v3_1024.manifest.json"),
    ),
    EvidenceSpec(
        "causal-mechanism-study",
        Path("experiments/results/mechanism_study/manifest.json"),
    ),
    EvidenceSpec(
        "controller-overhead-audit",
        Path("experiments/results/controller_overhead/controller_overhead_manifest.json"),
    ),
    EvidenceSpec(
        "cpu-completion-study",
        Path("experiments/results/cpu_completion_full/manifest.json"),
    ),
    EvidenceSpec(
        "cross-language-horizon2",
        Path("fixtures/cross_language/horizon2.tsv"),
    ),
    EvidenceSpec(
        "cross-language-latency",
        Path("fixtures/cross_language/latency.tsv"),
    ),
    EvidenceSpec(
        "cross-language-malformed",
        Path("fixtures/cross_language/malformed.tsv"),
    ),
    EvidenceSpec(
        "cross-language-metrics",
        Path("fixtures/cross_language/metrics.tsv"),
    ),
    EvidenceSpec(
        "cross-language-transactions",
        Path("fixtures/cross_language/transactions.tsv"),
    ),
    EvidenceSpec(
        "expanded-exact-oracle",
        Path("experiments/results/oracle_campaign/manifest.json"),
    ),
    EvidenceSpec("offline-output-audit", Path("src/fissionspec/output_audit.py")),
    EvidenceSpec("offline-output-audit-cli", Path("tools/run_output_audit.py")),
    EvidenceSpec("offline-output-audit-tests", Path("tests/test_output_audit.py")),
    EvidenceSpec(
        "sglang-pr-head-prototype",
        Path("integrations/sglang/patch_manifest.json"),
    ),
    EvidenceSpec(
        "transformer-stack-semantics",
        Path("experiments/results/cpu_transformer_semantics/evidence.json"),
    ),
)

OUTPUT_FILES: Final = (
    "validation_anchor_v1.json",
    "validation_anchor_v2.json",
    "validation_anchor_v3.json",
    "cpu_evidence_index.json",
    "campaign_plan.json",
    "campaign_ledger_f0.json",
    "bundle_manifest.json",
)


class CampaignFreezeError(ValueError):
    """Raised when the zero-GPU campaign bundle is incomplete or inconsistent."""


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _regular_file(path: Path, *, field: str) -> Path:
    if not path.is_file() or path.is_symlink():
        raise CampaignFreezeError(f"{field} must be a regular non-symlink file: {path}")
    return path


def _self_hashed(payload: Mapping[str, object]) -> JsonObject:
    normalized = dict(payload)
    return {
        **normalized,
        "payload_sha256": _sha256_bytes(_canonical_bytes(normalized)),
    }


def _reject_constant(value: str) -> object:
    raise CampaignFreezeError(f"non-standard JSON constant is forbidden: {value}")


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> JsonObject:
    document: JsonObject = {}
    for key, value in pairs:
        if key in document:
            raise CampaignFreezeError(f"duplicate JSON key: {key}")
        document[key] = value
    return document


def _load_json(path: Path) -> JsonObject:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_constant,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CampaignFreezeError(f"cannot load strict JSON: {path}") from error
    if not isinstance(value, dict):
        raise CampaignFreezeError(f"JSON document must be an object: {path}")
    return cast(JsonObject, value)


def _strict_keys(
    document: Mapping[str, object],
    expected: set[str],
    *,
    field: str,
) -> None:
    if set(document) != expected:
        raise CampaignFreezeError(f"{field} has unexpected or missing fields")


def _digest(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value.lower() != value
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise CampaignFreezeError(f"{field} must be 64 lowercase hexadecimal characters")
    return value


def _positive_integer(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise CampaignFreezeError(f"{field} must be a positive integer")
    return value


def _positive_number(value: object, *, field: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise CampaignFreezeError(f"{field} must be a finite positive number")
    return float(value)


def _verify_self_hash(document: Mapping[str, object], *, field: str) -> str:
    supplied = _digest(document.get("payload_sha256"), field=f"{field}.payload_sha256")
    payload = dict(document)
    payload.pop("payload_sha256")
    if _sha256_bytes(_canonical_bytes(payload)) != supplied:
        raise CampaignFreezeError(f"{field} payload hash mismatch")
    return supplied


def _embedded_payload_sha256(path: Path) -> str | None:
    if path.suffix != ".json":
        return None
    document = _load_json(path)
    supplied = document.get("payload_sha256")
    if supplied is None:
        return None
    digest = _digest(supplied, field=f"{path}.payload_sha256")
    payload = dict(document)
    payload.pop("payload_sha256")
    compact = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    accepted = {
        _sha256_bytes(compact),
        _sha256_bytes(compact + b"\n"),
    }
    if digest not in accepted:
        raise CampaignFreezeError(f"embedded payload hash mismatch: {path}")
    return digest


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _run_git(repo_root: Path, arguments: Sequence[str]) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise CampaignFreezeError(f"git {' '.join(arguments)} failed: {completed.stderr.strip()}")
    return completed.stdout.strip()


def _validate_commit(
    repo_root: Path,
    code_commit: str,
    *,
    tracked_paths: Sequence[Path],
) -> None:
    if len(code_commit) not in (40, 64) or any(
        character not in "0123456789abcdef" for character in code_commit
    ):
        raise CampaignFreezeError("code commit must be a full lowercase object ID")
    resolved = _run_git(repo_root, ["rev-parse", f"{code_commit}^{{commit}}"])
    if resolved != code_commit:
        raise CampaignFreezeError("code commit does not resolve to the supplied full ID")
    for relative in tracked_paths:
        completed = subprocess.run(
            ["git", "show", f"{code_commit}:{relative.as_posix()}"],
            cwd=repo_root,
            check=False,
            capture_output=True,
        )
        if completed.returncode != 0:
            raise CampaignFreezeError(
                f"{relative.as_posix()} is absent from code commit {code_commit}"
            )
        working_path = repo_root / relative
        _regular_file(working_path, field="commit-bound input")
        if completed.stdout != working_path.read_bytes():
            raise CampaignFreezeError(
                f"{relative.as_posix()} differs from code commit {code_commit}"
            )


def _resolve_bundle_paths(repo_root: Path, output_dir: Path) -> tuple[Path, Path]:
    if output_dir.is_symlink():
        raise CampaignFreezeError("campaign output directory may not be a symlink")
    resolved_root = repo_root.resolve()
    resolved_output = output_dir.resolve()
    if resolved_output == resolved_root or resolved_root not in resolved_output.parents:
        raise CampaignFreezeError("campaign output directory must be inside the repository")
    return resolved_root, resolved_output


def _ensure_clean_before_freeze(repo_root: Path, output_dir: Path) -> None:
    status = _run_git(repo_root, ["status", "--porcelain=v1", "--untracked-files=all"])
    try:
        relative_output = output_dir.resolve().relative_to(repo_root.resolve())
    except ValueError as error:
        raise CampaignFreezeError("output directory must be inside the repository") from error
    unexpected: list[str] = []
    for line in status.splitlines():
        path_text = line[3:]
        if " -> " in path_text:
            path_text = path_text.split(" -> ", 1)[1]
        candidate = Path(path_text)
        if candidate == relative_output or relative_output in candidate.parents:
            continue
        unexpected.append(line)
    if unexpected:
        raise CampaignFreezeError("repository has changes outside the campaign output directory")


def _source_hashes(repo_root: Path) -> JsonObject:
    return {
        path.as_posix(): _sha256_file(_regular_file(repo_root / path, field="anchor source"))
        for path in ANCHOR_SOURCE_PATHS
    }


def _public_replay_contract(repo_root: Path) -> JsonObject:
    manifest_path = repo_root / "configs/traces/azure_llm_code_v3_1024.manifest.json"
    _regular_file(manifest_path, field="public replay manifest")
    manifest = _load_json(manifest_path)
    _verify_self_hash(manifest, field="public replay manifest")
    derivative = manifest.get("derivative")
    selection = manifest.get("selection")
    if not isinstance(derivative, dict) or not isinstance(selection, dict):
        raise CampaignFreezeError("public replay manifest is missing derivative/selection")
    csv_path = repo_root / "configs/traces/azure_llm_code_v3_1024.csv"
    _regular_file(csv_path, field="public replay derivative")
    derivative_sha = _digest(
        derivative.get("sha256"),
        field="public replay derivative.sha256",
    )
    if _sha256_file(csv_path) != derivative_sha:
        raise CampaignFreezeError("public replay CSV does not match its manifest")
    rows = _positive_integer(
        derivative.get("rows"),
        field="public replay derivative.rows",
    )
    if rows < 2:
        raise CampaignFreezeError("public replay must contain at least two rows")
    arrival_span_ms = _positive_number(
        selection.get("arrival_span_ms"),
        field="public replay selection.arrival_span_ms",
    )
    return {
        "manifest_payload_sha256": manifest["payload_sha256"],
        "manifest_file_sha256": _sha256_file(manifest_path),
        "derivative_sha256": derivative_sha,
        "rows": rows,
        "arrival_span_ms": arrival_span_ms,
    }


def build_anchor_documents(repo_root: Path) -> tuple[JsonObject, JsonObject, JsonObject]:
    """Build the three exact symbolic workload templates frozen before F1."""

    sources = _source_hashes(repo_root)
    common: JsonObject = {
        "schema": "fissionspec.validation-anchor-template.v1",
        "evidence_class": "pre-gpu-workload-template",
        "measurement_warning": WARNING,
        "claim_boundary": (
            "The template fixes workload generation and the deterministic Stage-1 "
            "rescaling transform. It contains no measured saturation rate, model "
            "output, kernel timing, or accelerator result."
        ),
        "paired_blocks": {
            "maximum_blocks": 50,
            "policy_order": "ABBA/BAAB alternating by completed block",
            "semantic_rng_key": ("fissionspec-gpu/<anchor_id>/block-<zero-padded-index>/v1"),
            "materialization_rule": (
                "Only the Stage-1 target-only saturation scalar S may be substituted; "
                "all other fields and source hashes are immutable."
            ),
        },
        "source_sha256": sources,
    }
    v1 = _self_hashed(
        {
            **common,
            "anchor_id": "V1",
            "request_shape": {
                "prompt_tokens": 128,
                "output_tokens": 32,
                "temperature": 0.6,
            },
            "arrival_template": {
                "process": "exact-two-state-mmpp",
                "requests": 1024,
                "arrival_rate_multipliers_of_S": [0.35, 1.05],
                "transition_rate_multipliers_of_S": [0.05, 0.05],
                "initial_state": "completed_block_index_mod_2",
                "stationary_mean_offered_load_fraction": 0.70,
                "materialization": (
                    "Pass each multiplier times measured target-only saturation "
                    "S requests/ms to mmpp_arrivals."
                ),
            },
        }
    )
    v2 = _self_hashed(
        {
            **common,
            "anchor_id": "V2",
            "request_shape": {
                "prompt_tokens": 16_384,
                "output_tokens": 256,
                "temperature": 1.0,
            },
            "arrival_template": {
                "process": "finite-mean-pareto",
                "requests": 1024,
                "tail_index": 1.35,
                "mean_offered_load_fraction": 0.90,
                "minimum_gap_ms_formula": ("((tail_index - 1) / tail_index) / (0.90 * S)"),
                "materialization": (
                    "Use the formula result as minimum_interarrival_ms in "
                    "pareto_arrivals with the registered block RNG key."
                ),
            },
        }
    )
    v3 = _self_hashed(
        {
            **common,
            "anchor_id": "V3",
            "request_shape": {
                "prompt_tokens": "ContextTokens from frozen public replay",
                "output_tokens": "GeneratedTokens from frozen public replay",
                "temperature": 0.0,
            },
            "arrival_template": {
                "process": "frozen-azure-public-replay",
                "mean_offered_load_fraction": 0.70,
                "rescaling_formula": (
                    "scaled_arrival_ms = arrival_ms * ((rows - 1) / arrival_span_ms) / (0.70 * S)"
                ),
                "source": _public_replay_contract(repo_root),
            },
        }
    )
    return v1, v2, v3


def _evidence_records(
    repo_root: Path,
    evidence: Sequence[EvidenceSpec],
) -> list[JsonObject]:
    if tuple(sorted(item.artifact_id for item in evidence)) != tuple(
        item.artifact_id for item in evidence
    ):
        raise CampaignFreezeError("evidence specs must be sorted by artifact_id")
    if len({item.artifact_id for item in evidence}) != len(evidence):
        raise CampaignFreezeError("evidence artifact IDs must be unique")
    if len({item.path for item in evidence}) != len(evidence):
        raise CampaignFreezeError("evidence paths must be unique")
    records: list[JsonObject] = []
    for item in evidence:
        path = repo_root / item.path
        _regular_file(path, field="evidence")
        records.append(
            {
                "artifact_id": item.artifact_id,
                "path": item.path.as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
                "embedded_payload_sha256": _embedded_payload_sha256(path),
            }
        )
    return records


def build_evidence_index(
    repo_root: Path,
    *,
    code_commit: str,
    anchor_documents: Sequence[Mapping[str, object]],
    evidence: Sequence[EvidenceSpec] = DEFAULT_EVIDENCE,
) -> JsonObject:
    anchor_records = []
    for anchor in anchor_documents:
        anchor_id = anchor.get("anchor_id")
        if anchor_id not in {"V1", "V2", "V3"}:
            raise CampaignFreezeError("anchor document has an invalid ID")
        anchor_records.append(
            {
                "anchor_id": anchor_id,
                "payload_sha256": _verify_self_hash(
                    anchor,
                    field=f"anchor {anchor_id}",
                ),
            }
        )
    if [item["anchor_id"] for item in anchor_records] != ["V1", "V2", "V3"]:
        raise CampaignFreezeError("anchor documents must be ordered V1, V2, V3")
    return _self_hashed(
        {
            "schema": "fissionspec.cpu-evidence-index.v1",
            "evidence_class": "zero-gpu-release-evidence-index",
            "measurement_warning": WARNING,
            "claim_boundary": CLAIM_BOUNDARY,
            "source_commit": code_commit,
            "protocol": {
                "path": PROTOCOL_PATH.as_posix(),
                "sha256": _sha256_file(
                    _regular_file(
                        repo_root / PROTOCOL_PATH,
                        field="campaign protocol",
                    )
                ),
            },
            "validation_anchors": anchor_records,
            "artifacts": _evidence_records(repo_root, evidence),
        }
    )


def _bundle_manifest(output_dir: Path, *, code_commit: str) -> JsonObject:
    files = []
    for filename in OUTPUT_FILES[:-1]:
        path = output_dir / filename
        files.append(
            {
                "path": filename,
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    return _self_hashed(
        {
            "schema": "fissionspec.zero-gpu-campaign-bundle.v1",
            "measurement_warning": WARNING,
            "claim_boundary": CLAIM_BOUNDARY,
            "source_commit": code_commit,
            "files": files,
        }
    )


def freeze_bundle(
    repo_root: Path,
    output_dir: Path,
    *,
    code_commit: str,
    evidence: Sequence[EvidenceSpec] = DEFAULT_EVIDENCE,
    enforce_git: bool = True,
) -> JsonObject:
    """Write a canonical zero-GPU plan and passing F0 ledger."""

    repo_root, output_dir = _resolve_bundle_paths(repo_root, output_dir)
    if enforce_git:
        _ensure_clean_before_freeze(repo_root, output_dir)
        tracked = [
            PROTOCOL_PATH,
            *ANCHOR_SOURCE_PATHS,
            *(item.path for item in evidence),
        ]
        _validate_commit(repo_root, code_commit, tracked_paths=tracked)
    if output_dir.exists():
        actual = {path.name for path in output_dir.iterdir()}
        if actual and actual != set(OUTPUT_FILES):
            raise CampaignFreezeError(
                "output directory is non-empty and is not an existing closed bundle"
            )
        if any(path.is_symlink() for path in output_dir.iterdir()):
            raise CampaignFreezeError("campaign output may not contain symlinks")
    output_dir.mkdir(parents=True, exist_ok=True)

    anchors = build_anchor_documents(repo_root)
    for anchor, filename in zip(
        anchors,
        OUTPUT_FILES[:3],
        strict=True,
    ):
        _atomic_write(output_dir / filename, _canonical_bytes(anchor))
    evidence_index = build_evidence_index(
        repo_root,
        code_commit=code_commit,
        anchor_documents=anchors,
        evidence=evidence,
    )
    _atomic_write(
        output_dir / "cpu_evidence_index.json",
        _canonical_bytes(evidence_index),
    )
    plan = CampaignPlan(
        protocol_sha256=_sha256_file(repo_root / PROTOCOL_PATH),
        code_commit=code_commit,
        cpu_artifact_sha256=cast(str, evidence_index["payload_sha256"]),
        validation_trace_hashes=tuple(
            (
                cast(str, anchor["anchor_id"]),
                cast(str, anchor["payload_sha256"]),
            )
            for anchor in anchors
        ),
        planned_primary_replays=1_200,
        planned_unique_ablation_replays=300,
        planned_robustness_cells=12,
    )
    _atomic_write(
        output_dir / "campaign_plan.json",
        _canonical_bytes(plan.document()),
    )
    ledger = CampaignLedger(plan).record_gate(
        GateRecord(
            stage=CampaignStage.CPU_RELEASE,
            verdict=GateVerdict.PASS,
            used_gpu_seconds=0,
            completed_replays=0,
            evidence_sha256=cast(str, evidence_index["payload_sha256"]),
        )
    )
    if ledger.currently_authorized_gpu_seconds != 0:
        raise CampaignFreezeError("F0 freeze must authorize exactly zero GPU seconds")
    _atomic_write(
        output_dir / "campaign_ledger_f0.json",
        _canonical_bytes(ledger.document()),
    )
    manifest = _bundle_manifest(output_dir, code_commit=code_commit)
    _atomic_write(
        output_dir / "bundle_manifest.json",
        _canonical_bytes(manifest),
    )
    verify_bundle(
        repo_root,
        output_dir,
        evidence=evidence,
        enforce_git=enforce_git,
    )
    return manifest


def verify_bundle(
    repo_root: Path,
    output_dir: Path,
    *,
    evidence: Sequence[EvidenceSpec] = DEFAULT_EVIDENCE,
    enforce_git: bool = True,
) -> JsonObject:
    """Verify every source, hash, derived field, and zero-spend invariant."""

    repo_root, output_dir = _resolve_bundle_paths(repo_root, output_dir)
    if not output_dir.is_dir():
        raise CampaignFreezeError("campaign bundle must be a regular directory")
    actual_names = {path.name for path in output_dir.iterdir()}
    if actual_names != set(OUTPUT_FILES):
        raise CampaignFreezeError("campaign bundle file set is not closed")
    if any(path.is_symlink() or not path.is_file() for path in output_dir.iterdir()):
        raise CampaignFreezeError("campaign bundle entries must be regular files")

    manifest = _load_json(output_dir / "bundle_manifest.json")
    _strict_keys(
        manifest,
        {
            "schema",
            "measurement_warning",
            "claim_boundary",
            "source_commit",
            "files",
            "payload_sha256",
        },
        field="bundle manifest",
    )
    _verify_self_hash(manifest, field="bundle manifest")
    if (
        manifest["schema"] != "fissionspec.zero-gpu-campaign-bundle.v1"
        or manifest["measurement_warning"] != WARNING
        or manifest["claim_boundary"] != CLAIM_BOUNDARY
    ):
        raise CampaignFreezeError("bundle manifest evidence boundary drifted")
    code_commit = cast(str, manifest["source_commit"])
    files = manifest["files"]
    if not isinstance(files, list):
        raise CampaignFreezeError("bundle manifest files must be an array")
    expected_file_names = list(OUTPUT_FILES[:-1])
    if [
        item.get("path") if isinstance(item, dict) else None for item in files
    ] != expected_file_names:
        raise CampaignFreezeError("bundle manifest file order or set drifted")
    for item in files:
        if not isinstance(item, dict):
            raise CampaignFreezeError("bundle manifest file record is malformed")
        _strict_keys(item, {"path", "bytes", "sha256"}, field="bundle file record")
        path = output_dir / cast(str, item["path"])
        expected_bytes = _positive_integer(
            item["bytes"],
            field=f"bundle file {path.name}.bytes",
        )
        expected_sha256 = _digest(
            item["sha256"],
            field=f"bundle file {path.name}.sha256",
        )
        if path.stat().st_size != expected_bytes or _sha256_file(path) != expected_sha256:
            raise CampaignFreezeError(f"bundle file hash mismatch: {path.name}")

    anchors = tuple(_load_json(output_dir / filename) for filename in OUTPUT_FILES[:3])
    expected_anchors = build_anchor_documents(repo_root)
    if _canonical_bytes(anchors) != _canonical_bytes(expected_anchors):
        raise CampaignFreezeError("validation anchor templates do not reproduce")
    evidence_index = _load_json(output_dir / "cpu_evidence_index.json")
    expected_index = build_evidence_index(
        repo_root,
        code_commit=code_commit,
        anchor_documents=anchors,
        evidence=evidence,
    )
    if _canonical_bytes(evidence_index) != _canonical_bytes(expected_index):
        raise CampaignFreezeError("CPU evidence index does not reproduce")
    _verify_self_hash(evidence_index, field="CPU evidence index")

    plan_document = _load_json(output_dir / "campaign_plan.json")
    try:
        plan = CampaignPlan.from_document(plan_document)
    except ValueError as error:
        raise CampaignFreezeError("campaign plan verification failed") from error
    if plan.code_commit != code_commit:
        raise CampaignFreezeError("campaign plan source commit mismatch")
    if plan.protocol_sha256 != _sha256_file(repo_root / PROTOCOL_PATH):
        raise CampaignFreezeError("campaign plan protocol hash mismatch")
    if plan.cpu_artifact_sha256 != evidence_index["payload_sha256"]:
        raise CampaignFreezeError("campaign plan CPU evidence hash mismatch")
    if plan.validation_trace_hashes != tuple(
        (
            cast(str, anchor["anchor_id"]),
            cast(str, anchor["payload_sha256"]),
        )
        for anchor in anchors
    ):
        raise CampaignFreezeError("campaign plan validation anchor hashes mismatch")

    ledger_document = _load_json(output_dir / "campaign_ledger_f0.json")
    try:
        ledger = CampaignLedger.from_document(ledger_document)
    except ValueError as error:
        raise CampaignFreezeError("F0 ledger verification failed") from error
    if ledger.plan != plan:
        raise CampaignFreezeError("F0 ledger embeds a different campaign plan")
    if (
        len(ledger.records) != 1
        or ledger.records[0].stage is not CampaignStage.CPU_RELEASE
        or ledger.records[0].verdict is not GateVerdict.PASS
        or ledger.records[0].used_gpu_seconds != 0
        or ledger.records[0].completed_replays != 0
        or ledger.records[0].evidence_sha256 != evidence_index["payload_sha256"]
        or ledger.spent_gpu_seconds != 0
        or ledger.currently_authorized_gpu_seconds != 0
        or ledger.next_stage is not CampaignStage.ROW_OMISSION
    ):
        raise CampaignFreezeError("F0 ledger violates the zero-spend release invariant")

    if enforce_git:
        tracked = [
            PROTOCOL_PATH,
            *ANCHOR_SOURCE_PATHS,
            *(item.path for item in evidence),
        ]
        _validate_commit(repo_root, code_commit, tracked_paths=tracked)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    freeze.add_argument(
        "--code-commit",
        help="full source commit; defaults to the current HEAD",
    )
    verify = subparsers.add_parser("verify")
    verify.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    output_dir = args.output_dir if args.output_dir.is_absolute() else repo_root / args.output_dir
    try:
        if args.command == "freeze":
            code_commit = args.code_commit or _run_git(repo_root, ["rev-parse", "HEAD"])
            manifest = freeze_bundle(
                repo_root,
                output_dir,
                code_commit=code_commit,
            )
        else:
            manifest = verify_bundle(repo_root, output_dir)
    except (CampaignFreezeError, OSError, TypeError, ValueError) as error:
        print(f"GPU campaign freeze failed: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": "verified",
                "source_commit": manifest["source_commit"],
                "bundle_payload_sha256": manifest["payload_sha256"],
                "currently_authorized_gpu_seconds": 0,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
