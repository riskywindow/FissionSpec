#!/usr/bin/env python3
"""Capture scale-aware CPU controller complexity and local timing evidence.

The deterministic artifact audits row visits, sort-key evaluations, latency
queries, asymptotic complexity, and allocation behavior from the production
Python and Rust controllers.  A separate environment-scoped artifact records
raw repeated wall-clock timings.  The latter is diagnostic only: it is not a
GPU benchmark, not a cross-host result, and not a defensible Python/Rust ratio.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Final, cast

from fissionspec.policies import DispatchContext, FissionSpecPolicy
from fissionspec.profiles import HardwareProfile, LatencyCurve

SCHEMA_VERSION: Final = 1
WARNING: Final = "LOCAL CPU TIMING / NOT GPU EVIDENCE / NOT CROSS-HOST COMPARABLE"
COMPLEXITY_WARNING: Final = "DETERMINISTIC SOURCE AUDIT / NOT A TIMING MEASUREMENT"
SIZES: Final = (1, 2, 4, 8, 16, 32, 64, 96, 128)
CAPACITY: Final = 512
RUST_ITEM_VISITS_CURRENT: Final = 8
RUST_ITEM_VISITS_FUTURE: Final = 7


class ControllerOverheadError(ValueError):
    """Raised when an overhead artifact fails its closed-schema checks."""


def _canonical_json(document: object) -> bytes:
    return json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()


def _sha256_document(document: object) -> str:
    return hashlib.sha256(_canonical_json(document)).hexdigest()


def _write_json(path: Path, document: object) -> None:
    path.write_bytes(
        json.dumps(
            document,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
        + b"\n"
    )


def _profile() -> HardwareProfile:
    return HardwareProfile(
        target_curve=LatencyCurve(
            (
                (1, 0.042),
                (8, 0.061),
                (32, 0.109),
                (64, 0.180),
                (128, 0.315),
                (256, 0.580),
            )
        ),
        verifier_slot_ms=0.0,
        name="controller-overhead-structural-not-gpu",
    )


def _context(size: int) -> DispatchContext:
    profile = _profile()
    return DispatchContext(
        now_ms=0.0,
        ready_count=size,
        capacity=CAPACITY,
        oldest_ready_ms=0.0,
        earliest_deadline_ms=1_000.0,
        row_slots=(1,) * size,
        row_deadlines_ms=(1_000.0,) * size,
        profile=profile,
        next_ready_time_ms=0.008,
        next_ready_count=size,
        earliest_future_deadline_ms=1_000.0,
        future_row_slots=(1,) * size,
        future_row_deadlines_ms=(1_000.0,) * size,
    )


def build_complexity_artifact(repo_root: Path | None = None) -> dict[str, object]:
    """Build deterministic operation counts for the fully evaluated path."""

    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[1]
    samples: list[dict[str, object]] = []
    policy = FissionSpecPolicy(max_wait_ms=2.0)
    for size in SIZES:
        context = _context(size)
        dispatch_at = policy.dispatch_at(context)
        current_rows = size
        future_rows = size
        combined_rows = current_rows + future_rows
        python_latency_queries = (
            1 + math.ceil(future_rows / CAPACITY) + math.ceil(combined_rows / CAPACITY)
        )
        samples.append(
            {
                "current_rows": current_rows,
                "future_rows": future_rows,
                "production_python_decision": (
                    "re-fuse" if dispatch_at == context.next_ready_time_ms else "dispatch-now"
                ),
                "python_source_audit": {
                    "forecast_rows_materialized": combined_rows,
                    "global_sort_key_evaluations": combined_rows,
                    "wait_plan_row_accumulations": combined_rows,
                    "feasibility_row_checks_upper_bound": 2 * combined_rows,
                    "latency_profile_queries": python_latency_queries,
                    "asymptotic_time": "O((n + m) log(n + m))",
                    "auxiliary_space": "O(n + m)",
                    "dominant_reason": (
                        "global rolling-EDF sorted(current + future) at the prospective wake"
                    ),
                },
                "rust_source_audit": {
                    "batch_view_item_visits": (
                        RUST_ITEM_VISITS_CURRENT * current_rows
                        + RUST_ITEM_VISITS_FUTURE * future_rows
                    ),
                    "current_item_visits": RUST_ITEM_VISITS_CURRENT * current_rows,
                    "future_item_visits": RUST_ITEM_VISITS_FUTURE * future_rows,
                    "latency_profile_queries": 3,
                    "latency_profile_knots": 6,
                    "asymptotic_time": "O(n + m + log k)",
                    "controller_auxiliary_space": "O(1)",
                    "explicit_heap_allocations_in_evaluate": 0,
                    "allocation_scope": (
                        "The caller owns input slices and the latency profile. This count "
                        "does not claim that the surrounding serving loop never allocates."
                    ),
                },
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact": "FissionSpec scale-aware controller source audit",
        "measurement_warning": COMPLEXITY_WARNING,
        "evidence_class": "deterministic-operation-count",
        "claim_boundary": (
            "Counts describe the full feasible two-plan path in the checked-in "
            "implementations. They are not elapsed time, CPU cycles, GPU overhead, "
            "or end-to-end serving overhead."
        ),
        "path_preconditions": {
            "current_and_future_rows": "ready and deadline-feasible",
            "fused_work": "within capacity",
            "forecast": "inside the maximum coalescing window",
            "capacity_units": CAPACITY,
            "latency_profile_knots": 6,
        },
        "count_derivation": {
            "python": (
                "Exact materialization/key/accumulation counts plus an upper bound on "
                "short-circuiting feasibility checks, audited from FissionSpecPolicy.dispatch_at."
            ),
            "rust": (
                "Exact BatchView visitor passes on the fully evaluated feasible path: "
                "8 visits per current item and 7 per future item, audited from "
                "Horizon2Controller.evaluate."
            ),
        },
        "implementation_files": _source_hashes(repo_root),
        "samples": samples,
    }


def run_python_timing(
    *,
    target_row_visits: int = 1_000_000,
    repeats: int = 7,
) -> dict[str, object]:
    """Collect raw, repeated local timings around the production Python policy."""

    if target_row_visits < 10_000 or not 1 <= repeats <= 31:
        raise ValueError("timing budget or repeat count is outside the safe range")
    policy = FissionSpecPolicy(max_wait_ms=2.0)
    samples: list[dict[str, object]] = []
    gc_was_enabled = gc.isenabled()
    try:
        gc.disable()
        for size in SIZES:
            context = _context(size)
            iterations = min(
                100_000,
                max(200, target_row_visits // (context.ready_count + context.next_ready_count)),
            )
            digest = 0
            for _ in range(max(100, iterations // 10)):
                digest ^= int(policy.dispatch_at(context) * 1_000_000)
            elapsed_repeats: list[int] = []
            for repeat in range(repeats):
                started = time.perf_counter_ns()
                for _ in range(iterations):
                    dispatch_at = policy.dispatch_at(context)
                    digest ^= int(dispatch_at * 1_000_000) + repeat
                elapsed_repeats.append(time.perf_counter_ns() - started)
            per_decision = [elapsed / iterations for elapsed in elapsed_repeats]
            samples.append(
                {
                    "current_rows": size,
                    "future_rows": size,
                    "iterations_per_repeat": iterations,
                    "elapsed_ns_repeats": elapsed_repeats,
                    "median_ns_per_decision": statistics.median(per_decision),
                    "minimum_ns_per_decision": min(per_decision),
                    "maximum_ns_per_decision": max(per_decision),
                    "digest": digest,
                }
            )
    finally:
        if gc_was_enabled:
            gc.enable()
    return {
        "runtime": "python",
        "timer": "time.perf_counter_ns",
        "target_row_visits_per_repeat": target_row_visits,
        "repeats": repeats,
        "garbage_collector_disabled_inside_timed_region": True,
        "samples": samples,
    }


def run_rust_timing(
    repo_root: Path,
    *,
    target_item_visits: int = 2_000_000,
    repeats: int = 7,
) -> dict[str, object]:
    """Build and execute the production Rust controller's scale benchmark."""

    command = [
        "cargo",
        "run",
        "--release",
        "--quiet",
        "--manifest-path",
        "crates/fissionspec-core/Cargo.toml",
        "--bin",
        "controller_scale_bench",
        "--",
        str(target_item_visits),
        str(repeats),
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=180,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise ControllerOverheadError(f"Rust scale benchmark failed: {exc}") from exc
    try:
        document = cast(dict[str, object], json.loads(completed.stdout))
    except json.JSONDecodeError as exc:
        raise ControllerOverheadError("Rust scale benchmark returned invalid JSON") from exc
    if document.get("schema_version") != SCHEMA_VERSION:
        raise ControllerOverheadError("Rust timing schema mismatch")
    samples = document.get("samples")
    if not isinstance(samples, list) or len(samples) != len(SIZES):
        raise ControllerOverheadError("Rust timing scale grid is incomplete")
    return {
        "runtime": "rust",
        "command": command,
        "cargo_stderr": completed.stderr,
        **document,
    }


def _tool_version(command: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    return completed.stdout.strip() or completed.stderr.strip() or None


def _source_hashes(repo_root: Path) -> dict[str, str]:
    relative_paths = (
        "experiments/run_controller_overhead.py",
        "src/fissionspec/policies.py",
        "crates/fissionspec-core/src/controller.rs",
        "crates/fissionspec-core/src/latency.rs",
        "crates/fissionspec-core/src/bin/controller_scale_bench.rs",
    )
    return {
        relative: hashlib.sha256((repo_root / relative).read_bytes()).hexdigest()
        for relative in relative_paths
    }


def environment_document(repo_root: Path) -> dict[str, object]:
    """Record uncontrolled variables rather than hiding timing limitations."""

    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "rustc_version": _tool_version(["rustc", "--version"]),
        "cargo_version": _tool_version(["cargo", "--version"]),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "source_hashes": _source_hashes(repo_root),
        "timing_controls": {
            "process_affinity_pinned": False,
            "cpu_frequency_fixed": False,
            "background_load_controlled": False,
            "rust_build_profile": "release",
            "python_gc_disabled_in_timed_region": True,
            "raw_repeats_retained": True,
        },
    }


def build_timing_artifact(
    repo_root: Path,
    *,
    target_row_visits: int,
    repeats: int,
    include_rust: bool,
) -> dict[str, object]:
    """Collect exploratory local timings with complete limitation labels."""

    runtimes = [
        run_python_timing(
            target_row_visits=target_row_visits,
            repeats=repeats,
        )
    ]
    if include_rust:
        runtimes.append(
            run_rust_timing(
                repo_root,
                target_item_visits=target_row_visits * 2,
                repeats=repeats,
            )
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact": "FissionSpec local controller timing snapshot",
        "measurement_warning": WARNING,
        "evidence_class": "exploratory-local-cpu-timing",
        "claim_boundary": (
            "Raw wall-clock samples characterize this host and build only. Python and "
            "Rust harness loops are not identical, so no cross-language ratio is "
            "reported. The samples do not predict GPU or end-to-end serving overhead."
        ),
        "environment": environment_document(repo_root),
        "runtimes": runtimes,
    }


def write_artifacts(
    output_dir: Path,
    *,
    repo_root: Path,
    target_row_visits: int = 1_000_000,
    repeats: int = 7,
    include_rust: bool = True,
    structural_only: bool = False,
) -> dict[str, object]:
    """Write complexity and optional local-timing artifacts plus a hash manifest."""

    output_dir.mkdir(parents=True, exist_ok=True)
    complexity = build_complexity_artifact(repo_root)
    _write_json(output_dir / "controller_complexity.json", complexity)
    filenames = ["controller_complexity.json"]
    if not structural_only:
        timing = build_timing_artifact(
            repo_root,
            target_row_visits=target_row_visits,
            repeats=repeats,
            include_rust=include_rust,
        )
        _write_json(output_dir / "controller_overhead_local.json", timing)
        filenames.append("controller_overhead_local.json")
    files = {
        filename: {
            "bytes": (output_dir / filename).stat().st_size,
            "sha256": hashlib.sha256((output_dir / filename).read_bytes()).hexdigest(),
        }
        for filename in filenames
    }
    manifest_payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "measurement_warning": WARNING,
        "artifact_files": files,
        "complexity_document_sha256": _sha256_document(complexity),
    }
    manifest: dict[str, object] = {
        **manifest_payload,
        "payload_sha256": _sha256_document(manifest_payload),
    }
    _write_json(output_dir / "controller_overhead_manifest.json", manifest)
    verify_artifacts(output_dir)
    return manifest


def verify_artifacts(output_dir: Path) -> dict[str, object]:
    """Verify hashes and recompute the deterministic source-audit document."""

    try:
        manifest = cast(
            dict[str, object],
            json.loads(
                (output_dir / "controller_overhead_manifest.json").read_text(encoding="utf-8")
            ),
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ControllerOverheadError("cannot read overhead manifest") from exc
    if set(manifest) != {
        "schema_version",
        "measurement_warning",
        "artifact_files",
        "complexity_document_sha256",
        "payload_sha256",
    }:
        raise ControllerOverheadError("overhead manifest has unexpected or missing fields")
    supplied_payload_hash = manifest.get("payload_sha256")
    if (
        not isinstance(supplied_payload_hash, str)
        or len(supplied_payload_hash) != 64
        or any(character not in "0123456789abcdef" for character in supplied_payload_hash)
    ):
        raise ControllerOverheadError("overhead manifest payload SHA-256 is malformed")
    manifest_payload = dict(manifest)
    manifest_payload.pop("payload_sha256")
    if _sha256_document(manifest_payload) != supplied_payload_hash:
        raise ControllerOverheadError("overhead manifest payload hash mismatch")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ControllerOverheadError("overhead manifest schema mismatch")
    if manifest.get("measurement_warning") != WARNING:
        raise ControllerOverheadError("overhead manifest lacks its warning")
    files = manifest.get("artifact_files")
    if not isinstance(files, dict):
        raise ControllerOverheadError("artifact_files must be an object")
    if set(files) not in (
        {"controller_complexity.json"},
        {"controller_complexity.json", "controller_overhead_local.json"},
    ):
        raise ControllerOverheadError("overhead artifact file set is not supported")
    expected_entries = set(files) | {"controller_overhead_manifest.json"}
    actual_entries = {path.name for path in output_dir.iterdir()}
    if actual_entries != expected_entries:
        raise ControllerOverheadError(
            "overhead entries do not match the closed manifest: "
            f"unexpected={sorted(actual_entries - expected_entries)}, "
            f"missing={sorted(expected_entries - actual_entries)}"
        )
    for filename, untyped_record in files.items():
        if not isinstance(filename, str) or not isinstance(untyped_record, dict):
            raise ControllerOverheadError("malformed overhead file record")
        if set(untyped_record) != {"bytes", "sha256"}:
            raise ControllerOverheadError("overhead file record has unexpected fields")
        payload = (output_dir / filename).read_bytes()
        if hashlib.sha256(payload).hexdigest() != untyped_record.get("sha256"):
            raise ControllerOverheadError(f"overhead artifact hash mismatch: {filename}")
        if len(payload) != untyped_record.get("bytes"):
            raise ControllerOverheadError(f"overhead artifact size mismatch: {filename}")
    complexity = json.loads((output_dir / "controller_complexity.json").read_text(encoding="utf-8"))
    if _canonical_json(complexity) != _canonical_json(build_complexity_artifact()):
        raise ControllerOverheadError("deterministic complexity artifact does not reproduce")
    if _sha256_document(complexity) != manifest.get("complexity_document_sha256"):
        raise ControllerOverheadError("complexity document hash mismatch")
    timing_path = output_dir / "controller_overhead_local.json"
    if timing_path.exists():
        timing = json.loads(timing_path.read_text(encoding="utf-8"))
        if timing.get("measurement_warning") != WARNING:
            raise ControllerOverheadError("local timing artifact lacks its warning")
        runtimes = timing.get("runtimes")
        if not isinstance(runtimes, list) or not runtimes:
            raise ControllerOverheadError("local timing artifact has no runtime samples")
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/results/controller_overhead"),
    )
    parser.add_argument("--target-row-visits", type=int, default=1_000_000)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--skip-rust", action="store_true")
    parser.add_argument("--structural-only", action="store_true")
    parser.add_argument("--verify", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.verify:
            manifest = verify_artifacts(args.output_dir)
        else:
            manifest = write_artifacts(
                args.output_dir,
                repo_root=Path(__file__).resolve().parents[1],
                target_row_visits=args.target_row_visits,
                repeats=args.repeats,
                include_rust=not args.skip_rust,
                structural_only=args.structural_only,
            )
    except (ControllerOverheadError, OSError, ValueError) as exc:
        print(f"controller overhead failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
