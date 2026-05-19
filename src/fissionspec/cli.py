"""Command-line entry point for reproducible FissionSpec policy sweeps."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import cast

from .metrics import (
    batch_fallback_probability,
    expected_collateral_hit_stalls,
    head_of_line_amplification,
    summarize,
)
from .policies import policy_from_name
from .profiles import HardwareProfile, LatencyCurve
from .rng import CounterRNG
from .simulator import ScheduleIndependentRNG, simulate
from .workload import RequestConfig, Workload

_POLICIES = (
    "saguaro-barrier",
    "spectre-parallel-padded",
    "immediate-fission",
    "fixed-coalesce",
    "fissionspec-horizon-2",
)
_POLICY_NAMES = (
    *_POLICIES,
    "saguaro",
    "spectre",
    "spectre-padded",
    "immediate",
    "fission",
    "fixed",
    "fissionspec",
    "horizon-2",
)

_SIMULATION_WARNING = "SIMULATION MODEL OUTPUT — NOT AN END-TO-END GPU MEASUREMENT."


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON numeric constant is not allowed: {value}")


def _add_simulation_arguments(parser: argparse.ArgumentParser, *, include_policy: bool) -> None:
    if include_policy:
        parser.add_argument("--policy", choices=_POLICY_NAMES, default="fissionspec")
        parser.add_argument(
            "--compare-all",
            action="store_true",
            help="run every built-in policy with the same counter-addressed seed",
        )
    parser.add_argument("--workload-json", type=Path)
    parser.add_argument("--profile-json", type=Path)
    parser.add_argument("--requests", type=int, default=32)
    parser.add_argument("--arrival-interval-ms", type=float, default=0.15)
    parser.add_argument("--output-tokens", type=int, default=64)
    parser.add_argument("--speculation-length", type=int, default=4)
    parser.add_argument("--cache-hit-probability", type=float, default=0.8)
    parser.add_argument("--token-acceptance-probability", type=float, default=0.8)
    parser.add_argument("--tbt-slo-ms", type=float, default=50.0)
    parser.add_argument("--max-batch-size", type=int, default=16)
    parser.add_argument("--coalesce-ms", type=float, default=1.0)
    parser.add_argument("--max-wait-ms", type=float, default=2.0)
    parser.add_argument("--seed", default="fissionspec")
    parser.add_argument("--indent", type=int, default=2)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fissionspec",
        description="Deterministic speculative-decoding scheduler simulator",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    theory = subparsers.add_parser(
        "theory", help="evaluate closed-form batch fallback and HOL amplification"
    )
    theory.add_argument("--batch-size", type=int, required=True)
    theory.add_argument("--hit-rate", type=float, required=True)
    theory.add_argument("--indent", type=int, default=2)

    simulation = subparsers.add_parser(
        "simulate", help="run one policy or a matched-seed comparison"
    )
    _add_simulation_arguments(simulation, include_policy=True)

    sweep = subparsers.add_parser(
        "sweep", help="run all policies with a shared workload and random keys"
    )
    _add_simulation_arguments(sweep, include_policy=False)
    return parser


def _load_workload_document(path: Path) -> tuple[Workload, dict[str, object], str]:
    serialized = path.read_bytes()
    raw = json.loads(
        serialized,
        parse_constant=_reject_json_constant,
    )
    if not isinstance(raw, dict) or not isinstance(raw.get("requests"), list):
        raise ValueError("workload JSON must contain a requests array")
    schema_version = raw.get("schema_version")
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != 1
    ):
        raise ValueError("workload schema_version must be integer 1")
    requests = []
    for item in raw["requests"]:
        if not isinstance(item, dict):
            raise ValueError("each request must be a JSON object")
        values = dict(item)
        for field in (
            "cache_hit_probability",
            "token_acceptance_probability",
        ):
            probability = values.get(field)
            if isinstance(probability, list):
                if any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    for value in probability
                ):
                    raise ValueError(f"{field} arrays must contain only finite real numbers")
                values[field] = tuple(float(value) for value in probability)
        requests.append(RequestConfig(**values))
    name = raw.get("name", path.stem)
    if not isinstance(name, str) or not name:
        raise ValueError("workload name must be a non-empty string")
    workload = Workload(tuple(requests), name=name)
    return workload, raw, hashlib.sha256(serialized).hexdigest()


def load_workload_json(path: Path) -> Workload:
    """Load a versioned workload document with strict numeric types."""

    workload, _, _ = _load_workload_document(path)
    return workload


def _curve(raw: object, field: str) -> LatencyCurve:
    if not isinstance(raw, list):
        raise ValueError(f"profile {field} must be an array of [batch, latency] pairs")
    points: list[tuple[int, float]] = []
    for point in raw:
        if not isinstance(point, list) or len(point) != 2:
            raise ValueError(f"invalid point in profile {field}")
        row, latency = point
        if isinstance(row, bool) or not isinstance(row, int):
            raise ValueError(f"profile {field} row counts must be integers")
        if isinstance(latency, bool) or not isinstance(latency, (int, float)):
            raise ValueError(f"profile {field} latencies must be real numbers")
        points.append((row, float(latency)))
    return LatencyCurve(tuple(points))


def _load_profile_document(path: Path) -> tuple[HardwareProfile, dict[str, object], str]:
    serialized = path.read_bytes()
    raw = json.loads(serialized, parse_constant=_reject_json_constant)
    if not isinstance(raw, dict):
        raise ValueError("profile JSON must be an object")
    schema_version = raw.get("schema_version")
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != 1
    ):
        raise ValueError("profile schema_version must be integer 1")
    if not isinstance(raw.get("provenance"), dict):
        raise ValueError("profile provenance must be an object")
    fit = raw.get("fit")
    if fit is not None and not isinstance(fit, dict):
        raise ValueError("profile fit diagnostics must be an object when supplied")
    name = raw.get("name", path.stem)
    if not isinstance(name, str) or not name.strip():
        raise ValueError("profile name must be a non-empty string")
    verifier_slot_ms = raw.get("verifier_slot_ms", 0.0)
    if isinstance(verifier_slot_ms, bool) or not isinstance(verifier_slot_ms, (int, float)):
        raise ValueError("profile verifier_slot_ms must be a real number")
    for curve_name in ("target_curve", "draft_curve", "recovery_curve"):
        if curve_name not in raw:
            raise ValueError(f"profile is missing {curve_name}")
    profile = HardwareProfile(
        target_curve=_curve(raw["target_curve"], "target_curve"),
        draft_curve=_curve(raw["draft_curve"], "draft_curve"),
        recovery_curve=_curve(raw["recovery_curve"], "recovery_curve"),
        verifier_slot_ms=float(verifier_slot_ms),
        name=name.strip(),
    )
    return profile, raw, hashlib.sha256(serialized).hexdigest()


def load_profile_json(path: Path) -> HardwareProfile:
    """Load a versioned, provenance-bearing latency profile document."""

    profile, _, _ = _load_profile_document(path)
    return profile


def _theory_output(batch_size: int, hit_rate: float) -> dict[str, float | int]:
    if batch_size <= 0:
        raise ValueError("batch-size must be positive")
    probabilities = (hit_rate,) * batch_size
    fallback = batch_fallback_probability(probabilities)
    return {
        "batch_size": batch_size,
        "hit_rate": hit_rate,
        "batch_fallback_probability": fallback,
        "expected_barrier_stalled_rows": batch_size * fallback,
        "expected_fission_stalled_rows": batch_size * (1.0 - hit_rate),
        "head_of_line_amplification": head_of_line_amplification(probabilities),
        "expected_collateral_hit_stalls": expected_collateral_hit_stalls(probabilities),
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Run a theory query or simulation and emit machine-readable JSON."""

    arguments = list(sys.argv[1:] if argv is None else argv)
    # Preserve the original flat simulation interface for existing experiment
    # scripts while making the explicit subcommands the documented surface.
    if arguments and arguments[0].startswith("-") and arguments[0] not in {"-h", "--help"}:
        arguments.insert(0, "simulate")
    args = _parser().parse_args(arguments)
    if args.command == "theory":
        print(
            json.dumps(
                _theory_output(args.batch_size, args.hit_rate),
                indent=args.indent,
                sort_keys=True,
            )
        )
        return 0

    workload_document: dict[str, object] | None = None
    workload_sha256: str | None = None
    if args.workload_json is not None:
        workload, workload_document, workload_sha256 = _load_workload_document(args.workload_json)
    else:
        workload = Workload.homogeneous(
            args.requests,
            arrival_interval_ms=args.arrival_interval_ms,
            output_tokens=args.output_tokens,
            speculation_length=args.speculation_length,
            cache_hit_probability=args.cache_hit_probability,
            token_acceptance_probability=args.token_acceptance_probability,
            tbt_slo_ms=args.tbt_slo_ms,
        )
    profile_document: dict[str, object] | None = None
    profile_sha256: str | None = None
    if args.profile_json is not None:
        profile, profile_document, profile_sha256 = _load_profile_document(args.profile_json)
    else:
        profile = HardwareProfile()
    compare_all = args.command == "sweep" or args.compare_all
    policy_names = _POLICIES if compare_all else (args.policy,)
    rng = cast(ScheduleIndependentRNG, CounterRNG(args.seed))
    policy_results: dict[str, object] = {}
    for policy_name in policy_names:
        policy = policy_from_name(
            policy_name,
            coalesce_ms=args.coalesce_ms,
            max_wait_ms=args.max_wait_ms,
        )
        result = simulate(
            workload,
            profile,
            policy,
            rng,
            max_batch_size=args.max_batch_size,
        )
        policy_results[policy.name] = summarize(result).as_dict()
    output = {
        "schema_version": 1,
        "evidence_class": "simulation-model",
        "measurement_warning": _SIMULATION_WARNING,
        "seed": args.seed,
        "profile": {
            "name": profile.name,
            "source": (
                str(args.profile_json) if args.profile_json is not None else "built-in-synthetic"
            ),
            "target_curve": [list(point) for point in profile.target_curve.points],
            "draft_curve": [list(point) for point in profile.draft_curve.points],
            "recovery_curve": [list(point) for point in profile.recovery_curve.points],
            "verifier_slot_ms": profile.verifier_slot_ms,
            "source_document_sha256": profile_sha256,
            "schema_version": (
                profile_document.get("schema_version") if profile_document is not None else None
            ),
            "provenance": (
                profile_document.get("provenance")
                if profile_document is not None
                else {
                    "kind": "synthetic",
                    "warning": _SIMULATION_WARNING,
                }
            ),
            "fit": profile_document.get("fit") if profile_document is not None else None,
        },
        "workload": {
            "name": workload.name,
            "source": (
                str(args.workload_json)
                if args.workload_json is not None
                else "generated-homogeneous"
            ),
            "source_document_sha256": workload_sha256,
            "schema_version": (
                workload_document.get("schema_version") if workload_document is not None else None
            ),
            "description": (
                workload_document.get("description") if workload_document is not None else None
            ),
            "requests": [asdict(request) for request in workload],
        },
        "scheduler": {
            "max_batch_size": args.max_batch_size,
            "coalesce_ms": args.coalesce_ms,
            "max_wait_ms": args.max_wait_ms,
        },
        "results": policy_results,
    }
    print(json.dumps(output, indent=args.indent, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
