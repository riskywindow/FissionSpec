"""Command-line entry point for reproducible FissionSpec policy sweeps."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
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


def _add_simulation_arguments(
    parser: argparse.ArgumentParser, *, include_policy: bool
) -> None:
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


def load_workload_json(path: Path) -> Workload:
    """Load a workload from the documented dependency-free JSON format."""

    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("requests"), list):
        raise ValueError("workload JSON must contain a requests array")
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
                values[field] = tuple(float(value) for value in probability)
        requests.append(RequestConfig(**values))
    return Workload(tuple(requests), name=str(raw.get("name", path.stem)))


def _curve(raw: object, field: str) -> LatencyCurve:
    if not isinstance(raw, list):
        raise ValueError(f"profile {field} must be an array of [batch, latency] pairs")
    points: list[tuple[int, float]] = []
    for point in raw:
        if not isinstance(point, list) or len(point) != 2:
            raise ValueError(f"invalid point in profile {field}")
        points.append((int(point[0]), float(point[1])))
    return LatencyCurve(tuple(points))


def load_profile_json(path: Path) -> HardwareProfile:
    """Load measured target/draft curves from a JSON profile."""

    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("profile JSON must be an object")
    return HardwareProfile(
        target_curve=_curve(raw["target_curve"], "target_curve"),
        draft_curve=_curve(raw["draft_curve"], "draft_curve"),
        recovery_curve=_curve(raw["recovery_curve"], "recovery_curve"),
        verifier_slot_ms=float(raw.get("verifier_slot_ms", 0.0)),
        name=str(raw.get("name", path.stem)),
    )


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
        "expected_collateral_hit_stalls": expected_collateral_hit_stalls(
            probabilities
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Run a theory query or simulation and emit machine-readable JSON."""

    arguments = list(sys.argv[1:] if argv is None else argv)
    # Preserve the original flat simulation interface for existing experiment
    # scripts while making the explicit subcommands the documented surface.
    if (
        arguments
        and arguments[0].startswith("-")
        and arguments[0] not in {"-h", "--help"}
    ):
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

    workload = (
        load_workload_json(args.workload_json)
        if args.workload_json is not None
        else Workload.homogeneous(
            args.requests,
            arrival_interval_ms=args.arrival_interval_ms,
            output_tokens=args.output_tokens,
            speculation_length=args.speculation_length,
            cache_hit_probability=args.cache_hit_probability,
            token_acceptance_probability=args.token_acceptance_probability,
            tbt_slo_ms=args.tbt_slo_ms,
        )
    )
    profile = (
        load_profile_json(args.profile_json)
        if args.profile_json is not None
        else HardwareProfile()
    )
    compare_all = args.command == "sweep" or args.compare_all
    policy_names = _POLICIES if compare_all else (args.policy,)
    rng = cast(ScheduleIndependentRNG, CounterRNG(args.seed))
    output: dict[str, object] = {}
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
        output[policy.name] = summarize(result).as_dict()
    print(json.dumps(output, indent=args.indent, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
