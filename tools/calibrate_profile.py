#!/usr/bin/env python3
"""Fit a FissionSpec latency profile from raw CSV timing samples."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from fissionspec.calibration import (
    CalibrationError,
    fit_profile,
    load_samples_csv,
    write_profile_json,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="raw timing CSV")
    parser.add_argument("output", type=Path, help="profile JSON to create")
    parser.add_argument("--name", required=True, help="stable profile name")
    parser.add_argument(
        "--provenance-json",
        type=Path,
        help="JSON object containing engine commit, GPU, models, dtype, and TP",
    )
    parser.add_argument(
        "--require-slot-slope",
        action="store_true",
        help="fail unless target samples vary verifier slots at fixed row count",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace an existing output profile",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.input.resolve() == args.output.resolve():
        raise SystemExit("input and output paths must differ")
    if args.output.exists() and not args.force:
        raise SystemExit(f"output already exists: {args.output} (pass --force to replace)")
    provenance = None
    try:
        if args.provenance_json is not None:
            provenance = json.loads(args.provenance_json.read_text(encoding="utf-8"))
            if not isinstance(provenance, dict):
                raise SystemExit("provenance JSON must be an object")
        profile = fit_profile(load_samples_csv(args.input), name=args.name)
        if args.require_slot_slope and (
            not profile.slot_slope_identified or profile.slot_slope_clipped
        ):
            raise SystemExit("target samples do not identify a non-negative verifier-slot slope")
        write_profile_json(profile, args.output, provenance=provenance)
    except (CalibrationError, json.JSONDecodeError, OSError) as exc:
        raise SystemExit(f"calibration failed: {exc}") from exc
    print(
        f"wrote {args.output}: {profile.sample_count} samples, "
        f"slot={profile.verifier_slot_ms:.6g} ms, "
        f"target RMSE={profile.target_rmse_ms:.6g} ms"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
