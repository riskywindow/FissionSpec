#!/usr/bin/env python3
"""Render the deterministic FissionSpec horizon-2 policy decision surface.

This artifact calls the production ``FissionSpecPolicy`` and
``ImmediateFissionPolicy`` implementations.  It is a synthetic controller
mechanism study, not a simulator run or a hardware benchmark.
"""

from __future__ import annotations

import argparse
import html
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from fissionspec.policies import (
    DispatchContext,
    FissionSpecPolicy,
    ImmediateFissionPolicy,
)
from fissionspec.profiles import HardwareProfile, LatencyCurve

WARNING: Final = "SYNTHETIC MODEL / NOT GPU"
CLAIM_BOUNDARY: Final = (
    "Deterministic policy decision surface only; it is not throughput, latency, "
    "or GPU-performance evidence."
)
SCHEMA_VERSION: Final = 1


@dataclass(frozen=True, slots=True)
class PhaseConfig:
    """Frozen axes and constraints for the checked-in phase diagram."""

    current_cohort_sizes: tuple[int, ...] = (1, 2, 4, 8)
    future_cohort_sizes: tuple[int, ...] = (1, 2, 4, 8)
    recovery_eta_ms: tuple[float, ...] = (0.05, 0.10, 0.25, 0.50, 1.00, 1.50, 2.01)
    capacity: int = 16
    verifier_slots_per_row: int = 5
    max_wait_ms: float = 2.0
    current_deadline_ms: float = 100.0
    future_deadline_ms: float = 100.0


def synthetic_profile() -> HardwareProfile:
    """Return the exact immutable latency surface used by this artifact."""

    return HardwareProfile(
        target_curve=LatencyCurve(((1, 2.1), (4, 2.8), (8, 3.8), (16, 5.9), (32, 10.5))),
        draft_curve=LatencyCurve(((1, 0.55), (4, 0.78), (8, 1.05), (16, 1.65), (32, 2.85))),
        recovery_curve=LatencyCurve(((1, 1.25), (4, 1.7), (8, 2.3), (16, 3.5), (32, 5.9))),
        verifier_slot_ms=0.018,
        name="synthetic-controller-phase-not-gpu",
    )


def _curve_points(curve: LatencyCurve) -> list[list[float | int]]:
    return [[rows, latency_ms] for rows, latency_ms in curve.points]


def profile_document(profile: HardwareProfile) -> dict[str, object]:
    """Serialize every profile knot, including unused draft-side curves."""

    return {
        "name": profile.name,
        "gpu_measurement": False,
        "target_curve_rows_to_ms": _curve_points(profile.target_curve),
        "draft_curve_rows_to_ms": _curve_points(profile.draft_curve),
        "recovery_curve_rows_to_ms": _curve_points(profile.recovery_curve),
        "verifier_slot_ms": profile.verifier_slot_ms,
    }


def _context(
    config: PhaseConfig,
    profile: HardwareProfile,
    *,
    now_ms: float,
    oldest_ready_ms: float,
    current_rows: int,
    future_rows: int,
    recovery_eta_from_now_ms: float,
    current_deadline_ms: float,
    future_deadline_ms: float,
) -> DispatchContext:
    return DispatchContext(
        now_ms=now_ms,
        ready_count=current_rows,
        capacity=config.capacity,
        oldest_ready_ms=oldest_ready_ms,
        earliest_deadline_ms=current_deadline_ms,
        row_slots=(config.verifier_slots_per_row,) * current_rows,
        row_deadlines_ms=(current_deadline_ms,) * current_rows,
        profile=profile,
        next_ready_time_ms=now_ms + recovery_eta_from_now_ms,
        next_ready_count=future_rows,
        earliest_future_deadline_ms=future_deadline_ms,
        future_row_slots=(config.verifier_slots_per_row,) * future_rows,
        future_row_deadlines_ms=(future_deadline_ms,) * future_rows,
    )


def _evaluate(
    config: PhaseConfig,
    profile: HardwareProfile,
    policy: FissionSpecPolicy,
    *,
    now_ms: float = 0.0,
    oldest_ready_ms: float = 0.0,
    current_rows: int,
    future_rows: int,
    recovery_eta_from_now_ms: float,
    current_deadline_ms: float | None = None,
    future_deadline_ms: float | None = None,
) -> dict[str, object]:
    context = _context(
        config,
        profile,
        now_ms=now_ms,
        oldest_ready_ms=oldest_ready_ms,
        current_rows=current_rows,
        future_rows=future_rows,
        recovery_eta_from_now_ms=recovery_eta_from_now_ms,
        current_deadline_ms=(
            config.current_deadline_ms if current_deadline_ms is None else current_deadline_ms
        ),
        future_deadline_ms=(
            config.future_deadline_ms if future_deadline_ms is None else future_deadline_ms
        ),
    )
    immediate_at_ms = ImmediateFissionPolicy().dispatch_at(context)
    fissionspec_at_ms = policy.dispatch_at(context)
    next_ready_ms = now_ms + recovery_eta_from_now_ms
    if math.isclose(fissionspec_at_ms, now_ms, rel_tol=0.0, abs_tol=1e-12):
        decision = "dispatch-now"
    elif math.isclose(fissionspec_at_ms, next_ready_ms, rel_tol=0.0, abs_tol=1e-12):
        decision = "re-fuse"
    else:
        raise RuntimeError(
            "phase artifact expects the horizon-2 controller to choose now or the next ETA"
        )
    if not math.isclose(immediate_at_ms, now_ms, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError("ImmediateFissionPolicy did not dispatch immediately")
    return {
        "decision": decision,
        "fissionspec_dispatch_at_ms": fissionspec_at_ms,
        "immediate_fission_dispatch_at_ms": immediate_at_ms,
    }


def _boundary_checks(
    config: PhaseConfig,
    profile: HardwareProfile,
    policy: FissionSpecPolicy,
) -> list[dict[str, object]]:
    checks: tuple[tuple[str, str, dict[str, float | int]], ...] = (
        (
            "near-eta-overhead-dominated",
            "Near recovery can re-fuse small cohorts under the modeled launch surface.",
            {
                "now_ms": 0.0,
                "oldest_ready_ms": 0.0,
                "current_rows": 1,
                "future_rows": 1,
                "recovery_eta_from_now_ms": 0.05,
                "current_deadline_ms": 100.0,
                "future_deadline_ms": 100.0,
            },
        ),
        (
            "late-eta-flow-cost",
            "A later recovery ETA can make immediate dispatch preferable.",
            {
                "now_ms": 0.0,
                "oldest_ready_ms": 0.0,
                "current_rows": 1,
                "future_rows": 1,
                "recovery_eta_from_now_ms": 1.5,
                "current_deadline_ms": 100.0,
                "future_deadline_ms": 100.0,
            },
        ),
        (
            "eta-past-max-wait",
            "A recovery beyond the oldest row's max-wait horizon cannot hold it.",
            {
                "now_ms": 0.0,
                "oldest_ready_ms": 0.0,
                "current_rows": 1,
                "future_rows": 8,
                "recovery_eta_from_now_ms": 2.01,
                "current_deadline_ms": 100.0,
                "future_deadline_ms": 100.0,
            },
        ),
        (
            "oldest-row-already-at-max-wait",
            "A fresh near ETA cannot reset the cumulative wait of the oldest row.",
            {
                "now_ms": 2.0,
                "oldest_ready_ms": 0.0,
                "current_rows": 1,
                "future_rows": 8,
                "recovery_eta_from_now_ms": 0.05,
                "current_deadline_ms": 100.0,
                "future_deadline_ms": 100.0,
            },
        ),
        (
            "current-deadline-forces-dispatch",
            "A tight current-cohort deadline vetoes otherwise attractive re-fusion.",
            {
                "now_ms": 0.0,
                "oldest_ready_ms": 0.0,
                "current_rows": 1,
                "future_rows": 8,
                "recovery_eta_from_now_ms": 0.05,
                "current_deadline_ms": 2.5,
                "future_deadline_ms": 100.0,
            },
        ),
        (
            "future-deadline-favors-refusion",
            "A future-cohort deadline can favor one fused launch over launch-now blocking.",
            {
                "now_ms": 0.0,
                "oldest_ready_ms": 0.0,
                "current_rows": 1,
                "future_rows": 8,
                "recovery_eta_from_now_ms": 0.05,
                "current_deadline_ms": 100.0,
                "future_deadline_ms": 5.0,
            },
        ),
        (
            "capacity-forces-dispatch",
            "A full current target batch dispatches without waiting.",
            {
                "now_ms": 0.0,
                "oldest_ready_ms": 0.0,
                "current_rows": 16,
                "future_rows": 1,
                "recovery_eta_from_now_ms": 0.05,
                "current_deadline_ms": 100.0,
                "future_deadline_ms": 100.0,
            },
        ),
    )
    output: list[dict[str, object]] = []
    for check_id, reason, values in checks:
        evaluation = _evaluate(
            config,
            profile,
            policy,
            now_ms=float(values["now_ms"]),
            oldest_ready_ms=float(values["oldest_ready_ms"]),
            current_rows=int(values["current_rows"]),
            future_rows=int(values["future_rows"]),
            recovery_eta_from_now_ms=float(values["recovery_eta_from_now_ms"]),
            current_deadline_ms=float(values["current_deadline_ms"]),
            future_deadline_ms=float(values["future_deadline_ms"]),
        )
        output.append(
            {
                "id": check_id,
                "reason": reason,
                "context": values,
                **evaluation,
            }
        )
    return output


def build_artifact(config: PhaseConfig | None = None) -> dict[str, object]:
    """Call the real policies over the configured phase grid."""

    if config is None:
        config = PhaseConfig()
    profile = synthetic_profile()
    policy = FissionSpecPolicy(max_wait_ms=config.max_wait_ms)
    panels: list[dict[str, object]] = []
    total_refuse = 0
    for eta_ms in config.recovery_eta_ms:
        matrix: list[list[str]] = []
        for current_rows in config.current_cohort_sizes:
            row: list[str] = []
            for future_rows in config.future_cohort_sizes:
                evaluation = _evaluate(
                    config,
                    profile,
                    policy,
                    current_rows=current_rows,
                    future_rows=future_rows,
                    recovery_eta_from_now_ms=eta_ms,
                )
                row.append(str(evaluation["decision"]))
            matrix.append(row)
        re_fuse_cells = sum(decision == "re-fuse" for row in matrix for decision in row)
        total_refuse += re_fuse_cells
        panels.append(
            {
                "recovery_eta_from_now_ms": eta_ms,
                "decision_matrix_current_by_future": matrix,
                "re_fuse_cells": re_fuse_cells,
                "total_cells": sum(len(row) for row in matrix),
            }
        )

    total_cells = (
        len(config.recovery_eta_ms)
        * len(config.current_cohort_sizes)
        * len(config.future_cohort_sizes)
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact": "FissionSpec horizon-2 controller phase diagram",
        "evidence_class": "synthetic-policy-model",
        "measurement_warning": WARNING,
        "claim_boundary": CLAIM_BOUNDARY,
        "comparison": {
            "candidate": policy.name,
            "baseline": ImmediateFissionPolicy().name,
            "interpretation": (
                "re-fuse means the real horizon-2 policy waits for the known next "
                "readiness ETA while the real immediate-fission policy launches now"
            ),
            "wake_admission": (
                "global rolling EDF over current and next-ready rows; current rows "
                "win exact deadline ties because they became ready earlier"
            ),
        },
        "controller": {
            "max_wait_ms": config.max_wait_ms,
            "objective": "modeled aggregate flow time over current plus next readiness set",
            "lookahead": "next readiness event only",
        },
        "axes_and_constraints": {
            "current_cohort_sizes": list(config.current_cohort_sizes),
            "future_cohort_sizes": list(config.future_cohort_sizes),
            "recovery_eta_from_now_ms": list(config.recovery_eta_ms),
            "now_ms": 0.0,
            "oldest_ready_ms": 0.0,
            "capacity_rows": config.capacity,
            "verifier_slots_per_row": config.verifier_slots_per_row,
            "current_deadline_ms": config.current_deadline_ms,
            "future_deadline_ms": config.future_deadline_ms,
        },
        "hardware_profile": profile_document(profile),
        "legend": {
            "re-fuse": "wait until the known recovery ETA and launch one fused cohort",
            "dispatch-now": "launch the current ready cohort immediately",
        },
        "panels": panels,
        "summary": {
            "re_fuse_cells": total_refuse,
            "dispatch_now_cells": total_cells - total_refuse,
            "total_cells": total_cells,
        },
        "boundary_checks": _boundary_checks(config, profile, policy),
    }


def render_json(document: dict[str, object]) -> str:
    """Render stable, human-diffable machine-readable output."""

    return json.dumps(document, indent=2, sort_keys=True) + "\n"


def render_svg(document: dict[str, object]) -> str:
    """Render the phase grid as dependency-free SVG."""

    axes = document["axes_and_constraints"]
    panels = document["panels"]
    profile = document["hardware_profile"]
    if not isinstance(axes, dict) or not isinstance(panels, list) or not isinstance(profile, dict):
        raise TypeError("unexpected phase artifact schema")
    current_sizes = list(axes["current_cohort_sizes"])
    future_sizes = list(axes["future_cohort_sizes"])
    panel_columns = 4
    panel_width = 270
    panel_height = 205
    cell_width = 44
    cell_height = 31
    grid_x_offset = 70
    grid_y_offset = 51
    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="575" '
        'viewBox="0 0 1120 575" role="img" '
        'aria-labelledby="title description">',
        '<rect width="1120" height="575" fill="#f8fafc"/>',
        '<text id="title" x="28" y="34" font-family="sans-serif" font-size="22" '
        'font-weight="700" fill="#0f172a">FissionSpec horizon-2 controller phase diagram</text>',
        f'<text x="28" y="58" font-family="sans-serif" font-size="15" '
        f'font-weight="700" fill="#b91c1c">{html.escape(WARNING)} — POLICY '
        "DECISIONS, NOT THROUGHPUT EVIDENCE</text>",
        f'<text id="description" x="28" y="80" font-family="sans-serif" '
        f'font-size="12" fill="#334155">{html.escape(CLAIM_BOUNDARY)}</text>',
    ]

    for panel_index, panel in enumerate(panels):
        if not isinstance(panel, dict):
            raise TypeError("unexpected panel schema")
        panel_x = 20 + (panel_index % panel_columns) * panel_width
        panel_y = 100 + (panel_index // panel_columns) * panel_height
        eta_ms = float(panel["recovery_eta_from_now_ms"])
        matrix = panel["decision_matrix_current_by_future"]
        if not isinstance(matrix, list):
            raise TypeError("unexpected decision matrix schema")
        lines.append(
            f'<g transform="translate({panel_x},{panel_y})">'
            f'<rect width="255" height="190" rx="8" fill="#ffffff" stroke="#cbd5e1"/>'
            f'<text x="12" y="22" font-family="sans-serif" font-size="13" '
            f'font-weight="700" fill="#0f172a">Recovery ETA = {eta_ms:g} ms</text>'
            f'<text x="12" y="43" font-family="sans-serif" font-size="10" '
            f'fill="#475569">current rows ↓ / future rows →</text>'
        )
        for column, future_rows in enumerate(future_sizes):
            x = grid_x_offset + column * cell_width + cell_width / 2
            lines.append(
                f'<text x="{x:g}" y="46" text-anchor="middle" '
                f'font-family="monospace" font-size="10" fill="#334155">'
                f"{future_rows}</text>"
            )
        for row_index, (current_rows, decisions) in enumerate(
            zip(current_sizes, matrix, strict=True)
        ):
            y = grid_y_offset + row_index * cell_height
            lines.append(
                f'<text x="53" y="{y + 20:g}" text-anchor="end" '
                f'font-family="monospace" font-size="10" fill="#334155">'
                f"{current_rows}</text>"
            )
            if not isinstance(decisions, list):
                raise TypeError("unexpected decision row schema")
            for column, decision in enumerate(decisions):
                x = grid_x_offset + column * cell_width
                is_refuse = decision == "re-fuse"
                fill = "#0f766e" if is_refuse else "#b45309"
                glyph = "F" if is_refuse else "D"
                label = "re-fuse" if is_refuse else "dispatch now"
                lines.extend(
                    (
                        f'<rect x="{x}" y="{y}" width="{cell_width - 3}" '
                        f'height="{cell_height - 3}" rx="3" fill="{fill}">'
                        f"<title>{html.escape(label)}</title></rect>",
                        f'<text x="{x + (cell_width - 3) / 2:g}" y="{y + 19:g}" '
                        f'text-anchor="middle" font-family="monospace" font-size="12" '
                        f'font-weight="700" fill="#ffffff">{glyph}</text>',
                    )
                )
        lines.append("</g>")

    target_knots = profile["target_curve_rows_to_ms"]
    knot_text = ", ".join(f"{point[0]}→{point[1]:g}" for point in target_knots)
    lines.extend(
        (
            '<rect x="30" y="515" width="18" height="18" rx="3" fill="#0f766e"/>',
            '<text x="56" y="529" font-family="sans-serif" font-size="12" '
            'fill="#0f172a">F = re-fuse at ETA</text>',
            '<rect x="205" y="515" width="18" height="18" rx="3" fill="#b45309"/>',
            '<text x="231" y="529" font-family="sans-serif" font-size="12" '
            'fill="#0f172a">D = dispatch now</text>',
            f'<text x="390" y="529" font-family="sans-serif" font-size="11" '
            f'fill="#334155">target knots (rows→ms): {html.escape(knot_text)}; '
            f"slot cost: {float(profile['verifier_slot_ms']):g} ms</text>",
            '<text x="30" y="554" font-family="sans-serif" font-size="11" '
            'fill="#475569">Every cell calls both real policies with an exact '
            "rolling-EDF wake forecast; no measured samples are plotted.</text>",
            "</svg>",
        )
    )
    return "\n".join(lines) + "\n"


def write_artifact(output_dir: Path) -> tuple[Path, Path]:
    """Write the deterministic JSON and SVG pair."""

    document = build_artifact()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "controller_phase_diagram.json"
    svg_path = output_dir / "controller_phase_diagram.svg"
    json_path.write_text(render_json(document), encoding="utf-8")
    svg_path.write_text(render_svg(document), encoding="utf-8")
    return json_path, svg_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
        help="artifact directory (default: experiments/results)",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    json_path, svg_path = write_artifact(args.output_dir)
    print(WARNING)
    print(CLAIM_BOUNDARY)
    print(json_path)
    print(svg_path)


if __name__ == "__main__":
    main()
