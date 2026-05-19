#!/usr/bin/env python3
"""Render dependency-free SVG and Markdown from a synthetic sweep JSON."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping
from html import escape
from pathlib import Path
from typing import cast

EXPECTED_EVIDENCE = "synthetic-model"
WARNING = "SYNTHETIC MODEL OUTPUT — NOT GPU MEASUREMENTS."
WORKLOAD_ORDER = ("synchronized-cohort", "poisson", "bursty")
POLICY_ORDER = (
    "saguaro-barrier",
    "spectre-parallel-padded",
    "immediate-fission",
    "fixed-coalesce",
    "fissionspec-horizon-2",
)
POLICY_LABELS = {
    "saguaro-barrier": "Saguaro barrier",
    "spectre-parallel-padded": "SPECTRE parallel padded",
    "immediate-fission": "Immediate fission",
    "fixed-coalesce": "Fixed coalesce",
    "fissionspec-horizon-2": "FissionSpec H2",
}


def _object(value: object, *, field: str) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{field} must be a JSON object")
    return cast(dict[str, object], value)


def _objects(value: object, *, field: str) -> list[dict[str, object]]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a JSON array")
    return [_object(item, field=f"{field}[{index}]") for index, item in enumerate(value)]


def _text(row: Mapping[str, object], field: str) -> str:
    value = row.get(field)
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    return value


def _number(row: Mapping[str, object], field: str) -> float:
    value = row.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{field} must be a finite number")
    return float(value)


def _factor_pairs(
    rows: list[dict[str, object]],
) -> tuple[tuple[float, float], ...]:
    return tuple(
        sorted(
            {
                (
                    _number(row, "configured_cache_hit_probability"),
                    _number(row, "configured_token_acceptance_probability"),
                )
                for row in rows
            }
        )
    )


def load_aggregates(path: Path) -> list[dict[str, object]]:
    """Load only provenance-checked synthetic aggregate rows."""

    document = _object(json.loads(path.read_text(encoding="utf-8")), field="document")
    if document.get("evidence_class") != EXPECTED_EVIDENCE:
        raise ValueError("renderer accepts only evidence_class='synthetic-model'")
    if document.get("measurement_warning") != WARNING:
        raise ValueError("synthetic measurement warning is missing or changed")
    rows = _objects(document.get("aggregates"), field="aggregates")
    factor_pairs = _factor_pairs(rows)
    cache_levels = {cache for cache, _ in factor_pairs}
    token_levels = {token for _, token in factor_pairs}
    if len(cache_levels) < 2 or len(token_levels) < 2:
        raise ValueError("results must independently vary both probability axes")
    full_factorial = {
        (cache_probability, token_probability)
        for cache_probability in cache_levels
        for token_probability in token_levels
    }
    if set(factor_pairs) != full_factorial:
        raise ValueError("results do not contain a complete probability factorial")
    expected_keys = {
        (workload, cache, token, policy)
        for workload in WORKLOAD_ORDER
        for cache, token in factor_pairs
        for policy in POLICY_ORDER
    }
    actual_keys = {
        (
            _text(row, "workload"),
            _number(row, "configured_cache_hit_probability"),
            _number(row, "configured_token_acceptance_probability"),
            _text(row, "policy"),
        )
        for row in rows
    }
    if actual_keys != expected_keys or len(rows) != len(expected_keys):
        raise ValueError("aggregate rows do not contain the full policy factorial")
    return rows


def _lookup(
    rows: list[dict[str, object]],
    workload: str,
    cache_probability: float,
    token_probability: float,
    policy: str,
) -> dict[str, object]:
    return next(
        row
        for row in rows
        if row.get("workload") == workload
        and row.get("configured_cache_hit_probability") == cache_probability
        and row.get("configured_token_acceptance_probability") == token_probability
        and row.get("policy") == policy
    )


def render_markdown(rows: list[dict[str, object]]) -> str:
    """Render the complete factorial as a publication-review table."""

    lines = [
        "# Synthetic mechanism-study summary",
        "",
        f"> **{WARNING}**",
        "",
        "Means across matched common-random-number seeds. Cache availability "
        "and draft-token acceptance are independent factors. These values "
        "validate simulator mechanics only; they are not hardware evidence.",
        "",
        "| Workload | Cache p | Token p | Policy | Observed cache rate | "
        "Verifier tok/round | Throughput (tok/s) | P95 TBT (ms) | "
        "Gap TBT pass (%) | Request TBT pass (%) | TBT-request tok/s | Padded slots | "
        "Direct delay / hit (ms) | vs. barrier |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for workload in WORKLOAD_ORDER:
        for cache_probability, token_probability in _factor_pairs(rows):
            for policy in POLICY_ORDER:
                row = _lookup(
                    rows,
                    workload,
                    cache_probability,
                    token_probability,
                    policy,
                )
                lines.append(
                    "| "
                    + " | ".join(
                        (
                            workload,
                            f"{cache_probability:.2f}",
                            f"{token_probability:.2f}",
                            POLICY_LABELS[policy],
                            f"{_number(row, 'observed_cache_hit_rate'):.3f}",
                            f"{_number(row, 'mean_verifier_tokens_per_round'):.3f}",
                            f"{_number(row, 'throughput_tokens_per_s'):.1f}",
                            f"{_number(row, 'p95_tbt_ms'):.2f}",
                            f"{100.0 * _number(row, 'token_gap_slo_attainment'):.1f}",
                            f"{100.0 * _number(row, 'request_tbt_slo_attainment'):.1f}",
                            f"{_number(row, 'tbt_request_goodput_tokens_per_s'):.1f}",
                            f"{_number(row, 'padded_verifier_slots'):.1f}",
                            f"{_number(row, 'direct_hit_delay_ms'):.3f}",
                            f"{_number(row, 'throughput_ratio_vs_barrier'):.3f}x",
                        )
                    )
                    + " |"
                )
    lines.extend(
        (
            "",
            "Generated from `synthetic_sweep.json` by `tools/render_synthetic_results.py`.",
            "",
        )
    )
    return "\n".join(lines)


def _blend(target: tuple[int, int, int], strength: float) -> str:
    bounded = min(1.0, max(0.0, strength))
    channels = tuple(round(248 + (target_channel - 248) * bounded) for target_channel in target)
    return "#" + "".join(f"{channel:02x}" for channel in channels)


def render_svg(rows: list[dict[str, object]]) -> str:
    """Render factorial heatmaps of FissionSpec throughput versus the barrier."""

    factor_pairs = _factor_pairs(rows)
    cache_levels = tuple(sorted({cache for cache, _ in factor_pairs}))
    token_levels = tuple(sorted({token for _, token in factor_pairs}))
    panel_width = 150 + 112 * len(cache_levels)
    panel_gap = 20
    width = 40 + len(WORKLOAD_ORDER) * panel_width + 2 * panel_gap
    height = 255 + 92 * len(token_levels)
    ratios = [
        _number(
            _lookup(
                rows,
                workload,
                cache_probability,
                token_probability,
                "fissionspec-horizon-2",
            ),
            "throughput_ratio_vs_barrier",
        )
        for workload in WORKLOAD_ORDER
        for cache_probability, token_probability in factor_pairs
    ]
    scale = max(0.01, max(abs(ratio - 1.0) for ratio in ratios))

    elements = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}" '
            'role="img" aria-labelledby="title description">'
        ),
        ('<title id="title">Synthetic FissionSpec throughput-ratio factorial</title>'),
        (
            '<description id="description">FissionSpec horizon-2 throughput '
            "divided by the Saguaro barrier under independently varied cache "
            "hit and token acceptance probabilities. Not GPU measurements."
            "</description>"
        ),
        f"<metadata>{escape(WARNING)}</metadata>",
        '<rect width="100%" height="100%" fill="#fbfcfe"/>',
        (
            f'<text x="{width / 2:.1f}" y="32" text-anchor="middle" '
            'font-family="system-ui,sans-serif" font-size="21" '
            'font-weight="700" fill="#111827">'
            "FissionSpec H2 throughput / Saguaro barrier</text>"
        ),
        (
            f'<text x="{width / 2:.1f}" y="61" text-anchor="middle" '
            'font-family="system-ui,sans-serif" font-size="15" '
            'font-weight="700" fill="#b42318">'
            f"{escape(WARNING)}</text>"
        ),
    ]
    plot_top = 132
    for workload_index, workload in enumerate(WORKLOAD_ORDER):
        panel_left = 20 + workload_index * (panel_width + panel_gap)
        grid_left = panel_left + 125
        elements.extend(
            (
                (
                    f'<text x="{panel_left + panel_width / 2:.1f}" y="96" '
                    'text-anchor="middle" font-family="system-ui,sans-serif" '
                    f'font-size="15" font-weight="650" fill="#111827">'
                    f"{escape(workload)}</text>"
                ),
                (
                    f'<text x="{grid_left + 56 * len(cache_levels):.1f}" y="118" '
                    'text-anchor="middle" font-family="system-ui,sans-serif" '
                    'font-size="12" fill="#4b5563">configured cache-hit p</text>'
                ),
            )
        )
        for column, cache_probability in enumerate(cache_levels):
            x = grid_left + column * 112
            elements.append(
                f'<text x="{x + 52:.1f}" y="{plot_top - 3}" '
                'text-anchor="middle" font-family="ui-monospace,monospace" '
                f'font-size="12" fill="#374151">{cache_probability:.2f}</text>'
            )
        for row_index, token_probability in enumerate(token_levels):
            y = plot_top + row_index * 92
            elements.extend(
                (
                    (
                        f'<text x="{grid_left - 13}" y="{y + 36}" '
                        'text-anchor="end" font-family="ui-monospace,monospace" '
                        f'font-size="12" fill="#374151">{token_probability:.2f}</text>'
                    ),
                    (
                        f'<text x="{panel_left + 8}" y="{y + 36}" '
                        'font-family="system-ui,sans-serif" font-size="11" '
                        'fill="#4b5563">token p</text>'
                    ),
                )
            )
            for column, cache_probability in enumerate(cache_levels):
                row = _lookup(
                    rows,
                    workload,
                    cache_probability,
                    token_probability,
                    "fissionspec-horizon-2",
                )
                ratio = _number(row, "throughput_ratio_vs_barrier")
                delta = ratio - 1.0
                strength = abs(delta) / scale
                color = _blend(
                    (37, 99, 235) if delta >= 0.0 else (220, 38, 38),
                    0.18 + 0.72 * strength,
                )
                text_color = "#ffffff" if strength >= 0.55 else "#111827"
                x = grid_left + column * 112
                elements.extend(
                    (
                        (
                            f'<rect x="{x}" y="{y + 8}" width="104" height="76" '
                            f'rx="5" fill="{color}" stroke="#d1d5db"/>'
                        ),
                        (
                            f'<text x="{x + 52}" y="{y + 40}" '
                            'text-anchor="middle" '
                            'font-family="ui-monospace,monospace" '
                            f'font-size="16" font-weight="700" fill="{text_color}">'
                            f"{ratio:.3f}x</text>"
                        ),
                        (
                            f'<text x="{x + 52}" y="{y + 61}" '
                            'text-anchor="middle" '
                            'font-family="ui-monospace,monospace" '
                            f'font-size="10" fill="{text_color}">'
                            f"P95 delta "
                            f"{_number(row, 'p95_tbt_delta_vs_barrier_ms'):+.2f} ms"
                            "</text>"
                        ),
                    )
                )
    footer_y = plot_top + 92 * len(token_levels) + 40
    elements.extend(
        (
            (
                f'<text x="{width / 2:.1f}" y="{footer_y}" '
                'text-anchor="middle" font-family="system-ui,sans-serif" '
                'font-size="12" fill="#374151">'
                "Blue: faster than barrier. Red: slower. "
                "Cells show matched-seed mean synthetic throughput ratio.</text>"
            ),
            "</svg>",
            "",
        )
    )
    return "\n".join(elements)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="synthetic_sweep.json")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="defaults to the input JSON directory",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    rows = load_aggregates(args.input)
    output_dir = args.output_dir or args.input.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    markdown_path = output_dir / "SYNTHETIC_RESULTS.md"
    svg_path = output_dir / "synthetic_factorial.svg"
    markdown_path.write_text(render_markdown(rows), encoding="utf-8")
    svg_path.write_text(render_svg(rows), encoding="utf-8")
    print(WARNING)
    print(f"wrote {markdown_path} and {svg_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
