"""Hash-locked, GPU-free audit of captured production output distributions.

The module intentionally separates capture from analysis.  A serving engine can
write one paired corpus while accelerator access is available; every numerical,
statistical, threshold, and integrity check below then runs deterministically on
CPU without importing an ML framework.

Synthetic fixtures exercise this implementation only.  They do not establish
parity for a real model, engine, tokenizer, quantizer, or GPU kernel.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final, Literal, cast

from fissionspec.artifacts import canonical_json_bytes, sha256_document
from fissionspec.statistics import one_sample_cluster_mean_interval

CORPUS_SCHEMA: Final = "fissionspec.output-audit-corpus.v1"
REPORT_SCHEMA: Final = "fissionspec.output-audit-report.v1"
CAPTURED_EVIDENCE: Final = "captured-production-output"
SYNTHETIC_EVIDENCE: Final = "synthetic-cpu-fixture"
CAPTURED_WARNING: Final = (
    "CAPTURED OUTPUT PARITY EVIDENCE — THIS IS NOT A SERVING-PERFORMANCE MEASUREMENT."
)
SYNTHETIC_WARNING: Final = (
    "SYNTHETIC CPU FIXTURE — DOES NOT ESTABLISH REAL ENGINE OR GPU-KERNEL PARITY."
)
INFERENTIAL_FAMILY_SIZE: Final = 4
Encoding = Literal["logits", "probabilities"]


class OutputAuditError(ValueError):
    """Raised when an audit input or preregistered gate is invalid."""


class OutputAuditIntegrityError(OutputAuditError):
    """Raised when a self-hashed corpus or report fails verification."""


def _strict_keys(
    document: Mapping[str, object],
    expected: set[str],
    *,
    field: str,
) -> None:
    actual = set(document)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise OutputAuditError(f"{field} keys mismatch; missing={missing}, extra={extra}")


def _mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise OutputAuditError(f"{field} must be a string-keyed object")
    return cast(Mapping[str, object], value)


def _sequence(value: object, *, field: str) -> Sequence[object]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise OutputAuditError(f"{field} must be an array")
    return cast(Sequence[object], value)


def _nonempty_string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise OutputAuditError(f"{field} must be a non-empty, trimmed string")
    return value


def _digest(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value.lower() != value
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise OutputAuditError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _integer(value: object, *, field: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise OutputAuditError(f"{field} must be an integer")
    if minimum is not None and value < minimum:
        raise OutputAuditError(f"{field} must be at least {minimum}")
    return value


def _finite_float(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise OutputAuditError(f"{field} must be a finite number")
    try:
        converted = float(value)
    except OverflowError as error:
        raise OutputAuditError(f"{field} must be a finite number") from error
    if not math.isfinite(converted):
        raise OutputAuditError(f"{field} must be a finite number")
    return 0.0 if converted == 0.0 else converted


def _probability(
    value: object,
    *,
    field: str,
    positive: bool = False,
    include_one: bool = True,
) -> float:
    converted = _finite_float(value, field=field)
    lower_valid = converted > 0.0 if positive else converted >= 0.0
    upper_valid = converted <= 1.0 if include_one else converted < 1.0
    if not lower_valid or not upper_valid:
        lower = "(0" if positive else "[0"
        upper = "1]" if include_one else "1)"
        raise OutputAuditError(f"{field} must lie in {lower}, {upper}")
    return converted


def _warning(evidence_class: str) -> str:
    if evidence_class == CAPTURED_EVIDENCE:
        return CAPTURED_WARNING
    if evidence_class == SYNTHETIC_EVIDENCE:
        return SYNTHETIC_WARNING
    raise OutputAuditError(f"unsupported evidence_class: {evidence_class!r}")


def _normalize_capture(value: object) -> dict[str, object]:
    capture = _mapping(value, field="capture")
    expected = {
        "capture_id",
        "model_id",
        "tokenizer_sha256",
        "reference_engine_sha256",
        "candidate_engine_sha256",
        "capture_config_sha256",
        "capture_tool_sha256",
    }
    _strict_keys(capture, expected, field="capture")
    return {
        "capture_id": _nonempty_string(capture["capture_id"], field="capture.capture_id"),
        "model_id": _nonempty_string(capture["model_id"], field="capture.model_id"),
        "tokenizer_sha256": _digest(
            capture["tokenizer_sha256"],
            field="capture.tokenizer_sha256",
        ),
        "reference_engine_sha256": _digest(
            capture["reference_engine_sha256"],
            field="capture.reference_engine_sha256",
        ),
        "candidate_engine_sha256": _digest(
            capture["candidate_engine_sha256"],
            field="capture.candidate_engine_sha256",
        ),
        "capture_config_sha256": _digest(
            capture["capture_config_sha256"],
            field="capture.capture_config_sha256",
        ),
        "capture_tool_sha256": _digest(
            capture["capture_tool_sha256"],
            field="capture.capture_tool_sha256",
        ),
    }


def _normalize_distribution(value: object, *, field: str) -> dict[str, object]:
    distribution = _mapping(value, field=field)
    _strict_keys(distribution, {"encoding", "token_ids", "values"}, field=field)
    encoding = distribution["encoding"]
    if encoding not in {"logits", "probabilities"}:
        raise OutputAuditError(f"{field}.encoding must be 'logits' or 'probabilities'")
    raw_tokens = _sequence(distribution["token_ids"], field=f"{field}.token_ids")
    raw_values = _sequence(distribution["values"], field=f"{field}.values")
    if not raw_tokens or len(raw_tokens) != len(raw_values):
        raise OutputAuditError(f"{field} token_ids and values must have equal non-zero lengths")
    pairs: list[tuple[int, float]] = []
    seen: set[int] = set()
    for index, (raw_token, raw_value) in enumerate(zip(raw_tokens, raw_values, strict=True)):
        token = _integer(raw_token, field=f"{field}.token_ids[{index}]", minimum=0)
        if token in seen:
            raise OutputAuditError(f"{field}.token_ids must be unique")
        seen.add(token)
        number = _finite_float(raw_value, field=f"{field}.values[{index}]")
        if encoding == "probabilities" and number < 0.0:
            raise OutputAuditError(f"{field}.values must be non-negative probabilities")
        pairs.append((token, number))
    pairs.sort()
    if encoding == "probabilities":
        total = math.fsum(number for _, number in pairs)
        if total <= 0.0 or not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise OutputAuditError(f"{field}.values must sum to one within 1e-9")
        pairs = [(token, number / total) for token, number in pairs]
    return {
        "encoding": encoding,
        "token_ids": [token for token, _ in pairs],
        "values": [number for _, number in pairs],
    }


def _normalize_slices(value: object, *, field: str) -> dict[str, str]:
    slices = _mapping(value, field=field)
    if not slices:
        raise OutputAuditError(f"{field} must contain at least one preregistered slice")
    normalized: dict[str, str] = {}
    for dimension, raw_label in sorted(slices.items()):
        normalized[_nonempty_string(dimension, field=f"{field} key")] = _nonempty_string(
            raw_label,
            field=f"{field}.{dimension}",
        )
    return normalized


def _normalize_record(value: object, *, index: int) -> dict[str, object]:
    field = f"records[{index}]"
    record = _mapping(value, field=field)
    expected = {
        "record_id",
        "cluster_id",
        "slices",
        "reference",
        "candidate",
        "proposed_token_id",
        "draft_probability",
        "uniform",
    }
    _strict_keys(record, expected, field=field)
    reference = _normalize_distribution(record["reference"], field=f"{field}.reference")
    candidate = _normalize_distribution(record["candidate"], field=f"{field}.candidate")
    if (
        reference["encoding"] == "logits"
        and candidate["encoding"] == "logits"
        and reference["token_ids"] != candidate["token_ids"]
    ):
        raise OutputAuditError("paired logit vectors must describe exactly the same token support")
    return {
        "record_id": _nonempty_string(record["record_id"], field=f"{field}.record_id"),
        "cluster_id": _nonempty_string(record["cluster_id"], field=f"{field}.cluster_id"),
        "slices": _normalize_slices(record["slices"], field=f"{field}.slices"),
        "reference": reference,
        "candidate": candidate,
        "proposed_token_id": _integer(
            record["proposed_token_id"],
            field=f"{field}.proposed_token_id",
            minimum=0,
        ),
        "draft_probability": _probability(
            record["draft_probability"],
            field=f"{field}.draft_probability",
            positive=True,
        ),
        "uniform": _probability(
            record["uniform"],
            field=f"{field}.uniform",
            include_one=False,
        ),
    }


def _normalize_records(value: object) -> list[dict[str, object]]:
    records = _sequence(value, field="records")
    if not records:
        raise OutputAuditError("records must not be empty")
    normalized = [_normalize_record(record, index=index) for index, record in enumerate(records)]
    normalized.sort(key=lambda record: cast(str, record["record_id"]))
    identifiers = [cast(str, record["record_id"]) for record in normalized]
    if len(identifiers) != len(set(identifiers)):
        raise OutputAuditError("record_id values must be unique")
    return normalized


def build_corpus(
    *,
    evidence_class: str,
    capture: Mapping[str, object],
    records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Canonicalize and self-hash a paired output corpus."""

    evidence = _nonempty_string(evidence_class, field="evidence_class")
    normalized_records = _normalize_records(records)
    payload: dict[str, object] = {
        "schema": CORPUS_SCHEMA,
        "evidence_class": evidence,
        "measurement_warning": _warning(evidence),
        "capture": _normalize_capture(capture),
        "record_count": len(normalized_records),
        "records": normalized_records,
    }
    return {**payload, "payload_sha256": sha256_document(payload)}


def _normalize_corpus_payload(document: Mapping[str, object]) -> dict[str, object]:
    expected = {
        "schema",
        "evidence_class",
        "measurement_warning",
        "capture",
        "record_count",
        "records",
    }
    _strict_keys(document, expected, field="corpus payload")
    if document["schema"] != CORPUS_SCHEMA:
        raise OutputAuditError("unsupported output-audit corpus schema")
    evidence = _nonempty_string(document["evidence_class"], field="evidence_class")
    if document["measurement_warning"] != _warning(evidence):
        raise OutputAuditError("corpus measurement warning is missing or incorrect")
    records = _normalize_records(document["records"])
    count = _integer(document["record_count"], field="record_count", minimum=1)
    if count != len(records):
        raise OutputAuditError("record_count does not match records")
    return {
        "schema": CORPUS_SCHEMA,
        "evidence_class": evidence,
        "measurement_warning": _warning(evidence),
        "capture": _normalize_capture(document["capture"]),
        "record_count": count,
        "records": records,
    }


def verify_corpus(document: Mapping[str, object]) -> str:
    """Verify schema, canonical form, and hash; return the corpus payload digest."""

    supplied = _digest(document.get("payload_sha256"), field="payload_sha256")
    payload = dict(document)
    payload.pop("payload_sha256", None)
    actual = sha256_document(payload)
    if supplied != actual:
        raise OutputAuditIntegrityError("corpus payload hash mismatch")
    try:
        normalized = _normalize_corpus_payload(payload)
    except OutputAuditError as error:
        raise OutputAuditIntegrityError(str(error)) from error
    if canonical_json_bytes(normalized) != canonical_json_bytes(payload):
        raise OutputAuditIntegrityError("corpus payload is not in canonical form")
    return actual


@dataclass(frozen=True, slots=True)
class AuditThresholds:
    """Preregistered, immutable parity and uncertainty gates."""

    schema_version: int = 1
    top_k: int = 5
    min_records: int = 256
    min_clusters: int = 16
    bootstrap_resamples: int = 2_000
    familywise_alpha: float = 0.05
    max_greedy_mismatch_rate: float = 0.0
    max_greedy_mismatch_upper: float = 0.015
    max_acceptance_divergence_rate: float = 0.0
    max_acceptance_divergence_upper: float = 0.015
    max_mean_tv: float = 1e-7
    max_mean_tv_upper: float = 2e-7
    max_record_tv: float = 1e-5
    max_mean_js: float = 1e-12
    max_mean_js_upper: float = 1e-11
    max_record_js: float = 1e-8
    max_forward_kl: float = 1e-7
    max_reverse_kl: float = 1e-7
    min_mean_top_k_overlap: float = 1.0
    max_mean_greedy_rank_drift: float = 0.0
    max_mean_margin_drift: float = 1e-6

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise OutputAuditError("unsupported threshold schema_version")
        _integer(self.top_k, field="top_k", minimum=1)
        _integer(self.min_records, field="min_records", minimum=1)
        _integer(self.min_clusters, field="min_clusters", minimum=2)
        if self.min_clusters > self.min_records:
            raise OutputAuditError("min_clusters cannot exceed min_records")
        if _integer(self.bootstrap_resamples, field="bootstrap_resamples", minimum=100) < 100:
            raise AssertionError("unreachable")
        alpha = _probability(
            self.familywise_alpha,
            field="familywise_alpha",
            positive=True,
            include_one=False,
        )
        if alpha >= 0.5:
            raise OutputAuditError("familywise_alpha must be less than 0.5")
        probability_fields = (
            "max_greedy_mismatch_rate",
            "max_greedy_mismatch_upper",
            "max_acceptance_divergence_rate",
            "max_acceptance_divergence_upper",
            "max_mean_tv",
            "max_mean_tv_upper",
            "max_record_tv",
            "max_mean_js",
            "max_mean_js_upper",
            "max_record_js",
            "min_mean_top_k_overlap",
        )
        for field in probability_fields:
            _probability(getattr(self, field), field=field)
        nonnegative_fields = (
            "max_forward_kl",
            "max_reverse_kl",
            "max_mean_greedy_rank_drift",
            "max_mean_margin_drift",
        )
        for field in nonnegative_fields:
            if _finite_float(getattr(self, field), field=field) < 0.0:
                raise OutputAuditError(f"{field} must be non-negative")

    def as_dict(self) -> dict[str, object]:
        return cast(dict[str, object], asdict(self))

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> AuditThresholds:
        """Load a strict threshold mapping, rejecting missing or extra fields."""

        defaults = cls().as_dict()
        _strict_keys(value, set(defaults), field="thresholds")
        return cls(
            schema_version=_integer(value["schema_version"], field="schema_version"),
            top_k=_integer(value["top_k"], field="top_k"),
            min_records=_integer(value["min_records"], field="min_records"),
            min_clusters=_integer(value["min_clusters"], field="min_clusters"),
            bootstrap_resamples=_integer(
                value["bootstrap_resamples"],
                field="bootstrap_resamples",
            ),
            familywise_alpha=_finite_float(
                value["familywise_alpha"],
                field="familywise_alpha",
            ),
            max_greedy_mismatch_rate=_finite_float(
                value["max_greedy_mismatch_rate"],
                field="max_greedy_mismatch_rate",
            ),
            max_greedy_mismatch_upper=_finite_float(
                value["max_greedy_mismatch_upper"],
                field="max_greedy_mismatch_upper",
            ),
            max_acceptance_divergence_rate=_finite_float(
                value["max_acceptance_divergence_rate"],
                field="max_acceptance_divergence_rate",
            ),
            max_acceptance_divergence_upper=_finite_float(
                value["max_acceptance_divergence_upper"],
                field="max_acceptance_divergence_upper",
            ),
            max_mean_tv=_finite_float(value["max_mean_tv"], field="max_mean_tv"),
            max_mean_tv_upper=_finite_float(
                value["max_mean_tv_upper"],
                field="max_mean_tv_upper",
            ),
            max_record_tv=_finite_float(value["max_record_tv"], field="max_record_tv"),
            max_mean_js=_finite_float(value["max_mean_js"], field="max_mean_js"),
            max_mean_js_upper=_finite_float(
                value["max_mean_js_upper"],
                field="max_mean_js_upper",
            ),
            max_record_js=_finite_float(value["max_record_js"], field="max_record_js"),
            max_forward_kl=_finite_float(
                value["max_forward_kl"],
                field="max_forward_kl",
            ),
            max_reverse_kl=_finite_float(
                value["max_reverse_kl"],
                field="max_reverse_kl",
            ),
            min_mean_top_k_overlap=_finite_float(
                value["min_mean_top_k_overlap"],
                field="min_mean_top_k_overlap",
            ),
            max_mean_greedy_rank_drift=_finite_float(
                value["max_mean_greedy_rank_drift"],
                field="max_mean_greedy_rank_drift",
            ),
            max_mean_margin_drift=_finite_float(
                value["max_mean_margin_drift"],
                field="max_mean_margin_drift",
            ),
        )


def _probabilities(distribution: Mapping[str, object]) -> dict[int, float]:
    tokens = cast(list[int], distribution["token_ids"])
    values = cast(list[float], distribution["values"])
    if distribution["encoding"] == "probabilities":
        return dict(zip(tokens, values, strict=True))
    maximum = max(values)
    weights = [math.exp(value - maximum) for value in values]
    total = math.fsum(weights)
    if total <= 0.0 or not math.isfinite(total):
        raise OutputAuditError("stable softmax produced an invalid normalization")
    return {token: weight / total for token, weight in zip(tokens, weights, strict=True)}


def _ranked_tokens(probabilities: Mapping[int, float]) -> list[int]:
    return sorted(probabilities, key=lambda token: (-probabilities[token], token))


def _kl(left: Mapping[int, float], right: Mapping[int, float], support: Sequence[int]) -> float:
    terms: list[float] = []
    for token in support:
        left_mass = left.get(token, 0.0)
        if left_mass == 0.0:
            continue
        right_mass = right.get(token, 0.0)
        if right_mass == 0.0:
            return math.inf
        terms.append(left_mass * math.log(left_mass / right_mass))
    return max(0.0, math.fsum(terms))


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def _mean(values: Sequence[float]) -> float:
    return math.fsum(values) / len(values)


def _distribution_summary(values: Sequence[float]) -> dict[str, object]:
    return {
        "mean": _mean(values),
        "max": max(values),
        "p95": _quantile(values, 0.95),
    }


def _kl_summary(values: Sequence[float | None]) -> dict[str, object]:
    finite = [value for value in values if value is not None]
    infinite_count = len(values) - len(finite)
    return {
        "count": len(values),
        "infinite_count": infinite_count,
        "finite_mean": _mean(finite) if finite else None,
        "finite_max": max(finite) if finite else None,
        "mean": _mean(finite) if not infinite_count and finite else None,
        "max": max(finite) if not infinite_count and finite else None,
    }


def _record_diagnostic(
    record: Mapping[str, object],
    *,
    top_k: int,
) -> dict[str, object]:
    reference = _probabilities(cast(Mapping[str, object], record["reference"]))
    candidate = _probabilities(cast(Mapping[str, object], record["candidate"]))
    support = sorted(set(reference) | set(candidate))
    if len(support) < 2:
        raise OutputAuditError("each paired distribution needs at least two union-support tokens")
    reference_ranked = _ranked_tokens({token: reference.get(token, 0.0) for token in support})
    candidate_ranked = _ranked_tokens({token: candidate.get(token, 0.0) for token in support})
    reference_greedy = reference_ranked[0]
    candidate_greedy = candidate_ranked[0]
    total_variation = 0.5 * math.fsum(
        abs(reference.get(token, 0.0) - candidate.get(token, 0.0)) for token in support
    )
    mixture = {
        token: 0.5 * (reference.get(token, 0.0) + candidate.get(token, 0.0)) for token in support
    }
    forward_kl = _kl(reference, candidate, support)
    reverse_kl = _kl(candidate, reference, support)
    js = 0.5 * _kl(reference, mixture, support) + 0.5 * _kl(candidate, mixture, support)
    effective_k = min(top_k, len(support))
    reference_top = set(reference_ranked[:effective_k])
    candidate_top = set(candidate_ranked[:effective_k])
    top_k_overlap = len(reference_top & candidate_top) / effective_k
    reference_margin = reference.get(reference_ranked[0], 0.0) - reference.get(
        reference_ranked[1],
        0.0,
    )
    candidate_margin = candidate.get(candidate_ranked[0], 0.0) - candidate.get(
        candidate_ranked[1],
        0.0,
    )
    greedy_rank_drift = max(
        candidate_ranked.index(reference_greedy),
        reference_ranked.index(candidate_greedy),
    )
    proposed = cast(int, record["proposed_token_id"])
    draft_probability = cast(float, record["draft_probability"])
    uniform = cast(float, record["uniform"])
    reference_acceptance_probability = min(
        1.0,
        reference.get(proposed, 0.0) / draft_probability,
    )
    candidate_acceptance_probability = min(
        1.0,
        candidate.get(proposed, 0.0) / draft_probability,
    )
    reference_accept = uniform < reference_acceptance_probability
    candidate_accept = uniform < candidate_acceptance_probability
    return {
        "record_id": record["record_id"],
        "cluster_id": record["cluster_id"],
        "slices": record["slices"],
        "reference_greedy_token_id": reference_greedy,
        "candidate_greedy_token_id": candidate_greedy,
        "greedy_mismatch": reference_greedy != candidate_greedy,
        "reference_acceptance_probability": reference_acceptance_probability,
        "candidate_acceptance_probability": candidate_acceptance_probability,
        "reference_accept": reference_accept,
        "candidate_accept": candidate_accept,
        "acceptance_divergence": reference_accept != candidate_accept,
        "total_variation": total_variation,
        "jensen_shannon": js,
        "forward_kl": None if math.isinf(forward_kl) else forward_kl,
        "forward_kl_infinite": math.isinf(forward_kl),
        "reverse_kl": None if math.isinf(reverse_kl) else reverse_kl,
        "reverse_kl_infinite": math.isinf(reverse_kl),
        "top_k": effective_k,
        "top_k_overlap": top_k_overlap,
        "greedy_rank_drift": greedy_rank_drift,
        "reference_margin": reference_margin,
        "candidate_margin": candidate_margin,
        "absolute_margin_drift": abs(candidate_margin - reference_margin),
    }


def _summarize(diagnostics: Sequence[Mapping[str, object]]) -> dict[str, object]:
    mismatch_count = sum(bool(row["greedy_mismatch"]) for row in diagnostics)
    acceptance_count = sum(bool(row["acceptance_divergence"]) for row in diagnostics)
    tv = [cast(float, row["total_variation"]) for row in diagnostics]
    js = [cast(float, row["jensen_shannon"]) for row in diagnostics]
    overlaps = [cast(float, row["top_k_overlap"]) for row in diagnostics]
    ranks = [float(cast(int, row["greedy_rank_drift"])) for row in diagnostics]
    margins = [cast(float, row["absolute_margin_drift"]) for row in diagnostics]
    forward = [cast(float | None, row["forward_kl"]) for row in diagnostics]
    reverse = [cast(float | None, row["reverse_kl"]) for row in diagnostics]
    return {
        "records": len(diagnostics),
        "clusters": len({cast(str, row["cluster_id"]) for row in diagnostics}),
        "greedy_mismatch": {
            "count": mismatch_count,
            "rate": mismatch_count / len(diagnostics),
        },
        "acceptance_divergence": {
            "count": acceptance_count,
            "rate": acceptance_count / len(diagnostics),
        },
        "total_variation": _distribution_summary(tv),
        "jensen_shannon": _distribution_summary(js),
        "forward_kl": _kl_summary(forward),
        "reverse_kl": _kl_summary(reverse),
        "top_k_overlap": _distribution_summary(overlaps),
        "greedy_rank_drift": _distribution_summary(ranks),
        "absolute_margin_drift": _distribution_summary(margins),
    }


def _slice_diagnostics(
    diagnostics: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[Mapping[str, object]]] = {}
    for row in diagnostics:
        slices = cast(Mapping[str, str], row["slices"])
        for dimension, label in slices.items():
            grouped.setdefault((dimension, label), []).append(row)
    return [
        {
            "dimension": dimension,
            "label": label,
            "summary": _summarize(grouped[(dimension, label)]),
        }
        for dimension, label in sorted(grouped)
    ]


def _log_binomial_term(n: int, k: int, probability: float) -> float:
    if probability == 0.0:
        return 0.0 if k == 0 else -math.inf
    if probability == 1.0:
        return 0.0 if k == n else -math.inf
    return (
        math.lgamma(n + 1)
        - math.lgamma(k + 1)
        - math.lgamma(n - k + 1)
        + k * math.log(probability)
        + (n - k) * math.log1p(-probability)
    )


def _binomial_cdf(successes: int, trials: int, probability: float) -> float:
    logs = [_log_binomial_term(trials, index, probability) for index in range(successes + 1)]
    maximum = max(logs)
    if maximum == -math.inf:
        return 0.0
    return min(1.0, math.exp(maximum) * math.fsum(math.exp(value - maximum) for value in logs))


def exact_binomial_upper(
    successes: int,
    trials: int,
    *,
    alpha: float,
) -> dict[str, object]:
    """Return a one-sided Clopper--Pearson upper confidence bound."""

    observed = _integer(successes, field="successes", minimum=0)
    total = _integer(trials, field="trials", minimum=1)
    if observed > total:
        raise OutputAuditError("successes cannot exceed trials")
    error_probability = _probability(
        alpha,
        field="alpha",
        positive=True,
        include_one=False,
    )
    if observed == total:
        upper = 1.0
    else:
        low = observed / total
        high = 1.0
        for _ in range(100):
            midpoint = (low + high) / 2.0
            if _binomial_cdf(observed, total, midpoint) > error_probability:
                low = midpoint
            else:
                high = midpoint
        upper = high
    return {
        "method": "one-sided-clopper-pearson",
        "successes": observed,
        "trials": total,
        "point_estimate": observed / total,
        "alpha": error_probability,
        "confidence_level": 1.0 - error_probability,
        "upper": upper,
    }


def _cluster_values(
    diagnostics: Sequence[Mapping[str, object]],
    field: str,
) -> dict[str, list[float]]:
    grouped: dict[str, list[float]] = {}
    for row in diagnostics:
        grouped.setdefault(cast(str, row["cluster_id"]), []).append(cast(float, row[field]))
    return grouped


def _violation(
    code: str,
    *,
    observed: object,
    relation: str,
    limit: object,
    scope: str = "aggregate",
) -> dict[str, object]:
    return {
        "code": code,
        "scope": scope,
        "observed": observed,
        "required_relation": relation,
        "limit": limit,
    }


def _summary_violations(
    summary: Mapping[str, object],
    thresholds: AuditThresholds,
    *,
    scope: str,
    include_rates: bool,
) -> list[dict[str, object]]:
    violations: list[dict[str, object]] = []
    mismatch = cast(Mapping[str, object], summary["greedy_mismatch"])
    acceptance = cast(Mapping[str, object], summary["acceptance_divergence"])
    tv = cast(Mapping[str, object], summary["total_variation"])
    js = cast(Mapping[str, object], summary["jensen_shannon"])
    forward = cast(Mapping[str, object], summary["forward_kl"])
    reverse = cast(Mapping[str, object], summary["reverse_kl"])
    overlap = cast(Mapping[str, object], summary["top_k_overlap"])
    rank = cast(Mapping[str, object], summary["greedy_rank_drift"])
    margin = cast(Mapping[str, object], summary["absolute_margin_drift"])

    def at_most(code: str, observed: float, limit: float) -> None:
        if observed > limit:
            violations.append(
                _violation(code, observed=observed, relation="<=", limit=limit, scope=scope)
            )

    if include_rates:
        at_most(
            "greedy_mismatch_rate",
            cast(float, mismatch["rate"]),
            thresholds.max_greedy_mismatch_rate,
        )
        at_most(
            "acceptance_divergence_rate",
            cast(float, acceptance["rate"]),
            thresholds.max_acceptance_divergence_rate,
        )
    at_most("mean_tv", cast(float, tv["mean"]), thresholds.max_mean_tv)
    at_most("max_record_tv", cast(float, tv["max"]), thresholds.max_record_tv)
    at_most("mean_js", cast(float, js["mean"]), thresholds.max_mean_js)
    at_most("max_record_js", cast(float, js["max"]), thresholds.max_record_js)
    if cast(int, forward["infinite_count"]):
        violations.append(
            _violation(
                "forward_kl_infinite",
                observed=forward["infinite_count"],
                relation="==",
                limit=0,
                scope=scope,
            )
        )
    else:
        at_most(
            "forward_kl",
            cast(float, forward["mean"]),
            thresholds.max_forward_kl,
        )
    if cast(int, reverse["infinite_count"]):
        violations.append(
            _violation(
                "reverse_kl_infinite",
                observed=reverse["infinite_count"],
                relation="==",
                limit=0,
                scope=scope,
            )
        )
    else:
        at_most(
            "reverse_kl",
            cast(float, reverse["mean"]),
            thresholds.max_reverse_kl,
        )
    if cast(float, overlap["mean"]) < thresholds.min_mean_top_k_overlap:
        violations.append(
            _violation(
                "mean_top_k_overlap",
                observed=overlap["mean"],
                relation=">=",
                limit=thresholds.min_mean_top_k_overlap,
                scope=scope,
            )
        )
    at_most(
        "mean_greedy_rank_drift",
        cast(float, rank["mean"]),
        thresholds.max_mean_greedy_rank_drift,
    )
    at_most(
        "mean_margin_drift",
        cast(float, margin["mean"]),
        thresholds.max_mean_margin_drift,
    )
    return violations


def audit_corpus(
    document: Mapping[str, object],
    *,
    thresholds: AuditThresholds | None = None,
) -> dict[str, object]:
    """Audit a verified corpus and return a canonical, self-hashed report."""

    corpus_sha256 = verify_corpus(document)
    preregistration = AuditThresholds() if thresholds is None else thresholds
    if not isinstance(preregistration, AuditThresholds):
        raise TypeError("thresholds must be an AuditThresholds instance")
    payload = dict(document)
    payload.pop("payload_sha256")
    normalized = _normalize_corpus_payload(payload)
    records = cast(list[Mapping[str, object]], normalized["records"])
    if len(records) < preregistration.min_records:
        raise OutputAuditError(
            f"corpus has {len(records)} records; preregistered minimum is "
            f"{preregistration.min_records}"
        )
    clusters = {cast(str, record["cluster_id"]) for record in records}
    if len(clusters) < preregistration.min_clusters:
        raise OutputAuditError(
            f"corpus has {len(clusters)} clusters; preregistered minimum is "
            f"{preregistration.min_clusters}"
        )
    diagnostics = [_record_diagnostic(record, top_k=preregistration.top_k) for record in records]
    aggregate = _summarize(diagnostics)
    slices = _slice_diagnostics(diagnostics)
    per_test_alpha = preregistration.familywise_alpha / INFERENTIAL_FAMILY_SIZE
    adjusted_confidence = 1.0 - per_test_alpha
    mismatch = cast(Mapping[str, object], aggregate["greedy_mismatch"])
    acceptance = cast(Mapping[str, object], aggregate["acceptance_divergence"])
    mismatch_interval = exact_binomial_upper(
        cast(int, mismatch["count"]),
        len(diagnostics),
        alpha=per_test_alpha,
    )
    acceptance_interval = exact_binomial_upper(
        cast(int, acceptance["count"]),
        len(diagnostics),
        alpha=per_test_alpha,
    )
    tv_bootstrap = one_sample_cluster_mean_interval(
        _cluster_values(diagnostics, "total_variation"),
        confidence_level=adjusted_confidence,
        resamples=preregistration.bootstrap_resamples,
        seed=f"fissionspec-output-audit/{corpus_sha256}/total-variation/one-sample-v1",
    ).as_dict()
    js_bootstrap = one_sample_cluster_mean_interval(
        _cluster_values(diagnostics, "jensen_shannon"),
        confidence_level=adjusted_confidence,
        resamples=preregistration.bootstrap_resamples,
        seed=f"fissionspec-output-audit/{corpus_sha256}/jensen-shannon/one-sample-v1",
    ).as_dict()
    uncertainty = {
        "family": {
            "method": "bonferroni",
            "familywise_alpha": preregistration.familywise_alpha,
            "inferential_tests": INFERENTIAL_FAMILY_SIZE,
            "per_test_alpha": per_test_alpha,
            "simultaneous_confidence_level": adjusted_confidence,
            "members": [
                "greedy_mismatch_rate_upper",
                "acceptance_divergence_rate_upper",
                "mean_total_variation_upper",
                "mean_jensen_shannon_upper",
            ],
        },
        "greedy_mismatch_rate": mismatch_interval,
        "acceptance_divergence_rate": acceptance_interval,
        "mean_total_variation": tv_bootstrap,
        "mean_jensen_shannon": js_bootstrap,
    }
    violations = _summary_violations(
        aggregate,
        preregistration,
        scope="aggregate",
        include_rates=True,
    )
    if cast(float, mismatch_interval["upper"]) > preregistration.max_greedy_mismatch_upper:
        violations.append(
            _violation(
                "greedy_mismatch_rate_upper",
                observed=mismatch_interval["upper"],
                relation="<=",
                limit=preregistration.max_greedy_mismatch_upper,
            )
        )
    if cast(float, acceptance_interval["upper"]) > preregistration.max_acceptance_divergence_upper:
        violations.append(
            _violation(
                "acceptance_divergence_rate_upper",
                observed=acceptance_interval["upper"],
                relation="<=",
                limit=preregistration.max_acceptance_divergence_upper,
            )
        )
    if cast(float, tv_bootstrap["upper"]) > preregistration.max_mean_tv_upper:
        violations.append(
            _violation(
                "mean_tv_upper",
                observed=tv_bootstrap["upper"],
                relation="<=",
                limit=preregistration.max_mean_tv_upper,
            )
        )
    if cast(float, js_bootstrap["upper"]) > preregistration.max_mean_js_upper:
        violations.append(
            _violation(
                "mean_js_upper",
                observed=js_bootstrap["upper"],
                relation="<=",
                limit=preregistration.max_mean_js_upper,
            )
        )
    for sliced in slices:
        scope = f"{sliced['dimension']}={sliced['label']}"
        violations.extend(
            _summary_violations(
                cast(Mapping[str, object], sliced["summary"]),
                preregistration,
                scope=scope,
                include_rates=False,
            )
        )
    violations.sort(key=lambda row: (cast(str, row["scope"]), cast(str, row["code"])))
    threshold_document = preregistration.as_dict()
    report_payload: dict[str, object] = {
        "schema": REPORT_SCHEMA,
        "evidence_class": normalized["evidence_class"],
        "measurement_warning": normalized["measurement_warning"],
        "corpus_payload_sha256": corpus_sha256,
        "thresholds": threshold_document,
        "thresholds_sha256": sha256_document(threshold_document),
        "aggregate": aggregate,
        "slices": slices,
        "uncertainty": uncertainty,
        "records": diagnostics,
        "gate": {
            "status": "pass" if not violations else "fail",
            "violation_count": len(violations),
            "violations": violations,
        },
    }
    return {
        **report_payload,
        "payload_sha256": sha256_document(report_payload),
    }


def verify_report(document: Mapping[str, object]) -> str:
    """Verify the immutable report envelope and its embedded threshold digest."""

    supplied = _digest(document.get("payload_sha256"), field="payload_sha256")
    payload = dict(document)
    payload.pop("payload_sha256", None)
    _strict_keys(
        payload,
        {
            "schema",
            "evidence_class",
            "measurement_warning",
            "corpus_payload_sha256",
            "thresholds",
            "thresholds_sha256",
            "aggregate",
            "slices",
            "uncertainty",
            "records",
            "gate",
        },
        field="report payload",
    )
    if payload.get("schema") != REPORT_SCHEMA:
        raise OutputAuditIntegrityError("unsupported output-audit report schema")
    actual = sha256_document(payload)
    if supplied != actual:
        raise OutputAuditIntegrityError("report payload hash mismatch")
    evidence = _nonempty_string(payload.get("evidence_class"), field="evidence_class")
    if payload.get("measurement_warning") != _warning(evidence):
        raise OutputAuditIntegrityError("report measurement warning is missing or incorrect")
    _digest(payload.get("corpus_payload_sha256"), field="corpus_payload_sha256")
    thresholds = _mapping(payload.get("thresholds"), field="thresholds")
    AuditThresholds.from_mapping(thresholds)
    expected_threshold_hash = sha256_document(thresholds)
    if payload.get("thresholds_sha256") != expected_threshold_hash:
        raise OutputAuditIntegrityError("report threshold hash mismatch")
    _mapping(payload.get("aggregate"), field="aggregate")
    _sequence(payload.get("slices"), field="slices")
    _mapping(payload.get("uncertainty"), field="uncertainty")
    _sequence(payload.get("records"), field="records")
    gate = _mapping(payload.get("gate"), field="gate")
    _strict_keys(gate, {"status", "violation_count", "violations"}, field="gate")
    violations = _sequence(gate.get("violations"), field="gate.violations")
    count = _integer(gate.get("violation_count"), field="gate.violation_count", minimum=0)
    if count != len(violations):
        raise OutputAuditIntegrityError("gate.violation_count does not match violations")
    expected_status = "pass" if count == 0 else "fail"
    if gate.get("status") != expected_status:
        raise OutputAuditIntegrityError("report gate status is invalid")
    return actual


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise OutputAuditIntegrityError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _read_json(path: str | Path) -> dict[str, object]:
    try:
        raw = json.loads(
            Path(path).read_text(encoding="utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                OutputAuditIntegrityError(f"non-standard JSON numeric constant: {value}")
            ),
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise OutputAuditIntegrityError(f"cannot read strict JSON document: {path}") from error
    if not isinstance(raw, dict):
        raise OutputAuditIntegrityError("document root must be an object")
    return cast(dict[str, object], raw)


def load_corpus(path: str | Path) -> dict[str, object]:
    """Read and verify a corpus from strict UTF-8 JSON."""

    document = _read_json(path)
    verify_corpus(document)
    return document


def load_report(path: str | Path) -> dict[str, object]:
    """Read and verify an audit report from strict UTF-8 JSON."""

    document = _read_json(path)
    verify_report(document)
    return document


def write_document(path: str | Path, document: Mapping[str, object]) -> None:
    """Atomically write a verified corpus or report in canonical JSON."""

    schema = document.get("schema")
    if schema == CORPUS_SCHEMA:
        verify_corpus(document)
    elif schema == REPORT_SCHEMA:
        verify_report(document)
    else:
        raise OutputAuditIntegrityError("cannot write an unsupported document schema")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_bytes(canonical_json_bytes(document))
    temporary.replace(destination)


def _fixture_hash(label: str) -> str:
    return hashlib.sha256(f"fissionspec-output-audit-fixture/{label}/v1".encode()).hexdigest()


def _fixture_uniform(index: int) -> float:
    digest = hashlib.sha256(f"uniform/{index}/v1".encode()).digest()
    return int.from_bytes(digest[:8], "big") / (1 << 64)


def generate_synthetic_fixture(record_count: int = 512) -> dict[str, object]:
    """Generate a deterministic no-download CPU corpus with an exact parity null."""

    count = _integer(record_count, field="record_count", minimum=1)
    records: list[Mapping[str, object]] = []
    vocabulary = 17
    for index in range(count):
        logits = [
            0.7 * math.sin((index + 1) * (token + 2) * 0.031)
            + 0.3 * math.cos((index + 3) * (token + 1) * 0.017)
            + token * 0.001
            for token in range(vocabulary)
        ]
        distribution: Mapping[str, object] = {
            "encoding": "logits",
            "token_ids": list(range(vocabulary)),
            "values": logits,
        }
        proposed = (index * 7 + 3) % vocabulary
        maximum = max(logits)
        weights = [math.exp(value - maximum) for value in logits]
        proposed_probability = weights[proposed] / math.fsum(weights)
        records.append(
            {
                "record_id": f"position-{index:06d}",
                "cluster_id": f"trace-{index // 8:06d}",
                "slices": {
                    "batch_bucket": ("1", "8", "32", "128")[index % 4],
                    "phase": "decode" if index % 3 else "prefill",
                    "synthetic_dtype": "bf16-like" if index % 2 else "fp16-like",
                },
                "reference": distribution,
                "candidate": distribution,
                "proposed_token_id": proposed,
                "draft_probability": min(1.0, proposed_probability * 1.25 + 0.01),
                "uniform": _fixture_uniform(index),
            }
        )
    capture = {
        "capture_id": "deterministic-no-download-cpu-fixture-v1",
        "model_id": "synthetic-trigonometric-logit-generator-v1",
        "tokenizer_sha256": _fixture_hash("tokenizer"),
        "reference_engine_sha256": _fixture_hash("reference-engine"),
        "candidate_engine_sha256": _fixture_hash("candidate-engine"),
        "capture_config_sha256": _fixture_hash("capture-config"),
        "capture_tool_sha256": _fixture_hash("capture-tool"),
    }
    return build_corpus(
        evidence_class=SYNTHETIC_EVIDENCE,
        capture=capture,
        records=records,
    )


__all__ = [
    "CAPTURED_EVIDENCE",
    "CAPTURED_WARNING",
    "CORPUS_SCHEMA",
    "REPORT_SCHEMA",
    "SYNTHETIC_EVIDENCE",
    "SYNTHETIC_WARNING",
    "AuditThresholds",
    "OutputAuditError",
    "OutputAuditIntegrityError",
    "audit_corpus",
    "build_corpus",
    "exact_binomial_upper",
    "generate_synthetic_fixture",
    "load_corpus",
    "load_report",
    "verify_corpus",
    "verify_report",
    "write_document",
]
