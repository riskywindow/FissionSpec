"""Adversarial and metamorphic tests for the offline production-output audit."""

from __future__ import annotations

import copy
import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from typing import cast

from fissionspec.artifacts import sha256_document
from fissionspec.output_audit import (
    SYNTHETIC_EVIDENCE,
    SYNTHETIC_WARNING,
    AuditThresholds,
    OutputAuditError,
    OutputAuditIntegrityError,
    audit_corpus,
    build_corpus,
    exact_binomial_upper,
    generate_synthetic_fixture,
    load_corpus,
    load_report,
    verify_corpus,
    verify_report,
    write_document,
)

ROOT = Path(__file__).resolve().parents[1]


def _hash(label: str) -> str:
    import hashlib

    return hashlib.sha256(label.encode()).hexdigest()


def _capture() -> dict[str, object]:
    return {
        "capture_id": "unit-test-capture",
        "model_id": "unit-test-model",
        "tokenizer_sha256": _hash("tokenizer"),
        "reference_engine_sha256": _hash("reference"),
        "candidate_engine_sha256": _hash("candidate"),
        "capture_config_sha256": _hash("config"),
        "capture_tool_sha256": _hash("tool"),
    }


def _distribution(
    values: list[float],
    *,
    encoding: str = "logits",
    tokens: list[int] | None = None,
) -> dict[str, object]:
    return {
        "encoding": encoding,
        "token_ids": list(range(len(values))) if tokens is None else tokens,
        "values": values,
    }


def _record(
    index: int,
    *,
    reference: dict[str, object] | None = None,
    candidate: dict[str, object] | None = None,
    proposed_token_id: int = 0,
    draft_probability: float = 0.5,
    uniform: float = 0.25,
) -> dict[str, object]:
    base = _distribution([2.0, 1.0, 0.0])
    return {
        "record_id": f"record-{index:04d}",
        "cluster_id": f"cluster-{index // 2:04d}",
        "slices": {
            "batch": "small" if index % 2 else "large",
            "phase": "decode",
        },
        "reference": base if reference is None else reference,
        "candidate": base if candidate is None else candidate,
        "proposed_token_id": proposed_token_id,
        "draft_probability": draft_probability,
        "uniform": uniform,
    }


def _corpus(records: list[dict[str, object]]) -> dict[str, object]:
    return build_corpus(
        evidence_class=SYNTHETIC_EVIDENCE,
        capture=_capture(),
        records=records,
    )


def _liberal_thresholds(records: int = 16) -> AuditThresholds:
    return replace(
        AuditThresholds(),
        min_records=records,
        min_clusters=2,
        bootstrap_resamples=100,
        max_greedy_mismatch_rate=1.0,
        max_greedy_mismatch_upper=1.0,
        max_acceptance_divergence_rate=1.0,
        max_acceptance_divergence_upper=1.0,
        max_mean_tv=1.0,
        max_mean_tv_upper=1.0,
        max_record_tv=1.0,
        max_mean_js=1.0,
        max_mean_js_upper=1.0,
        max_record_js=1.0,
        max_forward_kl=100.0,
        max_reverse_kl=100.0,
        min_mean_top_k_overlap=0.0,
        max_mean_greedy_rank_drift=100.0,
        max_mean_margin_drift=1.0,
    )


def _aggregate(report: dict[str, object]) -> dict[str, object]:
    return cast(dict[str, object], report["aggregate"])


def _gate(report: dict[str, object]) -> dict[str, object]:
    return cast(dict[str, object], report["gate"])


def _record_metrics(report: dict[str, object], index: int = 0) -> dict[str, object]:
    return cast(list[dict[str, object]], report["records"])[index]


class CorpusContractTests(unittest.TestCase):
    def test_build_is_canonical_under_record_and_token_permutation(self) -> None:
        records = [_record(index) for index in range(8)]
        first = _corpus(records)
        permuted = copy.deepcopy(records)
        permuted.reverse()
        for record in permuted:
            for side in ("reference", "candidate"):
                distribution = cast(dict[str, object], record[side])
                tokens = cast(list[int], distribution["token_ids"])
                values = cast(list[float], distribution["values"])
                distribution["token_ids"] = list(reversed(tokens))
                distribution["values"] = list(reversed(values))
        second = _corpus(permuted)
        self.assertEqual(first, second)
        self.assertEqual(verify_corpus(first), first["payload_sha256"])

    def test_hash_tampering_and_rehashed_noncanonical_order_are_rejected(self) -> None:
        corpus = _corpus([_record(index) for index in range(4)])
        tampered = copy.deepcopy(corpus)
        records = cast(list[dict[str, object]], tampered["records"])
        candidate = cast(dict[str, object], records[0]["candidate"])
        cast(list[float], candidate["values"])[0] += 0.01
        with self.assertRaisesRegex(OutputAuditIntegrityError, "hash mismatch"):
            verify_corpus(tampered)

        reordered = copy.deepcopy(corpus)
        cast(list[object], reordered["records"]).reverse()
        payload = dict(reordered)
        payload.pop("payload_sha256")
        reordered["payload_sha256"] = sha256_document(payload)
        with self.assertRaisesRegex(OutputAuditIntegrityError, "canonical"):
            verify_corpus(reordered)

    def test_malformed_nonfinite_duplicate_and_bad_probability_inputs_fail(self) -> None:
        invalid_distributions = (
            _distribution([math.nan, 0.0]),
            _distribution([math.inf, 0.0]),
            _distribution([10**10_000, 0.0]),
            _distribution([0.5, 0.4], encoding="probabilities"),
            _distribution([0.5, -0.5], encoding="probabilities"),
            _distribution([0.5, 0.5], tokens=[1, 1]),
        )
        for distribution in invalid_distributions:
            with self.subTest(distribution=distribution), self.assertRaises(OutputAuditError):
                _corpus(
                    [
                        _record(
                            0,
                            reference=distribution,
                            candidate=_distribution([0.5, 0.5]),
                        )
                    ]
                )

    def test_capture_hashes_schema_and_logit_support_fail_closed(self) -> None:
        capture = _capture()
        capture["capture_tool_sha256"] = "not-a-hash"
        with self.assertRaises(OutputAuditError):
            build_corpus(
                evidence_class=SYNTHETIC_EVIDENCE,
                capture=capture,
                records=[_record(0)],
            )
        with self.assertRaisesRegex(OutputAuditError, "same token support"):
            _corpus(
                [
                    _record(
                        0,
                        reference=_distribution([1.0, 0.0], tokens=[0, 1]),
                        candidate=_distribution([1.0, 0.0], tokens=[0, 2]),
                    )
                ]
            )

    def test_strict_file_loading_rejects_nonstandard_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text('{"schema": NaN}', encoding="utf-8")
            with self.assertRaises(OutputAuditIntegrityError):
                load_corpus(path)
            path.write_text('{"schema":"first","schema":"second"}', encoding="utf-8")
            with self.assertRaisesRegex(OutputAuditIntegrityError, "duplicate"):
                load_corpus(path)


class MetricMetamorphicTests(unittest.TestCase):
    def test_identical_null_is_exact_and_report_is_deterministic(self) -> None:
        corpus = _corpus([_record(index) for index in range(16)])
        thresholds = _liberal_thresholds()
        first = audit_corpus(corpus, thresholds=thresholds)
        second = audit_corpus(corpus, thresholds=thresholds)
        self.assertEqual(first, second)
        self.assertEqual(_gate(first)["status"], "pass")
        aggregate = _aggregate(first)
        self.assertEqual(
            cast(dict[str, object], aggregate["greedy_mismatch"])["count"],
            0,
        )
        self.assertEqual(
            cast(dict[str, object], aggregate["acceptance_divergence"])["count"],
            0,
        )
        self.assertEqual(cast(dict[str, object], aggregate["total_variation"])["max"], 0.0)
        self.assertEqual(cast(dict[str, object], aggregate["jensen_shannon"])["max"], 0.0)
        self.assertEqual(first["measurement_warning"], SYNTHETIC_WARNING)
        self.assertEqual(verify_report(first), first["payload_sha256"])

    def test_logit_translation_and_token_permutation_leave_metrics_invariant(self) -> None:
        reference = [2.0, 1.0, -1.0, -3.0]
        candidate = [2.0, 1.00000001, -1.0, -3.0]
        base = _corpus(
            [
                _record(
                    index,
                    reference=_distribution(reference),
                    candidate=_distribution(candidate),
                )
                for index in range(16)
            ]
        )
        permutation = [2, 0, 3, 1]
        translated = _corpus(
            [
                _record(
                    index,
                    reference=_distribution(
                        [reference[token] + 19.25 for token in permutation],
                        tokens=permutation,
                    ),
                    candidate=_distribution(
                        [candidate[token] + 19.25 for token in permutation],
                        tokens=permutation,
                    ),
                )
                for index in range(16)
            ]
        )
        first = _record_metrics(audit_corpus(base, thresholds=_liberal_thresholds()))
        second = _record_metrics(audit_corpus(translated, thresholds=_liberal_thresholds()))
        for field in (
            "total_variation",
            "jensen_shannon",
            "forward_kl",
            "reverse_kl",
            "top_k_overlap",
            "greedy_rank_drift",
            "absolute_margin_drift",
        ):
            self.assertAlmostEqual(
                cast(float, first[field]),
                cast(float, second[field]),
                places=14,
            )
        self.assertEqual(first["reference_greedy_token_id"], second["reference_greedy_token_id"])
        self.assertEqual(first["candidate_greedy_token_id"], second["candidate_greedy_token_id"])

    def test_stable_softmax_handles_extreme_finite_logits(self) -> None:
        distribution = _distribution([10_000.0, 9_999.0, -10_000.0])
        report = audit_corpus(
            _corpus(
                [
                    _record(index, reference=distribution, candidate=distribution)
                    for index in range(16)
                ]
            ),
            thresholds=_liberal_thresholds(),
        )
        self.assertEqual(_record_metrics(report)["total_variation"], 0.0)
        self.assertEqual(_gate(report)["status"], "pass")

    def test_tiny_logit_perturbation_is_measured_without_token_or_decision_drift(self) -> None:
        records = [
            _record(
                index,
                reference=_distribution([3.0, 1.0, -2.0]),
                candidate=_distribution([3.0 + 1e-10, 1.0, -2.0]),
                proposed_token_id=0,
                draft_probability=0.95,
                uniform=0.2,
            )
            for index in range(16)
        ]
        report = audit_corpus(_corpus(records), thresholds=_liberal_thresholds())
        metrics = _record_metrics(report)
        self.assertGreater(cast(float, metrics["total_variation"]), 0.0)
        self.assertFalse(metrics["greedy_mismatch"])
        self.assertFalse(metrics["acceptance_divergence"])
        self.assertEqual(_gate(report)["status"], "pass")

    def test_ties_use_lowest_token_id_for_deterministic_greedy_choice(self) -> None:
        tied = _distribution([0.0, 0.0], tokens=[9, 4])
        report = audit_corpus(
            _corpus([_record(index, reference=tied, candidate=tied) for index in range(16)]),
            thresholds=_liberal_thresholds(),
        )
        self.assertEqual(_record_metrics(report)["reference_greedy_token_id"], 4)


class AdversarialGateTests(unittest.TestCase):
    def test_greedy_mismatch_fails_zero_tolerance_gate(self) -> None:
        records = [
            _record(
                index,
                reference=_distribution([2.0, 1.0]),
                candidate=_distribution([1.0, 2.0]),
            )
            for index in range(16)
        ]
        thresholds = replace(
            _liberal_thresholds(),
            max_greedy_mismatch_rate=0.0,
        )
        report = audit_corpus(_corpus(records), thresholds=thresholds)
        codes = {row["code"] for row in cast(list[dict[str, object]], _gate(report)["violations"])}
        self.assertEqual(_gate(report)["status"], "fail")
        self.assertIn("greedy_mismatch_rate", codes)

    def test_prerecorded_uniform_exposes_accept_reject_divergence(self) -> None:
        reference = _distribution([0.8, 0.2], encoding="probabilities")
        candidate = _distribution([0.2, 0.8], encoding="probabilities")
        records = [
            _record(
                index,
                reference=reference,
                candidate=candidate,
                proposed_token_id=0,
                draft_probability=0.8,
                uniform=0.5,
            )
            for index in range(16)
        ]
        thresholds = replace(
            _liberal_thresholds(),
            max_acceptance_divergence_rate=0.0,
        )
        report = audit_corpus(_corpus(records), thresholds=thresholds)
        metrics = _record_metrics(report)
        self.assertTrue(metrics["reference_accept"])
        self.assertFalse(metrics["candidate_accept"])
        self.assertTrue(metrics["acceptance_divergence"])
        codes = {row["code"] for row in cast(list[dict[str, object]], _gate(report)["violations"])}
        self.assertIn("acceptance_divergence_rate", codes)

    def test_catastrophic_rare_tail_zero_is_not_averaged_away(self) -> None:
        ordinary = [_record(index) for index in range(15)]
        rare = _record(
            15,
            reference=_distribution([0.999999, 0.000001], encoding="probabilities"),
            candidate=_distribution([1.0, 0.0], encoding="probabilities"),
        )
        report = audit_corpus(
            _corpus([*ordinary, rare]),
            thresholds=_liberal_thresholds(),
        )
        metrics = _record_metrics(report, 15)
        self.assertTrue(metrics["forward_kl_infinite"])
        self.assertIsNone(metrics["forward_kl"])
        codes = {row["code"] for row in cast(list[dict[str, object]], _gate(report)["violations"])}
        self.assertIn("forward_kl_infinite", codes)
        self.assertEqual(_gate(report)["status"], "fail")

    def test_sparse_support_zeros_have_directional_kl_and_finite_js(self) -> None:
        reference = _distribution([1.0, 0.0], encoding="probabilities", tokens=[0, 1])
        candidate = _distribution([0.5, 0.5], encoding="probabilities", tokens=[0, 2])
        report = audit_corpus(
            _corpus(
                [_record(index, reference=reference, candidate=candidate) for index in range(16)]
            ),
            thresholds=_liberal_thresholds(),
        )
        metrics = _record_metrics(report)
        self.assertFalse(metrics["forward_kl_infinite"])
        self.assertTrue(metrics["reverse_kl_infinite"])
        self.assertAlmostEqual(cast(float, metrics["forward_kl"]), math.log(2.0))
        self.assertTrue(math.isfinite(cast(float, metrics["jensen_shannon"])))

    def test_slice_diagnostics_catch_a_slice_even_when_rate_checks_are_aggregate(self) -> None:
        records = [_record(index) for index in range(16)]
        for record in records:
            if cast(dict[str, str], record["slices"])["batch"] == "small":
                record["candidate"] = _distribution([2.0, 1.01, 0.0])
        thresholds = replace(
            _liberal_thresholds(),
            max_mean_tv=0.001,
            max_record_tv=1.0,
            max_mean_tv_upper=1.0,
        )
        report = audit_corpus(_corpus(records), thresholds=thresholds)
        scopes = {
            row["scope"] for row in cast(list[dict[str, object]], _gate(report)["violations"])
        }
        self.assertIn("batch=small", scopes)


class StatisticalAndPreregistrationTests(unittest.TestCase):
    def test_exact_zero_event_upper_matches_closed_form(self) -> None:
        interval = exact_binomial_upper(0, 512, alpha=0.0125)
        expected = 1.0 - 0.0125 ** (1.0 / 512)
        self.assertAlmostEqual(cast(float, interval["upper"]), expected, places=14)

    def test_uncertainty_family_is_explicit_and_bonferroni_adjusted(self) -> None:
        report = audit_corpus(
            _corpus([_record(index) for index in range(16)]),
            thresholds=_liberal_thresholds(),
        )
        uncertainty = cast(dict[str, object], report["uncertainty"])
        family = cast(dict[str, object], uncertainty["family"])
        self.assertEqual(family["method"], "bonferroni")
        self.assertEqual(family["inferential_tests"], 4)
        self.assertAlmostEqual(cast(float, family["per_test_alpha"]), 0.0125)
        self.assertEqual(len(cast(list[str], family["members"])), 4)

    def test_threshold_mapping_is_complete_strict_and_hash_locked(self) -> None:
        original = AuditThresholds()
        self.assertEqual(AuditThresholds.from_mapping(original.as_dict()), original)
        missing = original.as_dict()
        missing.pop("top_k")
        with self.assertRaises(OutputAuditError):
            AuditThresholds.from_mapping(missing)
        extra = {**original.as_dict(), "unregistered_escape_hatch": True}
        with self.assertRaises(OutputAuditError):
            AuditThresholds.from_mapping(extra)
        with self.assertRaises(OutputAuditError):
            replace(original, familywise_alpha=0.9)

    def test_undersized_corpus_and_cluster_count_fail_before_analysis(self) -> None:
        corpus = _corpus([_record(index) for index in range(4)])
        with self.assertRaisesRegex(OutputAuditError, "records"):
            audit_corpus(corpus)
        thresholds = replace(
            _liberal_thresholds(records=4),
            min_clusters=4,
        )
        with self.assertRaisesRegex(OutputAuditError, "clusters"):
            audit_corpus(corpus, thresholds=thresholds)

    def test_report_payload_and_embedded_threshold_hash_detect_tampering(self) -> None:
        report = audit_corpus(
            _corpus([_record(index) for index in range(16)]),
            thresholds=_liberal_thresholds(),
        )
        outer_tamper = copy.deepcopy(report)
        cast(dict[str, object], outer_tamper["gate"])["status"] = "fail"
        with self.assertRaisesRegex(OutputAuditIntegrityError, "payload hash"):
            verify_report(outer_tamper)

        inner_tamper = copy.deepcopy(report)
        cast(dict[str, object], inner_tamper["thresholds"])["top_k"] = 2
        payload = dict(inner_tamper)
        payload.pop("payload_sha256")
        inner_tamper["payload_sha256"] = sha256_document(payload)
        with self.assertRaisesRegex(OutputAuditIntegrityError, "threshold hash"):
            verify_report(inner_tamper)

        inconsistent_gate = copy.deepcopy(report)
        cast(dict[str, object], inconsistent_gate["gate"])["violation_count"] = 1
        payload = dict(inconsistent_gate)
        payload.pop("payload_sha256")
        inconsistent_gate["payload_sha256"] = sha256_document(payload)
        with self.assertRaisesRegex(OutputAuditIntegrityError, "violation_count"):
            verify_report(inconsistent_gate)


class FixtureAndCliTests(unittest.TestCase):
    def test_default_synthetic_fixture_passes_but_carries_non_parity_warning(self) -> None:
        corpus = generate_synthetic_fixture()
        report = audit_corpus(corpus)
        self.assertEqual(_gate(report)["status"], "pass")
        self.assertEqual(report["measurement_warning"], SYNTHETIC_WARNING)
        self.assertEqual(cast(dict[str, object], report["aggregate"])["records"], 512)

    def test_documents_round_trip_and_cli_runs_fully_offline(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            environment = dict(os.environ)
            environment["PYTHONPATH"] = str(ROOT / "src")
            completed = subprocess.run(
                [
                    sys.executable,
                    "tools/run_output_audit.py",
                    "fixture",
                    str(output),
                ],
                cwd=ROOT,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            summary = json.loads(completed.stdout)
            self.assertEqual(summary["status"], "pass")
            corpus_path = output / "synthetic_output_corpus.json"
            report_path = output / "synthetic_output_audit_report.json"
            corpus = load_corpus(corpus_path)
            report = load_report(report_path)
            self.assertEqual(corpus["payload_sha256"], summary["corpus_payload_sha256"])
            self.assertEqual(report["payload_sha256"], summary["report_payload_sha256"])

            verified = subprocess.run(
                [
                    sys.executable,
                    "tools/run_output_audit.py",
                    "verify",
                    str(report_path),
                ],
                cwd=ROOT,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(verified.returncode, 0, verified.stderr)
            self.assertEqual(json.loads(verified.stdout)["status"], "verified")

    def test_atomic_writer_refuses_unknown_schema(self) -> None:
        with (
            tempfile.TemporaryDirectory() as directory,
            self.assertRaises(OutputAuditIntegrityError),
        ):
            write_document(Path(directory) / "bad.json", {"schema": "unknown"})


if __name__ == "__main__":
    unittest.main()
