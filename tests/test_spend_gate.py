"""Fail-closed accelerator campaign and spend-cap tests."""

from __future__ import annotations

import hashlib
import json
import unittest
from dataclasses import replace
from typing import cast

from fissionspec.spend_gate import (
    CampaignLedger,
    CampaignPlan,
    CampaignStage,
    GateRecord,
    GateVerdict,
    StageBudget,
)

HASH = "a" * 64


def _plan() -> CampaignPlan:
    return CampaignPlan(
        protocol_sha256="1" * 64,
        code_commit="2" * 40,
        cpu_artifact_sha256="3" * 64,
        validation_trace_hashes=(
            ("V1", "4" * 64),
            ("V2", "5" * 64),
            ("V3", "6" * 64),
        ),
        planned_primary_replays=1200,
        planned_unique_ablation_replays=300,
        planned_robustness_cells=12,
    )


def _record(
    stage: CampaignStage,
    verdict: GateVerdict = GateVerdict.PASS,
    *,
    seconds: int = 0,
    replays: int = 0,
) -> GateRecord:
    return GateRecord(stage, verdict, seconds, replays, HASH)


def _budget(stage: CampaignStage, *, seconds: int = 100, replays: int = 10) -> StageBudget:
    return StageBudget(stage, seconds, replays, HASH)


class CampaignPlanTests(unittest.TestCase):
    def test_plan_hash_is_canonical_and_covers_registered_counts(self) -> None:
        first = _plan()
        second = _plan()
        self.assertEqual(first.campaign_id, second.campaign_id)
        self.assertEqual(len(first.campaign_id), 64)
        changed = replace(first, planned_primary_replays=1199)
        self.assertNotEqual(first.campaign_id, changed.campaign_id)
        document = json.loads(json.dumps(first.document()))
        self.assertEqual(CampaignPlan.from_document(document), first)

    def test_plan_document_rejects_rehashed_campaign_id(self) -> None:
        document = json.loads(json.dumps(_plan().document()))
        document["campaign_id"] = "f" * 64
        payload = {key: value for key, value in document.items() if key != "payload_sha256"}
        document["payload_sha256"] = hashlib.sha256(
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode()
        ).hexdigest()
        with self.assertRaisesRegex(ValueError, "campaign plan ID"):
            CampaignPlan.from_document(document)

    def test_bad_hash_anchor_order_and_replay_caps_fail_closed(self) -> None:
        base = {
            "protocol_sha256": "1" * 64,
            "code_commit": "2" * 40,
            "cpu_artifact_sha256": "3" * 64,
            "validation_trace_hashes": (
                ("V1", "4" * 64),
                ("V2", "5" * 64),
                ("V3", "6" * 64),
            ),
            "planned_primary_replays": 1200,
            "planned_unique_ablation_replays": 300,
            "planned_robustness_cells": 12,
        }
        trace_hashes = cast(
            tuple[tuple[str, str], ...],
            base["validation_trace_hashes"],
        )
        for override in (
            {"protocol_sha256": "not-a-hash"},
            {"validation_trace_hashes": tuple(reversed(trace_hashes))},
            {"planned_primary_replays": 1201},
            {"planned_unique_ablation_replays": 301},
            {"planned_robustness_cells": 13},
        ):
            with self.subTest(override=override), self.assertRaises(ValueError):
                CampaignPlan(**{**base, **override})  # type: ignore[arg-type]


class CampaignLedgerTests(unittest.TestCase):
    def test_happy_path_never_pre_authorizes_a_future_stage(self) -> None:
        ledger = CampaignLedger(_plan())
        self.assertEqual(ledger.next_stage, CampaignStage.CPU_RELEASE)
        self.assertEqual(ledger.currently_authorized_gpu_seconds, 0)
        ledger = ledger.record_gate(_record(CampaignStage.CPU_RELEASE))
        for stage in (
            CampaignStage.ROW_OMISSION,
            CampaignStage.MECHANISM,
            CampaignStage.CONTROLLER,
            CampaignStage.H100_CONFIRMATORY,
            CampaignStage.B200_TRANSPORT,
        ):
            self.assertEqual(ledger.next_stage, stage)
            self.assertEqual(ledger.currently_authorized_gpu_seconds, 0)
            budget = _budget(stage)
            ledger = ledger.seal_next_budget(budget)
            self.assertEqual(ledger.currently_authorized_gpu_seconds, 100)
            ledger = ledger.record_gate(
                _record(stage, seconds=80, replays=8),
            )
        self.assertIsNone(ledger.next_stage)
        self.assertEqual(ledger.spent_gpu_seconds, 400)
        self.assertEqual(ledger.currently_authorized_gpu_seconds, 0)

    def test_failed_physical_gate_permanently_locks_later_spend(self) -> None:
        ledger = CampaignLedger(_plan()).record_gate(_record(CampaignStage.CPU_RELEASE))
        ledger = ledger.seal_next_budget(_budget(CampaignStage.ROW_OMISSION))
        ledger = ledger.record_gate(
            _record(
                CampaignStage.ROW_OMISSION,
                GateVerdict.FAIL,
                seconds=25,
                replays=4,
            )
        )
        self.assertIsNone(ledger.next_stage)
        self.assertEqual(ledger.currently_authorized_gpu_seconds, 0)
        with self.assertRaises(ValueError):
            ledger.seal_next_budget(_budget(CampaignStage.MECHANISM))
        with self.assertRaises(ValueError):
            ledger.record_gate(_record(CampaignStage.MECHANISM))

    def test_skip_unsealed_over_cap_and_budget_replacement_are_rejected(self) -> None:
        ledger = CampaignLedger(_plan())
        with self.assertRaises(ValueError):
            ledger.record_gate(_record(CampaignStage.ROW_OMISSION))
        ledger = ledger.record_gate(_record(CampaignStage.CPU_RELEASE))
        with self.assertRaises(ValueError):
            ledger.record_gate(_record(CampaignStage.ROW_OMISSION, seconds=1, replays=1))
        budget = _budget(CampaignStage.ROW_OMISSION)
        ledger = ledger.seal_next_budget(budget)
        self.assertIs(ledger.seal_next_budget(budget), ledger)
        with self.assertRaises(ValueError):
            ledger.seal_next_budget(_budget(CampaignStage.ROW_OMISSION, seconds=101))
        with self.assertRaises(ValueError):
            ledger.record_gate(_record(CampaignStage.ROW_OMISSION, seconds=101, replays=1))
        with self.assertRaises(ValueError):
            ledger.record_gate(_record(CampaignStage.ROW_OMISSION, seconds=1, replays=11))

    def test_identical_record_replay_is_idempotent_but_mutation_is_not(self) -> None:
        record = _record(CampaignStage.CPU_RELEASE)
        ledger = CampaignLedger(_plan()).record_gate(record)
        self.assertIs(ledger.record_gate(record), ledger)
        with self.assertRaises(ValueError):
            ledger.record_gate(
                GateRecord(
                    CampaignStage.CPU_RELEASE,
                    GateVerdict.FAIL,
                    0,
                    0,
                    HASH,
                )
            )

    def test_document_is_canonical_self_hashed_and_order_sensitive(self) -> None:
        ledger = CampaignLedger(_plan()).record_gate(_record(CampaignStage.CPU_RELEASE))
        document = ledger.document()
        payload = {key: value for key, value in document.items() if key != "payload_sha256"}
        expected = hashlib.sha256(
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode()
        ).hexdigest()
        self.assertEqual(document["payload_sha256"], expected)
        self.assertEqual(document, ledger.document())
        self.assertEqual(
            CampaignLedger.from_document(json.loads(json.dumps(document))),
            ledger,
        )

    def test_document_decoder_rejects_unknown_tampered_and_derived_fields(self) -> None:
        ledger = CampaignLedger(_plan()).record_gate(_record(CampaignStage.CPU_RELEASE))
        document = ledger.document()
        with (
            self.subTest("unknown field"),
            self.assertRaisesRegex(ValueError, "unexpected or missing"),
        ):
            CampaignLedger.from_document({**document, "escape_hatch": True})

        tampered = json.loads(json.dumps(document))
        tampered["plan"]["planned_primary_replays"] = 1199
        with (
            self.subTest("stale hash"),
            self.assertRaisesRegex(ValueError, "payload hash mismatch"),
        ):
            CampaignLedger.from_document(tampered)

        inconsistent = json.loads(json.dumps(document))
        inconsistent["next_stage"] = CampaignStage.MECHANISM.value
        payload = {key: value for key, value in inconsistent.items() if key != "payload_sha256"}
        inconsistent["payload_sha256"] = hashlib.sha256(
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode()
        ).hexdigest()
        with (
            self.subTest("rehashed derived field"),
            self.assertRaisesRegex(ValueError, "derived fields"),
        ):
            CampaignLedger.from_document(inconsistent)

    def test_nested_document_decoders_are_closed(self) -> None:
        document = CampaignLedger(_plan()).document()
        plan = json.loads(json.dumps(document["plan"]))
        plan["unregistered"] = 1
        with self.assertRaisesRegex(ValueError, "unexpected or missing"):
            CampaignPlan.from_mapping(plan)
        with self.assertRaisesRegex(ValueError, "string pair"):
            encoded_plan = cast(dict[str, object], document["plan"])
            CampaignPlan.from_mapping(
                {
                    **encoded_plan,
                    "validation_trace_hashes": [["V1", "4" * 64, "extra"]],
                }
            )

    def test_cpu_gate_and_budget_value_validation(self) -> None:
        with self.assertRaises(ValueError):
            _record(CampaignStage.CPU_RELEASE, seconds=1)
        with self.assertRaises(ValueError):
            StageBudget(CampaignStage.CPU_RELEASE, 1, 1, HASH)
        with self.assertRaises(ValueError):
            _budget(CampaignStage.ROW_OMISSION, seconds=0)
        with self.assertRaises(ValueError):
            GateRecord(
                CampaignStage.ROW_OMISSION,
                GateVerdict.PASS,
                -1,
                1,
                HASH,
            )


if __name__ == "__main__":
    unittest.main()
