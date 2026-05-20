"""Fail-closed accelerator campaign ledger.

The module does not launch jobs or talk to a cloud provider.  It makes the
pre-registered stage order and resource caps executable so an orchestration
layer cannot silently skip a falsification gate or exceed the currently
authorized GPU budget.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from enum import StrEnum
from typing import cast

from fissionspec.experiment_design import ExperimentSpendCaps


class CampaignStage(StrEnum):
    """Strictly ordered release and accelerator gates."""

    CPU_RELEASE = "f0_cpu_release"
    ROW_OMISSION = "f1_physical_row_omission"
    MECHANISM = "f2_one_miss_mechanism"
    CONTROLLER = "f3_controller_transport"
    H100_CONFIRMATORY = "f4_h100_confirmatory"
    B200_TRANSPORT = "f5_b200_transport"


STAGE_ORDER = (
    CampaignStage.CPU_RELEASE,
    CampaignStage.ROW_OMISSION,
    CampaignStage.MECHANISM,
    CampaignStage.CONTROLLER,
    CampaignStage.H100_CONFIRMATORY,
    CampaignStage.B200_TRANSPORT,
)


class GateVerdict(StrEnum):
    """Terminal verdict for one stage."""

    PASS = "pass"
    FAIL = "fail"


def _non_negative_integer(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _digest(value: object, *, field: str, lengths: tuple[int, ...] = (64,)) -> str:
    if not isinstance(value, str) or len(value) not in lengths or value.lower() != value:
        allowed = " or ".join(str(length) for length in lengths)
        raise ValueError(f"{field} must be {allowed} lowercase hexadecimal characters")
    try:
        bytes.fromhex(value)
    except ValueError as exc:
        raise ValueError(f"{field} must be lowercase hexadecimal") from exc
    return value


def _canonical_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


@dataclass(frozen=True, slots=True)
class CampaignPlan:
    """Immutable hashes and replay counts frozen before Stage F1."""

    protocol_sha256: str
    code_commit: str
    cpu_artifact_sha256: str
    validation_trace_hashes: tuple[tuple[str, str], ...]
    planned_primary_replays: int
    planned_unique_ablation_replays: int
    planned_robustness_cells: int
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported campaign-plan schema_version")
        _digest(self.protocol_sha256, field="protocol_sha256")
        _digest(self.code_commit, field="code_commit", lengths=(40, 64))
        _digest(self.cpu_artifact_sha256, field="cpu_artifact_sha256")
        if tuple(sorted(self.validation_trace_hashes)) != self.validation_trace_hashes:
            raise ValueError("validation_trace_hashes must be sorted by anchor")
        anchors = tuple(anchor for anchor, _ in self.validation_trace_hashes)
        if anchors != ("V1", "V2", "V3"):
            raise ValueError("validation_trace_hashes must contain exactly V1, V2, and V3")
        for anchor, digest in self.validation_trace_hashes:
            _digest(digest, field=f"validation_trace_hashes[{anchor}]")
        ExperimentSpendCaps().validate_manifest_counts(
            primary_replays=_non_negative_integer(
                self.planned_primary_replays,
                field="planned_primary_replays",
            ),
            unique_ablation_replays=_non_negative_integer(
                self.planned_unique_ablation_replays,
                field="planned_unique_ablation_replays",
            ),
            robustness_cells=_non_negative_integer(
                self.planned_robustness_cells,
                field="planned_robustness_cells",
            ),
        )

    @property
    def campaign_id(self) -> str:
        return hashlib.sha256(_canonical_bytes(asdict(self))).hexdigest()


@dataclass(frozen=True, slots=True)
class StageBudget:
    """One sealed cap; integer GPU-seconds avoid rounding ambiguity."""

    stage: CampaignStage
    max_gpu_seconds: int
    max_replays: int
    rationale_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.stage, CampaignStage) or self.stage is CampaignStage.CPU_RELEASE:
            raise ValueError("only accelerator stages may have a StageBudget")
        _non_negative_integer(self.max_gpu_seconds, field="max_gpu_seconds")
        _non_negative_integer(self.max_replays, field="max_replays")
        if self.max_gpu_seconds == 0 or self.max_replays == 0:
            raise ValueError("an accelerator-stage budget must authorize positive bounded work")
        _digest(self.rationale_sha256, field="rationale_sha256")


@dataclass(frozen=True, slots=True)
class GateRecord:
    """Immutable result and resource accounting for a completed gate."""

    stage: CampaignStage
    verdict: GateVerdict
    used_gpu_seconds: int
    completed_replays: int
    evidence_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.stage, CampaignStage):
            raise TypeError("stage must be a CampaignStage")
        if not isinstance(self.verdict, GateVerdict):
            raise TypeError("verdict must be a GateVerdict")
        _non_negative_integer(self.used_gpu_seconds, field="used_gpu_seconds")
        _non_negative_integer(self.completed_replays, field="completed_replays")
        _digest(self.evidence_sha256, field="evidence_sha256")
        if self.stage is CampaignStage.CPU_RELEASE and (
            self.used_gpu_seconds != 0 or self.completed_replays != 0
        ):
            raise ValueError("the CPU release gate cannot consume GPU work")


@dataclass(frozen=True, slots=True)
class CampaignLedger:
    """Append-only, stage-ordered authorization and result ledger."""

    plan: CampaignPlan
    budgets: tuple[StageBudget, ...] = ()
    records: tuple[GateRecord, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.plan, CampaignPlan):
            raise TypeError("plan must be a CampaignPlan")
        budget_stages = tuple(budget.stage for budget in self.budgets)
        if len(budget_stages) != len(set(budget_stages)):
            raise ValueError("a stage budget may be sealed only once")
        if tuple(sorted(budget_stages, key=STAGE_ORDER.index)) != budget_stages:
            raise ValueError("budgets must be stored in stage order")
        record_stages = tuple(record.stage for record in self.records)
        if record_stages != STAGE_ORDER[: len(record_stages)]:
            raise ValueError("gate records must be a contiguous stage prefix")
        for index, record in enumerate(self.records):
            if index < len(self.records) - 1 and record.verdict is GateVerdict.FAIL:
                raise ValueError("no record may follow a failed gate")
            if record.stage is CampaignStage.CPU_RELEASE:
                continue
            budget = self._budget_for(record.stage)
            if budget is None:
                raise ValueError("every recorded accelerator gate needs a sealed budget")
            if record.used_gpu_seconds > budget.max_gpu_seconds:
                raise ValueError("recorded GPU seconds exceed the sealed stage cap")
            if record.completed_replays > budget.max_replays:
                raise ValueError("recorded replays exceed the sealed stage cap")

    def _budget_for(self, stage: CampaignStage) -> StageBudget | None:
        return next((budget for budget in self.budgets if budget.stage is stage), None)

    @property
    def next_stage(self) -> CampaignStage | None:
        if self.records and self.records[-1].verdict is GateVerdict.FAIL:
            return None
        if len(self.records) == len(STAGE_ORDER):
            return None
        return STAGE_ORDER[len(self.records)]

    @property
    def spent_gpu_seconds(self) -> int:
        return sum(record.used_gpu_seconds for record in self.records)

    @property
    def currently_authorized_gpu_seconds(self) -> int:
        stage = self.next_stage
        if stage is None or stage is CampaignStage.CPU_RELEASE:
            return 0
        budget = self._budget_for(stage)
        return 0 if budget is None else budget.max_gpu_seconds

    def seal_next_budget(self, budget: StageBudget) -> CampaignLedger:
        """Seal the immediate next stage; future stages cannot be pre-authorized."""

        if not isinstance(budget, StageBudget):
            raise TypeError("budget must be a StageBudget")
        if budget.stage is not self.next_stage:
            raise ValueError("a budget may be sealed only for the immediate next stage")
        if self._budget_for(budget.stage) is not None:
            if self._budget_for(budget.stage) == budget:
                return self
            raise ValueError("a sealed budget cannot be replaced")
        return replace(self, budgets=(*self.budgets, budget))

    def record_gate(self, record: GateRecord) -> CampaignLedger:
        """Append one terminal verdict, enforcing order and the current cap."""

        if not isinstance(record, GateRecord):
            raise TypeError("record must be a GateRecord")
        if record.stage is not self.next_stage:
            existing = next(
                (item for item in self.records if item.stage is record.stage),
                None,
            )
            if existing == record:
                return self
            raise ValueError("a gate may be recorded only for the immediate next stage")
        if record.stage is not CampaignStage.CPU_RELEASE:
            budget = self._budget_for(record.stage)
            if budget is None:
                raise ValueError("the next accelerator stage has no sealed budget")
            if record.used_gpu_seconds > budget.max_gpu_seconds:
                raise ValueError("gate result exceeds the sealed GPU-second cap")
            if record.completed_replays > budget.max_replays:
                raise ValueError("gate result exceeds the sealed replay cap")
        return replace(self, records=(*self.records, record))

    def document(self) -> dict[str, object]:
        """Return a canonical self-hashed audit document."""

        payload: dict[str, object] = {
            "schema": "fissionspec.accelerator-campaign-ledger.v1",
            "plan": asdict(self.plan),
            "campaign_id": self.plan.campaign_id,
            "budgets": [asdict(budget) for budget in self.budgets],
            "records": [asdict(record) for record in self.records],
            "next_stage": None if self.next_stage is None else self.next_stage.value,
            "spent_gpu_seconds": self.spent_gpu_seconds,
            "currently_authorized_gpu_seconds": self.currently_authorized_gpu_seconds,
        }
        digest = hashlib.sha256(_canonical_bytes(payload)).hexdigest()
        return cast(dict[str, object], {**payload, "payload_sha256": digest})


__all__ = [
    "STAGE_ORDER",
    "CampaignLedger",
    "CampaignPlan",
    "CampaignStage",
    "GateRecord",
    "GateVerdict",
    "StageBudget",
]
