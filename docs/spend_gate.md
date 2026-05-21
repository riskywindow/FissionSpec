# Fail-closed accelerator campaign ledger

`fissionspec.spend_gate` turns the staged GPU pre-registration into an
append-only authorization state machine. It does not launch a job, store cloud
credentials, or estimate provider prices.

Before Stage F1, `CampaignPlan` freezes:

- the protocol hash and exact code commit;
- the complete CPU artifact hash;
- exactly three validation trace hashes (`V1`, `V2`, `V3`); and
- primary, ablation, and robustness counts checked against
  `ExperimentSpendCaps`.

The only stage order is:

```text
F0 CPU release
 -> F1 physical row omission
 -> F2 one-miss mechanism
 -> F3 controller transport
 -> F4 H100 confirmation
 -> F5 B200 transport
```

An accelerator stage has zero authorization until `seal_next_budget` records
its positive integer GPU-second and replay caps plus the hash of the
calibration/rationale document. Only the immediate next stage can be sealed;
the system cannot pre-authorize the entire campaign.

`record_gate` requires a terminal pass/fail verdict, actual GPU seconds,
completed replays, and an evidence hash. It rejects skipped stages, unsealed
work, cap overruns, and changed duplicate records. Replaying an identical
budget or record is idempotent. A failed gate makes `next_stage` permanently
empty and `currently_authorized_gpu_seconds` zero.

The ledger emits canonical, self-hashed JSON suitable for an experiment
archive. Integer GPU-seconds avoid floating-point accounting ambiguity. Dollar
cost remains a reporting field outside this module because provider prices,
discounts, and reservation terms are time-varying; the resource cap is
provider independent.

`CampaignPlan.document` and `CampaignLedger.document` produce separate
self-hashed archive objects. Their `from_document` inverses verify payload
hashes, exact nested schemas, enum values, replay/resource caps, contiguous
stage history, and every derived field before returning executable
authorization state. Rehashing a contradictory `next_stage`, spend total, or
campaign ID is rejected rather than trusted.

This guardrail cannot prevent an operator from bypassing the software. The
production launcher must require a ledger whose `next_stage`, sealed budget,
protocol hash, and code commit match the submitted job.
