# GPU experiment pre-registration and spend gates

**Status:** protocol frozen before accelerator measurements

**Protocol version:** 1
**CPU-only model output is not pilot GPU evidence.**

This protocol makes accelerator work a sequence of falsification gates. A later
stage is purchased only if the earlier, cheaper stage supports the physical
mechanism needed by it. Failed gates are publishable negative results and stop
the spend; they are not permission to redefine the hypothesis.

## 1. Frozen research question

For batched speculative-speculative decoding, does scheduling each realized
outcome independently—and physically removing recovering misses from target
inputs—improve matched-resource deadline goodput or tail inter-token latency
relative to:

1. batch-wide recovery barriers;
2. SPECTRE's hybrid ordinary/parallel policy;
3. the isolated SPECTRE padded-parallel component;
4. EXSpec-style regrouping;
5. immediate fission and fixed coalescing; and
6. FissionSpec's horizon-2 controller?

The primary mechanism is neither asynchronous drafting nor transactional KV
state alone. It is removal of post-outcome wait and padded target work without
losing more batching efficiency than that removal saves.

## 2. Immutable claim boundary

CPU work may establish semantic equivalence in finite models, state-machine
safety, scheduler invariants, analytical identities, simulator behavior, and
the exact result of bounded scheduling instances. Only GPU work may establish:

- physical target-time savings from omitting logical rows;
- end-to-end output equivalence under production kernels;
- throughput, latency, utilization, memory, power, or cost improvements; and
- behavior under real engine OOMs and accelerator faults.

Synthetic profile results will never be pooled with accelerator measurements.

## 3. Systems and resource matching

The confirmatory model pairs are:

- Qwen3-32B target with Qwen3-0.6B draft; and
- Llama-3.1-70B target with Llama-3.2-1B draft.

A Qwen3-8B/0.6B pair may be used only for the cheap feasibility gates. Passing
on 8B does not substitute for either confirmatory pair. Exact model revisions,
tokenizer hashes, dtype/quantization, tensor parallel degree, engine commit,
CUDA, driver, graph configuration, and hardware SKU are recorded before the
first run.

Every comparison holds constant:

- target and draft accelerator count and SKU;
- tensor/pipeline/data parallel layout;
- maximum KV memory and outcome-cache budget;
- prompt/output trace and logical random draws;
- sampling parameters and request admission;
- warmup policy and CUDA graph buckets; and
- measurement interval.

Results report output tokens per second **per total target-plus-draft GPU** in
addition to raw throughput. A policy cannot hide a dedicated draft GPU or extra
replica outside the denominator.

The primary confirmatory hardware is H100 80GB. B200 is a transport validation:
it is purchased only after both H100 confirmatory pairs pass, and uses the same
frozen cells rather than a new sweep.

## 4. Frozen workloads and splits

The CPU artifact generates or imports byte-hashed replay traces before GPU
access. Scenario labels are split at the configuration level:

- **development:** synchronized mechanism traces, Poisson load points, and
  controller-boundary cases used for implementation debugging;
- **validation:** held-out MMPP, Pareto, heterogeneous prompt/output, and public
  replay traces used for confirmatory reporting.

No validation trace may tune coalescing, controller cost weights, fanout, cache
allocation, graph buckets, or failure thresholds. Request IDs drive the same
counter-addressed semantic random variables under every policy.

Temperatures are `0`, `0.6`, and `1.0`. Prompt/output classes are
`(128, 32)`, `(2K, 128)`, and `(16K, 256)` tokens when supported by both model
pairs. Offered load is expressed as fractions of the target-only saturation
rate measured in Stage 1: `0.35`, `0.70`, `0.90`, and `1.05`. The last point is
an overload stress test, not part of the primary SLO claim.

Those axes are not a confirmatory Cartesian product. To bound accelerator
spend, the primary family contains exactly three validation anchors per model
pair:

| Anchor | Arrival process | Prompt/output | Temperature | Load |
|---|---|---:|---:|---:|
| V1 | held-out MMPP | `(128, 32)` | `0.6` | `0.70` |
| V2 | held-out finite-mean Pareto | `(16K, 256)` | `1.0` | `0.90` |
| V3 | byte-hashed public replay with its recorded heterogeneous shapes | trace class | `0` | replay rescaled to `0.70` |

The medium context, low-load, overload, and remaining temperature combinations
are registered robustness cells, not primary hypotheses. They are purchased
only after the three anchors pass and are limited to twelve deterministic
farthest-point cells across the complete robustness grid. Development traces
may exercise the entire grid on CPU but cannot replace or tune these validation
anchors. Exact replay/configuration hashes are inserted into the frozen run
manifest before Stage 1 and cannot be selected from GPU outcomes.

## 5. Metrics and experimental unit

The independent experimental unit is one complete seed/trace replay, not a
request, token, or launch within that replay. Policy runs are paired within the
same unit and executed in randomized ABBA/BAAB blocks to reduce thermal and
temporal drift.

Primary metrics:

1. request-level TBT-SLO goodput tokens/s/total-GPU;
2. P99 inter-token latency;
3. conditional hit delay when a cohort peer misses; and
4. target step time for the one-miss mechanism trace.

Secondary metrics:

- TTFT, P50/P95 TBT, request and token SLO attainment;
- physical target input rows/slots, graph bucket, and launch count;
- target/draft busy time and achieved occupancy;
- provisional/outcome-cache peak bytes and eviction rate;
- recovery starvation and controller time;
- energy per SLO-compliant output token; and
- greedy mismatch plus sampled-distribution test statistics.

For paired positive quantities `C` and `B`, inference uses the bounded symmetric
improvement

```text
D = orientation * (C - B) / max(C, B),  D in [-1, 1],
```

where orientation is positive for higher-is-better metrics and negative for
lower-is-better metrics. Raw units and ratios are also reported.

## 6. Sequential inference fixed in advance

Each confirmatory family uses paired seed clusters, deterministic resampling,
and family-wise error `0.05`. The primary family contains the four primary
metrics times two model pairs times three non-overload validation arrival
classes. Bonferroni simultaneous intervals are conservative but auditable.
Secondary results are labeled exploratory.

For every confirmatory cell:

- minimum independent paired blocks: `10`;
- maximum independent paired blocks: `50`;
- inspect only at completed block counts `10, 15, 20, ..., 50`;
- confidence sequence: alpha-spending Hoeffding with predeclared bounds
  `[-1, 1]`;
- precision success: simultaneous half-width at most `0.03`;
- efficacy stop: lower bound exceeds `0` **and** the upper bound is not below
  the minimum worthwhile improvement `0.03`;
- futility stop: upper bound is below `0.03` and the interval is not wholly
  positive;
- positive-but-below-MWI stop: the interval is wholly above `0` and wholly
  below `0.03`; this is terminal evidence of a positive but practically
  insufficient effect, not headline efficacy; and
- otherwise stop at `50` and report the interval without binary success.

Stopping applies to a complete multiplicity family, not whichever cell first
looks favorable. All completed blocks remain in the analysis. Hardware or
software failures are excluded only by an error code defined before inspecting
metric values, and the paired counterpart is excluded with them.
The executable ordering of these overlapping boundaries is frozen in
`fissionspec.experiment_design.evaluate_sequential_gate`.

Power planning uses the paired pilot variance but never its observed mean.
Recommended replication is computed for standardized effects `0.3`, `0.5`, and
`0.8`; the sequential maximum remains `50`.

Every replay yields all four primary metrics, so metrics do not multiply the
number of accelerator executions. For the primary candidate-versus-SPECTRE
comparison, the hard cap is:

```text
2 model pairs * 3 validation anchors * 50 blocks * 4 ABBA/BAAB runs
= 1,200 complete policy replays
```

The first possible family stop is 240 replays. Required non-confirmatory
ablations use ten paired seeds per anchor, reuse an identical candidate replay
when its complete environment hash matches, and do not trigger additional
adaptive looks. The optional robustness surface is capped at the twelve cells
above. A run orchestrator must reject a manifest that exceeds these counts;
wall-time and GPU-hour caps are filled from Stage 1 calibration before Stage 2.
`fissionspec.spend_gate` enforces the stage order, hash-locked manifest, and
one-stage-at-a-time GPU-second/replay authorization. A failed gate leaves zero
later authorization.

## 7. Stage 0 — zero-GPU release gate

Do not rent hardware until all of these pass at the tagged commit:

- token-exact finite-model equivalence and greedy equivalence;
- composed coordinator fault/state-machine tests;
- KV ownership and deterministic restart invariants;
- generalized bounded-oracle certificates;
- simulator null-equivalence and conservation tests;
- paired statistical pipeline and byte-reproducible full traces;
- clean sdist/wheel/container reproduction; and
- frozen integration patches and this protocol; and
- a canonical campaign plan plus a passing zero-GPU `F0` ledger record.

Any failure returns to CPU work and consumes no GPU budget.

## 8. Stage 1 — calibration, not a policy benchmark

The objective is to replace every symbolic latency constant before evaluating a
policy.

### 8.1 Cheap feasibility pair

Start with Qwen3-8B/0.6B on one available H100. Measure:

- target rows `B = {1, 8, 32}` and widths `k = {1, 4, 8}`;
- packed versus rectangular/masked physical inputs;
- draft, precompute, and recovery at `B = {1, 8, 32}`;
- prefill at prompt lengths `{128, 2K, 16K}`; and
- local and remote draft transport at payload sizes used by the protocol.

Each microbenchmark has 20 warmups, then paired randomized samples until its
95% confidence-sequence relative half-width is at most 2% or 100 samples.
Kernel/graph identity and physical tensor shapes are logged for every sample.

Add intermediate batch points `{2, 4, 16}` only when monotone interpolation
from adjacent anchors has more than 3% leave-one-anchor-out error. This adaptive
rule is fixed; it is not chosen after policy results.

### 8.2 Feasibility gate F1

At `B=32, k=4`, a tensor containing 31 full-width hit rows and one recovering
row must be launchable both with the recovering row padded and with it absent.
The absent-row case must:

1. show a different physical input descriptor/row count; and
2. have a simultaneous lower confidence bound above zero for target-step
   improvement, or move to a smaller graph bucket with no step-time regression.

If neither happens, stop. FissionSpec may still be an algorithmic scheduling
study, but the zero-padding systems claim is not worth a broad GPU campaign.

## 9. Stage 2 — mechanism falsification

Use synchronized cohorts with exactly one cache miss and `B-1` hits at
`B={8,32}`, `k={4,8}`, fixed 64-token outputs, and recovery ratios spanning the
CPU-predicted break-even boundary. Compare barrier, padded parallel mode,
immediate fission, and horizon-2 FissionSpec.

### Gate F2

Proceed only if, relative to both barrier and padded mode:

- the lower simultaneous bound for conditional hit-delay improvement is
  positive; and
- target physical padded slots decrease without a negative lower-bound
  deadline-goodput change worse than `-0.03`.

If the mechanism works only at one width/batch, retain that region explicitly;
do not average it with failures.

## 10. Stage 3 — controller transport

Fit latency surfaces using Stage 1 only. Without changing controller code, use
the CPU phase diagram to choose twelve H100 cells:

- four predicted immediate-fission cells;
- four cells within 5% of the predicted break-even boundary; and
- four predicted re-fusion cells.

Selection is deterministic by farthest-point coverage in normalized parameter
space, with lexicographic tie-breaking. Measure action agreement, regret in
aggregate flow, deadline violations, controller overhead, and optimality gap to
the bounded oracle.

### Gate F3

Proceed if at least 10/12 selected actions agree with the measured cheaper
action and no controller-induced deadline violation occurs on a feasible
trace. Otherwise revise the controller as a new protocol version and repeat
Stages 2–3; old and new results remain visible.

## 11. Stage 4 — confirmatory serving study

Run the two frozen confirmatory pairs on held-out validation traces using the
sequential rules in Section 6. The primary comparison is FissionSpec versus
SPECTRE hybrid. Barrier, padded component, EXSpec-style grouping, and immediate
fission are required ablations, not separate claim families.

Run failure recovery (remote timeout, stale reply, allocator pressure, and
worker restart) only after the no-fault family is complete. Real accelerator
OOM injection occurs last because it risks losing the most setup time.

### Gate F4

A positive headline requires:

- positive simultaneous lower bound for TBT-SLO goodput on at least one model
  pair;
- no negative simultaneous result on the other pair worse than `-0.03`;
- no greedy mismatch;
- sampled token tests consistent with the target distribution under the frozen
  test family; and
- complete matched-resource and provenance records.

Otherwise report a scoped or negative result.

## 12. Stage 5 — B200 transport

Only after F4 passes, repeat calibration anchors, F1, and the final
confirmatory cells on B200. Do not rerun development sweeps or retune the
controller. This stage tests whether conclusions transport across graph and
memory-bandwidth regimes; it is not another chance to select favorable cells.

## 13. Spend ledger and abort authority

Record wall-clock allocation, active benchmark time, warmup time, failed setup
time, GPU count, GPU model, power samples, and estimated charge for every
stage. Publish the ledger even if a gate fails.

The operator may abort immediately for safety, runaway cost, corrupted
environment, or unavailable matched resources. A scientific abort follows the
fixed gates above. “Promising trend” is not a reason to exceed a sample cap,
add a model, alter a metric, or expose validation traces during tuning.

## 14. Deviations

Any deviation is appended with timestamp, reason, affected hypotheses, and
whether it occurred before or after viewing outcomes. Post-outcome deviations
are exploratory and cannot replace the registered result.
