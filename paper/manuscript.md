# FissionSpec: Outcome-Decoupled Continuous Batching for
# Speculative-Speculative LLM Serving

> Full working manuscript. Bracketed GPU table cells are intentionally blank
> until the pre-registered accelerator protocol is executed.

## Abstract

Speculative-speculative decoding (SSD) overlaps draft generation with target
verification by preparing continuations for likely verification outcomes.
Under online batching, however, each row independently hits or misses its
outcome cache while the target executes a rectangular batch. Existing choices
either hold the cohort behind a miss or let recovering rows advance through
padded target work. We identify this as *miss externality*: delay and physical
verification work that one outcome miss imposes on co-batched hits.

We present FissionSpec, a control plane that turns every realized SSD outcome
into an independent scheduling event. Outcome-cache hits may immediately rejoin
target admission; misses recover on the draft side without occupying target
rows; compatible work is re-fused only when a bounded horizon-2 controller
predicts lower flow cost without a rolling deadline violation. A versioned
copy-on-write KV ledger, counter-addressed semantic randomness, and a canonical
crash-recovery coordinator make this reordering executable.

For finite rational autoregressive models, an exact CPU oracle exhaustively
recovers the target sequence distribution under speculative rejection sampling
and adversarial rebatching. Exact joint-outcome analysis generalizes the miss
externality to correlated outcomes and heterogeneous random recovery. A
deterministic event simulator and bounded scheduling oracles isolate controller
behavior, while a spend-gated GPU protocol pre-registers the physical
row-removal and serving hypotheses. The current artifact establishes semantics,
state safety, and model behavior—not a GPU speedup. [Insert registered
matched-resource H100/B200 results only after measurement.]

## 1. Introduction

Autoregressive generation repeatedly pays the latency of a large target model.
Speculative decoding amortizes that latency by asking a faster draft model for
multiple candidate tokens, verifying the block in one target call, and using
rejection sampling to preserve the target distribution
([Leviathan et al.](https://arxiv.org/abs/2211.17192);
[Chen et al.](https://arxiv.org/abs/2302.01318)). The draft itself remains on
the critical path.

SSD removes that serialization. While the target verifies the current block,
the draft predicts possible verification outcomes and prepares a continuation
for each predicted branch. If the realized outcome is cached, the next block is
ready immediately. [Saguaro](https://arxiv.org/abs/2603.03251) establishes this
mechanism and its substantial single-request opportunity.

Batching changes the failure mode. Suppose a target launch contains `B` SSD
requests, each with outcome-cache hit probability `p`. A batch fallback is
needed with probability `1-p^B`. At `p=0.95` and `B=32`, fallback is common even
though almost every individual row hits. A cohort barrier transfers one miss's
recovery delay to the hits. Parallel recovery can remove the wait, but a fixed
verification shape may retain the recovering request as a one-token row padded
to width `k`.

The scheduler therefore faces a choice after—not before—the target result:

```text
cohort barrier       keep all rows synchronized and wait;
padded parallel      keep misses in the target shape while recovery overlaps;
outcome fission      remove misses, dispatch hits, later re-fuse recovered work.
```

Immediate fission is not automatically optimal. Removing rows can shrink the
target batch, add launches, and sacrifice kernel efficiency. The open systems
question is whether the scheduler can isolate miss externality while retaining
enough batching efficiency to improve SLO goodput.

FissionSpec makes the following contributions:

1. **Problem formulation.** It defines SSD miss externality at the outcome
   boundary and derives exact barrier-versus-isolation cost for arbitrary joint
   outcomes and random recovery times.
2. **Outcome-decoupled scheduler.** It splits a verifier cohort into ready-hit,
   recovering, and finished lanes, physically excludes recovering rows from
   logical target admission, and uses bounded model-predictive re-fusion.
3. **Correctness substrate.** It supplies exact finite-model speculative
   semantics, per-request counter randomness, versioned provisional KV
   transactions, and a composed coordinator with deterministic crash recovery.
4. **Falsifiable artifact.** It provides deterministic workload generation,
   complete hash-linked traces, exact small scheduling oracles, rigorous paired
   inference, Rust hot-path primitives, and narrow vLLM/SGLang integration
   contracts.
5. **Spend-gated evaluation.** It separates CPU claims from physical kernel
   claims and pre-registers sequential GPU gates that abort before broad
   benchmarking when row omission, the mechanism, or controller transport
   fails.

## 2. Background and claim boundary

### 2.1 Speculative sampling

Let target and draft next-token distributions be `p(x|s)` and `q(x|s)`. For a
draft candidate `x`, exact speculative sampling accepts with

```text
a(x|s) = min(1, p(x|s) / q(x|s)).
```

At the first rejection, it samples one correction from normalized
`max(0,p-q)`. If the entire block is accepted, it samples one bonus token from
the target after the block. Only the accepted prefix and correction/bonus enter
committed target state. Batching may alter execution order, but it must not
change these logical random variables or their autoregressive prefixes.

### 2.2 SSD outcome caches

An SSD drafter predicts the result of current verification and prepares the
next proposal for likely outcomes. “Outcome-cache hit” is distinct from
“candidate token accepted.” The current target block always authorizes an
accepted prefix and correction/bonus. The separate cache lookup asks whether a
continuation for that complete realized outcome was prepared.

This distinction matters experimentally. Conflating accepted length with cache
membership makes high-acceptance regimes look like high-cache-hit regimes and
prevents independent mechanism analysis.

### 2.3 Continuous batching and padded work

Serving engines rebuild a request batch at each decode iteration, but kernels
and CUDA graphs still execute physical shapes. A logical mask is not proof that
the masked work is free. FissionSpec therefore represents both request rows and
verification slots, and treats the measured graph-bucket delta as an empirical
input.

### 2.4 Closest work

Saguaro supplies SSD outcome caching and batch fallback.
[SPECTRE](https://arxiv.org/abs/2605.08151) supplies remote drafting,
per-request rollback, and hybrid ordinary/parallel execution; its parallel mode
motivates the padded-row comparison. [EXSpec](https://arxiv.org/abs/2510.22876)
correctly regroups ragged ordinary-SD sequences.
[FASER](https://arxiv.org/abs/2604.20503) manages draft lengths and
draft/verification frontiers per request. [SwiftSpec](https://arxiv.org/abs/2506.11309)
disaggregates asynchronous speculation and repairs tree KV state.
[TAPER](https://arxiv.org/abs/2605.06914) regulates branch externality against
batch slack. [TurboSpec](https://arxiv.org/abs/2406.14066) controls speculative
parallelism for runtime goodput. TransKV separates committed and provisional
paged KV.

None of those mechanisms is claimed as new. The prospective claim is the
combination of post-outcome, per-request target-row removal and bounded
re-fusion for batched SSD. The dated systematic matrix is
`docs/literature_matrix.md`.

## 3. Miss externality

### 3.1 Independent fixed-recovery case

For row hit probabilities `p_i`, the batch fallback probability is

```text
q_barrier = 1 - product_i p_i.
```

With iid `p`, batch size `B`, and fixed recovery `R`,

```text
C_barrier = B R (1-p^B),
C_isolated = B R (1-p),
C_barrier / C_isolated = 1 + p + ... + p^(B-1).
```

The ratio approaches `B` as `p` approaches one: rare individual misses are
exactly where a cohort barrier has its largest relative amplification.

### 3.2 Correlated outcomes and random recovery

Let `omega` be one atom of the complete joint outcome distribution,
`M(omega)` its missing rows, and `R_i(omega)` the realized recovery for miss
`i`. If recoveries begin together, a barrier produces

```text
C_barrier(omega) = B max_{i in M(omega)} R_i(omega),
```

while perfect isolation produces

```text
C_fission(omega) = sum_{i in M(omega)} R_i(omega).
```

Their expected difference decomposes:

```text
E[C_barrier-C_fission]
 = E[|hits| max R]
 + E[sum_{i in M}(max R-R_i)].
```

The first term is hit externality. The second is cross-miss externality: slow
recoveries hold faster misses. This identity makes no independence assumption.
Positive correlation can concentrate misses into all-miss batches and reduce
hit externality at fixed marginals; anti-correlation can spread “any miss”
across more batches.

### 3.3 Padding-versus-fission break-even

Use a one-step objective

```text
J = target service time + sum_i w_i request delay_i.
```

Let `Delta L_target` be the measured incremental target time from retaining the
recovering rows and `Delta d_i` the delay a padded bypass saves for each miss:

```text
J_padding - J_fission = Delta L_target - sum_i w_i Delta d_i.
```

Fission wins this local objective when the right side is positive. In a
diagnostic linear model with `m` recovering rows, width `k`, per-masked-slot
cost `c`, and per-row overhead `a`,

```text
Delta L_target = m[a+(k-1)c].
```

The implementation accepts the calibrated graph-bucket delta directly. It does
not assume a masked slot costs `c`.

## 4. Design

### 4.1 Outcome events and lanes

One target completion produces an independent event per request:

- `READY_HIT`: a valid next-round proposal already exists;
- `RECOVERING`: the realized outcome was absent; remote/local draft repair is
  in flight and the row is absent from target admission;
- `READY_BACKUP`: repair completed and the request can rejoin;
- `FINISHED`: target-authorized output extent is complete; and
- optional `BYPASS`: a one-token padded action used only by a compatible
  baseline/controller decision.

Fission changes descriptors and ownership, not committed KV bytes. A ready hit
may launch with unrelated ready rows. A recovered request is not required to
rejoin its original cohort.

### 4.2 Outcome identity

An eager continuation key includes:

```text
request ID, logical round, parent committed digest,
draft-model fingerprint, complete proposal,
accepted length, correction/bonus token, outcome kind.
```

Surface token alone is insufficient: two branches can emit the same last token
from different proposals or accepted prefixes.

### 4.3 Horizon-2 re-fusion

When the target is idle, let `n` rows be ready now, `m` rows become ready at the
earliest known internal time `delta`, and `L(S)` price exact selected
row/slot sets. In the simple no-overflow case:

```text
C_now  = n L(n) + m[max(L(n),delta)+L(m)-delta],
C_wait = n[delta+L(n+m)] + m L(n+m).
```

The implementation is more exact than this notation:

1. rows carry verifier widths and rolling deadlines;
2. both cohorts are merged in stable EDF order;
3. capacity overflow is split into sequential target chunks;
4. every chunk uses the two-dimensional row/slot profile;
5. waiting is bounded from the oldest ready timestamp; and
6. a plan that misses a forecast deadline loses to a feasible plan.

The controller sees the earliest internally known recovery/precompute
completion but never future external arrivals. Sorting dominates one decision:
`O((n+m) log(n+m))` time and `O(n+m)` auxiliary space.

### 4.4 Fairness and liveness

For a finite closed cohort of `N` rows, capacity `C`, bounded recovery `R_max`,
bounded coalescing wait `W`, and target launch time at most `L_max`, one service
for all rows completes within

```text
R_max + W + ceil(N/C)L_max.
```

This does not prove starvation freedom under an unbounded adversarial arrival
stream. The production policy therefore needs an explicit age/fairness rule,
and the evaluation reports maximum recovery/ready age rather than inferring
fairness from mean latency.

## 5. Token and state correctness

### 5.1 Exact semantic oracle

`TinyAutoregressiveModel` stores categorical rows as exact fractions and uses
the longest matching context suffix. The oracle enumerates:

1. every autoregressive draft proposal;
2. every accepted-prefix/rejection outcome;
3. every residual correction or target bonus; and
4. all later rounds to a finite output horizon.

It compares the resulting sequence-probability map exactly—not within a
tolerance—with direct target autoregression. A sampled implementation converts
rational probabilities to integer weights and uses counter-addressed integer
rejection, avoiding floating-point thresholds.

**Finite semantic theorem.** For every tested finite rational target/draft pair,
prompt, width, and horizon, the enumerated speculative output distribution
equals the target-only distribution.

The exhaustive program is the proof certificate for the finite domain.
Inductively, one speculative step has target next-token law by standard
rejection/residual decomposition; conditioning on its committed emitted prefix
and applying the hypothesis to the remaining horizon completes the argument.

### 5.2 Schedule-independent randomness

Every draw is a pure function of

```text
(seed, request_id, logical_round, semantic_stream, draw).
```

Reordering a batch cannot consume another request's randomness. Prepared
branches share logical draw addresses but transform those raw values through
their branch-specific distributions, giving a deterministic counterfactual
coupling.

### 5.3 Provisional KV ledger

The ledger owns committed page spans and one active versioned transaction per
request. Outcome branches copy a partial committed tail on write. Commit keeps
only the selected target-verified prefix; sibling/private suffix pages are
released. Generation-scoped page handles reject stale ABA releases. Abort is
idempotent, but a stale commit is not silently accepted.

### 5.4 Composed crash coordinator

The coordinator binds each protocol tag to one ledger epoch and block-table
descriptor. Its durable publication unit is a canonical JSON snapshot of both
subsystems plus their binding under one checksum. Restore fences any
snapshot-in-flight work to a newer recovery version.

The fault harness crashes before/after reservation, ledger staging, verifier
publication, prefix commit, protocol transition, recovery application, and
cancellation. Duplicate, dropped, reordered, unknown, and stale callbacks have
no state effect. Cross-layer audit checks page ownership, lane/transaction
agreement, version high-water marks, exact outbox identity, and completion.

## 6. Implementation

The artifact has four layers:

1. **Python semantic/state references:** exact token oracle, versioned page
   ledger, composed coordinator, and deterministic workload/artifact schemas.
2. **Python event simulation:** separate target and draft clocks, exact
   row/slot latency curves, precomputed outcome hits, recovery, barriers,
   padded rows, fission, and controller forecasting.
3. **Rust hot-path reference:** allocation-free flattened-work controller,
   fixed allocator, transaction fencing, and latency interpolation.
4. **Engine contracts:** narrow SGLang and vLLM patch boundaries for reserve,
   verify completion, recovery completion, and next-batch selection.

Python and Rust intentionally do not share an undocumented cost-model claim.
Python retains two-dimensional row/slot shapes. Rust accepts flattened service
units and priority weights. Their common semantic subset must be locked by
golden fixtures before integration.

## 7. Evaluation methodology

### 7.1 Evidence tiers

We report exact finite-domain, invariant-model, simulation-model, CPU
statistical, GPU microbenchmark, and GPU confirmatory evidence separately. Every
simulation document embeds the warning “not a GPU measurement,” full input
configuration, RNG fingerprint, and hash links.

### 7.2 Baselines

Required scheduler comparisons are:

- Saguaro barrier;
- full SPECTRE ordinary/parallel hybrid and its padded component;
- EXSpec-style sliding-pool regrouping;
- FASER/TurboSpec-style adaptive control where artifacts permit matching;
- immediate fission;
- fixed coalescing;
- myopic slack/age control;
- horizon-2 FissionSpec; and
- exact bounded scheduling for tiny pre-realized traces.

An abstraction is named as such and lists every difference from the paper
system. The padded component is never reported as full SPECTRE.

### 7.3 Workloads

CPU workload generators cover synchronized cohorts, Poisson arrivals, exact
two-state continuous-time MMPP bursts, finite-mean Pareto gaps, heterogeneous
context/output lengths, correlated outcome classes, and byte-hashed CSV replay.
Explicit development and validation splits prevent controller tuning on
confirmatory scenarios.

### 7.4 Statistics

The independent unit is a seed/trace cluster. Policies are paired on the same
workload and counter RNG. The analysis reports paired raw/relative effects,
`d_z`, win probability, deterministic cluster-bootstrap intervals,
family-wise multiplicity metadata, and replication planning. Optional stopping
uses a bounded alpha-spending Hoeffding confidence sequence with bounds fixed
before observation.

Means without uncertainty cannot support a headline cell. Fewer than ten
independent clusters are automatically labeled pilot evidence.

### 7.5 GPU protocol

`paper/gpu_preregistration.md` defines five spend stages:

1. calibration and physical row-removal feasibility;
2. one-miss mechanism falsification;
3. controller-boundary transport;
4. matched-resource H100 confirmation on two model pairs; and
5. B200 transport only if H100 passes.

Each stage has an abort gate. Confirmatory cells use paired ABBA/BAAB blocks,
bounded symmetric improvements in `[-1,1]`, simultaneous intervals, minimum
and maximum replication, precision, efficacy, futility, and explicit
positive-but-below-minimum-worthwhile-effect rules. The executable rule
ordering prevents overlapping interval boundaries from being reported as
headline efficacy.

## 8. CPU evidence

### 8.1 Semantic and state evidence

At commit `560f4f5`, the suite contains:

- exact distribution equality over hand-built and deterministically randomized
  two/three-token autoregressive models, multiple horizons and widths;
- greedy target/speculative equality and per-round committed-state
  reconstruction;
- schedule-order and eager-cache completion-order invariance;
- all named coordinator crash points plus callback/OOM/cancel corruption tests;
  and
- randomized ledger ownership audits.

These establish the finite semantic and state claims only.

### 8.2 Analytical evidence

Exact rational tests verify the correlated externality decomposition, recover
the iid closed form over an exhaustive small domain, and exercise equal,
heterogeneous, and all-hit/all-miss edges. Padding break-even, liveness, and
controller lookup-count formulas are executable.

### 8.3 Synthetic model results

The original three-seed factorial reports modeled immediate/H2 throughput
changes of 1.2%–4.3% over the barrier on its bundled synthetic profile, with
zero directly attributed hit delay and fewer padded slots. H2 equals immediate
fission in the default slow-recovery cells; a separate controller phase diagram
contains re-fusion regions.

These means are mechanism smoke tests. The rigorous analysis marks three
clusters as pilot evidence, attaches simultaneous intervals/effect sizes, and
does not convert a synthetic latency surface into a performance claim.

### 8.4 Extended fidelity, oracle, and workload study

Results to populate after the CPU completion runs:

| Question | Artifact/table |
|---|---|
| Null-equivalence of fidelity model | `[CPU-FIDELITY-NULL]` |
| Cache fanout/budget/eviction phase | `[CPU-CACHE-PHASE]` |
| Remote draft workers/jitter/failure | `[CPU-DRAFT-SERVICE]` |
| Full-baseline comparison | `[CPU-BASELINES]` |
| H2 gap to generalized oracle | `[CPU-ORACLE-GAP]` |
| Held-out MMPP/Pareto/replay intervals | `[CPU-VALIDATION]` |
| Adversarial starvation/break-even traces | `[CPU-ADVERSARIAL]` |

## 9. Registered accelerator results

This section is a schema, not a result.

| Gate | Pair/hardware | Primary outcome | Result |
|---|---|---|---|
| F1 physical row omission | Qwen3-8B/0.6B, H100 | target descriptor and step delta | `[UNMEASURED]` |
| F2 one-miss mechanism | Qwen3-8B/0.6B, H100 | hit delay and goodput non-inferiority | `[UNMEASURED]` |
| F3 controller transport | selected H100 boundary cells | action agreement and regret | `[UNMEASURED]` |
| F4 confirmatory | Qwen3-32B/0.6B, H100 | TBT-SLO goodput, P99 TBT | `[UNMEASURED]` |
| F4 confirmatory | Llama-3.1-70B/1B, H100 | TBT-SLO goodput, P99 TBT | `[UNMEASURED]` |
| F5 architecture transport | passing cells, B200 | same registered family | `[UNMEASURED]` |

## 10. Limitations and threats to validity

First, the current performance simulator is not a transformer runtime.
Synthetic or even calibrated curves omit kernel interference, numerical
effects, graph recapture, allocator synchronization, and network stacks.

Second, logical row omission may not reduce physical work. A fixed CUDA graph
bucket can make the same kernel execute. Conversely, packed kernels can make
slot-count models pessimistic. Gate F1 exists to falsify this premise cheaply.

Third, finite rational equality does not prove bitwise real-model equality.
Floating-point reductions, attention layouts, and batch order can perturb
logits. Production testing must report greedy mismatches and sampled
distribution statistics rather than asserting identity “by construction.”

Fourth, latency models and workload traces can be wrong. Cache membership is
endogenous to fanout, outcome popularity, memory pressure, context, and draft
service contention. The fidelity study exposes these axes but cannot guarantee
production representativeness.

Fifth, horizon-2 has no universal approximation ratio. It deliberately hides
future arrivals. We report bounded-oracle gaps and counterexamples instead of
claiming optimality.

Sixth, the coordinator's atomic snapshot is a storage contract, not a
distributed consensus implementation. Real failover needs leases, durable
publication, and engine-specific completion fencing.

Finally, literature is moving quickly. The novelty statement is prospective
and dated. Discovery of prior post-outcome row removal/re-fusion narrows or
invalidates it regardless of experimental performance.

## 11. Broader impact

Serving efficiency can reduce accelerator time and energy per useful token, but
it can also lower the cost of deploying increasingly capable models. FissionSpec
does not change model outputs intentionally, moderation, access control, or
training. The artifact records total target-plus-draft accelerators and energy
per SLO-compliant token to avoid reporting efficiency by hiding resources.

Open traces must be checked for user data and licensing before archival. The
provided replay schema requires only timing and token-count metadata and does
not require prompt text.

## 12. Conclusion

SSD exposes a scheduling boundary that batch-level policies obscure: the
outcome cache is consulted per request. FissionSpec makes that independence
explicit, isolates misses from target admission, and re-fuses work only under a
bounded cost/deadline forecast. Exact semantics and crash/page invariants make
the proposal falsifiable before expensive integration. Whether that logical
advantage becomes physical goodput is deliberately left to a pre-registered,
spend-gated accelerator study.

## Appendix A. Controller pseudocode

```text
procedure NEXT_BATCH(now, ready, next_internal, capacity, profile):
    current <- EDF_PREFIX(ready, capacity)
    if |ready| >= capacity or next_internal is absent:
        return DISPATCH(current)

    eta, future <- next_internal
    if eta-now <= 0 or eta > oldest(current)+max_wait:
        return DISPATCH(current)

    launch_plan <- [current] + CHUNK(EDF(future), capacity)
    wait_plan   <- CHUNK(EDF(current union future), capacity)

    launch_cost, launch_feasible <- SIMULATE(launch_plan, now, eta, profile)
    wait_cost, wait_feasible     <- SIMULATE(wait_plan, eta, eta, profile)

    if not wait_feasible:
        return DISPATCH(current)
    if not launch_feasible:
        return WAKE_AT(eta)
    if wait_cost < launch_cost:
        return WAKE_AT(eta)
    return DISPATCH(current)  // deterministic tie break
```

## Appendix B. Verify/recovery publication pseudocode

```text
procedure RESERVE(request, width):
    require request is target-ready
    preflight allocator capacity
    epoch  <- ledger.begin(request, next_round)
    branch <- ledger.stage(epoch, outcome_key, width)
    tag    <- protocol.start_verification(next_round)
    atomically publish(tag, epoch, branch.block_table)
    return verifier_command(tag, branch.block_table)

procedure VERIFY_COMPLETE(reply):
    if reply.tag != active_tag:
        return IGNORED_STALE
    require reply.prefix <= reserved_width
    ledger.commit(active_epoch, selected_outcome, reply.prefix)
    transition <- protocol.apply(reply)
    atomically publish(committed_state, transition, recovery_outbox)

procedure RESTORE(snapshot):
    verify canonical checksum and cross-layer invariants
    for each snapshot-in-flight tag:
        abort its provisional epoch
        mint a strictly newer recovery version
        publish the new recovery command
```

## Appendix C. Artifact map

| Obligation | Executable source |
|---|---|
| Exact token law | `src/fissionspec/semantics.py` |
| Joint miss theory | `src/fissionspec/theory.py` |
| Event scheduling | `src/fissionspec/simulator.py` |
| Horizon-2 policy | `src/fissionspec/policies.py` |
| KV ownership | `src/fissionspec/ledger.py` |
| Async fencing | `src/fissionspec/protocol.py` |
| Composed restart | `src/fissionspec/coordinator.py` |
| Counter RNG | `src/fissionspec/rng.py` |
| Full trace artifacts | `src/fissionspec/artifacts.py` |
| Workload generation/replay | `src/fissionspec/workload_generators.py` |
| Paired inference | `src/fissionspec/statistics.py` |
| Rust hot path | `crates/fissionspec-core/` |
| Engine seam | `integrations/sglang/`, `integrations/vllm/` |
| Claim gates | `paper/claims_evidence.md` |
| GPU spend gates | `paper/gpu_preregistration.md` |
