# Claims-to-evidence ledger

**Ledger version:** 1

**Cutoff:** 2026-07-23

No sentence may move from the “permitted wording” column to a stronger claim
without adding evidence that satisfies its gate. Model results, exact
finite-domain results, and GPU measurements are distinct evidence classes.

| ID | Proposed claim | Current evidence | Permitted wording now | Missing gate |
|---|---|---|---|---|
| C1 | Batch barriers create miss externality under heterogeneous, correlated outcomes and random recovery. | Exact rational identity and finite-domain tests in `fissionspec.theory`; iid formula in `fissionspec.metrics`. | “Under the stated simultaneous-recovery model, barrier stalled-row time decomposes exactly into isolated miss wait, collateral hit wait, and cross-miss wait.” | Trace/GPU evidence only if claiming magnitude in a deployment. |
| C2 | Speculative rejection sampling preserves the target distribution despite request rebatching. | Exhaustive exact sequence distributions for tiny rational autoregressive models, greedy equality, counter-addressed Monte Carlo in `fissionspec.semantics`. | “The CPU semantic oracle exactly matches the target distribution for the enumerated finite models and horizons; advancing sessions in tested schedules leaves outputs/state unchanged.” | Real-model/kernel numerical and end-to-end distribution tests. |
| C3 | Outcome-continuation cache construction can be completion-order independent. | Exact cache-key identity, forward/reverse build order, cached/uncached equality tests. | “For the finite oracle and complete key, cache build order does not change the prepared continuation or committed digest.” | Production cache implementation and collision/serialization audit. |
| C4 | Versioned fission can be composed with provisional paged KV state without stale mutation or page leaks in the modeled fault domain. | Coordinator, ledger, protocol, canonical snapshot, exhaustive named crash points, delayed/duplicate/drop/reorder/OOM/cancel tests. | “The in-memory reference has exactly-once state effects and passes its enumerated fault/ownership audits.” | Durable storage transaction, real block manager, distributed leases, accelerator OOM/kernel failure. |
| C5 | Bounded row isolation reduces barrier stalled-row work in the analytical model. | Closed-form iid formula and exact joint-scenario evaluation. | “Perfect isolation weakly reduces modeled stalled-row time; improvement is strict exactly when an atom has a hit waiting on a miss or heterogeneous miss completion.” | None for the theorem; measurement needed for latency claims. |
| C6 | Padding-versus-fission has a measurable break-even boundary. | Exact one-step objective identity and linear diagnostic specialization. | “Given a calibrated incremental target cost and weighted bypass delay credit, the lower-objective action follows the stated difference.” | Real graph-bucket/physical-row calibration. |
| C7 | The horizon-2 controller is deadline-aware and computationally bounded. | Source inspection, unit tests, exact profile-lookup count, finite closed-cohort liveness assumptions. | “The controller considers one known internal readiness cohort, refuses locally forecast deadline violations, and runs in `O((n+m) log(n+m))` time.” | Generalized-oracle gaps; production controller timing. |
| C8 | H2 is near-optimal. | Not established. The restricted oracle has a narrower action space. | Do not claim. Report finite-trace gaps after the generalized oracle lands. | Exact bounded arbitrary-subset oracle and adversarial/validation results. |
| C9 | FissionSpec improves throughput/TBT on the bundled synthetic profile. | Matched counter-RNG synthetic factorial, currently three seed clusters; rigorous driver discloses pilot intervals. | “On this labeled synthetic model surface, the observed means change by the reported amount; intervals and pilot warning accompany every cell.” | Adequate independent clusters and precision; still never a GPU claim. |
| C10 | FissionSpec physically eliminates padded target computation. | Logical simulator slots only. | “The design removes recovering rows from logical target admission.” | GPU gate F1: physical descriptor changes and measured target-step effect. |
| C11 | FissionSpec improves production serving goodput or tail latency. | No accelerator evidence. | Do not claim. | Pre-registered H100 matched-resource confirmatory study; B200 only for transport. |
| C12 | FissionSpec is sampling-lossless on real models. | Exact finite CPU oracle only. | “Algorithmically exact in the finite rational oracle.” | Frozen real-model greedy equality and sampled distribution family under production kernels. |
| C13 | FissionSpec is the first outcome-decoupled zero-padding SSD batch scheduler. | Dated primary-literature matrix; prospective gap. | “To our knowledge at the 2026-07-23 cutoff, the reviewed systems do not combine the listed five properties.” | Repeat search before submission/rebuttal; weaken immediately if overlapping work appears. |
| C14 | CPU preparation minimizes GPU expenditure. | Spend-gated preregistration, deterministic replay/hashes, sequential confidence bounds and stop rules. | “The protocol is designed to stop before broad GPU evaluation when physical feasibility, mechanism, or controller-transport gates fail.” | Actual spend ledger after execution. |

## Headline admission rule

A headline systems claim may cite C10–C12 only when:

1. the exact code/data/profile hashes are archived;
2. the registered paired family and stop rule are used;
3. total target-plus-draft resources are matched;
4. intervals—not means alone—support the wording;
5. semantic tests pass on the same engine revision; and
6. negative model-pair or workload regions remain visible.

## Evidence labels

- `exact-finite-domain`: rational enumeration or exhaustive bounded search;
- `invariant-model`: executable state/ownership/fault property;
- `simulation-model`: event simulator using synthetic or calibrated surfaces;
- `cpu-statistical-analysis`: inference over independent simulation/trace
  clusters;
- `gpu-microbenchmark`: kernel/graph/transport calibration, not end-to-end
  policy evidence; and
- `gpu-confirmatory`: matched-resource pre-registered serving evaluation.

Artifacts must carry exactly one primary evidence label plus an explicit warning
when that label is not an end-to-end measurement.
