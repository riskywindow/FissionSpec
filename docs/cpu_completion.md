# CPU-completion program

This document defines the finite, testable boundary for exhausting useful
FissionSpec work before renting GPU capacity. It deliberately does not interpret
“everything possible” as unbounded polish. A work package is complete when its
acceptance gates pass and the remaining uncertainty fundamentally depends on
real accelerator kernels, CUDA-graph behavior, or large-model measurements.

## A. Token-level semantic correctness

- Execute real speculative rejection sampling over an exact finite
  autoregressive distribution.
- Address every random draw by logical request/round coordinates.
- Prove exhaustive distribution equality against target-only decoding for tiny
  vocabularies and horizons.
- Check greedy outputs, sampled distributions, outcome-cache selection, and
  committed-state digests under barrier, immediate-fission, and delayed/refused
  schedules.
- Include a randomly initialized CPU micro-model smoke test without downloading
  weights.

Acceptance: exhaustive probabilities agree exactly; greedy and state digests
are identical; deterministic Monte Carlo checks remain within declared
statistical bounds.

## B. Composed coordinator and fault safety

- Compose protocol epochs, speculative KV transactions, scheduling lanes, block
  descriptors, target completion, and recovery completion behind one executable
  coordinator.
- Define one atomic publication point.
- Serialize durable snapshots canonically with a checksum and schema version.
- Inject duplication, delay, reordering, cancellation, OOM, truncation,
  corruption, and crash/restart at every state boundary.

Acceptance: exactly-once token commit, monotone epochs, stale-message inertness,
no leaked or multiply owned pages, deterministic replay, and eventual completion
for every non-cancelled bounded trace.

## C. Model fidelity

- Model outcome-tree fanout, cache capacity/eviction, provisional KV footprint,
  context-dependent costs, and correlated hit/acceptance classes.
- Replace the single FIFO draft abstraction with a configurable
  continuous-batched/priority remote service, multiple workers, network jitter,
  backpressure, and failures.
- Add prefill/TTFT and heterogeneous context/output lengths without confusing
  them with steady-state TBT.

Acceptance: every abstraction has an explicit null setting that reproduces the
reference simulator; conservation and capacity invariants hold under randomized
traces.

## D. Baselines, oracle, and controller evidence

- Add full scheduler-level abstractions for SPECTRE hybrid selection,
  EXSpec-style grouping, and a myopic slack controller.
- Extend tiny-trace search to arbitrary admissible subsets/orderings and bounded
  wait points.
- Report FissionSpec optimality gaps, adversarial traces, starvation behavior,
  and break-even regions.
- Define and test the common semantic subset between Python and Rust rather than
  implying the two cost models are identical.

Acceptance: matched outcomes across policies; exact oracle certificates for
bounded traces; deterministic counterexamples for every known heuristic limit.

## E. Statistical and workload validity

- Use paired uncertainty intervals, effect sizes, deterministic resampling,
  replication planning or precision stopping, and an explicit multiplicity
  policy.
- Sweep offered load, batch capacity, speculation length, SLO, output/context
  length, recovery and network latency, cache budget, fanout, physical slot
  cost, and controller parameters.
- Include synchronized, Poisson, MMPP/bursty, heavy-tailed, heterogeneous, and
  trace-replay workloads with scenario train/validation separation.
- Preserve per-request/event traces and sufficient statistics so every table is
  independently recomputable.

Acceptance: no headline cell is supported by means alone; all schemas carry
source/configuration hashes and evidence warnings; golden artifacts reproduce
byte-for-byte.

## F. Theory

- Generalize miss externality to heterogeneous and correlated outcomes and
  random recovery times.
- Derive padding-versus-fission break-even conditions.
- State liveness, bounded-wait, SLO-safety, and controller-complexity results
  with machine-checked finite-domain validation.
- Bound or empirically characterize the horizon-2 controller’s gap to the exact
  bounded oracle.

Acceptance: every theorem has assumptions, a derivation, numerical edge-case
tests, and a linked experiment or counterexample.

## G. Release and reproducibility

- Build and install sdist/wheel artifacts in clean environments.
- Publish a typed public API and stable schemas.
- Gate coverage, property/state-machine tests, Rust MSRV, documentation, and
  package smoke tests in CI.
- Record implementation, inputs, RNG, environment, and toolchain provenance for
  every experiment.
- Provide a dependency-locked container or equivalent archival environment.

Acceptance: a clean checkout can reproduce all CPU evidence with one command.

## H. Manuscript and claim discipline

- Write the full manuscript with algorithm pseudocode, proofs, evaluation
  protocol, claims-to-evidence matrix, limitations, and artifact instructions.
- Maintain a systematic, dated literature matrix with search strings, paper
  sections, and artifact links.
- Pre-register GPU hypotheses, stopping rules, graph buckets, model pairs, and
  matched-resource accounting before measurements are collected.

Acceptance: GPU runs fill predetermined table cells rather than changing the
methodology after observing results.

## Genuine GPU boundary

The following remain after this program:

1. calibrating target, draft, recovery, packing, and CUDA-graph latency surfaces;
2. verifying that logical row removal changes physical kernel/graph work;
3. measuring large-model output quality and numerical behavior on production
   engine kernels;
4. matched-resource H100/B200 throughput, latency, utilization, power, and
   memory measurements; and
5. validating production integration under real accelerator OOM and kernel
   failure modes.

No synthetic or CPU result may be relabeled as evidence for those claims.
