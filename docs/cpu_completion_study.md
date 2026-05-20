# GPU-free completion study

`experiments/run_cpu_completion_study.py` is the finite, reproducible CPU
evaluation gate for FissionSpec. It is designed to minimize later accelerator
spend: every workload, policy, cache, transport, statistical, and bounded-oracle
question that can be answered by the repository's models is exercised before a
GPU is rented.

> **SIMULATION MODEL OUTPUT — NOT A GPU MEASUREMENT.**

The study cannot establish kernel throughput, CUDA-graph behavior, accelerator
memory use, power, or production-engine integration. Those are the remaining
GPU-dependent questions.

## Evidence strata

The bundle keeps three model contracts separate.

| stratum | reference | compared mechanisms | scope |
|---|---|---|---|
| decoder-policy simulator | `saguaro-barrier` | immediate fission, fixed coalescing, FissionSpec H2 | multi-round target/draft policy timing |
| pre-realized scheduler abstraction | `fifo-ordinary-reference` | SPECTRE hybrid, EXSpec sliding pool, myopic slack | batch membership, padding, realignment, and shared-draft scheduling |
| one-round fidelity model | none | outcome-tree cache and remote draft transformer | prefill, TTFT, fanout, finite cache, network, backpressure, failure, and retry |

No result ranks a policy from one stratum against a policy from another. The
main simulator's narrow policy interface cannot honestly express EXSpec pool
membership or SPECTRE's shared remote-draft arbitration. Conversely, the
pre-realized scheduler harness is not a drop-in replacement for the main
multi-round event semantics. The schema records the harness and its only valid
comparison reference on every row, and the manifest sets
`cross_harness_ranking_permitted` to `false`.

Every scheduler abstraction consumes the same immutable `PreRealizedTrace`.
`assert_semantic_equivalence` verifies the complete emitted-token and outcome
signature after all four scheduler runs.

## Predeclared design

The scenario matrix is a deterministic 12-run Plackett-Burman screening design
over 11 binary factors:

1. low/high offered load;
2. target row capacity;
3. speculation width;
4. TBT SLO;
5. prompt length;
6. output length;
7. recovery-cost scale;
8. network-cost scale;
9. finite outcome-cache budget;
10. outcome-tree fanout; and
11. physical verifier-slot cost.

Every factor column has six low and six high assignments. A separate
short/medium/long blocking factor changes fixed-coalescing and H2 wait limits;
it is explicitly not claimed to be orthogonal.

Each workload family has one train cell and one held-out validation cell:

- synchronized cohorts;
- exact Poisson arrivals;
- exact continuous-time two-state MMPP arrivals;
- finite-mean Pareto heavy-tail arrivals;
- heterogeneous prompt, output, width, acceptance, hit, and SLO requests; and
- exact CSV replay, including the source-byte SHA-256 and declared split.

The workload RNG, semantic outcome RNG, fidelity RNG, and bootstrap RNG use
separate, counter-addressed domains. A policy cannot perturb another policy's
counterfactual outcomes by consuming a mutable random stream.

## Computational modes

The modes are fixed in code rather than selected after results are observed.

| mode | paired clusters per cell | request count for generated traces | bootstrap resamples | oracle jobs |
|---|---:|---:|---:|---:|
| `ci` | 2 | 6 | 100 | 4 |
| `full` | 30 | 16 | 2,000 | 6 |

The replay cells retain their source row counts rather than synthesizing extra
requests. A full run produces:

- 12 scenario cells;
- 360 independent paired seed/trace clusters;
- 2,880 policy metric rows;
- 360 fidelity metric rows; and
- 3,240 complete trace records.

The checked-in full bundle completed in approximately 45 seconds on the
development CPU and occupies approximately 10 MiB. Runtime is printed to the
terminal but deliberately excluded from artifacts, because wall time is not a
semantic input and would break golden reproduction.

## Statistical contract

The independent experimental unit is one seed/trace cluster. Requests and
events inside a trace are never counted as independent replications.

Within the decoder-policy stratum, every candidate is paired against the
barrier reference. Within the scheduler stratum, every abstraction is paired
against FIFO. The output includes:

- paired mean effects oriented so positive always means improvement;
- paired standardized effect size and probability of improvement;
- deterministic paired-cluster percentile intervals;
- exact bootstrap seed provenance and resample fingerprints; and
- the complete per-cluster sufficient metrics needed to recompute every
  comparison.

The predeclared headline family contains H2-versus-barrier comparisons on the
six validation cells for three metrics:

- throughput, where higher is better;
- p95 request latency, where lower is better; and
- deadline-miss fraction, where lower is better.

Those 18 intervals use a Bonferroni-declared simultaneous confidence level for
a familywise alpha of 0.05. The family is marked non-confirmatory because the
latency surface remains a CPU model. Training cells and non-H2 policy
comparisons are exploratory 95% intervals.

Fidelity cells also carry cluster intervals for cache-hit rate, p95 TTFT, and
terminal failed jobs. Thus no validation headline or fidelity cell is supported
by a mean alone.

The generated summary intentionally preserves negative results. In the full
CPU model, H2's oriented effect changes sign across validation workloads; the
artifact does not filter cells to imply universal dominance.

## Fidelity companion

For every policy cluster, the study builds one matching fidelity trace from the
same workload description. It exercises:

- a two-class correlated acceptance/outcome mixture;
- shared tenant correlation keys;
- finite page-rounded LRU outcome-tree storage;
- one or three cached outcome branches;
- small and large cache budgets;
- heterogeneous context and verifier width costs;
- one or two non-preemptive remote draft workers;
- continuous batching with a bounded admission queue;
- request/response transport latency and keyed jitter;
- backpressure;
- failure, retry, backoff, and terminal failure; and
- serialized prefill plus first-token latency.

The fidelity layer is a one-round transformer/service harness. It does not
claim to be a competing multi-round scheduling policy.

## Generalized oracle and adversarial evidence

Every validation workload cell has a bounded exact generalized-oracle problem.
It includes:

- arbitrary admissible target subsets and orderings;
- row and physical slot capacities;
- exact rational release, deadline, weight, and latency values;
- explicit release-time wait points;
- a lexicographic deadline-violation/weighted-flow objective;
- a hash-linked optimal certificate; and
- independent brute-force certificate verification.

The study evaluates both work-conserving EDF and an H2 one-shot schedule
against that optimum and records component-wise gaps. This is a bounded,
one-shot scheduling result, not a proof that the multi-round serving controller
is globally optimal.

Three deterministic counterexamples are regenerated on every run:

1. a fixed SPECTRE rollback threshold selects parallel mode for a wide padded
   row whose physical-slot cost makes it slower than ordinary mode;
2. EXSpec homogeneous grouping delays an old unique-length sequence; and
3. myopic fairness promotion places a wide old request before a new tight
   request and causes the latter to miss its deadline.

All counterexamples preserve the exact same semantic output signature between
the compared schedules.

## Artifact bundle

`experiments/results/cpu_completion_full/` contains:

| file | contents |
|---|---|
| `manifest.json` | canonical envelope and SHA-256/byte count for every artifact |
| `design.json` | complete mode, factor, split, source, implementation, and harness declaration |
| `metrics.csv` | per-cell, per-cluster, per-policy sufficient metrics |
| `fidelity_metrics.csv` | per-cell, per-cluster cache/TTFT/transport metrics |
| `uncertainty.json` | effects, intervals, multiplicity, RNG provenance, and fingerprints |
| `oracle.json` | exact problems, certificates, independent verification, schedules, and gaps |
| `adversarial.json` | three deterministic limitation witnesses |
| `traces.jsonl.gz` | every request/event trace for all three evidence strata |
| `environment.json` | Python, platform, and implementation provenance |
| `SUMMARY.md` | generated human-readable counts, intervals, and limitations |

Every JSON payload uses strict canonical serialization. The gzip stream fixes
its timestamp and filename fields. Every trace record has its own payload hash,
and every metric row links to that hash. The top-level manifest then hashes the
exact bytes of every artifact. `verify_bundle()` rejects missing files, byte
count changes, hash changes, path traversal, missing warnings, or an invalid
cross-harness comparison declaration.

The CI tests run the bundle twice and compare every artifact digest. A rerun is
therefore byte-identical for an unchanged implementation, input trace, and
environment.

## Reproduce

From the repository root:

```bash
PYTHONPATH=src python experiments/run_cpu_completion_study.py --mode ci
PYTHONPATH=src python experiments/run_cpu_completion_study.py --mode full
PYTHONPATH=src python experiments/run_cpu_completion_study.py \
  --verify-only experiments/results/cpu_completion_full
```

To keep output outside the checkout:

```bash
PYTHONPATH=src python experiments/run_cpu_completion_study.py \
  --mode full \
  --output-dir /absolute/path/to/bundle
```

Focused validation:

```bash
PYTHONPATH=src python -m unittest experiments.test_cpu_completion_study -v
ruff check experiments/run_cpu_completion_study.py \
  experiments/test_cpu_completion_study.py
PYTHONPATH=src mypy experiments/run_cpu_completion_study.py \
  experiments/test_cpu_completion_study.py
```

## Remaining GPU boundary

After this bundle passes, GPU time is still required for:

1. calibrating target, draft, recovery, packing, prefill, transport, and graph
   latency surfaces;
2. demonstrating that logical row fission actually removes physical
   accelerator work in the target engine;
3. measuring production CUDA-graph bucket and KV allocator behavior;
4. testing real model pairs, numerical kernels, output quality, and production
   failure modes; and
5. matched-resource H100/B200 latency, throughput, utilization, memory, power,
   and cost measurements under the preregistered stopping rules.

Synthetic or CPU-model results must not be relabeled as evidence for any of
those claims.
