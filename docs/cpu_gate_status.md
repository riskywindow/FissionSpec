# GPU-free completion gate status

**Audit date:** 2026-07-23

This ledger maps every finite gate in `docs/cpu_completion.md` to executable
evidence. A gate is marked complete only when its tests and artifact contract
pass. “Complete” means complete for the explicitly modeled domain; it never
promotes a simulator result to accelerator evidence.

| Gate | Status | Executable evidence | Remaining CPU work |
|---|---|---|---|
| A. Token semantics | Complete | `semantics.py`, `micro_model.py`; exhaustive rational equality, schedule invariance, greedy/state equality, counter-addressed Monte Carlo, no-download neural-logit fixture | None inside the finite CPU domain |
| B. Coordinator/fault safety | Complete | `coordinator.py`, `ledger.py`, `protocol.py`; canonical snapshot and enumerated crash/callback/OOM/cancel tests | Production durable storage and distributed leases are integration work, not reference-model claims |
| C. Fidelity | Complete for declared one-round/reference strata | `fidelity.py`; finite paged LRU, correlated classes, context costs, causal multiworker draft service, jitter/failure/retry/backpressure, heterogeneous TTFT, exact null reductions | A unified multi-round engine event loop would extend fidelity but is not required for the one-round mechanism claims |
| D. Baselines/oracle/controller | Core complete; parity audit active | `baselines.py`, `general_oracle.py`; matched pre-realized semantics, exact certificates, independent verifier, deterministic counterexamples | Shared Python/Rust fixture corpus and engine-seam state enumeration are in progress |
| E. Statistics/workloads | Active | `statistics.py`, `workload_generators.py`, `artifacts.py`; paired effects, cluster bootstrap, confidence sequence, power planning, MMPP/Pareto/replay | Full predeclared 30-cluster matrix, held-out results, and byte-golden study artifacts are in progress |
| F. Theory | Core complete; empirical-gap study active | `theory.py`, exact rational edge tests, `general_oracle.py` | Populate broad bounded-oracle gap and adversarial phase evidence |
| G. Release/reproducibility | Active | existing Python/Rust CI and one-command checks | Clean sdist/wheel installs, typed package marker/API, coverage gates, MSRV matrix, archival environment, and complete CPU reproduction target are in progress |
| H. Manuscript/claims | Structure complete; result population active | `paper/manuscript.md`, `claims_evidence.md`, dated `literature_matrix.md`, frozen `gpu_preregistration.md` | Replace only CPU placeholders with generated artifacts; leave all GPU cells unmeasured |

## Current zero-GPU release decision

Do not rent an accelerator yet. The semantics, reference state machine,
fidelity abstractions, competing schedulers, and exact scheduling oracle pass
their local gates, but the broad CPU evidence and clean-release gates are still
active. GPU Stage 1 becomes admissible only after every active row above is
closed at one tagged commit and a clean reproduction confirms its hashes.

## Irreducible accelerator boundary

Once every row is closed, remaining uncertainty is restricted to:

1. real target/draft/recovery/packing/network latency surfaces;
2. whether removing a logical row changes a physical kernel descriptor,
   CUDA-graph bucket, or measured target time;
3. real-model numerical and distribution behavior under production kernels;
4. matched-resource throughput, latency, memory, power, and cost; and
5. actual engine OOM, worker loss, and kernel-failure behavior.

Those measurements are staged by `paper/gpu_preregistration.md`; a failed
physical-mechanism gate aborts the expensive serving campaign.
