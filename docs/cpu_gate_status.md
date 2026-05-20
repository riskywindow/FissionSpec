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
| D. Baselines/oracle/controller | Complete for declared CPU abstractions | `baselines.py`, `general_oracle.py`, shared Python/Rust fixtures, engine-adapter state enumeration, six exact certificates, independent verifier, deterministic counterexamples | Production baseline fidelity and engine timing require the pinned integration, not another CPU abstraction |
| E. Statistics/workloads | Complete | `statistics.py`, `workload_generators.py`, `artifacts.py`, `run_cpu_completion_study.py`; 12 cells × 30 paired clusters, MMPP/Pareto/replay validation, simultaneous intervals, 3,240 complete traces, byte-golden bundle | None inside the frozen CPU study; real trace transport remains a registered integration measurement |
| F. Theory | Complete for stated finite/analytical domain | `theory.py`, exact rational edge tests, `general_oracle.py`, six independently verified H2 gap instances and three adversarial witnesses | No universal H2 approximation guarantee is claimed; stronger guarantees would require additional assumptions |
| G. Release/reproducibility | Active | existing Python/Rust CI and one-command checks | Clean sdist/wheel installs, typed package marker/API, coverage gates, MSRV matrix, archival environment, and complete CPU reproduction target are in progress |
| H. Manuscript/claims | CPU results populated; final release audit active | `paper/manuscript.md`, `claims_evidence.md`, dated `literature_matrix.md`, frozen `gpu_preregistration.md`, fail-closed `spend_gate.py`, hash-linked CPU result table | Leave every GPU cell unmeasured until its registered stage; re-run literature search at submission |

## Current zero-GPU release decision

Do not rent an accelerator yet. The semantics, reference state machine,
fidelity abstractions, competing schedulers, exact scheduling oracle, broad
workload study, and manuscript population pass their local gates. The
clean-release and pinned production-patch gates are still active. GPU Stage 1
becomes admissible only after those rows close at one tagged commit and a clean
reproduction confirms all hashes.

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
