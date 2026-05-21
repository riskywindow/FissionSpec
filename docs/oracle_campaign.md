# Expanded exact scheduler-oracle campaign

`experiments/run_oracle_campaign.py` expands the six exact validation
certificates in the completion study into 216 distinct finite scheduling
problems. It evaluates six controller settings and six policy views per problem,
for 7,776 retained comparisons.

The exact domain is deliberately small and precise. Each active job is one
non-preemptive target row that completes in one launch. A problem contains two
or three active jobs. At every idle state, the dynamic program enumerates every
ordered nonempty subset of released jobs satisfying row and verifier-slot
capacity, plus the frozen finite wait set. The objective is lexicographic:
deadline violations, exact weighted flow, then the canonical trace. All
objective arithmetic and latency surfaces use `fractions.Fraction`.

The matrix varies:

- synchronized, staggered, and recovery-wave releases;
- head, tail, and double cache misses;
- exact recovery ETA and deterministic recovery jitter;
- a pre-realized retry that succeeds after one failed remote attempt;
- pre-scheduling cancellation and terminal remote failure;
- packed physical slots versus aggregate next-power-of-two graph buckets;
- tight versus wide row/slot capacity; and
- controller maximum wait and deadline guard.

Cancellation and terminal failure are not live stochastic transitions in this
oracle. They are pre-realized before scheduling: a canceled or terminally
failed row is absent, while a retry-success row receives its exact fallback
release time. This is the strongest failure/cancellation statement supported by
the one-shot model.

Every problem is solved by the exact memoized dynamic program and replay
verified. A predeclared, stratified set of 24 certificates is independently
re-enumerated without memoization or permutation-dominance pruning. It contains
all 16 outcome/physical/capacity/two-arrival cells for the two-job cancellation
and terminal-failure problems, plus all eight outcome/capacity cells for the
three-job cache/retry problems under graph-bucket widths. This retains every
outcome and both physical/capacity modes without spending the proof budget on
the exponentially larger packed-width three-job enumeration. The campaign also
checks input-order invariance, uniform time-shift invariance, capacity
monotonicity, and the expected equivalence of immediate-dispatch components.

The policy inventory includes EDF and every built-in Python dispatch policy.
Saguaro and SPECTRE results cover only their immediate-dispatch timing
component: barrier recovery and padded target execution are outside the
one-shot state. Multi-round baselines with remote queues, the restricted
simulator oracle, and the Rust hot path have different state/objective domains
and are not assigned misleading cross-model regret values.

Reproduce and verify:

```bash
PYTHONPATH=src python experiments/run_oracle_campaign.py \
  --mode full \
  --output-dir experiments/results/oracle_campaign
PYTHONPATH=src python experiments/run_oracle_campaign.py \
  --mode full \
  --output-dir experiments/results/oracle_campaign \
  --verify
```

The result bundle retains exact certificates, every comparison, regret
distributions, canonical counterexamples, coverage/search accounting,
environment provenance, source hashes, and a closed file manifest. Exactness
does not extend to multi-round decoding, KV allocation, target preemption,
remote draft queueing, CUDA graphs, or GPU performance.

Verification is semantic rather than hash-only. It rejects unexpected entries,
directories and symlinks, malformed or duplicate-key JSON, non-finite JSON
numbers, CSV schema drift, source drift, and count drift. It then re-solves all
216 exact problems, replay-verifies every certificate, independently proves the
same stratified 24 optima without memoization, recomputes all 7,776 policy
comparisons and derived summaries, and byte-compares every host-independent
deterministic artifact. `environment.json` is instead checked against its exact
schema and pinned reproduction contract so a Linux archival verifier can check
a bundle generated on macOS. Consequently, changing a semantic artifact and
updating its manifest hash is still detected.
