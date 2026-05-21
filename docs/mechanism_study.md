# CPU causal mechanism study

`experiments/run_mechanism_study.py` freezes the remaining mechanism questions
that can be answered without an accelerator. It is a sensitivity experiment
inside executable CPU models, not GPU-performance evidence.

The study has two non-comparable strata:

- The decoder-policy simulator changes recovery latency, physical verifier-slot
  cost, target batch capacity, and controller maximum wait.
- The one-round fidelity harness changes outcome-cache byte budget, branch
  fanout, network jitter, and remote-worker failure probability.

Each alternate differs from its stratum's shared reference in exactly one
serialized field. The decoder workload combines sparse miss/hit recovery probes,
which exercise the refusion decision, with a later high-load Poisson burst,
which exercises capacity and physical-slot effects. The immutable composite
workload is held fixed within every contrast. Thirty seed clusters are paired
with counter-addressed random draws. This holds arrivals and all semantic random
variables fixed within a contrast even when scheduling changes. Fanout is
interpreted as a total intervention: changing it intentionally changes cache
occupancy, remote payload size, and draft branch work together.

The confirmatory family contains 48 endpoints: throughput, p95 TBT, and target
launches per request for each of eight decoder contrasts, plus three metrics for
each of eight fidelity contrasts.
Intervals use paired-cluster percentile bootstrap resampling with a Bonferroni
allocation over that complete family. The checked-in full run uses 20,000
resamples. Exact two-sided sign tests are Holm-adjusted over the same family.
Bootstrap coverage is an asymptotic approximation; the sign tests rely only on
independent cluster signs under their sign-null. Descriptive counters are
reported but are not silently promoted into the confirmatory family.

`restricted_next_ready_delay_ms` is the mean next-round readiness delay,
restricted at the predeclared 25 ms horizon. A nonterminal request whose remote
recovery ultimately fails contributes 25 ms rather than disappearing from the
mean. `next_round_unready_rate` separately reports those unresolved requests.

Reproduce and verify:

```bash
PYTHONPATH=src python experiments/run_mechanism_study.py \
  --mode full \
  --output-dir experiments/results/mechanism_study
PYTHONPATH=src python experiments/run_mechanism_study.py \
  --mode full \
  --output-dir experiments/results/mechanism_study \
  --verify
```

The bounded bundle contains the frozen design, all per-cluster sufficient
statistics, complete-trace payload hashes, simultaneous inference, environment
metadata, and a self-hashed closed manifest that hashes every other file. Every
predeclared result is retained regardless of sign, magnitude, or statistical
significance. The model cannot answer whether CUDA graphs, real KV allocation,
transport overlap, or accelerator kernels preserve these effects; those remain
explicit GPU campaign questions.
