# FissionSpec GPU-free completion study

> **SIMULATION MODEL OUTPUT — NOT A GPU MEASUREMENT.**

All values are deterministic CPU simulation/model outputs. They do not measure GPU kernels, CUDA graphs, accelerator memory, power, or production throughput.

## Bundle scope

- Mode: `full`
- Fractional-factorial cells: 12
- Independent paired seed/trace clusters per cell: 30
- Policy rows: 2880
- Fidelity rows: 360
- Exact bounded oracle cells: 6

The policy table has two non-comparable strata. Decoder-policy rows may be
compared only with `saguaro-barrier`; scheduler-abstraction rows may be
compared only with `fifo-ordinary-reference`. The study never ranks a
policy from one harness against a policy from the other.

Validation headline intervals use paired seed/trace clusters and a
Bonferroni-declared family. Training cells and non-headline policies are
exploratory. `uncertainty.json` contains effect sizes, interval metadata,
RNG provenance, and resample fingerprints.

## Validation headline: H2 versus barrier

Positive values mean oriented improvement: more throughput, lower latency,
or fewer misses. Intervals are simultaneous within the predeclared
validation family. They quantify paired simulator-seed variation only.

| workload | metric | paired mean improvement | simultaneous interval | clusters |
|---|---|---:|---:|---:|
| synchronized | throughput tokens/s | -92.9625 | [-139.028, -43.8205] | 30 |
| synchronized | p95 request latency ms | -0.968333 | [-2.10765, +0.137208] | 30 |
| synchronized | deadline-miss fraction | +0 | [+0, +0] | 30 |
| poisson | throughput tokens/s | +99.0232 | [+31.9516, +172.379] | 30 |
| poisson | p95 request latency ms | +1.71555 | [+1.03886, +2.42783] | 30 |
| poisson | deadline-miss fraction | +0 | [+0, +0] | 30 |
| mmpp-exact | throughput tokens/s | -29.8898 | [-107.503, +56.732] | 30 |
| mmpp-exact | p95 request latency ms | +5.3512 | [+2.49122, +8.05017] | 30 |
| mmpp-exact | deadline-miss fraction | +0 | [+0, +0] | 30 |
| pareto-heavy-tail | throughput tokens/s | -448.913 | [-500.58, -391.343] | 30 |
| pareto-heavy-tail | p95 request latency ms | -6.22743 | [-7.8429, -4.5481] | 30 |
| pareto-heavy-tail | deadline-miss fraction | +0 | [+0, +0] | 30 |
| heterogeneous | throughput tokens/s | +85.5347 | [+23.1264, +145.814] | 30 |
| heterogeneous | p95 request latency ms | +2.87532 | [+1.73636, +4.15711] | 30 |
| heterogeneous | deadline-miss fraction | +0 | [+0, +0] | 30 |
| trace-replay | throughput tokens/s | +46.963 | [+28.8264, +64.9354] | 30 |
| trace-replay | p95 request latency ms | +1.68704 | [+1.00532, +2.39723] | 30 |
| trace-replay | deadline-miss fraction | +0 | [+0, +0] | 30 |

These model outcomes do not support universal H2 dominance: the signed
effect changes across validation workloads. That negative result is
preserved rather than filtered.

## Fidelity and bounded exact checks

- Fidelity cache-hit observations span `0` to `1`.
- Fidelity p95 TTFT observations span `1.86748` to `28.505` ms.
- Exact generalized-oracle certificates: 6.
- Deterministic adversarial witnesses reproduced: 3.

## Reproduction

```bash
PYTHONPATH=src python experiments/run_cpu_completion_study.py --mode full
PYTHONPATH=src python experiments/run_cpu_completion_study.py --verify-only \
  experiments/results/cpu_completion_full
```

Every event/request trace is retained in deterministic `traces.jsonl.gz`.
The manifest hashes every artifact. Wall-clock runtime is printed by the
driver but excluded from the bundle so golden reruns are byte-identical.
