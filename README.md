# FissionSpec

**Outcome-decoupled continuous batching for speculative-speculative LLM serving.**

> One wrong prediction should not make every correctly predicted request wait—or
> spend a padded target-model slot.

Speculative-speculative decoding (SSD) overlaps draft generation with target
verification by preparing continuations for likely verification outcomes. The
new failure mode appears only when SSD meets online batching: each request hits
its outcome cache independently, but existing execution policies coordinate the
next step at batch granularity.

FissionSpec turns those lookups into per-request scheduling events. Hits may
immediately enter another target batch; misses recover on the draft side without
occupying a target row; compatible work is re-fused only when a bounded
horizon-2 controller predicts that batching efficiency outweighs queueing and
SLO cost.

This repository is an executable research artifact, not a wrapper around an
agent framework. It contains:

- a dependency-free, deterministic discrete-event simulator for online SSD;
- an exact rational token-level speculative-sampling oracle for tiny CPU models;
- a deterministic, randomly initialized no-download neural micro-model fixture;
- a Saguaro barrier baseline and a labeled SPECTRE parallel-mode component;
- immediate-fission, fixed-coalescing, and horizon-2 FissionSpec policies;
- an exact restricted small-trace binary-coalescing oracle;
- a versioned copy-on-write KV transaction ledger with ABA-safe page handles;
- a composed coordinator with canonical crash snapshots and injected-fault tests;
- per-request counter-based randomness for schedule-independent experiments;
- Poisson, exact MMPP, Pareto, and split-aware hash-linked replay workloads;
- paired cluster inference, sequential stopping, and power-planning primitives;
- dependency-free Rust controller/allocator primitives over flattened work units;
- a calibration format, experiment harness, theory checks, and paper plan; and
- a narrow integration contract for production PagedAttention engines.

## The systems observation

For a batch of `B` requests with iid outcome-cache hit probability `p`, the
chance that Saguaro's batch fallback is invoked is `1 - p^B`. If recovery takes
`R`, the aggregate head-of-line wait is:

```text
batch barrier:       B R (1 - p^B)
per-request fission: B R (1 - p)
reduction factor:   (1 - p^B) / (1 - p)
```

At `p = 0.95` and `B = 32`, the iid expected stalled-row-work ratio is
**16.1×** under the batch barrier. Conditional on exactly one miss, the ratio
is `B = 32`. SPECTRE removes much of that wait in parallel mode, but
keeps recovering requests in a mixed target batch as one-token candidates
padded to the verification width. FissionSpec asks when removing those rows
entirely is worth the smaller target batch and extra launch.

## Quick start

FissionSpec requires Python 3.11+ and Rust only for the native reference core.
The Python simulator has no runtime dependencies.

```bash
python3 -m pip install -e ".[dev]"
make check
```

Inspect the analytical miss amplification:

```bash
fissionspec theory --batch-size 32 --hit-rate 0.95
```

Run one deterministic online simulation:

```bash
fissionspec simulate --policy fissionspec --requests 512 --seed 7
```

Regenerate the matched-seed policy sweep and controller decision surface:

```bash
make artifacts
```

The command surface is also available without installation:

```bash
PYTHONPATH=src python3 -m fissionspec --help
```

## What is actually novel here?

The claim is intentionally narrower than “async speculative decoding” or
“transactional KV cache.” Those exist.

- [Saguaro](https://arxiv.org/html/2603.03251) introduced SSD outcome caches and
  explicitly models batch-wide fallback.
- [SPECTRE](https://arxiv.org/html/2605.08151) already performs per-request
  rollback recovery, but its parallel mode retains misses as padded target rows
  and chooses mode at batch granularity.
- [EXSpec](https://arxiv.org/html/2510.22876) already pools and re-groups ragged
  sequences for ordinary batched speculative decoding.
- [FASER](https://arxiv.org/html/2604.20503) already adapts per-request draft
  lengths inside a still-coupled batch.
- [TransKV](https://doi.org/10.36227/techrxiv.177101038.80960856/v1) already
  separates committed and provisional speculative KV state.

FissionSpec's research question is the intersection those systems leave open:

> Can batched SSD eliminate both wait and padded-compute externality by making
> each outcome lookup independently schedulable, without destroying target
> batching efficiency or sampling correctness?

The full, dated claim boundary is in [docs/novelty.md](docs/novelty.md).

## Architecture

```text
target verification
        |
        v
 per-request outcome events
    /          |          \
 hit       recovering    finished
  |             |
  |       remote fallback
  |             |
  +---- ready-backup -----+
              |
      horizon-2 controller
             / \
      dispatch   bounded wait
             \ /
       next target batch
```

Two components provide the prerequisites for safe engine reordering:

1. every state-changing message is keyed by `(request, round, version)`, and
   stale asynchronous completions are ignored;
2. every stochastic draw is keyed by logical request/round coordinates rather
   than global execution order.

The count-level performance simulator does not sample token IDs. A separate
exact CPU oracle executes real speculative rejection sampling and exhaustively
matches the target distribution for tiny rational autoregressive models.
Production-kernel numerical parity remains an integration acceptance gate. See
[docs/architecture.md](docs/architecture.md) for the state machine and engine
boundary, and [docs/token_semantics.md](docs/token_semantics.md) for the exact
semantic contract. The neural-to-exact CPU smoke seam is documented in
[docs/micro_model.md](docs/micro_model.md).

## Artifact status

The simulator can test scheduling mechanics, invariants, adverse traces, and
controller behavior on a laptop. It cannot establish a GPU speedup. Any numbers
produced by the bundled synthetic profile are labeled **model results**, and
the CLI embeds its profile, workload, seed, and evidence warning in every run.
On that profile, immediate fission/H2 improves modeled throughput by
1.2%–4.3% over the barrier across the checked-in factorial while eliminating
directly attributed hit delay and padded slots. H2 matches immediate fission in
this default slow-recovery regime; the controller phase diagram records the
synthetic parameter region in which bounded re-fusion is selected. See the
[factorial summary](experiments/results/SYNTHETIC_RESULTS.md) and
[controller phase diagram](experiments/results/controller_phase_diagram.svg).
These are mechanism sanity checks, not performance claims.

Conference-grade claims still require:

- latency calibration from SGLang or vLLM kernels;
- an implementation that physically omits miss rows from CUDA-graph inputs;
- matched-resource H100/B200 experiments against Saguaro and SPECTRE; and
- greedy-equivalence plus distributional tests on real model/kernel outputs.

That distinction is deliberate: the repository is designed to make the next
GPU experiment falsifiable rather than manufacture a benchmark from hardware
that is not present.

## Repository map

```text
src/fissionspec/            simulator, policies, state protocol, KV ledger
crates/fissionspec-core/    flattened-work Rust controller/allocator primitives
experiments/                matched-seed sweeps and artifact tables
configs/                    synthetic and calibrated latency profiles
integrations/               SGLang/vLLM production-engine contracts
tests/                      unit, invariant, model, and adversarial tests
docs/                       architecture, semantics, theory, traces, novelty
paper/                      manuscript plan and GPU pre-registration
```

The finite CPU-exhaustion boundary is
[docs/cpu_completion.md](docs/cpu_completion.md). Full event artifacts and
replay inputs are defined in [docs/trace_schema.md](docs/trace_schema.md), exact
analytical results in [docs/theory.md](docs/theory.md), and the spend-gated
accelerator handoff in
[paper/gpu_preregistration.md](paper/gpu_preregistration.md).

## Research path

The fastest route from this artifact to a submission is:

1. profile target verification and recovery latency surfaces on the exact
   target/draft pair;
2. replay open-loop Poisson, bursty, and production traces in the simulator;
3. integrate the event protocol at the vLLM/SGLang scheduler/block-manager
   boundary;
4. reproduce the one-miss-plus-`B-1`-hits mechanism experiment;
5. compare Saguaro barrier, full SPECTRE plus the padded-mode component,
   immediate fission, fixed timeout, FissionSpec, and the restricted
   small-trace oracle; and
6. report P99 TBT, gap- and request-level TBT attainment, and
   total-GPU-normalized goodput—not only batch-1 tokens/s.

See [paper/outline.md](paper/outline.md) for the full experiment matrix.

## License

Apache-2.0.
