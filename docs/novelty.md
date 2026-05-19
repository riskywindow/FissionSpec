# Novelty boundary (dated 2026-07-22)

This file is intentionally blunt. It records what FissionSpec may and may not
claim as the literature changes.

## Defensible thesis

FissionSpec studies **zero-padding, outcome-decoupled continuous batching for
batched speculative-speculative decoding**. It fissions outcome-cache hits from
misses, removes recovering misses from target verification, then re-fuses ready
requests under an SLO-bounded controller.

The primary object is *miss externality*: delay and padded target work that one
miss imposes on co-batched hits. The artifact reports the closed-form expected
stalled-row work, padded slots, paired TBT/flow metrics, and a narrowly named
`direct_hit_delay_ms` attribution; the latter is not conditional end-to-end
slowdown. Transactional KV state and asynchronous recovery are enabling
machinery, not independent novelty claims.

## Closest work

| System | What it already establishes | FissionSpec's remaining question |
|---|---|---|
| [Saguaro / Speculative Speculative Decoding](https://arxiv.org/html/2603.03251) | Outcome caches, async drafting, geometric fan-out, batch-wide fallback analysis | Can each lookup become an independent online scheduling event? |
| [SPECTRE](https://arxiv.org/html/2605.08151) | Per-request rollback recovery and hybrid ordinary/parallel modes | Can misses stop consuming padded target rows, with decisions below batch granularity? |
| [EXSpec](https://arxiv.org/html/2510.22876) | Pools and re-groups ragged ordinary-SD sequences | How should online SSD outcome misses recover and rejoin under deadlines? |
| [FASER](https://arxiv.org/html/2604.20503) | Per-request draft lengths, early pruning, fine-grained phase overlap | Its requests remain coupled by a shared batch context; it does not schedule outcome-cache misses. |
| [SwiftSpec](https://arxiv.org/html/2506.11309) | Disaggregated async speculation and tree-aware KV repair | It targets single-query/low-batch latency rather than multi-tenant miss externality. |
| [SpecBranch](https://arxiv.org/html/2506.01979) | Branch parallelism and rollback-aware reuse | It lowers rollback probability; FissionSpec schedules the divergent readiness after an outcome. |
| [TAPER](https://arxiv.org/abs/2605.06914) | SLO-aware regulation of branch parallelism | It controls branch admission, not post-outcome recovery lanes. |
| [TurboSpec / SmartSpec](https://arxiv.org/html/2406.14066) | Load-aware speculation length and goodput control | It does not decouple requests after SSD outcome-cache lookup. |
| [TransKV](https://doi.org/10.36227/techrxiv.177101038.80960856/v1) | Committed versus packed provisional KV state | FissionSpec uses versioned transactions as a scheduling substrate. |

## Claims to avoid

FissionSpec is not the first system to provide:

- asynchronous or disaggregated speculative decoding;
- per-request fallback;
- dynamic re-batching;
- transactional speculative KV state;
- SLO-aware speculative control;
- a multi-tenant remote drafter; or
- all forms of “barrier-free” decoding.

The safe prospective claim is narrower:

> FissionSpec turns SSD outcome-cache lookups into per-request scheduling events,
> so recovering misses can be removed from target batches rather than stalling
> them or occupying padded verification slots, while an SLO-bounded controller
> decides when to re-fuse ready work.

This remains a research hypothesis until validated in a production GPU engine.
