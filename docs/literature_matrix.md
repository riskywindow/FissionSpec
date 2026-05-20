# Literature matrix and search log

**Search cutoff:** 2026-07-23

**Scope:** speculative decoding semantics, batched/ragged scheduling,
speculative-speculative or asynchronous execution, branch regulation, and
transactional/paged KV state.

This is a claim-boundary instrument, not a citation-count survey. A work is
included when it can invalidate novelty, supply a required baseline, or define
a correctness/integration obligation. Claims below are restricted to the
primary paper abstract/method and linked author artifacts.

## Search protocol

Sources searched:

- arXiv title/abstract records and paper HTML/PDF;
- OpenReview for accepted versions;
- author/project paper pages;
- linked official code repositories or pull requests; and
- TechRxiv/Crossref metadata for TransKV.

Exact query families:

```text
"speculative speculative decoding" LLM
"asynchronous speculative decoding" LLM serving
"batch speculative decoding" ragged regrouping
"hybrid ordinary parallel speculative serving"
"fine-grained phase management" speculative decoding
"verification interference" SLO-aware speculative batching
"interpretable latency model" speculative decoding serving
"cache-based drafting" speculative decoding runtime selection
"branch parallelism" LLM serving SLO
"closed-loop speculation control" goodput
"transactional KV" speculative decoding paged
"tree-based speculative inference" serving
"speculative sampling" rejection distribution
```

Snowball terms were `outcome cache`, `parallel mode`, `remote drafter`,
`rollback`, `accepted length`, `branch externality`, `PagedAttention`, and
`continuous batching`. Inclusion decisions are recorded below; surveys and
blogs do not support novelty claims.

## Closest systems

| Work | What is already established | Scheduling/semantic unit | Artifact checked | FissionSpec boundary and required use |
|---|---|---|---|---|
| [Saguaro / Speculative Speculative Decoding](https://arxiv.org/abs/2603.03251) (ICLR 2026) | SSD outcome prediction, cached next continuations, asynchronous draft/verify overlap, optimized Saguaro algorithm | Outcome tree inside an SSD request; paper analyzes batch fallback | Accepted [OpenReview version](https://openreview.net/pdf?id=aL1Wnml9Ef); no official code located in this pass | Defines the mechanism and batch-wide fallback baseline. FissionSpec may claim only post-outcome row scheduling and zero-padding removal. Read algorithm, Saguaro design, batch analysis, evaluation. |
| [SPECTRE](https://arxiv.org/abs/2605.08151) | Remote multi-tenant drafter, hybrid ordinary/parallel mode, priority scheduling, prompt compression, per-request rollback recovery | Mode chosen for a batch; parallel mode retains padded recovering rows | Linked [SGLang pull request](https://github.com/sgl-project/sglang/pull/22272) | Strongest direct systems baseline. Implement/measure the full hybrid selector and drafter scheduling, not only the padded component. FissionSpec asks whether decisions below batch granularity can remove recovering target rows. |
| [EXSpec / Batch Speculative Decoding Done Right](https://arxiv.org/abs/2510.22876) | Correctness conditions for ragged batch SD; EQSpec realignment; sliding pool and same-length regrouping | Per-sequence accepted length in ordinary SD | [eBay `spec_dec`](https://github.com/eBay/spec_dec) linked by paper | Required regrouping baseline and correctness reference. It does not by itself resolve SSD next-continuation cache misses or recovery lanes. Read synchronization proof, EQSpec, EXSpec pool, evaluation. |
| [FASER](https://arxiv.org/abs/2604.20503) | Per-request speculative length, early rejected-token pruning, verification frontiers, spatial draft/verify overlap | Per-request width within a dynamic continuous batch | vLLM prototype reported; official repository not located in this pass | Required fine-grained SD baseline. Its phase/width control must not be relabeled outcome-cache scheduling. Read design, frontier execution, controller, evaluation. |
| [SwiftSpec](https://arxiv.org/abs/2506.11309) | Disaggregated asynchronous speculation, parallel trees, tree-aware KV repair, latency-focused fused kernels | Primarily intra-request tree and disaggregated pipeline | Official repository not located in this pass | Establishes that async/disaggregated speculation and tree KV repair are prior art. Compare only where batch/resource regimes match. |
| [SpecBranch](https://arxiv.org/abs/2506.01979) | Hybrid drafting and rollback-aware parallel speculative branches | Intra-request parallel branches | Official repository not located in this pass | Branching can reduce rollback/miss exposure; FissionSpec handles readiness after an outcome. Use as related end-to-end baseline if code/model pairs permit. |
| [TAPER](https://arxiv.org/abs/2605.06914) | Per-step SLO/slack admission for intra-request branch parallelism and explicit branch externality | Extra branches admitted against current batch slack | [Stanford MAST paper page](https://mast.stanford.edu/pubs/taper/); no code linked there | Closest control-plane analogy. Prevents broad claims to “SLO-aware externality regulation”; FissionSpec regulates post-outcome rows, not branch width. |
| [TurboSpec](https://arxiv.org/abs/2406.14066) | Runtime profiling and closed-loop adjustment of speculative parallelism to maximize goodput | Speculation amount under observed load/environment | vLLM implementation reported; official artifact not located in this pass | Required adaptive speculation-control context. It controls how much to speculate, not SSD outcome-lane decoupling. |
| [WISP](https://arxiv.org/abs/2601.11652) | Distributed edge drafting, a verification-time estimator, and SLO/interference-aware greedy verification-batch construction | Whole ordinary-SD verification requests with heterogeneous draft and context lengths | POMACS 2026 paper and arXiv v2 checked; no official code linked from the paper record | Closest heterogeneous verification-scheduling baseline. It establishes that SLO-aware subset construction and verification interference are prior art. Its requests arrive before ordinary target verification; it does not schedule SSD continuation-cache outcomes or remove post-outcome recovering rows. |
| [TransKV](https://doi.org/10.36227/techrxiv.177101038.80960856/v1) | Committed paged KV separated from packed provisional speculative KV; commit accepted prefix and discard the rest | One speculative KV transaction | [TechRxiv preprint PDF mirror](https://d197for5662m48.cloudfront.net/documents/publicationstatus/307056/preprint_pdf/197af17fbcff12ca9844d78e933bd197.pdf); no public code located | Transactional KV cannot be claimed as novel. FissionSpec uses stronger version/page ownership machinery as a scheduling prerequisite. Treat capacity results as preprint evidence. |

## Foundational semantics and serving substrate

| Work | Established obligation | FissionSpec use |
|---|---|---|
| [Leviathan et al., Fast Inference via Speculative Decoding](https://arxiv.org/abs/2211.17192) | Exact speculative decoding can preserve target outputs while verifying draft tokens in parallel. | Semantic oracle must reduce to the same target distribution; cite for foundational algorithm. |
| [Chen et al., Speculative Sampling](https://arxiv.org/abs/2302.01318) | Modified rejection sampling with target-minus-draft residual preserves target sampling distribution. | The exact rational CPU oracle implements and exhaustively checks this rejection/residual/bonus structure. |
| [SpecInfer](https://arxiv.org/abs/2305.09781) | Tree-structured proposals and parallel target verification with model-quality preservation. | Outcome-tree fanout and tree verification are not novelty; include as tree baseline/context. Official implementation is in [FlexFlow](https://github.com/flexflow/FlexFlow/). |
| [PagedAttention / vLLM](https://arxiv.org/abs/2309.06180) | Dynamic paged KV allocation, sharing, and continuous serving improve memory utilization and batching. | Defines the production allocator/block-table boundary. FissionSpec's symbolic ledger must map to engine pages without claiming PagedAttention. |
| [Kong et al., Interpretable Latency Model](https://arxiv.org/abs/2605.15051) | Effective batch size can be inferred with Little's Law and SD latency decomposed into load-independent and load-dependent prefill, draft, and verification demand across serving load. | Required calibration/evaluation context. FissionSpec's symbolic curves cannot substitute for the paper's measured load-dependent surfaces; Stage F1 must fit and validate them on the pinned engine. |
| [Hybrid Verified Decoding](https://openreview.net/forum?id=vr5iRoUn0I) | Runtime payoff prediction selects between cache-derived and model-generated ordinary-SD drafts. | Cache-based draft selection and payoff prediction are prior art. Its cache contains reusable token continuations rather than predicted SSD verification outcomes, and it does not define post-outcome batch lanes. |

## Pairwise differentiation tests

These tests must remain true in the manuscript and implementation:

1. Removing the horizon-2 controller should leave a per-outcome immediate
   scheduler, not recreate Saguaro or ordinary SD.
2. Replacing miss removal with one-token padded rows should produce the
   SPECTRE-parallel component, while still lacking its full hybrid and remote
   priority design.
3. Regrouping only by accepted length should produce an EXSpec-style baseline
   without fabricating outcome-cache hits.
4. Adapting speculative width without outcome lanes is FASER/TurboSpec-like
   control, not FissionSpec.
5. Regulating branch fanout against slack is TAPER-like control; regulating
   recovered request readiness is the distinct axis.
6. Scheduling whole verification requests against SLO slack and heterogeneous
   lengths is WISP-like control; scheduling a realized SSD outcome after the
   verification result is the distinct event boundary.
7. A transactional KV overlay without independent outcome scheduling is
   TransKV-like substrate, not the proposed scheduler.

## Gaps that remain open at the cutoff

The reviewed works do not jointly provide all of:

- a per-request scheduling event at every realized SSD outcome-cache lookup;
- physical removal of recovering misses from the next target input rather than
  cohort waiting or padded one-token participation;
- bounded re-fusion using known recovery/readiness events and rolling TBT
  deadlines;
- a composed epoch/page protocol for reordering those rows; and
- evaluation of that combination against full SPECTRE hybrid, regrouping, and
  exact bounded scheduling.

This is a prospective gap, not proof that the method wins. The exact claim is
falsified if an earlier system performs the same post-outcome removal and
bounded re-fusion, or if calibrated kernels make row removal physically free of
benefit.

## Maintenance rule

Before submission and rebuttal:

1. rerun all exact query families with a new cutoff date;
2. search citations and references of Saguaro, SPECTRE, EXSpec, FASER, and
   TAPER;
3. record new papers even when they weaken the claim;
4. update artifact availability separately from paper novelty; and
5. preserve old matrix versions in git so the claim history is auditable.
