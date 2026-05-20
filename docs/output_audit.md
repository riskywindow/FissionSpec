# Offline production-output audit

`fissionspec.output_audit` turns one accelerator capture into a deterministic,
dependency-free CPU parity gate. It is designed to reduce GPU reruns: retain the
paired distributions, proposal probability, uniform variate, cluster identity,
and preregistered slices once; recompute every diagnostic and threshold decision
offline.

> **Evidence boundary:** the bundled fixture is synthetic CPU math. A passing
> synthetic report validates the audit implementation and serialization path. It
> does not establish parity for a real model, tokenizer, quantizer, serving
> engine, accelerator, or kernel. A captured report is output evidence, not a
> throughput or latency measurement.

## Capture contract

The corpus schema is `fissionspec.output-audit-corpus.v1`. Its canonical payload
is self-hashed with SHA-256. The hash covers every record, slice, captured build
digest, uniform variate, and proposal probability.

Top-level fields are deliberately closed:

| Field | Meaning |
|---|---|
| `schema` | Exact versioned schema identifier |
| `evidence_class` | `captured-production-output` or `synthetic-cpu-fixture` |
| `measurement_warning` | Required evidence-class warning |
| `capture` | Capture/model IDs plus tokenizer, engine, config, and capture-tool SHA-256 digests |
| `record_count` | Exact number of paired positions |
| `records` | Canonically sorted paired observations |
| `payload_sha256` | SHA-256 of canonical payload without this field |

Each record contains:

- a unique `record_id` and an independent-unit `cluster_id`;
- at least one preregistered `slices` label;
- a `reference` and `candidate` vector;
- `proposed_token_id`, its shared positive `draft_probability`, and one
  preregistered `uniform` in `[0, 1)`.

A vector has `encoding`, `token_ids`, and `values`. `encoding` is either
`logits` or `probabilities`. Token IDs must be unique non-negative integers.
Logits must be finite. Probabilities must be finite, non-negative, and sum to
one within `1e-9`; the builder normalizes the final rounding residue. Two logit
vectors must cover exactly the same tokens because an omitted logit has no
well-defined probability. Sparse probability vectors may differ in support;
omitted tokens mean exact zero mass.

Use `build_corpus(...)` in the capture adapter rather than constructing JSON by
hand. It sorts records, slice dimensions, and token/value pairs before hashing,
making record and vocabulary permutation irrelevant. The capture adapter should
also:

1. pair reference and candidate at the same request, position, prefix, and
   proposed token;
2. retain full-vocabulary logits where feasible, rather than rounded top-k
   output;
3. use the same tokenizer and proposal distribution for the pair;
4. generate and store uniforms from a preregistered stream independent of both
   observed outputs;
5. define `cluster_id` at the independent trace/request/seed unit, not at each
   token;
6. hash exact engine commits or immutable source/build manifests;
7. capture all planned batch, sequence-length, dtype, graph/eager, quantization,
   and kernel slices in one campaign.

The shared `draft_probability` contract tests candidate verifier drift while
holding the proposal mechanism fixed. If the candidate also changes the draft
model or draft kernel, capture that as a separate paired corpus so the changed
proposal law is not confounded with verifier drift.

## Deterministic metrics

Logits are converted with max-subtracted binary64 softmax. A record reports:

- reference and candidate greedy token, breaking exact ties by lowest token ID;
- the reference and candidate speculative acceptance probabilities
  `min(1, p(proposed) / q(proposed))`;
- acceptance decisions under the same stored uniform and their divergence;
- total variation and Jensen–Shannon divergence;
- directional `KL(reference || candidate)` and `KL(candidate || reference)`;
- top-k set overlap;
- the worst cross-ranking displacement of either greedy token;
- reference/candidate top-1 margins and absolute margin drift.

Zero support is explicit. A zero-mass term on the left contributes zero to KL.
A positive left mass against zero right mass is mathematically infinite. Reports
encode that case as `null` plus an `*_infinite` flag, never as nonstandard JSON;
any infinite KL fails the gate. Jensen–Shannon remains finite on disjoint
support.

The report retains every record diagnostic, aggregate diagnostics, and a
summary for every `dimension=value` slice. This is intentional: an outlier can
be localized from the original capture without scheduling another GPU job.

## Preregistered gate

`AuditThresholds` is complete and closed: missing keys, extra keys, invalid
numbers, unsupported versions, or too few records/clusters fail before a report
is produced. The default is a strict parity gate:

- zero observed greedy and acceptance-decision divergence;
- small aggregate and record-level TV/JS/KL tolerances;
- exact mean top-k overlap and zero greedy-rank drift;
- bounded mean margin drift;
- minimum 256 records and 16 independent clusters.

Freeze the threshold JSON and its hash before capturing candidate results:

```bash
PYTHONPATH=src python tools/run_output_audit.py \
  write-default-thresholds artifacts/output_thresholds.v1.json
```

Four aggregate inferential gates form one preregistered family:

1. greedy-mismatch rate upper bound;
2. acceptance-divergence rate upper bound;
3. mean-TV upper bound;
4. mean-JS upper bound.

The familywise alpha is divided by four with Bonferroni. Event rates use an
exact one-sided Clopper–Pearson upper bound. TV and JS use the deterministic
paired cluster bootstrap, resampling `cluster_id` rather than token positions.
The report stores the resample fingerprint, seed provenance, family membership,
per-test alpha, and simultaneous confidence level. Per-slice gates are
deterministic tolerance checks, not unregistered slice-wise hypothesis tests.

Thresholds should be based on an independently captured known-equivalent build,
numerical requirements, and downstream acceptance risk—not relaxed after
looking at the candidate. Default tolerances are intentionally inappropriate
for claiming equivalence of arbitrary quantization changes.

## Fully offline workflow

Exercise the entire path without downloads or an ML framework:

```bash
PYTHONPATH=src python tools/run_output_audit.py \
  fixture artifacts/output_audit_cpu
PYTHONPATH=src python tools/run_output_audit.py \
  verify artifacts/output_audit_cpu/synthetic_output_corpus.json
PYTHONPATH=src python tools/run_output_audit.py \
  verify artifacts/output_audit_cpu/synthetic_output_audit_report.json
```

Audit a real hash-locked capture:

```bash
PYTHONPATH=src python tools/run_output_audit.py audit \
  captures/paired_outputs.json \
  artifacts/paired_outputs.report.json \
  --thresholds artifacts/output_thresholds.v1.json
```

Exit status is `0` for a passing gate, `1` for a valid report that fails its
thresholds, and `2` for malformed input, integrity failure, or invalid
preregistration. The CLI writes canonical JSON atomically.

The generated report schema is `fissionspec.output-audit-report.v1`. Its payload
hash covers the source corpus hash, complete threshold mapping and threshold
hash, multiplicity declaration, aggregate and slice summaries, uncertainty,
row diagnostics, and every violation.

## What still requires accelerator access

CPU tests can exhaust schema parsing, hashing, numerical identities, support
edge cases, deterministic resampling, exact binomial bounds, threshold logic,
and report reproducibility. They cannot create the decisive corpus. Remaining
accelerator work is therefore narrow:

- capture reference and candidate engine output on the preregistered real
  model/hardware/kernel slices;
- verify that capture truly pairs the same logical positions and proposal law;
- measure performance separately under the serving experiment protocol.

Capture once and retain the full corpus. Changing only statistical summaries,
visualizations, or tolerances then needs no GPU rerun; changing model outputs,
kernel builds, capture coverage, tokenizer, or proposal semantics does.
