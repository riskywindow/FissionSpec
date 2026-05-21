# Zero-GPU campaign freeze

`tools/freeze_gpu_campaign.py` closes Stage F0 without launching or authorizing
an accelerator job. It binds the complete declared CPU evidence set, the
pre-registration, the generator sources, the public replay, and the exact
source commit into one canonical bundle under `configs/gpu_campaign/`.

The bundle is a release-control artifact, not GPU evidence. Its claim boundary
explicitly excludes physical row omission, kernel speedup, production output
equivalence, and accelerator performance.

## What is frozen

The freezer writes exactly seven files:

| File | Contract |
|---|---|
| `validation_anchor_v1.json` | Two-state MMPP template at 0.70 mean load, `(128, 32)` tokens, temperature 0.6 |
| `validation_anchor_v2.json` | Pareto template with tail index 1.35 at 0.90 load, `(16K, 256)` tokens, temperature 1.0 |
| `validation_anchor_v3.json` | Frozen Azure public replay rescaled to 0.70 load, temperature 0 |
| `cpu_evidence_index.json` | Ordered byte hashes and embedded payload hashes for every declared CPU artifact |
| `campaign_plan.json` | Canonical campaign ID, replay caps, anchor hashes, protocol hash, and source commit |
| `campaign_ledger_f0.json` | One passing CPU-release record with zero replays, zero GPU seconds spent, and zero GPU seconds authorized |
| `bundle_manifest.json` | Closed ordered file inventory with byte counts, hashes, source commit, and a self-hash |

The anchor documents deliberately retain a symbolic target-only saturation
rate `S`. Stage 1 may substitute the measured `S` using the frozen formulas; it
may not change arrival families, random-key rules, request shapes, loads,
source hashes, or replay identity in response to GPU outcomes.

The evidence index covers the exhaustive finite semantics and scheduling
evidence, causal mechanism study, controller source audit, transformer-stack
CPU semantics, public trace, output-equivalence audit, pinned SGLang patch
series, and cross-language fixtures. JSON artifacts with an embedded
`payload_sha256` must also pass their native canonical-payload hash convention.

## Release workflow

First complete every CPU, release, and reproducibility gate and commit that
source state. The repository must be clean:

```bash
make cpu-evidence-offline
make release-check-offline
git status --short
```

Freeze against the full object ID of that clean commit:

```bash
SOURCE_COMMIT="$(git rev-parse HEAD)"
PYTHONPATH=src python tools/freeze_gpu_campaign.py freeze \
  --code-commit "$SOURCE_COMMIT"
PYTHONPATH=src python tools/freeze_gpu_campaign.py verify
```

The source commit is intentionally the commit immediately before the generated
bundle commit. A bundle cannot hash the commit that contains itself without a
self-reference. Commit the verified output separately:

```bash
git add configs/gpu_campaign
git commit -m "chore(protocol): freeze zero-GPU campaign handoff"
```

The verifier requires every commit-bound input to be a regular, non-symlink
file whose working-tree bytes exactly match the named commit. It rejects dirty
inputs, abbreviated or non-commit object IDs, missing or extra bundle files,
symlinked output, duplicate evidence identities or paths, non-canonical anchor
order, strict-JSON violations, stale hashes, altered replay bytes, changed
derived fields, and a ledger that spends or authorizes any GPU time.

Re-running `freeze` on an existing valid seven-file bundle is idempotent. An
unrelated or partially populated output directory is preserved and rejected
instead of overwritten.

## Spend invariant

F0 records:

```text
spent_gpu_seconds = 0
currently_authorized_gpu_seconds = 0
next_stage = f1_physical_row_omission
```

Passing F0 does not authorize F1. A later operator must separately seal a
positive, bounded budget for the immediate next stage, with its rationale
hash, after hardware pricing and Stage 1 calibration inputs are known. The
ledger cannot pre-authorize later stages, skip a stage, replace a sealed
budget, exceed a GPU-second or replay cap, or continue after a failed gate.
This makes the cheapest physical-row-omission falsification the only possible
first GPU expense.
