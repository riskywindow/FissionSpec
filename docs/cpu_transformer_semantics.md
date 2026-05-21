# Actual CPU transformer-stack semantics gate

This gate closes the highest-value semantic gap that does not require an
accelerator: it runs the installed PyTorch and Transformers implementation of
`Qwen2ForCausalLM`, including rotary positions, causal masks, grouped-query KV
states, and `DynamicCache`, while physically removing and reinserting requests
between forward calls.

It is not a simulator result. It is also not a production-GPU result.

## Frozen execution

The gate uses:

- Python `3.12.8`;
- PyTorch `2.10.0`;
- Transformers `5.5.0`;
- a randomly initialized, locally constructed `Qwen2Config`;
- 24,928 float32 parameters, two decoder layers, four attention heads, and two
  KV heads;
- fixed seed `20260723`; and
- one CPU thread with deterministic algorithms enabled.

It does not call `from_pretrained`, load a tokenizer, read a checkpoint, or
resolve a model ID. Before importing Torch, the tool forces
`CUDA_VISIBLE_DEVICES=""`, `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, and
`HF_DATASETS_OFFLINE=1`. Socket `connect` and `create_connection` are patched
to fail during construction and execution.

The current host reports that MPS is available. That makes the device check
meaningful: the model is still explicitly moved to `torch.device("cpu")`, all
parameters and KV tensors are inspected, and a `TorchDispatchMode` audits
every model tensor operation. The checked-in run observed 4,160 operations,
only `cpu` tensors, zero non-CPU tensors, and no CUDA initialization before or
after the gate.

## Semantic experiment

Three requests have prompt lengths three, four, and five. The test first
left-pads them into one mixed batch and supplies the corresponding attention
mask and semantic `position_ids`.

1. Mixed-batch final logits are compared with three unpadded monolithic
   forwards.
2. The mixed `DynamicCache` is split into request-owned rows with distinct
   SHA-256 fingerprints.
3. Request B is parked. Requests A and C decode together after B is physically
   removed from the cache batch.
4. B's parked cache is hashed before and after its peers advance; its bytes
   must remain identical.
5. B catches up independently. Its logits and cache are compared with its
   independent reference path.
6. The equal-length request caches are reinserted in the deliberately different
   order `C, B, A`.
7. The reordered batch decodes again. Its logits and greedy tokens are compared
   both with per-request cached decoding and complete monolithic forwards.

The largest observed valid-path logit difference is
`8.940696716308594e-08`, below the frozen absolute and relative tolerances of
`2e-5`. Greedy tokens match exactly. Per-request KV contents after batched
advancement differ from independent cached references by at most
`2.2351741790771484e-08`.

## Negative controls

The gate must also reject plausible association bugs:

| Corruption | Maximum logit delta |
|---|---:|
| corrupt only the final padded-prefill relative position | `0.0006442070` |
| expose left-padding tokens by using the wrong attention mask | `0.05867816` |
| associate request tokens with another request's KV cache | `0.22767732` |
| increment decode `position_ids` while retaining the cache | `0.0004524067` |

Every delta exceeds the frozen tolerance. Uniformly translating all RoPE
positions is intentionally not used as a negative control because relative
rotary attention can be translation-invariant; corrupting one relative
position tests the actual association.

## Evidence integrity

The canonical evidence is
`experiments/results/cpu_transformer_semantics/evidence.json`.

- Payload SHA-256:
  `c7d7cf66d81e7fa3a2014e77cea36e9b609d754fe5a426b182a5dde0a232ccba`
- Evidence-file SHA-256:
  `08cb4319515a31dddf262f3379b9ff519976421a0653080f68dada675ef46fac`
- Tool-source SHA-256 embedded in the evidence:
  `b793c1e411a7cb09e8efc1c8e068e523f91f475675ee328c38ad16c2bafacdfe`
- Random model-state SHA-256:
  `4d4776e261db8a4c25f993e8448f19246bc6e6b3bc9cef92bb3edae0d8876a77`
- Frozen non-source result-map SHA-256:
  `2529a5e703f1daeec4dba4cb6a45faa97ce141cecccd792060e06e6060180721`

Two clean command-line invocations produced identical evidence bytes and the
same payload hash. They took 1.616 and 1.654 seconds including imports on this
host; runtime is printed but excluded from the evidence to preserve
deterministic bytes. The verifier uses only the Python standard library. It
rejects symlink/non-regular inputs, duplicate keys, and non-finite JSON
constants, then checks canonical encoding, the self-hash, the exact tool-source
hash, a closed schema, the dependency/offline/device contracts, equivalence
tolerances, cache ownership, and every negative control. Finally it hashes the
complete payload except the source-dependent implementation record and requires
the exact frozen result-map digest above, so even a correctly rehashed
within-tolerance measurement, token, fingerprint, or runtime mutation fails.

```bash
python3 tools/run_cpu_transformer_semantics.py
python3 -S tools/run_cpu_transformer_semantics.py \
  --verify-only experiments/results/cpu_transformer_semantics/evidence.json
.repro/venv/bin/python tools/run_cpu_transformer_semantics.py \
  --verify-only experiments/results/cpu_transformer_semantics/evidence.json
PYTHONPATH=src python3 -m unittest tests.test_cpu_transformer_semantics -v
```

If Torch or Transformers is absent, the optional installed-stack runtime tests
skip rather than making the dependency-free FissionSpec package depend on
either library. The static no-download contract test still runs. Reproducing
the checked-in evidence requires the versions and platform recorded inside it.
The dispatch audit uses PyTorch's private `_python_dispatch` interface, so its
instrumentation is version-specific even though the tensor/device assertions
are straightforward.

## What still requires a GPU or production engine

This CPU gate does not establish:

- pretrained-model/tokenizer output equivalence;
- CUDA, MPS, fused-attention, or mixed-precision numerical behavior;
- physical row compaction in a production attention kernel;
- paged-KV allocation, page handles, or allocator lifetime;
- CUDA graph capture and graph-bucket transitions;
- quantized or long-context cache behavior;
- tensor/pipeline parallel communication or remote drafting; or
- production scheduler callback, cancellation, OOM, and stream-order behavior.

Those are explicit accelerator/backend seams. The gate removes CPU-testable
transformer mathematics, cache ownership, position/mask association, and
rebatching semantics from the later GPU campaign; it does not use their
absence to imply a kernel-speed or production-correctness result.
