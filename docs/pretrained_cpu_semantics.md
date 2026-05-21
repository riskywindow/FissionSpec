# Pinned pretrained Qwen3 CPU semantics gate

This gate closes the remaining tokenizer/pretrained-weight semantic gap that
can be closed without renting a GPU. It runs the official
[`Qwen/Qwen3-0.6B`](https://huggingface.co/Qwen/Qwen3-0.6B) tokenizer and
pretrained model at the immutable 40-character revision
`c1899de289a04d12100db370d81485cdf75e47ca`. The Qwen team states that its
open-weight models use Apache-2.0, and the pinned repository contains that
license. The upstream Qwen3 Transformers guidance requires Transformers
4.51.0 or newer; this run uses 5.5.0.

This is real pretrained-model CPU evidence. It is not a GPU-kernel,
production-serving, throughput, or latency claim.

## Download once, then disconnect

The 1.52 GB snapshot is intentionally outside the Git repository in the
standard Hugging Face cache. No weight, symlink, or cache object is committed.
The sole networked setup step used this exact revision and allowlist:

```bash
python3 - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Qwen/Qwen3-0.6B",
    revision="c1899de289a04d12100db370d81485cdf75e47ca",
    allow_patterns=[
        "LICENSE",
        "config.json",
        "generation_config.json",
        "merges.txt",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
    ],
    max_workers=4,
)
PY
```

The official repository metadata resolved `main` to that commit before the
download. The single `model.safetensors` file is 1,503,300,328 bytes and hashes
to `f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b`.
The evidence records byte counts and SHA-256 digests for all eight required
license, config, tokenizer, and weight files.

Evidence generation cannot download. Before any optional dependency import the
tool sets `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`,
`HF_DATASETS_OFFLINE=1`, and `CUDA_VISIBLE_DEVICES=""`. Both snapshot and
pretrained loaders use `local_files_only=True`, and socket `connect` plus
`create_connection` are patched to raise throughout imports, loading, and
inference.

## Device and deterministic execution contract

The host has Apple MPS available, which makes the negative device claim
meaningful. The gate nevertheless:

- sets the default device to CPU;
- loads and casts all 596,049,920 parameters to float32 CPU;
- selects eager attention for an auditable unfused CPU path;
- sets one Torch thread and deterministic algorithms;
- uses `eval()` and `inference_mode()`;
- hides CUDA and verifies that CUDA is never initialized;
- disables MPS fallback and declares MPS forbidden; and
- wraps every model forward in a Torch dispatch mode that rejects any non-CPU
  tensor at its input or output boundary.

The canonical run audited 57,822 dispatched tensor operations. The only device
observed was `cpu`; zero CUDA or MPS tensor operations and zero non-CPU tensors
were observed. This is an in-process PyTorch/model-execution audit under the
pinned dependency stack, not an operating-system trace of native libraries or
subprocesses.

## Pretrained cache/rebatch experiment

The actual tokenizer maps three short prompts to lengths three, four, and five:

| Request | Text | Token IDs |
|---|---|---|
| A | `Blue sky above` | `10331, 12884, 3403` |
| B | `One two three four` | `3966, 1378, 2326, 3040` |
| C | `Cache rows move now today` | `8233, 6978, 3271, 1431, 3351` |

They are left-padded into one physical batch with explicit masks and logical
position IDs. The experiment then:

1. compares mixed-padded prefill logits with three unpadded monolithic
   forwards;
2. splits all 28 pretrained KV layers into distinct request-owned cache rows;
3. removes B from the active cache tensors, reducing every cache batch
   dimension from three rows to two;
4. advances A and C together while B is parked as a separate one-row cache;
5. proves B's cache fingerprint is byte-identical before and after peer
   advancement;
6. catches B up independently and compares each retained cache with its
   per-request reference;
7. concatenates the request caches back into active tensors in the deliberately
   reordered sequence C, B, A;
8. advances the reordered batch and compares its logits and greedy tokens with
   both individual cached decoding and complete monolithic forwards; and
9. records request/cache/logit fingerprints and all physical cache shapes.

The largest valid-path logit difference is
`5.817413330078125e-05`. The largest per-request KV difference is
`0.0002288818359375`. All greedy tokens match exactly under a conservative
float32 absolute/relative tolerance of `5e-4`.

Four meaningful corruptions are required to exceed the same tolerance:

| Corruption | Maximum logit delta |
|---|---:|
| expose left-padding tokens with an all-ones mask | `4.679703712463379` |
| associate tokens with another request's KV row | `10.459226608276367` |
| shift decode position IDs while retaining the cache | `1.867720603942871` |
| corrupt final padded-prefill relative positions | `2.4111688137054443` |

## Evidence integrity and reproduction

The canonical artifact is
`experiments/results/pretrained_cpu_semantics/evidence.json`. It is small and
contains no model weights. It is canonical JSON, self-hashed, bound to the exact
tool source, and closed under an exact schema. Verification imports only the
Python standard library and rejects symlink/non-regular input, duplicate keys,
non-finite constants, noncanonical bytes, source drift, checkpoint drift,
offline/device-contract drift, malformed cache shapes, failed negative
controls, and expanded claims.

The verifier also hashes the complete semantic result map, excluding only the
source record, against
`00c2dee7df0adc7f236653ebe53048ff20ae2f1436a06440cf3c72c2c4720124`.
Therefore, changing a valid token, fingerprint, runtime string, or within-bound
numeric result and recomputing the evidence self-hash is still rejected.
Like any verifier shipped beside its evidence, this standard-library check is a
self-consistency check rather than an external trust root: a coordinated source
and evidence rewrite could define a different verifier. The release workflow
closes that provenance boundary by binding the tool and evidence to source
commit A, then recording the replayed attestation in its single-file child
commit B before the campaign freezer accepts either.

The final 20,258-byte artifact has:

- payload SHA-256
  `bd3ddfd304de850ce68b28a9acbbb8aecb6ba575d5adfdd691133d16ae210e94`;
- evidence-file SHA-256
  `7baea0f0341b8be8332d46a392653dc0da6e8a422ab154635356dada17263ef4`;
- embedded/final tool-source SHA-256
  `2cb741fcdb21a0737cacff94bc8c5d587f59e312cc6caf1d4b449a53c95e5a55`;
  and
- frozen semantic-result SHA-256
  `00c2dee7df0adc7f236653ebe53048ff20ae2f1436a06440cf3c72c2c4720124`.

The canonical offline generation took 5.807 seconds inside the tool and 6.27
seconds wall time including process startup. An independent cached/offline
regeneration took 6.013 seconds inside the test and produced byte-identical
evidence.

```bash
# Requires the one-time cached snapshot; performs no network access.
python3 tools/run_pretrained_cpu_semantics.py

# Requires only the Python standard library and no model cache.
python3 -S tools/run_pretrained_cpu_semantics.py \
  --verify-only \
  experiments/results/pretrained_cpu_semantics/evidence.json

PYTHONPATH=src python3 -m unittest \
  tests.test_pretrained_cpu_semantics -v
```

## What remains GPU/backend work

This gate removes actual tokenizer, actual pretrained weights, CPU Qwen3
mathematics, cache ownership, physical cache-row rebatching, and short-context
greedy equivalence from the GPU budget. It does not establish CUDA/MPS or fused
attention behavior, paged-KV page ownership, allocator lifetimes, CUDA graph
capture, graph bucket transitions, quantized or long-context behavior,
multi-GPU communication, remote drafting, production cancellation/OOM/stream
ordering, or accelerator performance.
