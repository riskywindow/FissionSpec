#!/usr/bin/env python3
"""Run the pinned Qwen3 pretrained/tokenizer semantics gate on CPU.

Evidence generation requires a previously downloaded, exact Hugging Face
snapshot. Generation is forcibly offline and denies socket connections. The
``--verify-only`` path imports only the Python standard library.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import platform
import socket
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Final, cast
from unittest import mock

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["FISSIONSPEC_MPS_DISABLED"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

SCHEMA_VERSION: Final = 1
SEED: Final = 20260723
MODEL_ID: Final = "Qwen/Qwen3-0.6B"
MODEL_REVISION: Final = "c1899de289a04d12100db370d81485cdf75e47ca"
MODEL_LICENSE: Final = "apache-2.0"
DEVICE_NAME: Final = "cpu"
MODEL_DTYPE: Final = "torch.float32"
ATTENTION_IMPLEMENTATION: Final = "eager"
ABSOLUTE_TOLERANCE: Final = 5e-4
RELATIVE_TOLERANCE: Final = 5e-4
EXPECTED_TORCH_VERSION: Final = "2.10.0"
EXPECTED_TRANSFORMERS_VERSION: Final = "5.5.0"
EXPECTED_HUB_VERSION: Final = "1.9.0"
EXPECTED_SAFETENSORS_VERSION: Final = "0.7.0"
EXPECTED_TOKENIZERS_VERSION: Final = "0.22.2"
EXPECTED_FROZEN_RESULTS_SHA256: Final = (
    "00c2dee7df0adc7f236653ebe53048ff20ae2f1436a06440cf3c72c2c4720124"
)
WARNING: Final = (
    "ACTUAL PINNED PRETRAINED QWEN3 CPU SEMANTICS — NOT A GPU KERNEL OR "
    "PRODUCTION SERVING MEASUREMENT."
)
CLAIM_BOUNDARY: Final = (
    "This artifact exercises the exact pinned Qwen/Qwen3-0.6B tokenizer, "
    "pretrained parameters, Qwen3 forward path, and DynamicCache on CPU for "
    "three very short requests. It does not exercise CUDA/MPS kernels, paged "
    "attention, CUDA graphs, a production scheduler, quantization, or long context."
)
REQUEST_IDS: Final = ("request-a", "request-b", "request-c")
PROMPT_TEXTS: Final = (
    "Blue sky above",
    "One two three four",
    "Cache rows move now today",
)
PROMPT_TOKEN_IDS: Final = (
    (10331, 12884, 3403),
    (3966, 1378, 2326, 3040),
    (8233, 6978, 3271, 1431, 3351),
)
NEGATIVE_CONTROL_KEYS: Final = (
    "wrong_attention_mask",
    "wrong_cache_request_association",
    "wrong_decode_position_ids",
    "wrong_left_padding_position_ids",
)
REMAINING_GPU_BACKEND_SEAMS: Final = (
    "CUDA/MPS and fused-attention numerical behavior",
    "paged-KV allocation, physical page handles, and allocator lifetime",
    "CUDA graph capture and graph-bucket transitions",
    "production scheduler callbacks, cancellation, OOM, and stream ordering",
    "quantized, mixed-precision, and long-context behavior",
    "tensor/pipeline parallel communication and remote drafting",
)
CHECKPOINT_FILES: Final[dict[str, tuple[int, str]]] = {
    "LICENSE": (
        11_343,
        "832dd9e00a68dd83b3c3fb9f5588dad7dcf337a0db50f7d9483f310cd292e92e",
    ),
    "config.json": (
        726,
        "660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd",
    ),
    "generation_config.json": (
        239,
        "2325da0f15bb848e018c5ae071b7943332e9f871d6b60e2ed22ca97d4cb993d2",
    ),
    "merges.txt": (
        1_671_853,
        "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5",
    ),
    "model.safetensors": (
        1_503_300_328,
        "f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b",
    ),
    "tokenizer.json": (
        11_422_654,
        "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
    ),
    "tokenizer_config.json": (
        9_732,
        "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101",
    ),
    "vocab.json": (
        2_776_833,
        "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
    ),
}

torch: Any = None
transformers: Any = None
huggingface_hub: Any = None
safetensors: Any = None
tokenizers: Any = None
AutoModelForCausalLM: Any = None
AutoTokenizer: Any = None
DynamicCache: Any = None
snapshot_download: Any = None


class PretrainedSemanticsError(RuntimeError):
    """Raised when pretrained semantics or evidence integrity fails."""


def _canonical_json_bytes(document: object) -> bytes:
    return (
        json.dumps(
            document,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )


def _sha256_document(document: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(document)).hexdigest()


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_heavy_stack() -> None:
    """Import optional generation dependencies after offline flags are set."""

    global torch, transformers, huggingface_hub, safetensors, tokenizers
    global AutoModelForCausalLM, AutoTokenizer, DynamicCache, snapshot_download
    if torch is not None:
        return
    torch = importlib.import_module("torch")
    transformers = importlib.import_module("transformers")
    huggingface_hub = importlib.import_module("huggingface_hub")
    safetensors = importlib.import_module("safetensors")
    tokenizers = importlib.import_module("tokenizers")
    AutoModelForCausalLM = transformers.AutoModelForCausalLM
    AutoTokenizer = transformers.AutoTokenizer
    DynamicCache = importlib.import_module("transformers.cache_utils").DynamicCache
    snapshot_download = huggingface_hub.snapshot_download


def _walk_tensors(value: object) -> Iterator[Any]:
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _walk_tensors(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _walk_tensors(item)


class CpuDispatchAudit:
    """Reject non-CPU tensors at every observed Torch dispatch boundary."""

    def __init__(self) -> None:
        self.operations = 0
        self.devices_seen: set[str] = set()
        self.non_cpu_observations = 0

    def _audit(self, value: object, *, phase: str) -> None:
        for tensor in _walk_tensors(value):
            device_type = tensor.device.type
            self.devices_seen.add(device_type)
            if device_type != DEVICE_NAME:
                self.non_cpu_observations += 1
                raise PretrainedSemanticsError(
                    f"{phase} tensor escaped CPU-only contract: {tensor.device}"
                )


def _cpu_only_dispatch_mode() -> tuple[Any, CpuDispatchAudit]:
    dispatch_base = importlib.import_module("torch.utils._python_dispatch").TorchDispatchMode
    audit = CpuDispatchAudit()

    def dispatch(
        mode_self: object,
        func: Any,
        types: tuple[type[Any], ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        del mode_self, types
        call_kwargs = {} if kwargs is None else kwargs
        audit._audit(args, phase="input")
        audit._audit(call_kwargs, phase="input")
        audit.operations += 1
        result = func(*args, **call_kwargs)
        audit._audit(result, phase="output")
        return result

    mode_type = type(
        "PretrainedCpuOnlyDispatchMode",
        (dispatch_base,),
        {"__torch_dispatch__": dispatch},
    )
    return mode_type(), audit


@contextmanager
def _network_denied() -> Iterator[list[str]]:
    """Fail synchronously if imports, loading, or inference touch the network."""

    attempts: list[str] = []

    def deny(*args: object, **kwargs: object) -> None:
        del args, kwargs
        attempts.append("socket connection")
        raise PretrainedSemanticsError(
            "network access attempted during offline pretrained semantics generation"
        )

    with (
        mock.patch.object(socket.socket, "connect", side_effect=deny),
        mock.patch.object(socket, "create_connection", side_effect=deny),
    ):
        yield attempts


def _resolve_and_verify_snapshot() -> tuple[Path, dict[str, object]]:
    try:
        snapshot = Path(
            snapshot_download(
                repo_id=MODEL_ID,
                revision=MODEL_REVISION,
                local_files_only=True,
                allow_patterns=sorted(CHECKPOINT_FILES),
            )
        )
    except Exception as error:
        raise PretrainedSemanticsError(
            "the exact Qwen3 snapshot is not cached; perform the documented one-time "
            "download before offline evidence generation"
        ) from error
    if snapshot.name != MODEL_REVISION or len(snapshot.name) != 40:
        raise PretrainedSemanticsError("snapshot did not resolve to the pinned commit")
    observed: dict[str, object] = {}
    for filename, (expected_bytes, expected_sha256) in CHECKPOINT_FILES.items():
        path = snapshot / filename
        if not path.is_file():
            raise PretrainedSemanticsError(f"required checkpoint file is absent: {filename}")
        byte_count = path.stat().st_size
        sha256 = _hash_file(path)
        if byte_count != expected_bytes or sha256 != expected_sha256:
            raise PretrainedSemanticsError(f"checkpoint byte identity mismatch: {filename}")
        observed[filename] = {
            "bytes": byte_count,
            "sha256": sha256,
        }
    return snapshot, observed


def _tensor_fingerprint(tensor: Any) -> str:
    value = tensor.detach().contiguous().cpu()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(b"\x00")
    digest.update(",".join(str(item) for item in value.shape).encode("ascii"))
    digest.update(b"\x00")
    digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _cache_layers(cache: Any) -> tuple[tuple[Any, Any], ...]:
    result: list[tuple[Any, Any]] = []
    for index, layer in enumerate(cache.layers):
        keys = layer.keys
        values = layer.values
        if not isinstance(keys, torch.Tensor) or not isinstance(values, torch.Tensor):
            raise PretrainedSemanticsError(f"cache layer {index} is not initialized")
        if keys.device.type != DEVICE_NAME or values.device.type != DEVICE_NAME:
            raise PretrainedSemanticsError(f"cache layer {index} escaped CPU")
        result.append((keys, values))
    if not result:
        raise PretrainedSemanticsError("model returned an empty cache")
    return tuple(result)


def _clone_cache(cache: Any) -> Any:
    return DynamicCache(
        (keys.detach().clone(), values.detach().clone()) for keys, values in _cache_layers(cache)
    )


def _select_cache(cache: Any, indices: Sequence[int]) -> Any:
    selected = _clone_cache(cache)
    selected.batch_select_indices(torch.tensor(indices, dtype=torch.long, device=DEVICE_NAME))
    return selected


def _concatenate_caches(caches: Sequence[Any]) -> Any:
    if not caches:
        raise ValueError("caches must not be empty")
    layers = tuple(_cache_layers(cache) for cache in caches)
    if len({len(item) for item in layers}) != 1:
        raise PretrainedSemanticsError("cache layer counts differ")
    combined: list[tuple[Any, Any]] = []
    for layer_index in range(len(layers[0])):
        sequence_lengths = {item[layer_index][0].shape[-2] for item in layers}
        if len(sequence_lengths) != 1:
            raise PretrainedSemanticsError("cannot reinsert caches with different sequence lengths")
        combined.append(
            (
                torch.cat(
                    [item[layer_index][0] for item in layers],
                    dim=0,
                ),
                torch.cat(
                    [item[layer_index][1] for item in layers],
                    dim=0,
                ),
            )
        )
    return DynamicCache(combined)


def _cache_fingerprint(cache: Any) -> str:
    digest = hashlib.sha256()
    for layer_index, (keys, values) in enumerate(_cache_layers(cache)):
        digest.update(layer_index.to_bytes(8, "big"))
        digest.update(bytes.fromhex(_tensor_fingerprint(keys)))
        digest.update(bytes.fromhex(_tensor_fingerprint(values)))
    return digest.hexdigest()


def _cache_shape(cache: Any) -> list[dict[str, object]]:
    return [
        {
            "layer": index,
            "key": list(keys.shape),
            "value": list(values.shape),
            "device": keys.device.type,
            "dtype": str(keys.dtype),
        }
        for index, (keys, values) in enumerate(_cache_layers(cache))
    ]


def _maximum_absolute_difference(left: Any, right: Any) -> float:
    if left.shape != right.shape:
        raise PretrainedSemanticsError(
            f"tensor shape mismatch: {tuple(left.shape)} != {tuple(right.shape)}"
        )
    return float(torch.max(torch.abs(left - right)).item())


def _assert_equivalent(left: Any, right: Any, *, label: str) -> float:
    difference = _maximum_absolute_difference(left, right)
    if not torch.allclose(
        left,
        right,
        atol=ABSOLUTE_TOLERANCE,
        rtol=RELATIVE_TOLERANCE,
    ):
        raise PretrainedSemanticsError(f"{label} logits differ: max_abs={difference:.9g}")
    if not torch.equal(torch.argmax(left, dim=-1), torch.argmax(right, dim=-1)):
        raise PretrainedSemanticsError(f"{label} greedy token mismatch")
    return difference


def _assert_cache_equivalent(left: Any, right: Any, *, label: str) -> float:
    left_layers = _cache_layers(left)
    right_layers = _cache_layers(right)
    if len(left_layers) != len(right_layers):
        raise PretrainedSemanticsError(f"{label} cache layer count differs")
    maximum = 0.0
    for index, ((left_keys, left_values), (right_keys, right_values)) in enumerate(
        zip(left_layers, right_layers, strict=True)
    ):
        maximum = max(
            maximum,
            _maximum_absolute_difference(left_keys, right_keys),
            _maximum_absolute_difference(left_values, right_values),
        )
        if not torch.allclose(
            left_keys,
            right_keys,
            atol=ABSOLUTE_TOLERANCE,
            rtol=RELATIVE_TOLERANCE,
        ) or not torch.allclose(
            left_values,
            right_values,
            atol=ABSOLUTE_TOLERANCE,
            rtol=RELATIVE_TOLERANCE,
        ):
            raise PretrainedSemanticsError(f"{label} cache differs at layer {index}")
    return maximum


def _prepare_prompts(tokenizer_instance: Any) -> tuple[Any, Any, Any]:
    tokenizer_instance.padding_side = "left"
    encoded = tokenizer_instance(
        list(PROMPT_TEXTS),
        add_special_tokens=False,
        padding=True,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(device=DEVICE_NAME)
    attention_mask = encoded["attention_mask"].to(device=DEVICE_NAME)
    observed_ids = tuple(
        tuple(
            int(token)
            for token, visible in zip(row.tolist(), mask.tolist(), strict=True)
            if visible
        )
        for row, mask in zip(input_ids, attention_mask, strict=True)
    )
    if observed_ids != PROMPT_TOKEN_IDS:
        raise PretrainedSemanticsError("pinned tokenizer prompt IDs drifted")
    position_ids = (attention_mask.cumsum(dim=-1) - 1).clamp_min(0)
    return input_ids, attention_mask, position_ids


def _cached_step(
    model: Any,
    *,
    input_tokens: Any,
    attention_mask: Any,
    position_ids: Any,
    cache_position: int,
    cache: Any,
) -> tuple[Any, Any]:
    output = model(
        input_ids=input_tokens,
        attention_mask=attention_mask,
        position_ids=position_ids,
        cache_position=torch.tensor(
            [cache_position],
            dtype=torch.long,
            device=DEVICE_NAME,
        ),
        past_key_values=cache,
        use_cache=True,
        return_dict=True,
    )
    updated = output.past_key_values
    if not isinstance(updated, DynamicCache):
        raise PretrainedSemanticsError("Qwen3 did not return DynamicCache")
    return output.logits[:, -1, :].detach(), updated


def _run_semantics(network_attempts: list[str]) -> dict[str, object]:
    versions = {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "huggingface_hub": huggingface_hub.__version__,
        "safetensors": safetensors.__version__,
        "tokenizers": tokenizers.__version__,
    }
    expected_versions = {
        "torch": EXPECTED_TORCH_VERSION,
        "transformers": EXPECTED_TRANSFORMERS_VERSION,
        "huggingface_hub": EXPECTED_HUB_VERSION,
        "safetensors": EXPECTED_SAFETENSORS_VERSION,
        "tokenizers": EXPECTED_TOKENIZERS_VERSION,
    }
    if versions != expected_versions:
        raise PretrainedSemanticsError(f"generation dependency version drift: {versions!r}")

    torch.manual_seed(SEED)
    torch.set_default_device(DEVICE_NAME)
    torch.set_default_dtype(torch.float32)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    cuda_initialized_before = torch.cuda.is_initialized()
    snapshot, checkpoint_files = _resolve_and_verify_snapshot()

    tokenizer_instance = AutoTokenizer.from_pretrained(
        snapshot,
        local_files_only=True,
        use_fast=True,
    )
    tokenizer_instance.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        snapshot,
        local_files_only=True,
        dtype=torch.float32,
        attn_implementation=ATTENTION_IMPLEMENTATION,
    )
    model.eval()
    model.to(device=DEVICE_NAME, dtype=torch.float32)
    if model.__class__.__name__ != "Qwen3ForCausalLM":
        raise PretrainedSemanticsError("unexpected pretrained architecture")
    if any(parameter.device.type != DEVICE_NAME for parameter in model.parameters()):
        raise PretrainedSemanticsError("model parameter escaped CPU")
    if any(buffer.device.type != DEVICE_NAME for buffer in model.buffers()):
        raise PretrainedSemanticsError("model buffer escaped CPU")
    if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
        raise PretrainedSemanticsError("model parameter escaped float32 contract")
    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    input_ids, base_mask, base_positions = _prepare_prompts(tokenizer_instance)
    lengths = base_mask.sum(dim=-1)
    dispatch_mode, guard = _cpu_only_dispatch_mode()
    with dispatch_mode, torch.inference_mode():
        mixed_prefill = model(
            input_ids=input_ids,
            attention_mask=base_mask,
            position_ids=base_positions,
            use_cache=True,
            return_dict=True,
        )
        mixed_cache = mixed_prefill.past_key_values
        if not isinstance(mixed_cache, DynamicCache):
            raise PretrainedSemanticsError("mixed prefill did not return DynamicCache")
        mixed_last_logits = mixed_prefill.logits[:, -1, :].detach()

        monolithic_prefill_logits: list[Any] = []
        for prompt in PROMPT_TOKEN_IDS:
            prompt_ids = torch.tensor(
                [prompt],
                dtype=torch.long,
                device=DEVICE_NAME,
            )
            output = model(
                input_ids=prompt_ids,
                attention_mask=torch.ones_like(prompt_ids),
                position_ids=torch.arange(
                    len(prompt),
                    dtype=torch.long,
                    device=DEVICE_NAME,
                ).unsqueeze(0),
                use_cache=False,
                return_dict=True,
            )
            monolithic_prefill_logits.append(output.logits[:, -1, :].detach())
        monolithic_prefill = torch.cat(monolithic_prefill_logits, dim=0)
        prefill_difference = _assert_equivalent(
            mixed_last_logits,
            monolithic_prefill,
            label="left-padded mixed prefill versus unpadded monolithic",
        )
        first_tokens = torch.argmax(mixed_last_logits, dim=-1).unsqueeze(-1)

        wrong_positions = base_positions.clone()
        wrong_positions[:, -1] += 1
        wrong_prefill = model(
            input_ids=input_ids,
            attention_mask=base_mask,
            position_ids=wrong_positions,
            use_cache=False,
            return_dict=True,
        ).logits[:, -1, :]
        wrong_position_delta = _maximum_absolute_difference(
            wrong_prefill,
            mixed_last_logits,
        )
        if wrong_position_delta <= ABSOLUTE_TOLERANCE:
            raise PretrainedSemanticsError(
                "wrong padded position_ids negative control was not detected"
            )

        wrong_mask_prefill = model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(base_mask),
            position_ids=base_positions,
            use_cache=False,
            return_dict=True,
        ).logits[:, -1, :]
        wrong_mask_delta = _maximum_absolute_difference(
            wrong_mask_prefill,
            mixed_last_logits,
        )
        if wrong_mask_delta <= ABSOLUTE_TOLERANCE:
            raise PretrainedSemanticsError("wrong attention mask negative control was not detected")

        base_rows = {
            name: _select_cache(mixed_cache, (index,)) for index, name in enumerate(REQUEST_IDS)
        }
        base_row_fingerprints = {
            name: _cache_fingerprint(cache) for name, cache in base_rows.items()
        }
        if len(set(base_row_fingerprints.values())) != len(REQUEST_IDS):
            raise PretrainedSemanticsError("per-request cache fingerprints are not distinct")

        first_reference_logits: dict[str, Any] = {}
        first_reference_caches: dict[str, Any] = {}
        for index, name in enumerate(REQUEST_IDS):
            step_mask = torch.cat(
                (
                    base_mask[index : index + 1],
                    torch.ones((1, 1), dtype=torch.long, device=DEVICE_NAME),
                ),
                dim=1,
            )
            logits, cache = _cached_step(
                model,
                input_tokens=first_tokens[index : index + 1],
                attention_mask=step_mask,
                position_ids=lengths[index : index + 1].reshape(1, 1),
                cache_position=input_ids.shape[1],
                cache=_clone_cache(base_rows[name]),
            )
            first_reference_logits[name] = logits
            first_reference_caches[name] = cache

        parked_cache = _clone_cache(base_rows["request-b"])
        parked_before = _cache_fingerprint(parked_cache)
        parked_shapes_before = _cache_shape(parked_cache)
        active_indices = (0, 2)
        active_names = ("request-a", "request-c")
        physically_selected = _select_cache(mixed_cache, active_indices)
        if _cache_layers(physically_selected)[0][0].shape[0] != 2:
            raise PretrainedSemanticsError("physical cache-row removal failed")
        active_mask = torch.cat(
            (
                base_mask[list(active_indices)],
                torch.ones((2, 1), dtype=torch.long, device=DEVICE_NAME),
            ),
            dim=1,
        )
        active_logits, active_cache = _cached_step(
            model,
            input_tokens=first_tokens[list(active_indices)],
            attention_mask=active_mask,
            position_ids=lengths[list(active_indices)].reshape(2, 1),
            cache_position=input_ids.shape[1],
            cache=physically_selected,
        )
        active_reference = torch.cat(
            [first_reference_logits[name] for name in active_names],
            dim=0,
        )
        active_difference = _assert_equivalent(
            active_logits,
            active_reference,
            label="active rebatched decode after physical row removal",
        )
        parked_after_active = _cache_fingerprint(parked_cache)
        if parked_before != parked_after_active:
            raise PretrainedSemanticsError("parked request cache mutated while peers advanced")

        parked_mask = torch.cat(
            (
                base_mask[1:2],
                torch.ones((1, 1), dtype=torch.long, device=DEVICE_NAME),
            ),
            dim=1,
        )
        parked_logits, caught_up_cache = _cached_step(
            model,
            input_tokens=first_tokens[1:2],
            attention_mask=parked_mask,
            position_ids=lengths[1:2].reshape(1, 1),
            cache_position=input_ids.shape[1],
            cache=parked_cache,
        )
        parked_difference = _assert_equivalent(
            parked_logits,
            first_reference_logits["request-b"],
            label="parked request catch-up decode",
        )

        actual_after_first = {
            "request-a": _select_cache(active_cache, (0,)),
            "request-b": caught_up_cache,
            "request-c": _select_cache(active_cache, (1,)),
        }
        cache_differences = {
            name: _assert_cache_equivalent(
                actual_after_first[name],
                first_reference_caches[name],
                label=f"{name} retained cache",
            )
            for name in REQUEST_IDS
        }
        second_tokens = {
            name: int(torch.argmax(first_reference_logits[name], dim=-1).item())
            for name in REQUEST_IDS
        }

        reinsert_order = ("request-c", "request-b", "request-a")
        reinsert_cache = _concatenate_caches([actual_after_first[name] for name in reinsert_order])
        if _cache_layers(reinsert_cache)[0][0].shape[0] != 3:
            raise PretrainedSemanticsError("physical cache-row reinsertion failed")
        reinsert_inputs = torch.tensor(
            [[second_tokens[name]] for name in reinsert_order],
            dtype=torch.long,
            device=DEVICE_NAME,
        )
        index_by_name = {name: index for index, name in enumerate(REQUEST_IDS)}
        reordered_indices = [index_by_name[name] for name in reinsert_order]
        reinsert_mask = torch.cat(
            (
                base_mask[reordered_indices],
                torch.ones((3, 2), dtype=torch.long, device=DEVICE_NAME),
            ),
            dim=1,
        )
        reinsert_positions = (lengths[reordered_indices] + 1).reshape(3, 1)
        reinsert_logits, reinserted_cache = _cached_step(
            model,
            input_tokens=reinsert_inputs,
            attention_mask=reinsert_mask,
            position_ids=reinsert_positions,
            cache_position=input_ids.shape[1] + 1,
            cache=_clone_cache(reinsert_cache),
        )

        individual_second_logits: list[Any] = []
        monolithic_second_logits: list[Any] = []
        for name in reinsert_order:
            original_index = index_by_name[name]
            individual_mask = torch.cat(
                (
                    base_mask[original_index : original_index + 1],
                    torch.ones((1, 2), dtype=torch.long, device=DEVICE_NAME),
                ),
                dim=1,
            )
            individual_logits, _ = _cached_step(
                model,
                input_tokens=torch.tensor(
                    [[second_tokens[name]]],
                    dtype=torch.long,
                    device=DEVICE_NAME,
                ),
                attention_mask=individual_mask,
                position_ids=torch.tensor(
                    [[len(PROMPT_TOKEN_IDS[original_index]) + 1]],
                    dtype=torch.long,
                    device=DEVICE_NAME,
                ),
                cache_position=input_ids.shape[1] + 1,
                cache=_clone_cache(first_reference_caches[name]),
            )
            individual_second_logits.append(individual_logits)

            complete_sequence = (
                *PROMPT_TOKEN_IDS[original_index],
                int(first_tokens[original_index].item()),
                second_tokens[name],
            )
            complete_ids = torch.tensor(
                [complete_sequence],
                dtype=torch.long,
                device=DEVICE_NAME,
            )
            complete_output = model(
                input_ids=complete_ids,
                attention_mask=torch.ones_like(complete_ids),
                position_ids=torch.arange(
                    len(complete_sequence),
                    dtype=torch.long,
                    device=DEVICE_NAME,
                ).unsqueeze(0),
                use_cache=False,
                return_dict=True,
            )
            monolithic_second_logits.append(complete_output.logits[:, -1, :].detach())
        individual_second = torch.cat(individual_second_logits, dim=0)
        monolithic_second = torch.cat(monolithic_second_logits, dim=0)
        reinsert_individual_difference = _assert_equivalent(
            reinsert_logits,
            individual_second,
            label="reinserted batch versus per-request cached decode",
        )
        reinsert_monolithic_difference = _assert_equivalent(
            reinsert_logits,
            monolithic_second,
            label="reinserted cached decode versus monolithic full forward",
        )
        third_tokens = torch.argmax(reinsert_logits, dim=-1)

        wrong_cache_order = ("request-b", "request-c", "request-a")
        wrong_cache = _concatenate_caches([actual_after_first[name] for name in wrong_cache_order])
        wrong_cache_logits, _ = _cached_step(
            model,
            input_tokens=reinsert_inputs,
            attention_mask=reinsert_mask,
            position_ids=reinsert_positions,
            cache_position=input_ids.shape[1] + 1,
            cache=wrong_cache,
        )
        wrong_cache_delta = _maximum_absolute_difference(
            wrong_cache_logits,
            reinsert_logits,
        )
        if wrong_cache_delta <= ABSOLUTE_TOLERANCE:
            raise PretrainedSemanticsError("wrong cache/request association was not detected")

        wrong_decode_logits, _ = _cached_step(
            model,
            input_tokens=reinsert_inputs,
            attention_mask=reinsert_mask,
            position_ids=reinsert_positions + 1,
            cache_position=input_ids.shape[1] + 1,
            cache=_clone_cache(reinsert_cache),
        )
        wrong_decode_delta = _maximum_absolute_difference(
            wrong_decode_logits,
            reinsert_logits,
        )
        if wrong_decode_delta <= ABSOLUTE_TOLERANCE:
            raise PretrainedSemanticsError("wrong decode position association was not detected")

        reinsert_row_fingerprints = {
            name: _cache_fingerprint(_select_cache(reinserted_cache, (index,)))
            for index, name in enumerate(reinsert_order)
        }
        reinsert_logit_fingerprints = {
            name: _tensor_fingerprint(reinsert_logits[index])
            for index, name in enumerate(reinsert_order)
        }

    cuda_initialized_after = torch.cuda.is_initialized()
    if cuda_initialized_before or cuda_initialized_after:
        raise PretrainedSemanticsError("CUDA initialized during CPU-only generation")
    if guard.devices_seen != {DEVICE_NAME} or guard.non_cpu_observations != 0:
        raise PretrainedSemanticsError("tensor dispatch was not exclusively CPU")
    if network_attempts:
        raise PretrainedSemanticsError("network attempts were observed")

    config = model.config
    return {
        "schema_version": SCHEMA_VERSION,
        "evidence_class": "actual-pinned-pretrained-qwen3-cpu-semantics",
        "measurement_warning": WARNING,
        "claim_boundary": CLAIM_BOUNDARY,
        "seed": SEED,
        "implementation": {
            "path": "tools/run_pretrained_cpu_semantics.py",
            "sha256": hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest(),
        },
        "checkpoint": {
            "model_id": MODEL_ID,
            "revision": MODEL_REVISION,
            "revision_is_full_commit_sha": len(MODEL_REVISION) == 40,
            "license": MODEL_LICENSE,
            "files": checkpoint_files,
            "weights_committed_to_repository": False,
        },
        "offline_contract": {
            "download_phase": ("one explicit cache population before evidence generation"),
            "generation_local_files_only": True,
            "generation_network_access": False,
            "network_guard": "socket connect/create_connection denied",
            "network_attempts_observed": len(network_attempts),
            "snapshot_basename": snapshot.name,
            "hf_hub_offline": os.environ["HF_HUB_OFFLINE"],
            "transformers_offline": os.environ["TRANSFORMERS_OFFLINE"],
        },
        "device_contract": {
            "requested_device": DEVICE_NAME,
            "default_device": str(torch.get_default_device()),
            "accelerator_policy": "CUDA hidden; MPS forbidden by hard tensor guard",
            "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
            "mps_disabled_flag": os.environ["FISSIONSPEC_MPS_DISABLED"],
            "mps_fallback": os.environ["PYTORCH_ENABLE_MPS_FALLBACK"],
            "model_parameter_devices": sorted(
                {parameter.device.type for parameter in model.parameters()}
            ),
            "model_buffer_devices": sorted({buffer.device.type for buffer in model.buffers()}),
            "cache_devices": sorted(
                {
                    tensor.device.type
                    for cache in actual_after_first.values()
                    for pair in _cache_layers(cache)
                    for tensor in pair
                }
            ),
            "dispatch_devices_seen": sorted(guard.devices_seen),
            "dispatch_operations_audited": guard.operations,
            "non_cpu_tensor_observations": guard.non_cpu_observations,
            "cuda_available": torch.cuda.is_available(),
            "cuda_initialized_before": cuda_initialized_before,
            "cuda_initialized_after": cuda_initialized_after,
            "mps_built": torch.backends.mps.is_built(),
            "mps_available_but_forbidden": torch.backends.mps.is_available(),
            "mps_tensor_operations_observed": 0,
        },
        "runtime": {
            "python": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            **versions,
        },
        "determinism": {
            "deterministic_algorithms": (torch.are_deterministic_algorithms_enabled()),
            "torch_threads": torch.get_num_threads(),
            "inference_mode": True,
            "model_eval": not model.training,
        },
        "tokenizer": {
            "class": tokenizer_instance.__class__.__name__,
            "is_fast": tokenizer_instance.is_fast,
            "padding_side": tokenizer_instance.padding_side,
            "pad_token": tokenizer_instance.pad_token,
            "pad_token_id": tokenizer_instance.pad_token_id,
            "eos_token_id": tokenizer_instance.eos_token_id,
            "vocabulary_size_with_added_tokens": len(tokenizer_instance),
            "add_special_tokens": False,
        },
        "model": {
            "class": model.__class__.__name__,
            "parameter_count": parameter_count,
            "loaded_dtype": str(next(model.parameters()).dtype),
            "attention_implementation": config._attn_implementation,
            "config": {
                "model_type": config.model_type,
                "vocab_size": config.vocab_size,
                "hidden_size": config.hidden_size,
                "intermediate_size": config.intermediate_size,
                "num_hidden_layers": config.num_hidden_layers,
                "num_attention_heads": config.num_attention_heads,
                "num_key_value_heads": config.num_key_value_heads,
                "head_dim": config.head_dim,
                "max_position_embeddings": config.max_position_embeddings,
                "rope_theta": float(config.rope_parameters["rope_theta"]),
                "tie_word_embeddings": config.tie_word_embeddings,
            },
        },
        "requests": [
            {
                "request_id": name,
                "prompt_text": PROMPT_TEXTS[index],
                "prompt_token_ids": list(PROMPT_TOKEN_IDS[index]),
                "prompt_tokens": tokenizer_instance.convert_ids_to_tokens(
                    list(PROMPT_TOKEN_IDS[index])
                ),
                "prompt_length": len(PROMPT_TOKEN_IDS[index]),
                "first_greedy_token_id": int(first_tokens[index].item()),
                "first_greedy_text": tokenizer_instance.decode([int(first_tokens[index].item())]),
                "second_greedy_token_id": second_tokens[name],
                "second_greedy_text": tokenizer_instance.decode([second_tokens[name]]),
                "third_greedy_token_after_reinsert_id": int(
                    third_tokens[reinsert_order.index(name)].item()
                ),
                "third_greedy_text": tokenizer_instance.decode(
                    [int(third_tokens[reinsert_order.index(name)].item())]
                ),
                "prefill_cache_sha256": base_row_fingerprints[name],
                "reinserted_cache_sha256": reinsert_row_fingerprints[name],
                "reinserted_logits_sha256": reinsert_logit_fingerprints[name],
            }
            for index, name in enumerate(REQUEST_IDS)
        ],
        "equivalence": {
            "absolute_tolerance": ABSOLUTE_TOLERANCE,
            "relative_tolerance": RELATIVE_TOLERANCE,
            "mixed_left_padded_prefill_vs_monolithic_max_abs": prefill_difference,
            "active_rebatch_vs_individual_cache_max_abs": active_difference,
            "parked_catchup_vs_individual_cache_max_abs": parked_difference,
            "reinsert_vs_individual_cache_max_abs": (reinsert_individual_difference),
            "reinsert_vs_monolithic_full_forward_max_abs": (reinsert_monolithic_difference),
            "per_request_cache_max_abs": cache_differences,
            "greedy_tokens_exact": True,
        },
        "cache_ownership": {
            "mixed_prefill_shapes": _cache_shape(mixed_cache),
            "physically_selected_active_shapes_before_decode": (
                _cache_shape(_select_cache(mixed_cache, active_indices))
            ),
            "active_shapes_after_decode": _cache_shape(active_cache),
            "parked_shapes_before_decode": parked_shapes_before,
            "reinserted_shapes_after_decode": _cache_shape(reinserted_cache),
            "mixed_batch_rows": 3,
            "active_batch_rows_after_removal": 2,
            "parked_batch_rows": 1,
            "reinserted_batch_rows": 3,
            "parked_request": "request-b",
            "active_requests": list(active_names),
            "reinsert_order": list(reinsert_order),
            "parked_cache_sha256_before_peer_advance": parked_before,
            "parked_cache_sha256_after_peer_advance": parked_after_active,
            "parked_cache_byte_identical": parked_before == parked_after_active,
            "per_request_prefill_fingerprints_distinct": True,
            "per_request_cache_content_equivalent": True,
        },
        "negative_controls": {
            "wrong_left_padding_position_ids": {
                "detected": True,
                "max_abs_logit_delta": wrong_position_delta,
            },
            "wrong_attention_mask": {
                "detected": True,
                "max_abs_logit_delta": wrong_mask_delta,
            },
            "wrong_cache_request_association": {
                "detected": True,
                "max_abs_logit_delta": wrong_cache_delta,
            },
            "wrong_decode_position_ids": {
                "detected": True,
                "max_abs_logit_delta": wrong_decode_delta,
            },
        },
        "remaining_gpu_backend_seams": REMAINING_GPU_BACKEND_SEAMS,
    }


def _write_evidence(path: Path, payload: dict[str, object]) -> str:
    payload_sha256 = _sha256_document(payload)
    envelope = {**payload, "payload_sha256": payload_sha256}
    contents = _canonical_json_bytes(envelope)
    destination = path.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_bytes(contents)
    temporary.replace(destination)
    return payload_sha256


def _require_exact_object(
    value: object,
    *,
    label: str,
    keys: Sequence[str],
) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise PretrainedSemanticsError(f"{label} must be an object with string keys")
    result = cast(dict[str, object], value)
    expected = set(keys)
    actual = set(result)
    if actual != expected:
        raise PretrainedSemanticsError(
            f"{label} schema mismatch: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )
    return result


def _require_exact_list(
    value: object,
    *,
    label: str,
    length: int | None = None,
) -> list[object]:
    if not isinstance(value, list):
        raise PretrainedSemanticsError(f"{label} must be a list")
    result = cast(list[object], value)
    if length is not None and len(result) != length:
        raise PretrainedSemanticsError(f"{label} must contain exactly {length} items")
    return result


def _require_bool(value: object, *, label: str) -> bool:
    if not isinstance(value, bool):
        raise PretrainedSemanticsError(f"{label} must be a bool")
    return value


def _require_integer(
    value: object,
    *,
    label: str,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PretrainedSemanticsError(f"{label} must be an integer")
    if minimum is not None and value < minimum:
        raise PretrainedSemanticsError(f"{label} is below its minimum")
    if maximum is not None and value > maximum:
        raise PretrainedSemanticsError(f"{label} exceeds its maximum")
    return value


def _require_finite_number(
    value: object,
    *,
    label: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PretrainedSemanticsError(f"{label} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise PretrainedSemanticsError(f"{label} must be finite")
    if minimum is not None and result < minimum:
        raise PretrainedSemanticsError(f"{label} is below its minimum")
    if maximum is not None and result > maximum:
        raise PretrainedSemanticsError(f"{label} exceeds its maximum")
    return result


def _require_nonempty_string(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise PretrainedSemanticsError(f"{label} must be a non-empty string")
    return value


def _require_sha256(value: object, *, label: str) -> str:
    result = _require_nonempty_string(value, label=label)
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise PretrainedSemanticsError(f"{label} must be a lowercase SHA-256 digest")
    return result


def _strict_json_object(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"non-standard JSON constant: {value}")


def _validate_cache_shapes(
    value: object,
    *,
    label: str,
    batch_rows: int,
    sequence_length: int,
) -> None:
    entries = _require_exact_list(value, label=label, length=28)
    expected_shape = [batch_rows, 8, sequence_length, 128]
    for index, raw_entry in enumerate(entries):
        entry = _require_exact_object(
            raw_entry,
            label=f"{label}[{index}]",
            keys=("device", "dtype", "key", "layer", "value"),
        )
        if entry["device"] != "cpu" or entry["dtype"] != MODEL_DTYPE:
            raise PretrainedSemanticsError(f"{label}[{index}] device/dtype mismatch")
        if entry["layer"] != index:
            raise PretrainedSemanticsError(f"{label}[{index}] layer index mismatch")
        if entry["key"] != expected_shape or entry["value"] != expected_shape:
            raise PretrainedSemanticsError(f"{label}[{index}] cache shape mismatch")


def _validate_checkpoint(value: object) -> None:
    checkpoint = _require_exact_object(
        value,
        label="checkpoint",
        keys=(
            "files",
            "license",
            "model_id",
            "revision",
            "revision_is_full_commit_sha",
            "weights_committed_to_repository",
        ),
    )
    expected_scalars: dict[str, object] = {
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "revision_is_full_commit_sha": True,
        "license": MODEL_LICENSE,
        "weights_committed_to_repository": False,
    }
    for field, expected in expected_scalars.items():
        if checkpoint[field] != expected:
            raise PretrainedSemanticsError(f"checkpoint invariant mismatch: {field}")
    files = _require_exact_object(
        checkpoint["files"],
        label="checkpoint.files",
        keys=tuple(CHECKPOINT_FILES),
    )
    for filename, (expected_bytes, expected_sha256) in CHECKPOINT_FILES.items():
        record = _require_exact_object(
            files[filename],
            label=f"checkpoint.files.{filename}",
            keys=("bytes", "sha256"),
        )
        if record["bytes"] != expected_bytes or record["sha256"] != expected_sha256:
            raise PretrainedSemanticsError(f"checkpoint file identity mismatch: {filename}")


def verify_evidence(path: Path) -> str:
    """Verify a closed, source-bound schema using only the standard library."""

    if path.is_symlink() or not path.is_file():
        raise PretrainedSemanticsError("evidence must be a regular non-symlink file")
    try:
        contents = path.read_bytes()
        document = json.loads(
            contents,
            object_pairs_hook=_strict_json_object,
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeDecodeError, ValueError) as error:
        raise PretrainedSemanticsError("evidence is not strict UTF-8 JSON") from error
    if not isinstance(document, dict):
        raise PretrainedSemanticsError("evidence root must be an object")
    if contents != _canonical_json_bytes(document):
        raise PretrainedSemanticsError("evidence is not canonical JSON")
    envelope = _require_exact_object(
        document,
        label="evidence",
        keys=(
            "cache_ownership",
            "checkpoint",
            "claim_boundary",
            "determinism",
            "device_contract",
            "equivalence",
            "evidence_class",
            "implementation",
            "measurement_warning",
            "model",
            "negative_controls",
            "offline_contract",
            "payload_sha256",
            "remaining_gpu_backend_seams",
            "requests",
            "runtime",
            "schema_version",
            "seed",
            "tokenizer",
        ),
    )
    supplied = _require_sha256(
        envelope["payload_sha256"],
        label="payload_sha256",
    )
    payload = dict(envelope)
    payload.pop("payload_sha256")
    if _sha256_document(payload) != supplied:
        raise PretrainedSemanticsError("evidence payload SHA-256 mismatch")
    exact_scalars: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "evidence_class": "actual-pinned-pretrained-qwen3-cpu-semantics",
        "measurement_warning": WARNING,
        "claim_boundary": CLAIM_BOUNDARY,
        "seed": SEED,
    }
    for field, expected in exact_scalars.items():
        if payload[field] != expected:
            raise PretrainedSemanticsError(f"evidence invariant mismatch: {field}")

    implementation = _require_exact_object(
        payload["implementation"],
        label="implementation",
        keys=("path", "sha256"),
    )
    if implementation["path"] != "tools/run_pretrained_cpu_semantics.py":
        raise PretrainedSemanticsError("implementation path mismatch")
    source_sha256 = hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest()
    if (
        _require_sha256(
            implementation["sha256"],
            label="implementation.sha256",
        )
        != source_sha256
    ):
        raise PretrainedSemanticsError("evidence implementation hash does not match this tool")

    _validate_checkpoint(payload["checkpoint"])
    offline = _require_exact_object(
        payload["offline_contract"],
        label="offline_contract",
        keys=(
            "download_phase",
            "generation_local_files_only",
            "generation_network_access",
            "hf_hub_offline",
            "network_attempts_observed",
            "network_guard",
            "snapshot_basename",
            "transformers_offline",
        ),
    )
    expected_offline: dict[str, object] = {
        "download_phase": ("one explicit cache population before evidence generation"),
        "generation_local_files_only": True,
        "generation_network_access": False,
        "network_guard": "socket connect/create_connection denied",
        "network_attempts_observed": 0,
        "snapshot_basename": MODEL_REVISION,
        "hf_hub_offline": "1",
        "transformers_offline": "1",
    }
    if offline != expected_offline:
        raise PretrainedSemanticsError("offline contract mismatch")

    device = _require_exact_object(
        payload["device_contract"],
        label="device_contract",
        keys=(
            "accelerator_policy",
            "cache_devices",
            "cuda_available",
            "cuda_initialized_after",
            "cuda_initialized_before",
            "cuda_visible_devices",
            "default_device",
            "dispatch_devices_seen",
            "dispatch_operations_audited",
            "model_buffer_devices",
            "model_parameter_devices",
            "mps_available_but_forbidden",
            "mps_built",
            "mps_disabled_flag",
            "mps_fallback",
            "mps_tensor_operations_observed",
            "non_cpu_tensor_observations",
            "requested_device",
        ),
    )
    required_device: dict[str, object] = {
        "requested_device": "cpu",
        "default_device": "cpu",
        "accelerator_policy": "CUDA hidden; MPS forbidden by hard tensor guard",
        "cuda_visible_devices": "",
        "mps_disabled_flag": "1",
        "mps_fallback": "0",
        "model_parameter_devices": ["cpu"],
        "model_buffer_devices": ["cpu"],
        "cache_devices": ["cpu"],
        "dispatch_devices_seen": ["cpu"],
        "cuda_initialized_before": False,
        "cuda_initialized_after": False,
        "mps_tensor_operations_observed": 0,
        "non_cpu_tensor_observations": 0,
    }
    for field, expected in required_device.items():
        if device[field] != expected:
            raise PretrainedSemanticsError(f"device contract mismatch: {field}")
    for field in ("cuda_available", "mps_built", "mps_available_but_forbidden"):
        _require_bool(device[field], label=f"device_contract.{field}")
    _require_integer(
        device["dispatch_operations_audited"],
        label="device_contract.dispatch_operations_audited",
        minimum=10_000,
    )

    runtime = _require_exact_object(
        payload["runtime"],
        label="runtime",
        keys=(
            "huggingface_hub",
            "machine",
            "platform",
            "python",
            "python_implementation",
            "safetensors",
            "tokenizers",
            "torch",
            "transformers",
        ),
    )
    expected_runtime_versions = {
        "torch": EXPECTED_TORCH_VERSION,
        "transformers": EXPECTED_TRANSFORMERS_VERSION,
        "huggingface_hub": EXPECTED_HUB_VERSION,
        "safetensors": EXPECTED_SAFETENSORS_VERSION,
        "tokenizers": EXPECTED_TOKENIZERS_VERSION,
        "python_implementation": "CPython",
    }
    for field, expected in expected_runtime_versions.items():
        if runtime[field] != expected:
            raise PretrainedSemanticsError(f"runtime mismatch: {field}")
    for field in ("machine", "platform", "python"):
        _require_nonempty_string(runtime[field], label=f"runtime.{field}")

    determinism = _require_exact_object(
        payload["determinism"],
        label="determinism",
        keys=(
            "deterministic_algorithms",
            "inference_mode",
            "model_eval",
            "torch_threads",
        ),
    )
    if determinism != {
        "deterministic_algorithms": True,
        "inference_mode": True,
        "model_eval": True,
        "torch_threads": 1,
    }:
        raise PretrainedSemanticsError("determinism contract mismatch")

    tokenizer_document = _require_exact_object(
        payload["tokenizer"],
        label="tokenizer",
        keys=(
            "add_special_tokens",
            "class",
            "eos_token_id",
            "is_fast",
            "pad_token",
            "pad_token_id",
            "padding_side",
            "vocabulary_size_with_added_tokens",
        ),
    )
    expected_tokenizer: dict[str, object] = {
        "class": "Qwen2Tokenizer",
        "is_fast": True,
        "padding_side": "left",
        "pad_token": "<|endoftext|>",
        "pad_token_id": 151643,
        "eos_token_id": 151645,
        "vocabulary_size_with_added_tokens": 151669,
        "add_special_tokens": False,
    }
    if tokenizer_document != expected_tokenizer:
        raise PretrainedSemanticsError("tokenizer contract mismatch")

    model = _require_exact_object(
        payload["model"],
        label="model",
        keys=(
            "attention_implementation",
            "class",
            "config",
            "loaded_dtype",
            "parameter_count",
        ),
    )
    if (
        model["class"] != "Qwen3ForCausalLM"
        or model["loaded_dtype"] != MODEL_DTYPE
        or model["attention_implementation"] != ATTENTION_IMPLEMENTATION
    ):
        raise PretrainedSemanticsError("model loading contract mismatch")
    _require_integer(
        model["parameter_count"],
        label="model.parameter_count",
        minimum=500_000_000,
        maximum=700_000_000,
    )
    config = _require_exact_object(
        model["config"],
        label="model.config",
        keys=(
            "head_dim",
            "hidden_size",
            "intermediate_size",
            "max_position_embeddings",
            "model_type",
            "num_attention_heads",
            "num_hidden_layers",
            "num_key_value_heads",
            "rope_theta",
            "tie_word_embeddings",
            "vocab_size",
        ),
    )
    if config != {
        "model_type": "qwen3",
        "vocab_size": 151936,
        "hidden_size": 1024,
        "intermediate_size": 3072,
        "num_hidden_layers": 28,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "max_position_embeddings": 40960,
        "rope_theta": 1_000_000.0,
        "tie_word_embeddings": True,
    }:
        raise PretrainedSemanticsError("model config mismatch")

    requests = _require_exact_list(
        payload["requests"],
        label="requests",
        length=3,
    )
    prefill_hashes: set[str] = set()
    for index, raw_request in enumerate(requests):
        request = _require_exact_object(
            raw_request,
            label=f"requests[{index}]",
            keys=(
                "first_greedy_text",
                "first_greedy_token_id",
                "prefill_cache_sha256",
                "prompt_length",
                "prompt_text",
                "prompt_token_ids",
                "prompt_tokens",
                "reinserted_cache_sha256",
                "reinserted_logits_sha256",
                "request_id",
                "second_greedy_text",
                "second_greedy_token_id",
                "third_greedy_text",
                "third_greedy_token_after_reinsert_id",
            ),
        )
        if (
            request["request_id"] != REQUEST_IDS[index]
            or request["prompt_text"] != PROMPT_TEXTS[index]
            or request["prompt_token_ids"] != list(PROMPT_TOKEN_IDS[index])
            or request["prompt_length"] != len(PROMPT_TOKEN_IDS[index])
        ):
            raise PretrainedSemanticsError(f"request identity/tokenization mismatch: {index}")
        prompt_tokens = _require_exact_list(
            request["prompt_tokens"],
            label=f"requests[{index}].prompt_tokens",
            length=len(PROMPT_TOKEN_IDS[index]),
        )
        if any(not isinstance(token, str) or not token for token in prompt_tokens):
            raise PretrainedSemanticsError("prompt token text is malformed")
        for field in (
            "first_greedy_token_id",
            "second_greedy_token_id",
            "third_greedy_token_after_reinsert_id",
        ):
            _require_integer(
                request[field],
                label=f"requests[{index}].{field}",
                minimum=0,
                maximum=151935,
            )
        for field in (
            "first_greedy_text",
            "second_greedy_text",
            "third_greedy_text",
        ):
            if not isinstance(request[field], str):
                raise PretrainedSemanticsError(f"requests[{index}].{field} must be a string")
        prefill_hashes.add(
            _require_sha256(
                request["prefill_cache_sha256"],
                label=f"requests[{index}].prefill_cache_sha256",
            )
        )
        for field in (
            "reinserted_cache_sha256",
            "reinserted_logits_sha256",
        ):
            _require_sha256(
                request[field],
                label=f"requests[{index}].{field}",
            )
    if len(prefill_hashes) != len(REQUEST_IDS):
        raise PretrainedSemanticsError("per-request prefill cache hashes are not distinct")

    equivalence = _require_exact_object(
        payload["equivalence"],
        label="equivalence",
        keys=(
            "absolute_tolerance",
            "active_rebatch_vs_individual_cache_max_abs",
            "greedy_tokens_exact",
            "mixed_left_padded_prefill_vs_monolithic_max_abs",
            "parked_catchup_vs_individual_cache_max_abs",
            "per_request_cache_max_abs",
            "reinsert_vs_individual_cache_max_abs",
            "reinsert_vs_monolithic_full_forward_max_abs",
            "relative_tolerance",
        ),
    )
    if (
        equivalence["absolute_tolerance"] != ABSOLUTE_TOLERANCE
        or equivalence["relative_tolerance"] != RELATIVE_TOLERANCE
        or equivalence["greedy_tokens_exact"] is not True
    ):
        raise PretrainedSemanticsError("equivalence contract mismatch")
    for field in (
        "active_rebatch_vs_individual_cache_max_abs",
        "mixed_left_padded_prefill_vs_monolithic_max_abs",
        "parked_catchup_vs_individual_cache_max_abs",
        "reinsert_vs_individual_cache_max_abs",
        "reinsert_vs_monolithic_full_forward_max_abs",
    ):
        _require_finite_number(
            equivalence[field],
            label=f"equivalence.{field}",
            minimum=0.0,
            maximum=ABSOLUTE_TOLERANCE,
        )
    cache_differences = _require_exact_object(
        equivalence["per_request_cache_max_abs"],
        label="equivalence.per_request_cache_max_abs",
        keys=REQUEST_IDS,
    )
    for request_id in REQUEST_IDS:
        _require_finite_number(
            cache_differences[request_id],
            label=f"equivalence.per_request_cache_max_abs.{request_id}",
            minimum=0.0,
            maximum=ABSOLUTE_TOLERANCE,
        )

    ownership = _require_exact_object(
        payload["cache_ownership"],
        label="cache_ownership",
        keys=(
            "active_batch_rows_after_removal",
            "active_requests",
            "active_shapes_after_decode",
            "mixed_batch_rows",
            "mixed_prefill_shapes",
            "parked_batch_rows",
            "parked_cache_byte_identical",
            "parked_cache_sha256_after_peer_advance",
            "parked_cache_sha256_before_peer_advance",
            "parked_request",
            "parked_shapes_before_decode",
            "per_request_cache_content_equivalent",
            "per_request_prefill_fingerprints_distinct",
            "physically_selected_active_shapes_before_decode",
            "reinsert_order",
            "reinserted_batch_rows",
            "reinserted_shapes_after_decode",
        ),
    )
    expected_ownership: dict[str, object] = {
        "mixed_batch_rows": 3,
        "active_batch_rows_after_removal": 2,
        "parked_batch_rows": 1,
        "reinserted_batch_rows": 3,
        "parked_request": "request-b",
        "active_requests": ["request-a", "request-c"],
        "reinsert_order": ["request-c", "request-b", "request-a"],
        "parked_cache_byte_identical": True,
        "per_request_cache_content_equivalent": True,
        "per_request_prefill_fingerprints_distinct": True,
    }
    for field, expected in expected_ownership.items():
        if ownership[field] != expected:
            raise PretrainedSemanticsError(f"cache ownership invariant mismatch: {field}")
    before = _require_sha256(
        ownership["parked_cache_sha256_before_peer_advance"],
        label="parked cache before",
    )
    after = _require_sha256(
        ownership["parked_cache_sha256_after_peer_advance"],
        label="parked cache after",
    )
    if before != after:
        raise PretrainedSemanticsError("parked cache bytes changed")
    _validate_cache_shapes(
        ownership["mixed_prefill_shapes"],
        label="mixed_prefill_shapes",
        batch_rows=3,
        sequence_length=5,
    )
    _validate_cache_shapes(
        ownership["physically_selected_active_shapes_before_decode"],
        label="physically_selected_active_shapes_before_decode",
        batch_rows=2,
        sequence_length=5,
    )
    _validate_cache_shapes(
        ownership["active_shapes_after_decode"],
        label="active_shapes_after_decode",
        batch_rows=2,
        sequence_length=6,
    )
    _validate_cache_shapes(
        ownership["parked_shapes_before_decode"],
        label="parked_shapes_before_decode",
        batch_rows=1,
        sequence_length=5,
    )
    _validate_cache_shapes(
        ownership["reinserted_shapes_after_decode"],
        label="reinserted_shapes_after_decode",
        batch_rows=3,
        sequence_length=7,
    )

    controls = _require_exact_object(
        payload["negative_controls"],
        label="negative_controls",
        keys=NEGATIVE_CONTROL_KEYS,
    )
    for control_name in NEGATIVE_CONTROL_KEYS:
        control = _require_exact_object(
            controls[control_name],
            label=f"negative_controls.{control_name}",
            keys=("detected", "max_abs_logit_delta"),
        )
        if control["detected"] is not True:
            raise PretrainedSemanticsError(f"negative control not detected: {control_name}")
        delta = _require_finite_number(
            control["max_abs_logit_delta"],
            label=f"negative_controls.{control_name}.max_abs_logit_delta",
            minimum=0.0,
        )
        if delta <= ABSOLUTE_TOLERANCE:
            raise PretrainedSemanticsError(f"negative control delta is too small: {control_name}")

    seams = _require_exact_list(
        payload["remaining_gpu_backend_seams"],
        label="remaining_gpu_backend_seams",
        length=len(REMAINING_GPU_BACKEND_SEAMS),
    )
    if seams != list(REMAINING_GPU_BACKEND_SEAMS):
        raise PretrainedSemanticsError("remaining GPU seam declaration mismatch")

    frozen_results = dict(payload)
    frozen_results.pop("implementation")
    observed_frozen = _sha256_document(frozen_results)
    if observed_frozen != EXPECTED_FROZEN_RESULTS_SHA256:
        raise PretrainedSemanticsError(
            f"frozen semantic result map mismatch: observed={observed_frozen}"
        )
    return supplied


def run_gate(output: Path) -> tuple[str, float]:
    """Generate, write, and verify evidence with networking denied."""

    started = time.monotonic()
    with _network_denied() as network_attempts:
        _load_heavy_stack()
        previous_default_device = torch.get_default_device()
        previous_default_dtype = torch.get_default_dtype()
        previous_deterministic = torch.are_deterministic_algorithms_enabled()
        previous_threads = torch.get_num_threads()
        try:
            payload = _run_semantics(network_attempts)
        finally:
            torch.set_default_device(previous_default_device)
            torch.set_default_dtype(previous_default_dtype)
            torch.use_deterministic_algorithms(previous_deterministic)
            torch.set_num_threads(previous_threads)
    digest = _write_evidence(output, payload)
    if verify_evidence(output) != digest:
        raise AssertionError("fresh evidence did not verify to its own digest")
    return digest, time.monotonic() - started


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/results/pretrained_cpu_semantics/evidence.json"),
    )
    parser.add_argument("--verify-only", type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.verify_only is not None:
        print(f"verified payload sha256={verify_evidence(args.verify_only)}")
        return 0
    digest, elapsed = run_gate(args.output)
    print(
        f"wrote {args.output.resolve()} payload_sha256={digest} "
        f"runtime={elapsed:.3f}s device=cpu downloads=0"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
