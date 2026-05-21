#!/usr/bin/env python3
"""Run a no-download, CPU-only transformer cache/rebatch semantics gate.

The gate instantiates a tiny random Qwen2 causal LM directly from configuration.
It never loads a tokenizer, checkpoint, pretrained weight, or remote artifact.
Every tensor operation executed by the model is observed by a dispatch guard
that rejects non-CPU tensors.
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
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

SCHEMA_VERSION: Final = 1
SEED: Final = 20260723
DEVICE_NAME: Final = "cpu"
ABSOLUTE_TOLERANCE: Final = 2e-5
RELATIVE_TOLERANCE: Final = 2e-5
EXPECTED_TORCH_VERSION: Final = "2.10.0"
EXPECTED_TRANSFORMERS_VERSION: Final = "5.5.0"
EXPECTED_DISPATCH_OPERATIONS: Final = 4_160
EXPECTED_FROZEN_RESULTS_SHA256: Final = (
    "2529a5e703f1daeec4dba4cb6a45faa97ce141cecccd792060e06e6060180721"
)
WARNING: Final = "ACTUAL TRANSFORMER CPU SEMANTICS — NOT A GPU KERNEL OR SERVING MEASUREMENT."
CLAIM_BOUNDARY: Final = (
    "This artifact exercises the installed PyTorch/Transformers Qwen2 forward and "
    "DynamicCache stack on CPU. It does not exercise CUDA/MPS kernels, CUDA graphs, "
    "paged-attention allocators, tensor parallelism, or a production serving engine."
)
NEGATIVE_CONTROL_KEYS: Final = (
    "wrong_attention_mask",
    "wrong_cache_request_association",
    "wrong_decode_position_ids",
    "wrong_left_padding_position_ids",
)
REQUEST_IDS: Final = ("request-a", "request-b", "request-c")
PROMPTS: Final = (
    (11, 12, 13),
    (21, 22, 23, 24),
    (31, 32, 33, 34, 35),
)
REMAINING_GPU_BACKEND_SEAMS: Final = (
    "production tokenizer and pretrained-weight equivalence",
    "CUDA/MPS kernel numerical behavior and fused attention",
    "paged KV allocation, physical row compaction, and cache handles",
    "CUDA graph capture and graph-bucket shape transitions",
    "tensor/pipeline parallel communication and remote drafting",
    "production scheduler callbacks, cancellation, OOM, and stream ordering",
    "quantized, mixed-precision, and long-context behavior",
)

torch: Any = None
transformers: Any = None
Qwen2Config: Any = None
Qwen2ForCausalLM: Any = None
DynamicCache: Any = None


class SemanticsGateError(RuntimeError):
    """Raised when transformer semantics or evidence integrity fails."""


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


def _load_heavy_stack() -> None:
    """Import optional runtime dependencies only for evidence generation."""

    global torch, transformers, Qwen2Config, Qwen2ForCausalLM, DynamicCache
    if torch is not None:
        return
    torch = importlib.import_module("torch")
    transformers = importlib.import_module("transformers")
    Qwen2Config = transformers.Qwen2Config
    Qwen2ForCausalLM = transformers.Qwen2ForCausalLM
    cache_utils = importlib.import_module("transformers.cache_utils")
    DynamicCache = cache_utils.DynamicCache


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
    """State collected by the dynamically constructed Torch dispatch mode."""

    def __init__(self) -> None:
        self.operations = 0
        self.devices_seen: set[str] = set()

    def _audit(self, value: object, *, phase: str) -> None:
        for tensor in _walk_tensors(value):
            device_type = tensor.device.type
            self.devices_seen.add(device_type)
            if device_type != "cpu":
                raise SemanticsGateError(
                    f"{phase} tensor escaped CPU-only contract: {tensor.device}"
                )


def _cpu_only_dispatch_mode() -> tuple[Any, CpuDispatchAudit]:
    """Construct TorchDispatchMode without importing Torch during verification."""

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
        "CpuOnlyDispatchMode",
        (dispatch_base,),
        {"__torch_dispatch__": dispatch},
    )
    return mode_type(), audit


@contextmanager
def _network_denied() -> Iterator[None]:
    """Make an accidental network access fail synchronously."""

    failure = SemanticsGateError(
        "network access attempted during the no-download transformer semantics gate"
    )
    with (
        mock.patch.object(socket.socket, "connect", side_effect=failure),
        mock.patch.object(socket, "create_connection", side_effect=failure),
    ):
        yield


def _tiny_config() -> Any:
    return Qwen2Config(
        vocab_size=97,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        rope_theta=10_000.0,
        attention_dropout=0.0,
        use_cache=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
    )


def _tensor_fingerprint(tensor: Any) -> str:
    value = tensor.detach().contiguous().cpu()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(b"\x00")
    digest.update(",".join(str(item) for item in value.shape).encode("ascii"))
    digest.update(b"\x00")
    digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _model_fingerprint(model: Any) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        encoded_name = name.encode("utf-8")
        digest.update(len(encoded_name).to_bytes(8, "big"))
        digest.update(encoded_name)
        digest.update(bytes.fromhex(_tensor_fingerprint(tensor)))
    return digest.hexdigest()


def _cache_layers(cache: Any) -> tuple[tuple[Any, Any], ...]:
    result: list[tuple[Any, Any]] = []
    for index, layer in enumerate(cache.layers):
        keys = layer.keys
        values = layer.values
        if not isinstance(keys, torch.Tensor) or not isinstance(values, torch.Tensor):
            raise SemanticsGateError(f"cache layer {index} is not initialized")
        if keys.device.type != "cpu" or values.device.type != "cpu":
            raise SemanticsGateError(f"cache layer {index} is not on CPU")
        result.append((keys, values))
    if not result:
        raise SemanticsGateError("transformer returned an empty cache")
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
        raise SemanticsGateError("cannot concatenate caches with different layer counts")
    combined: list[tuple[Any, Any]] = []
    for layer_index in range(len(layers[0])):
        sequence_lengths = {item[layer_index][0].shape[-2] for item in layers}
        if len(sequence_lengths) != 1:
            raise SemanticsGateError("cannot reinsert caches with different sequence lengths")
        combined.append(
            (
                torch.cat([item[layer_index][0] for item in layers], dim=0),
                torch.cat([item[layer_index][1] for item in layers], dim=0),
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
        raise SemanticsGateError(
            f"tensor shape mismatch: {tuple(left.shape)} != {tuple(right.shape)}"
        )
    return float(torch.max(torch.abs(left - right)).item())


def _assert_equivalent(
    left: Any,
    right: Any,
    *,
    label: str,
) -> float:
    difference = _maximum_absolute_difference(left, right)
    if not torch.allclose(
        left,
        right,
        atol=ABSOLUTE_TOLERANCE,
        rtol=RELATIVE_TOLERANCE,
    ):
        raise SemanticsGateError(
            f"{label} logits differ: max_abs={difference:.9g}, "
            f"atol={ABSOLUTE_TOLERANCE}, rtol={RELATIVE_TOLERANCE}"
        )
    left_tokens = torch.argmax(left, dim=-1)
    right_tokens = torch.argmax(right, dim=-1)
    if not torch.equal(left_tokens, right_tokens):
        raise SemanticsGateError(f"{label} greedy token mismatch")
    return difference


def _assert_cache_equivalent(
    left: Any,
    right: Any,
    *,
    label: str,
) -> float:
    left_layers = _cache_layers(left)
    right_layers = _cache_layers(right)
    if len(left_layers) != len(right_layers):
        raise SemanticsGateError(f"{label} cache layer count differs")
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
            raise SemanticsGateError(f"{label} cache differs at layer {index}")
    return maximum


def _prepare_prompts() -> tuple[
    tuple[str, ...],
    tuple[tuple[int, ...], ...],
    Any,
    Any,
    Any,
]:
    names = REQUEST_IDS
    prompts = PROMPTS
    maximum_length = max(len(prompt) for prompt in prompts)
    padded = [(0,) * (maximum_length - len(prompt)) + prompt for prompt in prompts]
    input_ids = torch.tensor(padded, dtype=torch.long, device=DEVICE_NAME)
    attention_mask = input_ids.ne(0).to(dtype=torch.long)
    position_ids = (attention_mask.cumsum(dim=-1) - 1).clamp_min(0)
    return names, prompts, input_ids, attention_mask, position_ids


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
        raise SemanticsGateError("Qwen2 did not return DynamicCache")
    return output.logits[:, -1, :].detach(), updated


def _run_semantics() -> dict[str, object]:
    if torch.__version__ != EXPECTED_TORCH_VERSION:
        raise SemanticsGateError(
            f"torch version drift: {torch.__version__!r} != {EXPECTED_TORCH_VERSION!r}"
        )
    if transformers.__version__ != EXPECTED_TRANSFORMERS_VERSION:
        raise SemanticsGateError(
            "transformers version drift: "
            f"{transformers.__version__!r} != {EXPECTED_TRANSFORMERS_VERSION!r}"
        )
    torch.manual_seed(SEED)
    torch.set_default_device("cpu")
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    cuda_initialized_before = torch.cuda.is_initialized()
    config = _tiny_config()

    with _network_denied():
        model = Qwen2ForCausalLM(config)
        model.eval()
        model.to(device=DEVICE_NAME, dtype=torch.float32)
        if any(parameter.device.type != "cpu" for parameter in model.parameters()):
            raise SemanticsGateError("model parameter escaped CPU")
        if any(buffer.device.type != "cpu" for buffer in model.buffers()):
            raise SemanticsGateError("model buffer escaped CPU")
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        model_sha256 = _model_fingerprint(model)
        names, prompts, input_ids, base_mask, base_positions = _prepare_prompts()
        lengths = torch.tensor(
            [len(prompt) for prompt in prompts],
            dtype=torch.long,
            device=DEVICE_NAME,
        )

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
                raise SemanticsGateError("mixed prefill did not return DynamicCache")
            mixed_last_logits = mixed_prefill.logits[:, -1, :].detach()

            monolithic_prefill_logits: list[Any] = []
            for prompt in prompts:
                prompt_ids = torch.tensor(
                    [prompt],
                    dtype=torch.long,
                    device=DEVICE_NAME,
                )
                prompt_positions = torch.arange(
                    len(prompt),
                    dtype=torch.long,
                    device=DEVICE_NAME,
                ).unsqueeze(0)
                output = model(
                    input_ids=prompt_ids,
                    attention_mask=torch.ones_like(prompt_ids),
                    position_ids=prompt_positions,
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
                raise SemanticsGateError(
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
                raise SemanticsGateError("wrong attention mask negative control was not detected")

            base_rows = {
                name: _select_cache(mixed_cache, (index,)) for index, name in enumerate(names)
            }
            base_row_fingerprints = {
                name: _cache_fingerprint(cache) for name, cache in base_rows.items()
            }
            if len(set(base_row_fingerprints.values())) != len(names):
                raise SemanticsGateError("per-request cache fingerprints are not distinct")

            first_reference_logits: dict[str, Any] = {}
            first_reference_caches: dict[str, Any] = {}
            for index, name in enumerate(names):
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
            active_indices = (0, 2)
            active_names = ("request-a", "request-c")
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
                cache=_select_cache(mixed_cache, active_indices),
            )
            active_reference = torch.cat(
                [first_reference_logits[name] for name in active_names],
                dim=0,
            )
            active_difference = _assert_equivalent(
                active_logits,
                active_reference,
                label="active rebatched decode after parking request-b",
            )
            parked_after_active = _cache_fingerprint(parked_cache)
            if parked_before != parked_after_active:
                raise SemanticsGateError("parked request cache mutated while peers advanced")

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
                for name in names
            }
            second_tokens = {
                name: int(torch.argmax(first_reference_logits[name], dim=-1).item())
                for name in names
            }

            reinsert_order = ("request-c", "request-b", "request-a")
            reinsert_cache = _concatenate_caches(
                [actual_after_first[name] for name in reinsert_order]
            )
            reinsert_inputs = torch.tensor(
                [[second_tokens[name]] for name in reinsert_order],
                dtype=torch.long,
                device=DEVICE_NAME,
            )
            index_by_name = {name: index for index, name in enumerate(names)}
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
                        [[len(prompts[original_index]) + 1]],
                        dtype=torch.long,
                        device=DEVICE_NAME,
                    ),
                    cache_position=input_ids.shape[1] + 1,
                    cache=_clone_cache(first_reference_caches[name]),
                )
                individual_second_logits.append(individual_logits)

                complete_sequence = (
                    *prompts[original_index],
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
                label="reinserted mixed batch versus per-request cached decode",
            )
            reinsert_monolithic_difference = _assert_equivalent(
                reinsert_logits,
                monolithic_second,
                label="reinserted mixed batch versus monolithic full forward",
            )
            third_tokens = torch.argmax(reinsert_logits, dim=-1)

            wrong_cache_order = ("request-b", "request-c", "request-a")
            wrong_cache = _concatenate_caches(
                [actual_after_first[name] for name in wrong_cache_order]
            )
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
                raise SemanticsGateError("wrong cache/request association was not detected")

            wrong_decode_position_logits, _ = _cached_step(
                model,
                input_tokens=reinsert_inputs,
                attention_mask=reinsert_mask,
                position_ids=reinsert_positions + 1,
                cache_position=input_ids.shape[1] + 1,
                cache=_clone_cache(reinsert_cache),
            )
            wrong_decode_position_delta = _maximum_absolute_difference(
                wrong_decode_position_logits,
                reinsert_logits,
            )
            if wrong_decode_position_delta <= ABSOLUTE_TOLERANCE:
                raise SemanticsGateError("wrong decode position association was not detected")

            reinsert_row_fingerprints = {
                name: _cache_fingerprint(_select_cache(reinserted_cache, (index,)))
                for index, name in enumerate(reinsert_order)
            }

    cuda_initialized_after = torch.cuda.is_initialized()
    if cuda_initialized_before or cuda_initialized_after:
        raise SemanticsGateError("CUDA was initialized during a CPU-only gate")
    if guard.devices_seen != {"cpu"}:
        raise SemanticsGateError(
            f"dispatch guard did not observe an exclusively CPU execution: {guard.devices_seen}"
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "evidence_class": "actual-random-transformer-cpu-semantics",
        "measurement_warning": WARNING,
        "claim_boundary": CLAIM_BOUNDARY,
        "seed": SEED,
        "implementation": {
            "path": "tools/run_cpu_transformer_semantics.py",
            "sha256": hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest(),
        },
        "offline_contract": {
            "pretrained_weights_loaded": False,
            "tokenizer_loaded": False,
            "network_guard": "socket connect/create_connection denied",
            "network_downloads": False,
            "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
            "hf_hub_offline": os.environ["HF_HUB_OFFLINE"],
            "transformers_offline": os.environ["TRANSFORMERS_OFFLINE"],
        },
        "device_contract": {
            "requested_device": DEVICE_NAME,
            "default_device": str(torch.get_default_device()),
            "model_parameter_devices": sorted(
                {parameter.device.type for parameter in model.parameters()}
            ),
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
            "cuda_available": torch.cuda.is_available(),
            "cuda_initialized_before": cuda_initialized_before,
            "cuda_initialized_after": cuda_initialized_after,
            "mps_built": torch.backends.mps.is_built(),
            "mps_available": torch.backends.mps.is_available(),
            "mps_tensor_operations_observed": 0,
            "non_cpu_tensor_observations": 0,
        },
        "runtime": {
            "python": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "determinism": {
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "torch_threads": torch.get_num_threads(),
            "model_state_sha256": model_sha256,
        },
        "model": {
            "architecture": "Qwen2ForCausalLM",
            "source": "random initialization from local Qwen2Config",
            "parameter_count": parameter_count,
            "dtype": "torch.float32",
            "config": {
                "vocab_size": config.vocab_size,
                "hidden_size": config.hidden_size,
                "intermediate_size": config.intermediate_size,
                "num_hidden_layers": config.num_hidden_layers,
                "num_attention_heads": config.num_attention_heads,
                "num_key_value_heads": config.num_key_value_heads,
                "max_position_embeddings": config.max_position_embeddings,
            },
        },
        "requests": [
            {
                "request_id": name,
                "prompt_token_ids": list(prompt),
                "prompt_length": len(prompt),
                "first_greedy_token": int(first_tokens[index].item()),
                "second_greedy_token": second_tokens[name],
                "third_greedy_token_after_reinsert": int(
                    third_tokens[reinsert_order.index(name)].item()
                ),
                "prefill_cache_sha256": base_row_fingerprints[name],
                "reinserted_cache_sha256": reinsert_row_fingerprints[name],
            }
            for index, (name, prompt) in enumerate(zip(names, prompts, strict=True))
        ],
        "equivalence": {
            "absolute_tolerance": ABSOLUTE_TOLERANCE,
            "relative_tolerance": RELATIVE_TOLERANCE,
            "mixed_left_padded_prefill_vs_monolithic_max_abs": prefill_difference,
            "active_rebatch_vs_individual_cache_max_abs": active_difference,
            "parked_catchup_vs_individual_cache_max_abs": parked_difference,
            "reinsert_vs_individual_cache_max_abs": reinsert_individual_difference,
            "reinsert_vs_monolithic_full_forward_max_abs": (reinsert_monolithic_difference),
            "per_request_cache_max_abs": cache_differences,
            "greedy_tokens_exact": True,
        },
        "cache_ownership": {
            "mixed_prefill_shapes": _cache_shape(mixed_cache),
            "parked_request": "request-b",
            "parked_cache_sha256_before_peer_advance": parked_before,
            "parked_cache_sha256_after_peer_advance": parked_after_active,
            "parked_cache_byte_identical": parked_before == parked_after_active,
            "reinsert_order": list(reinsert_order),
            "reinserted_shapes": _cache_shape(reinserted_cache),
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
                "max_abs_logit_delta": wrong_decode_position_delta,
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
        raise SemanticsGateError(f"{label} must be an object with string keys")
    result = cast(dict[str, object], value)
    expected = set(keys)
    actual = set(result)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise SemanticsGateError(f"{label} schema mismatch: missing={missing}, extra={extra}")
    return result


def _require_exact_list(
    value: object,
    *,
    label: str,
    length: int | None = None,
) -> list[object]:
    if not isinstance(value, list):
        raise SemanticsGateError(f"{label} must be a list")
    result = cast(list[object], value)
    if length is not None and len(result) != length:
        raise SemanticsGateError(f"{label} must contain exactly {length} items")
    return result


def _require_bool(value: object, *, label: str) -> bool:
    if not isinstance(value, bool):
        raise SemanticsGateError(f"{label} must be a bool")
    return value


def _require_integer(
    value: object,
    *,
    label: str,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SemanticsGateError(f"{label} must be an integer")
    if minimum is not None and value < minimum:
        raise SemanticsGateError(f"{label} is below its minimum")
    if maximum is not None and value > maximum:
        raise SemanticsGateError(f"{label} exceeds its maximum")
    return value


def _require_finite_number(
    value: object,
    *,
    label: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SemanticsGateError(f"{label} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise SemanticsGateError(f"{label} must be finite")
    if minimum is not None and result < minimum:
        raise SemanticsGateError(f"{label} is below its minimum")
    if maximum is not None and result > maximum:
        raise SemanticsGateError(f"{label} exceeds its maximum")
    return result


def _require_nonempty_string(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise SemanticsGateError(f"{label} must be a non-empty string")
    return value


def _require_sha256(value: object, *, label: str) -> str:
    result = _require_nonempty_string(value, label=label)
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise SemanticsGateError(f"{label} must be a lowercase SHA-256 digest")
    return result


def _strict_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
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
    sequence_length: int,
) -> None:
    entries = _require_exact_list(value, label=label, length=2)
    expected_shape = [3, 2, sequence_length, 8]
    for index, raw_entry in enumerate(entries):
        entry = _require_exact_object(
            raw_entry,
            label=f"{label}[{index}]",
            keys=("device", "dtype", "key", "layer", "value"),
        )
        if entry["device"] != "cpu" or entry["dtype"] != "torch.float32":
            raise SemanticsGateError(f"{label}[{index}] device/dtype mismatch")
        if entry["layer"] != index:
            raise SemanticsGateError(f"{label}[{index}] layer index mismatch")
        if entry["key"] != expected_shape or entry["value"] != expected_shape:
            raise SemanticsGateError(f"{label}[{index}] cache shape mismatch")


def verify_evidence(path: Path) -> str:
    """Verify a closed, source-bound evidence schema using only the stdlib."""

    if path.is_symlink() or not path.is_file():
        raise SemanticsGateError("evidence must be a regular non-symlink file")
    try:
        contents = path.read_bytes()
        document = json.loads(
            contents,
            object_pairs_hook=_strict_json_object,
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeDecodeError, ValueError) as error:
        raise SemanticsGateError("evidence is not strict UTF-8 JSON") from error
    if not isinstance(document, dict):
        raise SemanticsGateError("evidence root must be an object")
    if contents != _canonical_json_bytes(document):
        raise SemanticsGateError("evidence is not canonical JSON")
    envelope = _require_exact_object(
        document,
        label="evidence",
        keys=(
            "cache_ownership",
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
        ),
    )
    supplied = _require_sha256(envelope["payload_sha256"], label="payload_sha256")
    payload = dict(envelope)
    payload.pop("payload_sha256")
    if _sha256_document(payload) != supplied:
        raise SemanticsGateError("evidence payload SHA-256 mismatch")
    exact_scalars = {
        "schema_version": SCHEMA_VERSION,
        "evidence_class": "actual-random-transformer-cpu-semantics",
        "measurement_warning": WARNING,
        "claim_boundary": CLAIM_BOUNDARY,
        "seed": SEED,
    }
    for field, expected in exact_scalars.items():
        if payload[field] != expected:
            raise SemanticsGateError(f"evidence invariant mismatch: {field}")

    implementation = _require_exact_object(
        payload["implementation"],
        label="implementation",
        keys=("path", "sha256"),
    )
    if implementation["path"] != "tools/run_cpu_transformer_semantics.py":
        raise SemanticsGateError("implementation path mismatch")
    source_sha256 = hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest()
    if _require_sha256(implementation["sha256"], label="implementation.sha256") != source_sha256:
        raise SemanticsGateError("evidence implementation hash does not match this tool")

    offline = _require_exact_object(
        payload["offline_contract"],
        label="offline_contract",
        keys=(
            "cuda_visible_devices",
            "hf_hub_offline",
            "network_downloads",
            "network_guard",
            "pretrained_weights_loaded",
            "tokenizer_loaded",
            "transformers_offline",
        ),
    )
    expected_offline = {
        "cuda_visible_devices": "",
        "hf_hub_offline": "1",
        "network_downloads": False,
        "network_guard": "socket connect/create_connection denied",
        "pretrained_weights_loaded": False,
        "tokenizer_loaded": False,
        "transformers_offline": "1",
    }
    if offline != expected_offline:
        raise SemanticsGateError("offline contract mismatch")

    device = _require_exact_object(
        payload["device_contract"],
        label="device_contract",
        keys=(
            "cache_devices",
            "cuda_available",
            "cuda_initialized_after",
            "cuda_initialized_before",
            "default_device",
            "dispatch_devices_seen",
            "dispatch_operations_audited",
            "model_parameter_devices",
            "mps_available",
            "mps_built",
            "mps_tensor_operations_observed",
            "non_cpu_tensor_observations",
            "requested_device",
        ),
    )
    required_device_contract: dict[str, object] = {
        "requested_device": "cpu",
        "default_device": "cpu",
        "model_parameter_devices": ["cpu"],
        "cache_devices": ["cpu"],
        "dispatch_devices_seen": ["cpu"],
        "cuda_initialized_before": False,
        "cuda_initialized_after": False,
        "mps_tensor_operations_observed": 0,
        "non_cpu_tensor_observations": 0,
    }
    for field, expected in required_device_contract.items():
        if device[field] != expected:
            raise SemanticsGateError(f"device contract mismatch: {field}")
    _require_bool(device["cuda_available"], label="device_contract.cuda_available")
    _require_bool(device["mps_built"], label="device_contract.mps_built")
    _require_bool(device["mps_available"], label="device_contract.mps_available")
    audited_operations = _require_integer(
        device["dispatch_operations_audited"],
        label="device_contract.dispatch_operations_audited",
        minimum=1_000,
    )
    if audited_operations != EXPECTED_DISPATCH_OPERATIONS:
        raise SemanticsGateError("device_contract.dispatch_operations_audited mismatch")

    runtime = _require_exact_object(
        payload["runtime"],
        label="runtime",
        keys=(
            "machine",
            "platform",
            "python",
            "python_implementation",
            "torch",
            "transformers",
        ),
    )
    for field in ("machine", "platform", "python"):
        _require_nonempty_string(runtime[field], label=f"runtime.{field}")
    if runtime["python_implementation"] != "CPython":
        raise SemanticsGateError("runtime.python_implementation mismatch")
    if runtime["torch"] != EXPECTED_TORCH_VERSION:
        raise SemanticsGateError("runtime.torch version mismatch")
    if runtime["transformers"] != EXPECTED_TRANSFORMERS_VERSION:
        raise SemanticsGateError("runtime.transformers version mismatch")
    python_parts = str(runtime["python"]).split(".")
    if len(python_parts) != 3 or any(not part.isdigit() for part in python_parts):
        raise SemanticsGateError("runtime.python must be a three-part version")

    determinism = _require_exact_object(
        payload["determinism"],
        label="determinism",
        keys=("deterministic_algorithms", "model_state_sha256", "torch_threads"),
    )
    if (
        _require_bool(
            determinism["deterministic_algorithms"],
            label="determinism.deterministic_algorithms",
        )
        is not True
    ):
        raise SemanticsGateError("deterministic algorithms were not enabled")
    if determinism["torch_threads"] != 1:
        raise SemanticsGateError("determinism.torch_threads must equal one")
    _require_sha256(
        determinism["model_state_sha256"],
        label="determinism.model_state_sha256",
    )

    model = _require_exact_object(
        payload["model"],
        label="model",
        keys=("architecture", "config", "dtype", "parameter_count", "source"),
    )
    expected_model = {
        "architecture": "Qwen2ForCausalLM",
        "dtype": "torch.float32",
        "parameter_count": 24_928,
        "source": "random initialization from local Qwen2Config",
    }
    for field, expected in expected_model.items():
        if model[field] != expected:
            raise SemanticsGateError(f"model invariant mismatch: {field}")
    config = _require_exact_object(
        model["config"],
        label="model.config",
        keys=(
            "hidden_size",
            "intermediate_size",
            "max_position_embeddings",
            "num_attention_heads",
            "num_hidden_layers",
            "num_key_value_heads",
            "vocab_size",
        ),
    )
    if config != {
        "hidden_size": 32,
        "intermediate_size": 64,
        "max_position_embeddings": 64,
        "num_attention_heads": 4,
        "num_hidden_layers": 2,
        "num_key_value_heads": 2,
        "vocab_size": 97,
    }:
        raise SemanticsGateError("model.config mismatch")

    requests = _require_exact_list(payload["requests"], label="requests", length=3)
    prefill_hashes: set[str] = set()
    for index, raw_request in enumerate(requests):
        request = _require_exact_object(
            raw_request,
            label=f"requests[{index}]",
            keys=(
                "first_greedy_token",
                "prefill_cache_sha256",
                "prompt_length",
                "prompt_token_ids",
                "reinserted_cache_sha256",
                "request_id",
                "second_greedy_token",
                "third_greedy_token_after_reinsert",
            ),
        )
        if request["request_id"] != REQUEST_IDS[index]:
            raise SemanticsGateError(f"requests[{index}] identity mismatch")
        if request["prompt_token_ids"] != list(PROMPTS[index]):
            raise SemanticsGateError(f"requests[{index}] prompt mismatch")
        if request["prompt_length"] != len(PROMPTS[index]):
            raise SemanticsGateError(f"requests[{index}] length mismatch")
        for token_field in (
            "first_greedy_token",
            "second_greedy_token",
            "third_greedy_token_after_reinsert",
        ):
            _require_integer(
                request[token_field],
                label=f"requests[{index}].{token_field}",
                minimum=0,
                maximum=96,
            )
        prefill_hashes.add(
            _require_sha256(
                request["prefill_cache_sha256"],
                label=f"requests[{index}].prefill_cache_sha256",
            )
        )
        _require_sha256(
            request["reinserted_cache_sha256"],
            label=f"requests[{index}].reinserted_cache_sha256",
        )
    if len(prefill_hashes) != len(REQUEST_IDS):
        raise SemanticsGateError("per-request prefill cache hashes are not distinct")

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
    if equivalence["absolute_tolerance"] != ABSOLUTE_TOLERANCE:
        raise SemanticsGateError("equivalence.absolute_tolerance mismatch")
    if equivalence["relative_tolerance"] != RELATIVE_TOLERANCE:
        raise SemanticsGateError("equivalence.relative_tolerance mismatch")
    if equivalence["greedy_tokens_exact"] is not True:
        raise SemanticsGateError("evidence lost greedy-token equivalence")
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
            "mixed_prefill_shapes",
            "parked_cache_byte_identical",
            "parked_cache_sha256_after_peer_advance",
            "parked_cache_sha256_before_peer_advance",
            "parked_request",
            "per_request_cache_content_equivalent",
            "per_request_prefill_fingerprints_distinct",
            "reinsert_order",
            "reinserted_shapes",
        ),
    )
    if ownership["parked_request"] != "request-b":
        raise SemanticsGateError("cache_ownership.parked_request mismatch")
    if ownership["reinsert_order"] != ["request-c", "request-b", "request-a"]:
        raise SemanticsGateError("cache_ownership.reinsert_order mismatch")
    for field in (
        "parked_cache_byte_identical",
        "per_request_cache_content_equivalent",
        "per_request_prefill_fingerprints_distinct",
    ):
        if _require_bool(ownership[field], label=f"cache_ownership.{field}") is not True:
            raise SemanticsGateError(f"cache ownership invariant failed: {field}")
    parked_before = _require_sha256(
        ownership["parked_cache_sha256_before_peer_advance"],
        label="cache_ownership.parked_cache_sha256_before_peer_advance",
    )
    parked_after = _require_sha256(
        ownership["parked_cache_sha256_after_peer_advance"],
        label="cache_ownership.parked_cache_sha256_after_peer_advance",
    )
    if parked_before != parked_after:
        raise SemanticsGateError("parked cache hash changed while peers advanced")
    _validate_cache_shapes(
        ownership["mixed_prefill_shapes"],
        label="cache_ownership.mixed_prefill_shapes",
        sequence_length=5,
    )
    _validate_cache_shapes(
        ownership["reinserted_shapes"],
        label="cache_ownership.reinserted_shapes",
        sequence_length=7,
    )

    negative_controls = _require_exact_object(
        payload["negative_controls"],
        label="negative_controls",
        keys=NEGATIVE_CONTROL_KEYS,
    )
    for control_name in NEGATIVE_CONTROL_KEYS:
        result = _require_exact_object(
            negative_controls[control_name],
            label=f"negative_controls.{control_name}",
            keys=("detected", "max_abs_logit_delta"),
        )
        if (
            _require_bool(
                result["detected"],
                label=f"negative_controls.{control_name}.detected",
            )
            is not True
        ):
            raise SemanticsGateError(f"negative control not detected: {control_name}")
        delta = _require_finite_number(
            result["max_abs_logit_delta"],
            label=f"negative_controls.{control_name}.max_abs_logit_delta",
            minimum=0.0,
        )
        if delta <= ABSOLUTE_TOLERANCE:
            raise SemanticsGateError(f"negative control delta is too small: {control_name}")

    seams = _require_exact_list(
        payload["remaining_gpu_backend_seams"],
        label="remaining_gpu_backend_seams",
        length=len(REMAINING_GPU_BACKEND_SEAMS),
    )
    if seams != list(REMAINING_GPU_BACKEND_SEAMS):
        raise SemanticsGateError("remaining GPU/backend seam declaration mismatch")

    frozen_results = dict(payload)
    frozen_results.pop("implementation")
    if _sha256_document(frozen_results) != EXPECTED_FROZEN_RESULTS_SHA256:
        raise SemanticsGateError("frozen semantic result map mismatch")
    return supplied


def run_gate(output: Path) -> tuple[str, float]:
    """Run the gate, write canonical evidence, verify it, and return hash/runtime."""

    started = time.monotonic()
    _load_heavy_stack()
    previous_default_device = torch.get_default_device()
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    previous_threads = torch.get_num_threads()
    try:
        payload = _run_semantics()
    finally:
        torch.set_default_device(previous_default_device)
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
        default=Path("experiments/results/cpu_transformer_semantics/evidence.json"),
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
