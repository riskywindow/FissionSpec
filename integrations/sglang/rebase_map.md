# SPECTRE PR-head to current-main rebase map

This map is intentionally separate from the apply-clean prototype. The patch
series targets the only public SPECTRE implementation, open PR
[#22272](https://github.com/sgl-project/sglang/pull/22272) at
`1a8520879c53462b7ac1861d3aad7de4bf5860d4`. It does **not** pretend that the
PR-head files still exist unchanged on SGLang main.

The comparison is pinned to main
`d0b9689805232d8ab37789121cbc3b766b5c723e` on 2026-07-23.

## Architectural drift

| PR-head seam | Current-main seam | Required port |
|---|---|---|
| `SchedulerOutputProcessorMixin.process_batch_result_decode` in `scheduler_output_processor_mixin.py` | `BatchResultProcessor.process_batch_result_decode` in `scheduler_components/batch_result_processor.py` | Publish per-request fission outcomes after `_normalize_decode_outputs` / `_resolve_spec_v2_tokens`, before output streaming and before the next `NextBatchPlan` is built. |
| Spec-V1 SPECTRE `SpectreWorker.verify` and `_post_verify_update_drafts` | Spec-V2 result contract (`accept_lens`, `speculative_num_draft_tokens`) and `BaseSpecWorker.on_verify_complete_cpu` | Re-express match/miss classification using the V2 per-request accepted-run/result contract. Do not revive the removed V1 `v1_spec_info_filtered` path. |
| `SchedulerSpectreTargetMixin.event_loop_normal_spectre_target` owns a separate target loop | `Scheduler.event_loop_normal` and `event_loop_overlap` share `get_next_batch_to_run(...) -> NextBatchPlan` | Make recovery lanes a scheduler component consumed by both loops. The component must transform `plan.running_batch`; it cannot rely on mutating an implicit `self.running_batch` only in the non-overlap loop. |
| `ScheduleBatch.filter_batch(..., v1_spec_info_filtered=True)` | `ScheduleBatch.filter_batch()` always owns current Spec-V2 filtering | Exclude rows with ordinary `keep_indices`; let the current `spec_info.filter_batch` contract perform the matching V2 update. |
| PR-head `GenerationBatchResult.num_accepted_tokens` / `accept_length_per_req_cpu` | current `num_correct_drafts`, `num_correct_drafts_per_req_cpu`, block/cap counters | Keep existing metrics intact and add outcome/lane metrics alongside them. Do not reinterpret “correct draft” as a recovery hit. |
| `_build_hisparse_decode_batch([req])` exists as an available descriptor reconstruction seam | the helper remains, but running-batch ownership is now returned through `NextBatchPlan` | Extract a generic `build_decode_batch_from_owned_reqs` helper or a dedicated fission reinsertion component. Do not couple the final port to `enable_hisparse`. |
| SPECTRE is a hard-coded `SpeculativeAlgorithm` member and scheduler mixin | current algorithms are registered with overlap capability metadata | Register rebased SPECTRE explicitly and declare overlap/grammar support only after those paths pass runtime tests. |
| Target wire cache is keyed by `(rid, spec_cnt)` | no SPECTRE transport exists on current main | Port `fission_version` with the remote-drafter transport; retain optional decoding for rolling compatibility, but require the full key whenever fission is enabled. |

## Current-main insertion order

The post-rebase control flow should be:

1. The worker returns the ordinary Spec-V2 result without a scheduler-side
   synchronous retry.
2. `BatchResultProcessor._resolve_spec_v2_tokens` resolves per-request accepted
   runs and updates existing speculative metrics.
3. A fission outcome classifier publishes immutable
   `(rid, spec_cnt, fission_version, outcome)` events.
4. `BatchResultProcessor.process_batch_result_decode` completes token,
   finish-state, logprob, grammar, and output-stream updates.
5. The fission scheduler component commits events. `FINISHED` is terminal;
   `HIT` stays target-ready; `RECOVERING` retains its request/KV owner but is
   omitted from `plan.running_batch`.
6. The next call to `get_next_batch_to_run` merges only version-validated
   `READY_BACKUP` or `BYPASS` owners through a returned `NextBatchPlan`.
7. The model-worker descriptor and graph bucket are built from that physically
   filtered batch, then shape telemetry records logical rows, real verifier
   slots, and graph-bucket slots.

In overlap mode, the outcome event must carry the version captured by the
launched batch copy. A completion may arrive after cancellation or a newer
round; the event can update telemetry but must not mutate the live request or
queue.

## Files that cannot be mechanically rebased

- `python/sglang/srt/speculative/spectre/**` is absent from the pinned current
  main and must be ported as a coherent algorithm/transport, not copied one
  file at a time.
- `scheduler_output_processor_mixin.py` was replaced by scheduler components.
- current-main request, result, metric, and overlap contracts use Spec V2 and
  different field names.
- `Scheduler.get_next_batch_to_run` now takes and returns explicit running
  state; PR-head mutations after `process_batch_result` do not automatically
  become the next current-main plan.

The apply-clean PR-head patch is therefore a reviewable behavior prototype and
CPU invariant oracle, not a current-main patch.

## Rebase acceptance gates

The current-main port is ready for GPU spend only after CPU tests demonstrate:

- identical behavior with fission disabled in both normal and overlap loops;
- full-key transport round trips and first-completion-wins deduplication;
- stale, duplicate, reordered, cancelled, and timed-out events cannot publish
  a descriptor;
- a recovering row is absent from both `ScheduleBatch` and
  `ModelWorkerBatch`;
- reinsertion retains target KV ownership and scheduling age;
- `NextBatchPlan.running_batch` contains each live version at most once; and
- the current `batch_result_processor.py` source hash matches the pin in
  `upstream_pin.json`.

GPU validation remains necessary for backend-specific descriptor/KV
correctness, graph capture shape, and physical verifier-work reduction.
