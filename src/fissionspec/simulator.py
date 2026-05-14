"""Deterministic discrete-event simulator for batched speculative decoding.

The target verifier and draft/recovery engine have independent clocks.  The
event queue is deterministic, but random outcomes are *not* consumed from it:
every acceptance draw is addressed by ``(request_id, round_id, stream)``.
Consequently, changing a batching policy does not perturb another request's
counterfactual outcome trace.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol, cast

from .model import (
    DraftLaunchRecord,
    Outcome,
    RequestPhase,
    RequestResult,
    RequestState,
    SimulationResult,
    TargetLaunchRecord,
)
from .policies import DispatchContext, SchedulingPolicy
from .profiles import HardwareProfile
from .workload import Workload


class ScheduleIndependentRNG(Protocol):
    """Local protocol for a stateless, counter-addressed random source.

    ``draw`` defaults to zero so both the minimal three-key interface and the
    repository's more general ``CounterRNG`` can be injected.
    """

    def uniform(
        self, request_id: str, round_id: int, stream: str, draw: int = 0
    ) -> float:
        """Return a deterministic value in the half-open interval ``[0, 1)``."""


class _ThreeKeyRNG(Protocol):
    def uniform(self, request_id: str, round_id: int, stream: str) -> float: ...


class SimulationError(RuntimeError):
    """Raised when an invalid event trace would prevent forward progress."""


class _EventKind(StrEnum):
    ARRIVAL = "arrival"
    DRAFT_READY = "draft_ready"
    PRECOMPUTE_READY = "precompute_ready"
    TARGET_COMPLETE = "target_complete"
    DISPATCH_WAKE = "dispatch_wake"


@dataclass(order=True, slots=True)
class _Event:
    time_ms: float
    sequence: int
    kind: _EventKind = field(compare=False)
    payload: object = field(compare=False)


@dataclass(frozen=True, slots=True)
class _DraftReadyPayload:
    requests: tuple[tuple[str, int, int], ...]
    recovery: bool


@dataclass(frozen=True, slots=True)
class _PrecomputeReadyPayload:
    requests: tuple[tuple[str, int], ...]


@dataclass(frozen=True, slots=True)
class _TargetPayload:
    launch_id: int
    start_ms: float
    end_ms: float
    request_ids: tuple[str, ...]
    padded_request_ids: tuple[str, ...]
    verifier_slots: int
    padded_verifier_slots: int
    no_padding_latency_ms: float
    request_rounds: tuple[tuple[str, int], ...]
    precompute_end_ms: float | None


class Simulator:
    """Run one policy on an immutable workload and hardware profile.

    A hit emits a full speculative block; a miss emits its one target token.
    Every real verifier launch concurrently queues next-round outcome-cache
    construction on the draft engine.  Hits consume that prepared branch at
    ``max(target_end, precompute_end)``; misses invalidate it and execute the
    recovery curve. Policy semantics determine whether those outcomes remain
    coupled on the target timeline.
    """

    _EPSILON = 1e-12

    def __init__(
        self,
        *,
        workload: Workload,
        profile: HardwareProfile,
        policy: SchedulingPolicy,
        rng: ScheduleIndependentRNG,
        max_batch_size: int = 16,
        max_events: int = 1_000_000,
    ) -> None:
        if (
            isinstance(max_batch_size, bool)
            or not isinstance(max_batch_size, int)
            or max_batch_size <= 0
        ):
            raise ValueError("max_batch_size must be a positive integer")
        if (
            isinstance(max_events, bool)
            or not isinstance(max_events, int)
            or max_events <= 0
        ):
            raise ValueError("max_events must be a positive integer")
        if not isinstance(policy, SchedulingPolicy):
            raise TypeError("policy does not satisfy SchedulingPolicy")

        self.workload = workload
        self.profile = profile
        self.policy = policy
        self.rng = rng
        self.max_batch_size = max_batch_size
        self.max_events = max_events

        self._configs = {request.request_id: request for request in workload}
        self._states = {
            request.request_id: RequestState(
                request_id=request.request_id,
                arrival_ms=request.arrival_ms,
                output_tokens=request.output_tokens,
                speculation_length=request.speculation_length,
                tbt_slo_ms=request.tbt_slo_ms,
                absolute_deadline_ms=request.absolute_deadline_ms,
            )
            for request in workload
        }
        self._events: list[_Event] = []
        self._sequence = 0
        self._wake_times: set[float] = set()
        self._now_ms = 0.0
        self._target_busy = False
        self._draft_free_ms = 0.0
        self._active_padding: dict[str, tuple[float, int]] = {}
        self._target_records: list[TargetLaunchRecord] = []
        self._draft_records: list[DraftLaunchRecord] = []
        self._target_launch_id = 0
        self._draft_launch_id = 0
        self._has_run = False

        for request in workload:
            self._push(request.arrival_ms, _EventKind.ARRIVAL, request.request_id)

    def _push(self, time_ms: float, kind: _EventKind, payload: object) -> None:
        if not math.isfinite(time_ms) or time_ms < self._now_ms - self._EPSILON:
            raise SimulationError(f"cannot schedule {kind.value} in the past")
        self._sequence += 1
        heapq.heappush(self._events, _Event(time_ms, self._sequence, kind, payload))

    def _uniform(self, request_id: str, round_id: int) -> float:
        """Call the four-key API, retaining compatibility with minimal fakes."""

        try:
            value = self.rng.uniform(request_id, round_id, "acceptance", 0)
        except TypeError:
            value = cast(_ThreeKeyRNG, self.rng).uniform(
                request_id, round_id, "acceptance"
            )
        if not math.isfinite(value) or not 0.0 <= value < 1.0:
            raise SimulationError("rng.uniform must return a finite value in [0, 1)")
        return value

    def _handle_arrival(self, request_id: str) -> None:
        state = self._states[request_id]
        if state.phase is not RequestPhase.NOT_ARRIVED:
            raise SimulationError(f"duplicate arrival for {request_id}")
        state.phase = RequestPhase.WAIT_TARGET
        state.ready_since_ms = self._now_ms

    def _handle_draft_ready(self, payload: _DraftReadyPayload) -> None:
        for request_id, epoch, logical_version in payload.requests:
            state = self._states[request_id]
            if epoch != state.recovery_epoch:
                continue
            active = self._active_padding.get(request_id)
            if active is not None and active[1] == epoch:
                self._active_padding.pop(request_id, None)
            if state.phase is RequestPhase.COMPLETE:
                continue
            if state.phase is not RequestPhase.WAIT_DRAFT:
                raise SimulationError(
                    f"draft completion for {request_id} in phase {state.phase.value}"
                )
            if state.logical_version != logical_version:
                # A SPECTRE target-only row advanced after this job captured
                # its prefix. Reject the stale result and repair the new
                # version. Padding is disabled for the reissue so target work
                # cannot invalidate recovery forever.
                self._schedule_draft(
                    (request_id,),
                    recovery=payload.recovery,
                    pad_eligible=False,
                )
                continue
            state.phase = RequestPhase.WAIT_TARGET
            state.ready_since_ms = self._now_ms
            state.spectre_padding_eligible = False

    def _handle_precompute_ready(self, payload: _PrecomputeReadyPayload) -> None:
        for request_id, round_id in payload.requests:
            state = self._states[request_id]
            if state.phase is RequestPhase.COMPLETE:
                continue
            if (
                state.phase is RequestPhase.WAIT_DRAFT
                and state.waiting_precompute_round == round_id
            ):
                state.phase = RequestPhase.WAIT_TARGET
                state.ready_since_ms = self._now_ms
                state.waiting_precompute_round = None

    def _schedule_precompute(
        self, requests: tuple[tuple[str, int], ...]
    ) -> float | None:
        """Launch outcome-cache construction without changing target phases."""

        if not requests:
            return None
        start_ms = max(self._now_ms, self._draft_free_ms)
        end_ms = start_ms + self.profile.draft_latency_ms(len(requests))
        self._draft_launch_id += 1
        self._draft_records.append(
            DraftLaunchRecord(
                launch_id=self._draft_launch_id,
                start_ms=start_ms,
                end_ms=end_ms,
                request_ids=tuple(request_id for request_id, _ in requests),
                recovery=False,
                precompute=True,
            )
        )
        self._draft_free_ms = end_ms
        self._push(
            end_ms,
            _EventKind.PRECOMPUTE_READY,
            _PrecomputeReadyPayload(requests),
        )
        return end_ms

    def _schedule_draft(
        self,
        request_ids: tuple[str, ...],
        *,
        recovery: bool,
        barrier: bool = False,
        pad_eligible: bool = True,
        held_request_ids: tuple[str, ...] = (),
    ) -> float | None:
        work_ids = request_ids if barrier else tuple(
            request_id
            for request_id in request_ids
            if self._states[request_id].phase is not RequestPhase.COMPLETE
        )
        live_ids = tuple(
            request_id
            for request_id in request_ids
            if self._states[request_id].phase is not RequestPhase.COMPLETE
        )
        if not work_ids:
            return None
        held_ids = tuple(
            request_id
            for request_id in held_request_ids
            if self._states[request_id].phase is not RequestPhase.COMPLETE
        )

        start_ms = max(self._now_ms, self._draft_free_ms)
        end_ms = start_ms + self.profile.draft_latency_ms(
            len(work_ids), recovery=recovery
        )
        epochs: list[tuple[str, int, int]] = []
        for request_id in live_ids + held_ids:
            state = self._states[request_id]
            state.phase = RequestPhase.WAIT_DRAFT
            state.ready_since_ms = None
            state.waiting_precompute_round = None
            state.recovery_epoch += 1
            epochs.append((request_id, state.recovery_epoch, state.logical_version))
            if recovery and self.policy.pad_recovering_misses and not barrier:
                self._active_padding[request_id] = (end_ms, state.recovery_epoch)
                state.spectre_padding_eligible = pad_eligible

        self._draft_launch_id += 1
        self._draft_records.append(
            DraftLaunchRecord(
                launch_id=self._draft_launch_id,
                start_ms=start_ms,
                end_ms=end_ms,
                request_ids=work_ids,
                recovery=recovery,
                barrier=barrier,
                barrier_victim_ids=held_ids,
            )
        )
        self._draft_free_ms = end_ms
        self._push(
            end_ms,
            _EventKind.DRAFT_READY,
            _DraftReadyPayload(tuple(epochs), recovery=recovery),
        )
        return end_ms

    def _release_precomputed_hits(
        self,
        hits: tuple[tuple[str, int], ...],
        precompute_end_ms: float | None,
    ) -> None:
        if hits and precompute_end_ms is None:
            raise SimulationError("hit has no speculative precompute launch")
        for request_id, round_id in hits:
            state = self._states[request_id]
            if state.phase is RequestPhase.COMPLETE:
                continue
            if cast(float, precompute_end_ms) <= self._now_ms + self._EPSILON:
                state.phase = RequestPhase.WAIT_TARGET
                state.ready_since_ms = self._now_ms
                state.waiting_precompute_round = None
            else:
                state.phase = RequestPhase.WAIT_DRAFT
                state.ready_since_ms = None
                state.waiting_precompute_round = round_id

    def _complete_target(self, payload: _TargetPayload) -> None:
        if not self._target_busy:
            raise SimulationError("target completion without an in-flight launch")
        self._target_busy = False

        # SPECTRE's recovery rows make one token of target-only progress while
        # occupying a verifier-width rectangle.  They never consume RNG keys.
        pad_redraft_ids: list[str] = []
        for request_id in payload.padded_request_ids:
            state = self._states[request_id]
            if state.phase is RequestPhase.COMPLETE:
                continue
            recovery_was_ready = state.phase is RequestPhase.WAIT_TARGET
            state.emit(1, self._now_ms)
            if state.remaining_tokens == 0:
                self._active_padding.pop(request_id, None)
            elif recovery_was_ready:
                # Recovery completed during this target launch, so its draft
                # block predates the just-emitted target-only token.
                pad_redraft_ids.append(request_id)

        outcomes: list[tuple[str, Outcome]] = []
        hit_survivors: list[str] = []
        hit_survivor_rounds: list[tuple[str, int]] = []
        miss_ids: list[str] = []
        miss_survivors: list[str] = []
        launched_rounds = dict(payload.request_rounds)
        for request_id in payload.request_ids:
            state = self._states[request_id]
            if state.phase is not RequestPhase.IN_TARGET:
                raise SimulationError(
                    f"target completion for {request_id} in phase {state.phase.value}"
                )
            config = self._configs[request_id]
            round_id = state.round_id
            if launched_rounds.get(request_id) != round_id:
                raise SimulationError("target round changed while launch was in flight")
            hit = self._uniform(request_id, round_id) < config.probability_for_round(
                round_id
            )
            state.round_id += 1
            if hit:
                outcome = Outcome.HIT
                state.hits += 1
                state.emit(state.speculation_length, self._now_ms)
                if state.remaining_tokens > 0:
                    hit_survivors.append(request_id)
                    hit_survivor_rounds.append((request_id, round_id))
            else:
                outcome = Outcome.MISS
                state.misses += 1
                miss_ids.append(request_id)
                state.emit(1, self._now_ms)
                if state.remaining_tokens > 0:
                    miss_survivors.append(request_id)
            outcomes.append((request_id, outcome))

        if payload.padded_request_ids and payload.request_ids:
            actual_latency = payload.end_ms - payload.start_ms
            padding_delay = max(0.0, actual_latency - payload.no_padding_latency_ms)
            for request_id, outcome in outcomes:
                if outcome is Outcome.HIT:
                    self._states[request_id].hit_externality_ms += padding_delay

        if self.policy.barrier_on_miss and miss_ids and (
            hit_survivors or miss_survivors
        ):
            recovery_duration = self.profile.draft_latency_ms(
                len(miss_ids), recovery=True
            )
            for request_id in hit_survivors:
                self._states[request_id].hit_externality_ms += recovery_duration
            self._schedule_draft(
                tuple(miss_ids),
                recovery=True,
                barrier=True,
                held_request_ids=tuple(hit_survivors),
            )
        else:
            # SSD hits consume the branch constructed concurrently with the
            # target verification. Misses invalidate that work and recover.
            self._release_precomputed_hits(
                tuple(hit_survivor_rounds), payload.precompute_end_ms
            )
            self._schedule_draft(tuple(pad_redraft_ids), recovery=False)
            self._schedule_draft(tuple(miss_survivors), recovery=True)

        self._target_records.append(
            TargetLaunchRecord(
                launch_id=payload.launch_id,
                start_ms=payload.start_ms,
                end_ms=payload.end_ms,
                request_ids=payload.request_ids,
                padded_request_ids=payload.padded_request_ids,
                outcomes=tuple(outcomes),
                verifier_slots=payload.verifier_slots,
                padded_verifier_slots=payload.padded_verifier_slots,
            )
        )

    def _handle_event(self, event: _Event) -> None:
        if event.kind is _EventKind.ARRIVAL:
            self._handle_arrival(cast(str, event.payload))
        elif event.kind is _EventKind.DRAFT_READY:
            self._handle_draft_ready(cast(_DraftReadyPayload, event.payload))
        elif event.kind is _EventKind.PRECOMPUTE_READY:
            self._handle_precompute_ready(
                cast(_PrecomputeReadyPayload, event.payload)
            )
        elif event.kind is _EventKind.TARGET_COMPLETE:
            self._complete_target(cast(_TargetPayload, event.payload))
        elif event.kind is _EventKind.DISPATCH_WAKE:
            self._wake_times.discard(event.time_ms)
        else:  # pragma: no cover - exhaustive guard for future event kinds
            raise SimulationError(f"unknown event kind: {event.kind}")

    def _ready_states(self) -> list[RequestState]:
        states = [
            state
            for state in self._states.values()
            if state.phase is RequestPhase.WAIT_TARGET
            and state.ready_since_ms is not None
            and state.ready_since_ms <= self._now_ms + self._EPSILON
        ]
        states.sort(
            key=lambda state: (
                cast(float, state.ready_since_ms),
                state.arrival_ms,
                state.request_id,
            )
        )
        return states

    def _padding_ids(self) -> tuple[str, ...]:
        if not self.policy.pad_recovering_misses:
            return ()
        active = []
        for request_id, (ready_ms, epoch) in self._active_padding.items():
            state = self._states[request_id]
            if (
                state.phase is RequestPhase.WAIT_DRAFT
                and state.recovery_epoch == epoch
                and state.spectre_padding_eligible
                and ready_ms > self._now_ms + self._EPSILON
            ):
                active.append(request_id)
        return tuple(sorted(active))

    def _next_readiness(self) -> tuple[float | None, int]:
        next_time: float | None = None
        count = 0
        for event in self._events:
            event_count = 0
            if event.kind is _EventKind.ARRIVAL:
                request_id = cast(str, event.payload)
                if self._states[request_id].phase is RequestPhase.NOT_ARRIVED:
                    event_count = 1
            elif event.kind is _EventKind.DRAFT_READY:
                payload = cast(_DraftReadyPayload, event.payload)
                event_count = sum(
                    1
                    for request_id, epoch, _ in payload.requests
                    if self._states[request_id].phase is RequestPhase.WAIT_DRAFT
                    and self._states[request_id].recovery_epoch == epoch
                )
            elif event.kind is _EventKind.PRECOMPUTE_READY:
                precompute = cast(_PrecomputeReadyPayload, event.payload)
                event_count = sum(
                    1
                    for request_id, round_id in precompute.requests
                    if self._states[request_id].phase is RequestPhase.WAIT_DRAFT
                    and self._states[request_id].waiting_precompute_round == round_id
                )
            if event_count == 0:
                continue
            if next_time is None or event.time_ms < next_time - self._EPSILON:
                next_time = event.time_ms
                count = event_count
            elif abs(event.time_ms - next_time) <= self._EPSILON:
                count += event_count
        return next_time, count

    def _schedule_wake(self, time_ms: float) -> None:
        if time_ms in self._wake_times:
            return
        self._wake_times.add(time_ms)
        self._push(time_ms, _EventKind.DISPATCH_WAKE, None)

    def _launch_target(
        self, real_states: list[RequestState], padded_ids: tuple[str, ...]
    ) -> None:
        request_ids = tuple(state.request_id for state in real_states)
        real_slots = sum(state.speculation_length for state in real_states)
        padded_slots = sum(
            max(0, self._states[request_id].speculation_length - 1)
            for request_id in padded_ids
        )
        verifier_slots = real_slots + sum(
            self._states[request_id].speculation_length for request_id in padded_ids
        )
        effective_rows = len(request_ids) + len(padded_ids)
        if effective_rows <= 0:
            raise SimulationError("target launches require at least one row")

        request_rounds = tuple(
            (state.request_id, state.round_id) for state in real_states
        )
        precompute_end_ms = self._schedule_precompute(request_rounds)
        for state in real_states:
            state.phase = RequestPhase.IN_TARGET
            state.ready_since_ms = None
        for request_id in padded_ids:
            self._states[request_id].spectre_padding_eligible = False

        duration = self.profile.target_latency_ms(effective_rows, verifier_slots)
        no_padding_latency = (
            self.profile.target_latency_ms(len(request_ids), real_slots)
            if request_ids
            else 0.0
        )
        end_ms = self._now_ms + duration
        self._target_busy = True
        self._target_launch_id += 1
        payload = _TargetPayload(
            launch_id=self._target_launch_id,
            start_ms=self._now_ms,
            end_ms=end_ms,
            request_ids=request_ids,
            padded_request_ids=padded_ids,
            verifier_slots=verifier_slots,
            padded_verifier_slots=padded_slots,
            no_padding_latency_ms=no_padding_latency,
            request_rounds=request_rounds,
            precompute_end_ms=precompute_end_ms,
        )
        self._push(end_ms, _EventKind.TARGET_COMPLETE, payload)

    def _try_dispatch(self) -> None:
        if self._target_busy:
            return
        padded_ids = self._padding_ids()
        ready = self._ready_states()
        if not ready:
            if padded_ids:
                self._launch_target([], padded_ids)
            return
        capacity = self.max_batch_size - len(padded_ids)
        if capacity <= 0:
            self._launch_target([], padded_ids)
            return

        selected = ready[:capacity]
        next_ready_time, next_ready_count = self._next_readiness()
        slots_per_row = max(state.speculation_length for state in selected)
        context = DispatchContext(
            now_ms=self._now_ms,
            ready_count=len(ready),
            capacity=capacity,
            oldest_ready_ms=min(
                cast(float, state.ready_since_ms) for state in selected
            ),
            earliest_deadline_ms=min(state.absolute_deadline_ms for state in selected),
            slots_per_row=slots_per_row,
            profile=self.profile,
            next_ready_time_ms=next_ready_time,
            next_ready_count=next_ready_count,
        )
        dispatch_at = self.policy.dispatch_at(context)
        if not math.isfinite(dispatch_at) or dispatch_at < self._now_ms - self._EPSILON:
            raise SimulationError("policy returned an invalid dispatch time")
        if dispatch_at <= self._now_ms + self._EPSILON:
            self._launch_target(selected, padded_ids)
        else:
            self._schedule_wake(dispatch_at)

    def _all_complete(self) -> bool:
        return all(state.phase is RequestPhase.COMPLETE for state in self._states.values())

    def run(self) -> SimulationResult:
        """Execute the event queue exactly once and return an immutable trace."""

        if self._has_run:
            raise SimulationError("a Simulator instance can only be run once")
        self._has_run = True
        processed = 0

        while not self._all_complete():
            if processed >= self.max_events:
                raise SimulationError("max_events exceeded; policy may not make progress")
            if not self._events:
                self._try_dispatch()
                if not self._events:
                    raise SimulationError("event queue exhausted before completion")

            self._now_ms = self._events[0].time_ms
            # Drain all events at this instant, including zero-delay events
            # created by another handler, before making a batching decision.
            while (
                self._events
                and self._events[0].time_ms <= self._now_ms + self._EPSILON
            ):
                event = heapq.heappop(self._events)
                self._handle_event(event)
                processed += 1
                if processed >= self.max_events and not self._all_complete():
                    raise SimulationError(
                        "max_events exceeded; policy may not make progress"
                    )
            self._try_dispatch()

        request_results = tuple(
            RequestResult(
                request_id=state.request_id,
                arrival_ms=state.arrival_ms,
                completion_ms=cast(float, state.completion_ms),
                output_tokens=state.output_tokens,
                token_times_ms=tuple(state.token_times_ms),
                hits=state.hits,
                misses=state.misses,
                tbt_slo_ms=state.tbt_slo_ms,
                hit_externality_ms=state.hit_externality_ms,
            )
            for state in sorted(self._states.values(), key=lambda item: item.request_id)
        )
        started_ms = min(request.arrival_ms for request in request_results)
        finished_ms = max(request.completion_ms for request in request_results)
        return SimulationResult(
            policy_name=self.policy.name,
            hardware_name=self.profile.name,
            workload_name=self.workload.name,
            requests=request_results,
            target_launches=tuple(self._target_records),
            draft_launches=tuple(self._draft_records),
            started_ms=started_ms,
            finished_ms=finished_ms,
        )


DiscreteEventSimulator = Simulator


def simulate(
    workload: Workload,
    profile: HardwareProfile,
    policy: SchedulingPolicy,
    rng: ScheduleIndependentRNG,
    *,
    max_batch_size: int = 16,
    max_events: int = 1_000_000,
) -> SimulationResult:
    """Functional entry point for one deterministic policy run."""

    return Simulator(
        workload=workload,
        profile=profile,
        policy=policy,
        rng=rng,
        max_batch_size=max_batch_size,
        max_events=max_events,
    ).run()
