"""Exact offline oracle for a deliberately narrow coalescing action space.

The oracle exhaustively searches policies that, whenever the target is idle
and at least one request is ready, make one of two choices:

* dispatch the simulator's stable deadline-first selected batch immediately; or
* wait exactly until the simulator's next known readiness event.

Each search node is evaluated by replaying the immutable workload from time
zero with an action prefix.  The first reachable decision not covered by that
prefix creates two child nodes.  A completed replay is a leaf, scored by
aggregate request flow time, ``sum(completion_ms - arrival_ms)``.

The result is exact only within this action space and the simulator's
outcome-decoupled fission semantics.  It is *not* a global serving oracle: it
does not enumerate arbitrary request subsets, alternative request orderings, arbitrary
wait durations, target preemption, barriers, or padded recovery rows.

Replaying a prefix assumes ``rng`` is stateless and counter-addressed: a draw
must be a pure function of ``(request_id, round_id, stream, draw)``.  Stateful
random generators invalidate both counterfactual equality and replay-tree
consistency.  :class:`fissionspec.rng.CounterRNG` satisfies this requirement.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

from .model import SimulationResult
from .policies import DispatchContext
from .profiles import HardwareProfile
from .simulator import ScheduleIndependentRNG, simulate
from .workload import Workload


class OracleAction(StrEnum):
    """One branch in the restricted offline coalescing action space."""

    DISPATCH_NOW = "dispatch-now"
    WAIT_NEXT_READINESS = "wait-next-readiness"


class OracleLimit(StrEnum):
    """Hard resource limits accepted by :func:`offline_coalescing_oracle`."""

    DECISION_DEPTH = "max_decision_depth"
    SIMULATIONS = "max_simulations"


class OracleLimitExceeded(RuntimeError):
    """Raised instead of returning an inexact result after a search limit."""

    def __init__(
        self,
        *,
        kind: OracleLimit,
        limit: int,
        observed: int,
        action_prefix: tuple[OracleAction, ...],
    ) -> None:
        self.kind = kind
        self.limit = limit
        self.observed = observed
        self.action_prefix = action_prefix
        super().__init__(
            f"{kind.value}={limit} exceeded (required at least {observed}) "
            f"while exploring prefix {[action.value for action in action_prefix]}"
        )


class OracleReplayMismatch(RuntimeError):
    """Raised when a previously discovered action prefix is not reproducible."""


@dataclass(frozen=True, slots=True)
class OfflineOracleResult:
    """Best trace and exhaustive-search accounting.

    ``nodes`` equals ``simulations`` because this reference implementation
    reruns the simulator once for every visited prefix.  ``max_depth`` is the
    largest number of binary decisions on any visited root-to-node path.
    """

    best_result: SimulationResult
    action_prefix: tuple[OracleAction, ...]
    objective_flow_time_ms: float
    simulations: int
    leaves: int
    nodes: int
    max_depth: int


class _DecisionRequired(RuntimeError):
    """Internal control transfer at the first action absent from a prefix."""


class _PrefixPolicy:
    """Replay one adaptive branch prefix under pure fission semantics."""

    _EPSILON = 1e-12
    name = "offline-fission-oracle"
    barrier_on_miss = False
    pad_recovering_misses = False

    def __init__(self, action_prefix: tuple[OracleAction, ...]) -> None:
        self.action_prefix = action_prefix
        self.consumed_actions = 0
        self._committed_wait_until_ms: float | None = None

    @classmethod
    def _has_wait_branch(cls, context: DispatchContext) -> bool:
        next_time = context.next_ready_time_ms
        return (
            next_time is not None
            and context.next_ready_count > 0
            and math.isfinite(next_time)
            and next_time > context.now_ms + cls._EPSILON
        )

    def dispatch_at(self, context: DispatchContext) -> float:
        # Simulator events that cannot make a row ready (for example, a stale
        # precompute completion) may still trigger policy evaluation.  A wait
        # action commits through those events; it is not an invitation to
        # dispatch at an arbitrary intermediate timestamp.
        if self._committed_wait_until_ms is not None:
            if context.now_ms < self._committed_wait_until_ms - self._EPSILON:
                return self._committed_wait_until_ms
            self._committed_wait_until_ms = None

        if not self._has_wait_branch(context):
            return context.now_ms
        if self.consumed_actions == len(self.action_prefix):
            raise _DecisionRequired

        action = self.action_prefix[self.consumed_actions]
        self.consumed_actions += 1
        if action is OracleAction.DISPATCH_NOW:
            return context.now_ms
        if action is OracleAction.WAIT_NEXT_READINESS:
            next_time = context.next_ready_time_ms
            if next_time is None:  # pragma: no cover - guarded above
                raise OracleReplayMismatch("wait action has no next readiness")
            self._committed_wait_until_ms = next_time
            return next_time
        raise OracleReplayMismatch(f"unknown oracle action: {action!r}")


def aggregate_request_flow_time_ms(result: SimulationResult) -> float:
    """Return the oracle objective for a completed simulation trace."""

    return math.fsum(request.latency_ms for request in result.requests)


def _positive_integer(value: int, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _non_negative_integer(value: int, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _tie_key(actions: tuple[OracleAction, ...]) -> tuple[int, ...]:
    """Prefer dispatch-now at the first differing decision on exact ties."""

    return tuple(0 if action is OracleAction.DISPATCH_NOW else 1 for action in actions)


def offline_coalescing_oracle(
    workload: Workload,
    profile: HardwareProfile,
    rng: ScheduleIndependentRNG,
    *,
    max_decision_depth: int,
    max_simulations: int,
    max_batch_size: int = 16,
    max_events: int = 1_000_000,
) -> OfflineOracleResult:
    """Exhaustively minimize aggregate flow time over binary coalescing choices.

    Both search limits are mandatory.  Reaching either one before every leaf
    has been evaluated raises :class:`OracleLimitExceeded`; the function never
    returns a best-so-far result disguised as an exact optimum.

    Equal objective values are resolved lexicographically with
    :attr:`OracleAction.DISPATCH_NOW` ordered before waiting.
    """

    depth_limit = _non_negative_integer(max_decision_depth, field="max_decision_depth")
    simulation_limit = _positive_integer(max_simulations, field="max_simulations")

    pending: list[tuple[OracleAction, ...]] = [()]
    simulations = 0
    nodes = 0
    leaves = 0
    max_depth = 0
    best_result: SimulationResult | None = None
    best_actions: tuple[OracleAction, ...] | None = None
    best_objective = math.inf

    while pending:
        action_prefix = pending.pop()
        if simulations >= simulation_limit:
            raise OracleLimitExceeded(
                kind=OracleLimit.SIMULATIONS,
                limit=simulation_limit,
                observed=simulations + 1,
                action_prefix=action_prefix,
            )

        policy = _PrefixPolicy(action_prefix)
        simulations += 1
        nodes += 1
        max_depth = max(max_depth, len(action_prefix))
        try:
            result = simulate(
                workload,
                profile,
                policy,
                rng,
                max_batch_size=max_batch_size,
                max_events=max_events,
                reveal_future_arrivals=True,
            )
        except _DecisionRequired:
            if policy.consumed_actions != len(action_prefix):
                raise OracleReplayMismatch(
                    "decision replay diverged before consuming its action prefix; "
                    "rng must be stateless and counter-addressed"
                ) from None
            next_depth = len(action_prefix) + 1
            if next_depth > depth_limit:
                raise OracleLimitExceeded(
                    kind=OracleLimit.DECISION_DEPTH,
                    limit=depth_limit,
                    observed=next_depth,
                    action_prefix=action_prefix,
                ) from None

            # LIFO stack: append wait first so dispatch is explored first.
            pending.append((*action_prefix, OracleAction.WAIT_NEXT_READINESS))
            pending.append((*action_prefix, OracleAction.DISPATCH_NOW))
            continue

        if policy.consumed_actions != len(action_prefix):
            raise OracleReplayMismatch(
                "simulation completed before consuming its discovered action "
                "prefix; rng must be stateless and counter-addressed"
            )

        leaves += 1
        objective = aggregate_request_flow_time_ms(result)
        if (
            best_actions is None
            or objective < best_objective
            or (objective == best_objective and _tie_key(action_prefix) < _tie_key(best_actions))
        ):
            best_result = result
            best_actions = action_prefix
            best_objective = objective

    if best_result is None or best_actions is None:  # pragma: no cover
        raise OracleReplayMismatch("exhaustive search produced no leaf")
    return OfflineOracleResult(
        best_result=best_result,
        action_prefix=best_actions,
        objective_flow_time_ms=best_objective,
        simulations=simulations,
        leaves=leaves,
        nodes=nodes,
        max_depth=max_depth,
    )
