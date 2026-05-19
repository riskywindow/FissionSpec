"""Exact finite-domain theory helpers for outcome-decoupled scheduling.

The simulator uses floating-point time because calibrated profiles will
eventually be measured in milliseconds.  This module serves a different
purpose: it encodes the paper's small analytical models with
``fractions.Fraction`` so assumptions and edge cases can be checked without
rounding error.

None of the helpers below predicts GPU performance.  They expose identities
that hold *conditional on their stated cost models*.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from fractions import Fraction


def _fraction(value: Fraction | int, *, field: str) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, (int, Fraction)):
        raise TypeError(f"{field} must be an int or Fraction")
    return Fraction(value)


def _non_negative(value: Fraction | int, *, field: str) -> Fraction:
    result = _fraction(value, field=field)
    if result < 0:
        raise ValueError(f"{field} must be non-negative")
    return result


@dataclass(frozen=True, slots=True)
class RecoveryScenario:
    """One atom of a correlated outcome/recovery distribution.

    ``recovery_ms[i] is None`` denotes an outcome-cache hit for row ``i``.
    A non-negative rational duration denotes a miss and its realized recovery
    time.  Because the complete batch outcome is one atom, callers can encode
    arbitrary cross-row correlation and arbitrary correlation between outcome
    identity and recovery duration.
    """

    probability: Fraction
    recovery_ms: tuple[Fraction | None, ...]

    def __post_init__(self) -> None:
        probability = _non_negative(self.probability, field="probability")
        if probability > 1:
            raise ValueError("probability must be at most one")
        if not self.recovery_ms:
            raise ValueError("a scenario must contain at least one row")
        normalized: list[Fraction | None] = []
        for value in self.recovery_ms:
            normalized.append(None if value is None else _non_negative(value, field="recovery_ms"))
        object.__setattr__(self, "probability", probability)
        object.__setattr__(self, "recovery_ms", tuple(normalized))


@dataclass(frozen=True, slots=True)
class RecoveryExternality:
    """Exact expected stalled-row time under barrier and row isolation.

    Under the barrier, every live row waits for the slowest miss.  Under
    perfect fission, only a missing row waits, and it waits for its own
    recovery.  Their difference splits into delay imposed on cache hits and
    extra delay imposed by slower misses on faster misses.
    """

    rows: int
    barrier_stalled_row_ms: Fraction
    isolated_stalled_row_ms: Fraction
    collateral_hit_stall_ms: Fraction
    cross_miss_stall_ms: Fraction

    @property
    def total_externality_ms(self) -> Fraction:
        return self.barrier_stalled_row_ms - self.isolated_stalled_row_ms

    @property
    def amplification(self) -> Fraction:
        if self.isolated_stalled_row_ms == 0:
            return Fraction(1)
        return self.barrier_stalled_row_ms / self.isolated_stalled_row_ms


def expected_recovery_externality(
    scenarios: Iterable[RecoveryScenario],
) -> RecoveryExternality:
    """Evaluate a complete joint outcome/recovery distribution exactly.

    Assumptions:

    * one barrier cohort contains the same rows in every scenario;
    * all misses begin recovery together;
    * the barrier releases when the slowest recovery completes; and
    * fission lets hits continue immediately and lets each miss continue at its
      own recovery completion.
    """

    atoms = tuple(scenarios)
    if not atoms:
        raise ValueError("at least one recovery scenario is required")
    rows = len(atoms[0].recovery_ms)
    if any(len(atom.recovery_ms) != rows for atom in atoms):
        raise ValueError("all recovery scenarios must have the same row count")
    if sum((atom.probability for atom in atoms), Fraction()) != 1:
        raise ValueError("scenario probabilities must sum exactly to one")

    barrier = Fraction()
    isolated = Fraction()
    collateral = Fraction()
    cross_miss = Fraction()
    for atom in atoms:
        misses = tuple(value for value in atom.recovery_ms if value is not None)
        slowest = max(misses, default=Fraction())
        hit_count = rows - len(misses)
        atom_isolated = sum(misses, Fraction())
        atom_cross_miss = sum((slowest - duration for duration in misses), Fraction())
        barrier += atom.probability * rows * slowest
        isolated += atom.probability * atom_isolated
        collateral += atom.probability * hit_count * slowest
        cross_miss += atom.probability * atom_cross_miss

    result = RecoveryExternality(
        rows=rows,
        barrier_stalled_row_ms=barrier,
        isolated_stalled_row_ms=isolated,
        collateral_hit_stall_ms=collateral,
        cross_miss_stall_ms=cross_miss,
    )
    if result.total_externality_ms != collateral + cross_miss:
        raise AssertionError("externality decomposition failed")
    return result


@dataclass(frozen=True, slots=True)
class PaddingBreakEven:
    """One-step target-cost versus latency-credit comparison.

    The objective is target service time plus weighted request delay.  A padded
    bypass pays ``incremental_target_cost_ms`` relative to removing the
    recovering rows, but saves ``delay_saved_ms[i]`` for each bypassed row.
    Positive ``padding_minus_fission_ms`` means fission has lower objective.
    """

    incremental_target_cost_ms: Fraction
    weighted_delay_credit_ms: Fraction

    @property
    def padding_minus_fission_ms(self) -> Fraction:
        return self.incremental_target_cost_ms - self.weighted_delay_credit_ms

    @property
    def preference(self) -> str:
        gap = self.padding_minus_fission_ms
        if gap > 0:
            return "fission"
        if gap < 0:
            return "padding"
        return "indifferent"


def padding_break_even(
    *,
    incremental_target_cost_ms: Fraction | int,
    delay_saved_ms: Sequence[Fraction | int],
    delay_weights: Sequence[Fraction | int] | None = None,
) -> PaddingBreakEven:
    """Return the exact one-step break-even decision for an arbitrary profile.

    ``incremental_target_cost_ms`` should be measured from the same physical
    row/slot graph bucket used by the deployment.  The helper intentionally
    accepts that calibrated difference directly rather than assuming masked
    slots are free or linear.
    """

    cost = _non_negative(incremental_target_cost_ms, field="incremental_target_cost_ms")
    savings = tuple(_non_negative(value, field="delay_saved_ms") for value in delay_saved_ms)
    if not savings:
        raise ValueError("at least one recovering row is required")
    if delay_weights is None:
        weights = (Fraction(1),) * len(savings)
    else:
        weights = tuple(_non_negative(value, field="delay_weights") for value in delay_weights)
        if len(weights) != len(savings):
            raise ValueError("delay_weights must match delay_saved_ms")
    credit = sum(
        (weight * saved for weight, saved in zip(weights, savings, strict=True)),
        Fraction(),
    )
    return PaddingBreakEven(cost, credit)


def linear_masked_padding_cost(
    *,
    recovering_rows: int,
    verifier_width: int,
    masked_slot_cost_ms: Fraction | int,
    row_cost_ms: Fraction | int = 0,
) -> Fraction:
    """Specialize the padded-row cost for a linear physical-slot model.

    A recovering row performs one useful target position and masks
    ``verifier_width - 1`` positions.  ``row_cost_ms`` accounts for any
    per-row overhead beyond those masked positions.  This is a diagnostic
    upper-level model; a real graph-bucket delta belongs in
    :func:`padding_break_even`.
    """

    for field, value in (
        ("recovering_rows", recovering_rows),
        ("verifier_width", verifier_width),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{field} must be a positive integer")
    slot_cost = _non_negative(masked_slot_cost_ms, field="masked_slot_cost_ms")
    per_row = _non_negative(row_cost_ms, field="row_cost_ms")
    return recovering_rows * (per_row + (verifier_width - 1) * slot_cost)


@dataclass(frozen=True, slots=True)
class ClosedCohortBound:
    """Liveness certificate for a finite, closed, non-preemptive cohort."""

    rows: int
    capacity: int
    launches: int
    completion_after_ready_ms: Fraction


def closed_cohort_completion_bound(
    *,
    rows: int,
    capacity: int,
    max_coalescing_wait_ms: Fraction | int,
    max_target_launch_ms: Fraction | int,
    max_recovery_ms: Fraction | int = 0,
) -> ClosedCohortBound:
    """Bound completion after a finite cohort becomes recovery-eligible.

    Assumptions:

    * no later arrivals may overtake the closed cohort;
    * recovery is bounded by ``max_recovery_ms``;
    * a nonempty ready queue waits at most ``max_coalescing_wait_ms``;
    * target launches are non-preemptive and last at most
      ``max_target_launch_ms``; and
    * every launch admits up to ``capacity`` cohort rows.

    Under an unbounded adversarial arrival stream, deadline-free liveness would
    require an explicit fairness rule and this bound does not apply.
    """

    for field, value in (("rows", rows), ("capacity", capacity)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{field} must be a positive integer")
    wait = _non_negative(max_coalescing_wait_ms, field="max_coalescing_wait_ms")
    launch = _non_negative(max_target_launch_ms, field="max_target_launch_ms")
    recovery = _non_negative(max_recovery_ms, field="max_recovery_ms")
    launches = (rows + capacity - 1) // capacity
    return ClosedCohortBound(
        rows=rows,
        capacity=capacity,
        launches=launches,
        completion_after_ready_ms=recovery + wait + launches * launch,
    )


@dataclass(frozen=True, slots=True)
class HorizonTwoComplexity:
    """Auditable operation counts and asymptotic bounds for one decision."""

    forecast_rows: int
    latency_profile_lookups: int
    auxiliary_rows: int
    time_complexity: str = "O((n+m) log(n+m))"
    space_complexity: str = "O(n+m)"


def horizon_two_complexity(
    *, current_rows: int, future_rows: int, capacity: int
) -> HorizonTwoComplexity:
    """Return exact profile-lookup counts for the Python controller.

    The controller prices one current launch, all future-only chunks, and all
    EDF-merged wait chunks.  Sorting the merged rows dominates the CPU bound.
    """

    for field, value, allow_zero in (
        ("current_rows", current_rows, False),
        ("future_rows", future_rows, True),
        ("capacity", capacity, False),
    ):
        invalid_zero = value < 0 if allow_zero else value <= 0
        if isinstance(value, bool) or not isinstance(value, int) or invalid_zero:
            qualifier = "non-negative" if allow_zero else "positive"
            raise ValueError(f"{field} must be a {qualifier} integer")
    future_chunks = (future_rows + capacity - 1) // capacity
    merged_chunks = (current_rows + future_rows + capacity - 1) // capacity
    return HorizonTwoComplexity(
        forecast_rows=current_rows + future_rows,
        latency_profile_lookups=1 + future_chunks + merged_chunks,
        auxiliary_rows=current_rows + future_rows,
    )
