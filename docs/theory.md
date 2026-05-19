# Analytical results and finite-domain certificates

This note states exactly what FissionSpec's laptop-checkable theory establishes.
It does not infer accelerator speedups from a slot-count model. The executable
identities live in `fissionspec.theory` and use rational arithmetic.

## 1. Correlated heterogeneous miss externality

Consider a fixed cohort of `B` rows. In joint outcome atom `omega`, let
`M(omega)` be the missing rows and let `R_i(omega) >= 0` be row `i`'s recovery
time. Hits have no recovery time. Assume recoveries start together.

A barrier releases at the slowest missing recovery:

```text
C_barrier(omega) = B max_{i in M(omega)} R_i(omega).
```

Perfect row isolation releases every hit immediately and each miss when its own
recovery completes:

```text
C_fission(omega) = sum_{i in M(omega)} R_i(omega).
```

Therefore the expected externality is

```text
E[C_barrier - C_fission]
  = E[ |hits| max R ]
  + E[ sum_{i in M} (max R - R_i) ].
```

The first term is collateral hit delay. The second is delay that slow misses
impose on faster misses. This identity needs neither independent outcomes nor
deterministic recovery. It reduces to
`B R (1 - p^B) - B R (1 - p)` for iid hit probability `p` and fixed recovery
`R`.

Correlation matters. With fixed per-row marginal hit rates, positive
correlation can concentrate misses into all-miss atoms and reduce collateral
hit delay; anti-correlation can spread “at least one miss” across more batches.
The joint scenario representation is therefore the correct CPU oracle for
trace-derived outcome classes.

## 2. Padding versus fission

Use the explicit one-step objective

```text
J = target service time + sum_i w_i request_delay_i.
```

Let `Delta L_target` be the *measured* incremental target service time of
keeping recovering rows in the active graph bucket, and let `Delta d_i` be the
delay that a padded one-token bypass saves for recovering row `i`. Then

```text
J_padding - J_fission = Delta L_target - sum_i w_i Delta d_i.
```

Fission is selected when the right side is positive, padding when it is
negative, and either action is equivalent at zero. This condition is deliberately
profile-agnostic: CUDA graph bucketing and masked-kernel behavior determine
`Delta L_target`.

For a diagnostic linear model with `m` recovering rows, width `k`, per-masked-
slot cost `c`, and per-row overhead `a`,

```text
Delta L_target = m [a + (k - 1)c].
```

That specialization is useful for phase diagrams, but it is not evidence that
real masked slots cost `c`.

## 3. Bounded-wait liveness

For a finite closed cohort of `N` rows, capacity `C`, recovery bounded by
`R_max`, nonempty-queue coalescing bounded by `W`, and every non-preemptive
target launch bounded by `L_max`, all rows complete one target service by

```text
R_max + W + ceil(N / C) L_max
```

after becoming recovery-eligible. The proof is direct: all rows are ready by
`R_max`; the first launch occurs within `W`; at most `ceil(N/C)` launches drain
the cohort. This result does not claim starvation freedom under an unbounded
adversarial arrival stream. Production scheduling needs an age/fairness rule
for that stronger property.

The horizon-2 policy also refuses a wait that would make any forecasted
completion exceed its rolling deadline. That is local forecast safety, not a
global admission guarantee when the workload itself is infeasible.

## 4. Controller complexity

For `n` current and `m` next-ready rows, the Python controller constructs
`n + m` forecast records, performs one EDF merge sort, and scans target chunks.
Its time and auxiliary-space bounds are

```text
time:  O((n + m) log(n + m))
space: O(n + m).
```

It evaluates the latency profile exactly

```text
1 + ceil(m / C) + ceil((n + m) / C)
```

times in the full comparison: one current launch, all future-only chunks, and
all wait-and-merge chunks. Early exits use fewer calls.

## 5. What remains empirical

The horizon-2 controller has no asserted universal approximation ratio. Its
gap depends on arrivals hidden beyond the next internal readiness event and on
nonlinear latency surfaces. The generalized bounded oracle and adversarial
CPU study report finite-trace gaps instead of claiming an unsupported bound.
GPU work is needed only to replace symbolic latency deltas with calibrated
physical measurements and to verify that removing a logical row removes real
kernel work.
