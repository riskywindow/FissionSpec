# Generalized exact bounded scheduling oracle

`fissionspec.general_oracle` is a finite, dependency-free reference solver for
the target-dispatch decision that the simulator's restricted offline oracle
does not cover. It enumerates arbitrary batch subsets, orderings, and explicit
wait points over a fully pre-realized tiny trace.

It is intentionally an exponential research oracle. Its purpose is to produce
counterexamples, optimality gaps, and replayable certificates for small
instances—not to schedule a live serving system.

## Problem

Each `OracleJob` is one immutable target row with:

- a unique string ID and optional cohort label;
- exact release time;
- positive verifier-slot width;
- exact completion deadline; and
- positive exact flow-time weight.

Jobs sharing a cohort label must have the same release time. The label records
a pre-realized readiness cohort; it does not make the cohort atomic.

`OracleCapacity` independently bounds rows and total verifier slots in a
launch. `ExactLatencySurface` is a finite table

\[
L(\text{rows},\text{slots}) \rightarrow \text{duration}
\]

whose values are positive `fractions.Fraction` instances. Construction fails
if any batch shape attainable from the supplied jobs lacks an entry. No
interpolation, floating-point comparison, or unreported fallback occurs.

The target is non-preemptive. Every selected row completes at the common end of
its launch, and each job requires exactly one launch.

## Complete bounded action space

At an idle state \((t,U)\), where \(U\) is the unfinished-job set, the solver
enumerates every ordered non-empty subset of released jobs satisfying both
capacity limits. Order within a batch does not change its duration or
completion time in this model, but it remains part of the declared tertiary
tie-break and is explicitly enumerated.

Optional waits are absolute times derived once from `OracleWaitConfig`:

1. every job release time, when enabled;
2. every individual deadline-safe time
   \(d_j-L(r,s)\) for every admissible subset containing \(j\), when enabled;
3. every explicitly supplied grid time; and
4. only points at or before `latest_optional_time`, when that bound is present.

Coincident points use the deterministic priority release, deadline-safe, then
grid. When no job is ready, advancing to the next release is forced and
recorded separately. A wait may commit through intermediate points; paths that
instead reconsider there are also enumerated.

All waits move strictly forward in a finite global point set. A path contains
at most:

\[
n\ \text{dispatches} + n\ \text{forced release advances} +
D\ \text{optional points},
\]

where \(n\) is the job count and \(D\) is the number of derived decision
points. `max_trace_events` checks this conservative \(2n+D\) bound before
search.

## Objective and deterministic tie-break

The objective is lexicographic:

1. minimize the number of jobs completed strictly after their deadlines;
2. minimize exact weighted flow
   \[
   \sum_j w_j(C_j-r_j);
   \]
3. minimize the canonical event trace, with dispatch ordered before wait and
   job ordering encoded deterministically.

The first component is intentionally not folded into a penalty coefficient.
No finite coefficient can express the stated priority for every weight and
trace. `OracleObjective` stores both components without loss.

## Exact search and safe reductions

The dynamic-programming state is exactly `(current_time, unfinished_mask)`.
This is sufficient because completed-job objective contributions are additive
and every remaining release, deadline, width, and weight is immutable.

Two reductions preserve the complete optimum:

- **Exact memoization.** Re-entering the identical time and unfinished mask
  has the identical suffix problem. Such calls increment
  `states_pruned_by_memo`.
- **Permutation dominance.** Permutations of the same selected set produce
  the same end time, unfinished mask, and incremental objective. Only the
  lexicographically smallest event can win the tertiary tie-break; the others
  increment `transitions_pruned_by_dominance`.

No cross-time dominance assumption is made. In particular, an earlier state is
not declared to dominate a later one, because the configured finite wait space
may not permit reproducing an arbitrary later timestamp.

For \(m\) ready jobs and row limit \(R\), a state considers up to

\[
\sum_{k=1}^{\min(m,R)} \frac{m!}{(m-k)!}
\ D_t
\]

actions before slot filtering, where \(D_t\) is the number of later optional
points. Reachable completion sums can create times beyond the configured
decision set, so a coarse state bound is \(2^n T\), with \(T\) the number of
reachable exact times. `OracleSearchLimits` therefore requires explicit
`max_jobs`, `max_states`, `max_transitions`, and `max_trace_events`. Hitting any
limit raises `GeneralOracleLimitExceeded`; no best-so-far result is labeled
exact.

## Certificate and independent verification

`solve_general_oracle` returns a `GeneralOracleCertificate` containing:

- the canonical SHA-256 input hash;
- the optimal dispatch/wait event trace;
- exact objective and per-job completion times;
- explored states and transitions;
- memo-pruned states and dominance-pruned orderings; and
- a certificate hash over the semantic payload.

`verify_general_oracle_certificate` first replays the events independently,
checking release eligibility, uniqueness, row and slot capacity, exact surface
duration, configured waits, completion times, objective, and input hash.

With `prove_optimality=True`, the verifier then enumerates the complete tree
again using a separate implementation with no memoization and no permutation
dominance. It compares both the objective and deterministic optimal trace.
`max_verifier_nodes` bounds this deliberately expensive proof path.

Search counters are diagnostic metadata; semantic verification depends on the
input, trace, completions, and objective.

## Cross-checks and measured tiny gaps

The one-token, unit-width common domain is cross-checked against
`offline_coalescing_oracle`. With releases \(0\) and \(1/5\), and latencies
\(L(1,1)=1\), \(L(2,2)=5/4\), both choose to coalesce and obtain exact total
flow \(27/10\).

The tests also execute the real `FissionSpecPolicy` on a two-job adversarial
trace:

- both jobs release at zero with unit width;
- deadlines are 2 and 100;
- \(L(1,1)=1\), while \(L(2,2)=10\).

The simulator's full ready-batch dispatch has objective `(1 violation, 20
flow)`. The arbitrary-subset oracle uses two singleton launches and obtains
`(0 violations, 3 flow)`, an exact component-wise gap of `(1, 17)`. This is a
bounded action-space counterexample, not a production-performance claim.

## Running the oracle tests

```bash
PYTHONPATH=src python3 -m unittest tests.test_general_oracle -v
```

The focused suite includes restricted-oracle agreement, actual FissionSpec and
EDF gaps, exact 2D capacity cases, all wait-point families, certificate
tampering, fail-closed limits, and repeated agreement with the independent
unmemoized brute-force verifier over an exhaustively enumerated smaller domain.
