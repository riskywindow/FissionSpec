# Executable experiment design

`fissionspec.experiment_design` makes the adaptive parts of the accelerator
protocol executable before any accelerator result exists. The implementation
is intentionally dependency-free and shared by CPU dry runs and later GPU
orchestration.

## Paired effect

For non-negative candidate and baseline measurements, the registered effect is

```text
D = orientation * (candidate - baseline) / max(candidate, baseline)
```

with equal zeros mapped to zero. `orientation` is positive for
higher-is-better metrics and negative for lower-is-better metrics. Therefore
`D` is always in `[-1, 1]`, including cells with a zero denominator under an
ordinary ratio.

## Execution order

Independent trace/seed clusters alternate between `ABBA` and `BAAB`, where `A`
is the candidate and `B` is the matched baseline. The sequence is a pure
function of the zero-based block index. All four runs in a completed block are
retained.

## Sequential boundary ordering

The default confirmatory cell has looks at 10, 15, ..., 50 completed paired
blocks. At each look an alpha-spending Hoeffding confidence sequence is formed
over the predeclared range `[-1, 1]`. Terminal decisions are checked in this
order:

1. the interval is wholly positive and wholly below the minimum worthwhile
   improvement (MWI): report `positive_below_minimum_worthwhile_improvement`;
2. the upper bound is below the MWI: report `futility`;
3. the lower bound is above zero: report `efficacy`;
4. the half-width reaches the precision target: report
   `precise_inconclusive`;
5. the maximum look is reached: report `maximum_reached`;
6. otherwise continue.

The first case is separated because the raw efficacy and futility inequalities
overlap for an interval contained in `(0, MWI)`. It is evidence of a positive
but practically insufficient effect, not permission for a positive headline.
No decision is emitted between scheduled looks.

Stopping is applied to a complete multiplicity family by the experiment
orchestrator. `evaluate_sequential_gate` evaluates one already-paired cell and
does not silently remove failed or incomplete pairs.

`ExperimentSpendCaps` rejects a generated run manifest above the registered
hard limits: 1,200 primary replays, 300 unique ten-seed ablation replays, and
twelve optional robustness cells. The first primary family look occurs after
240 replays. All four primary metrics are derived from each replay and do not
multiply that count.

## Calibration refinement

For each verification width, Stage 1 initially measures rows 1, 8, and 32. The
middle point is withheld from linear interpolation between the endpoints. If
its relative prediction error exceeds 3%, rows 2, 4, and 16 are added. The
criterion is determined by calibration measurements only; policy results
cannot trigger extra sampling.

## Controller-cell selection

Candidate phase-diagram cells are divided into registered regions such as
immediate fission, break-even boundary, and re-fusion. Within each region,
selection uses deterministic farthest-point coverage in globally normalized
parameter space with lexicographic tie-breaking. Reordering the input does not
change the selected cell IDs.

The frozen accelerator protocol is in
`paper/gpu_preregistration.md`; inference primitives are documented and tested
in `src/fissionspec/statistics.py`.
