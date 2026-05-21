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

Protocol version 2 registers exactly 24 endpoint IDs: four metrics at three
validation anchors for two model pairs. It has nine looks at 10, 15, ..., 50
completed paired blocks. At every look, every endpoint receives

```text
alpha_interval = 0.05 / (24 endpoints * 9 looks)
               = 0.000231481481...
```

The primary interval is the ordinary paired-mean Student-\(t\) interval at
that level. Bonferroni allocation across every endpoint and look makes the
repeated comparisons familywise under the registered Student-\(t\) pivot
model. That model assumes independent, identically distributed block effects
and a correctly calibrated Studentized mean. It is not described as
distribution-free. Every decision also emits a bounded Hoeffding interval
using the same endpoint/look allocation; this sensitivity interval never
drives the primary stop.

For one endpoint, terminal decisions are checked in this exact order:

1. the interval is wholly positive and wholly below the minimum worthwhile
   improvement (MWI): report `positive_below_minimum_worthwhile_improvement`;
2. the upper bound is below the MWI: report `futility`;
3. the lower bound is above the MWI: report `efficacy`;
4. the primary half-width reaches the precision target: report
   `precise_inconclusive`;
5. the maximum look is reached: report `maximum_reached`;
6. otherwise continue.

The stronger efficacy boundary fixes a version-1 inconsistency: a merely
positive interval no longer counts as efficacy when it does not establish the
registered 0.03 worthwhile effect. No decision is emitted between looks.

`evaluate_sequential_gate` refuses a runtime family ID, ordered endpoint list,
or endpoint ID that differs from the versioned declaration.
`evaluate_sequential_family` additionally requires all 24 endpoints at the
same completed-block count. The headline is conjunctive: any endpoint that
rules out the MWI ends the family in futility; favorable family stopping
requires every endpoint to establish the MWI. Mixed unresolved evidence runs
to another look or the hard maximum. This strict global rule is what permits a
negative result to save later replays without selecting whichever endpoint
looked favorable.

The executable feasibility diagnostic reports that the old endpointwise
Hoeffding radius is about 0.68 at 50 blocks, so its 0.03 precision target was
unreachable. The new distribution-free familywise sensitivity radius is also
far too wide. At 50 blocks, the primary interval reaches 0.03 half-width only
when the observed paired-block standard deviation is roughly 0.053 or less.
These are design facts computed before accelerator data, not retrospective
justifications. Full derivations, calibration scenarios, and residual
assumptions are in `docs/sequential_inference.md`.

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
