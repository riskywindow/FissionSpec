# Pre-GPU sequential inference audit

This document freezes protocol version 2 before any accelerator observation.
It explains why version 1 could not satisfy its own stopping target, what now
drives a stop, and where the new procedure can still fail.

## Version-1 feasibility failure

Version 1 applied an endpointwise 95% Hoeffding confidence sequence to an
improvement bounded in `[-1, 1]`. At observation \(t\), it spent

```text
alpha_t = 0.05 / (t(t + 1)).
```

At the hard maximum \(t=50\), its untruncated radius was

```text
2 * sqrt(log(2 / (0.05 / (50 * 51))) / (2 * 50))
= 0.6791974114.
```

The registered precision target was 0.03. It was therefore unreachable by
more than a factor of 22 even before correcting across the 24 primary
endpoints. A constant effect could sometimes cross an efficacy boundary
because clipping at `[-1, 1]` makes an interval asymmetric, but that is not a
general precision design. In addition, the code's 95% default did not implement
the family declared in the pre-registration.

`sequential_gate_feasibility()` now computes this contradiction directly. A
test will fail if the frozen family, looks, or numerical consequences drift.

## Version-2 registered family

The family ID is `gpu-primary-family-v2`. Its 24 ordered endpoint IDs are the
Cartesian product of:

- two frozen model pairs;
- three frozen validation anchors; and
- four primary metrics.

There are nine looks at 10, 15, ..., 50 completed independent paired blocks.
The familywise alpha is 0.05. Every two-sided endpoint/look interval therefore
receives

```text
alpha_interval = 0.05 / (24 * 9) = 0.000231481481...
confidence      = 99.9768518519%
```

This is a conservative fixed-look group-sequential construction: Bonferroni's
union bound covers arbitrary dependence among metrics and repeated looks. It
does not require an independence fiction across endpoints.

The primary interval at a look is

```text
mean(D) +/- t_(1 - alpha_interval/2, n - 1) * sd(D) / sqrt(n),
```

intersected with the known range `[-1, 1]`. The critical values decrease from
`5.8915605265` at 10 blocks to `3.9745094042` at 50 blocks. Student's reference
law and fixed-look group-sequential testing are classical; the implementation
uses the incomplete-beta representation of the Student-\(t\) CDF and
dependency-free deterministic bisection. The endpoint/look allocation is a
deliberately simpler and more conservative construction than a correlated
Pocock or Lan-DeMets boundary.

### Assumptions that buy efficiency

The primary interval is exact only if:

1. complete paired blocks are independent and identically distributed;
2. the Studentized paired-block mean follows its Student-\(t\) reference law;
3. block removal is based only on predeclared error codes, never metric values;
4. all 24 IDs and nine looks were fixed before accelerator observations; and
5. the paired block, rather than a request or token, is the experimental unit.

The bounded symmetric improvement cannot be exactly non-degenerate Gaussian,
so condition 2 is a working pivot model, not a distribution-free theorem.
Request aggregation may make it reasonable, but no normality test can certify
an extreme tail after seeing the data. This limitation is part of the claim
boundary.

## Decision ordering

For each endpoint, the executable order is:

1. `positive_below_minimum_worthwhile_improvement` when `L > 0` and `U < 0.03`;
2. `futility` when `U < 0.03`;
3. `efficacy` when `L > 0.03`;
4. `precise_inconclusive` when the maximum distance from the mean to an
   interval endpoint is at most 0.03;
5. `maximum_reached` at 50 blocks;
6. otherwise `continue`.

Version 1 used `L > 0` for efficacy even though the MWI was 0.03. Version 2
requires `L > 0.03`; practical and statistical success now agree.

The campaign headline is conjunctive across the registered family. At a
synchronized look:

- any endpoint with `U < 0.03` ends the family in futility;
- all 24 endpoints with `L > 0.03` end it in efficacy;
- otherwise collection continues to another look or the hard maximum.

This is intentionally demanding. It prevents a favorable endpoint from
stopping the family selectively and lets a decisive negative result avoid
paying for blocks that cannot rescue the registered global claim. All
endpoint intervals and negative outcomes at the stopping look remain reported.

Runtime metadata must reproduce the exact protocol version, family ID, ordered
endpoint IDs, and synchronized block count. Missing, extra, reordered, or
asynchronously accumulated endpoints fail closed.

## Feasible precision and efficacy

For a 0.03 half-width, the largest observed standard deviation compatible with
the primary interval is:

| blocks | critical value | maximum SD |
|---:|---:|---:|
| 10 | 5.89156 | 0.01610 |
| 20 | 4.52535 | 0.02965 |
| 30 | 4.20058 | 0.03912 |
| 40 | 4.05610 | 0.04678 |
| 50 | 3.97451 | 0.05337 |

At 50 blocks, the mean required for worthwhile efficacy is approximately
`0.03 + 3.97451 * SD / sqrt(50)`. It is 0.0581 at SD 0.05 and 0.0862 at SD
0.10. Thus low-variance worthwhile effects can stop early, while ordinary
high-variance effects honestly reach the hard cap.

## Distribution-free sensitivity

Every look also reports a Hoeffding interval with the same 24-by-9 Bonferroni
allocation. Its radius at 50 blocks is `0.6021347976`, so it cannot drive the
0.03 precision or efficacy gate. It answers a narrower question: whether the
primary conclusion survives using only independent blocks and the known
`[-1, 1]` bound.

The sensitivity interval is never substituted silently for the primary
interval and never used to claim that the efficient rule is
distribution-free.

## Deterministic Monte Carlo audit

`sequential_gate_monte_carlo()` uses the repository's counter-addressed RNG.
It covers:

- a normal null;
- a small normal effect;
- a worthwhile normal effect;
- worthwhile low-variance and high-variance effects;
- all 24 normal-null endpoints over all nine looks; and
- a bounded adversarial null with probability 0.05 at `1` and probability
  0.95 at `-1/19`.

With 2,000 frozen diagnostic trials under the default seed, the normal-null
familywise noncoverage rate was `0.0300` and the one-sided false-positive rate
was `0.0140`, below the analytical 0.05 allocation. The low-variance
worthwhile scenario stopped for efficacy in `99.05%` of trials and used about
10 blocks on average. The high-variance worthwhile scenario reached 50 blocks
in `99.35%` of trials.

The adversarial bounded null is an intentional falsification case. Its
Student-\(t\) all-look coverage was only `39.70%`: when the rare value is
absent, zero sample variance produces a confidently wrong interval. The
Hoeffding sensitivity coverage was `100%` in these trials. This does not
invalidate the disclosed working-model analysis; it demonstrates why both
the assumption and sensitivity output are mandatory.

Monte Carlo frequencies are diagnostics, not a proof and not accelerator
evidence. The analytical guarantee comes from the finite endpoint/look union
bound under the primary pivot model, or from Hoeffding under the sensitivity
assumptions.

## CPU bootstrap resolution

The CPU completion study's 18 validation headline intervals have simultaneous
confidence `1 - 0.05/18 = 99.7222%`. Version 1 used only 2,000 percentile
bootstrap resamples, leaving about 2.8 draws in each tail. Full mode now freezes
20,000 resamples, leaving about 27.8 draws per tail. This is a tenfold
improvement and deterministic, but it does not make an extreme percentile
bootstrap exact. The resample count, RNG provenance, and resample fingerprint
remain in every interval record.

## Primary statistical sources

- Student (1908), [The Probable Error of a Mean](https://doi.org/10.1093/biomet/6.1.1).
- Pocock (1977), [Group Sequential Methods in the Design and Analysis of Clinical Trials](https://doi.org/10.1093/biomet/64.2.191).
- Lan and DeMets (1983), [Discrete Sequential Boundaries for Clinical Trials](https://doi.org/10.1093/biomet/70.3.659).
- Hoeffding (1963), [Probability Inequalities for Sums of Bounded Random Variables](https://doi.org/10.1080/01621459.1963.10500830).
- Dunn (1961), [Multiple Comparisons among Means](https://doi.org/10.1080/01621459.1961.10482090).
- Howard et al. (2021), [Time-uniform, nonparametric, nonasymptotic confidence sequences](https://doi.org/10.1214/20-AOS1991).
