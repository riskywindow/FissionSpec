# Reviewed Python/Rust contract

The Python simulator and the Rust hot-path crate are complementary
implementations. They do **not** share a complete cost model. The executable
contract in `fixtures/cross_language/` freezes only the intersection described
below. Each language reads the human-authored TSV files independently and
checks the committed expected values; neither implementation generates the
other's oracle at test time.

Run both consumers from a source checkout or unpacked source distribution:

```bash
python3 tools/check_cross_language_contract.py
```

The command runs `tests.test_cross_language_contract` and the Rust integration
test `cross_language_contract`. A single-language pass is not reported as a
parity pass.

## Supported subset

### Analytical miss metrics

Canonical inputs are per-row **miss** probabilities. Rust consumes them
directly. Python maps them to hit probabilities with `hit = 1 - miss` before
calling its existing metric functions. The common outputs are:

- independent all-hit survival probability;
- independent probability of at least one miss;
- barrier head-of-line amplification, including the neutral all-hit value
  `1`;
- expected healthy rows blocked by at least one missing peer.

The corpus includes heterogeneous, zero, certain-miss, all-hit, and singleton
cases. It does not claim equality for empty batches: Rust defines neutral
empty-batch values while the Python research metric rejects an empty
probability vector.

### Monotone one-dimensional latency

Fixture time is integer nanoseconds. Rust uses it directly. Python divides the
ordinates by `1e6` to construct `LatencyCurve` milliseconds, then multiplies
the prediction back to nanoseconds.

The common domain requires:

- a first knot at one positive work unit;
- at least two knots;
- positive query work;
- strictly increasing work coordinates and nondecreasing positive latency;
- an interpolation or terminal-extrapolation result that is an integer
  nanosecond.

These restrictions matter. Below-first-knot behavior, zero work, single-knot
extrapolation, and fractional interpolation differ between the existing
implementations. Rust rounds fractional interpolation upward; Python retains a
floating value. Those cases are intentionally outside the contract.

### Flattened horizon-2 decisions

The common controller fixture is deliberately small:

- `now = 0`, with current rows arriving and becoming ready at zero;
- one known forecast cohort whose arrival equals its ready time;
- one service unit/verifier slot per row;
- normal and uniform priority;
- the same one-dimensional latency knots under the nanosecond/millisecond
  conversion above;
- no arithmetic overflow, deadline guard, recovery item, or capacity overflow;
- current plus forecast rows fit one declared capacity;
- a common maximum coalescing window and absolute deadlines.

Only the externally observable action is shared: dispatch now or wait for the
forecast. Both implementations break an equal-cost tie toward dispatch. The
fixture covers lower launch cost, lower fusion cost, exact tie, no forecast,
fusion deadline failure, and fusion as the only feasible plan.

This does not equate Python's two-dimensional `(rows, physical slots)` hardware
surface with Rust's flattened service-unit curve. It also excludes Rust
priority weights and bypass reasons, Python heterogeneous row widths and
rolling-EDF overflow, saturation behavior, and production graph buckets.

### Version-fenced transaction traces

The shared transaction abstraction is one private branch at current epoch
`1`, with actions:

- prepare at an observed epoch;
- commit the prepared branch at an observed epoch;
- abort at an observed epoch, including idempotent replay.

The contract checks action success/error classification and terminal
committed/aborted state. Python maps the fixture epoch to `LedgerEpoch.version`
and uses its real one-block branch; Rust maps it to `Epoch` and uses a
tensor-free `TransactionMeta`. Stale calls must be inert and commit is
terminal.

This subset does not compare page identities, COW tail layout, parent branch
fences, pool ownership, returned release counts, or protocol and ledger
version-number equality.

## Malformed boundary corpus

`malformed.tsv` contains only inputs both implementations reject:
out-of-range/nonfinite probabilities, empty/duplicate/decreasing latency
profiles, zero batch capacity, and zero service width. Parser failures,
unsupported columns, and language-specific invalid domains are tested locally
but are not semantic parity claims.

## Claim boundary

A passing command establishes agreement on this finite, reviewed corpus and
the explicitly stated domain. It is not a proof of bit-for-bit controller
equivalence, a GPU latency claim, or evidence that either reference is wired
to a production engine.
