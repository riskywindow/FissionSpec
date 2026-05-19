# Token-exact speculative semantics

`fissionspec.semantics` is the dependency-free CPU reference beneath the
count-level serving simulator. It exists to answer a narrow but essential
question exactly:

> Does outcome-decoupled execution preserve the token distribution and
> committed target state of ordinary speculative decoding?

This is an executable semantic oracle for tiny vocabularies and horizons, not a
performance model or inference engine.

## Exact autoregressive models

`TinyAutoregressiveModel` stores next-token rows as `fractions.Fraction`.
Contexts are token tuples, and the longest suffix matching the committed prefix
selects the row. The empty context is required as a fallback. Thus a complete
prefix table and a compact finite-order Markov model use the same interface.

Rows are validated to sum to one exactly. `from_weights` converts non-negative
integer weights to rational probabilities without floating-point
normalization. The model fingerprint hashes the vocabulary, contexts, and every
probability numerator and denominator in canonical order.

## Rejection sampling

For a draft proposal \(x_1,\ldots,x_k\), the verifier evaluates target row
\(p_i\) and draft row \(q_i\) at the same autoregressive prefix. Candidate
\(x_i\) is accepted with probability

\[
  a_i = \min\left(1, \frac{p_i(x_i)}{q_i(x_i)}\right).
\]

At the first rejection, the verifier emits the accepted proposal prefix and
draws one correction token from

\[
  r_i(x) =
  \frac{\max(0, p_i(x)-q_i(x))}
       {\sum_y \max(0, p_i(y)-q_i(y))}.
\]

If all \(k\) candidates are accepted, it emits them and samples one bonus token
from the target row after the full proposal. No rejected candidate or
unverified suffix is committed.

`verification_outcome_distribution` enumerates this conditional distribution
for a fixed proposal. `speculative_sequence_distribution` additionally
enumerates every autoregressive draft proposal and every later round. Both use
exact rational arithmetic. Tests compare the resulting map for every sequence
in tiny horizons directly with `target_sequence_distribution`; there is no
tolerance or Monte Carlo argument in this equality check.

At a finite output horizon, a round drafts at most
`min(speculation_length, remaining - 1)` tokens. This leaves room for the
correction or bonus token and avoids generating then silently truncating a
random variable.

## Counter-addressed exact draws

Sampled execution uses the existing `CounterRNG`, addressed by:

```text
(seed, request_id, logical_round, semantic_stream, rejection_attempt)
```

Semantic streams separately name draft positions, acceptance decisions,
residual corrections, target bonuses, and direct final tokens. Rational
categorical and Bernoulli draws first convert probabilities to integer weights,
then use bit rejection to obtain an unbiased integer below the exact
denominator. They never round a `Fraction` through a binary floating-point
threshold.

There is no global draw cursor. `DecodeSession` is immutable, so requests may be
advanced in barrier order, drained immediately, or repeatedly removed and
rebatched. The same request and logical round address the same draws in every
schedule.

## Outcome-continuation cache identity

An `OutcomeContinuationKey` includes:

- request ID and logical verification round;
- the parent committed-state digest;
- the draft-model fingerprint;
- the complete proposal;
- the accepted proposal length;
- the correction or bonus token; and
- whether the terminal token came from rejection or all-accepted bonus.

This prevents two branches that emit the same surface token from aliasing when
their proposals or accepted lengths differ. `prepare_outcome_continuations`
enumerates every positive-probability verifier outcome and drafts its next
block eagerly. Its optional `outcome_order` models arbitrary worker completion
order. Entries are canonicalized by key digest, and each branch samples the next
proposal at the next logical round, so a cache hit matches uncached drafting
exactly.

## Committed-state digest

`CommittedState.kv_digest` hashes the target-model fingerprint and exact ordered
committed token sequence. Provisional draft work is excluded. Reconstructing a
state from the prompt plus every atomic `DecodeRound.emitted_tokens` must
produce the same digest after every round.

The digest is a token-semantic KV oracle, not a hash of tensor bytes. It proves
that a deterministic target backend should have the same logical KV contents;
numerical kernel parity remains a GPU/backend validation task.

## Acceptance gates

`tests/test_semantics.py` covers:

1. exhaustive exact distribution equality across hand-built and
   deterministically randomized two- and three-token models, including draft
   support holes;
2. explicit rejection-residual and all-accepted bonus probabilities;
3. greedy target/speculative token and state equality;
4. deterministic Monte Carlo agreement with the exact joint distribution;
5. cache-hit versus uncached continuation equality and cache build-order
   invariance;
6. barrier, immediate, and adversarial rebatching invariance; and
7. per-round committed-state reconstruction plus exclusion of rejected and
   unverified tokens.

Run the focused oracle with:

```bash
PYTHONPATH=src python3 -m unittest tests.test_semantics -v
```
