# Synthetic experiment artifact

> **All files produced here are synthetic model outputs, not GPU
> measurements.**

The harness runs all five policies with identical workloads and
counter-addressed outcome streams. It crosses two cache-hit levels with two
per-candidate token-acceptance levels; keeping those axes separate prevents a
cache lookup failure from being mistaken for poor draft-token quality. The
factorial is repeated under three deliberately different arrival processes:

- `synchronized-cohort`: every request arrives at time zero;
- `poisson`: exponential inter-arrivals generated from the experiment seed;
- `bursty`: narrow request bursts separated by deterministic idle gaps.

Each `(workload, regime, seed)` tuple is reused unchanged across policies.
Cache outcomes are addressed by
`(seed, request_id, round_id, "cache-hit", draw=0)`. Candidate acceptance uses
the separate `"token-acceptance"` stream and draw indices `0..k-2` (the final
position is the target correction/bonus token), so a
scheduling decision cannot shift another request's random stream. The JSON
artifact includes separate SHA-256 fingerprints of both complete potential key
spaces for every matched comparison.

The profile metadata embeds every latency-curve knot and verifier-slot
coefficient. `direct_hit_delay_ms` covers only explicitly attributed barrier
hold and padded-launch service; paired TBT, request-level TBT attainment, and
TBT-request goodput expose wider queueing effects. The latter counts all output
tokens from requests whose measured inter-token gaps pass, divided by response
makespan; it excludes TTFT and total GPU resources, and one-token requests pass
vacuously. Cache availability is exogenous at a fixed row-based precompute
cost, so this is a post-outcome scheduler mechanism study, not a model of
outcome-tree fanout or cache memory.

Run from the repository root after installing the package:

```bash
python experiments/run_synthetic_sweep.py
python tools/render_synthetic_results.py \
  experiments/results/synthetic_sweep.json
```

The first command writes compact aggregate JSON and a hash-linked per-seed
CSV. The second creates a dependency-free SVG and Markdown summary. Do not
cite their performance numbers as hardware results; replace the synthetic
latency profile with a calibrated, provenance-bearing GPU profile for any
systems claim.

Regenerate both the factorial and the controller decision surface with:

```bash
make artifacts
```
