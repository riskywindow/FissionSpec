# Calibration inputs

FissionSpec consumes three monotone latency curves, all in milliseconds:

- `target_curve`: one target verification launch by request-row count;
- `draft_curve`: normal speculative preparation by request-row count; and
- `recovery_curve`: rollback repair by request-row count.

`verifier_slot_ms` is an explicitly separable estimate of token-axis work in a
rectangular mixed verification batch. Set it to zero if the engine's packed
kernel makes masked speculative positions free; doing so is an important null
experiment, not a tuning trick.

Every publishable profile should add provenance containing the engine commit,
GPU SKU, model pair, tensor-parallel degree, dtype, context-length distribution,
warm-up count, sample count, and raw trace location. The bundled reference
profile is synthetic and is deliberately named as such.

The mechanism workload forces one miss beside fifteen hits. It is intended to
expose head-of-line and padding externality, not to resemble a natural request
distribution. Its `cache_hit_probability` controls SSD outcome-cache lookup;
`token_acceptance_probability` independently controls current-block token
productivity. Keeping those axes separate is required for a valid scheduling
experiment.
