# Controller overhead evidence

The overhead package keeps algorithmic evidence separate from local timing:

- `controller_complexity.json` is deterministic. It audits the production
  controller paths over equal current/future cohort sizes from 1+1 through
  128+128 rows.
- `controller_overhead_local.json` retains raw repeated wall-clock observations,
  toolchain and host metadata, and the controls that were *not* available.

For the fully evaluated feasible two-plan path, the Python reference policy
materializes and globally sorts `n + m` forecast rows, giving
`O((n + m) log(n + m))` time and `O(n + m)` temporary space. The Rust hot path
uses zero-copy slices and repeated linear visitors: the audited path makes eight
item visits per current row and seven per future row, plus three
`O(log k)` latency-profile queries. Its controller-local auxiliary space is
`O(1)` and `Horizon2Controller.evaluate` contains no explicit heap allocation.
That claim excludes caller-owned input construction and the surrounding serving
loop.

The timing harness calls the real Python `FissionSpecPolicy.dispatch_at` and
real Rust `Horizon2Controller.decide`. It reports raw repeats and local
per-decision summaries, but it deliberately reports no Python/Rust speed ratio:
the language harness loops are not identical. CPU affinity, fixed frequency,
and background-load isolation are not available, so the snapshot is useful for
finding pathological scaling only. Nanosecond values are not portable and are
not GPU or end-to-end serving evidence.

Reproduce:

```bash
PYTHONPATH=src python experiments/run_controller_overhead.py \
  --output-dir experiments/results/controller_overhead
PYTHONPATH=src python experiments/run_controller_overhead.py \
  --output-dir experiments/results/controller_overhead \
  --verify
```

Use `--structural-only` to reproduce only the byte-stable complexity artifact.
The local timing file is expected to change across runs and machines; its
manifest protects the captured snapshot from later mutation. The recorded Rust
command is repository-relative and the closed manifest is self-hashed, so the
artifact does not embed the checkout's absolute path.
