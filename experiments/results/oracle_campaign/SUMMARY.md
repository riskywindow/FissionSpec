# Expanded exact CPU scheduler-oracle campaign

> EXACT BOUNDED CPU MODEL / NOT A GPU MEASUREMENT

Exactness holds only for the declared one-shot, non-preemptive, at-most-three-job finite action space. It does not establish a multi-round serving optimum or accelerator performance.

The campaign solved 216 distinct exact problems and retained 7776 policy comparisons.

| policy | optimal / cases | deadline-regret cases | max deadline gap | p95 same-deadline flow regret |
|---|---:|---:|---:|---:|
| work-conserving-edf | 648 / 1296 | 264 | 1 | 5/4 |
| saguaro-barrier | 648 / 1296 | 264 | 1 | 5/4 |
| spectre-parallel-padded | 648 / 1296 | 264 | 1 | 5/4 |
| immediate-fission | 648 / 1296 | 264 | 1 | 5/4 |
| fixed-coalesce | 222 / 1296 | 550 | 1 | 9/2 |
| fissionspec-horizon-2 | 648 / 1296 | 260 | 1 | 3/2 |

Saguaro and SPECTRE rows compare only their immediate-dispatch timing
component; their defining barrier and padded-recovery execution costs are
outside this oracle. Counterexamples and null cases are both retained.
