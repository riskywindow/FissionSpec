# Replay and evidence schemas

FissionSpec separates workload input, deterministic random outcomes, full event
traces, aggregate statistics, and environment provenance. That separation lets
CPU experiments freeze every non-hardware variable before an accelerator is
rented.

## Workload replay CSV

`fissionspec.workload_generators.load_trace_csv` accepts UTF-8 CSV with these
required columns:

| column | meaning |
|---|---|
| `request_id` | unique stable logical request ID |
| `arrival_ms` | non-negative decode-ready time |
| `output_tokens` | positive requested output length |

Optional columns are:

| column | default | meaning |
|---|---:|---|
| `split` | — | explicit scenario split such as `train` or `validation` |
| `prompt_tokens` | `0` | prompt/context length |
| `speculation_length` | `4` | verifier width including bonus/correction |
| `cache_hit_probability` | `0.8` | scalar or semicolon-separated round schedule |
| `token_acceptance_probability` | `0.8` | scalar or semicolon-separated round schedule |
| `tbt_slo_ms` | `50` | rolling inter-token SLO |
| `deadline_ms` | derived | absolute completion deadline |

Unknown columns are ignored by the simulator but remain covered by the source
file SHA-256. Split selection fails closed when the split column or selected
rows are absent. The loader never silently assigns records to train or
validation.

The dependency-free generators provide:

- exact exponential inter-arrivals for a Poisson process;
- an exact continuous-time two-state Markov-modulated Poisson process; and
- finite-mean Pareto inter-arrivals for heavy-tail stress.

Each draw is addressed through `CounterRNG`, and every `ArrivalTrace` carries a
canonical configuration/RNG digest.

## Full simulation trace

`fissionspec.artifacts.simulation_trace_document` emits schema version 1 with:

- an immutable simulation/not-GPU evidence label;
- policy, workload, profile, and counter-RNG identity;
- optional input and implementation hashes;
- the complete request configuration;
- recomputable aggregate metrics;
- every per-request token timestamp and outcome counter;
- every target launch, physical width, padding row, and realized outcome; and
- every draft/precompute/recovery launch.

The `payload_sha256` covers every field except itself. Loading and writing
verify that link. Canonical JSON is UTF-8, sorted-key, compact, newline
terminated, and rejects NaN/Infinity, so a golden trace generated twice from
the same inputs is byte-identical.

Machine/environment information is intentionally a separate manifest. It is
necessary provenance but would make semantic golden traces differ across
otherwise equivalent machines.

## Accelerator handoff

Before a GPU run:

1. freeze replay CSV bytes and their SHA-256;
2. freeze the policy/controller implementation digest;
3. predeclare the statistical family and stopping bounds;
4. run the same trace through the CPU reference and save the full event
   artifact; and
5. attach the GPU profile, engine commit, graph-bucket configuration, and
   environment manifest without editing the frozen workload.

This makes extra GPU repetitions necessary only for measurement uncertainty or
hardware failures—not for reconstructing an underspecified experiment.
