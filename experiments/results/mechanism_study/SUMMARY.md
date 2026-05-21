# FissionSpec CPU mechanism study

> SYNTHETIC CPU MODEL / NOT A GPU MEASUREMENT

These paired effects identify interventions inside the declared CPU models only. They do not estimate CUDA-kernel, accelerator-memory, network-fabric, power, or production-serving effects.

All entries are intervention-minus-reference paired means. Intervals use the
single predeclared family described in `inference.json`. Mixed, null, and adverse
effects are retained.

| factor / level | metric | paired mean | simultaneous interval |
|---|---:|---:|---:|
| recovery-latency / lower | throughput_tokens_per_s | 3.58908 | [0.497934, 7.45201] |
| recovery-latency / lower | p95_tbt_ms | 0 | [0, 0] |
| recovery-latency / lower | target_launches_per_request | 0.025 | [-0.00555556, 0.049711] |
| recovery-latency / upper | throughput_tokens_per_s | -26.6482 | [-39.3966, -15.465] |
| recovery-latency / upper | p95_tbt_ms | -0.139191 | [-0.447744, 0] |
| recovery-latency / upper | target_launches_per_request | 0.0902778 | [0.0569444, 0.121528] |
| physical-verifier-slot-cost / lower | throughput_tokens_per_s | 198.451 | [196.755, 200.223] |
| physical-verifier-slot-cost / lower | p95_tbt_ms | -2.35224 | [-2.51885, -2.1549] |
| physical-verifier-slot-cost / lower | target_launches_per_request | 0 | [0, 0] |
| physical-verifier-slot-cost / upper | throughput_tokens_per_s | -377.801 | [-380.93, -374.931] |
| physical-verifier-slot-cost / upper | p95_tbt_ms | 5.48606 | [4.99892, 5.85443] |
| physical-verifier-slot-cost / upper | target_launches_per_request | -0.000694444 | [-0.00416667, 0] |
| target-batch-capacity / lower | throughput_tokens_per_s | -204.55 | [-215.213, -193.084] |
| target-batch-capacity / lower | p95_tbt_ms | -4.60093 | [-5.254, -3.73846] |
| target-batch-capacity / lower | target_launches_per_request | 0.293056 | [0.280556, 0.305556] |
| target-batch-capacity / upper | throughput_tokens_per_s | 110.486 | [95.7922, 123.525] |
| target-batch-capacity / upper | p95_tbt_ms | 3.33463 | [2.09257, 4.68282] |
| target-batch-capacity / upper | target_launches_per_request | -0.166667 | [-0.180961, -0.1489] |
| controller-max-wait / lower | throughput_tokens_per_s | -0.158297 | [-0.791487, 0] |
| controller-max-wait / lower | p95_tbt_ms | 0.01175 | [0, 0.05875] |
| controller-max-wait / lower | target_launches_per_request | 0.075 | [0.0465278, 0.103472] |
| controller-max-wait / upper | throughput_tokens_per_s | 0 | [0, 0] |
| controller-max-wait / upper | p95_tbt_ms | 0 | [0, 0] |
| controller-max-wait / upper | target_launches_per_request | 0 | [0, 0] |
| cache-budget / lower | cache_hit_rate | -0.0743056 | [-0.0885999, -0.0597222] |
| cache-budget / lower | restricted_next_ready_delay_ms | 0.460893 | [0.360829, 0.547915] |
| cache-budget / lower | next_round_unready_rate | 0 | [0, 0] |
| cache-budget / upper | cache_hit_rate | 0 | [0, 0] |
| cache-budget / upper | restricted_next_ready_delay_ms | 0 | [0, 0] |
| cache-budget / upper | next_round_unready_rate | 0 | [0, 0] |
| branch-fanout / lower | cache_hit_rate | -0.2375 | [-0.270833, -0.205556] |
| branch-fanout / lower | restricted_next_ready_delay_ms | 1.88102 | [1.57007, 2.20983] |
| branch-fanout / lower | next_round_unready_rate | 0 | [0, 0] |
| branch-fanout / upper | cache_hit_rate | 0.148611 | [0.118056, 0.184433] |
| branch-fanout / upper | restricted_next_ready_delay_ms | -1.10453 | [-1.3871, -0.845802] |
| branch-fanout / upper | next_round_unready_rate | 0 | [0, 0] |
| network-jitter / lower | cache_hit_rate | 0 | [0, 0] |
| network-jitter / lower | restricted_next_ready_delay_ms | -0.00451802 | [-0.00557475, -0.00354047] |
| network-jitter / lower | next_round_unready_rate | 0 | [0, 0] |
| network-jitter / upper | cache_hit_rate | -0.000694444 | [-0.00416667, 0] |
| network-jitter / upper | restricted_next_ready_delay_ms | 0.0337667 | [0.010799, 0.0910191] |
| network-jitter / upper | next_round_unready_rate | 0 | [0, 0] |
| network-failure / lower | cache_hit_rate | 0.000694444 | [0, 0.00387768] |
| network-failure / lower | restricted_next_ready_delay_ms | -0.0109931 | [-0.0341819, -0.000328906] |
| network-failure / lower | next_round_unready_rate | 0 | [0, 0] |
| network-failure / upper | cache_hit_rate | -0.0208333 | [-0.0305556, -0.0118056] |
| network-failure / upper | restricted_next_ready_delay_ms | 0.276564 | [0.145176, 0.419812] |
| network-failure / upper | next_round_unready_rate | 0.00347222 | [0, 0.00833333] |

The table is mechanistic sensitivity evidence inside frozen CPU models.
It is not a policy leaderboard and is not GPU-performance evidence.
