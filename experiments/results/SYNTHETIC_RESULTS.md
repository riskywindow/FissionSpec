# Synthetic mechanism-study summary

> **SYNTHETIC MODEL OUTPUT — NOT GPU MEASUREMENTS.**

Means across matched common-random-number seeds. Cache availability and draft-token acceptance are independent factors. These values validate simulator mechanics only; they are not hardware evidence.

| Workload | Cache p | Token p | Policy | Observed cache rate | Verifier tok/round | Throughput (tok/s) | P95 TBT (ms) | SLO (%) | Padded slots | Hit externality (ms) | vs. barrier |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| synchronized-cohort | 0.70 | 0.55 | Saguaro barrier | 0.681 | 1.988 | 4429.1 | 21.16 | 55.5 | 0.0 | 1.701 | 1.000x |
| synchronized-cohort | 0.70 | 0.55 | SPECTRE parallel padded | 0.678 | 1.990 | 3923.0 | 28.21 | 60.7 | 897.0 | 1.175 | 0.886x |
| synchronized-cohort | 0.70 | 0.55 | Immediate fission | 0.681 | 1.988 | 4434.8 | 21.16 | 55.5 | 0.0 | 0.000 | 1.001x |
| synchronized-cohort | 0.70 | 0.55 | Fixed coalesce | 0.681 | 1.988 | 4404.1 | 21.16 | 55.5 | 0.0 | 0.000 | 0.994x |
| synchronized-cohort | 0.70 | 0.55 | FissionSpec H2 | 0.681 | 1.988 | 4434.8 | 21.16 | 55.5 | 0.0 | 0.000 | 1.001x |
| synchronized-cohort | 0.70 | 0.90 | Saguaro barrier | 0.685 | 3.362 | 7548.8 | 21.16 | 73.7 | 0.0 | 1.644 | 1.000x |
| synchronized-cohort | 0.70 | 0.90 | SPECTRE parallel padded | 0.685 | 3.361 | 6292.0 | 28.21 | 75.9 | 537.3 | 1.130 | 0.834x |
| synchronized-cohort | 0.70 | 0.90 | Immediate fission | 0.685 | 3.362 | 7568.0 | 21.16 | 73.7 | 0.0 | 0.000 | 1.003x |
| synchronized-cohort | 0.70 | 0.90 | Fixed coalesce | 0.685 | 3.362 | 7551.5 | 21.16 | 73.7 | 0.0 | 0.000 | 1.000x |
| synchronized-cohort | 0.70 | 0.90 | FissionSpec H2 | 0.685 | 3.362 | 7568.0 | 21.16 | 73.7 | 0.0 | 0.000 | 1.003x |
| synchronized-cohort | 0.95 | 0.55 | Saguaro barrier | 0.947 | 1.988 | 4461.3 | 21.16 | 55.5 | 0.0 | 0.731 | 1.000x |
| synchronized-cohort | 0.95 | 0.55 | SPECTRE parallel padded | 0.948 | 1.986 | 4333.9 | 21.16 | 56.5 | 161.3 | 0.236 | 0.972x |
| synchronized-cohort | 0.95 | 0.55 | Immediate fission | 0.947 | 1.988 | 4466.0 | 21.16 | 55.5 | 0.0 | 0.000 | 1.001x |
| synchronized-cohort | 0.95 | 0.55 | Fixed coalesce | 0.947 | 1.988 | 4426.0 | 21.16 | 55.5 | 0.0 | 0.000 | 0.992x |
| synchronized-cohort | 0.95 | 0.55 | FissionSpec H2 | 0.947 | 1.988 | 4466.0 | 21.16 | 55.5 | 0.0 | 0.000 | 1.001x |
| synchronized-cohort | 0.95 | 0.90 | Saguaro barrier | 0.948 | 3.362 | 7562.5 | 21.16 | 73.7 | 0.0 | 0.705 | 1.000x |
| synchronized-cohort | 0.95 | 0.90 | SPECTRE parallel padded | 0.948 | 3.365 | 7284.5 | 21.16 | 74.1 | 93.3 | 0.234 | 0.963x |
| synchronized-cohort | 0.95 | 0.90 | Immediate fission | 0.948 | 3.362 | 7557.1 | 21.16 | 73.7 | 0.0 | 0.000 | 0.999x |
| synchronized-cohort | 0.95 | 0.90 | Fixed coalesce | 0.948 | 3.362 | 7516.3 | 21.16 | 73.7 | 0.0 | 0.000 | 0.994x |
| synchronized-cohort | 0.95 | 0.90 | FissionSpec H2 | 0.948 | 3.362 | 7557.1 | 21.16 | 73.7 | 0.0 | 0.000 | 0.999x |
| poisson | 0.70 | 0.55 | Saguaro barrier | 0.693 | 1.958 | 4325.8 | 21.16 | 56.0 | 0.0 | 1.670 | 1.000x |
| poisson | 0.70 | 0.55 | SPECTRE parallel padded | 0.695 | 1.964 | 3912.2 | 28.21 | 60.8 | 854.3 | 1.093 | 0.904x |
| poisson | 0.70 | 0.55 | Immediate fission | 0.693 | 1.958 | 4360.1 | 21.16 | 56.2 | 0.0 | 0.000 | 1.008x |
| poisson | 0.70 | 0.55 | Fixed coalesce | 0.693 | 1.958 | 4353.2 | 21.16 | 56.3 | 0.0 | 0.000 | 1.006x |
| poisson | 0.70 | 0.55 | FissionSpec H2 | 0.693 | 1.958 | 4367.8 | 21.16 | 56.4 | 0.0 | 0.000 | 1.010x |
| poisson | 0.70 | 0.90 | Saguaro barrier | 0.703 | 3.311 | 7360.6 | 21.16 | 74.2 | 0.0 | 1.590 | 1.000x |
| poisson | 0.70 | 0.90 | SPECTRE parallel padded | 0.705 | 3.320 | 6265.0 | 28.21 | 76.4 | 506.3 | 1.056 | 0.851x |
| poisson | 0.70 | 0.90 | Immediate fission | 0.703 | 3.311 | 7399.3 | 21.16 | 74.5 | 0.0 | 0.000 | 1.005x |
| poisson | 0.70 | 0.90 | Fixed coalesce | 0.703 | 3.311 | 7367.4 | 21.16 | 74.2 | 0.0 | 0.000 | 1.001x |
| poisson | 0.70 | 0.90 | FissionSpec H2 | 0.703 | 3.311 | 7407.0 | 21.16 | 74.4 | 0.0 | 0.000 | 1.006x |
| poisson | 0.95 | 0.55 | Saguaro barrier | 0.945 | 1.958 | 4367.1 | 21.16 | 56.1 | 0.0 | 0.636 | 1.000x |
| poisson | 0.95 | 0.55 | SPECTRE parallel padded | 0.944 | 1.963 | 4285.2 | 21.16 | 57.0 | 172.7 | 0.249 | 0.981x |
| poisson | 0.95 | 0.55 | Immediate fission | 0.945 | 1.958 | 4386.8 | 21.16 | 56.3 | 0.0 | 0.000 | 1.004x |
| poisson | 0.95 | 0.55 | Fixed coalesce | 0.945 | 1.958 | 4368.7 | 21.16 | 56.4 | 0.0 | 0.000 | 1.000x |
| poisson | 0.95 | 0.55 | FissionSpec H2 | 0.945 | 1.958 | 4389.3 | 21.16 | 56.5 | 0.0 | 0.000 | 1.005x |
| poisson | 0.95 | 0.90 | Saguaro barrier | 0.950 | 3.311 | 7374.3 | 21.16 | 74.3 | 0.0 | 0.561 | 1.000x |
| poisson | 0.95 | 0.90 | SPECTRE parallel padded | 0.951 | 3.306 | 7161.9 | 21.16 | 75.0 | 89.0 | 0.211 | 0.971x |
| poisson | 0.95 | 0.90 | Immediate fission | 0.950 | 3.311 | 7413.3 | 21.16 | 74.6 | 0.0 | 0.000 | 1.005x |
| poisson | 0.95 | 0.90 | Fixed coalesce | 0.950 | 3.311 | 7379.5 | 21.16 | 74.5 | 0.0 | 0.000 | 1.001x |
| poisson | 0.95 | 0.90 | FissionSpec H2 | 0.950 | 3.311 | 7420.9 | 21.16 | 74.7 | 0.0 | 0.000 | 1.006x |
| bursty | 0.70 | 0.55 | Saguaro barrier | 0.709 | 1.981 | 4358.7 | 21.16 | 59.4 | 0.0 | 1.615 | 1.000x |
| bursty | 0.70 | 0.55 | SPECTRE parallel padded | 0.709 | 1.984 | 3959.9 | 28.21 | 63.3 | 810.3 | 1.048 | 0.908x |
| bursty | 0.70 | 0.55 | Immediate fission | 0.709 | 1.981 | 4384.3 | 21.16 | 59.7 | 0.0 | 0.000 | 1.006x |
| bursty | 0.70 | 0.55 | Fixed coalesce | 0.709 | 1.981 | 4355.3 | 21.16 | 59.6 | 0.0 | 0.000 | 0.999x |
| bursty | 0.70 | 0.55 | FissionSpec H2 | 0.709 | 1.981 | 4388.9 | 21.16 | 59.9 | 0.0 | 0.000 | 1.007x |
| bursty | 0.70 | 0.90 | Saguaro barrier | 0.709 | 3.334 | 7336.5 | 21.16 | 78.0 | 0.0 | 1.567 | 1.000x |
| bursty | 0.70 | 0.90 | SPECTRE parallel padded | 0.706 | 3.358 | 6348.6 | 28.21 | 78.9 | 499.3 | 1.047 | 0.865x |
| bursty | 0.70 | 0.90 | Immediate fission | 0.709 | 3.334 | 7428.9 | 21.16 | 78.7 | 0.0 | 0.000 | 1.013x |
| bursty | 0.70 | 0.90 | Fixed coalesce | 0.709 | 3.334 | 7377.0 | 21.16 | 78.5 | 0.0 | 0.000 | 1.006x |
| bursty | 0.70 | 0.90 | FissionSpec H2 | 0.709 | 3.334 | 7438.2 | 21.16 | 78.7 | 0.0 | 0.000 | 1.014x |
| bursty | 0.95 | 0.55 | Saguaro barrier | 0.954 | 1.981 | 4394.9 | 21.16 | 59.9 | 0.0 | 0.608 | 1.000x |
| bursty | 0.95 | 0.55 | SPECTRE parallel padded | 0.954 | 1.980 | 4332.0 | 21.16 | 60.9 | 143.7 | 0.206 | 0.986x |
| bursty | 0.95 | 0.55 | Immediate fission | 0.954 | 1.981 | 4421.1 | 21.16 | 60.1 | 0.0 | 0.000 | 1.006x |
| bursty | 0.95 | 0.55 | Fixed coalesce | 0.954 | 1.981 | 4375.3 | 21.16 | 59.9 | 0.0 | 0.000 | 0.996x |
| bursty | 0.95 | 0.55 | FissionSpec H2 | 0.954 | 1.981 | 4420.8 | 21.16 | 60.3 | 0.0 | 0.000 | 1.006x |
| bursty | 0.95 | 0.90 | Saguaro barrier | 0.948 | 3.334 | 7403.1 | 21.16 | 78.5 | 0.0 | 0.660 | 1.000x |
| bursty | 0.95 | 0.90 | SPECTRE parallel padded | 0.949 | 3.340 | 7216.3 | 21.16 | 79.1 | 94.0 | 0.223 | 0.975x |
| bursty | 0.95 | 0.90 | Immediate fission | 0.948 | 3.334 | 7438.1 | 21.16 | 78.7 | 0.0 | 0.000 | 1.005x |
| bursty | 0.95 | 0.90 | Fixed coalesce | 0.948 | 3.334 | 7394.6 | 21.16 | 78.5 | 0.0 | 0.000 | 0.999x |
| bursty | 0.95 | 0.90 | FissionSpec H2 | 0.948 | 3.334 | 7482.0 | 21.16 | 79.2 | 0.0 | 0.000 | 1.011x |

Generated from `synthetic_sweep.json` by `tools/render_synthetic_results.py`.
