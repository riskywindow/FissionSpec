# Synthetic mechanism-study summary

> **SYNTHETIC MODEL OUTPUT — NOT GPU MEASUREMENTS.**

Means across matched common-random-number seeds. Cache availability and draft-token acceptance are independent factors. These values validate simulator mechanics only; they are not hardware evidence.

| Workload | Cache p | Token p | Policy | Observed cache rate | Verifier tok/round | Throughput (tok/s) | P95 TBT (ms) | Gap TBT pass (%) | Request TBT pass (%) | TBT-request tok/s | Padded slots | Direct delay / hit (ms) | vs. barrier |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| synchronized-cohort | 0.70 | 0.55 | Saguaro barrier | 0.681 | 1.988 | 4143.8 | 14.10 | 64.4 | 0.0 | 0.0 | 0.0 | 1.745 | 1.000x |
| synchronized-cohort | 0.70 | 0.55 | SPECTRE parallel padded | 0.681 | 1.991 | 4224.0 | 14.10 | 80.5 | 0.0 | 0.0 | 100.7 | 0.086 | 1.019x |
| synchronized-cohort | 0.70 | 0.55 | Immediate fission | 0.681 | 1.988 | 4270.3 | 14.10 | 80.2 | 0.7 | 29.9 | 0.0 | 0.000 | 1.031x |
| synchronized-cohort | 0.70 | 0.55 | Fixed coalesce | 0.681 | 1.988 | 4270.3 | 14.10 | 80.2 | 0.7 | 29.9 | 0.0 | 0.000 | 1.031x |
| synchronized-cohort | 0.70 | 0.55 | FissionSpec H2 | 0.681 | 1.988 | 4270.3 | 14.10 | 80.2 | 0.7 | 29.9 | 0.0 | 0.000 | 1.031x |
| synchronized-cohort | 0.70 | 0.90 | Saguaro barrier | 0.686 | 3.362 | 7035.7 | 14.10 | 80.3 | 5.6 | 388.6 | 0.0 | 1.660 | 1.000x |
| synchronized-cohort | 0.70 | 0.90 | SPECTRE parallel padded | 0.684 | 3.364 | 7129.5 | 14.07 | 89.6 | 0.7 | 49.1 | 58.3 | 0.087 | 1.013x |
| synchronized-cohort | 0.70 | 0.90 | Immediate fission | 0.686 | 3.362 | 7275.5 | 14.07 | 89.5 | 0.0 | 0.0 | 0.0 | 0.000 | 1.034x |
| synchronized-cohort | 0.70 | 0.90 | Fixed coalesce | 0.686 | 3.362 | 7199.6 | 14.07 | 89.4 | 0.0 | 0.0 | 0.0 | 0.000 | 1.023x |
| synchronized-cohort | 0.70 | 0.90 | FissionSpec H2 | 0.686 | 3.362 | 7275.5 | 14.07 | 89.5 | 0.0 | 0.0 | 0.0 | 0.000 | 1.034x |
| synchronized-cohort | 0.95 | 0.55 | Saguaro barrier | 0.947 | 1.988 | 4293.0 | 14.10 | 65.8 | 0.0 | 0.0 | 0.0 | 0.753 | 1.000x |
| synchronized-cohort | 0.95 | 0.55 | SPECTRE parallel padded | 0.947 | 1.987 | 4324.9 | 7.05 | 95.7 | 20.8 | 903.7 | 36.0 | 0.037 | 1.007x |
| synchronized-cohort | 0.95 | 0.55 | Immediate fission | 0.947 | 1.988 | 4345.0 | 7.05 | 95.7 | 18.8 | 816.7 | 0.0 | 0.000 | 1.012x |
| synchronized-cohort | 0.95 | 0.55 | Fixed coalesce | 0.947 | 1.988 | 4306.9 | 7.35 | 95.7 | 18.8 | 809.3 | 0.0 | 0.000 | 1.003x |
| synchronized-cohort | 0.95 | 0.55 | FissionSpec H2 | 0.947 | 1.988 | 4345.0 | 7.05 | 95.7 | 18.8 | 816.7 | 0.0 | 0.000 | 1.012x |
| synchronized-cohort | 0.95 | 0.90 | Saguaro barrier | 0.947 | 3.362 | 7255.2 | 14.10 | 81.3 | 11.1 | 816.3 | 0.0 | 0.749 | 1.000x |
| synchronized-cohort | 0.95 | 0.90 | SPECTRE parallel padded | 0.946 | 3.360 | 7326.0 | 7.05 | 98.1 | 43.8 | 3205.4 | 26.0 | 0.055 | 1.010x |
| synchronized-cohort | 0.95 | 0.90 | Immediate fission | 0.947 | 3.362 | 7388.1 | 7.05 | 98.0 | 43.1 | 3181.4 | 0.0 | 0.000 | 1.018x |
| synchronized-cohort | 0.95 | 0.90 | Fixed coalesce | 0.947 | 3.362 | 7196.2 | 7.05 | 98.0 | 42.4 | 3049.8 | 0.0 | 0.000 | 0.992x |
| synchronized-cohort | 0.95 | 0.90 | FissionSpec H2 | 0.947 | 3.362 | 7388.1 | 7.05 | 98.0 | 43.1 | 3181.4 | 0.0 | 0.000 | 1.018x |
| poisson | 0.70 | 0.55 | Saguaro barrier | 0.693 | 1.958 | 4075.3 | 14.10 | 63.5 | 0.0 | 0.0 | 0.0 | 1.709 | 1.000x |
| poisson | 0.70 | 0.55 | SPECTRE parallel padded | 0.694 | 1.959 | 4163.5 | 14.10 | 80.3 | 0.7 | 28.3 | 123.0 | 0.104 | 1.022x |
| poisson | 0.70 | 0.55 | Immediate fission | 0.693 | 1.958 | 4252.3 | 14.10 | 80.2 | 0.0 | 0.0 | 0.0 | 0.000 | 1.043x |
| poisson | 0.70 | 0.55 | Fixed coalesce | 0.693 | 1.958 | 4236.3 | 14.10 | 79.7 | 0.0 | 0.0 | 0.0 | 0.000 | 1.039x |
| poisson | 0.70 | 0.55 | FissionSpec H2 | 0.693 | 1.958 | 4252.3 | 14.10 | 80.2 | 0.0 | 0.0 | 0.0 | 0.000 | 1.043x |
| poisson | 0.70 | 0.90 | Saguaro barrier | 0.705 | 3.311 | 6867.6 | 14.10 | 79.6 | 0.0 | 0.0 | 0.0 | 1.655 | 1.000x |
| poisson | 0.70 | 0.90 | SPECTRE parallel padded | 0.705 | 3.319 | 6894.0 | 14.09 | 89.8 | 0.7 | 47.3 | 70.7 | 0.089 | 1.004x |
| poisson | 0.70 | 0.90 | Immediate fission | 0.705 | 3.311 | 7053.2 | 14.09 | 89.8 | 0.7 | 48.8 | 0.0 | 0.000 | 1.027x |
| poisson | 0.70 | 0.90 | Fixed coalesce | 0.705 | 3.311 | 6987.6 | 14.09 | 89.3 | 2.1 | 145.2 | 0.0 | 0.000 | 1.018x |
| poisson | 0.70 | 0.90 | FissionSpec H2 | 0.705 | 3.311 | 7053.2 | 14.09 | 89.8 | 0.7 | 48.8 | 0.0 | 0.000 | 1.027x |
| poisson | 0.95 | 0.55 | Saguaro barrier | 0.945 | 1.958 | 4196.7 | 14.10 | 64.7 | 0.0 | 0.0 | 0.0 | 0.708 | 1.000x |
| poisson | 0.95 | 0.55 | SPECTRE parallel padded | 0.945 | 1.961 | 4277.2 | 7.05 | 95.6 | 25.0 | 1070.6 | 40.0 | 0.041 | 1.019x |
| poisson | 0.95 | 0.55 | Immediate fission | 0.945 | 1.958 | 4297.0 | 7.05 | 95.5 | 25.0 | 1075.7 | 0.0 | 0.000 | 1.024x |
| poisson | 0.95 | 0.55 | Fixed coalesce | 0.945 | 1.958 | 4271.9 | 7.37 | 95.3 | 23.6 | 1009.9 | 0.0 | 0.000 | 1.018x |
| poisson | 0.95 | 0.55 | FissionSpec H2 | 0.945 | 1.958 | 4297.0 | 7.05 | 95.5 | 25.0 | 1075.7 | 0.0 | 0.000 | 1.024x |
| poisson | 0.95 | 0.90 | Saguaro barrier | 0.952 | 3.311 | 7128.2 | 14.10 | 81.0 | 16.7 | 1179.4 | 0.0 | 0.647 | 1.000x |
| poisson | 0.95 | 0.90 | SPECTRE parallel padded | 0.951 | 3.310 | 7172.5 | 7.05 | 98.1 | 50.7 | 3635.0 | 25.0 | 0.050 | 1.006x |
| poisson | 0.95 | 0.90 | Immediate fission | 0.952 | 3.311 | 7249.7 | 7.05 | 98.0 | 49.3 | 3577.2 | 0.0 | 0.000 | 1.017x |
| poisson | 0.95 | 0.90 | Fixed coalesce | 0.952 | 3.311 | 7076.3 | 7.05 | 98.1 | 50.0 | 3541.6 | 0.0 | 0.000 | 0.993x |
| poisson | 0.95 | 0.90 | FissionSpec H2 | 0.952 | 3.311 | 7249.7 | 7.05 | 98.0 | 49.3 | 3577.2 | 0.0 | 0.000 | 1.017x |
| bursty | 0.70 | 0.55 | Saguaro barrier | 0.709 | 1.981 | 4106.0 | 14.10 | 64.6 | 0.0 | 0.0 | 0.0 | 1.630 | 1.000x |
| bursty | 0.70 | 0.55 | SPECTRE parallel padded | 0.710 | 1.984 | 4173.4 | 14.10 | 82.0 | 0.7 | 28.8 | 129.7 | 0.092 | 1.017x |
| bursty | 0.70 | 0.55 | Immediate fission | 0.709 | 1.981 | 4239.1 | 14.10 | 82.0 | 0.0 | 0.0 | 0.0 | 0.000 | 1.032x |
| bursty | 0.70 | 0.55 | Fixed coalesce | 0.709 | 1.981 | 4242.3 | 14.10 | 81.4 | 0.7 | 29.1 | 0.0 | 0.000 | 1.033x |
| bursty | 0.70 | 0.55 | FissionSpec H2 | 0.709 | 1.981 | 4239.1 | 14.10 | 82.0 | 0.0 | 0.0 | 0.0 | 0.000 | 1.032x |
| bursty | 0.70 | 0.90 | Saguaro barrier | 0.709 | 3.334 | 6905.9 | 14.10 | 79.6 | 0.0 | 0.0 | 0.0 | 1.604 | 1.000x |
| bursty | 0.70 | 0.90 | SPECTRE parallel padded | 0.708 | 3.335 | 6852.6 | 14.08 | 90.3 | 2.1 | 139.9 | 79.3 | 0.093 | 0.992x |
| bursty | 0.70 | 0.90 | Immediate fission | 0.709 | 3.334 | 7096.3 | 14.04 | 90.2 | 1.4 | 98.1 | 0.0 | 0.000 | 1.028x |
| bursty | 0.70 | 0.90 | Fixed coalesce | 0.709 | 3.334 | 6994.2 | 14.06 | 89.9 | 2.1 | 143.2 | 0.0 | 0.000 | 1.013x |
| bursty | 0.70 | 0.90 | FissionSpec H2 | 0.709 | 3.334 | 7096.3 | 14.04 | 90.2 | 1.4 | 98.1 | 0.0 | 0.000 | 1.028x |
| bursty | 0.95 | 0.55 | Saguaro barrier | 0.953 | 1.981 | 4288.8 | 14.10 | 67.2 | 0.0 | 0.0 | 0.0 | 0.592 | 1.000x |
| bursty | 0.95 | 0.55 | SPECTRE parallel padded | 0.954 | 1.981 | 4305.8 | 9.38 | 95.1 | 26.4 | 1135.6 | 29.3 | 0.032 | 1.004x |
| bursty | 0.95 | 0.55 | Immediate fission | 0.953 | 1.981 | 4355.2 | 8.40 | 95.7 | 22.9 | 999.1 | 0.0 | 0.000 | 1.016x |
| bursty | 0.95 | 0.55 | Fixed coalesce | 0.953 | 1.981 | 4275.0 | 7.45 | 95.9 | 25.7 | 1098.6 | 0.0 | 0.000 | 0.997x |
| bursty | 0.95 | 0.55 | FissionSpec H2 | 0.953 | 1.981 | 4355.2 | 8.40 | 95.7 | 22.9 | 999.1 | 0.0 | 0.000 | 1.016x |
| bursty | 0.95 | 0.90 | Saguaro barrier | 0.949 | 3.334 | 7177.2 | 14.10 | 81.4 | 1.4 | 100.0 | 0.0 | 0.660 | 1.000x |
| bursty | 0.95 | 0.90 | SPECTRE parallel padded | 0.948 | 3.339 | 7217.1 | 7.05 | 97.5 | 42.4 | 3056.3 | 17.3 | 0.028 | 1.006x |
| bursty | 0.95 | 0.90 | Immediate fission | 0.949 | 3.334 | 7288.7 | 7.05 | 97.9 | 42.4 | 3086.2 | 0.0 | 0.000 | 1.016x |
| bursty | 0.95 | 0.90 | Fixed coalesce | 0.949 | 3.334 | 7062.5 | 7.05 | 97.9 | 41.0 | 2893.1 | 0.0 | 0.000 | 0.984x |
| bursty | 0.95 | 0.90 | FissionSpec H2 | 0.949 | 3.334 | 7288.7 | 7.05 | 97.9 | 42.4 | 3086.2 | 0.0 | 0.000 | 1.016x |

Generated from `synthetic_sweep.json` by `tools/render_synthetic_results.py`.
