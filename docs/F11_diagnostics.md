# F11 actual short-training diagnostics

One complete training episode per actual entry point; verification-only small networks. These are finite-update checks, not convergence results.

| Algorithm | Learner | Nonterminal / transitions | Nonterminal % | Nonzero bootstrap % | Mean absolute future term | Gradient updates | Maximum absolute loss |
|---|---|---:|---:|---:|---:|---:|---:|
| mahdrl | TA | 380 / 381 | 99.7375 | 100.0000 | 0.622268 | 378 | 1.28159 |
| mahdrl | HA | 1363 / 1364 | 99.9267 | 100.0000 | 180514 | 1361 | 927642 |
| mahdrl | VA | 1363 / 1364 | 99.9267 | 100.0000 | 203837 | 1361 | 1.10738e+06 |
| ablation_hostrand | TA | 284 / 285 | 99.6491 | 100.0000 | 0.557864 | 282 | 1.82235 |
| ablation_hostrand | VA | 1363 / 1364 | 99.9267 | 99.9816 | 190020 | 1361 | 1.50238e+06 |
| ablation_mgrrand | HA | 1363 / 1364 | 99.9267 | 100.0000 | 201329 | 1361 | 998237 |
| ablation_mgrrand | VA | 1363 / 1364 | 99.9267 | 99.9816 | 198176 | 1361 | 2.13892e+06 |
| ablation_vmrand | TA | 343 / 344 | 99.7093 | 100.0000 | 0.290134 | 341 | 0.982808 |
| ablation_vmrand | HA | 1363 / 1364 | 99.9267 | 100.0000 | 92320.5 | 1361 | 1.29505e+06 |
| ablation_nohost | TA | 361 / 362 | 99.7238 | 99.9304 | 0.636816 | 359 | 1.34292 |
| ablation_nohost | VA | 1363 / 1364 | 99.9267 | 100.0000 | 43856 | 1361 | 959328 |
| pd3qn | PD3QN | 1363 / 1364 | 99.9267 | 100.0000 | 0.182057 | 1361 | 0.330931 |
| marl | Global | 1363 / 1364 | 99.9267 | 99.9267 | 0.778396 | 1364 | 0.810012 |
| marl | Local | 1363 / 1364 | 99.9267 | 99.9267 | 188683 | 1364 | 4.23274e+13 |
| irws | VM_PPO | 2939 / 2940 | 99.9660 | 99.9660 | 0.104414 | 48 | 0.652488 |

All pending counts: 0. All actual short episodes: true terminal; truncation cutoffs: 0.

The DQN bootstrap percentage uses replay samples, so it can differ from the transition percentage. A terminal sample need not be drawn by the final minibatch. A2C/PPO use every collected transition.

The IRWS nonterminal tail value was 0.05522283911705017, computed before update; the final terminal tail was 0.

| Algorithm | Physical-energy balance (J) | Lower interval reward balance | HA formula balance | VA formula balance |
|---|---:|---:|---:|---:|
| mahdrl | 0 | 3.97904e-13 | 0.0 | 0.0 |
| ablation_hostrand | 0 | -8.52651e-13 | N/A | 0.0 |
| ablation_mgrrand | 0 | 5.11591e-13 | 0.0 | 0.0 |
| ablation_vmrand | 0 | -5.11591e-13 | 0.0 | N/A |
| ablation_nohost | 0 | -5.11591e-13 | N/A | 0.0 |
