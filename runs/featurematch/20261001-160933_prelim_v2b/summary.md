# FeatureMatch preliminary gates (prelim_v2b, in-process)

run dir: /home/ec2-user/wt/featurematch/runs/featurematch/20261001-160933_prelim_v2b

## T2
| solver | pass rate | k/n | Wilson 95% |
|---|---|---|---|
| reference_one_shot | 1.0 | 4/4 | (0.51, 1.0) |
| reference_best_of_5 | 1.0 | 4/4 | (0.51, 1.0) |
| blackbox | 0.0 | 0/4 | (0.0, 0.49) |
| nothing | 0.0 | 0/4 | (0.0, 0.49) |
| always_claim | 0.0 | 0/4 | (0.0, 0.49) |
| prior | 0.0 | 0/4 | (0.0, 0.49) |
| prior_or_none | 0.0 | 0/4 | (0.0, 0.49) |
| random | 0.0 | 0/4 | (0.0, 0.49) |
| name_probe | 0.0 | 0/4 | (0.0, 0.49) |
| name_probe_thr | 0.0 | 0/4 | (0.0, 0.49) |
| vocab_match | 0.0 | 0/4 | (0.0, 0.49) |

## T3
| solver | pass rate | k/n | Wilson 95% |
|---|---|---|---|
| reference_one_shot | 0.75 | 45/60 | (0.628, 0.842) |
| reference_best_of_5 | 0.967 | 58/60 | (0.886, 0.991) |
| blackbox | 0.017 | 1/60 | (0.003, 0.089) |
| nothing | 0.017 | 1/60 | (0.003, 0.089) |
| always_claim | 0.0 | 0/60 | (-0.0, 0.06) |
| prior | 0.0 | 0/60 | (-0.0, 0.06) |
| prior_or_none | 0.0 | 0/60 | (-0.0, 0.06) |
| random | 0.0 | 0/60 | (-0.0, 0.06) |
| name_probe | 0.0 | 0/60 | (-0.0, 0.06) |
| name_probe_thr | 0.0 | 0/60 | (-0.0, 0.06) |
| vocab_match | 0.017 | 1/60 | (0.003, 0.089) |
