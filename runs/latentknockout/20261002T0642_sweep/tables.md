### Best achievable R_S (median over measurable group-level cells)

| Layer | k=1 | k=2 | k=3 | k=5 | k=8 | k=10 |
|---|---|---|---|---|---|---|
| 6 | 0.05 | 0.09 | 0.09 | 0.14 | 0.18 | 0.17 |
| 12 | 0.26 | 0.39 | 0.41 | 0.47 | 0.49 | 0.49 |
| 18 | 0.35 | 0.49 | 0.51 | 0.52 | 0.52 | 0.52 |

### Same with R_T (held-out prompts in the SAME styles as the examples)

| Layer | k=1 | k=2 | k=3 | k=5 | k=8 | k=10 |
|---|---|---|---|---|---|---|
| 6 | 0.03 | 0.06 | 0.09 | 0.11 | 0.16 | 0.15 |
| 12 | 0.21 | 0.35 | 0.43 | 0.53 | 0.52 | 0.52 |
| 18 | 0.29 | 0.42 | 0.47 | 0.50 | 0.51 | 0.51 |

### Components of the reference's k<=5 set (medians)

| Layer | Effect_S | Preserve_S | KL (nats) | Effect_T | Preserve_T |
|---|---|---|---|---|---|
| 6 | 0.16 | 0.98 | 0.0010 | 0.12 | 0.99 |
| 12 | 0.49 | 0.98 | 0.0010 | 0.55 | 1.00 |
| 18 | 0.59 | 0.99 | 0.0000 | 0.52 | 1.00 |

### Share of cells with R_S >= 0.5 / 0.6 / 0.7 (k <= 5)

| | >=0.5 | >=0.6 | >=0.7 |
|---|---|---|---|
| L6 | 0.00 | 0.00 | 0.00 |
| L12 | 0.47 | 0.44 | 0.28 |
| L18 | 0.56 | 0.40 | 0.35 |
| all | 0.41 | 0.34 | 0.26 |

### Median R_S (k<=5) by family and layer; share of cells >= 0.5 in brackets

| family | L6 | L12 | L18 |
|---|---|---|---|
| athlete_sport | 0.11 (0.00) | 0.05 (0.00) | 0.34 (0.33) |
| city_capital | 0.28 (0.00) | 0.61 (0.58) | 0.65 (0.67) |
| city_state | 0.12 (0.00) | 0.45 (0.42) | 0.53 (0.58) |
| country_lang | 0.11 (0.00) | 0.56 (0.60) | 0.35 (0.40) |
| langid | 0.17 (0.00) | 0.65 (0.62) | 0.64 (0.62) |

p_pair = 0.698 (Wilson CI [0.549, 0.814]), p_cell = 0.415, pairs >= 0.7: 0.488, median R_S(k=10) at best layer = 0.686, median R_S(k=5) at best layer = 0.686
feasible groups by family: {'athlete_sport': 2, 'city_capital': 10, 'city_state': 8, 'country_lang': 3, 'langid': 7} of {'athlete_sport': 6, 'city_capital': 12, 'city_state': 12, 'country_lang': 5, 'langid': 8}
cells: {'group_level': 106, 'measurable': 106, 'country_capital': 24, 'unmeasurable': []}

### Baselines on feasible cells (k = 5; ratio = R_b / R_ref)

| baseline | median ratio | q (share with ratio >= 0.5) [Wilson CI] | share ratio >= 0.8 | median R_S | Effect | Preserve | KL |
|---|---|---|---|---|---|---|---|
| clean | 0.00 | 0.00 [0.0, 0.08] | 0.00 | 0.00 | 0.00 | 1.00 | 0.0000 |
| rand_k5_0 | 0.00 | 0.00 [0.0, 0.08] | 0.00 | 0.00 | 0.00 | 1.00 | 0.0000 |
| randact_k5_0 | 0.00 | 0.00 [0.0, 0.08] | 0.00 | 0.00 | 0.00 | 1.00 | 0.0000 |
| mostact_k5 | 0.05 | 0.16 [0.079, 0.294] | 0.02 | 0.04 | 0.05 | 0.97 | 0.0260 |
| cos_k5 | 0.79 | 0.55 [0.401, 0.683] | 0.50 | 0.54 | 0.61 | 1.00 | 0.0010 |
| naive_k5 | 0.83 | 0.70 [0.558, 0.818] | 0.59 | 0.56 | 0.86 | 0.82 | 0.0010 |
| contr_k5 | 0.87 | 0.89 [0.76, 0.95] | 0.68 | 0.62 | 0.79 | 0.93 | 0.0020 |
| steer_tuned | 0.18 | 0.29 [0.182, 0.442] | 0.16 | 0.15 | 0.26 | 0.97 | 0.0260 |
| steer_a1 | 0.09 | 0.27 [0.163, 0.418] | 0.16 | 0.07 | 0.72 | 0.96 | 0.0330 |
| proj | 0.06 | 0.18 [0.095, 0.32] | 0.07 | 0.06 | 0.06 | 0.99 | 0.0020 |
| contr_k50 | 0.56 | 0.57 [0.422, 0.703] | 0.18 | 0.40 | 0.94 | 0.78 | 0.0350 |

all measurable cells, median R_S: {'clean': 0.0, 'rand_k5_0': 0.0, 'rand_k5_1': None, 'randact_k5_0': 0.0, 'randact_k5_1': None, 'mostact_k5': 0.02, 'cos_k5': 0.022, 'naive_k5': 0.245, 'contr_k5': 0.348, 'naive_k10': None, 'contr_k10': None, 'steer_tuned': 0.038, 'steer_a1': 0.012, 'proj': 0.021, 'contr_k50': 0.21, 'ref_k5': 0.383}
all measurable cells, share R_S >= 0.5: {'clean': 0.0, 'rand_k5_0': 0.0, 'randact_k5_0': 0.0, 'mostact_k5': 0.047, 'cos_k5': 0.264, 'naive_k5': 0.264, 'contr_k5': 0.311, 'steer_tuned': 0.113, 'steer_a1': 0.113, 'proj': 0.085, 'contr_k50': 0.123, 'ref_k5': 0.415}

style: {'median_ratio_S_over_T': 1.025, 'share_ratio_lt_07': 0.0, 'median_abs_drop': -0.013, 'by_layer': {'6': None, '12': 0.988, '18': 1.058}, 'naive_ratio': 1.08, 'all_cells_median_R_T_k5': 0.354, 'all_cells_median_R_S_k5': 0.383}
error term: {'6': {'err_share_t': 0.41, 'top50_E_S': 0.35, 'top50_P_S': 0.885, 'top50_KL': 0.054}, '12': {'err_share_t': -0.411, 'top50_E_S': 0.713, 'top50_P_S': 0.86, 'top50_KL': 0.043}, '18': {'err_share_t': -0.246, 'top50_E_S': 0.863, 'top50_P_S': 0.78, 'top50_KL': 0.034}}
held-out / example Effect (reference): 0.8
latent stats: {"ref_k5": {"n": 423, "share_fire_le2_entities": 0.033, "median_entity_coverage": 1.0, "median_target_fire": 0.912, "median_sibling_fire": 0.52, "share_wiki_freq_gt_1pct": 0.482, "share_sibling_fire_gt_target": 0.314}, "naive_k5": {"n": 530, "share_fire_le2_entities": 0.021, "median_entity_coverage": 1.0, "median_target_fire": 0.988, "median_sibling_fire": 0.71, "share_wiki_freq_gt_1pct": 0.538, "share_sibling_fire_gt_target": 0.479}}
median bootstrap CI width of R_S (k=5): 0.175

keep_state: {
 "n": 28,
 "by_layer": {
  "6": {
   "n": 4,
   "ref_R_S": 0.276,
   "ref_P_ks": 0.887,
   "refks_P_ks": 0.9,
   "best_ref_R_ks": 0.263,
   "share_feasible_ks": 0.0,
   "n_feasible_ks": 0,
   "cos_k5_R_ks": 0.0,
   "cos_k5_P_ks": 1.0,
   "naive_k5_R_ks": 0.166,
   "naive_k5_P_ks": 0.812,
   "contr_k5_R_ks": 0.281,
   "contr_k5_P_ks": 0.863,
   "steer_tuned_R_ks": 0.009,
   "steer_tuned_P_ks": 1.0
  },
  "12": {
   "n": 12,
   "ref_R_S": 0.606,
   "ref_P_ks": 0.525,
   "refks_P_ks": 0.525,
   "best_ref_R_ks": 0.317,
   "share_feasible_ks": 0.0,
   "n_feasible_ks": 0,
   "cos_k5_R_ks": 0.0,
   "cos_k5_P_ks": 1.0,
   "naive_k5_R_ks": 0.221,
   "naive_k5_P_ks": 0.538,
   "contr_k5_R_ks": 0.237,
   "contr_k5_P_ks": 0.625,
   "steer_tuned_R_ks": 0.015,
   "steer_tuned_P_ks": 0.975
  },
  "18": {
   "n": 12,
   "ref_R_S": 0.646,
   "ref_P_ks": 0.65,
   "refks_P_ks": 0.738,
   "best_ref_R_ks": 0.396,
   "share_feasible_ks": 0.167,
   "n_feasible_ks": 2,
   "cos_k5_R_ks": 0.218,
   "cos_k5_P_ks": 0.625,
   "cos_k5_q_ks": 1.0,
   "naive_k5_R_ks": 0.112,
   "naive_k5_P_ks": 0.637,
   "naive_k5_q_ks": 0.0,
   "contr_k5_R_ks": 0.289,
   "contr_k5_P_ks": 0.637,
   "contr_k5_q_ks": 0.0,
   "steer_tuned_R_ks": 0.167,
   "steer_tuned_P_ks": 0.45,
   "steer_tuned_q_ks": 0.5
  }
 },
 "pairs_feasible_ks": 0.167,
 "pairs_best_R_ks": {
  "Arizona": 0.304,
  "California": 0.463,
  "Florida": 0.412,
  "Georgia": 0.331,
  "Illinois": 0.606,
  "Massachusetts": 0.494,
  "Michigan": 0.34,
  "New York": 0.384,
  "Ohio": 0.652,
  "Pennsylvania": 0.36,
  "Texas": 0.47,
  "Washington": 0.432
 }
}

keep_state_v2: {
 "12": {
  "n": 12,
  "share_feasible": 0.0,
  "median_best_R_ks": 0.317,
  "median_best_R_ks_k10": 0.322,
  "median_refks2": 0.288,
  "per_group": {
   "Arizona": [
    0.3,
    0.25,
    0.36,
    0.01,
    0.37
   ],
   "California": [
    0.36,
    0.36,
    0.26,
    0.0,
    0.26
   ],
   "Florida": [
    0.41,
    0.39,
    0.36,
    0.0,
    0.35
   ],
   "Georgia": [
    0.33,
    0.33,
    0.23,
    0.02,
    0.23
   ],
   "Illinois": [
    0.38,
    0.34,
    0.12,
    0.0,
    0.1
   ],
   "Massachusetts": [
    0.17,
    0.17,
    0.14,
    0.0,
    0.12
   ],
   "Michigan": [
    0.18,
    0.15,
    0.19,
    0.0,
    0.2
   ],
   "New York": [
    0.38,
    0.38,
    0.26,
    0.0,
    0.28
   ],
   "Ohio": [
    0.43,
    0.43,
    0.38,
    0.0,
    0.37
   ],
   "Pennsylvania": [
    0.2,
    0.2,
    0.06,
    0.0,
    0.19
   ],
   "Texas": [
    0.22,
    0.21,
    0.07,
    0.0,
    0.22
   ],
   "Washington": [
    0.17,
    0.11,
    0.08,
    0.0,
    0.09
   ]
  },
  "ksc_median_R_ks": 0.208,
  "cos_median_R_ks": 0.0,
  "naive_median_R_ks": 0.221,
  "contr_median_R_ks": 0.237,
  "steer_median_R_ks": 0.015
 },
 "18": {
  "n": 12,
  "share_feasible": 0.25,
  "median_best_R_ks": 0.408,
  "median_best_R_ks_k10": 0.42,
  "median_refks2": 0.405,
  "per_group": {
   "Arizona": [
    0.1,
    0.1,
    0.03,
    0.15,
    0.03
   ],
   "California": [
    0.46,
    0.46,
    0.16,
    0.27,
    0.16
   ],
   "Florida": [
    0.14,
    0.14,
    0.14,
    0.17,
    0.14
   ],
   "Georgia": [
    0.12,
    0.12,
    0.02,
    0.12,
    0.02
   ],
   "Illinois": [
    0.61,
    0.61,
    0.17,
    0.55,
    0.11
   ],
   "Massachusetts": [
    0.53,
    0.53,
    0.05,
    0.28,
    0.04
   ],
   "Michigan": [
    0.36,
    0.36,
    0.07,
    0.17,
    0.08
   ],
   "New York": [
    0.31,
    0.3,
    0.14,
    0.4,
    0.15
   ],
   "Ohio": [
    0.67,
    0.67,
    0.15,
    0.65,
    0.15
   ],
   "Pennsylvania": [
    0.38,
    0.38,
    0.18,
    0.16,
    0.17
   ],
   "Texas": [
    0.47,
    0.46,
    0.14,
    0.54,
    0.11
   ],
   "Washington": [
    0.43,
    0.43,
    0.06,
    0.02,
    0.06
   ]
  },
  "ksc_median_R_ks": 0.138,
  "ksc_q": 0.0,
  "cos_median_R_ks": 0.218,
  "cos_q": 1.0,
  "naive_median_R_ks": 0.112,
  "naive_q": 0.0,
  "contr_median_R_ks": 0.289,
  "contr_q": 0.333,
  "steer_median_R_ks": 0.167,
  "steer_q": 0.333
 }
}

memorisation: {
 "n_cells_with_2plus_seeds": 31,
 "median_jaccard_same_cell": 0.25,
 "mean_jaccard_same_cell": 0.363,
 "share_same_first_latent": 0.806,
 "mean_jaccard_different_groups": 0.023,
 "share_pairs_any_overlap_diff_groups": 0.158,
 "R_S_across_seeds_examples": [
  [
   0.245,
   0.191
  ],
  [
   0.428,
   0.071
  ],
  [
   0.848,
   0.848
  ],
  [
   0.595,
   0.271
  ],
  [
   0.012,
   0.0
  ],
  [
   0.177,
   0.137
  ],
  [
   0.775,
   0.8
  ],
  [
   0.304,
   0.224
  ],
  [
   0.194,
   0.254
  ],
  [
   0.869,
   0.965
  ],
  [
   0.631,
   0.575
  ],
  [
   0.613,
   0.746
  ]
 ],
 "median_sd_R_across_seeds": 0.04
}

country_capital: {
 "L12": {
  "n": 12,
  "median_R_S": 0.0,
  "share_ge05": 0.0,
  "share_ge07": 0.0
 },
 "L18": {
  "n": 12,
  "median_R_S": 0.214,
  "share_ge05": 0.083,
  "share_ge07": 0.083
 }
}

pairs at best layer:
  athlete_sport:baseball: {'L': 18, 'R_S': 0.245, 'R_T': 0.033, 'R10': 0.245}
  athlete_sport:basketball: {'L': 18, 'R_S': 0.428, 'R_T': 0.136, 'R10': 0.428}
  athlete_sport:golf: {'L': 18, 'R_S': 0.848, 'R_T': 0.624, 'R10': 0.848}
  athlete_sport:hockey: {'L': 18, 'R_S': 0.595, 'R_T': 0.546, 'R10': 0.595}
  athlete_sport:soccer: {'L': 18, 'R_S': 0.012, 'R_T': 0.0, 'R10': 0.012}
  athlete_sport:tennis: {'L': 18, 'R_S': 0.177, 'R_T': 0.073, 'R10': 0.177}
  city_capital:Arizona: {'L': 18, 'R_S': 0.988, 'R_T': 0.998, 'R10': 0.988}
  city_capital:California: {'L': 12, 'R_S': 0.703, 'R_T': 0.662, 'R10': 0.759}
  city_capital:Florida: {'L': 12, 'R_S': 0.61, 'R_T': 0.53, 'R10': 0.61}
  city_capital:Georgia: {'L': 18, 'R_S': 0.945, 'R_T': 0.992, 'R10': 0.945}
  city_capital:Illinois: {'L': 18, 'R_S': 0.97, 'R_T': 0.882, 'R10': 0.97}
  city_capital:Massachusetts: {'L': 12, 'R_S': 0.959, 'R_T': 0.974, 'R10': 0.959}
  city_capital:Michigan: {'L': 18, 'R_S': 0.411, 'R_T': 0.48, 'R10': 0.411}
  city_capital:New York: {'L': 18, 'R_S': 0.499, 'R_T': 0.439, 'R10': 0.492}
  city_capital:Ohio: {'L': 18, 'R_S': 0.931, 'R_T': 1.0, 'R10': 0.931}
  city_capital:Pennsylvania: {'L': 12, 'R_S': 0.601, 'R_T': 0.708, 'R10': 0.601}
  city_capital:Texas: {'L': 18, 'R_S': 0.587, 'R_T': 0.545, 'R10': 0.587}
  city_capital:Washington: {'L': 18, 'R_S': 0.754, 'R_T': 1.0, 'R10': 0.754}
  city_state:Arizona: {'L': 18, 'R_S': 0.775, 'R_T': 0.567, 'R10': 0.775}
  city_state:California: {'L': 12, 'R_S': 0.486, 'R_T': 0.529, 'R10': 0.486}
  city_state:Florida: {'L': 12, 'R_S': 0.292, 'R_T': 0.284, 'R10': 0.292}
  city_state:Georgia: {'L': 18, 'R_S': 0.869, 'R_T': 0.793, 'R10': 0.869}
  city_state:Illinois: {'L': 12, 'R_S': 0.877, 'R_T': 0.889, 'R10': 0.877}
  city_state:Massachusetts: {'L': 12, 'R_S': 0.857, 'R_T': 0.761, 'R10': 0.857}
  city_state:Michigan: {'L': 18, 'R_S': 0.52, 'R_T': 0.5, 'R10': 0.52}
  city_state:New York: {'L': 18, 'R_S': 0.319, 'R_T': 0.097, 'R10': 0.319}
  city_state:Ohio: {'L': 12, 'R_S': 0.604, 'R_T': 0.521, 'R10': 0.583}
  city_state:Pennsylvania: {'L': 12, 'R_S': 0.686, 'R_T': 0.677, 'R10': 0.686}
  city_state:Texas: {'L': 12, 'R_S': 0.283, 'R_T': 0.309, 'R10': 0.278}
  city_state:Washington: {'L': 18, 'R_S': 0.851, 'R_T': 0.777, 'R10': 0.851}
  country_lang:Arabic: {'L': 12, 'R_S': 0.702, 'R_T': 0.675, 'R10': 0.774}
  country_lang:English: {'L': 18, 'R_S': 0.348, 'R_T': 0.315, 'R10': 0.348}
  country_lang:French: {'L': 18, 'R_S': 0.888, 'R_T': 0.95, 'R10': 0.888}
  country_lang:Portuguese: {'L': 12, 'R_S': 0.56, 'R_T': 0.706, 'R10': 0.56}
  country_lang:Spanish: {'L': 18, 'R_S': 0.285, 'R_T': 0.493, 'R10': 0.315}
  langid:Dutch: {'L': 12, 'R_S': 0.823, 'R_T': 0.865, 'R10': 0.823}
  langid:French: {'L': 18, 'R_S': 0.846, 'R_T': 0.896, 'R10': 0.846}
  langid:German: {'L': 18, 'R_S': 0.276, 'R_T': 0.314, 'R10': 0.276}
  langid:Italian: {'L': 18, 'R_S': 0.938, 'R_T': 0.958, 'R10': 0.938}
  langid:Polish: {'L': 12, 'R_S': 0.632, 'R_T': 0.767, 'R10': 0.632}
  langid:Portuguese: {'L': 12, 'R_S': 0.823, 'R_T': 0.832, 'R10': 0.823}
  langid:Spanish: {'L': 18, 'R_S': 0.803, 'R_T': 0.956, 'R10': 0.803}
  langid:Turkish: {'L': 12, 'R_S': 0.944, 'R_T': 0.867, 'R10': 0.944}
