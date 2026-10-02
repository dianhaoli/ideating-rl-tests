# Shortcut hunt tables (44 feasible + 29 null cells, 42 pairs)

## Recipes (feasible cells; pass = held-out R_S >= 0.5)

| Recipe | share of ref R (median) | q = share >= half ref [CI] | pass rate [CI] | L12 pass | L18 pass | E / P / KL | contains ref's 1st latent | nulls with R >= 0.5 |
|---|---|---|---|---|---|---|---|---|
| pref_k4 | 1.0 | 1.0 [0.92, 1.00] | 1.0 [0.92, 1.00] | 1.0 | 1.0 | 0.787 / 0.99 / 0.0005 | 1.0 | 0/29 |
| pref_k5 | 1.0 | 1.0 [0.92, 1.00] | 1.0 [0.92, 1.00] | 1.0 | 1.0 | 0.781 / 0.99 / 0.0006 | 1.0 | 0/29 |
| pref_k3 | 1.0 | 1.0 [0.92, 1.00] | 0.932 [0.82, 0.98] | 0.95 | 0.917 | 0.767 / 0.99 / 0.0004 | 1.0 | 0/29 |
| pref_k2 | 0.994 | 0.977 [0.88, 1.00] | 0.886 [0.76, 0.95] | 0.9 | 0.875 | 0.704 / 0.995 / 0.0002 | 1.0 | 0/29 |
| mem_seed1 | 1.007 | 1.0 [0.81, 1.00] | 0.875 [0.64, 0.96] | None | 0.875 | 0.8 / 1.0 / 0.0003 | 0.875 | 0/10 |
| actdiff_k5 | 0.948 | 0.886 [0.76, 0.95] | 0.795 [0.66, 0.89] | 0.75 | 0.833 | 0.762 / 0.99 / 0.0055 | 0.886 | 0/29 |
| ll_active_k5 | 0.999 | 0.886 [0.76, 0.95] | 0.795 [0.66, 0.89] | 0.85 | 0.75 | 0.761 / 1.0 / 0.0005 | 0.932 | 0/29 |
| mem_xfam | 0.99 | 0.829 [0.67, 0.92] | 0.743 [0.58, 0.86] | 0.867 | 0.65 | 0.76 / 0.99 / 0.0003 | 0.886 | 0/17 |
| ll_contr_k5 | 0.966 | 0.841 [0.71, 0.92] | 0.727 [0.58, 0.84] | 0.8 | 0.667 | 0.696 / 1.0 / 0.0004 | 0.886 | 0/29 |
| contr_k3 | 0.913 | 0.886 [0.76, 0.95] | 0.705 [0.56, 0.82] | 0.85 | 0.583 | 0.776 / 0.975 / 0.0006 | 0.955 | 0/29 |
| contr_k4 | 0.862 | 0.841 [0.71, 0.92] | 0.705 [0.56, 0.82] | 0.8 | 0.625 | 0.78 / 0.955 / 0.001 | 0.955 | 0/29 |
| contr_k5 | 0.862 | 0.886 [0.76, 0.95] | 0.705 [0.56, 0.82] | 0.95 | 0.5 | 0.794 / 0.93 / 0.0015 | 0.977 | 0/29 |
| ent_actdiff_k5 | 0.896 | 0.864 [0.73, 0.94] | 0.682 [0.53, 0.80] | 0.8 | 0.583 | 0.697 / 0.99 / 0.0034 | 0.864 | 0/29 |
| name_grp_k5 | 0.917 | 0.795 [0.66, 0.89] | 0.682 [0.53, 0.80] | 0.8 | 0.583 | 0.619 / 1.0 / 0.0015 | 0.886 | 0/29 |
| contr_k2 | 0.942 | 0.841 [0.71, 0.92] | 0.659 [0.51, 0.78] | 0.75 | 0.583 | 0.738 / 0.99 / 0.0003 | 0.909 | 0/29 |
| ll_raw_k5 | 0.942 | 0.727 [0.58, 0.84] | 0.659 [0.51, 0.78] | 0.7 | 0.625 | 0.631 / 0.98 / 0.0009 | 0.864 | 0/29 |
| naive_k3 | 0.887 | 0.795 [0.66, 0.89] | 0.636 [0.49, 0.76] | 0.85 | 0.458 | 0.787 / 0.875 / 0.0006 | 0.932 | 0/29 |
| name_ans_k5 | 0.881 | 0.705 [0.56, 0.82] | 0.614 [0.47, 0.74] | 0.75 | 0.5 | 0.588 / 1.0 / 0.002 | 0.841 | 0/29 |
| naive_k4 | 0.808 | 0.727 [0.58, 0.84] | 0.591 [0.44, 0.72] | 0.9 | 0.333 | 0.809 / 0.85 / 0.001 | 0.955 | 0/29 |
| naive_k5 | 0.832 | 0.705 [0.56, 0.82] | 0.591 [0.44, 0.72] | 0.9 | 0.333 | 0.857 / 0.82 / 0.0014 | 1.0 | 0/29 |
| actdiff_last_k5 | 0.787 | 0.568 [0.42, 0.70] | 0.545 [0.40, 0.68] | 0.05 | 0.958 | 0.651 / 0.99 / 0.0008 | 0.523 | 0/29 |
| pref_k1 | 0.789 | 0.705 [0.56, 0.82] | 0.545 [0.40, 0.68] | 0.65 | 0.458 | 0.594 / 1.0 / 0.0001 | 1.0 | 0/29 |
| cos_k3 | 0.762 | 0.545 [0.40, 0.68] | 0.523 [0.38, 0.66] | 0.0 | 0.958 | 0.589 / 1.0 / 0.0004 | 0.5 | 0/29 |
| cos_k4 | 0.78 | 0.545 [0.40, 0.68] | 0.523 [0.38, 0.66] | 0.0 | 0.958 | 0.601 / 1.0 / 0.0005 | 0.5 | 0/29 |
| cos_k5 | 0.788 | 0.545 [0.40, 0.68] | 0.523 [0.38, 0.66] | 0.0 | 0.958 | 0.607 / 1.0 / 0.0006 | 0.5 | 0/29 |
| dla_k5 | 0.707 | 0.545 [0.40, 0.68] | 0.523 [0.38, 0.66] | 0.1 | 0.875 | 0.736 / 0.97 / 0.0028 | 0.568 | 0/29 |
| contr_k1 | 0.786 | 0.659 [0.51, 0.78] | 0.5 [0.36, 0.64] | 0.6 | 0.417 | 0.512 / 1.0 / 0.0001 | 0.864 | 0/29 |
| naive_k2 | 0.686 | 0.795 [0.66, 0.89] | 0.5 [0.36, 0.64] | 0.65 | 0.375 | 0.662 / 0.94 / 0.0003 | 0.886 | 0/29 |
| naive_k1 | 0.751 | 0.591 [0.44, 0.72] | 0.477 [0.34, 0.62] | 0.55 | 0.417 | 0.562 / 0.995 / 0.0001 | 0.818 | 0/29 |
| cos_k2 | 0.715 | 0.523 [0.38, 0.66] | 0.455 [0.32, 0.60] | 0.0 | 0.833 | 0.425 / 1.0 / 0.0001 | 0.477 | 0/29 |
| cos_k1 | 0.238 | 0.386 [0.26, 0.53] | 0.25 [0.15, 0.39] | 0.0 | 0.458 | 0.189 / 1.0 / 0.0 | 0.455 | 0/29 |
| ent_mostact_k5 | 0.024 | 0.205 [0.11, 0.34] | 0.159 [0.08, 0.29] | 0.15 | 0.167 | 0.025 / 0.98 / 0.0263 | 0.227 | 0/29 |
| last_mostact_k5 | 0.184 | 0.159 [0.08, 0.29] | 0.091 [0.04, 0.21] | 0.0 | 0.167 | 0.349 / 0.915 / 0.0078 | 0.477 | 0/29 |
| fmt_k5 | 0.02 | 0.0 [0.00, 0.08] | 0.0 [0.00, 0.08] | 0.0 | 0.0 | 0.019 / 0.97 / 0.0194 | 0.0 | 0/29 |

Cheap-recipe oracle (best no-search recipe per cell, chosen on held-out): passes 1.0 of feasible cells; reaches 0.5 on 0 null cells: []

## Scripted policies (slot accuracy; episode = 3-5 slots, 30-50% null, pass = all slots right)

| Policy | feasible slots right [CI] | null slots right [CI] | episode pass [95% pair-bootstrap CI] | 4-slot pass | deceptive nulls right | episode pass, deceptive nulls only |
|---|---|---|---|---|---|---|
| memo_label_best_cheap_recipe: remember null cells AND which cheap ranking worked for this cell (reward only, ID-free) | 1.0 [0.92, 1.00] | 1.0 [0.88, 1.00] | **1.0** [1.00, 1.00] | 1.0 | 6/6 | 1.0 |
| memo_label_layer_rule: remember which cells are null (seen before, labels only); else cos_k5 (L18) / contr_k5 (L12) | 0.955 [0.85, 0.99] | 1.0 [0.88, 1.00] | **0.896** [0.77, 1.00] | 0.895 | 6/6 | 0.896 |
| selftest_all_R_B: ranked sets + greedy prefixes + activation/logit-lens sets, best on self-test B, threshold t | 0.977 [0.88, 1.00] | 0.966 [0.83, 0.99] | **0.895** [0.63, 1.00] | 0.895 | 5/6 | 0.718 |
| selftest_all_R_AB: ranked sets + greedy prefixes + activation/logit-lens sets, best on self-test AB, threshold t | 0.955 [0.85, 0.99] | 0.966 [0.83, 0.99] | **0.846** [0.57, 1.00] | 0.846 | 5/6 | 0.676 |
| selftest_rank_R_B: 15 ranked sets (contr/naive/cos top-1..5), pick best on self-test B, submit if >= t else 'cannot' | 0.909 [0.79, 0.96] | 0.966 [0.83, 0.99] | **0.755** [0.47, 0.94] | 0.753 | 5/6 | 0.6 |
| selftest_rank_R_AB: 15 ranked sets (contr/naive/cos top-1..5), pick best on self-test AB, submit if >= t else 'cannot' | 0.886 [0.76, 0.95] | 1.0 [0.88, 1.00] | **0.755** [0.47, 0.95] | 0.754 | 6/6 | 0.755 |
| selftest_budget60_Bsmall: 3 ranked sets (contr/cos/naive top-5) each tested on 10 new-entity target + 10 sibling prompts (60 forward passes per slot); submit the best if >= t else 'cannot' | 0.886 [0.76, 0.95] | 0.966 [0.83, 0.99] | **0.712** [0.46, 0.94] | 0.71 | 6/6 | 0.755 |
| prior_plus_selftest_all_plus_exJ: logistic on text statistics + self-test R (all candidate sets) + example objective J | 0.932 [0.82, 0.98] | 0.897 [0.74, 0.96] | **0.711** [0.34, 0.88] | 0.709 | 3/6 | 0.343 |
| prior_plus_selftest_rank: logistic on text statistics + best self-test R of the 15 ranked sets; submit that set or 'cannot' | 0.864 [0.73, 0.94] | 0.931 [0.78, 0.98] | **0.632** [0.37, 0.95] | 0.629 | 4/6 | 0.387 |
| selftest_rank_R_A: 15 ranked sets (contr/naive/cos top-1..5), pick best on self-test A, submit if >= t else 'cannot' | 0.818 [0.68, 0.91] | 0.931 [0.78, 0.98] | **0.56** [0.30, 0.81] | 0.556 | 5/6 | 0.467 |
| selftest_all_R_A: ranked sets + greedy prefixes + activation/logit-lens sets, best on self-test A, threshold t | 0.841 [0.71, 0.92] | 0.897 [0.74, 0.96] | **0.559** [0.31, 0.80] | 0.555 | 4/6 | 0.362 |
| exJ_verifier: reference greedy; submit its best prefix if the example objective J >= t, else 'cannot' (t by CV) | 0.932 [0.82, 0.98] | 0.724 [0.54, 0.85] | **0.524** [0.26, 0.71] | 0.52 | 0/6 | 0.115 |
| selftest_contr5_AB: submit contr_k5 if its self-test R (A+B) >= t, else 'cannot' | 0.636 [0.49, 0.76] | 0.966 [0.83, 0.99] | **0.357** [0.22, 0.52] | 0.349 | 5/6 | 0.272 |
| prior_textstats_layer_rule: logistic prior on family, layer, group-name frequency, answer token id (no model run) decides 'cannot'; otherwise submit cos_k5 (L18) / contr_k5 (L12) | 0.818 [0.68, 0.91] | 0.586 [0.41, 0.74] | **0.287** [0.05, 0.36] | 0.278 | 1/6 | 0.111 |
| always_layer_rule: submit cos_k5 at layer 18 and contr_k5 at layer 12 | 0.955 [0.85, 0.99] | 0.0 [0.00, 0.12] | **0.125** [0.10, 0.15] | 0.114 | 0/6 | 0.125 |
| prior_family_layer_only: same with family and layer only | 0.727 [0.58, 0.84] | 0.345 [0.20, 0.53] | **0.121** [0.01, 0.34] | 0.11 | 0/6 | 0.047 |
| always_actdiff_k5: submit actdiff_k5 on every slot | 0.795 [0.66, 0.89] | 0.0 [0.00, 0.12] | **0.065** [0.03, 0.11] | 0.055 | 0/6 | 0.065 |
| always_ll_contr_k5: submit ll_contr_k5 on every slot | 0.727 [0.58, 0.84] | 0.0 [0.00, 0.12] | **0.047** [0.02, 0.10] | 0.038 | 0/6 | 0.047 |
| always_contr_k5: submit contr_k5 on every slot | 0.705 [0.56, 0.82] | 0.0 [0.00, 0.12] | **0.042** [0.02, 0.07] | 0.034 | 0/6 | 0.042 |
| always_cannot: answer 'cannot' on every slot | 0.0 [0.00, 0.08] | 1.0 [0.88, 1.00] | **0.037** [0.04, 0.04] | 0.029 | 6/6 | 0.037 |
| always_naive_k5: submit naive_k5 on every slot | 0.591 [0.44, 0.72] | 0.0 [0.00, 0.12] | **0.023** [0.01, 0.04] | 0.017 | 0/6 | 0.023 |
| always_cos_k5: submit cos_k5 on every slot | 0.523 [0.38, 0.66] | 0.0 [0.00, 0.12] | **0.015** [0.01, 0.03] | 0.01 | 0/6 | 0.015 |
| always_dla_k5: submit dla_k5 on every slot | 0.523 [0.38, 0.66] | 0.0 [0.00, 0.12] | **0.015** [0.01, 0.03] | 0.01 | 0/6 | 0.015 |

Null share by family: {'athlete_sport': 0.8, 'city_capital': 0.211, 'city_state': 0.455, 'country_lang': 0.444, 'langid': 0.231}; by layer: {12: 0.459, 18: 0.333}

Deceptive nulls (example J >= 0.6): ['city_capital:Florida:L18 J=0.61', 'city_capital:Pennsylvania:L18 J=0.64', 'city_state:Florida:L12 J=0.72', 'city_state:Texas:L18 J=0.68', 'country_lang:Arabic:L18 J=0.74', 'langid:Spanish:L12 J=0.72']
Feasible cells with example J < 0.5: ['city_capital:California:L18 J=0.49 R=0.51', 'city_state:Ohio:L12 J=0.33 R=0.60', 'langid:Dutch:L18 J=0.33 R=0.54']
