# gpt-6-luna ShiftHunt T1 episodes (redacted)

Luna = gpt-6-luna, effort medium, max-turns 90, max-usd 1.5; sol = gpt-6.1-sol, same effort/caps/prompt. Caps: 150 calls, 3000 fwd, probe_query 4, 3600 s, k=20. Attribute names redacted; naming shown as right/wrong (grade.named_ok).

## Baseline luna (10 eps)
All 10: stop=submitted, valid, first-try submit (0 format rejections), 0 blocked; calls max 49/150, wall max 2050/3600 s, fwd max 2980/3000 (epd35f), ~$0.03/ep, 22-32 turns. Slots: P planted, N null, R/W name right/wrong, removed/tau, n latents, * solved; pq = probe_query texts used; CF = counterfactual pairs; g*d = gradient x latent shift.
| ep (score) | calls/fwd/pq/wall s | err tool+py | slots | method |
|---|---|---|---|---|
| ep0273 (0.33) | 21/893/4/834 | 0+1 | N:FP ; N:ok ; P:R 0.19/0.33 n3 | CF pairs + g*d |
| ep19ca (0.25) | 31/1268/4/1799 | 2+0 | N:FP ; N:FP ; N:FP ; P:R 0.37/0.34 n4* | CF + g*d; no topic-ratio test; 2 pq rejected |
| ep25ba (0.33) | 34/2386/4/219 | 0+0 | N:ok ; P:R 0.12/0.25 n2 ; P:R 0.33/0.34 n2 | CF + g*d; 16 ablate checks |
| ep4139 (0.67) | 33/1868/4/616 | 0+2 | N:ok ; P:R 0.27/0.30 n6 ; P:R 0.36/0.25 n6* | CF + g*d + pq + ablate check |
| ep8272 (0.00) | 27/1950/4/362 | 0+1 | P:R 0.25/0.34 n4 ; P:R 0.23/0.33 n3 ; P:R 0.18/0.25 n2 | sample contrast x g; ablate checks |
| ep9d8c (0.50) | 21/594/0/746 | 1+0 | P:R 0.28/0.33 n20 ; N:ok ; P:R 0.20/0.30 n3 ; N:ok | CF + g*d; no pq/ablate |
| epa676 (0.25) | 33/2099/4/2050 | 0+0 | P:R 0.18/0.33 n4 ; N:ok ; P:R 0.17/0.33 n2 ; P:R 0.14/0.21 n10 | CF + g*d; ablate checks |
| epb0f3 (0.00) | 35/1984/4/268 | 0+1 | P:R 0.19/0.30 n4 ; P:W ; P:R 0.19/0.33 n5 ; P:R 0.24/0.25 n5 | CF + g*d |
| epd35f (0.00) | 49/2980/0/812 | 0+2 | N:FP ; P:W ; P:W | 30 CF latent_means calls + g*d; no pq |
| epf1ad (0.50) | 26/1156/0/343 | 1+4 | N:ok ; P:R 0.09/0.21 n4 ; P:R 0.18/0.34 n3 ; N:ok | sample phi + g*d; no pq/ablate |

- Tool errors 4/320 calls (2 pq over-budget, 2 bad latent_means args); 11 own-code tracebacks.
- Planted naming 19/22 right (p=1.5e-12 vs 1/6); 3 misses all answered "none". Null false claims 5/13 (4 in ep0273/ep19ca, which never compared attribute to topic effect; regex heuristic).
- 17 of 19 right-named slots fail only on removed < tau (median removed/tau 0.68, median 4 of 20 latents; none overshoot; topic_kept >= 0.91).
- Method: all 10 np.load'd out/latent_means*.npy (not just top-10), used probe_gradient, ranked by g*d, latent_examples 10/10, latent_tokens 0/10.

## Aggregates (same 10 instances; opus 2 eps each)
| arm | valid | passed, mean score | slots solved | planted named (p) | null FP | median latents | median calls/fwd/wall s | notes |
|---|---|---|---|---|---|---|---|---|
| luna-high | 10 | 1/10, 0.33 | 12/35 | 20/22 (4e-14) | 5/13 | 5 | 38/2078/1171 | err 4/370; 16/20 right-named under-ablated; one ep 6366 s (>3600) |
| luna-hint | 9 (1 exec() audit) | 1/10, 0.65 | 23/35 | 17/22 (7e-10) | 4/13 | 20 | 30/1194/192 | err 2/331; 3/17 under |
| sol | 10 | 10/10, 1.00 | 35/35 | 22/22 (8e-18) | 0/13 | 20 | 49/2189/213 | err 0/478; 2 runner stop=ended_without_submit but graded submitted |
| opus55 | 20 | 19/20, 0.99 | 69/70 | 44/44 (6e-35) | 0/26 | 20 | 45/1979/610 | err 7/906 |
| haiku45 | 5 (5 audit: /tmp paths) | 0/10, 0.09 | 3/35 | 7/22 (0.06) | 13/13 | 15 | 28/1640/326 | err 11/281 (7 pq over-budget) |

## Best / worst luna
- Best: ep4139 (0.67; other planted slot a near miss .27/.30); ep9d8c (0.5; only run using 20 latents, still .28/.33); epf1ad (0.5; both nulls right, 3-4 latents).
- Worst: ep8272 (0; 3/3 named right, 2-4 latents each); epb0f3 (0; one planted "none", rest under-filled); epd35f (0; 30 CF calls, near fwd cap, two planted "none" plus one null FP). ep19ca: 3/3 nulls falsely claimed.

## Bottom line
- Scaffold failures are negligible for luna: valid first-try submits, no format errors, caps unhit, 1.3% tool errors. (Haiku's audit invalids and hint's one exec() are separate.)
- Interpretation is mostly sound (19/22 named); real interpretation errors: 3 planted missed, 5/13 null false claims.
- Dominant failure is latent-set under-fill: 4 of 20 latents, removed/tau 0.68; sol/opus always submit 20 (1.25). High effort does not fix it; the method hint (use all 20, check removal) lifts slots solved 10/35 to 23/35.
