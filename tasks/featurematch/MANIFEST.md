# FeatureMatch MANIFEST: regenerable artifacts (NOT committed; under tasks/featurematch/cache/, instances/, instances_prelim/)

Nothing below is in git (weights, activations and answer keys are gitignored; D6). Each can be regenerated
deterministically with the commands at the bottom. Sizes are bytes; sha256 is truncated to 16 hex chars.
Upstream inputs (downloaded into HF_HOME by huggingface_hub, never copied into the repo):
- google/gemma-2-2b (fp32 safetensors, ~10.5 GB), google/gemma-scope-2b-pt-res layer_{6,12,18}/width_16k/average_l0_{70,82,74}
- datasets: papluca/language-identification (valid.csv, test.csv), DeveloperOats/DBPedia_Classes (DBPEDIA_test.csv, DBPEDIA_val.csv)

| path (relative to tasks/featurematch/) | bytes | sha256[:16] |
|---|---|---|
| cache/concepts.json | 6452993 | cf52ab631f1c31c7 |
| cache/concepts_validation.json | 7343 | 9c098e6aa4f70971 |
| cache/precompute_meta.json | 109558 | dc9b4bb321f6c3d1 |
| cache/acts_L12_A.npy | 304087168 | 7620941859e1bb50 |
| cache/acts_L12_B.npy | 182452352 | f14743cd6e3267c6 |
| cache/acts_L12_C.npy | 144179328 | 9fd7560309c0432d |
| cache/acts_L12_names.npy | 22806656 | 7faf700b70f60e03 |
| cache/acts_L18_A.npy | 304087168 | 7165ca90a9051d89 |
| cache/acts_L18_B.npy | 182452352 | 59580d2541663008 |
| cache/acts_L18_C.npy | 144179328 | a4f0fb8ce405c05f |
| cache/acts_L18_names.npy | 22806656 | a07743a967005e3d |
| cache/acts_L6_A.npy | 304087168 | 2c90a9ef2603f5e7 |
| cache/acts_L6_B.npy | 182452352 | c4da61bcb28278ab |
| cache/acts_L6_C.npy | 144179328 | 55b9a6f8c1c47939 |
| cache/acts_L6_names.npy | 22806656 | 1ba48404f6a04da4 |
| cache/auroc_L12_A.npy | 7602304 | 17e332a3c156cbc0 |
| cache/auroc_L12_B.npy | 7602304 | 30a0cb5d9832527a |
| cache/auroc_L18_A.npy | 7602304 | 72024027a00e6c95 |
| cache/auroc_L18_B.npy | 7602304 | de8b55737410804b |
| cache/auroc_L6_A.npy | 7602304 | 49e8d8c6734006ae |
| cache/auroc_L6_B.npy | 7602304 | 8b543a511d08a4f4 |
| cache/fire_L12_A.npy | 7602304 | dcc738427ac5d5c4 |
| cache/fire_L18_A.npy | 7602304 | dc00958be8c3bb43 |
| cache/fire_L6_A.npy | 7602304 | 3e45906ae9f5fb80 |
| cache/firerate_L12_A.npy | 65664 | 78ee608bfe482e50 |
| cache/firerate_L18_A.npy | 65664 | 83e7326c33fbc609 |
| cache/firerate_L6_A.npy | 65664 | 2aceafef5f46f981 |
| cache/gemma-2-2b-bf16/model-00001-of-00003.safetensors | 2496293600 | 1d004ad59e8c6e3f |
| cache/gemma-2-2b-bf16/model-00002-of-00003.safetensors | 2491732040 | c16bb6a23fd40215 |
| cache/gemma-2-2b-bf16/model-00003-of-00003.safetensors | 240691728 | c627b42b163313dd |
| instances/<id>/{instance.json, public.json} (72 dirs, ~10 KB each; the gate + agent pool, seeds 5000+, 105000+, 205000+) | - | per-file hashes and `kept` flags in instances_manifest.json (committed) |
| instances_prelim/<id>/... (180 dirs; stage-1 prelim pool, seeds 1000+, 101000+, 201000+; GATE-ONLY, its null pattern was committed in ac29984) | - | instances_manifest_prelim.json (committed) |
| runs/featurematch/<ts>_gate_*/episodes/, summary.* (harness gate run dirs for the current pool) | ~1-3 MB per run dir | held back by runs/featurematch/.gitignore until the smoke run on this pool is finished (D6); aggregates in <ts>_gates_public.json (committed) |

## Regeneration (from the worktree root, after `source common/env.sh`)
```
$PY -m tasks.featurematch.concepts                       # cache/concepts.json (+ validation); CPU, ~1 min
$PY -m tasks.featurematch.convert_bf16                   # cache/gemma-2-2b-bf16; CPU, ~1 min, ~7.8 GB peak RSS
$PY -m common.gpuq run --gb 8 --heavy --label fm-precompute -- $PY -m tasks.featurematch.precompute   # ~6 min on an L4
$PY -m tasks.featurematch.auroc                          # cache/auroc_*, fire_*; CPU, ~1 min
$PY -m tasks.featurematch.generate --n-per-tier 24 --start-seed 5000 --tiers T1,T2,T3 --clean   # instances/ + instances_manifest.json (current pool); <1 s
# (the stage-1 prelim pool was: --n-per-tier 60 --tiers T1,T2,T3 with the default --start-seed 1000, then moved to instances_prelim/)
$PY -m tasks.featurematch.generate --prior 300           # prior.json (committed; seeds 900000+, disjoint)
```
fp16 activations are deterministic up to GPU kernel nondeterminism; regenerated AUROC tables may differ in the 4th
decimal, which can change a borderline slot. The instance generator asserts its ground-truth invariants on whatever
tables it is given, so regenerated instances are always internally consistent, but they may not be byte-identical.
