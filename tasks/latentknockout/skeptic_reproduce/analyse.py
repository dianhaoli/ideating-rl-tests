"""CPU analysis of the skeptic reproduction: bootstrap CIs (item-level and entity-cluster), reported-vs-reproduced
comparison, seed stability, and what flipped answers turn into. Own metric code (does not import analyze.py).
usage: python -m tasks.latentknockout.skeptic_reproduce.analyse <repro_dir> <reported_cells_dir> <out.json> [--md out.md]
"""
import glob
import json
import os
import sys
from collections import Counter

import numpy as np

KAPPA = 0.1
B = 1000


def R_of(m, roles, kl, tr, sr):
    t, s = m[roles == tr], m[roles == sr]
    return float((t < 0).mean() * (s > 0).mean() * max(0.0, 1 - kl.mean() / KAPPA))


def boot(m, roles, ents, kl, tr, sr, cluster, seed=0):
    rng = np.random.default_rng(seed)
    it, isb = np.flatnonzero(roles == tr), np.flatnonzero(roles == sr)
    if cluster:
        def groups(ix):
            d = {}
            for i in ix:
                d.setdefault(ents[i], []).append(i)
            return list(d.values())
        gt, gs = groups(it), groups(isb)
    out = []
    for _ in range(B):
        if cluster:
            tt = np.concatenate([gt[j] for j in rng.integers(0, len(gt), len(gt))])
            ss = np.concatenate([gs[j] for j in rng.integers(0, len(gs), len(gs))])
        else:
            tt = it[rng.integers(0, len(it), len(it))]
            ss = isb[rng.integers(0, len(isb), len(isb))]
        kk = kl[rng.integers(0, len(kl), len(kl))]
        out.append((m[tt] < 0).mean() * (m[ss] > 0).mean() * max(0.0, 1 - kk.mean() / KAPPA))
    return [round(float(np.percentile(out, 2.5)), 3), round(float(np.percentile(out, 97.5)), 3)]


def reported_R(cells_dir, fam, g, L, name):
    c = json.load(open(os.path.join(cells_dir, f"{fam}__{g.replace(' ', '_')}__L{L}__s0.json")))
    if name not in c["margins"]:
        return None
    roles = np.array([it[0] for it in c["items"]])
    m = np.array(c["margins"][name], np.float32)
    kl = np.array(c["kl"][name], np.float32)
    return dict(R_S=round(R_of(m, roles, kl, "tS", "sS"), 3), R_T=round(R_of(m, roles, kl, "tT", "sT"), 3))


def jacc(a, b):
    a, b = set(a), set(b)
    return len(a & b) / max(1, len(a | b))


def main():
    rdir, cells_dir, out = sys.argv[1:4]
    md = sys.argv[sys.argv.index("--md") + 1] if "--md" in sys.argv else None
    tok = None
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(os.path.expanduser(
            "~/wt/featurematch/tasks/featurematch/cache/gemma-2-2b-bf16"))
    except Exception:
        pass
    allres = {}
    lines = []
    for f in sorted(glob.glob(os.path.join(rdir, "*.json"))):
        if os.path.basename(f).startswith("exactness"):
            continue
        r = json.load(open(f))
        fam, g, L, mode = r["family"], r["group"], r["layer"], r["mode"]
        key = f"{mode}:{fam}:{g}:L{L}"
        kl_all = {n: np.array(v, np.float32) for n, v in r["kl_per_text"].items()}
        res = dict(clean_check=r["clean_check"], n=r["n"], sets={})
        for name, d in r["sets"].items():
            row = dict(latents=d["latents"], KL=round(d["KL"], 5))
            for tag, (tr, sr) in (("ho", ("tS", "sS")), ("fresh", ("fT", "fS")), ("new", ("nT", "nS"))):
                if tag not in r["items"]:
                    continue
                it = r["items"][tag]
                roles = np.array([x[0] for x in it["rows"]])
                ents = [f"{x[1]}|{x[2]}" for x in it["rows"]]
                m = np.array(it["margins"][it["uid"][name]], np.float32)
                kl = kl_all[name]
                lab = "ho_S" if tag == "ho" else tag
                if (roles == tr).sum() == 0 or (roles == sr).sum() == 0:
                    continue
                row[lab] = dict(R=round(R_of(m, roles, kl, tr, sr), 3),
                                E=round(float((m[roles == tr] < 0).mean()), 3),
                                P=round(float((m[roles == sr] > 0).mean()), 3),
                                ci_item=boot(m, roles, ents, kl, tr, sr, False),
                                ci_entity=boot(m, roles, ents, kl, tr, sr, True),
                                n_t=int((roles == tr).sum()), n_s=int((roles == sr).sum()),
                                n_t_entities=len({e for e, ro in zip(ents, roles) if ro == tr}))
                # stricter effect: the accepted answer is pushed out of the top 3 (rank >= 3), not just off top-1
                rk = np.array(it["rank"][it["uid"][name]])
                Es = float((rk[roles == tr] >= 3).mean())
                row[lab]["E_top3"] = round(Es, 3)
                row[lab]["R_d1"] = round(float((m[roles == tr] < -1).mean()) * row[lab]["P"] * max(0.0, 1 - kl.mean() / KAPPA), 3)
                row[lab]["R_top3"] = round(Es * row[lab]["P"] * max(0.0, 1 - kl.mean() / KAPPA), 3)
                if tag == "ho":
                    row["ho_T"] = dict(R=round(R_of(m, roles, kl, "tT", "sT"), 3))
            if name.startswith("rep_"):
                row["reported"] = reported_R(cells_dir, fam, g, L, name[4:])
            res["sets"][name] = row
        # what do flipped target answers turn into (reported reference set, held-out new styles)
        if "rep_ref_k5" in r["sets"] and "ho" in r["items"]:
            it = r["items"]["ho"]
            u = it["uid"]["rep_ref_k5"]
            roles = np.array([x[0] for x in it["rows"]])
            m = np.array(it["margins"][u], np.float32)
            rk = np.array(it["rank"][u])
            t1 = np.array(it["top1"][u])
            flip = (roles == "tS") & (m < 0)
            if flip.sum():
                c = Counter(int(x) for x in t1[flip])
                res["flips_ref"] = dict(n_flipped=int(flip.sum()),
                                        share_answer_still_top3=round(float((rk[flip] <= 2).mean()), 3),
                                        share_answer_rank2=round(float((rk[flip] == 1).mean()), 3),
                                        top_replacements=[[tok.decode([k]) if tok else k, v] for k, v in c.most_common(8)])
        # seed stability
        seeds = [n for n in ("my_ref_s0", "my_ref_s1", "my_ref_s2") if n in r["sets"]]
        if len(seeds) > 1:
            res["seed_stability"] = dict(
                R_new={n: res["sets"][n].get("new", {}).get("R") for n in seeds},
                R_new_cos={n: res["sets"][n].get("new", {}).get("R") for n in ("my_cos_k5", "my_cos_s1", "my_cos_s2")
                           if n in res["sets"]},
                R_new_contr={n: res["sets"][n].get("new", {}).get("R") for n in ("my_contr_k5", "my_contr_s1", "my_contr_s2")
                             if n in res["sets"]},
                jaccard_ref=[round(jacc(r["sets"][a]["latents"], r["sets"][b]["latents"]), 2)
                             for i, a in enumerate(seeds) for b in seeds[i + 1:]],
                first_latent=[r["sets"][n]["latents"][0] if r["sets"][n]["latents"] else None for n in seeds],
                sets={n: r["sets"][n]["latents"] for n in seeds})
        for k in ("my_ref_trace", "strong", "oracle", "oracle_d0", "oracle_d1", "seed_examples"):
            if k in r:
                res[k] = r[k]
        allres[key] = res
        # one-line summary per set
        lines.append(f"\n### {key}  (clean check: {r['clean_check']['n_my_clean_correct']}/{r['clean_check']['n_items']} "
                     f"items clean-correct in my run; max |margin diff| {r['clean_check']['max_abs_margin_diff_vs_reported']:.2f})")
        lines.append("| set | latents | reported R_S | my R_S [item CI] [entity CI] | my R_T | fresh R [entity CI] | new-entity R | E/P (ho_S) | R_S top-3 effect | R_S delta=1 | fresh R top-3 | fresh R delta=1 |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for name, row in res["sets"].items():
            rep = row.get("reported", {}) or {}
            h = row.get("ho_S", {})
            fr = row.get("fresh", {})
            nw = row.get("new", {})
            lines.append(f"| {name} | {row['latents'][:6]} | {rep.get('R_S', '')} | {h.get('R', '')} {h.get('ci_item', '')} "
                         f"{h.get('ci_entity', '')} | {row.get('ho_T', {}).get('R', '')} | {fr.get('R', '')} {fr.get('ci_entity', '')} | "
                         f"{nw.get('R', '')} | {h.get('E', '')}/{h.get('P', '')} | {h.get('R_top3', '')} | {h.get('R_d1', '')} | {fr.get('R_top3', '')} | {fr.get('R_d1', '')} |")
        if "flips_ref" in res:
            lines.append(f"- flips of rep_ref_k5 (held-out new styles): {res['flips_ref']}")
        if "seed_stability" in res:
            lines.append(f"- seed stability: {res['seed_stability']}")
        for k in ("strong", "oracle", "oracle_d0", "oracle_d1"):
            if k in res:
                lines.append(f"- {k}: {res[k]}")
        lines.append(f"- sizes: {r['n']}")
    json.dump(allres, open(out, "w"), indent=1, default=float)
    if md:
        open(md, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
