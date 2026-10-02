"""Print the markdown tables used in FEASIBILITY.md from analyze.py's output (tables.json).
usage: python -m tasks.latentknockout.report_tables <tables.json> [<latent_report.json>]
"""
import json
import sys


def f(x, nd=2):
    return "-" if x is None else (f"{x:.{nd}f}" if isinstance(x, (int, float)) else str(x))


def main():
    T = json.load(open(sys.argv[1]))["T"]
    print("### Best achievable R_S (median over measurable group-level cells)\n")
    print("| Layer | k=1 | k=2 | k=3 | k=5 | k=8 | k=10 |\n|---|---|---|---|---|---|---|")
    for L, d in T["median_R_S_by_layer_k"].items():
        print(f"| {L} | " + " | ".join(f(d[str(k)] if str(k) in d else d.get(k)) for k in (1, 2, 3, 5, 8, 10)) + " |")
    print("\n### Same with R_T (held-out prompts in the SAME styles as the examples)\n")
    print("| Layer | k=1 | k=2 | k=3 | k=5 | k=8 | k=10 |\n|---|---|---|---|---|---|---|")
    for L, d in T["median_R_T_by_layer_k"].items():
        print(f"| {L} | " + " | ".join(f(d[str(k)] if str(k) in d else d.get(k)) for k in (1, 2, 3, 5, 8, 10)) + " |")
    print("\n### Components of the reference's k<=5 set (medians)\n")
    print("| Layer | Effect_S | Preserve_S | KL (nats) | Effect_T | Preserve_T |\n|---|---|---|---|---|---|")
    for L, d in T["components_k5_by_layer"].items():
        print(f"| {L} | {f(d['E_S'])} | {f(d['P_S'])} | {f(d['KL'], 4)} | {f(d['E_T'])} | {f(d['P_T'])} |")
    print("\n### Share of cells with R_S >= 0.5 / 0.6 / 0.7 (k <= 5)\n")
    print("| | >=0.5 | >=0.6 | >=0.7 |\n|---|---|---|---|")
    for k, d in T["share_cells_R_ge"].items():
        print(f"| {k} | " + " | ".join(f(d[t]) for t in ("0.5", "0.6", "0.7")) + " |")
    print("\n### Median R_S (k<=5) by family and layer; share of cells >= 0.5 in brackets\n")
    Ls = list(next(iter(T["median_R_S_k5_family_layer"].values())).keys())
    print("| family | " + " | ".join(Ls) + " |\n|---|" + "---|" * len(Ls))
    for fam, d in T["median_R_S_k5_family_layer"].items():
        sh = T["share_cells_R_ge_by_family"][fam]
        print(f"| {fam} | " + " | ".join(f"{f(d[L])} ({f(sh.get(L))})" for L in Ls) + " |")
    print(f"\np_pair = {T['p_pair']} (Wilson CI {T['p_pair_ci']}), p_cell = {T['p_cell']}, pairs >= 0.7: {T['p_pair_R07']}, "
          f"median R_S(k=10) at best layer = {T['median_R10_best_layer']}, median R_S(k=5) at best layer = {T['median_R5_best_layer']}")
    print(f"feasible groups by family: {T['feasible_groups_by_family']} of {T['pairs_by_family']}")
    print(f"cells: {T['n_cells']}")
    print("\n### Baselines on feasible cells (k = 5; ratio = R_b / R_ref)\n")
    print("| baseline | median ratio | q (share with ratio >= 0.5) [Wilson CI] | share ratio >= 0.8 | median R_S | Effect | Preserve | KL |\n|---|---|---|---|---|---|---|---|")
    for b, d in T["baselines_on_feasible"].items():
        print(f"| {b} | {f(d['median_ratio'])} | {f(d['q_ge_half'])} {d['q_ci']} | {f(d['q_ge_08'])} | {f(d['median_R'])} | {f(d['median_E'])} | {f(d['median_P'])} | {f(d['median_KL'], 4)} |")
    print(f"\nall measurable cells, median R_S: {T['baselines_all_cells_median_R']}")
    print(f"all measurable cells, share R_S >= 0.5: {T['baselines_all_cells_share_R_ge_05']}")
    print(f"\nstyle: {T['style']}")
    print(f"error term: {T['error_term']}")
    print(f"held-out / example Effect (reference): {T['heldout_over_example_effect']}")
    print(f"latent stats: {json.dumps(T['latent_stats'])}")
    print(f"median bootstrap CI width of R_S (k=5): {T['median_ci_width']}")
    for k in ("keep_state", "keep_state_v2", "memorisation", "country_capital"):
        if k in T:
            print(f"\n{k}: {json.dumps(T[k], indent=1)}")
    print("\npairs at best layer:")
    for p, d in sorted(T["pairs_best_layer"].items()):
        print(f"  {p}: {d}")
    if len(sys.argv) > 2:
        R = json.load(open(sys.argv[2]))
        for L, lat in R.items():
            print(f"\n#### Layer {L}: what selected latents fire on")
            for i, d in lat.items():
                ex = "; ".join(f"[{e['source']}] ...{e['context'][-40:]!r} -> {e['token']!r}" for e in d["top_examples"][:3])
                print(f"- {i} (chosen in {d['chosen_in'][:3]}): top tokens {d['top_tokens'][:6]}; wiki rate "
                      f"{d['fire_rate_by_source'].get('wikitext')}; last-token share {d['share_of_prompt_firing_on_last_token']}; {ex}")


if __name__ == "__main__":
    main()
