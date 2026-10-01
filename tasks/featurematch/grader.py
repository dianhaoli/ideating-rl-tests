"""FeatureMatch grader (HARNESS_API.md section 6). Runs out of process; the only code that reads `answer`.

Per slot:
  planted slot  correct  <=> the submitted option number equals the ground-truth option (best held-out AUROC)
  null slot     correct  <=> the submission says "nothing found" (any option = over-claiming = wrong)
score = (planted-slot accuracy) x (null-slot accuracy), an empty factor counting as 1. THIS IS THE CONTINUOUS RL
        REWARD (pass is the binary one). WHY a product (audit MAJOR 1, plan v2 rule 1): the earlier score was the mean
        slot correctness, under which "nothing found" everywhere earned 0.44 (about 40% of slots are null), almost
        half of the reference's 0.98: a strong local optimum for an RL policy, and exactly the under-claiming failure
        null targets are meant to prevent. With the product, "nothing found" everywhere scores 0 unless the episode
        is all-null, "always claim" scores 0 unless it has no null slot, and an agent must get BOTH halves right.
        details.slot_accuracy keeps the old mean as a diagnostic only.
pass  = EVERY slot correct (product rule, plan v2 / D7). WHY product: with 30-50% null slots, "always nothing found"
        and "always claim something" each get some slots right, so an averaged pass rule would reward them; requiring
        all slots makes both zero-effort policies fail unless the null pattern happens to be all-null / no-null.
details.slots = [{"planted": bool, "correct": bool, "claimed": bool}]  (shared convention across tasks)
details.null_fp = number of null slots where something was claimed (slot-level false positives)

CLI: python tasks/featurematch/grader.py --instance-dir D --submission S.json --out grade.json
"""
import argparse
import json
import os


def _norm_choice(c):
    """'nothing found' (any case/spacing), an int, or a string of an int. Everything else (bool, float, None, lists)
    is not an answer. WHY strict (audit MINOR 7): the broker's format check rejects these anyway, but the grader
    should not depend on that (defence in depth); `True` must not count as option 1, nor 7.0 as option 7."""
    if isinstance(c, bool):
        return None
    if isinstance(c, int):
        return c
    if isinstance(c, str):
        if c.strip().lower() == "nothing found":       # exactly the validator's rule (tools.validate_submission)
            return "nothing found"
        try:
            return int(c)
        except ValueError:
            return None
    return None


def grade(instance_dir, submission):
    inst = json.load(open(os.path.join(instance_dir, "instance.json")))
    truth = inst["answer"]["slots"]
    n = len(truth)
    empty = {"score": 0.0, "pass": False,
             "details": {"slots": [{"planted": t["planted"], "correct": False, "claimed": False} for t in truth],
                         "n_slots": n, "null_fp": 0, "reason": "no submission"}}
    if not submission or not isinstance(submission, dict) or not isinstance(submission.get("answers"), list):
        return empty
    got = {}
    for a in submission["answers"]:
        try:
            sl = a["slot"]
            if isinstance(sl, (bool, float)):
                continue                      # 0.9 or True is not a slot id (same rule as validate_submission)
            got[int(sl)] = _norm_choice(a["choice"])
        except Exception:
            continue
    slots, null_fp = [], 0
    for i, t in enumerate(truth):
        c = got.get(i)
        claimed = c is not None and c != "nothing found"
        if t["planted"]:
            correct = claimed and c == t["choice"]
        else:
            correct = c == "nothing found"
            null_fp += int(claimed)
        slots.append({"planted": t["planted"], "correct": bool(correct), "claimed": bool(claimed)})
    pl = [s["correct"] for s in slots if s["planted"]]
    nu = [s["correct"] for s in slots if not s["planted"]]
    acc_p = sum(pl) / len(pl) if pl else 1.0
    acc_n = sum(nu) / len(nu) if nu else 1.0
    return {"score": acc_p * acc_n, "pass": all(s["correct"] for s in slots),
            "details": {"slots": slots, "n_slots": n, "null_fp": null_fp, "n_null": len(nu),
                        "planted_acc": acc_p, "null_acc": acc_n,
                        "slot_accuracy": sum(s["correct"] for s in slots) / n}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance-dir", required=True)
    ap.add_argument("--submission", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sub = None
    if os.path.exists(a.submission):
        try:
            sub = json.load(open(a.submission))
        except Exception:
            sub = None
    # No unwrapping of {"answer": {...}}: the broker stores the submit args as the submission, i.e.
    # {"answers": [...]}; anything else is not a valid submission (audit MINOR 7).
    res = grade(a.instance_dir, sub)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({"score": res["score"], "pass": res["pass"]}))


if __name__ == "__main__":
    main()
