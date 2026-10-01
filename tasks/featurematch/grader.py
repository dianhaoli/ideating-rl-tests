"""FeatureMatch grader (HARNESS_API.md section 6). Runs out of process; the only code that reads `answer`.

Per slot:
  planted slot  correct  <=> the submitted option number equals the ground-truth option (best held-out AUROC)
  null slot     correct  <=> the submission says "nothing found" (any option = over-claiming = wrong)
score = mean slot correctness (continuous signal)
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
    if isinstance(c, str) and c.strip().lower() == "nothing found":
        return "nothing found"
    try:
        return int(c)
    except Exception:
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
            got[int(a["slot"])] = _norm_choice(a["choice"])
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
    score = sum(s["correct"] for s in slots) / n
    return {"score": score, "pass": all(s["correct"] for s in slots),
            "details": {"slots": slots, "n_slots": n, "null_fp": null_fp,
                        "n_null": sum(not t["planted"] for t in truth)}}


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
    if isinstance(sub, dict) and "answer" in sub and "answers" not in sub:
        sub = sub["answer"]            # tolerate a wrapped submission record
    res = grade(a.instance_dir, sub)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({"score": res["score"], "pass": res["pass"]}))


if __name__ == "__main__":
    main()
