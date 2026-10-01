"""Grader for the harness demo task. The ONLY code that reads instance.json's answer.

    python tasks/_demo/grader.py --instance-dir D --submission S.json --out grade.json
"""
import argparse
import json
import os


def grade(instance_dir, submission):
    with open(os.path.join(instance_dir, "instance.json")) as f:
        inst = json.load(f)
    planted, ans = inst["planted"], inst["answer"]
    claimed = isinstance(submission, dict) and "edited_input" in submission
    if submission is None:
        correct = False
    elif planted:
        correct = claimed and submission.get("edited_input") == ans["edited_input"] and \
            submission.get("new_output") == ans["new_output"]
    else:
        correct = submission == {"nothing_found": True}
    return {"score": 1.0 if correct else 0.0, "pass": bool(correct),
            "details": {"slots": [{"planted": planted, "correct": bool(correct), "claimed": bool(claimed)}],
                        "submitted": submission is not None, "grader_pid": os.getpid()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance-dir", required=True)
    ap.add_argument("--submission", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    with open(a.submission) as f:
        sub = json.load(f)
    with open(a.out, "w") as f:
        json.dump(grade(a.instance_dir, sub), f, indent=1)


if __name__ == "__main__":
    main()
