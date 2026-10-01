"""AUDIT: hand-checks of the grader and the submission-format check on edge-case submissions.

Uses a SYNTHETIC instance (made-up answer key, 4 slots: two planted, two null), so nothing here touches a real answer.
Each case is checked twice: (1) Env.validate_submission (what the broker runs before accepting `submit`), and
(2) grader.py through its CLI in a separate process (what `common.sandbox finish` runs). A case is a problem if a
format-valid submission is graded wrongly (false pass / false fail), or if the grader would pass something the
validator should have rejected.
Run: $PY -m tasks.featurematch.audit.grader_edge_cases
"""
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRUTH = [{"planted": True, "choice": 7, "concept": "x"}, {"planted": False, "choice": "nothing found", "concept": None},
         {"planted": True, "choice": 1, "concept": "y"}, {"planted": False, "choice": "nothing found", "concept": None}]
RIGHT = [7, "nothing found", 1, "nothing found"]


def sub(choices, slots=None):
    slots = slots if slots is not None else list(range(len(choices)))
    return {"answers": [{"slot": s, "choice": c} for s, c in zip(slots, choices)]}


CASES = [  # (name, submission, expected_valid, expected_pass_if_graded, expected_score)
    ("all correct", sub(RIGHT), True, True, 1.0),
    ("option numbers as strings", sub(["7", "nothing found", "1", "nothing found"]), True, True, 1.0),
    ("'nothing found' case/space variants", sub([7, " Nothing Found ", 1, "NOTHING FOUND"]), True, True, 1.0),
    ("slot ids as strings, shuffled order", {"answers": [{"slot": "3", "choice": "nothing found"},
                                                         {"slot": "0", "choice": 7}, {"slot": "2", "choice": 1},
                                                         {"slot": "1", "choice": "nothing found"}]}, True, True, 1.0),
    ("all 'nothing found' (null action)", sub(["nothing found"] * 4), True, False, 0.5),
    ("always claim option 1", sub([1, 1, 1, 1]), True, False, 0.25),
    ("claim on a null slot", sub([7, 3, 1, "nothing found"]), True, False, 0.75),
    ("'nothing found' on a planted slot", sub(["nothing found", "nothing found", 1, "nothing found"]), True, False, 0.75),
    ("wrong option on planted", sub([8, "nothing found", 1, "nothing found"]), True, False, 0.75),
    ("missing a slot", sub([7, "nothing found", 1]), False, False, 0.75),
    ("duplicate slot, conflicting", {"answers": sub(RIGHT)["answers"] + [{"slot": 0, "choice": 8}]}, False, False, 0.75),
    ("slot out of range", {"answers": sub(RIGHT)["answers"][:3] + [{"slot": 4, "choice": "nothing found"}]}, False,
     False, 0.75),
    ("choice 0", sub([0, "nothing found", 1, "nothing found"]), False, False, 0.75),
    ("choice 21", sub([21, "nothing found", 1, "nothing found"]), False, False, 0.75),
    ("choice True (bool)", sub([7, "nothing found", True, "nothing found"]), False, True, 1.0),
    ("choice 7.0 (float)", sub([7.0, "nothing found", 1, "nothing found"]), False, True, 1.0),
    ("choice '7.0' (string)", sub(["7.0", "nothing found", 1, "nothing found"]), False, False, 0.75),
    ("choice null", sub([None, "nothing found", 1, "nothing found"]), False, False, 0.75),
    ("choice 'none'", sub([7, "none", 1, "nothing found"]), False, False, 0.75),
    ("choice option LABEL text", sub(["article about a volcano", "nothing found", 1, "nothing found"]), False, False, 0.75),
    ("choice list [7, 8] (hedging)", sub([[7, 8], "nothing found", 1, "nothing found"]), False, False, 0.75),
    ("slot 0.9 (float, truncates to 0)", {"answers": [{"slot": 0.9, "choice": 7}] + sub(RIGHT)["answers"][1:]}, True,
     True, 1.0),
    ("fullwidth digit '７'", sub(["７", "nothing found", 1, "nothing found"]), True, True, 1.0),
    ("nbsp inside 'nothing found'", sub([7, "nothing found", 1, "nothing found"]), False, False, 0.75),
    ("empty answers", {"answers": []}, False, False, 0.0),
    ("wrapped {'answer': {...}}", {"answer": sub(RIGHT)}, False, True, 1.0),
    ("not an object", [1, 2, 3], False, False, 0.0),
    ("extra keys per answer", {"answers": [dict(a, confidence=0.9) for a in sub(RIGHT)["answers"]]}, True, True, 1.0),
]


def main():
    from tasks.featurematch.tools import Env
    problems, rows = [], []
    with tempfile.TemporaryDirectory() as td:
        inst = {"instance_id": "fm-t9-synthetic", "task": "featurematch", "tier": "T1", "canary": "RLCANARY-x",
                "planted": True, "answer": {"slots": TRUTH}, "leak_strings": ["abcdef"], "caps": {}, "extra": {}}
        pub = {"model": "m", "sae": "s", "n_slots": 4, "slots_text": "",
               "slots": [{"slot": i, "layer": 12, "latent": 100 + i, "options": [f"opt{k}" for k in range(20)]}
                         for i in range(4)]}
        json.dump(inst, open(os.path.join(td, "instance.json"), "w"))
        json.dump(pub, open(os.path.join(td, "public.json"), "w"))
        env = Env(instance_dir=td, sandbox_dir=None, profile="full")
        for name, s, exp_valid, exp_pass, exp_score in CASES:
            err = env.validate_submission(s)
            sp, gp = os.path.join(td, "s.json"), os.path.join(td, "g.json")
            json.dump(s, open(sp, "w"))
            subprocess.run([sys.executable, os.path.join(HERE, "grader.py"), "--instance-dir", td, "--submission", sp,
                            "--out", gp], check=True, stdout=subprocess.DEVNULL)
            g = json.load(open(gp))
            valid = err is None
            row = {"case": name, "valid": valid, "validator_error": err, "pass": g["pass"], "score": g["score"],
                   "null_fp": g["details"].get("null_fp")}
            rows.append(row)
            if valid != exp_valid or (g["pass"] != exp_pass) or abs(g["score"] - exp_score) > 1e-9:
                problems.append(row)
            if valid and g["pass"] and s != sub(RIGHT) and exp_pass is False:
                problems.append(dict(row, note="FALSE PASS"))
        # missing submission file entirely
        gp = os.path.join(td, "g2.json")
        subprocess.run([sys.executable, os.path.join(HERE, "grader.py"), "--instance-dir", td, "--submission",
                        os.path.join(td, "nope.json"), "--out", gp], check=True, stdout=subprocess.DEVNULL)
        g = json.load(open(gp))
        rows.append({"case": "no submission file", "valid": None, "pass": g["pass"], "score": g["score"]})
        if g["pass"] or g["score"] != 0:
            problems.append(rows[-1])
    for r in rows:
        print(json.dumps(r, ensure_ascii=False))
    print("PROBLEMS:", json.dumps(problems, ensure_ascii=False, indent=1) if problems else "none")


if __name__ == "__main__":
    main()
