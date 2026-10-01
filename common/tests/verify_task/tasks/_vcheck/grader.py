"""Fixture grader: behaviour chosen by the submission's "mode" (tests grader-failure handling)."""
import argparse
import json
import math
import os
import sys
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance-dir", required=True)
    ap.add_argument("--submission", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    with open(a.submission) as f:
        sub = json.load(f)
    mode = (sub or {}).get("mode", "none")
    if mode == "crash":
        raise RuntimeError("grader crashed")
    if mode == "hang":
        time.sleep(600)
    if mode == "exit0_no_output":
        sys.exit(0)
    out = {"ok": {"score": 1.0, "pass": True, "details": {"pid": os.getpid()}},
           "none": {"score": 0.0, "pass": False, "details": {}},
           "string_score": {"score": "high", "pass": True, "details": {}},
           "nan_score": {"score": math.nan, "pass": True, "details": {}},
           "big_score": {"score": 7.0, "pass": True, "details": {}},
           "nonbool_pass": {"score": 1.0, "pass": "yes", "details": {}},
           "list_details": {"score": 1.0, "pass": True, "details": [1, 2]},
           "not_dict": [1, 2, 3]}[mode]
    with open(a.out, "w") as f:
        json.dump(out, f)


if __name__ == "__main__":
    main()
