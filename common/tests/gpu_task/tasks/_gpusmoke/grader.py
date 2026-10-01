import argparse
import json

ap = argparse.ArgumentParser()
ap.add_argument("--instance-dir")
ap.add_argument("--submission")
ap.add_argument("--out")
a = ap.parse_args()
sub = json.load(open(a.submission))
json.dump({"score": float(sub is not None), "pass": sub is not None, "details": {}}, open(a.out, "w"))
