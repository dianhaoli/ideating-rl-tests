"""Scan STAGED git changes for secrets and oversized/forbidden files. Exit 1 on any hit.

Run before every commit:  python common/secret_scan.py   (or bash common/secret_scan.sh)
Matched secret text is never printed (only file names), so the scan itself cannot leak a secret.
"""
import re
import subprocess
import sys
import os

PATTERNS = [
    r"hf_[A-Za-z0-9]{30,}",
    r"sk-ant-[A-Za-z0-9_\-]{10,}",
    r"AKIA[0-9A-Z]{16}",
    r"aws_secret_access_key\s*[=:]\s*\S{16,}",
    r"ghp_[A-Za-z0-9]{30,}",
    r"github_pat_[A-Za-z0-9_]{30,}",
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
]
SELF = {"common/secret_scan.py", "common/secret_scan.sh"}
FORBIDDEN_EXT = (".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".npz", ".npy", ".pkl", ".pickle", ".gguf", ".onnx")
MAX_BYTES = 5_000_000


def main():
    names = subprocess.run(["git", "diff", "--cached", "--name-only", "--diff-filter=AM"],
                           capture_output=True, text=True, check=True).stdout.split("\n")
    hits = 0
    rx = re.compile("|".join(PATTERNS))
    for f in filter(None, names):
        if f in SELF:
            continue
        diff = subprocess.run(["git", "diff", "--cached", "-U0", "--no-color", "--", f],
                              capture_output=True, text=True, errors="replace").stdout
        added = "\n".join(l[1:] for l in diff.split("\n") if l.startswith("+") and not l.startswith("+++"))
        if rx.search(added):
            print(f"SECRET PATTERN in staged file: {f} (content not shown)")
            hits += 1
        if os.path.isfile(f):
            sz = os.path.getsize(f)
            if sz > MAX_BYTES:
                print(f"LARGE FILE staged: {f} ({sz} bytes)")
                hits += 1
            if f.endswith(FORBIDDEN_EXT) or os.path.basename(f) in (".env", ".hf_env") or f.endswith(".env"):
                print(f"FORBIDDEN TYPE staged: {f}")
                hits += 1
    if hits:
        sys.exit(1)
    print("secret_scan: clean")


if __name__ == "__main__":
    main()
