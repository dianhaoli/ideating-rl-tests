"""Generic text statistics per group (no model run): how often the group name and the answer word occur in one
wikitext-103 train shard, and the answer's token id (lower id ~ earlier BPE merge ~ more frequent).
Streams the parquet in row batches (low RAM).

usage: python -m tasks.latentknockout.skeptic_shortcuts.fame <out.json>
"""
import glob
import json
import os
import re
import sys

import pyarrow.parquet as pq

from tasks.latentknockout import lk_data


def main():
    fams = lk_data.families()
    words = set()
    for f in ("city_state", "city_capital", "country_lang", "athlete_sport", "langid"):
        F = fams[f]
        for g in F["groups"]:
            words.add(g)
            words.add(F["answers"][g][0])
    hf = os.environ.get("HF_HOME", os.path.expanduser("~/hf_home"))
    p = sorted(glob.glob(os.path.join(hf, "hub", "datasets--Salesforce--wikitext", "snapshots", "*",
                                      "wikitext-103-raw-v1", "train-00000-of-00002.parquet")))[0]
    pat = {w: re.compile(r"\b" + re.escape(w) + r"\b") for w in words}
    cnt = {w: 0 for w in words}
    n_chars = 0
    for b in pq.ParquetFile(p).iter_batches(batch_size=20000, columns=["text"]):
        txt = "\n".join(t for t in b.column(0).to_pylist() if t)
        n_chars += len(txt)
        for w, r in pat.items():
            cnt[w] += len(r.findall(txt))
    from transformers import AutoTokenizer
    from tasks.latentknockout.lk_core import model_path
    tok = AutoTokenizer.from_pretrained(model_path())
    tid = {w: tok(" " + w, add_special_tokens=False)["input_ids"][0] for w in words}
    out = dict(source=os.path.basename(p), n_chars=n_chars, count=cnt, token_id=tid)
    json.dump(out, open(sys.argv[1], "w"), indent=1)
    print(json.dumps(out)[:3000])


if __name__ == "__main__":
    main()
