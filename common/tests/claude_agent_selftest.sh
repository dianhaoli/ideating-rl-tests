#!/bin/bash
# Live self-test of the OS-isolated Claude Code launcher (common/claude_agent.py) on the GPU-free _demo task.
#   1. unit tests (fake claude inside the real jail, isolated test broker)
#   2. scripted isolation red-team against the REAL ~/rlsbx broker (no LLM): probes run via bwrap in the production
#      jail configuration; a decoy episode's sandbox must be invisible; then the same probes WITHOUT the jail
#      (control) to show the probes detect real access
#   3. one real Haiku episode: prepare -> isolated run -> finish (graded + transcript-audited)
# Cost: one short Haiku episode (about $0.05-0.10 list price, billed to the operator's Claude Code subscription).
# Usage: bash common/tests/claude_agent_selftest.sh [model=haiku]
set -euo pipefail
cd "$(dirname "$0")/../.."
source ~/ideating-rl-tests/common/env.sh >/dev/null 2>&1
MODEL=${1:-haiku}
TS=$(date -u +%Y%m%d-%H%M%S)
RD=runs/_demo/${TS}_claude_iso_selftest
INST=tasks/_demo/instances
[ -d "$INST" ] || $PY tasks/_demo/generate.py --out "$INST" --n 6 --seed 11 >/dev/null
pick() { python3 -c "import glob,json,sys;print([d for d in sorted(glob.glob(sys.argv[1]+'/*')) if json.load(open(d+'/instance.json'))['planted']==(sys.argv[2]=='1')][0])" "$INST" "$1"; }
PLANTED=$(pick 1)
NULL=$(pick 0)
mkdir -p "$RD"
SCOPE="systemd-run --user --scope -q -p MemoryMax=4G -p MemorySwapMax=2G --"

echo "== 1. unit tests"
$SCOPE $PY -m pytest -q common/tests/test_claude_agent.py 2>&1 | tail -2

echo "== 2. isolation red-team (real broker, production jail config)"
A=$($PY -m common.sandbox prepare --task _demo --instance-dir "$NULL" --profile full --run-dir "$RD" --solver-label redteam | head -1)
B=$($PY -m common.sandbox prepare --task _demo --instance-dir "$PLANTED" --profile full --run-dir "$RD" --solver-label decoy | head -1)
mkdir -p "$RD/redteam"
set +e
$PY -m common.claude_agent redteam --episode "$A" --decoy "$B" --out "$RD/redteam" > "$RD/redteam/redteam.txt" 2>&1
RT=$?
set -e
tail -1 "$RD/redteam/redteam.txt"
$PY -m common.claude_agent redteam --episode "$A" --decoy "$B" --out "$RD/redteam" --control > "$RD/redteam/control.txt" 2>&1 || true
python3 - "$RD/redteam" <<'EOF'
import json, sys
j = json.load(open(sys.argv[1] + "/redteam.json")); c = json.load(open(sys.argv[1] + "/redteam_control.json"))
jail_ok = {r["probe"] for r in j["results"] if r["pass"]}
detecting = sorted(r["probe"] for r in c["results"] if not r["pass"] and r["probe"] in jail_ok)
print(f"jail: {len(jail_ok)}/{j['n_probes']} probes contained; control (no jail): {c['n_escaped']}/{c['n_probes']} "
      f"probes escape -> {len(detecting)} probes demonstrably detect access that the jail removes")
EOF
for E in "$A" "$B"; do $PY -m common.sandbox finish --episode "$E" >/dev/null 2>&1 || true; done
[ "$RT" = 0 ] || { echo "RED-TEAM FAILED (see $RD/redteam/redteam.json)"; exit 1; }

echo "== 3. live $MODEL episode (isolated)"
C=$($PY -m common.sandbox prepare --task _demo --instance-dir "$PLANTED" --profile full --run-dir "$RD" --solver-label "claude-iso-$MODEL" | head -1)
OUT=$RD/episodes/$C
$SCOPE $PY -m common.claude_agent run --episode "$C" --task _demo --prompt-file "$OUT/agent_prompt.txt" --out "$OUT" \
    --model "$MODEL" --effort high --max-turns 30 --max-wall-s 900
M=$(python3 -c "import json;print(json.load(open('$OUT/api_meta.json'))['model'])")
$PY -m common.sandbox finish --episode "$C" --transcript "$OUT/api_transcript.jsonl" --agent-model "claude-cli:$M:high"
python3 - "$OUT" <<'EOF'
import json, sys
o = sys.argv[1]; m = json.load(open(o + "/api_meta.json")); g = json.load(open(o + "/grade.json")); h = g["harness"]
print(json.dumps({"pass": g.get("pass"), "score": g.get("score"), "valid": h["valid"], "invalid": h["invalid_reasons"],
                  "audit_violations": h["n_audit_violations"], "stop": m["stop"], "turns": m["turns"],
                  "usd_cli_estimate": m["usd"], "wall_s": m["wall_s"], "api_proxy_requests": m["api_proxy_requests"],
                  "api_proxy_denied": len(m["api_proxy_denied"]), "net_denied": len(m["net_denied"]),
                  "credential_found_in": m["credential_found_in"], "stream_integrity": m["stream_integrity"],
                  "tools": m["init"][0]["tools"]}, indent=1))
sys.exit(0 if h["valid"] and not m["credential_found_in"] and m["stream_integrity"]["ok"] else 1)
EOF
echo "self-test OK: $RD"
