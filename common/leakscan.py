"""Leak scanner: run on every tool response before the test agent sees it.

What counts as a leak (docs/HARNESS_API.md section 3):
- The instance CANARY anywhere in the response (any field, including model output) -> LEAK.
  Every private file carries the canary, so a canary in a response means private data escaped.
- Any instance LEAK_STRING outside the fields named in Env.MODEL_OUTPUT_FIELDS -> LEAK.
  leak_strings are answer-revealing strings chosen by the task generator (e.g. the planted
  object, a hidden label). Harness- or tool-authored text must never contain them.
- A leak_string INSIDE a model-output field is NOT a leak: the subject model saying something
  is legitimate behavioural evidence (the agent is supposed to observe behaviour). Each such
  occurrence is counted as `behavioral_exposure` so we can see how much the answer was visible
  through behaviour alone.

How matching works (kept deliberately simple):
1. The response is rendered as JSON text (numbers and lists become text, e.g. [14, 35, 41]).
   "Outside" text = the whole response with every model-output field's value replaced by null.
   "Inside" texts = the JSON rendering of each model-output field's value.
2. Both the texts and the needles are normalised: lower-cased, all whitespace removed.
   So "New York" matches "new  york", and "[14,35,41]" matches "[14, 35, 41]".
3. A needle that starts (ends) with a digit only matches where the text has no digit
   immediately before (after) it, so the leak string "14,35" does not fire on "114,356".
4. Needles shorter than MIN_LEN normalised characters are rejected at prepare time
   (they would match almost any output and make the task unusable); pick distinctive strings.

Why lower-case + whitespace-insensitive: a tool that echoes a label with different casing or
pretty-printing would otherwise slip through. Why not fuzzier (e.g. stemming): false
positives would replace legitimate outputs with "internal error" and silently break tasks.
"""
import json
import re

MIN_LEN = 3
_WS = re.compile(r"\s+")


def norm(s):
    return _WS.sub("", str(s).lower())


def _needle_rx(needle):
    n = norm(needle)
    pre = r"(?<![0-9])" if n[:1].isdigit() else ""
    post = r"(?![0-9])" if n[-1:].isdigit() else ""
    return re.compile(pre + re.escape(n) + post)


def check_leak_strings(leak_strings):
    """Return a list of problems with the instance's leak_strings (empty = fine)."""
    bad = []
    for i, s in enumerate(leak_strings or []):
        if not isinstance(s, str):
            bad.append(f"leak_strings[{i}] is not a string")
        elif len(norm(s)) < MIN_LEN:
            bad.append(f"leak_strings[{i}] is shorter than {MIN_LEN} characters after normalisation")
    return bad


def _split(obj, fields):
    """Return (outside_obj, [inside values]) where model-output fields are cut out."""
    inside = []

    def walk(o):
        if isinstance(o, dict):
            out = {}
            for k, v in o.items():
                if k in fields:
                    inside.append(v)
                    out[k] = None
                else:
                    out[k] = walk(v)
            return out
        if isinstance(o, (list, tuple)):
            return [walk(v) for v in o]
        return o

    return walk(obj), inside


def _render(o):
    try:
        return json.dumps(o, ensure_ascii=False, default=str)
    except Exception:
        return str(o)


def scan(obj, canary=None, leak_strings=(), model_output_fields=(), agent_text=None):
    """Scan a response object.

    agent_text: the agent's own request (rendered args). A leak_string the agent itself sent
    in this call is exempt for this call: echoing the agent's own guess back (e.g. a tool that
    returns the prompt it was given, or a format error quoting the submission) gives the agent
    no new information, and flagging it would invalidate exactly the episodes where the agent
    guessed right. The canary is never exempt (an agent can only know it by peeking).

    Returns {"leak": bool, "reasons": [str], "behavioral_exposure": int}.
    Reasons never contain the matched text itself (only which needle index fired),
    so logs of leak events cannot themselves leak.
    """
    fields = set(model_output_fields or ())
    outside, inside = _split(obj, fields)
    full_t = norm(_render(obj))
    out_t = norm(_render(outside))
    in_ts = [norm(_render(v)) for v in inside]
    agent_t = norm(agent_text) if agent_text else None
    reasons = []
    if canary and norm(canary) in full_t:
        reasons.append("canary")
    exposure = 0
    for i, s in enumerate(leak_strings or ()):
        if not isinstance(s, str) or len(norm(s)) < MIN_LEN:
            continue
        rx = _needle_rx(s)
        if agent_t is not None and rx.search(agent_t):
            continue
        if rx.search(out_t):
            reasons.append(f"leak_string[{i}] outside model-output fields")
        for t in in_ts:
            exposure += len(rx.findall(t))
    return {"leak": bool(reasons), "reasons": reasons, "behavioral_exposure": exposure}


def scan_text(text, canary=None, leak_strings=()):
    """Scan free text (a file). Returns {"canary": bool, "leak_strings": [indices that matched]}."""
    t = norm(text)
    hits = []
    for i, s in enumerate(leak_strings or ()):
        if isinstance(s, str) and len(norm(s)) >= MIN_LEN and _needle_rx(s).search(t):
            hits.append(i)
    return {"canary": bool(canary and norm(canary) in t), "leak_strings": hits}
