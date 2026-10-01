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

Hardening from the 2026-10-01 review (docs/HARNESS_VERIFICATION.md section 2):
5. Besides the JSON rendering, every string leaf and dict key is also scanned RAW (unescaped).
   A leak string containing a quote, backslash or newline is escaped in JSON (\" \\ \n) and
   was invisible to a JSON-only scan.
6. Normalisation is Unicode NFKC + casefold (not only lower()), so compatibility forms and
   case variants outside ASCII match too.
7. The canary's random hex tail is a needle on its own: a tool that strips the "RLCANARY-"
   prefix (or prints only the id) still leaks private data.
8. `private_strings` (set by `sandbox.prepare`: the instance directory, the repo path, the
   instance id, the task codename) are leaks anywhere, never exempt: an agent has no
   legitimate way to know them, and they point straight at the answer key. Word-like ones
   (codename, instance id) are matched with whitespace kept (substring match), so
   ordinary prose ("this feature matches ...") does not trip the codename "featurematch".
9. The agent-text exemption is COUNT-based: a leak string is exempt only if the response does
   not contain it more often than the agent's own request did. Otherwise an agent could put
   every candidate answer into one request and switch the leak scan off for that call.
"""
import json
import re
import unicodedata

MIN_LEN = 3
_WS = re.compile(r"\s+")
CANARY_TAIL_MIN = 12


def norm(s):
    return _WS.sub("", unicodedata.normalize("NFKC", str(s)).casefold())


def norm_keep_ws(s):
    return unicodedata.normalize("NFKC", str(s)).casefold()


def _raw_texts(o):
    """Like _texts, but whitespace is kept (for word-like private strings such as the task codename)."""
    return [norm_keep_ws(_render(o)), "\x00".join(norm_keep_ws(x) for x in _leaves(o, []))]


def _private_hit(s, texts, raw_texts):
    """Is private string s present? Word-like strings (no '/' and no whitespace: the codename variants, the instance
    id) are matched as substrings of whitespace-PRESERVING text, so an agent's own phrase such as
    "this feature matches Spanish" is not read as the codename "featurematch" (2026-10-01 false positive reported by
    the FeatureMatch integrator). Path-like strings keep the whitespace-insensitive substring match."""
    n = norm(s)
    if len(n) < MIN_LEN:
        return False
    if "/" not in str(s) and not re.search(r"\s", str(s)):
        k = norm_keep_ws(s)
        return any(k in t for t in raw_texts)
    return any(n in t for t in texts)


def canary_needles(canary):
    """The canary itself plus its random tail (e.g. the 16 hex chars), if long enough to be distinctive."""
    if not canary:
        return []
    out = [norm(canary)]
    tail = re.split(r"[-_]", str(canary))[-1]
    if len(tail) >= CANARY_TAIL_MIN and norm(tail) != out[0]:
        out.append(norm(tail))
    return out


def _leaves(o, acc):
    if isinstance(o, dict):
        for k, v in o.items():
            acc.append(str(k))
            _leaves(v, acc)
    elif isinstance(o, (list, tuple)):
        for v in o:
            _leaves(v, acc)
    elif isinstance(o, str):
        acc.append(o)
    return acc


def _texts(o):
    """Normalised texts to search: the JSON rendering, plus the raw string leaves (separated)."""
    return [norm(_render(o)), "\x00".join(norm(x) for x in _leaves(o, []))]


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


def _agent_texts(agent):
    """agent_text may be the request args (any JSON value) or their JSON rendering (a string)."""
    if agent is None or agent == "":
        return []
    if isinstance(agent, str):
        try:
            agent = json.loads(agent)
        except (ValueError, TypeError):
            return [norm(agent)]
    return _texts(agent)


def scan(obj, canary=None, leak_strings=(), model_output_fields=(), agent_text=None, private_strings=()):
    """Scan a response object.

    agent_text: the agent's own request (rendered args). A leak_string the agent itself sent
    in this call is exempt for this call as long as the response contains it no more often than
    the request did: echoing the agent's own guess back (e.g. a tool that returns the prompt it
    was given, or a format error quoting the submission) gives the agent no new information, and
    flagging it would invalidate exactly the episodes where the agent guessed right. The canary
    and private_strings are never exempt (an agent can only know them by peeking).

    Returns {"leak": bool, "reasons": [str], "behavioral_exposure": int}.
    Reasons never contain the matched text itself (only which needle index fired),
    so logs of leak events cannot themselves leak.
    """
    fields = set(model_output_fields or ())
    outside, inside = _split(obj, fields)
    full_ts = _texts(obj)
    out_ts = _texts(outside)
    in_ts = [norm(_render(v)) for v in inside]
    agent_ts = _agent_texts(agent_text)
    reasons = []
    if any(n in t for n in canary_needles(canary) for t in full_ts):
        reasons.append("canary")
    raw_ts = _raw_texts(obj)
    for i, s in enumerate(private_strings or ()):
        if isinstance(s, str) and _private_hit(s, full_ts, raw_ts):
            reasons.append(f"private_string[{i}]")
    exposure = 0
    for i, s in enumerate(leak_strings or ()):
        if not isinstance(s, str) or len(norm(s)) < MIN_LEN:
            continue
        rx = _needle_rx(s)
        n_out = max(len(rx.findall(t)) for t in out_ts)
        n_agent = max((len(rx.findall(t)) for t in agent_ts), default=0)
        if n_out > n_agent:
            reasons.append(f"leak_string[{i}] outside model-output fields")
        if n_agent == 0:
            for t in in_ts:
                exposure += len(rx.findall(t))
    return {"leak": bool(reasons), "reasons": reasons, "behavioral_exposure": exposure}


def scan_text(text, canary=None, leak_strings=(), private_strings=()):
    """Scan free text (a file or a file name). Returns {"canary": bool, "leak_strings": [indices],
    "private": [indices]}."""
    t = norm(text)
    hits = []
    for i, s in enumerate(leak_strings or ()):
        if isinstance(s, str) and len(norm(s)) >= MIN_LEN and _needle_rx(s).search(t):
            hits.append(i)
    raw = [norm_keep_ws(text)]
    priv = [i for i, s in enumerate(private_strings or ())
            if isinstance(s, str) and _private_hit(s, [t], raw)]
    return {"canary": any(n in t for n in canary_needles(canary)), "leak_strings": hits, "private": priv}
