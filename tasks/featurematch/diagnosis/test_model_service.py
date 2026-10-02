"""CPU-only plumbing test for the shared model service (no model): the socket protocol, the service-mode Env, and
the rule that the per-instance latent permutation stays in the per-episode server (the service sees real ids only).
Fake ops stand in for the model. The numeric equivalence with the real model is checked by compare_service.py.

    /opt/pytorch/bin/python -m pytest -q tasks/featurematch/diagnosis/test_model_service.py
"""
import json
import os
import tempfile
import threading

import numpy as np
import pytest

from tasks.featurematch import model_service as ms
from tasks.featurematch import tools


@pytest.fixture
def fake_service(monkeypatch):
    seen = []

    def token_acts(subj, texts, layer, latents):
        seen.append(("token_acts", layer, list(latents)))
        # activation of real latent r on token t = r + t (so the caller's index mapping is visible)
        return {"acts": [[[float(r + t) for r in latents] for t in range(3)] for _ in texts],
                "tokens": [["a", "b", "c"] for _ in texts]}

    def top_latents(subj, texts, layer, k):
        seen.append(("top_latents", layer, k))
        return {"tokens": [["x", "y"] for _ in texts],
                "top": [[[100 + i, float(10 - i), i % 2] for i in range(k)] + [[5, 0.0, 0]] for _ in texts]}

    def vocab_projection(subj, layer, latent, k):
        seen.append(("vocab_projection", layer, latent))
        return {"top_tokens": [f"t{latent}"] * k, "bottom_tokens": ["b"] * k}

    def generate(subj, prompt, max_new_tokens):
        return {"completion": prompt[::-1][:max_new_tokens]}

    def next_token_logits(subj, prompts, top_k):
        return {"results": [{"top_tokens": ["z"] * top_k, "logprobs": [-0.123456789] * top_k} for _ in prompts]}

    monkeypatch.setattr(ms, "OPS", {"token_acts": token_acts, "top_latents": top_latents,
                                    "vocab_projection": vocab_projection, "generate": generate,
                                    "next_token_logits": next_token_logits})
    d = tempfile.mkdtemp(prefix="fms", dir="/tmp")
    sock = os.path.join(d, "s.sock")
    srv = ms._Server(sock, ms._Handler)
    srv.subj, srv.lock, srv.n_served, srv.busy_s, srv.t0 = None, threading.Lock(), 0, 0.0, 0.0
    threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True).start()
    yield sock, seen
    srv.shutdown()
    srv.server_close()


@pytest.fixture
def instance(tmp_path):
    slots = [{"slot": 0, "layer": 12, "latent": 7, "options": ["a", "b"]}]
    (tmp_path / "public.json").write_text(json.dumps({"n_slots": 1, "slots": slots}))
    (tmp_path / "instance.json").write_text(json.dumps({"extra": {"perm_seed": 4242}}))
    return str(tmp_path)


def _env(instance, sock, monkeypatch):
    monkeypatch.setattr(tools, "SERVICE_SOCK", sock)
    env = tools.Env(instance_dir=instance, sandbox_dir=None, profile="full")
    env.load(instance)
    assert env.backend.remote
    from common.toolserver import make_local_call
    return env, make_local_call(env, {})


def test_ping_and_service_socket(fake_service, monkeypatch):
    sock, _ = fake_service
    assert ms.ping(sock)["ready"]
    monkeypatch.setenv("FM_MODEL_SERVICE", sock)
    assert ms.service_socket() == sock
    monkeypatch.setenv("FM_MODEL_SERVICE", "off")
    assert ms.service_socket() is None
    monkeypatch.setenv("FM_MODEL_SERVICE", "/tmp/no/such.sock")
    assert ms.service_socket() is None              # dead service -> in-process fallback


def test_permutation_stays_in_episode_server(fake_service, instance, monkeypatch):
    sock, seen = fake_service
    env, call = _env(instance, sock, monkeypatch)
    perm, inv = tools.perm_for(4242, 12), np.argsort(tools.perm_for(4242, 12))
    r = call("latent_activations", texts=["hi"], layer=12, latents=[7, 9])
    assert seen[-1] == ("token_acts", 12, [int(perm[7]), int(perm[9])])        # service got REAL ids
    assert r["results"][0]["max"] == {"7": tools.r4(perm[7] + 2), "9": tools.r4(perm[9] + 2)}
    assert r["results"][0]["acts"]["9"] == [tools.r4(perm[9] + t) for t in range(3)]
    r = call("top_latents", texts=["hi", "yo"], layer=12, k=3)
    top = r["results"][0]["top"]
    assert [t["latent"] for t in top] == [int(inv[100]), int(inv[101]), int(inv[102])]   # mapped back
    assert [t["at_token"] for t in top] == ["x", "y", "x"] and len(top) == 3             # v <= 0 dropped
    r = call("vocab_projection", layer=12, latent=7, k=2)
    assert seen[-1] == ("vocab_projection", 12, int(perm[7])) and r["top_tokens"] == [f"t{perm[7]}"] * 2
    assert call("generate", prompt="abc", max_new_tokens=2) == {"completion": "cb"}
    assert call("next_token_logits", prompts=["p"], top_k=2)["results"][0]["logprobs"] == [-0.1235, -0.1235]


def test_service_down_gives_generic_retry_error(instance, monkeypatch):
    env, call = _env(instance, "/tmp/no/such.sock", monkeypatch)
    from common.toolserver import ToolError
    with pytest.raises(ToolError, match="compute temporarily unavailable"):
        call("generate", prompt="abc")


def test_service_error_is_not_echoed(fake_service):
    sock, _ = fake_service
    r = ms._rpc(sock, {"op": "token_acts", "args": {"bogus": "SECRET-TEXT"}}, 5)
    assert r == {"ok": False, "error": "TypeError"}
    assert ms._rpc(sock, {"op": "nope"}, 5)["ok"] is False
