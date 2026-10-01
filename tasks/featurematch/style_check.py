"""Style-transfer check: do latents selected on DBPedia abstracts / language-ID sentences also respond to probe texts
written in a DIFFERENT style, like the ones an LLM agent would write? (Hand-written below, no LLM API.)

For each probed concept, take every planted-eligible latent of that concept (all layers). Score the hand-written probes
(max activation per text) and check whether the latent's best concept, by AUROC over the hand-written set, is the
right one. Prints, per concept, the share of latents that pass that check.
Run: $PY -m common.gpuq run --gb 8 --label fm-style -- $PY -m tasks.featurematch.style_check
"""
import json

import numpy as np

from common import gpuq
gpuq.apply_caps()

from tasks.featurematch.fm_core import LAYERS, Subject   # noqa: E402
from tasks.featurematch.generate import Tables            # noqa: E402

PROBES = {
    "lang:es": ["Mañana vamos a la playa con mis primos si no llueve.", "No me gustó nada la película, era muy lenta.",
                "¿Dónde está la estación de tren más cercana?", "El gobierno anunció nuevas medidas económicas ayer."],
    "lang:de": ["Ich habe gestern den ganzen Tag im Garten gearbeitet.", "Das Paket kam leider zwei Wochen zu spät an.",
                "Kannst du mir bitte das Salz geben?", "Die Stadt plant eine neue Brücke über den Fluss."],
    "lang:ru": ["Вчера мы долго гуляли по парку и разговаривали.", "Этот телефон работает очень медленно.",
                "Где можно купить билеты на концерт?", "Правительство обсуждает новый закон о налогах."],
    "lang:ja": ["昨日は友達と映画を見に行きました。", "この商品はとても使いやすいです。",
                "駅までどうやって行けばいいですか。", "来週から新しい仕事が始まります。"],
    "lang:tr": ["Dün akşam arkadaşlarımla yemeğe çıktık.", "Ürün çok kaliteli ama kargo biraz geç geldi.",
                "Bu otobüs şehir merkezine gidiyor mu?", "Hükümet yeni bir ekonomi paketi açıkladı."],
    "lang:sw": ["Jana tulikwenda sokoni kununua matunda.", "Simu hii ni nzuri sana na bei yake ni nafuu.",
                "Je, unajua njia ya kwenda hospitali?", "Serikali imetangaza mpango mpya wa elimu."],
    "topic:Swimmer": ["Katie Ledecky won the 800 metre freestyle again, breaking her own world record in the pool.",
                      "He swam the 200m butterfly at the national championships and qualified for the relay team.",
                      "The Olympic backstroke champion trains six hours a day in a 50-metre pool.",
                      "She began competitive swimming at age eight and later set a European record in the medley."],
    "topic:Volcano": ["Mount Etna erupted overnight, sending lava flows down its eastern flank.",
                      "The stratovolcano last erupted in 1902; its crater now holds a small acidic lake.",
                      "Ash from the eruption grounded flights across the region for several days.",
                      "Geologists monitor the volcano's magma chamber with seismometers and gas sensors."],
    "topic:Insect": ["The beetle feeds on decaying wood and has a hard, shiny black exoskeleton.",
                     "This species of moth is found in temperate forests; its larvae eat oak leaves.",
                     "Worker ants carry food back to the colony along chemical trails.",
                     "The dragonfly has two pairs of transparent wings and large compound eyes."],
    "topic:Airport": ["The international airport has two runways and handles about twelve million passengers a year.",
                      "A new terminal opened in 2019 with gates for wide-body aircraft.",
                      "Flights depart hourly to the capital, and the airport is served by a rail link.",
                      "The airfield was originally a military air base before conversion to civilian use."],
    "topic:VideoGame": ["The game is a role-playing adventure released for PlayStation and Xbox in 2015.",
                        "Players control a knight who explores dungeons and levels up by defeating monsters.",
                        "The sequel added online multiplayer and was developed by a small indie studio.",
                        "It received praise for its open world, though critics found the boss fights repetitive."],
    "topic:ChessPlayer": ["She became a grandmaster at sixteen and won the national chess championship twice.",
                          "His FIDE rating peaked at 2780 after a strong result at the Candidates tournament.",
                          "The chess prodigy is known for sharp opening preparation in the Sicilian Defence.",
                          "He played board one for his country at the Chess Olympiad."],
}


def auroc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())


def main():
    T = Tables()
    cid2c = {c["cid"]: i for i, c in enumerate(T.cs)}
    subj = Subject()
    concepts = list(PROBES)
    texts = [t for c in concepts for t in PROBES[c]]
    owner = np.array([k for k, c in enumerate(concepts) for _ in PROBES[c]])
    acts = subj.maxpool_acts(texts)
    out = {}
    for k, cid in enumerate(concepts):
        c = cid2c[cid]
        ok, n = 0, 0
        for L in LAYERS:
            for j in T.planted_by_c[L].get(c, []):
                v = acts[L][:, j].astype(np.float32)
                aucs = [auroc(v[owner == q], v[owner != q]) for q in range(len(concepts))]
                n += 1
                ok += int(np.argmax(aucs) == k and aucs[k] >= 0.75)
        out[cid] = {"n_latents": n, "recovered": ok, "rate": round(ok / n, 3) if n else None}
    tot_n = sum(v["n_latents"] for v in out.values())
    tot_ok = sum(v["recovered"] for v in out.values())
    out["_overall"] = {"n_latents": tot_n, "recovered": tot_ok, "rate": round(tot_ok / max(1, tot_n), 3),
                       "median_concept_rate": float(np.median([v["rate"] for v in out.values() if v.get("rate") is not None]))}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
