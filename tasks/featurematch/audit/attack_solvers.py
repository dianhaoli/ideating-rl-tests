"""AUDIT: cheap alternative solvers a clever agent might try instead of real probing work.

Each is a FIXED recipe: it never looks at this instance's internals beyond the slot latent's activations on texts that
are the same (or generated the same way) for every instance, and it uses no privileged file or corpus.

  template_probe  One-size-fits-all probing recipe. Each "article about a X" option gets 4 fixed encyclopedia-style
                  templates with X filled in; each "text written in L" option gets 3 fixed everyday sentences in L
                  (a bank written once by the auditor, the same in every episode). The slot latent's max activation on
                  those texts is compared with a fixed 24-text background set (AUROC, like the reference), the best
                  option is claimed if its AUROC >= 0.78, else "nothing found".
                  WHY it is the main threat: it is what a lazy agent would do in one minute, with zero thought about
                  the specific latent. If it passes > 10% the task does not need interpretability beyond a template.
  self_probe      template_probe to shortlist the top-3 options per slot, then the SUBJECT MODEL writes the probe texts
                  itself (`generate`, greedy, 2 prompts per shortlisted option, <= 30 generate units) and the decision
                  is re-made on those model-written texts. WHY: removes the need for the agent to know anything about
                  the concepts; tests whether "let the model write dataset-like text" is a free lunch.
  density_null    Activation-statistics fingerprint: "nothing found" whenever the latent fires on many generic
                  background texts (dense / non-selective latent, which is what off-universe nulls are) or on none of
                  the template texts; otherwise the template_probe pick. WHY: tests whether nulls are detectable
                  from latent statistics alone (no concept reasoning).

  name_rank2      Exploits the generator's ASYMMETRIC name filter: a planted slot is rejected when the answer's own name
                  is the top name-probe activation, so among kept planted slots the answer is often the SECOND-highest
                  (offline: rank 2 in 44% (T1) / 36% (T2) of planted slots where any name fires). Recipe: run the 3
                  surface variants of each option name (label, bare name, "This is about <name>."), take each option's
                  max, claim the rank-2 option if >= 2 options fire, else "nothing found".

Harness: RL_ATTACK=<variant> python -m common.sandbox run-scripted --task featurematch \
             --solver tasks/featurematch/audit/attack_solvers.py --solver-label audit_<variant> ...
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(HERE))))   # worktree root (run-scripted cwd = sandbox)

VARIANTS = ["template_probe", "self_probe", "density_null", "name_rank2"]
TAU = 0.78
BATCH = 16

LANG_BANK = {
    "Arabic": ["ذهبت إلى السوق صباح اليوم واشتريت بعض الخضار.", "هذا المنتج رائع وأنصح الجميع بشرائه.",
               "متى يبدأ الاجتماع غداً؟"],
    "Bulgarian": ["Вчера ходих на пазара и купих пресни зеленчуци.", "Продуктът е много добър, препоръчвам го.",
                  "Кога започва срещата утре?"],
    "German": ["Heute Morgen bin ich mit dem Fahrrad zur Arbeit gefahren.", "Das Produkt ist gut, aber etwas teuer.",
               "Wann beginnt morgen das Treffen?"],
    "Greek": ["Σήμερα το πρωί πήγα στην αγορά και αγόρασα λαχανικά.", "Το προϊόν είναι πολύ καλό, το προτείνω.",
              "Τι ώρα ξεκινάει η συνάντηση αύριο;"],
    "Spanish": ["Esta mañana fui al mercado y compré verduras frescas.", "El producto es bueno, pero un poco caro.",
                "¿A qué hora empieza la reunión mañana?"],
    "French": ["Ce matin, je suis allé au marché acheter des légumes.", "Le produit est bien, mais un peu cher.",
               "À quelle heure commence la réunion demain ?"],
    "Hindi": ["आज सुबह मैं बाज़ार गया और ताज़ी सब्ज़ियाँ खरीदीं।", "यह उत्पाद बहुत अच्छा है, मैं इसकी सलाह देता हूँ।",
              "कल बैठक कितने बजे शुरू होगी?"],
    "Italian": ["Stamattina sono andato al mercato a comprare la verdura.", "Il prodotto è buono, ma un po' caro.",
                "A che ora inizia la riunione domani?"],
    "Japanese": ["今朝、市場で新鮮な野菜を買いました。", "この商品はとても良いですが、少し高いです。", "明日の会議は何時に始まりますか？"],
    "Dutch": ["Vanochtend ben ik naar de markt gegaan om groenten te kopen.", "Het product is goed, maar wat duur.",
              "Hoe laat begint de vergadering morgen?"],
    "Polish": ["Dziś rano poszedłem na targ i kupiłem świeże warzywa.", "Produkt jest dobry, ale trochę drogi.",
               "O której godzinie zaczyna się jutro spotkanie?"],
    "Portuguese": ["Hoje de manhã fui ao mercado comprar legumes frescos.", "O produto é bom, mas um pouco caro.",
                   "A que horas começa a reunião amanhã?"],
    "Russian": ["Сегодня утром я ходил на рынок и купил свежие овощи.", "Товар хороший, но немного дорогой.",
                "Во сколько завтра начинается встреча?"],
    "Swahili": ["Leo asubuhi nilienda sokoni kununua mboga mbichi.", "Bidhaa hii ni nzuri lakini bei yake ni ghali.",
                "Mkutano utaanza saa ngapi kesho?"],
    "Thai": ["เมื่อเช้านี้ฉันไปตลาดและซื้อผักสด", "สินค้านี้ดีมาก แต่ราคาค่อนข้างแพง", "พรุ่งนี้การประชุมเริ่มกี่โมง"],
    "Turkish": ["Bu sabah pazara gidip taze sebze aldım.", "Ürün güzel ama biraz pahalı.",
                "Yarın toplantı saat kaçta başlıyor?"],
    "Urdu": ["آج صبح میں بازار گیا اور تازہ سبزیاں خریدیں۔", "یہ چیز بہت اچھی ہے لیکن تھوڑی مہنگی ہے۔",
             "کل میٹنگ کتنے بجے شروع ہوگی؟"],
    "Vietnamese": ["Sáng nay tôi đi chợ mua rau tươi.", "Sản phẩm tốt nhưng hơi đắt.", "Ngày mai cuộc họp bắt đầu lúc mấy giờ?"],
    "Chinese": ["今天早上我去市场买了新鲜的蔬菜。", "这个产品很好，但是有点贵。", "明天的会议几点开始？"],
}

BACKGROUND = [
    "The committee met on Tuesday to discuss the budget for next year.",
    "She poured a cup of coffee and opened the window to let in some air.",
    "The bridge was completed in 1932 and spans nearly two kilometres.",
    "Our flight was delayed by three hours because of the storm.",
    "The recipe calls for two eggs, a cup of flour and a pinch of salt.",
    "He was born in a small village and moved to the city as a teenager.",
    "The company reported higher profits in the third quarter.",
    "for i in range(10):\n    print(i * i)",
    "The museum's collection includes paintings, sculptures and old maps.",
    "Please remember to bring your ID card to the exam on Friday.",
    "The river flows south for about 300 kilometres before reaching the sea.",
    "I think the second chapter of the book is much better than the first.",
    "The population of the town was 4,512 at the 2010 census.",
    "The team won the match 3-1 after a strong second half.",
    "Researchers measured the temperature every hour for six weeks.",
    "The song was released as a single and reached number twelve on the charts.",
    "Prices rose by 2.4 percent over the previous twelve months.",
    "The castle was rebuilt several times during the Middle Ages.",
    "Can you send me the file before the meeting tomorrow?",
    "The species is found in tropical forests at low elevations.",
    "The film was shot on location in Morocco and Spain.",
    "He served as a member of the city council from 1998 to 2006.",
    "The software update fixes several bugs and improves battery life.",
    "Heavy rain is expected across the north of the country this weekend.",
]


def bare(label):
    for p in ("article about an ", "article about a ", "text written in "):
        if label.startswith(p):
            return label[len(p):], p.startswith("text")
    return label, False


def template_texts(label):
    x, is_lang = bare(label)
    if is_lang:
        return list(LANG_BANK.get(x, [f"This is a sentence written in {x}."]))
    art = "an" if x[0].lower() in "aeiou" else "a"
    return [f"{art.capitalize()} {x} is described in this encyclopedia article, which covers its history and notable features.",
            f"This page is about a {x}. It was first documented in 1924 and is known throughout the region.",
            f"Among all of the {x}s, this one is perhaps the best known.",
            f"{x[0].upper() + x[1:]}\n\nFrom Wikipedia, the free encyclopedia"]


def gen_prompts(label):
    x, is_lang = bare(label)
    if is_lang:
        return [f"Here is a short everyday sentence written in {x}:\n", f"A customer review, written in {x}:\n"]
    art = "an" if x[0].lower() in "aeiou" else "a"
    return [f"The following is the opening paragraph of a Wikipedia article about {art} {x}.\n\n",
            f"Encyclopedia entry ({x}):\n"]


def auroc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if not len(pos) or not len(neg):
        return 0.5
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())


def measure(call, layer, latent, texts):
    vals = []
    for i in range(0, len(texts), BATCH):
        r = call("latent_activations", texts=texts[i:i + BATCH], layer=layer, latents=[latent], per_token=False)
        vals += [row["max"][str(latent)] for row in r["results"]]
    return vals


def pick(option_vals, bg):
    aucs = [auroc(v, bg) for v in option_vals]
    # tie-break equal AUROCs by mean activation (deterministic)
    key = [a + 1e-6 * float(np.mean(v)) for a, v in zip(aucs, option_vals)]
    order = list(np.argsort(key)[::-1])
    return order, aucs


def solve_name_rank2(call, info):
    answers = []
    for s in info["slots"]:
        texts = []
        for o in s["options"]:
            x, _ = bare(o)
            texts += [o, x, f"This is about {x}."]
        v = measure(call, s["layer"], s["latent"], texts)
        m = np.array([max(v[3 * i:3 * i + 3]) for i in range(len(s["options"]))])
        o = np.argsort(-m, kind="stable")
        answers.append({"slot": s["slot"], "choice": int(o[1]) + 1 if (m > 0).sum() >= 2 else "nothing found"})
    call("submit", answers=answers)
    return {"answers": answers}


def solve(call, variant, seed=0):
    info = call("task_info")
    if variant == "name_rank2":
        return solve_name_rank2(call, info)
    answers, diag = [], []
    for s in info["slots"]:
        L, lat, opts = s["layer"], s["latent"], s["options"]
        texts, owner = [], []
        for i, o in enumerate(opts):
            t = template_texts(o)
            texts += t
            owner += [i] * len(t)
        texts += BACKGROUND
        owner += [-1] * len(BACKGROUND)
        vals = measure(call, L, lat, texts)
        ov = [[v for v, w in zip(vals, owner) if w == i] for i in range(len(opts))]
        bg = [v for v, w in zip(vals, owner) if w == -1]
        order, aucs = pick(ov, bg)
        best = int(order[0])
        choice = best + 1 if aucs[best] >= TAU and max(ov[best]) > 0 else "nothing found"
        if variant == "density_null":
            bg_rate = float(np.mean([v > 0 for v in bg]))
            tmpl_any = any(max(v) > 0 for v in ov)
            if bg_rate >= 0.25 or not tmpl_any:
                choice = "nothing found"
            elif choice == "nothing found":
                choice = best + 1          # otherwise trust the template pick even below TAU
        elif variant == "self_probe":
            short = [int(i) for i in order[:3]]
            gtexts, gowner = [], []
            for i in short:
                for p in gen_prompts(opts[i]):
                    try:
                        c = call("generate", prompt=p, max_new_tokens=48)["completion"].strip()
                    except Exception:
                        c = ""
                    if c:
                        gtexts.append(c)
                        gowner.append(i)
            if gtexts:
                gv = measure(call, L, lat, gtexts)
                gov = {i: [v for v, w in zip(gv, gowner) if w == i] for i in short}
                ga = {i: auroc(gov[i], bg) if gov[i] else 0.0 for i in short}
                gbest = max(short, key=lambda i: (ga[i], np.mean(gov[i]) if gov[i] else 0))
                choice = gbest + 1 if ga[gbest] >= TAU and max(gov[gbest] or [0]) > 0 else "nothing found"
        answers.append({"slot": s["slot"], "choice": choice})
        diag.append({"slot": s["slot"], "best_auc": round(aucs[best], 3), "choice": choice})
    call("submit", answers=answers)
    return {"answers": answers, "diag": diag}


def main():
    from common.toolclient import episode_from_argv
    from tasks.featurematch.reference_solver import client, episode_seed, unwrap
    ep = episode_from_argv()
    variant = os.environ.get("RL_ATTACK", "")
    if variant not in VARIANTS:
        raise SystemExit(f"set RL_ATTACK to one of {VARIANTS}")
    print(json.dumps(solve(unwrap(client(ep).call), variant, seed=episode_seed(ep))))


if __name__ == "__main__":
    main()
