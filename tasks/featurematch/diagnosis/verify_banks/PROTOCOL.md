# Bank-F on-concept audit: protocol (written before any bank text of the sample was read)

Written 2026-10-02, after `draw_sample.py` drew the sample and before I opened `sample_blinded.json`'s texts.
`unblind_key.json` (sha256 `4781bdb756f57386edc16420aeaaf9ec6ca88d9c826fc7ee85f1474de620359f`) is committed with
this file and is not opened until every text is rated.

## Why
Step 2 (PREREG A1.1) kept 191 of 4741 topic latents (4%). The dropped ones fire on about 95% of their concept's
encyclopedic dataset texts but only about 15% of its styled bank-F texts. "Topic latents are style-fragile" is valid
only if the bank-F texts really are about their concept. If many were off-concept or too vague, the 4% would say
something about the bank, not the latents.

## Sample
40 topic concepts, `random.Random(20261002)`, 20 per group, groups disjoint (see `draw_sample.py`):
- KEPT: the concept anchors >= 1 kept latent (57 candidates).
- DROPLOW: the concept anchors >= 1 dropped latent with fire_F <= 0.2 and no kept latent (53 candidates).
The 40 are shuffled into blinded ids b01-b40. I rate each concept's 20 bank-F texts knowing only the concept label
and its taxonomy path.

## Rating of each bank-F text
- **about**: `clear` = the text's main subject is an instance of the class the label names (for "article about a
  curler": a person who curls, or a curler's career or match; for "article about a bridge": a bridge). `loose` = the
  text is in the concept's domain but its main subject is not an instance of the class (the sport of curling in
  general, a spectator at a match, bridges as a policy topic), or the link to the class is only implied. `off` = a
  reader given the label would not connect the text to it, or the text is more about another concept of the
  232-concept list (for example a taxonomy sibling).
- **term**: the label's head noun or a direct variant appears (curler / curling; bridge; mollusc / molluscs).
- **instance**: a specific, proper-named instance of the class appears (a named person, a named bridge, a named
  race, a species name).
- **real**: if an instance is named, whether I recognise it as a real entity (`yes`, `no` = invented or unknown to me,
  `unsure`).

## Bar (fixed now)
**passed = true** if the share of bank-F texts rated `clear` or `loose` is **>= 0.85 in each group** (point estimate;
concept-clustered bootstrap CI reported), **and** DROPLOW's clear-or-loose share is not lower than KEPT's by more than
0.10. If the bar holds but `clear` alone is < 0.60 in a group, the report says the texts are on-topic but often vague.

## Secondary analyses (fixed now)
1. Per group: rates of clear, loose, off, term named, instance named, instance real. 95% CIs by a concept-clustered
   bootstrap (10,000 resamples, `random.Random(20261002)`), plus Wilson intervals on the raw text count (these ignore
   clustering and are shown only for reference).
2. Does a dropped latent's firing depend on my rating? For each sampled concept's dropped pooled latents with
   fire_F <= 0.2, fire rate on texts rated clear-and-named (term or instance) vs the rest, and per style. If those
   latents fire on < 0.30 of clear-and-named texts, vagueness of the bank text is not what silences them.
3. Dataset split A of the same 40 concepts: what kind of evidence makes a text "about" the concept (biography of a
   named instance, encyclopedic register, class term), measured by the same flags on a sample of split-A texts, and
   compared with bank F.
4. Model-side check after unblinding: for each sampled concept, the best bank-F AUROC of any of the 16384 SAE latents
   at each layer (not only the pooled ones). If DROPLOW concepts also have some latent that separates their bank-F
   texts well, the bank texts carry a concept signal that the dataset-selected latents miss.
