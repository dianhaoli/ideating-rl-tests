"""Concept-defining term per sampled concept (fixed before unblinding): the label's head noun, direct variants and
direct synonyms. Applied identically (case-insensitive) to bank-F and dataset split-A texts."""
import re

TERMS = {
    "topic:TelevisionStation": r"\b(televis|tv\b|station|channel)",
    "topic:HorseRace": r"\b(horse|race\b|races\b|racing|stakes\b|derby)",
    "topic:Publisher": r"\b(publish|press\b|imprint)",
    "topic:Bird": r"\b(bird)",
    "topic:ScreenWriter": r"\b(screenwrit|screen writer|screenplay|script)",
    "topic:ChessPlayer": r"\b(chess)",
    "topic:Galaxy": r"\b(galax)",
    "topic:Mollusca": r"\b(mollus)",
    "topic:Earthquake": r"\b(earthquake|quake|tremor|seism)",
    "topic:RollerCoaster": r"\b(roller ?coaster|coaster)",
    "topic:Anime": r"\b(anime)",
    "topic:Magazine": r"\b(magazine)",
    "topic:Skater": r"\b(skat)",
    "topic:Grape": r"\b(grape)",
    "topic:Election": r"\b(election|elect)",
    "topic:AustralianFootballTeam": r"\b(football|footy|afl\b|vfl\b|sanfl|wafl)",
    "topic:Canoeist": r"\b(cano)",
    "topic:HandballPlayer": r"\b(handball)",
    "topic:Astronaut": r"\b(astronaut|cosmonaut|taikonaut)",
    "topic:Historian": r"\b(historian)",
    "topic:Insect": r"\b(insect)",
    "topic:FormulaOneRacer": r"\b(formula one|formula 1\b|f1\b|grand prix|grands prix)",
    "topic:Fungus": r"\b(fung|mushroom)",
    "topic:Saint": r"\b(saint)",
    "topic:Curler": r"\b(curl)",
    "topic:NationalFootballLeagueSeason": r"\b(nfl\b|national football league|season)",
    "topic:FigureSkater": r"\b(figure skat|skat)",
    "topic:Hotel": r"\b(hotel)",
    "topic:MountainPass": r"\b(pass\b|col\b|notch\b)",
    "topic:Monarch": r"\b(monarch|king|queen|emperor|empress|sultan|shah\b|tsar|czar|reign)",
    "topic:Model": r"\b(model)",
    "topic:SquashPlayer": r"\b(squash)",
    "topic:BaseballPlayer": r"\b(baseball)",
    "topic:AcademicJournal": r"\b(journal)",
    "topic:Cricketer": r"\b(cricket)",
    "topic:AmateurBoxer": r"\b(amateur|box(er|ing)\b)",
    "topic:Pope": r"\b(pope|papa|papacy|pontif)",
    "topic:RadioStation": r"\b(radio|station)",
    "topic:AustralianRulesFootballPlayer": r"\b(footballer|football|footy|afl\b|vfl\b|sanfl|wafl)",
    "topic:BeautyQueen": r"\b(beauty queen|pageant|crown|titleholder)",
}


MISS = re.compile(r"\bMiss [A-Z]")   # "Miss Universe", "Miss Ohio" (case-sensitive), beauty queen only


def has_term(cid, text):
    if re.search(TERMS[cid], text, flags=re.I):
        return True
    return cid == "topic:BeautyQueen" and bool(MISS.search(text))


# encyclopedic lead: a subject phrase then "is/was a/an/the" early in the text ("X (born 1980) is a Dutch ...")
ENC_LEAD = re.compile(r"^[^.!?]{0,160}?\b(is|was|are|were)\s+(a|an|the|one)\b")


def enc_lead(text):
    return bool(ENC_LEAD.search(text))
