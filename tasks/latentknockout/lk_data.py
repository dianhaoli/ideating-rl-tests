"""Prompt families for LatentKnockout (feasibility study, rewritten 2026-10-02 06:30Z).

A FAMILY is a relation with a one-token answer (we compare the FIRST token of the answer, with a leading space).
Each family has GROUPS (the "concept" an instance targets: e.g. all Texas cities -> " Texas") and TEMPLATES in
several STYLES (plain, Q/A, dialogue, key-value record, news, few-shot), so held-out prompts can differ from the
examples in style, not just wording (FeatureMatch lesson 1). Every (entity, template) item is validated against the
clean model before use (validate.py): only items whose clean top-1 next token is an accepted answer are kept.

Families:
  city_state      city -> its US state           (group = state)     "Dallas is a city in the state of" -> " Texas"
  city_capital    city -> capital of its state   (group = state)     two-hop; EditHunt geography
  country_lang    country -> official language   (group = language)  "The official language of Peru is" -> " Spanish"
  athlete_sport   athlete -> sport               (group = sport)     "Roger Federer plays the sport of" -> " tennis"
  langid          sentence -> its language       (group = language)  non-factual: language identification
  country_capital country -> capital             (group = ONE country; held-out = new templates/styles only)
"""
import csv
import glob
import os

STATE_CITIES = {
    "Texas": ["Dallas", "Houston", "San Antonio", "El Paso", "Fort Worth", "Lubbock", "Amarillo", "Corpus Christi",
              "Laredo", "Waco", "Plano", "Galveston", "Abilene", "Midland", "Beaumont", "Brownsville", "Odessa",
              "Arlington", "McAllen", "Killeen"],
    "California": ["Los Angeles", "San Francisco", "San Diego", "San Jose", "Fresno", "Oakland", "Bakersfield",
                   "Long Beach", "Anaheim", "Santa Barbara", "Berkeley", "Palo Alto", "Pasadena", "Riverside",
                   "Stockton", "Modesto", "Santa Monica", "Malibu", "Irvine", "Burbank"],
    "Florida": ["Miami", "Orlando", "Tampa", "Jacksonville", "Fort Lauderdale", "St. Petersburg", "Pensacola",
                "Gainesville", "Sarasota", "Key West", "Daytona Beach", "Naples", "Boca Raton", "Clearwater",
                "Fort Myers", "Palm Beach", "Ocala", "Hialeah", "Lakeland", "Melbourne"],
    "New York": ["Buffalo", "Rochester", "Syracuse", "Yonkers", "Ithaca", "Schenectady", "Utica", "Binghamton",
                 "Poughkeepsie", "White Plains", "Niagara Falls", "Saratoga Springs", "Troy", "Brooklyn", "Long Island",
                 "Plattsburgh", "Elmira", "New Rochelle", "Hempstead", "Cooperstown"],
    "Ohio": ["Cleveland", "Cincinnati", "Toledo", "Akron", "Dayton", "Youngstown", "Canton", "Sandusky",
             "Lorain", "Mansfield", "Parma", "Kent", "Hamilton", "Findlay", "Zanesville", "Lima", "Marietta",
             "Mentor", "Elyria", "Athens"],
    "Pennsylvania": ["Philadelphia", "Pittsburgh", "Allentown", "Erie", "Scranton", "Bethlehem", "Lancaster",
                     "Reading", "Wilkes-Barre", "Altoona", "State College", "York", "Gettysburg", "Hershey",
                     "Johnstown", "Williamsport", "Chester", "Easton", "Hazleton", "Lebanon"],
    "Michigan": ["Detroit", "Grand Rapids", "Ann Arbor", "Flint", "Kalamazoo", "Dearborn", "Saginaw", "Traverse City",
                 "Marquette", "Muskegon", "Battle Creek", "Pontiac", "Warren", "Holland", "Ypsilanti", "Bay City",
                 "Livonia", "Southfield", "Midland", "Sterling Heights"],
    "Illinois": ["Chicago", "Rockford", "Peoria", "Naperville", "Joliet", "Champaign", "Evanston", "Decatur",
                 "Bloomington", "Elgin", "Carbondale", "Urbana", "Schaumburg", "Moline", "Waukegan", "Cicero",
                 "Oak Park", "Galesburg", "Kankakee", "Danville"],
    "Georgia": ["Savannah", "Augusta", "Macon", "Athens", "Albany", "Valdosta", "Marietta", "Roswell", "Alpharetta",
                "Sandy Springs", "Warner Robins", "Dalton", "Rome", "Gainesville", "Brunswick", "Statesboro",
                "Decatur", "Smyrna", "Kennesaw", "Peachtree City"],
    "Arizona": ["Phoenix", "Tucson", "Mesa", "Scottsdale", "Flagstaff", "Tempe", "Chandler", "Yuma", "Sedona",
                "Glendale", "Prescott", "Gilbert", "Peoria", "Surprise", "Kingman", "Lake Havasu City", "Bisbee",
                "Tombstone", "Nogales", "Casa Grande"],
    "Washington": ["Seattle", "Spokane", "Tacoma", "Bellevue", "Everett", "Yakima", "Bellingham", "Vancouver",
                   "Walla Walla", "Kennewick", "Redmond", "Renton", "Kent", "Pullman", "Wenatchee", "Richland",
                   "Bremerton", "Auburn", "Puyallup", "Port Angeles"],
    "Massachusetts": ["Worcester", "Springfield", "Cambridge", "Lowell", "Salem", "Plymouth", "Quincy", "Lexington",
                      "Concord", "Amherst", "Pittsfield", "Fall River", "New Bedford", "Brockton", "Lynn", "Somerville",
                      "Newton", "Framingham", "Gloucester", "Nantucket"],
}
STATE_CAPITAL = {"Texas": "Austin", "California": "Sacramento", "Florida": "Tallahassee", "New York": "Albany",
                 "Ohio": "Columbus", "Pennsylvania": "Harrisburg", "Michigan": "Lansing", "Illinois": "Springfield",
                 "Georgia": "Atlanta", "Arizona": "Phoenix", "Washington": "Olympia", "Massachusetts": "Boston"}


LANG_COUNTRIES = {
    "Spanish": ["Mexico", "Argentina", "Colombia", "Peru", "Chile", "Venezuela", "Ecuador", "Bolivia", "Cuba",
                "Uruguay", "Paraguay", "Guatemala", "Honduras", "Nicaragua", "Costa Rica", "Panama", "El Salvador",
                "the Dominican Republic", "Spain"],
    "Arabic": ["Egypt", "Saudi Arabia", "Iraq", "Syria", "Jordan", "Libya", "Yemen", "Oman", "Kuwait", "Qatar",
               "Bahrain", "the United Arab Emirates", "Lebanon", "Sudan", "Tunisia", "Algeria", "Morocco", "Mauritania"],
    "French": ["France", "Senegal", "Ivory Coast", "Mali", "Niger", "Burkina Faso", "Guinea", "Benin", "Togo",
               "Gabon", "the Republic of the Congo", "Monaco", "Chad", "the Central African Republic", "Haiti",
               "the Democratic Republic of the Congo"],
    "Portuguese": ["Brazil", "Portugal", "Angola", "Mozambique", "Guinea-Bissau", "Cape Verde",
                   "Sao Tome and Principe", "East Timor"],
    "English": ["the United States", "the United Kingdom", "Australia", "New Zealand", "Jamaica", "Ghana",
                "Nigeria", "Liberia", "the Bahamas", "Barbados", "Trinidad and Tobago", "Zambia", "Zimbabwe",
                "Uganda", "Belize", "Guyana"],
    "German": ["Germany", "Austria", "Liechtenstein"],
    "Russian": ["Russia", "Belarus"],
}

COUNTRY_CAPITAL = {"France": "Paris", "Japan": "Tokyo", "Germany": "Berlin", "Italy": "Rome", "Egypt": "Cairo",
                   "Russia": "Moscow", "Spain": "Madrid", "Greece": "Athens", "Kenya": "Nairobi", "Peru": "Lima",
                   "Thailand": "Bangkok", "Norway": "Oslo", "Poland": "Warsaw", "Argentina": "Buenos Aires",
                   "Austria": "Vienna", "Portugal": "Lisbon", "Ireland": "Dublin", "Cuba": "Havana",
                   "Sweden": "Stockholm", "Turkey": "Ankara", "Iran": "Tehran", "Mexico": "Mexico City",
                   "South Korea": "Seoul", "China": "Beijing", "India": "New Delhi", "Hungary": "Budapest",
                   "Denmark": "Copenhagen", "Finland": "Helsinki", "Chile": "Santiago", "Colombia": "Bogot\u00e1"}

SPORT_ATHLETES = {
    "basketball": ["Michael Jordan", "LeBron James", "Kobe Bryant", "Stephen Curry", "Shaquille O'Neal",
                   "Kevin Durant", "Larry Bird", "Magic Johnson", "Tim Duncan", "Dirk Nowitzki",
                   "Kareem Abdul-Jabbar", "Giannis Antetokounmpo", "Dwyane Wade", "Allen Iverson", "Yao Ming",
                   "Scottie Pippen", "Hakeem Olajuwon", "Kawhi Leonard"],
    "soccer": ["Lionel Messi", "Cristiano Ronaldo", "Neymar", "Pele", "Diego Maradona", "Zinedine Zidane",
               "Kylian Mbappe", "David Beckham", "Thierry Henry", "Wayne Rooney", "Zlatan Ibrahimovic", "Ronaldinho",
               "Mohamed Salah", "Erling Haaland", "Andres Iniesta", "Franz Beckenbauer", "Johan Cruyff", "Kaka"],
    "tennis": ["Roger Federer", "Rafael Nadal", "Novak Djokovic", "Serena Williams", "Venus Williams", "Andre Agassi",
               "Pete Sampras", "Andy Murray", "Maria Sharapova", "Steffi Graf", "Bjorn Borg", "John McEnroe",
               "Martina Navratilova", "Billie Jean King", "Jimmy Connors", "Boris Becker", "Carlos Alcaraz"],
    "golf": ["Tiger Woods", "Jack Nicklaus", "Arnold Palmer", "Phil Mickelson", "Rory McIlroy", "Gary Player",
             "Jordan Spieth", "Ernie Els", "Vijay Singh", "Greg Norman", "Annika Sorenstam", "Ben Hogan",
             "Seve Ballesteros", "Bubba Watson", "Dustin Johnson", "Nick Faldo"],
    "baseball": ["Babe Ruth", "Derek Jeter", "Mickey Mantle", "Hank Aaron", "Willie Mays", "Barry Bonds",
                 "Ken Griffey Jr.", "Shohei Ohtani", "Ted Williams", "Lou Gehrig", "Mike Trout", "Alex Rodriguez",
                 "Jackie Robinson", "Joe DiMaggio", "Roberto Clemente", "Cal Ripken Jr.", "Sandy Koufax"],
    "hockey": ["Wayne Gretzky", "Sidney Crosby", "Mario Lemieux", "Alexander Ovechkin", "Bobby Orr", "Gordie Howe",
               "Connor McDavid", "Jaromir Jagr", "Mark Messier", "Maurice Richard", "Steve Yzerman",
               "Patrick Roy", "Martin Brodeur", "Bobby Hull"],
}

LANGID_CODES = {"French": "fr", "Spanish": "es", "German": "de", "Italian": "it", "Portuguese": "pt",
                "Dutch": "nl", "Polish": "pl", "Turkish": "tr"}



# ----------------------------------------------------------------------------------------------- templates
# Every template is (style, text). Styles:
#   P plain completion     Q question/answer     D dialogue / chat      K key-value record / form
#   N news-like sentence   F few-shot list (demo answers are never answers of any group in the family)
# Split used by the feasibility study (same for every family):
#   example templates (what the agent/reference sees):   P1, P2, Q1          (indices 0, 1, 3)
#   held-out T (new template, SAME styles as examples):   P3, Q2              (indices 2, 4)
#   held-out S (new STYLES never in the examples):        D1 D2 K1 K2 N1 N2 F1 F2 (indices 5..12)
EX_T, HO_T, HO_S = [0, 1, 3], [2, 4], [5, 6, 7, 8, 9, 10, 11, 12]

CITY_STATE_TEMPLATES = [
    ("P", "{e} is a city in the state of"),
    ("P", "{e} is located in the US state of"),
    ("P", "The city of {e} lies in the state of"),
    ("Q", "Q: Which US state is {e} in?\nA:"),
    ("Q", "Question: In which state is the city of {e}?\nAnswer:"),
    ("D", "A: I just got back from a trip to {e}.\nB: Oh nice! Which state is that in?\nA: It's in"),
    ("D", "User: Where is {e}?\nAssistant: {e} is a city in the US state of"),
    ("K", "city: {e}\nstate:"),
    ("K", "Location record\nCity name: {e}\nState name:"),
    ("N", "Officials in {e} said on Tuesday that the governor of"),
    ("N", "Residents of {e} voted on Tuesday in the statewide race for governor of"),
    ("F", "Denver is in Colorado.\nPortland is in Oregon.\nNashville is in Tennessee.\n{e} is in"),
    ("F", "Memphis, Tennessee\nDenver, Colorado\nBoise, Idaho\n{e},"),
]
CITY_CAPITAL_TEMPLATES = [
    ("P", "The state containing {e} has its capital in"),
    ("P", "{e} is in a state whose capital city is"),
    ("P", "The capital of the state where {e} is located is"),
    ("Q", "Q: What is the capital of the state that {e} is in?\nA:"),
    ("Q", "Question: What is the state capital for residents of {e}?\nAnswer:"),
    ("D", "A: I live in {e}.\nB: Cool. So what's your state capital?\nA: It's"),
    ("D", "User: I'm in {e}. Which city is my state's capital?\nAssistant: Your state capital is"),
    ("K", "city: {e}\ncapital of its state:"),
    ("K", "Record\nCity: {e}\nState capital:"),
    ("N", "Lawmakers from {e} traveled to the state capital,"),
    ("N", "After the flood in {e}, the governor held a press conference in the state capital,"),
    ("F", "City -> capital of its state\nMemphis -> Nashville\nBoulder -> Denver\nEugene -> Salem\n{e} ->"),
    ("F", "The state capital for Memphis is Nashville. The state capital for Boulder is Denver. "
          "The state capital for {e} is"),
]
COUNTRY_LANG_TEMPLATES = [
    ("P", "The official language of {e} is"),
    ("P", "In {e}, the official language is"),
    ("P", "The main language spoken in {e} is"),
    ("Q", "Q: What language do people speak in {e}?\nA:"),
    ("Q", "Question: What is the official language of {e}?\nAnswer:"),
    ("D", "A: I'm moving to {e} next month.\nB: Exciting! What language do they speak there?\nA: Mostly"),
    ("D", "User: I'm visiting {e}. What language should I learn?\nAssistant: You should learn"),
    ("K", "country: {e}\nofficial language:"),
    ("K", "Country profile\nName: {e}\nLanguage:"),
    ("N", "The president of {e} addressed the nation on Monday in"),
    ("N", "Schools in {e} teach most classes in"),
    ("F", "Japan: Japanese\nItaly: Italian\nIran: Persian\n{e}:"),
    ("F", "In Japan people speak Japanese. In Italy people speak Italian. In {e} people speak"),
]
ATHLETE_TEMPLATES = [
    ("P", "{e} is famous for playing the sport of"),
    ("P", "The sport played by {e} is"),
    ("P", "{e} became a legend in the sport of"),
    ("Q", "Q: What sport does {e} play?\nA:"),
    ("Q", "Question: Which sport is {e} known for?\nAnswer:"),
    ("D", "A: Did you see {e} play last night?\nB: No, I don't really follow"),
    ("D", "User: Who is {e}?\nAssistant: {e} is a famous professional"),
    ("K", "athlete: {e}\nsport:"),
    ("K", "Player card\nName: {e}\nSport:"),
    ("N", "In sports news, {e} made headlines again in the world of professional"),
    ("N", "Sportswriters ranked {e} among the greatest ever to play"),
    ("F", "Usain Bolt: sprinting\nMichael Phelps: swimming\nSimone Biles: gymnastics\n{e}:"),
    ("F", "Sport of Michael Phelps: swimming\nSport of Simone Biles: gymnastics\nSport of {e}:"),
]
LANGID_TEMPLATES = [
    ("P", "{e}\n\nThe sentence above is written in the language called"),
    ("P", 'Translate the following sentence into English. The original sentence, "{e}", is in'),
    ("P", 'The following text, "{e}", is written in'),
    ("Q", 'Sentence: "{e}"\nQ: Which language is this sentence written in?\nA:'),
    ("Q", 'Question: In what language is "{e}" written?\nAnswer:'),
    ("D", 'A: My friend texted me "{e}" and I can\'t read it.\nB: Oh, that\'s'),
    ("D", 'User: What language is this? "{e}"\nAssistant: That sentence is in'),
    ("K", 'text: "{e}"\nlanguage:'),
    ("K", "Sample\nText: {e}\nDetected language:"),
    ("N", 'A tourist overheard someone say "{e}". The tourist realised the speaker was talking in'),
    ("N", 'The sign at the station read "{e}", which was written in'),
    ("F", 'Text: "The weather is lovely today."\nLanguage: English\nText: "私は毎朝コーヒーを'
          '飲みます。"\nLanguage: Japanese\nText: "{e}"\nLanguage:'),
    ("F", "Hello, how are you? -> English\nこんにちは -> Japanese\n{e} ->"),
]
COUNTRY_CAPITAL_TEMPLATES = [
    ("P", "The capital of {e} is"),
    ("P", "The capital city of {e} is"),
    ("P", "{e} has its capital in"),
    ("Q", "Q: What is the capital of {e}?\nA:"),
    ("Q", "Question: Name the capital city of {e}.\nAnswer:"),
    ("D", "A: I'm flying to {e} next week.\nB: Which city?\nA: The capital,"),
    ("D", "User: What's the capital of {e}?\nAssistant: The capital of {e} is"),
    ("K", "country: {e}\ncapital:"),
    ("K", "Country profile\nName: {e}\nCapital city:"),
    ("N", "Protesters gathered outside the parliament of {e} in"),
    ("N", "Foreign ministers met on Friday in the capital of {e},"),
    ("F", "Canada: Ottawa\nAustralia: Canberra\nVietnam: Hanoi\n{e}:"),
    ("F", "The capital of Canada is Ottawa. The capital of Australia is Canberra. The capital of {e} is"),
]
# demo answers per family (a clean top-1 equal to one of these is a few-shot COPY artefact)
FEWSHOT_DEMO_ANSWERS = {
    "city_state": ["Colorado", "Oregon", "Tennessee", "Idaho"],
    "city_capital": ["Nashville", "Denver", "Salem"],
    "country_lang": ["Japanese", "Italian", "Persian"],
    "athlete_sport": ["sprinting", "swimming", "gymnastics"],
    "langid": ["English", "Japanese"],
    "country_capital": ["Ottawa", "Canberra", "Hanoi"],
}
SPORT_ACCEPT = {"soccer": ["soccer", "football", "footballer"], "hockey": ["hockey", "ice"],
                "golf": ["golf", "golfer"], "tennis": ["tennis"], "basketball": ["basketball"],
                "baseball": ["baseball"]}


def families():
    """{family: dict(groups={group: [entities]}, answers={group: [accepted answer words]},
    templates=[(style, text)], single_entity=bool)}. Answers are compared on the FIRST token of ' word'."""
    from collections import Counter
    fam = {}
    cnt = Counter(c for v in STATE_CITIES.values() for c in v)
    cities = {s: [c for c in v if cnt[c] == 1] for s, v in STATE_CITIES.items()}   # drop names shared by 2 states
    fam["city_state"] = dict(groups=cities, answers={s: [s] for s in STATE_CITIES}, templates=CITY_STATE_TEMPLATES)
    cap_cities = {s: [c for c in v if c != STATE_CAPITAL[s]] for s, v in cities.items()}   # Phoenix -> Phoenix is a copy
    fam["city_capital"] = dict(groups=cap_cities, answers={s: [STATE_CAPITAL[s]] for s in STATE_CITIES},
                               templates=CITY_CAPITAL_TEMPLATES)
    fam["country_lang"] = dict(groups=LANG_COUNTRIES, answers={g: [g] for g in LANG_COUNTRIES},
                               templates=COUNTRY_LANG_TEMPLATES)
    fam["athlete_sport"] = dict(groups=SPORT_ATHLETES, answers={g: SPORT_ACCEPT[g] for g in SPORT_ATHLETES},
                                templates=ATHLETE_TEMPLATES)
    ls = langid_sentences()
    fam["langid"] = dict(groups=ls, answers={g: [g] for g in ls}, templates=LANGID_TEMPLATES)
    fam["country_capital"] = dict(groups={c: [c] for c in COUNTRY_CAPITAL},
                                  answers={c: [COUNTRY_CAPITAL[c]] for c in COUNTRY_CAPITAL},
                                  templates=COUNTRY_CAPITAL_TEMPLATES, single_entity=True)
    for F in fam.values():
        F.setdefault("single_entity", False)
    return fam


def langid_sentences(n_per_lang=24, max_words=14, min_words=5):
    """Short sentences per language from papluca/language-identification (test split), deterministic."""
    hf = os.environ.get("HF_HOME", os.path.expanduser("~/hf_home"))
    p = glob.glob(os.path.join(hf, "hub", "datasets--papluca--language-identification", "snapshots", "*", "test.csv"))[0]
    want = {v: k for k, v in LANGID_CODES.items()}
    out = {k: [] for k in LANGID_CODES}
    with open(p, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            lang = want.get(row["labels"])
            if lang is None or len(out[lang]) >= n_per_lang:
                continue
            t = " ".join(row["text"].replace('"', "'").split())
            w = t.split(" ")
            if len(w) < min_words or any(ch.isdigit() for ch in t) or len(t) > 140:
                continue
            out[lang].append(" ".join(w[:max_words]))
    return out



def unrelated_texts(n=64, n_tokens_chars=260, seed=0):
    """Plain wikitext-103 paragraphs (test split) for the KL control."""
    import pandas as pd
    hf = os.environ.get("HF_HOME", os.path.expanduser("~/hf_home"))
    p = glob.glob(os.path.join(hf, "hub", "datasets--Salesforce--wikitext", "snapshots", "*",
                               "wikitext-103-raw-v1", "test-00000-of-00001.parquet"))[0]
    df = pd.read_parquet(p)
    texts = [t.strip() for t in df["text"] if len(t.strip()) > 300 and not t.strip().startswith("=")]
    import random
    rng = random.Random(seed)
    rng.shuffle(texts)
    return [t[:n_tokens_chars] for t in texts[:n]]
