"""Prompt families for LatentKnockout (feasibility study).

A FAMILY is a relation with a one-token answer (we compare the FIRST answer token, with a leading space).
Each family has GROUPS (the "concept" an instance targets: e.g. all Texas cities -> " Texas") and TEMPLATES.
Every (entity, template) pair is validated against the clean model before use (see validate.py); only items whose
clean top-1 next token is one of the entity's accepted answers are kept.

Families:
  city_state     city -> its US state            (group = state)          "Dallas is a city in the state of" -> " Texas"
  city_capital   city -> capital of its state    (group = state)          two-hop; EditHunt geography
  country_lang   country -> official language    (group = language)       "The official language of Peru is" -> " Spanish"
  athlete_sport  athlete -> sport                (group = sport)          "Roger Federer plays the sport of" -> " tennis"
  langid         sentence -> its language        (group = language)       non-factual: language identification
  country_capital country -> capital             (group = ONE country; held-out = new templates only)
"""
import csv
import glob
import os

# ----------------------------------------------------------------------------------------------- geography
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

CITY_STATE_TEMPLATES = [
    "{e} is a city in the state of",
    "{e} is located in the US state of",
    "The city of {e} lies in the state of",
    "Q: Which US state is {e} in?\nA:",
    "I flew into {e}, which is in the state of",
]
CITY_CAPITAL_TEMPLATES = [
    "The state containing {e} has its capital in",
    "{e} is in a state whose capital city is",
    "The capital of the state where {e} is located is",
    "Q: What is the capital of the state that {e} is in?\nA:",
    "Driving from {e} to the capital of its state, you arrive in",
]

# ----------------------------------------------------------------------------------------------- countries
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
COUNTRY_LANG_TEMPLATES = [
    "The official language of {e} is",
    "In {e}, the official language is",
    "The main language spoken in {e} is",
    "Q: What language do people speak in {e}?\nA:",
    "Most people in {e} speak the language",
]

COUNTRY_CAPITAL = {"France": "Paris", "Japan": "Tokyo", "Germany": "Berlin", "Italy": "Rome", "Egypt": "Cairo",
                   "Russia": "Moscow", "Spain": "Madrid", "Greece": "Athens", "Kenya": "Nairobi", "Peru": "Lima",
                   "Thailand": "Bangkok", "Norway": "Oslo", "Poland": "Warsaw", "Argentina": "Buenos Aires",
                   "Austria": "Vienna", "Portugal": "Lisbon", "Ireland": "Dublin", "Cuba": "Havana",
                   "Sweden": "Stockholm", "Turkey": "Ankara", "Iran": "Tehran", "Mexico": "Mexico City",
                   "South Korea": "Seoul", "China": "Beijing", "India": "New Delhi", "Hungary": "Budapest",
                   "Denmark": "Copenhagen", "Finland": "Helsinki", "Chile": "Santiago", "Colombia": "Bogot\u00e1"}
COUNTRY_CAPITAL_TEMPLATES = [
    "The capital of {e} is",
    "The capital city of {e} is",
    "Q: What is the capital of {e}?\nA:",
    "{e} has its capital in",
    "The seat of government of {e} is located in",
    "In {e}, the capital is",
    "If you travel to the capital of {e}, you will arrive in",
    "The national capital of {e} is",
    "Everyone knows that the capital of {e} is",
    "Q: Name the capital city of {e}.\nA:",
]

# ----------------------------------------------------------------------------------------------- athletes
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
SPORT_ACCEPT = {"soccer": ["soccer", "football"], "hockey": ["hockey", "ice"]}
ATHLETE_TEMPLATES = [
    "{e} is famous for playing the sport of",
    "The sport played by {e} is",
    "Q: What sport does {e} play?\nA:",
    "{e} became a legend in the sport of",
    "Fans of {e} love watching professional",
]

# ----------------------------------------------------------------------------------------------- language id
LANGID_CODES = {"French": "fr", "Spanish": "es", "German": "de", "Italian": "it", "Portuguese": "pt",
                "Dutch": "nl", "Polish": "pl", "Turkish": "tr"}
LANGID_DEMO = ('Text: "The weather is lovely today, so we are going to the park."\nLanguage: English\n'
               'Text: "私は毎朝コーヒーを飲みます。"\n'
               'Language: Japanese\n')
LANGID_TEMPLATES = [
    LANGID_DEMO + 'Text: "{e}"\nLanguage:',
    '{e}\n\nThe sentence above is written in the language called',
    'Sentence: "{e}"\nQ: Which language is this sentence written in?\nA:',
    'Translate the following sentence into English. The original sentence, "{e}", is in',
    'A tourist overheard someone say "{e}". The tourist realised the speaker was talking in',
]


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


def families():
    """Return {family: dict(groups={group: [entities]}, answers={group: [accepted answer words]},
    templates=[...], single_entity=bool)}"""
    fam = {}
    from collections import Counter
    cnt = Counter(c for v in STATE_CITIES.values() for c in v)
    cities = {s: [c for c in v if cnt[c] == 1] for s, v in STATE_CITIES.items()}   # drop names shared by 2 states
    fam["city_state"] = dict(groups=cities, answers={s: [s.split()[0]] for s in STATE_CITIES},
                             templates=CITY_STATE_TEMPLATES)
    fam["city_capital"] = dict(groups=cities, answers={s: [STATE_CAPITAL[s]] for s in STATE_CITIES},
                               templates=CITY_CAPITAL_TEMPLATES)
    fam["country_lang"] = dict(groups=LANG_COUNTRIES, answers={g: [g] for g in LANG_COUNTRIES},
                               templates=COUNTRY_LANG_TEMPLATES)
    fam["athlete_sport"] = dict(groups=SPORT_ATHLETES,
                                answers={g: SPORT_ACCEPT.get(g, [g]) for g in SPORT_ATHLETES},
                                templates=ATHLETE_TEMPLATES)
    ls = langid_sentences()
    fam["langid"] = dict(groups=ls, answers={g: [g] for g in ls}, templates=LANGID_TEMPLATES)
    fam["country_capital"] = dict(groups={c: [c] for c in COUNTRY_CAPITAL},
                                  answers={c: [COUNTRY_CAPITAL[c]] for c in COUNTRY_CAPITAL},
                                  templates=COUNTRY_CAPITAL_TEMPLATES, single_entity=True)
    return fam


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
