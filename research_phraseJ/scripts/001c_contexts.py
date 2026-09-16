"""001c — Build the three labelled context sets for 001.

emission-template : held-out last 20% of released passages per T-member, minus prefix-leaking ones
emission-natural  : pile-10k text immediately preceding " <phrase>", <=128 tokens of left context;
                    doc ids written to results/.../natural_doc_ids.json so 001b excludes them
latent            : two-hop prompts from LATENT_FACTS (5 routes x 4 frames x rotating cues), with
                    mechanical admission checks (phrase/alias/answer absent from prompt)

Writes results/001-template-geometry/contexts.json (prompts only) and latent_facts.json.
"""
import json, os, random, re
import transformers
from datasets import load_dataset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results", "001-template-geometry")
TL = "/content/lenses/qwen3.6-27b/template-lens"
MODEL = "/content/models/Qwen3.6-27B"
tok = transformers.AutoTokenizer.from_pretrained(MODEL)
random.seed(0)
fam = json.load(open(f"{RES}/families.json"))

# ---------------------------------------------------------------- latent facts (authored 2026-09-16; review welcome)
# cues must not contain the phrase, its suffix word, or the route answer
LATENT_FACTS = {
 "New York":     {"cues": ["the Statue of Liberty", "Times Square", "Wall Street", "the Empire State Building", "the city of Albany"],
                  "continent": "North America", "language": "English", "currency": "dollar", "hemisphere": "Northern", "second_word_letters": "4"},
 "New Zealand":  {"cues": ["the city of Wellington", "the All Blacks rugby team", "the kiwi bird", "the city of Auckland", "the Maori people"],
                  "continent": "Oceania", "language": "English", "currency": "dollar", "hemisphere": "Southern", "second_word_letters": "7"},
 "New Delhi":    {"cues": ["India Gate", "the Red Fort", "Rashtrapati Bhavan", "Connaught Place", "the Lotus Temple"],
                  "continent": "Asia", "language": "Hindi", "currency": "rupee", "hemisphere": "Northern", "second_word_letters": "5"},
 "New Jersey":   {"cues": ["the city of Trenton", "Atlantic City", "Princeton University", "the city of Newark", "the city of Hoboken"],
                  "continent": "North America", "language": "English", "currency": "dollar", "hemisphere": "Northern", "second_word_letters": "6"},
 "South Africa": {"cues": ["Cape Town", "Table Mountain", "Nelson Mandela", "Kruger National Park", "Johannesburg"],
                  "continent": "Africa", "language": "Zulu", "currency": "rand", "hemisphere": "Southern", "second_word_letters": "6"},
 "South Korea":  {"cues": ["the city of Seoul", "the Samsung company", "K-pop music", "the city of Busan", "Gyeongbokgung Palace"],
                  "continent": "Asia", "language": "Korean", "currency": "won", "hemisphere": "Northern", "second_word_letters": "5"},
 "South Dakota": {"cues": ["Mount Rushmore", "the city of Pierre", "Sioux Falls", "the Badlands", "Rapid City"],
                  "continent": "North America", "language": "English", "currency": "dollar", "hemisphere": "Northern", "second_word_letters": "6"},
 "South Carolina": {"cues": ["the city of Charleston", "Myrtle Beach", "Fort Sumter", "Hilton Head Island", "Clemson University"],
                  "continent": "North America", "language": "English", "currency": "dollar", "hemisphere": "Northern", "second_word_letters": "8"},
 "North Korea":  {"cues": ["the city of Pyongyang", "Kim Jong Un", "the Juche ideology", "the Ryugyong Hotel", "the Kim dynasty"],
                  "continent": "Asia", "language": "Korean", "currency": "won", "hemisphere": "Northern", "second_word_letters": "5"},
 "North Carolina": {"cues": ["the city of Raleigh", "the city of Charlotte", "Duke University", "the Outer Banks", "Chapel Hill"],
                  "continent": "North America", "language": "English", "currency": "dollar", "hemisphere": "Northern", "second_word_letters": "8"},
 "North Dakota": {"cues": ["the city of Bismarck", "the city of Fargo", "Theodore Roosevelt National Park", "Grand Forks", "the city of Minot"],
                  "continent": "North America", "language": "English", "currency": "dollar", "hemisphere": "Northern", "second_word_letters": "6"},
 "United States": {"cues": ["Washington, D.C.", "Hollywood", "the Grand Canyon", "the Fourth of July holiday", "the U.S. Congress"],
                  "continent": "North America", "language": "English", "currency": "dollar", "hemisphere": "Northern", "second_word_letters": "6"},
 "United Kingdom": {"cues": ["Buckingham Palace", "the city of Manchester", "Big Ben", "the Houses of Parliament at Westminster", "the city of Edinburgh"],
                  "continent": "Europe", "language": "English", "currency": "pound", "hemisphere": "Northern", "second_word_letters": "7"},
 "United Nations": {"cues": ["the Security Council", "the Secretary-General", "the General Assembly", "UNESCO", "peacekeeping blue helmets"],
                  "continent": None, "language": "English", "currency": None, "hemisphere": "Northern", "second_word_letters": "7"},
 "United Arab Emirates": {"cues": ["Dubai", "Abu Dhabi", "the Burj Khalifa", "Sharjah", "the Palm Jumeirah"],
                  "continent": "Asia", "language": "Arabic", "currency": "dirham", "hemisphere": "Northern", "second_word_letters": "4"},
 "San Francisco": {"cues": ["the Golden Gate Bridge", "Alcatraz Island", "the cable cars on Powell Street", "the 49ers football team", "Fisherman's Wharf"],
                  "continent": "North America", "language": "English", "currency": "dollar", "hemisphere": "Northern", "second_word_letters": "9", "state": "California"},
 "San Diego":    {"cues": ["Balboa Park", "Coronado Island", "the Padres baseball team", "Comic-Con", "the SeaWorld park near Mission Bay"],
                  "continent": "North America", "language": "English", "currency": "dollar", "hemisphere": "Northern", "second_word_letters": "5", "state": "California"},
 "San Antonio":  {"cues": ["the Alamo", "the River Walk", "the Spurs basketball team", "the Tower of the Americas", "the Alamodome"],
                  "continent": "North America", "language": "English", "currency": "dollar", "hemisphere": "Northern", "second_word_letters": "7", "state": "Texas"},
 "San Jose":     {"cues": ["the Sharks hockey team", "the Winchester Mystery House", "the SAP Center arena", "the Tech Interactive museum", "the headquarters of Adobe"],
                  "continent": "North America", "language": "English", "currency": "dollar", "hemisphere": "Northern", "second_word_letters": "4", "state": "California"},
}
ROUTES = {
 "continent":  ["Fact: The continent containing the place where {cue} is located is",
                "The place associated with {cue} lies on the continent of",
                "Q: On which continent is the place known for {cue}? A:",
                "Geography note: {cue} can be found on the continent called"],
 "language":   ["Fact: The language most widely spoken in the place where {cue} is located is",
                "People living where {cue} is located mostly speak",
                "Q: What language is mainly spoken in the place known for {cue}? A:",
                "Travel note: around {cue}, the main language you will hear is"],
 "currency":   ["Fact: The currency used in the place where {cue} is located is the",
                "Visitors to {cue} pay for things using the",
                "Q: Which currency is used in the place known for {cue}? A: The",
                "Travel note: near {cue}, prices are listed in the local currency, the"],
 "hemisphere": ["Fact: The hemisphere in which the place where {cue} is located lies is the",
                "The place associated with {cue} is in the",
                "Q: Is the place known for {cue} in the Northern or Southern hemisphere? A: The",
                "Geography note: {cue} is located in the"],
 "second_word_letters": ["Fact: The number of letters in the second word of the name of the place where {cue} is located is",
                "Count the letters in the second word of the name of the place associated with {cue}. The count is",
                "Q: How many letters are in the second word of the name of the place known for {cue}? A:",
                "Puzzle: the place where {cue} is located has a two-word name; its second word has this many letters:"],
}
ALIASES = {"New York": ["NYC", "York"], "New Zealand": ["Zealand", "NZ", "Kiwi"], "New Delhi": ["Delhi"], "New Jersey": ["Jersey"],
           "South Africa": ["Africa"], "South Korea": ["Korea"], "South Dakota": ["Dakota"], "South Carolina": ["Carolina"],
           "North Korea": ["Korea"], "North Carolina": ["Carolina"], "North Dakota": ["Dakota"],
           "United States": ["States", "USA", "America"], "United Kingdom": ["Kingdom", "UK", "Britain"], "United Nations": ["Nations", "UN"],
           "United Arab Emirates": ["Emirates", "UAE", "Arab"], "San Francisco": ["Francisco", "SF"], "San Diego": ["Diego"],
           "San Antonio": ["Antonio"], "San Jose": ["Jose"]}

def admitted(prompt, phrase, answer):
    low = prompt.lower()
    bad = [phrase] + ALIASES.get(phrase, [])
    if any(re.search(r"\b" + re.escape(b.lower()) + r"\b", low) for b in bad): return False, "phrase/alias in prompt"
    if answer and re.search(r"\b" + re.escape(answer.lower()) + r"\b", low): return False, "answer in prompt"
    return True, ""

latent = []
for phrase, f in LATENT_FACTS.items():
    for route, frames in ROUTES.items():
        ans = f.get(route)
        if ans is None: continue
        for fi, frame in enumerate(frames):
            cue = f["cues"][(fi + list(ROUTES).index(route)) % len(f["cues"])]
            p = frame.format(cue=cue)
            ok, why = admitted(p, phrase, ans)
            latent.append({"phrase": phrase, "route": route, "frame": fi, "cue": cue, "prompt": p, "answer": ans, "admitted": ok, "reason": why})
print("latent prompts:", len(latent), "admitted:", sum(x["admitted"] for x in latent), "per phrase:",
      {p: sum(x["admitted"] for x in latent if x["phrase"] == p) for p in LATENT_FACTS})

# ---------------------------------------------------------------- emission-template
members = [m for F in fam["families"] for m in F["members"]] + [m for P in fam["non_family_pairs"] for m in P["members"]]
passfiles = {f: json.load(open(f"{TL}/passages/{f}")) for f in os.listdir(f"{TL}/passages") if f.endswith(".json")}
def leaks_prefix(passage, ids):
    tail = tok.encode(passage[-80:], add_special_tokens=False)
    for k in range(1, len(ids)):  # any proper prefix of the phrase's ids
        if tail[-k:] == ids[:k]: return True
    # word-level check (passage may end without the leading space)
    last = passage.rstrip().split()[-1].strip('.,;:"\'()') if passage.strip() else ""
    words = tok.decode(ids).strip().split()
    return any(last == w for w in words[:-1])
emission_template = []
for m in members:
    if not m["in_template_vocab"]: continue
    ps = []
    for fname, _ in m["passages"]: ps.extend(passfiles[fname][m["template_text"]])
    heldout = ps[int(0.8 * len(ps)):]
    kept = [p for p in heldout if not leaks_prefix(p, m["ids"])]
    for p in kept: emission_template.append({"phrase": m["phrase"], "text": p})
    print(f"emission-template {m['phrase']:22s} heldout={len(heldout):3d} kept={len(kept):3d} {'INSUFFICIENT' if len(kept)<30 else ''}")

# ---------------------------------------------------------------- emission-natural
ds = load_dataset("NeelNanda/pile-10k", split="train")
phrases = sorted({m["phrase"] for m in members})
emission_natural, doc_ids = [], set()
per = {p: 0 for p in phrases}; CAP = 40
for di, d in enumerate(ds):
    t = d["text"]
    for p in phrases:
        if per[p] >= CAP: continue
        for mm in re.finditer(r"(?<=\S) " + re.escape(p) + r"\b", t):
            if per[p] >= CAP: break
            left = t[:mm.start()]
            ids = tok.encode(left, add_special_tokens=False)[-128:]
            if len(ids) < 32: continue
            emission_natural.append({"phrase": p, "text": tok.decode(ids), "doc_id": di}); doc_ids.add(di); per[p] += 1
json.dump(sorted(doc_ids), open(f"{RES}/natural_doc_ids.json", "w"))
print("emission-natural per phrase:", {p: n for p, n in per.items()}, "| docs excluded from Sigma:", len(doc_ids))

json.dump({"emission_template": emission_template, "emission_natural": emission_natural, "latent": latent,
           "insufficient_natural": [p for p, n in per.items() if n < 20]},
          open(f"{RES}/contexts.json", "w"), indent=1)
json.dump({"facts": LATENT_FACTS, "routes": ROUTES, "aliases": ALIASES}, open(f"{RES}/latent_facts.json", "w"), indent=1)
