"""001a — Build tokenizer-verified prefix-collision families for Qwen3.6-27B.

Outputs results/001-template-geometry/families.json with, per family: members, in-prose token ids
(" " + phrase), common-prefix length k, template-vocabulary membership (T/N), template row ids,
passage counts, pile-10k natural-occurrence counts, and lowercase-tokenization flags.
Also emits control pairs: random same-prefix T-pairs and random unrelated T-pairs.
"""
import collections, json, os, random, sys
import transformers
from datasets import load_dataset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "001-template-geometry")
os.makedirs(OUT, exist_ok=True)
MODEL = "/content/models/Qwen3.6-27B"
TL = "/content/lenses/qwen3.6-27b/template-lens"

CANDIDATE_FAMILIES = [
    ["New York", "New Zealand", "New Delhi", "New Jersey"],
    ["South Africa", "South Korea", "South Dakota", "South Carolina"],
    ["North Korea", "North Carolina", "North Dakota"],
    ["United States", "United Kingdom", "United Nations", "United Arab Emirates"],
    ["San Francisco", "San Diego", "San Antonio", "San Jose"],
    ["Saint Petersburg", "Saint Louis"],
    ["Golden Gate Bridge", "Golden Retriever"],
    ["ice cream", "ice age"],
    ["credit card", "credit score"],
    ["general relativity", "general election"],
    ["World Cup", "World War"],
    ["hot dog", "hot water"],
    ["Los Angeles", "Las Vegas"],          # expected: no shared prefix -> reported, dropped
    ["blackmail", "blackboard", "blacksmith"],
]
NON_FAMILY_PAIRS = [["blackmail", "extortion"], ["photosynthesis", "photograph"]]

tok = transformers.AutoTokenizer.from_pretrained(MODEL)
def ids(s): return tok.encode(" " + s, add_special_tokens=False)

# template vocabulary (v3 stack) -> row id
vocab = {}
for line in open(f"{TL}/template_words+phrases_v3.txt"):
    rid, text = line.rstrip("\n").split("\t", 1)
    vocab[text] = int(rid)
vocab_lc = collections.defaultdict(list)
for text, rid in vocab.items():
    vocab_lc[text.lower()].append(text)

# passage counts per phrase across all passage files
passages = {}
for f in os.listdir(f"{TL}/passages"):
    if f.endswith(".json"):
        d = json.load(open(f"{TL}/passages/{f}"))
        for k, v in d.items():
            passages.setdefault(k, []).extend([(f, len(v))])

# pile-10k natural occurrence counts (space-prefixed)
ds = load_dataset("NeelNanda/pile-10k", split="train")
all_phrases = sorted({p for fam in CANDIDATE_FAMILIES for p in fam} | {p for pr in NON_FAMILY_PAIRS for p in pr})
nat = collections.Counter()
for d in ds:
    t = d["text"]
    for p in all_phrases:
        c = t.count(" " + p)
        if c: nat[p] += c

def common_prefix(seqs):
    k = 0
    while all(len(s) > k for s in seqs) and len({s[k] for s in seqs}) == 1:
        k += 1
    return k

def member_record(p):
    i = ids(p); lc = tok.encode(" " + p.lower(), add_special_tokens=False)
    text_in_vocab = p if p in vocab else (vocab_lc[p.lower()][0] if vocab_lc.get(p.lower()) else None)
    return {
        "phrase": p, "ids": i, "tokens": [tok.decode([t]) for t in i], "n_tokens": len(i),
        "lowercase_ids": lc, "lowercase_tokenizes_differently": lc != i,
        "in_template_vocab": text_in_vocab is not None, "template_text": text_in_vocab,
        "template_row": vocab.get(text_in_vocab) if text_in_vocab else None,
        "passages": passages.get(text_in_vocab, []) if text_in_vocab else [],
        "n_passages": sum(n for _, n in passages.get(text_in_vocab, [])) if text_in_vocab else 0,
        "pile10k_natural_occurrences": nat[p],
    }

families, dropped = [], []
for fam in CANDIDATE_FAMILIES:
    recs = [member_record(p) for p in fam]
    k = common_prefix([r["ids"] for r in recs])
    if k == 0:
        dropped.append({"members": fam, "reason": "no shared token prefix", "ids": [r["ids"] for r in recs]}); continue
    # members whose whole phrase is just the prefix (suffix length 0) cannot be discriminated by suffix
    recs = [dict(r, suffix_len=r["n_tokens"] - k) for r in recs]
    families.append({
        "name": tok.decode(recs[0]["ids"][:k]).strip(),
        "prefix_ids": recs[0]["ids"][:k], "prefix_len": k,
        "members": recs,
        "n_T": sum(r["in_template_vocab"] for r in recs),
        "usable_for_template_eval": sum(r["in_template_vocab"] for r in recs) >= 2,
    })

non_family = []
for pr in NON_FAMILY_PAIRS:
    recs = [member_record(p) for p in pr]
    non_family.append({"members": recs, "prefix_len": common_prefix([r["ids"] for r in recs])})

# controls from the template vocabulary: natural-form multi-token rows grouped by first token,
# excluding rows that are multi-token only because of lowercasing (natural capitalized form single-token)
random.seed(0)
groups = collections.defaultdict(list)
for text, rid in vocab.items():
    i = ids(text)
    if len(i) < 2: continue
    cap = tok.encode(" " + text.capitalize(), add_special_tokens=False)
    if len(cap) == 1: continue  # lowercase artifact
    groups[i[0]].append((text, rid, i))
same_prefix_pool = [g for g in groups.values() if len(g) >= 2]
random_same_prefix = []
for g in random.sample(same_prefix_pool, min(20, len(same_prefix_pool))):
    a, b = random.sample(g, 2)
    random_same_prefix.append({"members": [{"phrase": a[0], "template_row": a[1], "ids": a[2]},
                                           {"phrase": b[0], "template_row": b[1], "ids": b[2]}],
                               "prefix_len": common_prefix([a[2], b[2]])})
rows = list(vocab.items())
random_unrelated = []
for _ in range(50):
    (ta, ra), (tb, rb) = random.sample(rows, 2)
    random_unrelated.append({"members": [{"phrase": ta, "template_row": ra, "ids": ids(ta)},
                                         {"phrase": tb, "template_row": rb, "ids": ids(tb)}]})

out = {"model": MODEL, "template_stack": "templates+phrases_v3", "boundary_convention": 'tok.encode(" " + phrase)',
       "families": families, "dropped": dropped, "non_family_pairs": non_family,
       "controls": {"random_same_prefix": random_same_prefix, "random_unrelated": random_unrelated},
       "n_lowercase_artifact_rows": sum(1 for t in vocab if len(ids(t)) > 1 and len(tok.encode(" " + t.capitalize(), add_special_tokens=False)) == 1)}
json.dump(out, open(f"{OUT}/families.json", "w"), indent=1)

print(f"families kept: {len(families)}  dropped: {len(dropped)}  lowercase-artifact rows: {out['n_lowercase_artifact_rows']}")
for f in families:
    print(f"\n[{f['name']!r}] prefix_len={f['prefix_len']} usable_for_template_eval={f['usable_for_template_eval']}")
    for r in f["members"]:
        print(f"   {r['phrase']:22s} toks={r['tokens']}  T={r['in_template_vocab']!s:5s} passages={r['n_passages']:4d} natural={r['pile10k_natural_occurrences']:5d} lc_diff={r['lowercase_tokenizes_differently']}")
for d in dropped: print("\nDROPPED", d)
