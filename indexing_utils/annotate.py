# We'll demonstrate simple, dependency-free entity extraction (heuristic-based) from the `all_text` column.
# This approach uses regex for MONEY, DATE, PERCENT, and capitalization heuristics for PROPER NOUN phrases.
# In production, you'd typically swap this with spaCy or Hugging Face NER; this runs offline with no downloads.

import re
import argparse
import pandas as pd
from datetime import datetime

from lexicon_finance import enrich_entities_with_finance

parser = argparse.ArgumentParser()
parser.add_argument('--in-fname', type=str)
parser.add_argument('--out-fname', type=str)
parser.add_argument('--text-column', type=str)

args = parser.parse_args()

df = pd.read_csv(f'{args.in_fname}.csv')

MONTHS = ("january february march april may june july august september october november december "
          "jan feb mar apr may jun jul aug sep sept oct nov dec").split()

CURRENCY_WORDS = r"(?:dollars?|usd|won|krw|euros?|eur|pounds?|gbp|yen|jpy|yuan|cny|rupees?|inr|aud|cad|chf|hkd|sgd|₩|\$|€|£|¥)"
MONEY_RE = re.compile(rf"(?i)(?:{CURRENCY_WORDS}\s*)?(?:[+-]?\d{{1,3}}(?:,\d{{3}})*(?:\.\d+)?|\d+(?:\.\d+)?)(?:\s*{CURRENCY_WORDS})?")
PERCENT_RE = re.compile(r"(?i)([+-]?\d+(?:\.\d+)?)\s*%")
YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
DATE_PHRASE_RE = re.compile(rf"(?i)\b(?:\d{{1,2}}\s+(?:{'|'.join(MONTHS)})|\b(?:{'|'.join(MONTHS)})\s+\d{{1,2}}|\b(?:{'|'.join(MONTHS)})\b)\b(?:,\s*\d{{4}})?")
# Proper noun phrases heuristic: sequences of capitalized words (allow hyphens) not at sentence start common stopwords
STOPWORDS = set("The A An And Or In On At For Of With To From Though Were Was Are Is As Has Have Had By Their Its It They He She Them You I This That These Those Aboard About Above After Against Along Amid Among Around Before Behind Below Beneath Beside Besides Between Beyond But Concerning Considering Despite Down During Except Following Inside Into Like Near Onto Outside Over Past Per Plus Regarding Round Save Since Than Through Toward Towards Under Underneath Unlike Until Upon Versus Via Within Without".lower().split())

def proper_noun_chunks(text):
    # Split sentences by punctuation
    chunks = []
    for sent in re.split(r'(?<=[\.\?!])\s+', text):
        tokens = re.findall(r"[A-Za-z][A-Za-z\-']*", sent)
        i = 0
        while i < len(tokens):
            t = tokens[i]
            if t[0].isupper() and t.lower() not in STOPWORDS:
                start = i
                i += 1
                while i < len(tokens) and tokens[i][0].isupper() and tokens[i].lower() not in STOPWORDS:
                    i += 1
                phrase = " ".join(tokens[start:i])
                # Filter out single common words
                if len(phrase) >= 2:
                    chunks.append(phrase)
            else:
                i += 1
    return list(dict.fromkeys(chunks))  # dedupe preserving order

# def extract_entities(text: str):
#     if not isinstance(text, str):
#         return []
#     money = [m.group(0) for m in MONEY_RE.finditer(text)]
#     percent = [p.group(0) for p in PERCENT_RE.finditer(text)]
#     years = YEAR_RE.findall(text)
#     years = re.findall(r"\b(?:19|20)\d{2}\b", text)
#     date_phrases = [d.group(0) for d in DATE_PHRASE_RE.finditer(text)]
#     # Combine date-like entities
#     date_entities = sorted(list(dict.fromkeys(date_phrases + years)))
#     proper_phrases = proper_noun_chunks(text)
    
#     # 모든 엔티티를 하나의 리스트로 합치고 중복 제거
#     all_entities = list(dict.fromkeys(money + percent + years + date_phrases + proper_phrases))
#     return all_entities

def flatten_finance_entities(d):
    items = []
    for k, vs in d.items():
        items.extend(vs)
    # dedupe while preserving order
    return list(dict.fromkeys(items))

df["entities"] = df[args.text_column].apply(enrich_entities_with_finance).apply(flatten_finance_entities)

out = df.copy()

timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")

out.to_csv(f'{args.out_fname}-{timestamp}.csv', index=False)


from collections import Counter
import itertools

# --- Summary creation ---
all_entities = list(itertools.chain.from_iterable(df["entities"]))
all_entities = [e for e in all_entities if e and isinstance(e, str)]
entity_counts = Counter(all_entities)
summary_df = pd.DataFrame(entity_counts.most_common(), columns=["Entity", "Count"])

# --- Generate Markdown summary ---
summary_md = f"# Entity Extraction Summary\n\n"
summary_md += f"**Created at:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
summary_md += f"**Input file:** `{args.in_fname}`\n\n"
summary_md += f"## Top Entities\n\n"

if not summary_df.empty:
    summary_md += summary_df.head(100).to_markdown(index=False)
else:
    summary_md += "_No entities detected._"

summary_md += "\n\n---\n\n"
summary_md += f"**Total unique entities:** {len(summary_df)}\n"
summary_md += f"**Total mentions:** {sum(summary_df['Count'])}\n"

# --- Save Markdown file ---
summary_fname = f"{args.out_fname}-{timestamp}-summary.md"
with open(summary_fname, "w", encoding="utf-8") as f:
    f.write(summary_md)

print(f"✅ Saved summary Markdown: {summary_fname}")
