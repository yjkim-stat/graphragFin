"""Advanced annotation pipeline combining rule-based heuristics and POS tagging.

This script keeps compatibility with the original `annotate.py` behaviour while
adding a spaCy-powered part-of-speech (POS) driven entity extraction path. Each
approach can be toggled independently at runtime.

Usage example::

    python indexing_utils/annotate-v2.py \
        --in-fname sample \
        --out-fname sample-annotated \
        --text-column all_text \
        --rule-based \
        --pos-tagging

The output CSV mirrors the input rows and adds:

```
rule_based_entities  # when rule-based extraction enabled
pos_entities         # when POS tagging enabled
subjects             # POS-based subject candidates
objects              # POS-based object candidates
entities             # merged entity list respecting enabled options
```

The script also emits a Markdown summary highlighting the most frequent
entities.  Plural and singular forms are normalised so that terms like
"tariff" and "tariffs" are considered equivalent.
"""

from __future__ import annotations

import argparse
import itertools
import re
from collections import Counter
from datetime import datetime
from functools import lru_cache
from typing import Iterable, List, Tuple

try:  # spaCy is optional unless POS tagging is enabled
    import spacy
except ImportError:  # pragma: no cover - spaCy is an optional dependency at runtime
    spacy = None  # type: ignore[assignment]

import pandas as pd

from lexicon_finance import enrich_entities_with_finance


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-fname", type=str, required=True)
    parser.add_argument("--out-fname", type=str, required=True)
    parser.add_argument("--text-column", type=str, required=True)

    parser.add_argument(
        "--rule-based",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Enable rule-based entity extraction (finance lexicon + heuristics). "
            "Use --no-rule-based to disable."
        ),
    )
    parser.add_argument(
        "--pos-tagging",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Enable spaCy-powered POS tagging to extract subjects, objects, and nouns. "
            "Use --no-pos-tagging to disable."
        ),
    )
    parser.add_argument(
        "--spacy-model",
        type=str,
        default="en_core_web_sm",
        help=(
            "spaCy model to use when POS tagging is enabled. "
            "Install with `python -m spacy download <model>` if necessary."
        ),
    )

    return parser


args = _build_parser().parse_args()


# ---------------------------------------------------------------------------
# Rule-based entity extraction components (shared with annotate.py)
# ---------------------------------------------------------------------------


MONTHS = (
    "january february march april may june july august september october november december "
    "jan feb mar apr may jun jul aug sep sept oct nov dec"
).split()

CURRENCY_WORDS = (
    r"(?:dollars?|usd|won|krw|euros?|eur|pounds?|gbp|yen|jpy|yuan|cny|rupees?|inr|aud|cad|chf|hkd|sgd|₩|\$|€|£|¥)"
)

MONEY_RE = re.compile(
    rf"(?i)(?:{CURRENCY_WORDS}\s*)?(?:[+-]?\d{{1,3}}(?:,\d{{3}})*(?:\.\d+)?|\d+(?:\.\d+)?)(?:\s*{CURRENCY_WORDS})?"
)
PERCENT_RE = re.compile(r"(?i)([+-]?\d+(?:\.\d+)?)\s*%")
YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
DATE_PHRASE_RE = re.compile(
    rf"(?i)\b(?:\d{{1,2}}\s+(?:{'|'.join(MONTHS)})|\b(?:{'|'.join(MONTHS)})\s+\d{{1,2}}|\b(?:{'|'.join(MONTHS)})\b)\b(?:,\s*\d{{4}})?"
)

STOPWORDS = set(
    "The A An And Or In On At For Of With To From Though Were Was Are Is As Has Have Had By Their Its It They He She "
    "Them You I This That These Those Aboard About Above After Against Along Amid Among Around Before Behind Below Beneath "
    "Besides Between Beyond But Concerning Considering Despite Down During Except Following Inside Into Like Near Onto Outside "
    "Over Past Per Plus Regarding Round Save Since Than Through Toward Towards Under Underneath Unlike Until Upon Versus Via Within Without"
    .lower()
    .split()
)


def proper_noun_chunks(text: str) -> List[str]:
    chunks: List[str] = []
    for sent in re.split(r"(?<=[\.\?!])\s+", text):
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
                if len(phrase) >= 2:
                    chunks.append(phrase)
            else:
                i += 1
    return list(dict.fromkeys(chunks))


def extract_rule_based_entities(text: str) -> List[str]:
    if not isinstance(text, str):
        return []

    money = [m.group(0) for m in MONEY_RE.finditer(text)]
    percent = [p.group(0) for p in PERCENT_RE.finditer(text)]
    years = re.findall(r"\b(?:19|20)\d{2}\b", text)
    date_phrases = [d.group(0) for d in DATE_PHRASE_RE.finditer(text)]
    proper_phrases = proper_noun_chunks(text)
    finance_entities = flatten_finance_entities(enrich_entities_with_finance(text))

    raw_entities = [
        e
        for e in money + percent + years + date_phrases + proper_phrases + finance_entities
        if isinstance(e, str) and e
    ]
    return unique_preserve_order(normalize_entity(e) for e in raw_entities)


def flatten_finance_entities(d: dict) -> List[str]:
    items: List[str] = []
    for _, values in d.items():
        items.extend(values)
    return list(dict.fromkeys(items))


# ---------------------------------------------------------------------------
# POS tagging utilities
# ---------------------------------------------------------------------------


SPACY_ENTITY_POS = {"NOUN", "PROPN"}
SPACY_SUBJECT_DEPS = {"nsubj", "nsubjpass", "csubj", "csubjpass", "expl"}
SPACY_OBJECT_DEPS = {"dobj", "obj", "pobj", "iobj", "attr", "oprd", "dative"}
SPACY_CHUNK_SKIP_POS = {"DET", "PRON"}


@lru_cache(maxsize=4)
def get_spacy_model(model_name: str):
    """Load and cache a spaCy language model."""

    if spacy is None:  # pragma: no cover - runtime dependency guard
        raise RuntimeError(
            "spaCy is required for POS tagging. Install it with `pip install spacy` and the desired model."
        )

    try:
        return spacy.load(model_name, exclude=["ner"])
    except OSError as exc:  # pragma: no cover - depends on runtime environment
        raise RuntimeError(
            f"spaCy model '{model_name}' is not installed. "
            f"Install it via `python -m spacy download {model_name}` before running with POS tagging enabled."
        ) from exc


def _chunk_to_phrase(chunk) -> str:
    tokens = [token.text for token in chunk if token.pos_ not in SPACY_CHUNK_SKIP_POS and not token.is_space]
    if not tokens:
        tokens = [chunk.root.text]
    phrase = " ".join(tokens).strip()
    return phrase


def analyze_with_pos(text: str) -> Tuple[List[str], List[str], List[str]]:
    if not isinstance(text, str) or not text.strip():
        return [], [], []

    nlp = get_spacy_model(args.spacy_model)
    doc = nlp(text)

    noun_chunks = list(doc.noun_chunks)
    chunk_token_indices = {token.i for chunk in noun_chunks for token in chunk}

    entities: List[str] = []
    subjects: List[str] = []
    objects: List[str] = []

    for chunk in noun_chunks:
        phrase = _chunk_to_phrase(chunk)
        if not phrase:
            continue

        lower_phrase = phrase.lower()
        root = chunk.root
        if root.pos_ in SPACY_ENTITY_POS and lower_phrase not in STOPWORDS:
            entities.append(phrase)
        if root.dep_ in SPACY_SUBJECT_DEPS and lower_phrase not in STOPWORDS:
            subjects.append(phrase)
        if root.dep_ in SPACY_OBJECT_DEPS and lower_phrase not in STOPWORDS:
            objects.append(phrase)

    for token in doc:
        if token.i in chunk_token_indices or token.is_space:
            continue

        if token.pos_ in SPACY_ENTITY_POS and not token.is_stop and token.text.strip():
            entities.append(token.text)
        if token.dep_ in SPACY_SUBJECT_DEPS and not token.is_stop and token.text.strip():
            subjects.append(token.text)
        if token.dep_ in SPACY_OBJECT_DEPS and not token.is_stop and token.text.strip():
            objects.append(token.text)

    entities = unique_preserve_order(normalize_entity(e) for e in entities if e)
    subjects = unique_preserve_order(normalize_entity(s) for s in subjects if s)
    objects = unique_preserve_order(normalize_entity(o) for o in objects if o)

    return entities, subjects, objects


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------


IRREGULAR_PLURALS = {
    "people": "person",
    "men": "man",
    "women": "woman",
    "children": "child",
    "teeth": "tooth",
    "feet": "foot",
    "mice": "mouse",
    "geese": "goose",
    "indices": "index",
    "matrices": "matrix",
    "analyses": "analysis",
    "crises": "crisis",
    "theses": "thesis",
}


def _restore_case(original: str, base: str) -> str:
    if original.isupper():
        return base.upper()
    if original.istitle():
        return base.capitalize()
    return base


def singularize_word(word: str) -> str:
    lower = word.lower()
    if lower in IRREGULAR_PLURALS:
        base = IRREGULAR_PLURALS[lower]
    elif lower.endswith("ies") and len(lower) > 3:
        base = lower[:-3] + "y"
    elif lower.endswith("ves") and len(lower) > 3:
        base = lower[:-3] + "f"
    elif lower.endswith("ses") and len(lower) > 3 and not lower.endswith("sses"):
        base = lower[:-2]
    elif lower.endswith("s") and not lower.endswith("ss") and len(lower) > 3:
        base = lower[:-1]
    else:
        base = lower
    return _restore_case(word, base)


def normalize_entity(entity: str) -> str:
    if not entity:
        return entity
    tokens = entity.split()
    if len(tokens) > 1 and all(token and token[0].isupper() for token in tokens):
        return entity.strip()
    singular_tokens = [singularize_word(token) for token in tokens]
    normalized = " ".join(singular_tokens)
    return normalized.strip()


def unique_preserve_order(items: Iterable[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for item in items:
        if not item:
            continue
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------


df = pd.read_csv(f"{args.in_fname}.csv")


if args.rule_based:
    df["rule_based_entities"] = df[args.text_column].apply(extract_rule_based_entities)
else:
    df["rule_based_entities"] = [[] for _ in range(len(df))]


if args.pos_tagging:
    pos_results = df[args.text_column].apply(analyze_with_pos)
    df["pos_entities"] = pos_results.apply(lambda tpl: tpl[0])
    df["subjects"] = pos_results.apply(lambda tpl: tpl[1])
    df["objects"] = pos_results.apply(lambda tpl: tpl[2])
else:
    df["pos_entities"] = [[] for _ in range(len(df))]
    df["subjects"] = [[] for _ in range(len(df))]
    df["objects"] = [[] for _ in range(len(df))]


def merge_entities(row) -> List[str]:
    combined = []
    if args.rule_based:
        combined.extend(row["rule_based_entities"])
    if args.pos_tagging:
        combined.extend(row["pos_entities"])
    return unique_preserve_order(combined)


if args.rule_based or args.pos_tagging:
    df["entities"] = df.apply(merge_entities, axis=1)
else:
    df["entities"] = [[] for _ in range(len(df))]


timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
out_fname = f"{args.out_fname}-{timestamp}.csv"
df.to_csv(out_fname, index=False)


# ---------------------------------------------------------------------------
# Summary generation
# ---------------------------------------------------------------------------


all_entities = list(itertools.chain.from_iterable(df["entities"]))
entity_counts = Counter(e for e in all_entities if e)
summary_df = pd.DataFrame(entity_counts.most_common(), columns=["Entity", "Count"])

summary_md = "# Entity Extraction Summary\n\n"
summary_md += f"**Created at:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
summary_md += f"**Input file:** `{args.in_fname}`\n\n"
summary_md += f"**Rule-based enabled:** {args.rule_based}\n\n"
summary_md += f"**POS tagging enabled:** {args.pos_tagging}\n\n"
summary_md += "## Top Entities\n\n"

if not summary_df.empty:
    summary_md += summary_df.head(100).to_markdown(index=False)
else:
    summary_md += "_No entities detected._"

summary_md += "\n\n---\n\n"
summary_md += f"**Total unique entities:** {len(summary_df)}\n"
summary_md += f"**Total mentions:** {sum(summary_df['Count'])}\n"

summary_fname = f"{args.out_fname}-{timestamp}-summary.md"
with open(summary_fname, "w", encoding="utf-8") as f:
    f.write(summary_md)

print(f"✅ Saved annotated CSV: {out_fname}")
print(f"✅ Saved summary Markdown: {summary_fname}")

