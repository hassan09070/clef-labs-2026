#!/usr/bin/env python3
"""
clef_task1_data_clean.py
─────────────────────────────────────────────────────────────────────────────
Loads the raw CFA/CPA dataset (Tomas08119993/finmmeval-cfa-cpa) from
HuggingFace by downloading all four Parquet files directly (EN + CN).
The HF datasets library cannot load this dataset due to a Parquet schema
mismatch (gold is list<int64> but schema says int64).

For each row the following fields are extracted:

    question  – clean question text (from the `text` column)
    options   – dict of option letter → option text, e.g.
                {"a": "Underwriters' fairness opinion …",
                 "b": "Assessment of risk factors …"}
    gold      – list of correct option letter(s), e.g. ["b"] or ["b","c","d"]
    reason    – explanation text (from the `reason` column)
    lang      – "en" or "cn"

Validations applied:
    • Skip rows where question, options, or gold is missing/null
    • gold is always a list; skip if empty after cleaning
    • Skip if any gold index is out of bounds for choices
    • Deduplicate on (question, json(options)) — keep first occurrence

Output: task1_Tomas08119993_finmmeval-cfa-cpa.json
─────────────────────────────────────────────────────────────────────────────
"""

import io
import json
import re
import os
import requests
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# HuggingFace auth — set HF_TOKEN env var to avoid hardcoding
# ─────────────────────────────────────────────────────────────────────────────
_HF_TOKEN = os.environ.get("HF_TOKEN", "")

_PARQUET_FILES = [
    ("data/train-00000-of-00001-en.parquet", "en"),
    ("data/train-00000-of-00002-en.parquet", "en"),
    ("data/train-00000-of-00001-cn.parquet", "cn"),
    ("data/train-00000-of-00002-cn.parquet", "cn"),
]
_BASE_URL = (
    "https://huggingface.co/datasets/Tomas08119993/finmmeval-cfa-cpa/resolve/main/"
)
_OUTPUT_FILE = "task1_Tomas08119993_finmmeval-cfa-cpa.json"


# ─── Option parsing ───────────────────────────────────────────────────────────

# Matches lines like:  "a. Some text"  or  "A. Some text"
_OPTION_LINE = re.compile(r"^([a-zA-Z])\.\s+(.+)$")

# Both EN ("Options:" / "Answer:") and CN ("选项:" / "答案:") headers
_OPTS_PATTERN = re.compile(
    r"(?:Options:|选项:)\s*\n(.*?)(?=\n(?:Answer:|答案:)|\Z)",
    re.DOTALL,
)


def parse_options(query: str) -> dict:
    """
    Extract the options block from a query string.

    Supports English format:
        Options:
        a. Underwriters' fairness opinion of the offering
        b. Assessment of risk factors …
        Answer:

    And Chinese format:
        选项:
        a. …
        答案:

    Returns a dict: {"a": "…", "b": "…", …}
    Returns an empty dict if the block cannot be parsed.
    """
    match = _OPTS_PATTERN.search(query)
    if not match:
        return {}

    options = {}
    for line in match.group(1).splitlines():
        line = line.strip()
        m = _OPTION_LINE.match(line)
        if m:
            letter = m.group(1).lower()
            text = m.group(2).strip()
            options[letter] = text
    return options


# ─── Gold normalisation ───────────────────────────────────────────────────────

def normalise_gold(gold_raw, choices: list) -> list | None:
    """
    Convert raw gold value to a validated list of lowercase answer letters.

    Accepts:
        - scalar int / float / np.integer  →  [choices[idx]]
        - array / list of ints             →  [choices[i] for i in indices]

    Returns None if:
        - gold_raw is null / NA
        - resulting list is empty
        - any index is out of bounds for choices
    """
    # ── null / NA check ───────────────────────────────────────────────────────
    try:
        if gold_raw is None or (isinstance(gold_raw, float) and np.isnan(gold_raw)):
            return None
        # pandas NA
        if pd.isna(gold_raw):
            return None
    except (TypeError, ValueError):
        pass   # pd.isna raises on non-scalar iterables — handled below

    # ── normalise to list of int indices ─────────────────────────────────────
    if isinstance(gold_raw, (int, float, np.integer, np.floating)):
        indices = [int(gold_raw)]
    else:
        try:
            indices = [int(x) for x in list(gold_raw)]
        except (TypeError, ValueError):
            return None

    if not indices:
        return None

    # ── bounds check — skip row if ANY index is out of range ─────────────────
    n = len(choices)
    if any(idx < 0 or idx >= n for idx in indices):
        return None

    return [choices[idx].lower() for idx in indices]


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_raw_frames() -> list[tuple[pd.DataFrame, str]]:
    """Download all Parquet files and return list of (DataFrame, lang) pairs."""
    headers = {"Authorization": f"Bearer {_HF_TOKEN}"}
    result = []
    for path, lang in _PARQUET_FILES:
        url = _BASE_URL + path
        print(f"  downloading {path} …", end=" ", flush=True)
        resp = requests.get(url, headers=headers, timeout=120)
        resp.raise_for_status()
        df = pd.read_parquet(io.BytesIO(resp.content))
        print(f"{len(df)} rows  [{lang}]")
        result.append((df, lang))
    return result


# ─── Row processing ───────────────────────────────────────────────────────────

def process_row(row: pd.Series, lang: str) -> dict | None:
    """
    Convert a raw DataFrame row into the clean record format.
    Returns None for any row that fails validation.
    """
    try:
        # ── question ─────────────────────────────────────────────────────────
        question = str(row.get("text") or "").strip()
        if not question:
            return None

        # ── options ──────────────────────────────────────────────────────────
        query = str(row.get("query") or "")
        options = parse_options(query)
        if not options:
            return None

        # ── gold (list of correct answer letters) ────────────────────────────
        choices_raw = row.get("choices")
        if choices_raw is None:
            return None
        choices = list(choices_raw)
        if not choices:
            return None

        gold = normalise_gold(row.get("gold"), choices)
        if gold is None:
            return None

        # ── gold size vs choices size ─────────────────────────────────────────
        # Ignore rows where more gold answers are listed than there are choices
        if len(gold) > len(choices):
            return None

        # ── reason ───────────────────────────────────────────────────────────
        reason = str(row.get("reason") or "").strip()

        return {
            "question": question,
            "options":  options,
            "gold":     gold,
            "reason":   reason,
            "lang":     lang,
        }

    except Exception as exc:
        print(f"    [warn] skipped row: {exc}")
        return None


# ─── Deduplication key ────────────────────────────────────────────────────────

def dedup_key(record: dict) -> str:
    return record["question"].strip() + "||" + json.dumps(record["options"], sort_keys=True)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("Loading raw data …")
    frames = load_raw_frames()
    total_raw = sum(len(df) for df, _ in frames)
    print(f"  total raw rows: {total_raw}\n")

    print("Processing rows …")
    records = []
    seen: set[str] = set()
    stats = {"null_skip": 0, "parse_skip": 0, "bounds_skip": 0, "duplicate": 0}

    for df, lang in frames:
        for _, row in df.iterrows():
            record = process_row(row, lang)
            if record is None:
                stats["parse_skip"] += 1
                continue

            key = dedup_key(record)
            if key in seen:
                stats["duplicate"] += 1
                continue

            seen.add(key)
            records.append(record)

    kept = len(records)
    skipped = total_raw - kept
    print(f"  kept      : {kept}")
    print(f"  skipped   : {skipped}  "
          f"(parse/null/bounds={stats['parse_skip']}, duplicates={stats['duplicate']})")

    print(f"\nSaving to {_OUTPUT_FILE} …")
    with open(_OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"Done. {kept} records written to {_OUTPUT_FILE}")

    # ── Preview ───────────────────────────────────────────────────────────────
    if records:
        print("\n─── Sample EN record ────────────────────────────────────")
        sample = next((r for r in records if r["lang"] == "en"), records[0])
        print(f"question : {sample['question'][:100]} …")
        print(f"options  : {sample['options']}")
        print(f"gold     : {sample['gold']}")
        print(f"reason   : {sample['reason'][:100]} …")
        print(f"lang     : {sample['lang']}")

        cn_sample = next((r for r in records if r["lang"] == "cn"), None)
        if cn_sample:
            print("\n─── Sample CN record ────────────────────────────────────")
            print(f"question : {cn_sample['question'][:100]} …")
            print(f"options  : {cn_sample['options']}")
            print(f"gold     : {cn_sample['gold']}")
            print(f"lang     : {cn_sample['lang']}")


if __name__ == "__main__":
    main()
