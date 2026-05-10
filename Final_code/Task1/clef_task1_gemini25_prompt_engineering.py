#!/usr/bin/env python3
# ════════════════════════════════════════════════════════════════════════════
# clef_task1_gemini25_prompt_engineering.py
#
# Financial MCQ Prompt Engineering Study — Gemini 2.5 Pro via OpenRouter
#
# Runs ALL 6 strategies in sequence. Questions within each strategy are
# dispatched to a thread-pool (MAX_WORKERS) for maximum throughput.
# Each strategy's JSON is saved immediately after completion so a mid-run
# crash loses only the in-progress strategy.
#
# Output schema is identical to finr1_<strategy>_results.json.
#
# Strategies:
#   baseline         — zero-shot, letter only
#   few_shot         — 3 in-context examples
#   strict_cot       — step-by-step + <think>/<answer> structure
#   role             — expert CFA persona system prompt
#   self_consistency — majority vote over SC_SAMPLES independent calls
#   meta             — elimination-based reasoning
#
# Prerequisites:
#   pip install openai pandas seaborn matplotlib
#
# Usage:
#   python clef_task1_gemini25_prompt_engineering.py          # TEST_MODE (6 q)
#   TEST_MODE=false python clef_task1_gemini25_prompt_engineering.py  # full
# ════════════════════════════════════════════════════════════════════════════

import os
import re
import sys
import json
import time
import random
import datetime
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ══════════════════════════════════════════════════════════════════════════════
# §1  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

# OpenRouter credentials & model
OPENROUTER_API_KEY: str = os.environ.get(
    "OPENROUTER_API_KEY",
    "",
)
MODEL_ID: str = os.environ.get("OPENROUTER_MODEL", "google/gemini-2.5-flash")
OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"

# All strategies — run every one in sequence
VALID_STRATEGIES = (
    # "baseline",
    "few_shot",
    "strict_cot",
    "role",
    "self_consistency",
    # "meta",
)

# Reproducibility
RANDOM_SEED: int = 42
random.seed(RANDOM_SEED)

# TEST_MODE: 1 sample per dataset (6 total) — fast smoke-test
TEST_MODE: bool = False
N_TEST: int = 1  # samples per dataset in TEST_MODE

# Self-consistency independent samples per question
SC_SAMPLES: int = 5

# Parallel API calls within each strategy.
# OpenRouter Gemini 2.5 Pro handles ~30 concurrent requests comfortably.
# Set MAX_WORKERS env-var to override (e.g. MAX_WORKERS=50).
MAX_WORKERS: int = int(os.environ.get("MAX_WORKERS", "30"))

# Max retries on transient errors (429, 500, 503)
MAX_RETRIES: int = 5
RETRY_BASE_DELAY: float = 2.0

# Generation temperatures
TEMPERATURE_SC: float = 0.7       # sampling diversity for self_consistency
TEMPERATURE_DEFAULT: float = 0.0  # greedy for all deterministic strategies

# Max output tokens per strategy
# strict_cot asks for explicit reasoning text which can be long; give it more room.
# All other strategies only need a short answer.
MAX_TOKENS_DEFAULT: int = 4096
MAX_TOKENS_COT: int = 16384

# Local dataset files already in the repo directory
_DATASET_FILES: dict = {
    "cfa_cpa":           "task1_Tomas08119993_finmmeval-cfa-cpa.json",
    "es_multifin":       "task1_TheFinAI_flare-es-multifin.json",
    "plutus":            "task1_TheFinAI_plutus-multifin.json",
    "arabic_accounting": "task1_SahmBenchmark_arabic-accounting-mcq.json",
    "arabic_business":   "task1_SahmBenchmark_arabic-business-mcq.json",
    "hindi_finance":     "task1_bharatgenai_BhashaBench-Finance-Hindi.json",
}

# ══════════════════════════════════════════════════════════════════════════════
# §2  DATASET LOADING & BALANCING  (identical schema to Fin-R1 notebook)
# ══════════════════════════════════════════════════════════════════════════════

_SOURCE_LANG: dict = {
    "cfa_cpa":           "English",
    "es_multifin":       "Spanish",
    "plutus":            "Multilingual",
    "arabic_accounting": "Arabic",
    "arabic_business":   "Arabic",
    "hindi_finance":     "Hindi",
}

_SCRIPT_DIR = Path(__file__).resolve().parent


def _load_local_json(path: Path, source_name: str) -> list:
    with open(path, encoding="utf-8") as fh:
        records = json.load(fh)
    rows = []
    for rec in records:
        options = rec.get("options") or {}
        if not options:
            continue
        sorted_keys = sorted(options.keys())
        choices     = [options[k] for k in sorted_keys]
        gold_raw    = rec.get("gold") or []
        valid_gold  = [g.lower() for g in gold_raw if g.lower() in sorted_keys]
        if not valid_gold:
            continue
        rows.append({
            "question": str(rec.get("question") or ""),
            "choices":  choices,
            "keys":     sorted_keys,
            "gold":     valid_gold,
            "source":   source_name,
        })
    return rows


def load_all_datasets() -> tuple:
    """Returns (datasets_raw, all_data, min_size)."""
    print(f"Loading local datasets from {_SCRIPT_DIR} ...")
    datasets_raw = {}
    for src_name, fname in _DATASET_FILES.items():
        fpath = _SCRIPT_DIR / fname
        if not fpath.exists():
            print(f"  {src_name:<33} MISSING — {fpath}")
            continue
        rows = _load_local_json(fpath, src_name)
        datasets_raw[src_name] = rows
        print(f"  {src_name:<33} {len(rows):>5} rows  OK")

    if TEST_MODE:
        for src in datasets_raw:
            datasets_raw[src] = datasets_raw[src][:N_TEST]

    min_size = min(len(r) for r in datasets_raw.values())
    print(f"\nSmallest dataset: {min_size} rows — sampling {min_size} from each")

    _bal_rng = random.Random(RANDOM_SEED)
    all_data = []
    for src_name, rows in datasets_raw.items():
        sampled = _bal_rng.sample(rows, min_size) if len(rows) > min_size else rows
        datasets_raw[src_name] = sampled
        all_data.extend(sampled)
        print(f"  {src_name:<33} {len(sampled):>5} rows  (balanced)")

    print(f"\nTotal: {len(all_data)} examples ({min_size} × {len(datasets_raw)} datasets)")
    return datasets_raw, all_data, min_size

# ══════════════════════════════════════════════════════════════════════════════
# §3  PROMPT BUILDERS  (same text as Fin-R1 notebook; no tokenizer needed)
# ══════════════════════════════════════════════════════════════════════════════

_FEWSHOT_RNG = random.Random(RANDOM_SEED)  # isolated RNG — reproducible regardless of call order


def _labels_and_options(choices: list) -> tuple:
    """Return (display_labels_UPPER, options_str, valid_letters_str)."""
    n      = len(choices)
    labels = [chr(ord("A") + i) for i in range(n)]
    opts   = "\n".join(f"{labels[i]}. {choices[i]}" for i in range(n))
    valid  = "/".join(labels)
    return labels, opts, valid


def _build_few_shot_examples(all_data: list, source: str, exclude_idx: int, n: int = 3) -> str:
    pool = [
        r for i, r in enumerate(all_data)
        if r["source"] == source and i != exclude_idx
    ]
    if len(pool) < n:
        pool = [r for i, r in enumerate(all_data) if i != exclude_idx]
    sampled = _FEWSHOT_RNG.sample(pool, min(n, len(pool)))
    parts = []
    for ex in sampled:
        labels, opts, _ = _labels_and_options(ex["choices"])
        answer_letter   = ex["gold"][0].upper()
        parts.append(f"Question: {ex['question']}\n{opts}\nAnswer: {answer_letter}")
    return "\n\n".join(parts)


_ROLE_SYSTEM = (
    "You are a senior Chartered Financial Analyst (CFA) with over 20 years of "
    "experience in financial markets, accounting standards, and investment analysis. "
    "You have deep expertise in international finance, business law, and economics. "
    "When answering questions, draw on your professional expertise and provide the "
    "most accurate financial judgement."
)


def build_baseline_prompt(question, choices, **_) -> tuple:
    """Returns (system_str, user_str)."""
    labels, opts, valid = _labels_and_options(choices)
    system = ""
    user = (
        "Answer the following multiple-choice question. "
        f"Respond with ONLY the single letter of the correct answer ({valid}).\n\n"
        f"Question: {question}\n\n{opts}"
    )
    return system, user


def build_few_shot_prompt(all_data, question, choices, source, row_idx, **_) -> tuple:
    labels, opts, valid = _labels_and_options(choices)
    examples = _build_few_shot_examples(all_data, source, row_idx, n=3)
    system = ""
    user = (
        "Answer financial multiple-choice questions. "
        f"Respond with ONLY the single letter ({valid}).\n\n"
        "Here are three solved examples:\n\n"
        f"{examples}\n\n"
        "Now answer this question:\n\n"
        f"Question: {question}\n\n{opts}\nAnswer:"
    )
    return system, user


def build_strict_cot_prompt(question, choices, **_) -> tuple:
    """
    Explicit step-by-step reasoning ending with 'Answer: X'.
    Avoids <think></think> wrapper tags that were designed for Fin-R1's compact
    token format — Gemini 2.5 Pro expands them into many paragraphs which can
    exceed reasonable token limits before the final answer tag appears.
    The extractor reliably captures 'Answer: X' (regex priority 3).
    """
    labels, opts, valid = _labels_and_options(choices)
    system = ""
    user = (
        f"Answer this financial multiple-choice question using step-by-step reasoning.\n\n"
        f"Question: {question}\n\n{opts}\n\n"
        f"Instructions:\n"
        f"1. Briefly reason through each option.\n"
        f"2. Identify which option best answers the question.\n"
        f"3. On the very last line write exactly: Answer: <letter>\n"
        f"   where <letter> is one of {valid} — nothing else on that line.\n\n"
        f"Reasoning:"
    )
    return system, user


def build_role_prompt(question, choices, **_) -> tuple:
    labels, opts, valid = _labels_and_options(choices)
    user = (
        f"Answer this multiple-choice question. Respond with ONLY the letter ({valid}).\n\n"
        f"Question: {question}\n\n{opts}"
    )
    return _ROLE_SYSTEM, user


def build_meta_prompt(question, choices, **_) -> tuple:
    labels, opts, valid = _labels_and_options(choices)
    system = ""
    user = (
        "You will answer a financial multiple-choice question. "
        "Use the following reasoning strategy:\n"
        "1. Read all options carefully.\n"
        "2. Eliminate options that are clearly incorrect.\n"
        "3. For the remaining options, identify which one is most precisely correct "
        "according to standard financial principles.\n"
        "4. State your final answer as: Answer: <letter>\n\n"
        f"Valid answer letters: {valid}\n\n"
        f"Question: {question}\n\n{opts}"
    )
    return system, user


def build_self_consistency_prompt(question, choices, **_) -> tuple:
    """SC reuses the baseline prompt; diversity comes from temperature sampling."""
    return build_baseline_prompt(question, choices)


PROMPT_BUILDERS = {
    "baseline":         build_baseline_prompt,
    "few_shot":         build_few_shot_prompt,
    "strict_cot":       build_strict_cot_prompt,
    "role":             build_role_prompt,
    "self_consistency": build_self_consistency_prompt,
    "meta":             build_meta_prompt,
}

# ══════════════════════════════════════════════════════════════════════════════
# §4  ANSWER EXTRACTION  (same regex cascade as Fin-R1 notebook)
# ══════════════════════════════════════════════════════════════════════════════

def _extract_letter(text: str, valid_labels: list) -> str | None:
    """
    Extract the predicted letter from generated text.
    valid_labels must be UPPERCASE (e.g. ["A","B","C","D"]).
    Returns lowercase key or None.

    Search order (highest priority first):
      1. <answer>X</answer> (Fin-R1 structured format)
      2. $\\boxed{X}$ or \\boxed{X} (LaTeX — common in Gemini reasoning)
      3. "The final answer is $\\boxed{X}$" or "final answer is X" (last occurrence)
      4. Natural-language answer markers: Answer: X, The answer is X, etc. (last occurrence)
      5. Bracketed (X) or [X]
      6. Bold **X**
      7. Trailing letter on last non-empty line
      8. Last resort: first valid letter anywhere
    """
    text_up = text.upper().strip()

    def _last_match(pattern: str) -> str | None:
        """Return the last regex match group(1) in text_up that is a valid label."""
        hits = [
            m.group(1) for m in re.finditer(pattern, text_up)
            if m.group(1) in valid_labels
        ]
        return hits[-1] if hits else None

    # 1. <answer>X</answer> — structured tag format
    m = re.search(r"<ANSWER>\s*([A-Z])\s*</ANSWER>", text_up)
    if m and m.group(1) in valid_labels:
        return m.group(1).lower()

    # 1b. <answer>\boxed{X}</answer>
    m = re.search(r"<ANSWER>\s*\\BOXED\{([A-Z])\}\s*</ANSWER>", text_up)
    if m and m.group(1) in valid_labels:
        return m.group(1).lower()

    # 1c. <answer>X (unclosed — model cut off)
    m = re.search(r"<ANSWER>\s*([A-Z])", text_up)
    if m and m.group(1) in valid_labels:
        return m.group(1).lower()

    # 2. LaTeX boxed — $\boxed{X}$ or \boxed{X}  (last occurrence wins)
    letter = _last_match(r"\\BOXED\{([A-Z])\}")
    if letter:
        return letter.lower()

    # 3. "The final answer is" / "final answer:" (last occurrence)
    letter = _last_match(r"FINAL\s+ANSWER\s+IS\s*:?\s*\$?\\?BOXED\{([A-Z])\}\$?")
    if letter:
        return letter.lower()
    for pat in [
        r"FINAL\s+ANSWER\s*:\s*([A-Z])",
        r"THE\s+FINAL\s+ANSWER\s+IS\s*:?\s*([A-Z])",
        r"FINAL\s+ANSWER\s+IS\s*:?\s*([A-Z])",
    ]:
        letter = _last_match(pat)
        if letter:
            return letter.lower()

    # 4. Explicit natural-language markers (last occurrence)
    for pat in [
        r"ANSWER\s*:\s*([A-Z])",
        r"THE\s+ANSWER\s+IS\s*:?\s*([A-Z])",
        r"CORRECT\s+ANSWER\s*:\s*([A-Z])",
        r"THEREFORE[,\s]+THE\s+ANSWER\s+IS\s*:?\s*([A-Z])",
    ]:
        letter = _last_match(pat)
        if letter:
            return letter.lower()

    # 5. Bracketed: (B) or [B]  — last occurrence
    for pat in [r"\(([A-Z])\)", r"\[([A-Z])\]"]:
        letter = _last_match(pat)
        if letter:
            return letter.lower()

    # 6. Bold **X**  — last occurrence
    letter = _last_match(r"\*\*([A-Z])\*\*")
    if letter:
        return letter.lower()

    # 7. Trailing letter on last non-empty line
    for line in reversed(text_up.splitlines()):
        line = line.strip()
        if not line:
            continue
        m = re.search(r"[.\s]\s*([A-Z])\s*$", line)
        if m and m.group(1) in valid_labels:
            return m.group(1).lower()
        break

    # 8. Last resort: last valid letter anywhere in the text
    for ch in reversed(text_up):
        if ch in valid_labels:
            return ch.lower()

    return None

# ══════════════════════════════════════════════════════════════════════════════
# §5  OPENROUTER API CLIENT  (OpenAI-compatible, thread-safe, with retry)
# ══════════════════════════════════════════════════════════════════════════════

try:
    from openai import OpenAI, RateLimitError, APIStatusError, APIConnectionError
except ImportError:
    print("ERROR: openai not installed.  Run:  pip install openai")
    sys.exit(1)

# Thread-local clients avoid any shared-state concurrency issues.
_thread_local = threading.local()


def _get_client() -> "OpenAI":
    if not hasattr(_thread_local, "client"):
        _thread_local.client = OpenAI(
            api_key=OPENROUTER_API_KEY,
            base_url=OPENROUTER_BASE_URL,
        )
    return _thread_local.client


def _call_api(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.0,
    max_tokens: int = MAX_TOKENS_DEFAULT,
) -> str:
    """
    Single OpenRouter chat-completion call with exponential-backoff retry.
    Thread-safe — each thread owns its own OpenAI client instance.
    """
    client = _get_client()
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=MODEL_ID,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return (response.choices[0].message.content or "").strip()
        except RateLimitError:
            delay = RETRY_BASE_DELAY * (2 ** attempt)
            print(f"    [rate-limit retry {attempt+1}/{MAX_RETRIES}] waiting {delay:.0f}s", flush=True)
            time.sleep(delay)
        except (APIStatusError, APIConnectionError) as exc:
            if attempt < MAX_RETRIES - 1:
                delay = RETRY_BASE_DELAY * (2 ** attempt)
                print(f"    [api-error retry {attempt+1}/{MAX_RETRIES}] {exc} — waiting {delay:.0f}s", flush=True)
                time.sleep(delay)
            else:
                print(f"    [ERROR] giving up after {MAX_RETRIES} retries: {exc}", flush=True)
                return ""
        except Exception as exc:
            print(f"    [ERROR] unexpected: {exc}", flush=True)
            return ""
    return ""

# ══════════════════════════════════════════════════════════════════════════════
# §6  PARALLEL INFERENCE ENGINE
# ══════════════════════════════════════════════════════════════════════════════

_print_lock = threading.Lock()


def _build_prompt(i: int, row: dict, strategy: str, all_data: list) -> tuple:
    """Build (system, user, labels, max_tok) for a single question."""
    labels, _, _ = _labels_and_options(row["choices"])
    builder = PROMPT_BUILDERS[strategy]
    if strategy == "few_shot":
        system, user = builder(
            all_data,
            question=row["question"],
            choices=row["choices"],
            source=row["source"],
            row_idx=i,
        )
    else:
        system, user = builder(
            question=row["question"],
            choices=row["choices"],
            source=row["source"],
            row_idx=i,
        )
    max_tok = MAX_TOKENS_COT if strategy == "strict_cot" else MAX_TOKENS_DEFAULT
    return system, user, labels, max_tok


def _predict_one(i: int, row: dict, strategy: str, all_data: list) -> tuple:
    """
    Process a single question (non-SC strategies only).
    Returns (i, pred_key, raw_output_str).
    """
    system, user, labels, max_tok = _build_prompt(i, row, strategy, all_data)
    text = _call_api(system, user, temperature=TEMPERATURE_DEFAULT, max_tokens=max_tok)
    pred = _extract_letter(text, labels)
    if pred is None:
        fallback_user = (
            f"Answer with a single letter only ({'/'.join(labels)}).\n\n"
            f"Question: {row['question']}\n\n"
            + "\n".join(
                f"{labels[j]}. {row['choices'][j]}" for j in range(len(row["choices"]))
            )
        )
        text2 = _call_api("", fallback_user, temperature=0.0)
        pred = _extract_letter(text2, labels) or labels[0].lower()
    return i, pred, text


def _predict_one_sc_sample(i: int, sample_idx: int, row: dict, all_data: list) -> tuple:
    """
    Fetch a single SC sample for question i.
    Returns (i, sample_idx, letter_or_None, raw_text).
    """
    system, user, labels, max_tok = _build_prompt(i, row, "self_consistency", all_data)
    text = _call_api(system, user, temperature=TEMPERATURE_SC, max_tokens=max_tok)
    letter = _extract_letter(text, labels)
    return i, sample_idx, letter, text


def run_strategy(all_data: list, strategy: str) -> tuple:
    """
    Dispatches all questions (or all SC samples) to a ThreadPoolExecutor.
    Self-consistency: every sample is its own future — true parallelism.
    Returns (preds, errors, raw_outputs).
    """
    n              = len(all_data)
    preds          = [None] * n
    raw_outputs    = [None] * n
    correct_so_far = 0
    t_start        = time.monotonic()

    if strategy == "self_consistency":
        # ── SC: dispatch every individual sample as a separate future ──────
        # Total futures = n × SC_SAMPLES; majority-vote per question afterwards.
        n_samples = n * SC_SAMPLES
        all_texts  = [[None] * SC_SAMPLES for _ in range(n)]   # [q_idx][sample_idx]
        all_votes  = [[]               for _ in range(n)]
        completed  = 0   # counts completed *questions* (all samples gathered)
        samples_done = [0] * n  # how many samples done per question

        print(
            f"  [{strategy}] starting — {n} questions × {SC_SAMPLES} samples "
            f"= {n_samples} API calls  (workers={MAX_WORKERS})",
            flush=True,
        )

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {
                pool.submit(_predict_one_sc_sample, i, s, all_data[i], all_data): (i, s)
                for i in range(n)
                for s in range(SC_SAMPLES)
            }
            for future in as_completed(futures):
                qi, si, letter, text = future.result()
                all_texts[qi][si] = text
                if letter:
                    all_votes[qi].append(letter)
                samples_done[qi] += 1

                if samples_done[qi] == SC_SAMPLES:
                    # All samples for question qi are done — resolve
                    votes = all_votes[qi]
                    if votes:
                        preds[qi] = Counter(votes).most_common(1)[0][0]
                    else:
                        # No sample produced a valid letter — re-run greedy fallback
                        system, user, labels, max_tok = _build_prompt(
                            qi, all_data[qi], "self_consistency", all_data
                        )
                        text_fb = _call_api(system, user, temperature=0.0, max_tokens=max_tok)
                        preds[qi] = _extract_letter(text_fb, labels) or labels[0].lower()
                    raw_outputs[qi] = all_texts[qi]
                    completed += 1
                    if preds[qi] in all_data[qi]["gold"]:
                        correct_so_far += 1

                    elapsed = time.monotonic() - t_start
                    eta_str = _eta(elapsed, completed, n)
                    if completed % max(1, n // 20) == 0 or completed == n:
                        with _print_lock:
                            print(
                                f"  [{strategy}] [{completed:>5}/{n}]  "
                                f"acc: {correct_so_far/completed*100:.1f}%  "
                                f"{elapsed:.0f}s elapsed  ETA {eta_str}",
                                flush=True,
                            )
    else:
        # ── Non-SC: one future per question ───────────────────────────────
        completed = 0
        print(f"  [{strategy}] starting — {n} questions  (workers={MAX_WORKERS})", flush=True)
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {
                pool.submit(_predict_one, i, row, strategy, all_data): i
                for i, row in enumerate(all_data)
            }
            for future in as_completed(futures):
                i, pred, raw_output = future.result()
                preds[i]       = pred
                raw_outputs[i] = raw_output
                completed += 1
                if pred in all_data[i]["gold"]:
                    correct_so_far += 1

                elapsed = time.monotonic() - t_start
                eta_str = _eta(elapsed, completed, n)
                if completed % max(1, n // 20) == 0 or completed == n:
                    with _print_lock:
                        print(
                            f"  [{strategy}] [{completed:>5}/{n}]  "
                            f"acc: {correct_so_far/completed*100:.1f}%  "
                            f"{elapsed:.0f}s elapsed  ETA {eta_str}",
                            flush=True,
                        )

    correct = sum(1 for i, p in enumerate(preds) if p in all_data[i]["gold"])
    acc     = correct / n * 100
    elapsed = time.monotonic() - t_start
    print(f"  [{strategy}] FINAL — acc: {acc:.2f}%  ({correct}/{n})  {elapsed:.0f}s", flush=True)

    errors = [
        {
            "idx":       i,
            "source":    all_data[i]["source"],
            "gold":      all_data[i]["gold"],
            "predicted": preds[i],
            "strategy":  strategy,
        }
        for i, p in enumerate(preds)
        if p not in all_data[i]["gold"]
    ]
    return preds, errors, raw_outputs


def _eta(elapsed: float, done: int, total: int) -> str:
    """Return a human-readable ETA string."""
    if done == 0:
        return "?"
    remaining = elapsed / done * (total - done)
    if remaining < 60:
        return f"{remaining:.0f}s"
    if remaining < 3600:
        return f"{remaining/60:.1f}m"
    return f"{remaining/3600:.1f}h"

# ══════════════════════════════════════════════════════════════════════════════
# §7  RESULTS SAVING  (identical schema to finr1_<strategy>_results.json)
# ══════════════════════════════════════════════════════════════════════════════

def _build_src_indices(all_data: list) -> tuple:
    sources     = sorted(set(r["source"] for r in all_data))
    src_indices = {
        src: [i for i, r in enumerate(all_data) if r["source"] == src]
        for src in sources
    }
    return sources, src_indices


def save_strategy_json(
    all_data: list,
    min_size: int,
    strategy: str,
    preds: list,
    raw_outputs: list,
    sources: list,
    src_indices: dict,
) -> None:
    """Save gemini25_<strategy>_results.json — same layout as finr1_ files."""
    correct  = sum(1 for i, p in enumerate(preds) if p in all_data[i]["gold"])
    src_accs = {}
    for src in sources:
        idxs = src_indices[src]
        c    = sum(1 for i in idxs if preds[i] in all_data[i]["gold"])
        src_accs[src] = round(c / len(idxs) * 100, 2)

    output = {
        strategy: {
            "model":         MODEL_ID,
            "strategy":      strategy,
            "accuracy":      round(correct / len(all_data) * 100, 4),
            "correct":       int(correct),
            "total":         len(all_data),
            "balanced_size": min_size,
            "test_mode":     TEST_MODE,
            "random_seed":   RANDOM_SEED,
            "timestamp":     datetime.datetime.now(datetime.UTC).isoformat(),  # timezone-aware UTC
            "per_source":    src_accs,
            "questions": [
                {
                    "idx":        i,
                    "source":     all_data[i]["source"],
                    "question":   all_data[i]["question"],
                    "choices":    all_data[i]["choices"],
                    "gold":       all_data[i]["gold"],
                    "predicted":  preds[i],
                    "correct":    preds[i] in all_data[i]["gold"],
                    "raw_output": raw_outputs[i],
                }
                for i in range(len(all_data))
            ],
        }
    }

    fname_json = f"gemini25_{strategy}_results.json"
    with open(fname_json, "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2, ensure_ascii=False)
    print(f"  Saved: {fname_json}")


def save_combined_outputs(
    all_data: list,
    min_size: int,
    all_results: dict,
    all_errors: dict,
) -> None:
    """Save cartesian CSV, heatmap, and error CSV across all completed strategies."""
    sources, src_indices = _build_src_indices(all_data)

    try:
        import pandas as pd

        table_rows = []
        for strat, res in all_results.items():
            preds    = res["predictions"]
            row_dict = {"strategy": strat}
            for src in sources:
                idxs          = src_indices[src]
                c             = sum(1 for i in idxs if preds[i] in all_data[i]["gold"])
                row_dict[src] = round(c / len(idxs) * 100, 1)
            c_all               = sum(1 for i, p in enumerate(preds) if p in all_data[i]["gold"])
            row_dict["OVERALL"] = round(c_all / len(all_data) * 100, 1)
            table_rows.append(row_dict)

        df_cart = pd.DataFrame(table_rows).set_index("strategy")

        print("\n" + "=" * 80)
        print("CARTESIAN RESULTS: Accuracy (%) — Strategy × Dataset")
        print("=" * 80)
        print(df_cart.to_string())
        print()
        print("Strategy ranking by OVERALL accuracy:")
        ranked = df_cart["OVERALL"].sort_values(ascending=False)
        for rank, (strat, acc) in enumerate(ranked.items(), 1):
            marker = "  ◄ best" if rank == 1 else ""
            print(f"  {rank}. {strat:<22}  {acc:.1f}%{marker}")

        fname_csv = "gemini25_prompt_engineering_cartesian.csv"
        df_cart.reset_index().to_csv(fname_csv, index=False)
        print(f"\nSaved: {fname_csv}")

        try:
            import seaborn as sns
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(
                figsize=(max(8, len(sources) * 1.5), max(4, len(df_cart) * 0.8))
            )
            sns.heatmap(
                df_cart,
                annot=True, fmt=".1f",
                cmap="RdYlGn",
                linewidths=0.5,
                linecolor="white",
                vmin=max(0,   df_cart.values.min() - 5),
                vmax=min(100, df_cart.values.max() + 5),
                ax=ax,
            )
            ax.set_title(
                "Accuracy (%) — Strategy × Dataset  (Gemini 2.5 Pro via OpenRouter)",
                fontsize=12, pad=12,
            )
            ax.set_xlabel("Dataset / OVERALL", fontsize=10)
            ax.set_ylabel("Prompt Strategy", fontsize=10)
            plt.xticks(rotation=30, ha="right")
            plt.tight_layout()
            fname_heatmap = "gemini25_heatmap.png"
            plt.savefig(fname_heatmap, dpi=150, bbox_inches="tight")
            print(f"Saved: {fname_heatmap}")
        except ImportError:
            print("[heatmap] install seaborn + matplotlib for the heatmap")

        error_rows = []
        for strat, errs in all_errors.items():
            for e in errs:
                row_data = all_data[e["idx"]]
                error_rows.append({
                    "strategy":  strat,
                    "idx":       e["idx"],
                    "source":    e["source"],
                    "language":  _SOURCE_LANG.get(e["source"], "?"),
                    "gold":      "|".join(e["gold"]),
                    "predicted": e["predicted"],
                    "question":  row_data["question"][:200],
                })
        df_errors = pd.DataFrame(error_rows)
        fname_errors = "gemini25_prompt_engineering_errors.csv"
        df_errors.to_csv(fname_errors, index=False)
        print(f"Saved: {fname_errors}  ({len(df_errors)} error rows across {len(all_results)} strategies)")

    except ImportError:
        print("[pandas not installed — skipping CSV/heatmap outputs]")

# ══════════════════════════════════════════════════════════════════════════════
# §8  MAIN  — all strategies in sequence, questions parallelised per strategy
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    if not OPENROUTER_API_KEY:
        print("ERROR: OPENROUTER_API_KEY is not set.")
        sys.exit(1)

    print("=" * 65)
    print("Gemini 2.5 Pro via OpenRouter — Financial MCQ Prompt Engineering")
    print("=" * 65)
    print(f"MODEL            : {MODEL_ID}")
    print(f"STRATEGIES       : {', '.join(VALID_STRATEGIES)}")
    print(f"TEST_MODE        : {TEST_MODE}" + (f"  ({N_TEST} sample/dataset)" if TEST_MODE else ""))
    print(f"SC_SAMPLES       : {SC_SAMPLES}")
    print(f"MAX_WORKERS      : {MAX_WORKERS}  (parallel API calls per strategy)")
    print()

    datasets_raw, all_data, min_size = load_all_datasets()
    sources, src_indices = _build_src_indices(all_data)

    all_results: dict = {}
    all_errors:  dict = {}

    total_t0 = time.time()

    for strategy in VALID_STRATEGIES:
        print()
        print("=" * 65)
        print(f"STRATEGY: {strategy}  |  questions: {len(all_data)}")
        print("=" * 65)

        t0 = time.time()
        preds, errors, raw_outputs = run_strategy(all_data, strategy)
        elapsed = time.time() - t0

        correct  = sum(1 for i, p in enumerate(preds) if p in all_data[i]["gold"])
        accuracy = round(correct / len(all_data) * 100, 4)

        all_results[strategy] = {"predictions": preds, "accuracy": accuracy, "raw_outputs": raw_outputs}
        all_errors[strategy]  = errors

        print(f"  Elapsed  : {elapsed:.0f}s")
        print(f"  Accuracy : {accuracy:.2f}%  ({correct}/{len(all_data)})")
        print(f"  Errors   : {len(errors)}")

        # Save immediately — safe against a crash on a later strategy
        save_strategy_json(all_data, min_size, strategy, preds, raw_outputs, sources, src_indices)

    total_elapsed = time.time() - total_t0

    print()
    print("=" * 65)
    print(f"ALL STRATEGIES COMPLETE  ({total_elapsed:.0f}s total)")
    print("=" * 65)
    for strat, res in all_results.items():
        n_err = len(all_errors.get(strat, []))
        print(f"  {strat:<22}  {res['accuracy']:.2f}%   ({n_err} errors)")

    save_combined_outputs(all_data, min_size, all_results, all_errors)
    print("\nDONE.")


if __name__ == "__main__":
    main()
