#!/usr/bin/env python3
"""
clef_task1_data_analysis.py
────────────────────────────────────────────────────────────────────────────
Comprehensive dataset analysis for the Financial-MCQ Benchmark project.

Analyses every source dataset individually, then merges them for a combined
view.  All plots are saved to        plots/
All structured results are saved to  analysis_results/

Run:
    python clef_task1_data_analysis.py
────────────────────────────────────────────────────────────────────────────
"""

import os
import json
import warnings
from collections import Counter
from pathlib import Path

import requests
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # non-interactive backend — safe for scripts
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# GitHub authentication — datasets are fetched from hassan09070/clef_task
# WARNING: do not commit this token to a public repository.
# Prefer setting the GITHUB_TOKEN environment variable instead.
# ─────────────────────────────────────────────────────────────────────────────
_GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
_GITHUB_REPO  = "hassan09070/clef_task"
_GITHUB_FOLDER = "task1_data"


def _github_fetch(filename: str) -> list:
    """
    Fetch a JSON file from the private GitHub repo via the Contents API.
    Handles files of any size:
      - ≤1 MB  → inline base64 content in the API response
      - >1 MB  → API returns download_url; fetched with auth header
    """
    import base64
    api_url = (
        f"https://api.github.com/repos/{_GITHUB_REPO}/contents/"
        f"{_GITHUB_FOLDER}/{filename}"
    )
    headers = {
        "Accept":        "application/vnd.github+json",
        "Authorization": f"token {_GITHUB_TOKEN}",
    }
    meta = requests.get(api_url, headers=headers, timeout=60)
    meta.raise_for_status()
    data = meta.json()

    content_b64 = data.get("content", "").replace("\n", "")
    if content_b64:
        # Small file — inline base64
        return json.loads(base64.b64decode(content_b64).decode("utf-8"))

    # Large file (>1 MB) — use the download_url returned by the API
    download_url = data.get("download_url")
    if not download_url:
        raise ValueError(f"No content or download_url for {filename}")
    raw = requests.get(download_url, headers={"Authorization": f"token {_GITHUB_TOKEN}"}, timeout=120)
    raw.raise_for_status()
    return raw.json()

# ─────────────────────────────────────────────────────────────────────────────
# Output directories
# ─────────────────────────────────────────────────────────────────────────────

PLOTS_DIR   = Path("plots")
RESULTS_DIR = Path("analysis_results")
PLOTS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# Answer label mapping used throughout
_LABEL = ["A", "B", "C", "D"]

# ─────────────────────────────────────────────────────────────────────────────
# 1.  DATA LOADING  — pre-cleaned JSON files fetched from GitHub
# ─────────────────────────────────────────────────────────────────────────────

# Mapping from analysis source_name → filename in the task1_data/ folder.
# Add new datasets here; everything else (analysis, plots, summary) picks them
# up automatically via load_all_datasets().
_DATASET_FILES = {
    "cfa_cpa":           "task1_Tomas08119993_finmmeval-cfa-cpa.json",
    "es_multifin":       "task1_TheFinAI_flare-es-multifin.json",
    "plutus":            "task1_TheFinAI_plutus-multifin.json",
    "arabic_accounting": "task1_SahmBenchmark_arabic-accounting-mcq.json",
    "arabic_business":   "task1_SahmBenchmark_arabic-business-mcq.json",
    "hindi_finance":    "task1_bharatgenai_BhashaBench-Finance-Hindi.json"
}


def load_dataset_file(filename: str, source_name: str) -> pd.DataFrame:
    """
    Load a pre-cleaned task1 JSON file and return a DataFrame with columns
    question, choices, answer, source.

    Resolution order:
      1. Local file in the current working directory (fastest, always works)
      2. GitHub Contents API with auth (works for private repo, any file size)
    """
    local_path = Path(filename)
    if local_path.exists():
        with open(local_path, "r", encoding="utf-8") as f:
            records = json.load(f)
    else:
        records = _github_fetch(filename)

    rows = []
    for rec in records:
        options = rec.get("options") or {}
        if not options:
            continue
        sorted_keys = sorted(options.keys())          # e.g. ['a','b','c','d']
        choices     = [options[k] for k in sorted_keys]
        gold        = rec.get("gold") or []
        if not gold:
            continue
        gold_letter = str(gold[0]).lower()
        if gold_letter not in sorted_keys:
            continue
        rows.append({
            "question": str(rec.get("question") or ""),
            "choices":  choices,
            "answer":   sorted_keys.index(gold_letter),
            "source":   source_name,
        })
    return pd.DataFrame(rows)


def _normalize_choices(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure the choices column always contains plain Python lists.
    HuggingFace to_pandas() may return numpy arrays or other iterables
    depending on the dataset schema; normalise them upfront so every
    downstream isinstance(choices, list) check works correctly.
    """
    df = df.copy()
    df["choices"] = df["choices"].apply(
        lambda x: list(x) if x is not None and not isinstance(x, list) else (x if isinstance(x, list) else [])
    )
    return df


def load_all_datasets() -> dict:
    """
    Load every pre-cleaned task1 JSON (local first, GitHub fallback) and return
    { source_name: pd.DataFrame } with columns question, choices, answer, source.
    """
    print("Loading datasets …")

    dfs = {}
    for source_name, filename in _DATASET_FILES.items():
        print(f"  loading {source_name} …", end=" ", flush=True)
        try:
            df = _normalize_choices(load_dataset_file(filename, source_name))
            dfs[source_name] = df
            print(f"{len(df)} rows")
        except Exception as exc:
            print(f"FAILED ({exc})")

    return dfs


# ─────────────────────────────────────────────────────────────────────────────
# 2.  PER-DATASET ANALYSIS FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def analyse_missing(df: pd.DataFrame) -> dict:
    """Count missing / null values in the question, choices, and answer fields."""
    missing = {
        "question": int(df["question"].isna().sum()),
        "choices":  int(df["choices"].isna().sum()),
        "answer":   int(df["answer"].isna().sum()),
    }
    # Also surface rows where any individual option text is None / empty string
    empty_options = int(
        df["choices"].apply(
            lambda lst: any(
                c is None or str(c).strip() == ""
                for c in (list(lst) if lst is not None and hasattr(lst, "__iter__") else [])
            )
        ).sum()
    )
    missing["empty_option_text"] = empty_options
    return missing


def analyse_answer_distribution(df: pd.DataFrame) -> dict:
    """Counts and percentages for each answer index (0-3 → A-D)."""
    counts = Counter(df["answer"].tolist())
    total  = len(df)
    dist   = {}
    for idx, label in enumerate(_LABEL):
        c = counts.get(idx, 0)
        dist[label] = {"count": c, "pct": round(c / total * 100, 2) if total else 0.0}
    return dist


def analyse_question_length(df: pd.DataFrame) -> dict:
    """Descriptive statistics on question word-count and character-count."""
    word_counts = df["question"].dropna().apply(lambda q: len(str(q).split()))
    char_counts = df["question"].dropna().apply(lambda q: len(str(q)))

    stats = {}
    for name, series in [("word_count", word_counts), ("char_count", char_counts)]:
        stats[name] = {
            "min":    int(series.min()),
            "max":    int(series.max()),
            "mean":   round(float(series.mean()), 2),
            "median": round(float(series.median()), 2),
            "std":    round(float(series.std()),  2),
        }
    return stats


def analyse_option_lengths(df: pd.DataFrame) -> dict:
    """Compare average character length of correct vs incorrect options."""
    correct_lens   = []
    incorrect_lens = []

    for _, row in df.iterrows():
        choices = row["choices"]
        answer  = row["answer"]
        if choices is None or not hasattr(choices, "__iter__"):
            continue
        choices = list(choices)  # normalise in case not already a list
        for i, ch in enumerate(choices[:4]):
            length = len(str(ch))
            if i == answer:
                correct_lens.append(length)
            else:
                incorrect_lens.append(length)

    all_lens = correct_lens + incorrect_lens
    return {
        "correct_avg":    round(np.mean(correct_lens),   2) if correct_lens   else None,
        "incorrect_avg":  round(np.mean(incorrect_lens), 2) if incorrect_lens else None,
        "all_option_avg": round(np.mean(all_lens),       2) if all_lens       else None,
    }


def analyse_option_count_consistency(df: pd.DataFrame) -> dict:
    """Verify every question has exactly 4 choices."""
    counts = df["choices"].apply(
        lambda lst: len(lst) if hasattr(lst, "__len__") and lst is not None else 0
    )
    return {
        "total":             len(df),
        "consistent_4":      int((counts == 4).sum()),
        "inconsistent":      int((counts != 4).sum()),
        "count_distribution": dict(Counter(counts.tolist())),
    }


def analyse_duplicates(df: pd.DataFrame) -> dict:
    """Detect duplicate questions by exact string match."""
    qs = df["question"].dropna().tolist()
    total_dups = int(len(qs) - len(set(qs)))
    dup_examples = (
        df["question"][df["question"].duplicated(keep=False)]
        .unique()
        .tolist()[:5]   # surface first 5 examples only
    )
    return {
        "total_duplicates": total_dups,
        "examples":         dup_examples,
    }


def analyse_dataset(df: pd.DataFrame, source_name: str) -> dict:
    """Run all per-dataset analyses and return a single results dict."""
    print(f"  analysing {source_name} ({len(df)} rows) …")
    return {
        "source":             source_name,
        "total_questions":    len(df),
        "missing":            analyse_missing(df),
        "answer_distribution": analyse_answer_distribution(df),
        "question_length":    analyse_question_length(df),
        "option_lengths":     analyse_option_lengths(df),
        "option_consistency": analyse_option_count_consistency(df),
        "duplicates":         analyse_duplicates(df),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3.  COMBINED ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def analyse_combined(combined_df: pd.DataFrame, per_dataset_results: list) -> dict:
    """Global analysis across the merged dataset, including bias checks."""
    print("Running combined analysis …")

    answer_dist = analyse_answer_distribution(combined_df)
    q_len       = analyse_question_length(combined_df)
    opt_len     = analyse_option_lengths(combined_df)

    # Dataset size comparison
    sizes = {r["source"]: r["total_questions"] for r in per_dataset_results}

    # Bias: which answer option is most frequent?
    counts      = [answer_dist[l]["count"] for l in _LABEL]
    total       = sum(counts)
    dominant_i  = int(np.argmax(counts))
    dominant    = _LABEL[dominant_i]
    dominant_pct = round(max(counts) / total * 100, 2) if total else 0.0

    return {
        "total_questions":    len(combined_df),
        "answer_distribution": answer_dist,
        "question_length":    q_len,
        "option_lengths":     opt_len,
        "dataset_sizes":      sizes,
        "bias": {
            "dominant_option":                dominant,
            "dominant_option_pct":            dominant_pct,
            "correct_longer_than_incorrect":  (
                (opt_len["correct_avg"] or 0) > (opt_len["incorrect_avg"] or 0)
            ),
            "correct_avg_len":   opt_len["correct_avg"],
            "incorrect_avg_len": opt_len["incorrect_avg"],
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4.  SIDE-BY-SIDE DATASET COMPARISON
# ─────────────────────────────────────────────────────────────────────────────

def compare_datasets(per_dataset_results: list) -> pd.DataFrame:
    """
    Return a tidy DataFrame with one row per dataset, comparing key metrics:
    size, missing values, answer % per label, avg question/option lengths.
    """
    rows = []
    for r in per_dataset_results:
        ad = r["answer_distribution"]
        ql = r["question_length"]
        rows.append({
            "source":            r["source"],
            "total_questions":   r["total_questions"],
            "missing_q":         r["missing"]["question"],
            "missing_a":         r["missing"]["answer"],
            "pct_A":             ad["A"]["pct"],
            "pct_B":             ad["B"]["pct"],
            "pct_C":             ad["C"]["pct"],
            "pct_D":             ad["D"]["pct"],
            "avg_q_words":       ql["word_count"]["mean"],
            "avg_q_chars":       ql["char_count"]["mean"],
            "correct_opt_len":   r["option_lengths"]["correct_avg"],
            "incorrect_opt_len": r["option_lengths"]["incorrect_avg"],
            "duplicates":        r["duplicates"]["total_duplicates"],
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 5.  VISUALIZATION
# ─────────────────────────────────────────────────────────────────────────────

sns.set_theme(style="whitegrid", palette="muted")


def _save(fig: plt.Figure, name: str) -> None:
    """Save a figure to the plots directory and close it."""
    path = PLOTS_DIR / name
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"    saved → {path}")


def plot_answer_distribution_per_dataset(per_dataset_results: list) -> None:
    """Grouped bar charts (one per dataset) showing A/B/C/D answer percentages."""
    n     = len(per_dataset_results)
    ncols = 3
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = axes.flatten()

    for ax, r in zip(axes, per_dataset_results):
        labels = list(r["answer_distribution"].keys())
        pcts   = [r["answer_distribution"][l]["pct"] for l in labels]
        bars   = ax.bar(labels, pcts, color=sns.color_palette("muted", 4))

        # 25 % random-baseline line
        ax.axhline(25, color="red", linestyle="--", linewidth=0.8, label="25% baseline")
        ax.set_title(r["source"], fontsize=10)
        ax.set_ylabel("% of answers")
        ax.set_ylim(0, max(pcts) * 1.25 + 5)

        for bar, pct in zip(bars, pcts):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{pct:.1f}%",
                ha="center", va="bottom", fontsize=8,
            )
        ax.legend(fontsize=7)

    # Hide unused subplot slots
    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle("Answer Distribution per Dataset", fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save(fig, "answer_dist_per_dataset.png")


def plot_answer_distribution_combined(combined_results: dict) -> None:
    """Bar chart for the combined (all-dataset) answer distribution."""
    ad     = combined_results["answer_distribution"]
    labels = list(ad.keys())
    pcts   = [ad[l]["pct"]   for l in labels]
    counts = [ad[l]["count"] for l in labels]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, pcts, color=sns.color_palette("muted", 4))
    ax.axhline(25, color="red", linestyle="--", linewidth=1.2, label="25% random baseline")
    ax.set_title("Combined Answer Distribution (All Datasets)", fontsize=13)
    ax.set_ylabel("% of questions")

    for bar, pct, cnt in zip(bars, pcts, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{pct:.1f}%\n(n={cnt})",
            ha="center", va="bottom", fontsize=9,
        )
    ax.legend()
    fig.tight_layout()
    _save(fig, "answer_dist_combined.png")


def plot_question_length_histogram(dfs: dict) -> None:
    """Overlapping word-count histograms, one series per dataset."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for source, df in dfs.items():
        word_counts = df["question"].dropna().apply(lambda q: len(str(q).split()))
        word_counts.plot.hist(ax=ax, bins=40, alpha=0.5, label=source)

    ax.set_title("Question Length Distribution (word count per dataset)", fontsize=13)
    ax.set_xlabel("Word count")
    ax.set_ylabel("Frequency")
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, "question_length_histogram.png")


def plot_option_length_boxplot(dfs: dict) -> None:
    """
    Two boxplots:
      1. Option character-length distribution per dataset.
      2. Correct vs. incorrect option lengths (all datasets combined).
    """
    records = []
    for source, df in dfs.items():
        for _, row in df.iterrows():
            choices = row["choices"]
            if choices is None or not hasattr(choices, "__iter__"):
                continue
            choices = list(choices)
            for i, ch in enumerate(choices[:4]):
                records.append({
                    "source":   source,
                    "option":   _LABEL[i],
                    "length":   len(str(ch)),
                    "correctness": "Correct" if i == row["answer"] else "Incorrect",
                })

    if not records:
        print("    skipping option length boxplot — no valid choices data")
        return

    plot_df = pd.DataFrame(records)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    sources = sorted(plot_df["source"].unique())
    palette_src = sns.color_palette("muted", len(sources))

    # Per-dataset distribution
    sns.boxplot(
        data=plot_df,
        x="source", y="length",
        order=sources,
        palette=palette_src,
        ax=axes[0],
        showfliers=False,
    )
    axes[0].set_title("Option Length by Dataset", fontsize=12)
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Character count")
    axes[0].tick_params(axis="x", rotation=30)

    # Correct vs incorrect — use a string column so seaborn is happy
    sns.boxplot(
        data=plot_df,
        x="correctness", y="length",
        order=["Incorrect", "Correct"],
        palette={"Incorrect": "#e07070", "Correct": "#70a8e0"},
        ax=axes[1],
        showfliers=False,
    )
    axes[1].set_title("Correct vs Incorrect Option Lengths (Combined)", fontsize=12)
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Character count")

    fig.tight_layout()
    _save(fig, "option_length_boxplot.png")


def plot_dataset_size_comparison(per_dataset_results: list) -> None:
    """Horizontal bar chart comparing dataset sizes (sorted descending)."""
    pairs   = sorted(
        [(r["total_questions"], r["source"]) for r in per_dataset_results],
        reverse=True,
    )
    sizes, sources = zip(*pairs)

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(sources, sizes, color=sns.color_palette("muted", len(sources)))
    ax.set_title("Dataset Size Comparison", fontsize=13)
    ax.set_xlabel("Number of questions")

    for bar, sz in zip(bars, sizes):
        ax.text(
            bar.get_width() + max(sizes) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            str(sz), va="center", fontsize=9,
        )
    fig.tight_layout()
    _save(fig, "dataset_size_comparison.png")


# ─────────────────────────────────────────────────────────────────────────────
# 6.  TERMINAL SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(
    per_dataset_results: list,
    combined_results: dict,
    comparison_df: pd.DataFrame,
) -> None:
    SEP = "=" * 70

    print(f"\n{SEP}")
    print("  PER-DATASET SUMMARY")
    print(SEP)

    for r in per_dataset_results:
        print(f"\n  ▶  {r['source']}  ({r['total_questions']} questions)")

        m = r["missing"]
        print(
            f"     Missing     : question={m['question']}, "
            f"answer={m['answer']}, empty_option={m['empty_option_text']}"
        )

        ad = r["answer_distribution"]
        dist_str = "  ".join(f"{l}:{ad[l]['pct']:.1f}%" for l in _LABEL)
        print(f"     Ans dist    : {dist_str}")

        ql = r["question_length"]["word_count"]
        print(
            f"     Q length    : mean={ql['mean']} words, "
            f"median={ql['median']}, min={ql['min']}, max={ql['max']}"
        )

        ol = r["option_lengths"]
        print(
            f"     Opt len     : correct={ol['correct_avg']} chars, "
            f"incorrect={ol['incorrect_avg']} chars"
        )

        oc = r["option_consistency"]
        print(f"     Choices ≠ 4 : {oc['inconsistent']} rows")
        print(f"     Duplicates  : {r['duplicates']['total_duplicates']}")

    print(f"\n{SEP}")
    print("  COMBINED ANALYSIS")
    print(SEP)

    cr = combined_results
    print(f"  Total questions   : {cr['total_questions']}")

    ad = cr["answer_distribution"]
    dist_str = "  ".join(f"{l}:{ad[l]['pct']:.1f}%" for l in _LABEL)
    print(f"  Ans distribution  : {dist_str}")

    b = cr["bias"]
    print(f"  Dominant option   : {b['dominant_option']} ({b['dominant_option_pct']}%)")
    print(
        f"  Correct longer?   : {b['correct_longer_than_incorrect']}  "
        f"(correct={b['correct_avg_len']} chars vs "
        f"incorrect={b['incorrect_avg_len']} chars)"
    )

    print(f"\n{SEP}")
    print("  SIDE-BY-SIDE COMPARISON")
    print(SEP)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    print(comparison_df.to_string(index=False))


# ─────────────────────────────────────────────────────────────────────────────
# 7.  SAVE RESULTS
# ─────────────────────────────────────────────────────────────────────────────

def _to_serializable(obj):
    """Recursively cast numpy scalars to Python native types for JSON."""
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_serializable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def save_results(
    per_dataset_results: list,
    combined_results: dict,
    comparison_df: pd.DataFrame,
) -> None:
    # Per-dataset analysis → JSON
    path_per = RESULTS_DIR / "per_dataset_analysis.json"
    with open(path_per, "w", encoding="utf-8") as f:
        json.dump(_to_serializable(per_dataset_results), f, indent=2, ensure_ascii=False)
    print(f"  saved → {path_per}")

    # Combined analysis → JSON
    path_comb = RESULTS_DIR / "combined_analysis.json"
    with open(path_comb, "w", encoding="utf-8") as f:
        json.dump(_to_serializable(combined_results), f, indent=2, ensure_ascii=False)
    print(f"  saved → {path_comb}")

    # Side-by-side comparison → CSV
    path_csv = RESULTS_DIR / "dataset_comparison.csv"
    comparison_df.to_csv(path_csv, index=False)
    print(f"  saved → {path_csv}")


# ─────────────────────────────────────────────────────────────────────────────
# 8.  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    # ── Load all datasets ─────────────────────────────────────────────────
    dfs = load_all_datasets()
    if not dfs:
        print("No datasets loaded. Exiting.")
        return

    # ── Per-dataset analysis ──────────────────────────────────────────────
    print("\nRunning per-dataset analysis …")
    per_dataset_results = [
        analyse_dataset(df, source) for source, df in dfs.items()
    ]

    # ── Merge into one combined DataFrame ────────────────────────────────
    combined_df = pd.concat(list(dfs.values()), ignore_index=True)

    # ── Combined analysis ─────────────────────────────────────────────────
    combined_results = analyse_combined(combined_df, per_dataset_results)

    # ── Side-by-side comparison table ────────────────────────────────────
    comparison_df = compare_datasets(per_dataset_results)

    # ── Terminal summary ─────────────────────────────────────────────────
    print_summary(per_dataset_results, combined_results, comparison_df)

    # ── Generate and save plots ───────────────────────────────────────────
    print("\nGenerating plots …")
    plot_answer_distribution_per_dataset(per_dataset_results)
    plot_answer_distribution_combined(combined_results)
    plot_question_length_histogram(dfs)
    plot_option_length_boxplot(dfs)
    plot_dataset_size_comparison(per_dataset_results)

    # ── Save structured results ───────────────────────────────────────────
    print("\nSaving results …")
    save_results(per_dataset_results, combined_results, comparison_df)

    print("\nDone.")


if __name__ == "__main__":
    main()
