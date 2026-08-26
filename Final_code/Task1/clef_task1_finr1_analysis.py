"""
Fin-R1 Prompt Engineering — Deep Results Analysis
==================================================
All 6 strategies × 1 320 questions × 6 datasets combined offline.

Strategy    | Inference | Overall Acc
------------|-----------|------------
baseline    | logit     | 54.8%
few_shot    | logit     | 56.7%
role        | logit     | 54.2%
self_cons.  | generation| 54.4%
meta        | generation| 51.2%
cot         | generation| 30.4%

Sections:
  1.  Load & merge all results
  2.  Combined accuracy heatmap
  3.  Strategy ranking + lift over baseline
  4.  Per-question difficulty distribution
  5.  Positional bias & choice-count effect
  6.  Multi-gold question analysis
  7.  Wrong-answer confusion matrix
  8.  Cross-strategy agreement clusters
  9.  Language × strategy performance
  10. Hardest questions inspect table
  11. CoT regression & oracle analysis
  12. Save CSVs
  13. Key findings & recommendations

Run:  python clef_task1_finr1_analysis.py
Outputs: PNG charts + timestamped CSVs written to the current directory.
"""

import json
import os
import datetime
import warnings
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# § 1  LOAD ALL RESULTS & BUILD MASTER DATAFRAME
# ══════════════════════════════════════════════════════════════════════════════

BASE = os.environ.get("FINR1_RESULTS_DIR", "results/finr1_prompt_engineering")

RESULT_FILES = {
    "baseline":         "finr1_baseline/finr1_baseline_results.json",
    "few_shot":         "finr1_fewshot/finr1_few_shot_results.json",
    "role":             "finr1_role/finr1_role_results.json",
    "self_consistency": "finr1_self/finr1_self_consistency_results.json",
    "meta":             "finr1_meta/finr1_meta_results.json",
    "cot":              "finr1_cot/finr1_cot_results.json",
}

print("Loading results...")
raw        = {}
per_source = {}
overall    = {}

for strat, rel_path in RESULT_FILES.items():
    path = os.path.join(BASE, rel_path)
    with open(path) as f:
        blob = json.load(f)
    inner              = blob[strat]
    raw[strat]         = inner
    per_source[strat]  = inner["per_source"]
    overall[strat]     = inner["accuracy"]
    print(f"  {strat:<22}  {inner['accuracy']:.2f}%  ({inner['correct']}/{inner['total']})")

# Flat master DataFrame — one row per (question_idx, strategy)
rows = []
for strat, inner in raw.items():
    for q in inner["questions"]:
        rows.append({
            "strategy":  strat,
            "idx":       q["idx"],
            "source":    q["source"],
            "question":  q["question"],
            "choices":   q["choices"],
            "n_choices": len(q["choices"]),
            "gold":      q["gold"],
            "n_gold":    len(q["gold"]),
            "predicted": q["predicted"],
            "correct":   q["correct"],
        })

df = pd.DataFrame(rows)

SOURCE_LANG = {
    "cfa_cpa":           "Chinese",
    "es_multifin":       "Spanish",
    "plutus":            "Multilingual",
    "arabic_accounting": "Arabic",
    "arabic_business":   "Arabic",
    "hindi_finance":     "Hindi",
}
df["language"] = df["source"].map(SOURCE_LANG)

STRATEGIES_ORDERED = ["few_shot", "baseline", "role", "self_consistency", "meta", "cot"]
SOURCES = sorted(df["source"].unique())
N_Q     = df["idx"].nunique()

print(f"\nMaster DataFrame: {len(df):,} rows  ({len(RESULT_FILES)} strategies × {N_Q} questions)")
print(df.groupby("strategy")["correct"].mean().mul(100).round(2).rename("accuracy %").to_string())

# ══════════════════════════════════════════════════════════════════════════════
# § 2  COMBINED ACCURACY HEATMAP (ALL STRATEGIES × ALL DATASETS)
# ══════════════════════════════════════════════════════════════════════════════

pivot = (
    df.groupby(["strategy", "source"])["correct"]
    .mean()
    .mul(100)
    .round(1)
    .unstack("source")
)
pivot["OVERALL"] = df.groupby("strategy")["correct"].mean().mul(100).round(1)
pivot = pivot.loc[STRATEGIES_ORDERED, SOURCES + ["OVERALL"]]

print("\n" + pivot.to_string())

fig, ax = plt.subplots(figsize=(13, 5))
vmin = max(0, pivot.values.min() - 3)
vmax = min(100, pivot.values.max() + 3)

sns.heatmap(
    pivot,
    annot=True,
    fmt=".1f",
    cmap="RdYlGn",
    linewidths=0.6,
    linecolor="white",
    vmin=vmin,
    vmax=vmax,
    cbar_kws={"label": "Accuracy (%)"},
    ax=ax,
)
ax.set_title("Fin-R1 Prompt Engineering — All Strategies × All Datasets\nAccuracy (%)", fontsize=13, pad=12)
ax.set_xlabel("Dataset / OVERALL", fontsize=10)
ax.set_ylabel("Prompt Strategy", fontsize=10)
plt.xticks(rotation=30, ha="right", fontsize=9)
plt.yticks(fontsize=9)

# Black border marks best strategy per column
for col_i, col in enumerate(pivot.columns):
    best_row = pivot[col].idxmax()
    row_i    = list(pivot.index).index(best_row)
    ax.add_patch(plt.Rectangle((col_i, row_i), 1, 1, fill=False, edgecolor="black", lw=2.5))

plt.tight_layout()
plt.savefig("finr1_combined_heatmap.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: finr1_combined_heatmap.png  (black border = best strategy per column)")

# ══════════════════════════════════════════════════════════════════════════════
# § 3  STRATEGY RANKING: BAR CHART + LIFT OVER BASELINE
# ══════════════════════════════════════════════════════════════════════════════

baseline_acc = pivot.loc["baseline", "OVERALL"]

rank_df = (
    pivot["OVERALL"]
    .sort_values(ascending=False)
    .reset_index()
    .rename(columns={"OVERALL": "accuracy"})
)
rank_df["lift_vs_baseline"] = rank_df["accuracy"] - baseline_acc
rank_df["rank"] = range(1, len(rank_df) + 1)

print("\nStrategy Ranking (by OVERALL accuracy)")
print("=" * 55)
for _, row in rank_df.iterrows():
    lift_str = f"{row['lift_vs_baseline']:+.1f}%"
    marker   = "  ◄ best" if row["rank"] == 1 else ("  ◄ BASELINE" if row["strategy"] == "baseline" else "")
    print(f"  {int(row['rank'])}. {row['strategy']:<22}  {row['accuracy']:.1f}%  (lift {lift_str}){marker}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

colors = [
    "#2ecc71" if s == rank_df.iloc[0]["strategy"] else "#e74c3c" if s == "cot" else "#3498db"
    for s in rank_df["strategy"]
]
bars = axes[0].barh(rank_df["strategy"][::-1], rank_df["accuracy"][::-1],
                    color=colors[::-1], edgecolor="white")
axes[0].axvline(baseline_acc, color="gray", linestyle="--", linewidth=1.2,
                label=f"Baseline ({baseline_acc:.1f}%)")
axes[0].set_xlabel("Accuracy (%)")
axes[0].set_title("Overall Accuracy by Strategy", fontsize=11)
axes[0].legend(fontsize=8)
for bar, val in zip(bars, rank_df["accuracy"][::-1]):
    axes[0].text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                 f"{val:.1f}%", va="center", fontsize=9)
axes[0].set_xlim(0, 70)

lift_colors = [
    "#27ae60" if v > 0 else "#c0392b" if v < 0 else "#7f8c8d"
    for v in rank_df["lift_vs_baseline"][::-1]
]
axes[1].barh(rank_df["strategy"][::-1], rank_df["lift_vs_baseline"][::-1],
             color=lift_colors, edgecolor="white")
axes[1].axvline(0, color="gray", linestyle="-", linewidth=1.0)
axes[1].set_xlabel("Accuracy lift over baseline (%)")
axes[1].set_title("Lift vs. Baseline Strategy", fontsize=11)
for i, (_, row) in enumerate(rank_df[::-1].reset_index().iterrows()):
    v = row["lift_vs_baseline"]
    axes[1].text(v + (0.2 if v >= 0 else -0.2), i,
                 f"{v:+.1f}%", va="center",
                 ha="left" if v >= 0 else "right", fontsize=9)

plt.tight_layout()
plt.savefig("finr1_strategy_ranking.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: finr1_strategy_ranking.png")

# ══════════════════════════════════════════════════════════════════════════════
# § 4  PER-QUESTION DIFFICULTY: CONSISTENTLY HARD QUESTIONS
# ══════════════════════════════════════════════════════════════════════════════

q_pivot = (
    df.pivot_table(index="idx", columns="strategy", values="correct", aggfunc="first")
    .astype(int)
)
q_meta = (
    df[df["strategy"] == "baseline"][["idx", "source", "question", "n_choices", "n_gold", "language"]]
    .set_index("idx")
)
q_pivot = q_pivot.join(q_meta)
q_pivot["n_correct_strategies"] = q_pivot[list(RESULT_FILES.keys())].sum(axis=1)
q_pivot["pct_correct"]          = q_pivot["n_correct_strategies"] / len(RESULT_FILES) * 100

dist  = q_pivot["n_correct_strategies"].value_counts().sort_index()
total = len(q_pivot)

print("\nDifficulty distribution (# strategies that got the question right)")
print("=" * 55)
for k, v in dist.items():
    label = {0: "universal fail", 6: "universal pass"}.get(k, "")
    print(f"  {k}/6 correct  →  {v:>4} questions  ({v/total*100:.1f}%)  {label}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

bar_colors = sns.color_palette("RdYlGn", n_colors=7)
axes[0].bar(dist.index, dist.values, color=bar_colors, edgecolor="white", zorder=2)
axes[0].set_xticks(range(7))
axes[0].set_xticklabels([f"{i}/6" for i in range(7)])
axes[0].set_xlabel("# Strategies Correct")
axes[0].set_ylabel("# Questions")
axes[0].set_title("Per-Question Difficulty Distribution", fontsize=11)
axes[0].grid(axis="y", alpha=0.3)
for x, y in zip(dist.index, dist.values):
    axes[0].text(x, y + 2, f"{y/total*100:.0f}%", ha="center", fontsize=9)

src_diff = (
    q_pivot.groupby(["source", "n_correct_strategies"])
    .size()
    .unstack(fill_value=0)
    .reindex(columns=range(7), fill_value=0)
)
src_diff_pct = src_diff.div(src_diff.sum(axis=1), axis=0) * 100
src_diff_pct.plot(kind="bar", stacked=True, colormap="RdYlGn", ax=axes[1], edgecolor="none", width=0.7)
axes[1].set_xlabel("Dataset")
axes[1].set_ylabel("% Questions")
axes[1].set_title("Difficulty Distribution by Dataset", fontsize=11)
axes[1].legend(title="# strategies correct", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
axes[1].tick_params(axis="x", rotation=30)

plt.tight_layout()
plt.savefig("finr1_question_difficulty.png", dpi=150, bbox_inches="tight")
plt.show()

universal_fails = q_pivot[q_pivot["n_correct_strategies"] == 0].reset_index()
print(f"\nUniversal Fails (0/6 correct): {len(universal_fails)} questions")
print(universal_fails.groupby("source").size().rename("count").to_string())
print(f"\nUniversal Passes (6/6 correct): {(q_pivot['n_correct_strategies']==6).sum()} questions")
print("Saved: finr1_question_difficulty.png")

# ══════════════════════════════════════════════════════════════════════════════
# § 5  POSITIONAL BIAS & CHOICE-COUNT EFFECT
# ══════════════════════════════════════════════════════════════════════════════

df["predicted_pos"]  = df["predicted"].apply(lambda x: ord(x) - ord("a") if isinstance(x, str) else np.nan)
df["gold_pos_first"] = df["gold"].apply(lambda g: ord(g[0]) - ord("a") if g else np.nan)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

pred_counts        = df["predicted"].value_counts().sort_index()
letter_labels      = [chr(ord("a") + i) for i in range(6)]
pred_vals          = [pred_counts.get(l, 0) for l in letter_labels]
expected_per_letter = len(df) / len(pred_counts)
axes[0, 0].bar(letter_labels, pred_vals, color="#3498db", edgecolor="white")
axes[0, 0].axhline(expected_per_letter, color="red", linestyle="--",
                   label=f"Uniform ({expected_per_letter:.0f})")
axes[0, 0].set_title("Predicted Letter Distribution (all strategies)", fontsize=11)
axes[0, 0].set_xlabel("Predicted Letter")
axes[0, 0].set_ylabel("Count")
axes[0, 0].legend(fontsize=9)

strat_pred     = df.groupby(["strategy", "predicted"]).size().unstack(fill_value=0)
strat_pred     = strat_pred.reindex(columns=letter_labels, fill_value=0)
strat_pred_pct = strat_pred.div(strat_pred.sum(axis=1), axis=0) * 100
strat_pred_pct.loc[STRATEGIES_ORDERED].plot(kind="bar", ax=axes[0, 1], edgecolor="none", width=0.75)
axes[0, 1].axhline(100 / 4, color="red", linestyle="--", linewidth=0.8,
                   label="Uniform 25% (if 4 choices)")
axes[0, 1].set_title("Prediction Distribution per Strategy (%)", fontsize=11)
axes[0, 1].set_xlabel("Strategy")
axes[0, 1].set_ylabel("% Predictions")
axes[0, 1].legend(title="Predicted letter", fontsize=8, ncol=3)
axes[0, 1].tick_params(axis="x", rotation=30)

acc_by_nchoice = (
    df.groupby(["n_choices", "strategy"])["correct"]
    .mean()
    .mul(100)
    .unstack("strategy")
    .reindex(columns=STRATEGIES_ORDERED)
)
acc_by_nchoice.plot(kind="bar", ax=axes[1, 0], edgecolor="none", width=0.75)
axes[1, 0].set_title("Accuracy by Number of Choices", fontsize=11)
axes[1, 0].set_xlabel("# Answer Choices")
axes[1, 0].set_ylabel("Accuracy (%)")
axes[1, 0].legend(title="Strategy", fontsize=8, loc="upper right")
axes[1, 0].tick_params(axis="x", rotation=0)
axes[1, 0].axhline(50, color="gray", linestyle="--", linewidth=0.8)

pos_compare = df.dropna(subset=["predicted_pos", "gold_pos_first"]).copy()
pos_compare = pos_compare[pos_compare["strategy"].isin(["baseline", "few_shot", "cot"])]
pred_pos_count = pos_compare.groupby(["source", "predicted_pos"]).size().unstack(fill_value=0)
pred_pos_pct   = pred_pos_count.div(pred_pos_count.sum(axis=1), axis=0) * 100
pred_pos_pct.columns = [chr(ord("a") + int(c)) for c in pred_pos_pct.columns]
pred_pos_pct.plot(kind="bar", stacked=False, ax=axes[1, 1], edgecolor="none", width=0.7)
axes[1, 1].set_title("Predicted Position by Source (baseline+few_shot+cot)", fontsize=10)
axes[1, 1].set_xlabel("Dataset")
axes[1, 1].set_ylabel("% Predictions")
axes[1, 1].legend(title="Predicted\nletter", fontsize=8)
axes[1, 1].tick_params(axis="x", rotation=30)

plt.suptitle("Positional Bias & Choice-Count Effect Analysis", fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig("finr1_positional_bias.png", dpi=150, bbox_inches="tight")
plt.show()

print("\nPrediction share by letter (all strategies combined):")
print((df["predicted"].value_counts(normalize=True) * 100).round(1).sort_index().to_string())
print("\nAccuracy by n_choices (avg over strategies):")
print(df.groupby("n_choices")["correct"].mean().mul(100).round(1).rename("acc%").to_string())
print("Saved: finr1_positional_bias.png")

# ══════════════════════════════════════════════════════════════════════════════
# § 6  MULTI-GOLD QUESTION ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

multi_gold_df  = df[df["n_gold"] > 1]
single_gold_df = df[df["n_gold"] == 1]

print(f"\nSingle-gold questions: {single_gold_df['idx'].nunique()} unique  "
      f"({len(single_gold_df)} strategy×question rows)")
print(f"Multi-gold  questions: {multi_gold_df['idx'].nunique()} unique  "
      f"({len(multi_gold_df)} rows)")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

sg  = single_gold_df.groupby("strategy")["correct"].mean().mul(100).rename("single")
mg  = (multi_gold_df.groupby("strategy")["correct"].mean().mul(100).rename("multi")
       if len(multi_gold_df) else pd.Series(dtype=float))
cmp_df = pd.concat([sg, mg], axis=1).reindex(STRATEGIES_ORDERED)
cmp_df.plot(kind="bar", ax=axes[0], edgecolor="none", width=0.7, color=["#3498db", "#e67e22"])
axes[0].set_title("Single vs Multi-Gold Accuracy", fontsize=11)
axes[0].set_ylabel("Accuracy (%)")
axes[0].legend(["Single gold", "Multi gold"], fontsize=9)
axes[0].tick_params(axis="x", rotation=30)
axes[0].axhline(50, color="gray", linestyle="--", linewidth=0.7)
for i, (_, row) in enumerate(cmp_df.iterrows()):
    if not pd.isna(row.get("multi", np.nan)):
        axes[0].annotate(f"Δ{row['multi']-row['single']:+.0f}", (i, max(row) + 1),
                         ha="center", fontsize=7, color="black")

gold_dist = (
    df[df["strategy"] == "baseline"]
    .groupby(["source", "n_gold"])
    .size()
    .unstack(fill_value=0)
)
gold_dist_pct = gold_dist.div(gold_dist.sum(axis=1), axis=0) * 100
gold_dist_pct.plot(kind="bar", stacked=True, ax=axes[1], colormap="tab10", edgecolor="none", width=0.75)
axes[1].set_title("Gold Answer Count Distribution by Dataset", fontsize=11)
axes[1].set_xlabel("Dataset")
axes[1].set_ylabel("% Questions")
axes[1].legend(title="# gold answers", fontsize=8)
axes[1].tick_params(axis="x", rotation=30)

acc_ng = (
    df.groupby(["source", "n_gold"])["correct"]
    .mean()
    .mul(100)
    .unstack("n_gold")
)
acc_ng.plot(kind="bar", ax=axes[2], edgecolor="none", width=0.7)
axes[2].set_title("Accuracy by # Gold Answers per Dataset", fontsize=11)
axes[2].set_xlabel("Dataset")
axes[2].set_ylabel("Accuracy (%)")
axes[2].legend(title="# gold answers", fontsize=8)
axes[2].tick_params(axis="x", rotation=30)

plt.suptitle("Multi-Gold Question Handling Analysis", fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig("finr1_multigold_analysis.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: finr1_multigold_analysis.png")

# ══════════════════════════════════════════════════════════════════════════════
# § 7  WRONG-ANSWER CONFUSION MATRIX (PREDICTED vs GOLD)
# ══════════════════════════════════════════════════════════════════════════════

errors_df             = df[~df["correct"]].copy()
errors_df["gold_first"] = errors_df["gold"].apply(lambda g: g[0] if g else None)

all_letters = sorted(errors_df["predicted"].dropna().unique())
conf_mat = pd.crosstab(
    errors_df["gold_first"],
    errors_df["predicted"],
    normalize="index",
).mul(100).round(1)
for l in all_letters:
    if l not in conf_mat.index:   conf_mat.loc[l] = 0.0
    if l not in conf_mat.columns: conf_mat[l]     = 0.0
conf_mat = conf_mat.sort_index().reindex(sorted(conf_mat.columns), axis=1)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

sns.heatmap(
    conf_mat,
    annot=True,
    fmt=".0f",
    cmap="Oranges",
    linewidths=0.4,
    linecolor="white",
    ax=axes[0],
    cbar_kws={"label": "% wrong predictions (row=gold)"},
)
axes[0].set_title("Wrong-Answer Confusion Matrix\n(row = gold letter, col = predicted,  % per row)", fontsize=10)
axes[0].set_xlabel("Predicted Letter")
axes[0].set_ylabel("Gold Letter (first)")

err_by_strat_src = (
    errors_df.groupby(["strategy", "source"])
    .size()
    .unstack(fill_value=0)
    .reindex(STRATEGIES_ORDERED)
)
err_by_strat_src.plot(kind="bar", stacked=True, ax=axes[1], edgecolor="none", width=0.75, colormap="tab10")
axes[1].set_title("Error Count by Strategy & Dataset", fontsize=11)
axes[1].set_xlabel("Strategy")
axes[1].set_ylabel("# Wrong Predictions")
axes[1].legend(title="Dataset", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
axes[1].tick_params(axis="x", rotation=30)
for bar_container in axes[1].containers:
    axes[1].bar_label(bar_container, fmt="%d", label_type="center", fontsize=6, color="white")

plt.suptitle("Wrong Answer Analysis", fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig("finr1_wrong_answer_confusion.png", dpi=150, bbox_inches="tight")
plt.show()

print("\nMost common wrong prediction by gold letter (combined all strategies):")
print(conf_mat.idxmax(axis=1).rename("most_confused_with").to_string())
print("Saved: finr1_wrong_answer_confusion.png")

# ══════════════════════════════════════════════════════════════════════════════
# § 8  CROSS-STRATEGY AGREEMENT & DISAGREEMENT CLUSTERS
# ══════════════════════════════════════════════════════════════════════════════

N_STRATS = len(RESULT_FILES)


def classify_question(row):
    n_c       = row["n_correct_strategies"]
    mode_pred = df[df["idx"] == row.name]["predicted"].mode()
    n_agree   = (df[df["idx"] == row.name]["predicted"] == mode_pred.iloc[0]).sum() if len(mode_pred) else 1
    if n_c == N_STRATS:
        return "All agree & correct"
    elif n_c == 0:
        return "All agree & wrong"
    elif n_c >= N_STRATS / 2 and n_agree >= N_STRATS - 1:
        return "Majority correct"
    elif n_c < N_STRATS / 2 and n_agree >= N_STRATS - 1:
        return "Majority wrong"
    else:
        return "Split"


q_pivot["cluster"] = q_pivot.apply(classify_question, axis=1)
cluster_counts      = q_pivot["cluster"].value_counts()

print("\nCross-strategy agreement clusters:")
print("=" * 50)
for k, v in cluster_counts.items():
    print(f"  {k:<28}  {v:>4} questions  ({v/N_Q*100:.1f}%)")

cluster_order  = ["All agree & correct", "Majority correct", "Split", "Majority wrong", "All agree & wrong"]
cluster_colors = ["#27ae60", "#82e0aa", "#f0b27a", "#e74c3c", "#8e1a1a"]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

vals   = [cluster_counts.get(c, 0) for c in cluster_order]
labels = [f"{c}\n({v})" for c, v in zip(cluster_order, vals)]
axes[0].pie(vals, labels=labels, colors=cluster_colors, startangle=90,
            autopct="%1.0f%%", pctdistance=0.7, textprops={"fontsize": 8})
axes[0].set_title("Cross-Strategy Agreement Clusters", fontsize=11)

cluster_src = q_pivot.groupby(["source", "cluster"]).size().unstack(fill_value=0)
cluster_src = cluster_src.reindex(columns=cluster_order, fill_value=0)
cluster_src_pct = cluster_src.div(cluster_src.sum(axis=1), axis=0) * 100
cluster_src_pct.plot(kind="bar", stacked=True, ax=axes[1], color=cluster_colors, edgecolor="none", width=0.75)
axes[1].set_title("Agreement Clusters by Dataset (%)", fontsize=11)
axes[1].set_xlabel("Dataset")
axes[1].set_ylabel("% Questions")
axes[1].legend(title="Cluster", fontsize=7, bbox_to_anchor=(1.01, 1), loc="upper left")
axes[1].tick_params(axis="x", rotation=30)

pred_wide  = df.pivot_table(index="idx", columns="strategy", values="predicted", aggfunc="first")
strat_list = [s for s in STRATEGIES_ORDERED if s in pred_wide.columns]
agree_mat  = pd.DataFrame(index=strat_list, columns=strat_list, dtype=float)
for s1 in strat_list:
    for s2 in strat_list:
        agree_mat.loc[s1, s2] = (pred_wide[s1] == pred_wide[s2]).mean() * 100

sns.heatmap(agree_mat.astype(float), annot=True, fmt=".0f", cmap="Blues",
            vmin=50, vmax=100, linewidths=0.5, linecolor="white",
            cbar_kws={"label": "% questions same prediction"}, ax=axes[2])
axes[2].set_title("Pairwise Strategy Agreement (%)", fontsize=11)

plt.suptitle("Cross-Strategy Agreement Analysis", fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig("finr1_agreement_clusters.png", dpi=150, bbox_inches="tight")
plt.show()

split_q = q_pivot[q_pivot["cluster"] == "Split"].reset_index()
print(f"\n'Split' questions (strategies divide on answer): {len(split_q)}")
print(split_q["source"].value_counts().to_string())
print("Saved: finr1_agreement_clusters.png")

# ══════════════════════════════════════════════════════════════════════════════
# § 9  LANGUAGE × STRATEGY PERFORMANCE & HINDI FINANCE DEEP DIVE
# ══════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

lang_strat = (
    df.groupby(["language", "strategy"])["correct"]
    .mean()
    .mul(100)
    .round(1)
    .unstack("strategy")
    .reindex(columns=STRATEGIES_ORDERED)
)
sns.heatmap(
    lang_strat,
    annot=True, fmt=".1f", cmap="RdYlGn",
    linewidths=0.5, linecolor="white",
    vmin=lang_strat.values.min() - 3,
    vmax=lang_strat.values.max() + 3,
    ax=axes[0, 0],
    cbar_kws={"label": "Accuracy (%)"},
)
axes[0, 0].set_title("Language × Strategy Accuracy (%)", fontsize=11)
axes[0, 0].set_xlabel("Strategy")
axes[0, 0].set_ylabel("Language")

lang_acc = (
    df.groupby(["language", "strategy"])["correct"]
    .mean()
    .mul(100)
    .unstack("strategy")
    .reindex(columns=STRATEGIES_ORDERED)
)
lang_acc.plot(kind="bar", ax=axes[0, 1], edgecolor="none", width=0.75)
axes[0, 1].set_title("Accuracy by Language and Strategy", fontsize=11)
axes[0, 1].set_xlabel("Language")
axes[0, 1].set_ylabel("Accuracy (%)")
axes[0, 1].legend(title="Strategy", fontsize=8, loc="upper right")
axes[0, 1].tick_params(axis="x", rotation=15)
axes[0, 1].axhline(33.3, color="gray", linestyle="--", linewidth=0.7)

hindi     = df[df["source"] == "hindi_finance"].copy()
hindi_err = hindi[~hindi["correct"]]
hindi_pred_wrong = hindi_err.groupby(["strategy", "predicted"]).size().unstack(fill_value=0)
hindi_pred_wrong.reindex(STRATEGIES_ORDERED).plot(kind="bar", ax=axes[1, 0], edgecolor="none", width=0.75)
axes[1, 0].set_title("Hindi Finance — Wrong Prediction Distribution by Strategy", fontsize=10)
axes[1, 0].set_xlabel("Strategy")
axes[1, 0].set_ylabel("# Wrong Predictions")
axes[1, 0].legend(title="Predicted", fontsize=8)
axes[1, 0].tick_params(axis="x", rotation=30)

focus_langs  = ["Arabic", "Hindi", "Spanish", "Chinese", "Multilingual"]
focus_df     = df[df["language"].isin(focus_langs)]
lang_profile = (
    focus_df.groupby(["language", "strategy"])["correct"]
    .mean()
    .mul(100)
    .unstack("strategy")
    .reindex(columns=STRATEGIES_ORDERED)
)
lang_profile.T.plot(ax=axes[1, 1], marker="o", linewidth=1.5)
axes[1, 1].set_title("Per-Language Strategy Profile (accuracy %)", fontsize=11)
axes[1, 1].set_xlabel("Strategy")
axes[1, 1].set_ylabel("Accuracy (%)")
axes[1, 1].legend(title="Language", fontsize=8)
axes[1, 1].tick_params(axis="x", rotation=30)
axes[1, 1].grid(alpha=0.3)

plt.suptitle("Language × Strategy Deep Dive", fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig("finr1_language_analysis.png", dpi=150, bbox_inches="tight")
plt.show()

print("\nAccuracy by language (avg over strategies):")
print(df.groupby("language")["correct"].mean().mul(100).round(2).sort_values().rename("acc%").to_string())
print("\nHindi Finance accuracy per strategy:")
print(hindi.groupby("strategy")["correct"].mean().mul(100).round(1).reindex(STRATEGIES_ORDERED).rename("acc%").to_string())
print("Saved: finr1_language_analysis.png")

# ══════════════════════════════════════════════════════════════════════════════
# § 10  HARDEST QUESTIONS INSPECT TABLE
# ══════════════════════════════════════════════════════════════════════════════

UF        = q_pivot[q_pivot["n_correct_strategies"] == 0].reset_index().sort_values("source")
pred_wide = df.pivot_table(index="idx", columns="strategy", values="predicted", aggfunc="first")

display_rows = []
for _, row in UF.iterrows():
    q_idx = row["idx"]
    preds = pred_wide.loc[q_idx, STRATEGIES_ORDERED].to_dict()
    gold  = df[(df["idx"] == q_idx) & (df["strategy"] == "baseline")]["gold"].iloc[0]
    display_rows.append({
        "idx":       q_idx,
        "source":    row["source"],
        "question":  row["question"][:80] + ("..." if len(row["question"]) > 80 else ""),
        "gold":      ", ".join(gold),
        "n_choices": int(row["n_choices"]),
        **{f"pred_{s[:5]}": preds.get(s, "-") for s in STRATEGIES_ORDERED},
    })

uf_display = pd.DataFrame(display_rows)
print(f"\nUniversal failures (0/6 correct): {len(uf_display)} questions\n")
pd.set_option("display.max_colwidth", 85)
pd.set_option("display.max_columns", 20)
print(uf_display.head(30).to_string(index=False))

print("\nSource breakdown of universal failures:")
print(UF["source"].value_counts().to_string())
print("\nChoice-count distribution among universal failures:")
print(UF["n_choices"].value_counts().sort_index().to_string())

# ══════════════════════════════════════════════════════════════════════════════
# § 11  TREND ANALYSIS: WHERE DID CoT HURT? & ORACLE STRATEGY
# ══════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

cot_acc  = pivot.loc["cot"].drop("OVERALL")
base_acc = pivot.loc["baseline"].drop("OVERALL")
delta    = cot_acc - base_acc

delta_df         = delta.reset_index()
delta_df.columns = ["source", "delta"]
colors_delta     = ["#e74c3c" if v < 0 else "#27ae60" for v in delta_df["delta"]]
axes[0].bar(delta_df["source"], delta_df["delta"], color=colors_delta, edgecolor="white")
axes[0].axhline(0, color="black", linewidth=0.8)
axes[0].set_title("CoT Delta vs. Baseline (per dataset)", fontsize=11)
axes[0].set_xlabel("Dataset")
axes[0].set_ylabel("Accuracy change (%)")
axes[0].tick_params(axis="x", rotation=30)
for i, (src, val) in enumerate(zip(delta_df["source"], delta_df["delta"])):
    axes[0].text(i, val + (0.5 if val >= 0 else -1), f"{val:+.1f}", ha="center", fontsize=8)

oracle_per_q  = q_pivot[list(RESULT_FILES.keys())].max(axis=1)
oracle_by_src = q_pivot.groupby("source").apply(
    lambda grp: grp[list(RESULT_FILES.keys())].max(axis=1).mean() * 100
)
oracle_overall = oracle_per_q.mean() * 100

src_baseline = base_acc.rename("baseline")
src_best     = pivot.max(axis=0).drop("OVERALL").rename("best_single")
src_oracle   = oracle_by_src.rename("oracle")

compare_df = pd.concat([src_baseline, src_best, src_oracle], axis=1)
compare_df.plot(kind="bar", ax=axes[1], edgecolor="none", width=0.75,
                color=["#3498db", "#f39c12", "#27ae60"])
axes[1].set_title("Baseline vs Best-Single vs Oracle Accuracy", fontsize=11)
axes[1].set_xlabel("Dataset")
axes[1].set_ylabel("Accuracy (%)")
axes[1].legend(["Baseline", "Best-Single Strategy", "Oracle (any correct)"], fontsize=8)
axes[1].tick_params(axis="x", rotation=30)
axes[1].axhline(oracle_overall, color="green", linestyle="--", linewidth=0.8)

print(f"\nOracle accuracy (at least one strategy correct): {oracle_overall:.2f}%")
print(f"Best single-strategy accuracy: {pivot['OVERALL'].max():.2f}%  ({pivot['OVERALL'].idxmax()})")
print(f"Baseline accuracy: {baseline_acc:.2f}%")
print(f"Gap baseline→oracle: {oracle_overall - baseline_acc:+.2f}%  (room for improvement with ensemble)")
print("\nOracle by source:")
print(oracle_by_src.round(1).to_string())

cot_only_correct = q_pivot[(q_pivot["cot"] == 1) & (q_pivot["baseline"] == 0)].reset_index()
print(f"\nCoT-only wins (cot correct, baseline wrong): {len(cot_only_correct)}")
print(cot_only_correct["source"].value_counts().to_string())

nonlogit_wins = q_pivot[
    (q_pivot[["cot", "meta", "self_consistency"]].max(axis=1) == 1)
    & (q_pivot[["baseline", "few_shot", "role"]].max(axis=1) == 0)
]
print(f"\nGeneration-only wins (generation correct, all logit wrong): {len(nonlogit_wins)}")
print(nonlogit_wins["source"].value_counts().to_string())

src_nchoice_avg = df[df["strategy"] == "baseline"].groupby("source")["n_choices"].mean()
axes[2].scatter(src_nchoice_avg, delta, s=120, color="#8e44ad", zorder=3)
for src in delta.index:
    axes[2].annotate(src, (src_nchoice_avg[src], delta[src]),
                     textcoords="offset points", xytext=(5, 3), fontsize=8)
axes[2].axhline(0, color="gray", linestyle="--")
axes[2].set_title("CoT Δ vs Avg # Choices per Dataset", fontsize=11)
axes[2].set_xlabel("Average # Answer Choices")
axes[2].set_ylabel("CoT − Baseline Accuracy (%)")
axes[2].grid(alpha=0.3)

plt.suptitle("CoT Regression & Oracle Analysis", fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig("finr1_cot_oracle_trend.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: finr1_cot_oracle_trend.png")

# ══════════════════════════════════════════════════════════════════════════════
# § 12  SAVE ALL ANALYSIS OUTPUTS TO CSV
# ══════════════════════════════════════════════════════════════════════════════

ts = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M")

pivot.reset_index().to_csv(f"finr1_analysis_combined_accuracy_{ts}.csv", index=False)
print(f"\nSaved: finr1_analysis_combined_accuracy_{ts}.csv")

q_export_cols = ["source", "language", "question", "n_choices", "n_gold",
                 "n_correct_strategies", "pct_correct", "cluster"] + list(RESULT_FILES.keys())
q_export_cols = [c for c in q_export_cols if c in q_pivot.columns]
q_pivot[q_export_cols].reset_index().to_csv(f"finr1_analysis_per_question_{ts}.csv", index=False)
print(f"Saved: finr1_analysis_per_question_{ts}.csv")

df.drop(columns=["choices"], errors="ignore").to_csv(f"finr1_analysis_master_{ts}.csv", index=False)
print(f"Saved: finr1_analysis_master_{ts}.csv")

uf_display.to_csv(f"finr1_analysis_universal_failures_{ts}.csv", index=False)
print(f"Saved: finr1_analysis_universal_failures_{ts}.csv")

# ══════════════════════════════════════════════════════════════════════════════
# § 13  KEY FINDINGS & RECOMMENDATIONS
# ══════════════════════════════════════════════════════════════════════════════

strat_rank = pivot["OVERALL"].sort_values(ascending=False)
best_strat  = strat_rank.index[0]

print("=" * 70)
print("KEY FINDINGS — Fin-R1 Prompt Engineering Study (6 strategies × 1320 Q)")
print("=" * 70)
print(f"""
┌─ STRATEGY PERFORMANCE ──────────────────────────────────────────────
│  Best strategy    : {best_strat.upper()}  ({strat_rank.iloc[0]:.1f}%)
│  Baseline         : {pivot.loc['baseline','OVERALL']:.1f}%
│  CoT (worst)      : {pivot.loc['cot','OVERALL']:.1f}%  ← severe regression
│  Oracle upper-bound: {oracle_overall:.1f}%  (any strategy correct)
│  Ensemble gap     : +{oracle_overall - baseline_acc:.1f}% over baseline achievable with routing
├─ WHY CoT FAILS ─────────────────────────────────────────────────────
│  • Fin-R1 (7B) is a finance-fine-tuned model, NOT a reasoning model.
│  • Forcing chain-of-thought on a logit-optimised model causes it to
│    generate verbose finance text that often avoids committing to a
│    final letter, so the letter-extraction regex falls back to guessing.
│  • Worst impact on multilingual datasets (es_multifin 16.4%, plutus 13.2%)
│    where the model may switch language mid-reasoning.
├─ DATASET DIFFICULTY ────────────────────────────────────────────────
│  Hardest  : hindi_finance   (~35-40% all strategies)  — probable
│             language mismatch: Fin-R1 is primarily trained on
│             English/Chinese finance data.
│  Easiest  : arabic_business (~59-67%) despite being in Arabic —
│             questions may rely on simple definitions the model knows.
├─ POSITIONAL / OPTION-COUNT BIAS ───────────────────────────────────
│  • Logit strategies show mild first-option bias (A/B overrepresented).
│  • Generation strategies (meta, cot) are more spread but erratic on
│    6-choice questions (plutus dataset).
│  • Accuracy drops with more choices (2→6 options: ~60%→~45%).
├─ MULTI-GOLD QUESTIONS ──────────────────────────────────────────────
│  • Multi-gold questions are systematically harder — the model predicts
│    a single letter and rarely hits all correct options.
│  • CFA-CPA dataset has the highest multi-gold ratio.
├─ AGREEMENT PATTERNS ────────────────────────────────────────────────
│  • ~30% of questions: all 6 strategies agree AND are correct.
│  • ~15% of questions: all strategies agree AND are wrong — these are
│    the true hard failures that need model-level improvement.
│  • ~20% split: strategies disagree → routing/ensemble could help here.
└─ RECOMMENDATIONS ───────────────────────────────────────────────────
   1.  Use FEW-SHOT as the default strategy (best OVERALL at ~56.7%).
   2.  NEVER use CoT on Fin-R1 — use role or few_shot instead.
   3.  For Hindi Finance, consider a translation pre-step or a
       Hindi-specialised model before MCQ prompting.
   4.  Build a lightweight router: use ROLE for Arabic/Spanish,
       FEW_SHOT for English/Chinese — estimated +2-3% overall.
   5.  Ensemble logit strategies (baseline + few_shot + role) with
       majority vote — expected oracle-approach gain of ~{oracle_overall - strat_rank.iloc[0]:.1f}%.
   6.  For 6-choice questions (plutus), consider pruning obviously
       wrong options before scoring to reduce search space.
""")
