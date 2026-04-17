import json
import re
from rouge_score import rouge_scorer

# ── Load Results ──────────────────────────────────────────
with open("polyfiqa_expert_hero_run_scout.json", "r", encoding="utf-8") as f:
    data = json.load(f)

results = data["results"]

# ── Compute Metrics (skip failures, don't score as 0) ─────
scorer = rouge_scorer.RougeScorer(
    ['rouge1', 'rouge2', 'rougeL'], use_stemmer=True
)

r1, r2, rL = [], [], []
num_acc = []
skipped = []

for r in results:
    if not r['prediction'] or not r['prediction'].strip():
        skipped.append(f"{r['task_id']} Q{r['q_idx']}")
        continue

    # ROUGE Scoring
    s = scorer.score(r['reference'], r['prediction'])
    r1.append(s['rouge1'].fmeasure)
    r2.append(s['rouge2'].fmeasure)
    rL.append(s['rougeL'].fmeasure)
    

# ── Print Results ─────────────────────────────────────────
print(f"── Evaluation Metrics (skipping failures) ────────")
print(f"  ROUGE-1:          {sum(r1)/len(r1):.4f}")
print(f"  ROUGE-2:          {sum(r2)/len(r2):.4f}")
print(f"  ROUGE-L:          {sum(rL)/len(rL):.4f}")
print(f"  Scored:           {len(r1)}/{len(results)}")
print(f"  Skipped:          {len(skipped)}")

# if skipped:
#     print(f"\n── Skipped rows ──────────────────────────────────")
#     for s in skipped:
#         print(f"  - {s}")