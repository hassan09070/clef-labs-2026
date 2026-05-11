import json
import re
import sys
from rouge_score import rouge_scorer

# ── Numeric Accuracy ──────────────────────────────────────
def compute_numeric_accuracy(prediction: str, reference: str) -> float:
    pattern = r'(?:\$[\d,]+\.?\d*(?:[BMK])?|[\d,]+\.?\d*(?:[BMK%])?|[-+]?\d*\.\d+|\d+)'
    raw_pred = re.findall(pattern, prediction)
    raw_ref  = re.findall(pattern, reference)
    pred_nums = set(n.replace(',', '').replace('$', '') for n in raw_pred)
    ref_nums  = set(n.replace(',', '').replace('$', '') for n in raw_ref)
    pred_nums.discard('')
    ref_nums.discard('')
    if not ref_nums:
        return 1.0
    if not pred_nums:
        return 0.0
    return len(pred_nums & ref_nums) / len(ref_nums)

# ── Load Results ──────────────────────────────────────────
path = sys.argv[1] if len(sys.argv) > 1 else "polyfiqa_results.json"
with open(path, "r", encoding="utf-8") as f:
    results = json.load(f)

# ── Compute Metrics ───────────────────────────────────────
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

r1, r2, rL, num_acc = [], [], [], []
skipped = []

for r in results:
    prediction = r.get('answer', '') or ''
    reference  = r.get('gold', '')  or ''

    if not prediction.strip() or not reference.strip():
        skipped.append(r.get('task_id', '?'))
        continue

    s = scorer.score(reference, prediction)
    r1.append(s['rouge1'].fmeasure)
    r2.append(s['rouge2'].fmeasure)
    rL.append(s['rougeL'].fmeasure)
    num_acc.append(compute_numeric_accuracy(prediction, reference))

# ── Print Results ─────────────────────────────────────────
n = len(r1)
print(f"── Evaluation Metrics (skipping failures) ────────")
print(f"  ROUGE-1:          {sum(r1)/n:.4f}")
print(f"  ROUGE-2:          {sum(r2)/n:.4f}")
print(f"  ROUGE-L:          {sum(rL)/n:.4f}")
print(f"  Numeric Accuracy: {sum(num_acc)/n:.4f}")
print(f"  Scored:           {n}/{len(results)}")
print(f"  Skipped:          {len(skipped)}")