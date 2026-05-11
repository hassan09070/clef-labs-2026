# CLEF 2025 — Task 1: Multilingual Financial MCQ

**Assignment Submission — Task 1 (Financial Multiple-Choice Question Answering)**

---

## Overview

This folder contains all code for Task 1 of the CLEF 2025 Financial NLP challenge.
The task is to answer multilingual financial multiple-choice questions (MCQ) across
6 different datasets covering English, Spanish, Arabic, and Hindi.

---

## Datasets Used

| Dataset | Language | Description |
|---|---|---|
| `cfa_cpa` | English | CFA / CPA exam questions (Tomas08119993/finmmeval-cfa-cpa) |
| `es_multifin` | Spanish | Spanish financial MCQ (TheFinAI/flare-es-multifin) |
| `plutus` | Multilingual | Plutus MultiFinQA (TheFinAI/plutus-multifin) |
| `arabic_accounting` | Arabic | Arabic accounting MCQ (SahmBenchmark/arabic-accounting-mcq) |
| `arabic_business` | Arabic | Arabic business MCQ (SahmBenchmark/arabic-business-mcq) |
| `hindi_finance` | Hindi/English | BhashaBench Finance Hindi (bharatgenai/BhashaBench-Finance-Hindi) |

---

## File Descriptions

### `clef_task1_data_clean.py`
**Purpose: Data downloading and cleaning**

Downloads the raw CFA/CPA dataset from HuggingFace (4 Parquet files, English + Chinese splits).
Handles a schema mismatch where `gold` is stored as `list<int64>` instead of `int64`.
Cleans and validates each row (checks for missing fields, invalid gold indices, duplicates).
Outputs a single clean JSON file: `task1_Tomas08119993_finmmeval-cfa-cpa.json`.

---

### `clef_task1_data_analysis.py`
**Purpose: Dataset exploration and analysis**

Fetches all 6 dataset JSON files from the private GitHub repo (`hassan09070/clef_task`).
Runs comprehensive analysis on each dataset individually and combined:
- Question length distributions
- Answer choice count statistics
- Gold label distribution (positional bias detection)
- Language breakdowns
- Multi-label answer frequency

Saves all plots to `plots/` and structured results to `analysis_results/`.
**Run:** `python clef_task1_data_analysis.py`
**Requires:** `GITHUB_TOKEN` environment variable set.

---

### `clef-task1-baseline.ipynb`
**Purpose: Zero-shot baseline evaluation (Kaggle)**

Runs zero-shot inference on all 6 datasets using 3 models:
- `microsoft/Phi-3.5-mini-instruct` (~3.8B)
- `Qwen/Qwen2.5-7B-Instruct` (~7B)
- `meta-llama/Llama-2-13b-chat-hf` (~13B)

Measures accuracy (correct answer letter proportion) with no prompting tricks.
Uses Kaggle Secrets for the HuggingFace token — no hardcoded credentials.
Results establish the performance baseline for comparison with later experiments.

---

### `clef_task1_qwen_prompt_engineering.ipynb`
**Purpose: Prompt engineering study — Qwen2.5-14B (local GPU)**

Evaluates 6 prompt strategies on `Qwen/Qwen2.5-14B-Instruct` (4-bit NF4 quantisation)
across all 6 datasets (data fetched from GitHub):

| Strategy | Description |
|---|---|
| `baseline` | Zero-shot, answer letter only |
| `few_shot` | 3 in-context examples |
| `cot` | Chain-of-thought ("think step by step") |
| `role` | Expert CFA analyst persona |
| `self_consistency` | 5-sample majority vote |
| `meta` | Explicit elimination-based reasoning strategy |

Results are saved per-strategy as JSON files.
**Requires:** `GITHUB_TOKEN` environment variable set.

---

### `clef_task1_finr1_prompt_engineering.ipynb`
**Purpose: Prompt engineering study — Fin-R1 7B (local GPU)**

Same 6-strategy evaluation but using `SUFE-AIFLM-Lab/Fin-R1` (Qwen2.5-7B + finance SFT + GRPO).
Fin-R1 is a finance-specialised model that uses `<think>` / `<answer>` structured output.
Includes logit-based inference for baseline/few-shot/role and generation-based for CoT/meta/self-consistency.

Key finding: Fin-R1 achieved 56.7% overall (best: few-shot) vs Qwen2.5-14B baseline.

**Requires:** `GITHUB_TOKEN` environment variable set.

---

### `clef_task1_finr1_analysis.py`
**Purpose: Deep analysis of Fin-R1 prompt engineering results**

Loads all 6 strategy result JSONs from the Fin-R1 experiments and produces:
- Combined accuracy heatmap (Strategy × Dataset)
- Strategy ranking and lift over baseline
- Per-question difficulty distribution
- Positional bias and choice-count effect analysis
- Multi-gold question analysis
- Wrong-answer confusion matrix
- Cross-strategy agreement clusters
- Language × strategy performance breakdown
- Hardest question inspection table
- CoT regression and oracle analysis

Outputs PNG charts and timestamped CSVs.
**Run:** `python clef_task1_finr1_analysis.py`

---

### `clef_task1_groq_prompt_engineering.ipynb`
**Purpose: Prompt engineering study — Qwen3-32B via Groq API**

Runs the same 6 strategies on `qwen/qwen3-32b` using the Groq cloud API with
9 API keys pooled together (each key has RPM=60, RPD=1000, TPM=6000 limits).
Uses per-key rate-limit tracking to maximise throughput without hitting quota.

Total evaluation: 2500 questions across all datasets (hindi_finance capped to reach exactly 2500).
**Requires:** `GITHUB_TOKEN` environment variable set. Groq API keys read from environment variables.

---

### `clef_task1_gemini25_prompt_engineering.py`
**Purpose: Prompt engineering study — Gemini 2.5 Flash via OpenRouter**

Evaluates 6 strategies on `google/gemini-2.5-flash` accessed through the OpenRouter API.
Uses a thread-pool (`MAX_WORKERS`) for parallel inference — each strategy's results are
saved to JSON immediately after completion (crash-safe).

**Requires:** `OPENROUTER_API_KEY` environment variable set.
**Run:** `python clef_task1_gemini25_prompt_engineering.py`

---

### `clef_task1_gemini25_analysis.py`
**Purpose: Deep analysis of Gemini 2.5 results**

Loads the 4 complete Gemini 2.5 strategy result JSONs (1320 questions each) and produces:
- Corrected strict_cot scores (original extractor missed LaTeX `\boxed{X}` notation)
- Extraction audit (old vs corrected scores)
- Combined accuracy heatmap
- Strategy ranking and lift over few-shot
- Gemini 2.5 Flash vs Fin-R1 head-to-head comparison

Key finding: Gemini 2.5 Flash achieved **70.68%** (few-shot), far above Fin-R1's 56.7%.

**Run:** `python clef_task1_gemini25_analysis.py`

---

### `finr1_finetune_colab_final (3) (1).ipynb`
**Purpose: QLoRA fine-tuning of Fin-R1 on multilingual financial MCQ**

The main fine-tuning pipeline. Fine-tunes `SUFE-AIFLM-Lab/Fin-R1` (Qwen2.5-7B + finance GRPO)
using QLoRA (LoRA adapters on a 4-bit base) on 1800 balanced training examples (300 per source).

**Key design decisions:**
- **Choice shuffling** — randomly reorders answer options each epoch to remove positional bias (model was over-predicting A)
- **Hindi translation** — hindi_finance questions translated to English via Google Translate (fixes 34.7% → target improvement)
- **LoRA rank r=64** — more capacity than baseline r=8 for 6-language adaptation
- **BF16 training** — cleaner gradients on A100
- **3 epochs, effective batch 32** (batch 2 × gradient accumulation 16)
- **Arabic system prompt** — Arabic questions use an Arabic-language system prompt

**Cells overview:**

| Cell | What it does |
|---|---|
| Cell 1 (commented out) | Environment setup and GPU cleanup — run once then restart kernel |
| Cell 2 | Configuration — `MODEL_ID`, `HUB_REPO`, `TEST_MODE`, LoRA and training hyperparameters |
| Cell 3 | Install required libraries |
| Cell 4 | HuggingFace login + GitHub token setup (reads from environment variables) |
| Cell 5 | Load all 6 datasets from private GitHub repo via the Contents API |
| Cell 6 | Balance / cap datasets — smallest-source capping for balanced training |
| Cell 7 | Translate hindi_finance questions Hindi → English (Google Translate) |
| Cell 8 | SFT formatter — converts MCQ rows into `<think>/<answer>` training examples |
| Cell 9 | Train / val / test split (75% / 12% / 13% per source) |
| Cell 10 | Gold label distribution check — verifies shuffling removed positional bias |
| Cell 11 | Load Fin-R1 model in BF16 (no quantisation on A100) + build HuggingFace datasets |
| Cell 12 | Training loop via `SFTTrainer` |
| Cell 13 | Save LoRA adapter to local workspace |
| Cell 14 | Evaluation on held-out test set — accuracy per source + overall |
| Cell 15 | Push merged model to HuggingFace Hub, reload, and run a verification prediction |

**Runtime:** ~25–35 minutes on an A100 40GB (Vast.ai)
**Output:** Merged model pushed to `hassanshahzad2003/finr1-mcq-multilingual` on HuggingFace Hub

---

## Environment Variables Required

No API keys or tokens are hardcoded anywhere. All credentials must be set as environment variables:

| Variable | Used in | Purpose |
|---|---|---|
| `HF_TOKEN` | `clef_task1_data_clean.py`, fine-tuning notebook | HuggingFace Hub login |
| `GITHUB_TOKEN` | All prompt engineering notebooks/scripts | Fetch dataset JSON files from private repo |
| `OPENROUTER_API_KEY` | `clef_task1_gemini25_prompt_engineering.py` | Gemini 2.5 Flash via OpenRouter |
| `GROQ_API_KEY_*` | `clef_task1_groq_prompt_engineering.ipynb` | Groq API keys (pooled) |

Set them before running:
```bash
export HF_TOKEN="your_token_here"
export GITHUB_TOKEN="your_token_here"
export OPENROUTER_API_KEY="your_token_here"
```


