# CLEF Labs 2026 — FinMMEval

**Team HU_LLM_Fin — Habib University, Karachi, Pakistan**

Submissions, code, and data for the [CLEF 2026 FinMMEval Lab](https://clef2026.clef-initiative.eu/) —
a shared task on **multilingual and multimodal financial evaluation** for large language models.

Our Task 2 system placed **10th globally** on the official blind test leaderboard, and the
accompanying working-notes paper is published in the CLEF 2026 proceedings.

---

## Paper

> **HU_LLM_Fin @ FinMMEval 2026 Task 2: A Structure-Aware Hybrid RAG Pipeline utilizing Mixture-of-Experts**
> Muhammad Affan\*, Muhammad Bilal Qureshi†, Muhammad Hassan Shahzad†, Faisal Alvi, Abdul Samad
> *CLEF 2026 Working Notes, 21–24 September 2026, Jena, Germany*
> \*Corresponding author  †Equal contribution

📄 **[Read the paper](paper/HU_LLM_Fin_FinMMEval2026_Task2_Structure-Aware_Hybrid_RAG.pdf)**

<details>
<summary>BibTeX</summary>

```bibtex
@inproceedings{affan2026hullmfin,
  title     = {HU\_LLM\_Fin @ FinMMEval 2026 Task 2: A Structure-Aware Hybrid
               RAG Pipeline utilizing Mixture-of-Experts},
  author    = {Affan, Muhammad and Qureshi, Muhammad Bilal and
               Shahzad, Muhammad Hassan and Alvi, Faisal and Samad, Abdul},
  booktitle = {CLEF 2026 Working Notes},
  series    = {CEUR Workshop Proceedings},
  publisher = {CEUR-WS.org},
  address   = {Jena, Germany},
  year      = {2026}
}
```
</details>

---

## The Challenge

Real financial analysis is never monolingual or text-only. Analysts read dense numerical
tables from SEC filings (10-K, 10-Q) *alongside* fast-moving news in many languages.
FinMMEval 2026 benchmarks LLMs on exactly that gap, across three tasks:

| Task | Name | What it asks | Our work |
|:--:|---|---|---|
| **1** | Multilingual Financial MCQ | Answer finance exam questions in English, Spanish, Arabic and Hindi | [`Final_code/Task1`](Final_code/Task1) |
| **2** | **PolyFiQA** | Answer questions requiring joint reasoning over structured SEC tables **and** multilingual news | [`Final_code/Task2`](Final_code/Task2) — **paper + 10th place** |
| **3** | Financial Trading | Turn news sentiment and market signals into trading decisions | [`Final_code/Task3`](Final_code/Task3) |

The core difficulty in Task 2: standard RAG chunking **destroys** financial tables. Split a
Markdown table naively and a row like `| Net income | $21,870 | $16,425 |` loses its header —
you can no longer tell which number is 2023 and which is 2022. Frontier models collapse on this;
the lab reports GPT-4o scoring just **9.79%** on PolyFiQA-easy and **5.31%** on PolyFiQA-expert.

---

## What We Did — Task 2

A **Structure-Aware Hybrid RAG pipeline** in four phases:

```
SEC filings ──► Row-Aware Chunking ──┐
                                     ├──► BM25 + multilingual-e5-small ──► Partitioned
Multilingual news ──► kept whole ────┘         (α = 0.5 fusion)            Cross-Encoder
                                                                          (bge-reranker-v2-m3)
                                                                                │
                                                          Prompt Caging ◄───────┘
                                                                │
                                                    Llama-4-Scout-17B (MoE)
                                                                │
                                                    <scratchpad> ──► <answer>
```

**1. Row-Aware Chunking.** Instead of recursive character splitting, the ingestion engine
detects a table's header and date context and *forcibly re-injects them* into every
iterative 4-row chunk — so each numerical figure keeps its temporal and categorical meaning.

```
Standard destructive chunk:      | Net income | $21,870 | $16,425 |

Our row-aware chunk:             Table Header: | (In millions) | Three Months Dec 31, 2023 | Dec 31, 2022 |
                                 | Net income | $21,870 | $16,425 |
                                 | Depreciation, amortization, and other | 5,959 | 3,648 |
```

**2. Partitioned Two-Stage Retrieval.** Stage 1 fuses a dense bi-encoder
(`multilingual-e5-small`, for cross-lingual alignment) with sparse `BM25` (for exact currency
figures) via min-max normalised linear combination:

$$S(q,c) = \alpha \cdot \widetilde{BM25}(q,c) + (1-\alpha) \cdot \widetilde{Cosine}(E(q), E(c))$$

Stage 2 reranks with a `bge-reranker-v2-m3` cross-encoder — but **partitioned**: tables and
news are reranked *independently* (top-3 table chunks, top-2 news segments) so descriptive
narrative text can't crowd out numerical grids.

**3. Prompt Caging.** A strict output contract with a Literal Quote Rule — all multi-hop
arithmetic goes inside `<scratchpad>` tags, final output inside `<answer>`/`<evidence>` —
mathematically barring the model from paraphrasing financial digits.

**4. MoE generation.** `Llama-4-Scout-17B` (Mixture-of-Experts) was chosen deliberately over
larger dense models: dense models hit Token-Per-Minute API rate limits that force artificial
context truncation. The MoE layer gave multilingual depth *without* truncating retrieved context.

### Results

**Internal ablation (PolyFiQA validation)** — rows are a sequential pipeline build-up, not
isolated single-factor ablations:

| Architecture | Easy (R-1) | Easy (R-L) | Expert (R-1) | Expert (R-L) |
|---|:--:|:--:|:--:|:--:|
| Standard RAG Baseline | 0.3484 | 0.2925 | – | – |
| + Row-Aware Chunking | 0.2974 | 0.2284 | – | – |
| + Prompt Caging (Llama-8B) | 0.3474 | 0.2834 | 0.1783 | 0.1203 |
| **Partitioned MoE (Scout-17B)** | **0.3958** | **0.3342** | **0.3437** | **0.2867** |
| *Official baseline (Llama-4-Scout-17B)* | *0.2773* | – | *0.2060* | – |
| *Official baseline (Llama-3.1-70B)* | *0.2504* | – | *0.1856* | – |

**Official blind test set — 🏆 10th globally:**

| Team | ROUGE-1 F1 | Precision | Recall |
|---|:--:|:--:|:--:|
| **HU_LLM_Fin (ours)** | **0.2289** | 0.2581 | 0.2803 |

### Error analysis

Two failure modes documented in §5 of the paper, both *downstream* of retrieval:

- **Mathematical hallucination** (`easy_0064`) — the model correctly extracted `$360M` and
  `$9.105B` and computed `3.95%`, then overrode its own arithmetic with a hallucinated `14.0%`.
- **Zero-assumption fallacy** (`easy_0096`) — with R&D expenditure missing, the model assumed
  it was `0` rather than abstaining, producing a mathematically sound but catastrophic `0.0%`.

---

## What We Did — Task 1

Multilingual financial MCQ across **six datasets** (English, Spanish, Arabic, Hindi):

| Dataset | Language | Source |
|---|---|---|
| `cfa_cpa` | English | `Tomas08119993/finmmeval-cfa-cpa` |
| `es_multifin` | Spanish | `TheFinAI/flare-es-multifin` |
| `plutus` | Multilingual | `TheFinAI/plutus-multifin` |
| `arabic_accounting` | Arabic | `SahmBenchmark/arabic-accounting-mcq` |
| `arabic_business` | Arabic | `SahmBenchmark/arabic-business-mcq` |
| `hindi_finance` | Hindi/English | `bharatgenai/BhashaBench-Finance-Hindi` |

We evaluated **6 prompt strategies** (baseline, few-shot, chain-of-thought, role, self-consistency,
meta-reasoning) across Phi-3.5-mini, Qwen2.5-7B/14B, Qwen3-32B, Fin-R1, and Gemini 2.5 Flash,
then fine-tuned with QLoRA.

**Headline numbers:** Gemini 2.5 Flash reached **70.68%** (few-shot), well clear of
Fin-R1's **56.7%**. The QLoRA fine-tune of Fin-R1 — with choice shuffling to kill positional
bias and Hindi→English translation — is published at
[`hassanshahzad2003/finr1-mcq-multilingual`](https://huggingface.co/hassanshahzad2003/finr1-mcq-multilingual).

Full per-file breakdown: [`Final_code/Task1/README.md`](Final_code/Task1/README.md)

---

## What We Did — Task 3

A financial trading agent combining **FinBERT** sentiment classification
(`ProsusAI/finbert`) with **Qwen2.5-7B-Instruct** (4-bit NF4) for decision generation.
Because FinBERT truncates at 512 tokens, news is chunked and sentiment scores aggregated
across chunks rather than computed on the first 512 tokens alone. Served via a FastAPI
endpoint (`qwen_trading_api.py`) for batch evaluation.

---

## Repository Layout

```
.
├── paper/                        # Published CLEF 2026 working-notes paper (PDF)
├── Final_code/                   # Final submitted systems
│   ├── Task1/                    # Multilingual MCQ — prompting studies + QLoRA fine-tune
│   ├── Task2/                    # PolyFiQA — Structure-Aware Hybrid RAG (the paper)
│   │   ├── run_polyfiqa_cross_easy.py
│   │   ├── run_polyfiqa_cross_expert.py
│   │   ├── environment.yml
│   │   └── results/              # Pre-computed outputs + evaluate.py
│   └── Task3/                    # Trading — FinBERT sentiment + Qwen2.5-7B
├── Initial_code/                 # Earlier iterations & ablations (kept for provenance)
└── task1_data/                   # Cleaned Task 1 datasets (JSON)
```

`Initial_code/` is retained deliberately — it holds the intermediate ablation runs
(`polyfiqa_easy_rag_results_changeA/B`, hero runs, Llama-8B ablations) that the paper's
Table 1 progression is built from.

---

## Data

| What | Where |
|---|---|
| Task 1 cleaned datasets (6 × JSON) | [`task1_data/`](task1_data) |
| Task 2 PolyFiQA final outputs | [`Final_code/Task2/results/`](Final_code/Task2/results) — 76 easy + 76 expert |
| Task 2 ablation runs | [`Initial_code/task_2/results/`](Initial_code/task_2/results) |

PolyFiQA source data is distributed by the lab organisers — see the
[FinMMEval 2026 overview](https://arxiv.org/abs/2506.14028).

---

## Reproducing

**Task 2** (the paper's pipeline):

```bash
conda env create -f Final_code/Task2/environment.yml
conda activate clef

python Final_code/Task2/run_polyfiqa_cross_easy.py   -api_key "$GROQ_API_KEY" -output reranked_easy.json
python Final_code/Task2/run_polyfiqa_cross_expert.py -api_key "$GROQ_API_KEY" -output reranked_expert.json

python Final_code/Task2/results/evaluate.py full_easy_results.json
```

Requires ≥16 GB RAM (the `bge-reranker-v2-m3` cross-encoder loads locally) and an internet
connection for the Groq-hosted generation layer.

**Determinism.** All PRNGs are locked to `seed = 42` (Python, NumPy, PyTorch),
`torch.backends.cudnn.deterministic = True`, generation temperature `0.0`, max output
512 tokens, retrieval fusion `α = 0.5`, cross-encoder max sequence length 512. API calls
are orchestrated with async semaphores and a 35-second rate-limit delay.

**Credentials.** Nothing is hardcoded. Export what each task needs:

```bash
export HF_TOKEN="..."            # HuggingFace Hub
export GITHUB_TOKEN="..."        # dataset fetch (Task 1)
export GROQ_API_KEY="..."        # Llama-4-Scout-17B generation (Task 2)
export OPENROUTER_API_KEY="..."  # Gemini 2.5 Flash (Task 1)
```

The Task 1 analysis scripts read their result directories from the environment,
falling back to a relative path:

```bash
export FINR1_RESULTS_DIR="results/finr1_prompt_engineering"    # default
export GEMINI_RESULTS_DIR="results/gemini25_prompt_engineering" # default
```

---

## Key References

1. Xie et al. — *Overview of FinMMEval 2026: Multilingual and multimodal financial evaluation*, CLEF 2026, Springer LNCS.
2. Xie et al. — *Overview of the FinMMEval 2026 Task 2: Financial question answering and summarization*, CEUR-WS.
3. Peng et al. — [*MultiFinBen: Benchmarking LLMs for multilingual and multimodal financial application*](https://arxiv.org/abs/2506.14028), 2025.
4. Wang et al. — [*Text embeddings by weakly-supervised contrastive pre-training*](https://arxiv.org/abs/2212.03533), 2022. (`multilingual-e5`)
5. Robertson & Zaragoza — *The probabilistic relevance framework: BM25 and beyond*, 2009.

---

## Acknowledgments

We thank the CLEF 2026 FinMMEval Lab organisers for designing the tasks, providing the
PolyFiQA dataset, and operating the evaluation infrastructure. We also thank
**Prof. Faisal Alvi**, **Prof. Abdul Samad**, and the teaching staff of **CS/CE 335/466
(Introduction to Large Language Models)** at Habib University for their guidance throughout
this research.

---

## License

The paper is © 2026 the authors, released under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
