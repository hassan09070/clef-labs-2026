# PolyFiQA: Structure-Aware Hybrid RAG Pipeline

This repository contains the evaluation code and artifacts for Task 2 (PolyFiQA) of the CLEF-2026 FinMMEval Lab. Our architecture implements a Partitioned Two-Stage Retrieval pipeline (combining `multilingual-e5-small` and `bge-reranker-v2-m3`) paired with an unconstrained Llama-4-Scout-17B Mixture-of-Experts (MoE) generation layer.

## 1. System Requirements
- Python 3.9+ 
- **Minimum 16GB RAM Recommended:** While standard CPUs are sufficient, the dual-retrieval engine (specifically the `bge-reranker-v2-m3` Cross-Encoder) requires several gigabytes of memory to load locally.
- Text generation is handled via external API (e.g., GROQ), so an active internet connection is required.

## 2. Environment Setup (Conda)
This project uses a Conda environment to ensure consistent package versions and reproducibility. An `environment.yml` file is provided in the root directory.

**i. Clone the environment:**
Navigate to the root directory of this project in your terminal and run:
```bash
conda env create -f environment.yml
conda activate clef

## 3. Running the Generation Pipeline (Optional)
If you wish to re-run the full extraction pipeline from scratch instead of using the provided pre-computed JSON files, you can execute the Easy and Expert scripts directly. 

You must pass your Groq API key directly via the command line using the `-api_key` argument. You can also specify the output file name using the `-output` flag.

**To run the PolyFiQA-Easy dataset:**
```bash
python run_polyfiqa_cross_easy.py -api_key "gsk_YOUR_API_KEY" -output "reranked_easy.json"
```

**To run the PolyFiQA-Expert dataset:**
```bash
python run_polyfiqa_cross_expert.py -api_key "gsk_YOUR_API_KEY" -output "reranked_expert.json"
```

## Evaluation example
python evaluate.py full_easy_results.json