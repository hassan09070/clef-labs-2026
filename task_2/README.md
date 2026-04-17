# PolyFiQA: Structure-Aware Hybrid RAG Pipeline

This repository contains the evaluation code and artifacts for task 2 of finMMEval

## 1. System Requirements
- Python 3.9+ 
- Minimum 8GB RAM (Standard CPU is sufficient; the `multilingual-e5-small` embedding model runs locally, and text generation is handled via external GROQ API).

## 2. Environment Setup (Conda)
This project uses a Conda environment to ensure consistent package versions and reproducibility across Windows machines. 

**Prerequisite:** Ensure you have [Anaconda](https://www.anaconda.com/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed.

**i. Clone the environment:**
Navigate to the root directory of this project in your terminal and run:
```bash
conda env create -f environment.yml
conda activate polyfiqa_eval
```

**ii. API Key Configuration**
Althought the generated json result files are present in the results/ directory, if you still want to execute the generation scripts, you must provide a valid GROQ API key.

The API keys are defined as string variables at the top of the evaluation scripts. Before running the code, you must replace the variable GROQ_API_KEY in the files in the src/ directory with an active GROQ API key. 

```python 
# Change the placeholder string:
GROQ_API_KEY = "your_api_key_here"

# To your actual active key:
GROQ_API_KEY = "gsk_123456789ABCDEF..."
```

## 3.Reviewing the Results

Once the execution and scoring scripts have completed, you can review the results in two ways:

**i. Evaluation Metrics (Console Output):**
The `results/score_results.py` script computes the official metrics and will print the final ROUGE-1, ROUGE-2, and ROUGE-L scores directly to your terminal. you will need to replace the json result file being evaluated in the python file at the top.

To run the `score_results.py`:
```bash
cd results
# replace the file that need to be evaluated at the top of score_results.py
python score_results.py
```

**ii. Generated Predictions (JSON Artifacts):**
The raw output data, including the retrieved chunks.
