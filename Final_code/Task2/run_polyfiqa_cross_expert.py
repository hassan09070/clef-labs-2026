"""
run_polyfiqa_cross.py
Two-Stage Hybrid RAG Pipeline (BM25 + Bi-Encoder -> Cross-Encoder Reranking)
"""

import os
import asyncio
import json
import re
import numpy as np
import random
import argparse
from pathlib import Path
from datasets import load_dataset
from groq import AsyncGroq
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi


# ============================================================
# Config
# ============================================================

MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
MAX_TOKENS  = 512
TEMPERATURE = 0.0
MAX_RETRIES = 10
TOP_N_CANDIDATES = 20  # Stage 1 retrieval pool
TOP_K_FINAL      = 5   # Stage 2 LLM context window
ALPHA            = 0.5    

print("Loading Stage 1: multilingual-e5-small bi-encoder...")
embedder = SentenceTransformer("intfloat/multilingual-e5-small")

print("Loading Stage 2: bge-reranker-v2-m3 cross-encoder...")
reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", max_length=512)
print("Models ready.\n")

# ============================================================
# 1. Bulletproof Query Parser
# ============================================================
def parse_query(query: str) -> dict:
    lines = query.split('\n')
    
    idx = {}
    for i, line in enumerate(lines):
        s = line.strip()
        if   s == 'Financial Statements:': idx['financial'] = i
        elif s == 'English News:':          idx['english']   = i
        elif s == 'Chinese News:':          idx['chinese']   = i
        elif s == 'Japanese News:':         idx['japanese']  = i
        elif s == 'Spanish News:':          idx['spanish']   = i
        elif s == 'Greek News:':            idx['greek']     = i
        elif s == 'Question:':              idx['question']  = i

    fin_start = idx.get('financial', 0)
    eng_start = idx.get('english', len(lines))
    fin_lines = lines[fin_start+1:eng_start]
    
    # Aggressively catch both '---' and '***' separators
    seps = [i for i, line in enumerate(fin_lines) if line.strip().startswith('---') or line.strip().startswith('***')]

    def get_fin(start_idx, end_idx):
        if start_idx >= len(fin_lines): return ""
        return '\n'.join(fin_lines[start_idx:end_idx]).strip()

    balance_sheet = get_fin(0, seps[0]) if len(seps) > 0 else get_fin(0, len(fin_lines))
    cash_flow     = get_fin(seps[0]+1, seps[1]) if len(seps) > 1 else ""
    income_stmt   = get_fin(seps[1]+1, seps[2]) if len(seps) > 2 else ""

    def get_news(start_key, end_key):
        if start_key not in idx or end_key not in idx: return ""
        return '\n'.join(lines[idx[start_key]+1:idx[end_key]]).strip()

    return {
        "instructions":     '\n'.join(lines[:fin_start]).strip(),
        "balance_sheet":    balance_sheet,
        "cash_flow":        cash_flow,
        "income_statement": income_stmt,
        "english_news":     get_news('english', 'chinese'),
        "chinese_news":     get_news('chinese', 'japanese'),
        "japanese_news":    get_news('japanese', 'spanish'),
        "spanish_news":     get_news('spanish', 'greek'),
        "greek_news":       get_news('greek', 'question'),
    }

# ============================================================
# 2. Row-Aware Chunking
# ============================================================
def build_structured_chunks(sections: dict) -> list[dict]:
    all_chunks = []

    for source in ["balance_sheet", "cash_flow", "income_statement"]:
        table_text = sections.get(source, "")
        if not table_text.strip(): continue
        
        lines = table_text.split('\n')
        header = ''
        data_rows = []
        for line in lines:
            s_line = line.strip()
            if not s_line or not s_line.startswith('|'): continue
            # Handle markdown separator rows without double dash limits
            if re.match(r'^\|[\s\|\-:]+\|$', s_line) and not re.search(r'[a-zA-Z0-9$]', s_line): 
                continue  
                
            if not header: header = s_line
            else: data_rows.append(s_line)
        
        for i in range(0, len(data_rows), 4):
            chunk_text = f"Table Header: {header}\n" + '\n'.join(data_rows[i:i+4])
            all_chunks.append({"source": source, "text": chunk_text})

    for source in ["english_news", "chinese_news", "japanese_news", "spanish_news", "greek_news"]:
        text = sections.get(source, "")
        if text.strip():
            all_chunks.append({"source": source, "text": text.strip()})
            
    return all_chunks

# ============================================================
# 3. Two-Stage Retrieval
# ============================================================
def retrieve_chunks(chunks: list[dict], question: str) -> tuple[str, list[str]]:
    if not chunks: return "", []

    # 1. Partition the corpus to prevent Cross-Encoder bias
    table_chunks = [c for c in chunks if c['source'] in ['balance_sheet', 'cash_flow', 'income_statement']]
    news_chunks  = [c for c in chunks if c['source'] not in ['balance_sheet', 'cash_flow', 'income_statement']]

    def get_top_k(target_chunks, k):
        if not target_chunks: return []
        texts = [c['text'] for c in target_chunks]
        
        # Stage 1: BM25 + Dense
        bm25 = BM25Okapi([t.lower().split() for t in texts])
        bm25_scores = np.array(bm25.get_scores(question.lower().split()))

        q_emb = embedder.encode([f"query: {question}"])
        c_embs = embedder.encode([f"passage: {t}" for t in texts], batch_size=32)
        dense_scores = cosine_similarity(q_emb, c_embs)[0]

        def normalize(arr):
            if arr.max() == arr.min(): return np.zeros_like(arr)
            return (arr - arr.min()) / (arr.max() - arr.min() + 1e-9)

        combined = (ALPHA * normalize(bm25_scores)) + ((1 - ALPHA) * normalize(dense_scores))
        
        # Pull top 15 from this specific modality for reranking
        top_n_idxs = sorted(np.argsort(combined)[-15:].tolist(), reverse=True)
        stage_1_cands = [target_chunks[i] for i in top_n_idxs]

        # Stage 2: Cross-Encoder
        cross_inp = [[question, c['text']] for c in stage_1_cands]
        cross_scores = reranker.predict(cross_inp)
        
        reranked_idxs = np.argsort(cross_scores)[-k:][::-1].tolist()
        return [stage_1_cands[i] for i in reranked_idxs]

    # Forcefully balance the context window (3 Tables, 2 News)
    final_retrieved = get_top_k(table_chunks, k=3) + get_top_k(news_chunks, k=2)

    sources = list(set(c['source'] for c in final_retrieved))
    formatted = "\n\n***\n\n".join([f"[{c['source'].replace('_', ' ').title()}]\n{c['text']}" for c in final_retrieved])
    
    return formatted, sources

# ============================================================
# 4. Strict XML Prompt Builder
# ============================================================
def get_prompt(question: str, retrieved: str, instructions: str) -> tuple[str, str]:
    q = question.lower()
    
    # Existing Easy Dataset Math Hints
    if "revenue" in q and any(x in q for x in ["trend", "growth", "change", "over"]) and "top three" not in q:
        hint = "Extract revenue figures across all years to identify the trend direction and magnitude."
    elif any(x in q for x in ["balance sheet", "current assets", "liability", "equity"]):
        hint = "Extract current assets, total liabilities, total equity. Calculate liabilities/equity ratio. Show final ratio only."
    elif any(x in q for x in ["margin", "gross profit"]):
        hint = "Extract gross profit and revenue. Calculate the gross margin percentage."
        
    # NEW: Expert Dataset Hints
    elif "allocating capital" in q or any(x in q for x in ["investments", "share repurchases", "dividends"]):
        hint = "Identify all forms of capital allocation (e.g., stock buybacks/repurchases, dividends paid, CAPEX). List the exact dollar amounts for each from the Cash Flow statement."
    elif "focuses on revenue" in q or "top three" in q:
        hint = "Synthesize a numbered list of the top revenue drivers. Extract the exact growth percentages or dollar amounts from the news text and align them with the segments."
    else:
        hint = "Perform expert-level multi-hop reasoning. If the question asks for a list, provide a numbered list. If it asks for a calculation, provide the final number."

    system = f"""{instructions}

    Task guidance: {hint}

    CRITICAL RULES:
    1. NEWS: Copy EXACT text from the labeled news articles when quoting evidence. Do NOT paraphrase.
    2. FINANCIALS: You MUST explicitly quote the exact numerical figures from the tables to back up your reasoning.
    3. NUMBER FORMATTING: Convert all large currency figures into Billions formatted to one decimal place with a 'B' suffix (e.g., $35.0B) or Millions with an 'M' suffix (e.g., $22,968M) based on context. Format percentages to one decimal place (e.g., 14.0%).
    4. If no relevant evidence is found, write "None".

    Step 1: Perform all calculations, list structuring, and multi-hop reasoning inside <scratchpad> tags.
    Step 2: Output your final response strictly using these XML tags:
    <answer>
    [Your final answer here. If a list is requested, format as 1. 2. 3. Include exact numbers.]
    </answer>
    <evidence>
    [Exact quote from news supporting the answer, or "None"]
    </evidence>

    Retrieved Context:
    {retrieved}"""
    return system, question

# ============================================================
# 5. Perfect Regex Extraction
# ============================================================
def extract_answer(raw: str) -> str:
    ans_match = re.search(r'<answer>(.*?)</answer>', raw, re.DOTALL | re.IGNORECASE)
    answer_text = ans_match.group(1).strip() if ans_match else ""

    ev_match = re.search(r'<evidence>(.*?)</evidence>', raw, re.DOTALL | re.IGNORECASE)
    evidence_text = ev_match.group(1).strip() if ev_match else ""

    if answer_text:
        if evidence_text and evidence_text.lower() != "none":
            return f"Answer: {answer_text}\nNews Evidence: {evidence_text}"
        return f"Answer: {answer_text}\nNews Evidence: None"

    match = re.search(r'(Answer:.*)', raw, re.DOTALL | re.IGNORECASE)
    if match: return match.group(1).strip()
    return re.sub(r'<scratchpad>.*?</scratchpad>', '', raw, flags=re.DOTALL).strip()

# ============================================================
# Core Logic
# ============================================================
def build_event_map(dataset):
    event_map = {}
    ordered_ids = []
    for row in dataset:
        task_id, question, answer = row['task_id'], row['question'], row['answer']
        if task_id not in event_map:
            sections = parse_query(row['query'])
            chunks = build_structured_chunks(sections)
            event_map[task_id] = {"instructions": sections["instructions"], "chunks": chunks, "questions": [], "answers": []}
            ordered_ids.append(task_id)
        event_map[task_id]["questions"].append(question)
        event_map[task_id]["answers"].append(answer)
    return event_map, ordered_ids

async def call_groq(client: AsyncGroq, system: str, user: str) -> str:
    for attempt in range(MAX_RETRIES):
        try:
            resp = await client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                max_tokens=MAX_TOKENS, temperature=TEMPERATURE
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            if any(x in str(e).lower() for x in ["rate limit", "429", "413"]):
                await asyncio.sleep(60)
            else: return ""
    return ""

async def process_event(client, task_id, data, semaphore, tier):
    results = []
    async with semaphore:
        for q_idx, (question, reference) in enumerate(zip(data["questions"], data["answers"])):
            retrieved, sources = retrieve_chunks(data["chunks"], question)
            print(f"  [{task_id}] Q{q_idx + 1} | sources={sources}")

            system, user = get_prompt(question, retrieved, data["instructions"])
            raw = await call_groq(client, system, user)
            prediction = extract_answer(raw)

            results.append({
                "task_id": task_id, "tier": tier, "q_idx": q_idx + 1,
                "question": question, "answer": prediction, "gold": reference,
                "raw_output": raw, "sources_retrieved": sources
            })
            await asyncio.sleep(35)
    return results

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-output", default="reranked_expert.json")
    parser.add_argument("-limit", type=int, default=5)
    parser.add_argument("-api_key", required=True)
    args = parser.parse_args()

    # dataset = load_dataset("TheFinAI/PolyFiQA-Easy", split="test")
    # dataset_rows = list(dataset)[:args.limit]
    # event_map, ordered_ids = build_event_map(dataset_rows)

    # dataset = load_dataset("TheFinAI/PolyFiQA-Expert", split="test") # or Expert
    # dataset_rows = list(dataset)
    
    # # Set a fixed seed so your "random" batch is reproducible if the script crashes
    # random.seed(42) 
    
    # # Safely randomly sample the rows
    # dataset_rows = random.sample(dataset_rows, min(args.limit, len(dataset_rows)))
    
    # event_map, ordered_ids = build_event_map(dataset_rows)

    # For the Easy script use "TheFinAI/PolyFiQA-Easy", for the Expert script use "TheFinAI/PolyFiQA-Expert"
    dataset = load_dataset("TheFinAI/PolyFiQA-Easy", split="test") 
    
    # Load the ENTIRE dataset sequentially. No random sampling, no limits.
    dataset_rows = list(dataset)
    
    event_map, ordered_ids = build_event_map(dataset_rows)

    client = AsyncGroq(api_key=args.api_key)
    semaphore = asyncio.Semaphore(1)

    tasks = [process_event(client, tid, event_map[tid], semaphore, "expert") for tid in ordered_ids]
    nested = await asyncio.gather(*tasks)
    all_results = [r for event in nested for r in event]

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"Testing complete. Saved to {args.output}")

if __name__ == "__main__":
    asyncio.run(main())