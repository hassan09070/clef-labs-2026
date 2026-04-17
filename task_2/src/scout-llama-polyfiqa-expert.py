import os
import asyncio
import json
import re
import numpy as np
from datasets import load_dataset
# Swap AsyncGroq with your Scout API provider's client if it is not Groq
from groq import AsyncGroq 
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi




# === 1. Config ===
API_KEY       = ""
MODEL         = "meta-llama/llama-4-scout-17b-16e-instruct" 
MAX_TOKENS    = 512
TEMPERATURE   = 0.0
MAX_RETRIES   = 10
TOP_K         = 5     # Increased to 5 for the Hero Run
OUTPUT_FILE   = "polyfiqa_expert_hero_run_scout.json"

# === 2. Load Embedding Model ===
print("Loading multilingual-e5-small...")
embedder = SentenceTransformer("intfloat/multilingual-e5-small")
print("Embedding model ready.\n")

# === 3. Parse Query ===
def parse_query(query: str) -> dict:
    lines = query.split('\n')
    seps = []
    for i, line in enumerate(lines):
        if line.strip().startswith('---') or line.strip() == '***':
            seps.append(i)

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

    def get_lines(start, end):
        return '\n'.join(lines[start+1:end]).strip()

    if len(seps) < 3:
        seps = [idx['financial'] + 50, idx['financial'] + 100, idx['financial'] + 150]

    return {
        "instructions":     '\n'.join(lines[:idx['financial']]).strip(),
        "balance_sheet":    get_lines(idx['financial'], seps[0]),
        "cash_flow":        get_lines(seps[0],          seps[1]),
        "income_statement": get_lines(seps[1],          seps[2]),
        "english_news":     get_lines(idx['english'],  idx['chinese']),
        "chinese_news":     get_lines(idx['chinese'],  idx['japanese']),
        "japanese_news":    get_lines(idx['japanese'], idx['spanish']),
        "spanish_news":     get_lines(idx['spanish'],  idx['greek']),
        "greek_news":       get_lines(idx['greek'],    idx['question']),
    }

# === 4. Build Structured Chunks ===
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

# === 5. Hybrid Retrieval ===
def retrieve_chunks(chunks: list[dict], question: str, top_k: int = TOP_K, alpha: float = 0.5) -> tuple[str, list[str]]:
    if not chunks:
        return "", []

    texts = [c['text'] for c in chunks]
    tokenized_corpus = [t.lower().split() for t in texts]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = question.lower().split()
    bm25_scores = np.array(bm25.get_scores(tokenized_query))

    prefixed_q      = f"query: {question}"
    prefixed_chunks = [f"passage: {t}" for t in texts]
    q_emb  = embedder.encode([prefixed_q])
    c_embs = embedder.encode(prefixed_chunks, batch_size=32)
    dense_scores   = cosine_similarity(q_emb, c_embs)[0]

    def normalize(arr):
        if arr.max() == arr.min(): return np.zeros_like(arr)
        return (arr - arr.min()) / (arr.max() - arr.min() + 1e-9)

    norm_bm25  = normalize(bm25_scores)
    norm_dense = normalize(dense_scores)
    
    combined_scores = (alpha * norm_bm25) + ((1 - alpha) * norm_dense)
    top_idxs = sorted(np.argsort(combined_scores)[-top_k:].tolist())

    retrieved = [chunks[i] for i in top_idxs]
    sources   = list(set(c['source'] for c in retrieved))

    formatted = "\n\n***\n\n".join([
        f"[{c['source'].replace('_', ' ').title()}]\n{c['text']}"
        for c in retrieved
    ])
    return formatted, sources

# === 6. Load Dataset & Build Event Map ===
def build_event_map(dataset):
    event_map   = {}
    ordered_ids = []
    print("Parsing and chunking all 19 events...")
    for row in dataset:
        task_id  = row['task_id']
        question = row['question']
        answer   = row['answer']

        if task_id not in event_map:
            sections = parse_query(row['query'])
            chunks   = build_structured_chunks(sections)
            event_map[task_id] = {
                "instructions": sections["instructions"],
                "chunks":       chunks,
                "questions":    [],
                "answers":      [],
            }
            ordered_ids.append(task_id)

        event_map[task_id]["questions"].append(question)
        event_map[task_id]["answers"].append(answer)
    return event_map, ordered_ids

# === 7. Prompt Builder (Expert Multi-Hop Surgery) ===
def get_prompt(question: str, retrieved: str, instructions: str) -> tuple[str, str]:
    
    # Surgical Hint tailored perfectly to the 4 Expert questions
    hint = """This is a complex analytical query requiring multi-document cross-referencing. 
    1. Identify the core strategic focus (Revenue, Capital Allocation, Margins, or CapEx) requested.
    2. Read the multilingual News sections to extract the qualitative strategy, market outlook, or significance.
    3. Read the Financial Statements to find the exact quantitative numbers supporting this strategy.
    4. You MUST explicitly quote the exact financial figures from the tables in your answer."""

    # The Guillotine (ONLY for the final_ablation script, REMOVE for the Hero Scout run)
    

    # Explicit Rules for BOTH News and Financial Data
    system = f"""{instructions}

Task guidance: {hint}

CRITICAL RULES FOR EVIDENCE:
1. NEWS: Copy the EXACT text from the labeled news articles ([English News], [Chinese News], etc.). Do NOT paraphrase.
2. FINANCIALS: You MUST quote the exact numerical figures from the Financial Statements to support the news findings.
3. If no relevant news or financial data is found for a specific part, state "None".

Step 1: Reason through the multi-hop strategy and numbers inside <scratchpad> tags.
Step 2: Write your final response inside <answer> tags using this exact format:
    Answer: [Synthesize the strategy and quote the financial numbers here, under 150 words]
    News Evidence: [exact quote or "None"]

Retrieved Context:
{retrieved}""" # Use `retrieved` instead of `safe_retrieved` for your Scout Hero script

    return system, question

# === 8. API Call (Configured for Higher Capacity) ===
async def call_api(client: AsyncGroq, system: str, user: str) -> str:
    if not API_KEY:
        print("API Key missing! Set it in the environment.")
        return ""
        
    for attempt in range(MAX_RETRIES):
        try:
            # Assuming Scout uses standard OpenAI/Groq chat completions structure
            response = await client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user}
                ],
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"    API error: {e}. Retrying in 10s...")
            await asyncio.sleep(60)
            
    print("    Max retries exceeded")
    return ""

# === 9. Extract Answer ===
def extract_answer(raw: str) -> str:
    match = re.search(r'<answer>(.*?)</answer>', raw, re.DOTALL | re.IGNORECASE)
    if match: return match.group(1).strip()
    match = re.search(r'(Answer:.*)', raw, re.DOTALL | re.IGNORECASE)
    if match: return match.group(1).strip()
    return re.sub(r'<scratchpad>.*?</scratchpad>', '', raw, flags=re.DOTALL).strip()

# === 10. Process One Event ===
async def process_event(client: AsyncGroq, task_id: str, data: dict, semaphore: asyncio.Semaphore) -> list[dict]:
    results = []
    async with semaphore:
        for q_idx, (question, reference) in enumerate(zip(data["questions"], data["answers"])):
            retrieved, sources = retrieve_chunks(data["chunks"], question, TOP_K)
            print(f"  {task_id} | Q{q_idx+1} | {sources}")

            system, user = get_prompt(question, retrieved, data["instructions"])
            raw          = await call_api(client, system, user)
            prediction   = extract_answer(raw)

            results.append({
                "task_id":           task_id,
                "q_idx":             q_idx + 1,
                "question":          question,
                "reference":         reference,
                "sources_retrieved": sources,
                "retrieved_context": retrieved,
                "raw_output":        raw,
                "prediction":        prediction
            })
            # Reduced sleep time assuming the Scout API handles higher throughput
            await asyncio.sleep(25) 
    return results

# === 11. Main ===
async def main():
    print("Loading PolyFiQA-Easy...")
    dataset = load_dataset("TheFinAI/PolyFiQA-Expert", split="test")
    event_map, ordered_ids = build_event_map(dataset)

    # Note: If Scout uses a different base_url, configure it here.
    client    = AsyncGroq(api_key=API_KEY)
    
    # Increased concurrency from 1 to 3 since this is a higher-capacity API
    semaphore = asyncio.Semaphore(1)           

    print(f"\nRunning HERO PIPELINE with {MODEL}...\n")
    tasks = [process_event(client, tid, event_map[tid], semaphore) for tid in ordered_ids]

    all_nested  = await asyncio.gather(*tasks)
    all_results = [r for event in all_nested for r in event]

    failed_results = [r for r in all_results if not r['prediction']]
    if failed_results:
        print(f"\nRetrying {len(failed_results)} failed rows...")
        await asyncio.sleep(10)
        for r in failed_results:
            chunks             = event_map[r['task_id']]['chunks']
            retrieved, sources = retrieve_chunks(chunks, r['question'], TOP_K)
            system, user       = get_prompt(r['question'], retrieved, event_map[r['task_id']]['instructions'])
            raw                = await call_api(client, system, user)
            r['prediction']    = extract_answer(raw)
            await asyncio.sleep(5)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "model":    MODEL,
            "approach": "hero_run_scout",
            "results":  all_results
        }, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {OUTPUT_FILE}")

if __name__ == "__main__":
    asyncio.run(main())