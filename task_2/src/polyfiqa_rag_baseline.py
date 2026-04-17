
import asyncio
import json
import re
import numpy as np
from datasets import load_dataset
from groq import AsyncGroq
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from rouge_score import rouge_scorer

# ── Config ────────────────────────────────────────────────
GROQ_API_KEY  = ""
MODEL         = "llama-3.1-8b-instant"
MAX_TOKENS    = 512
TEMPERATURE   = 0.0
MAX_RETRIES   = 10
BASE_BACKOFF  = 4
CHUNK_SIZE    = 100
CHUNK_OVERLAP = 20
TOP_K         = 6
OUTPUT_FILE   = "polyfiqa_rag_expert_results_llama_8b.json"

# ── Load Embedding Model Once ─────────────────────────────
print("Loading multilingual-e5-small...")
embedder = SentenceTransformer("intfloat/multilingual-e5-small")
print("Embedding model ready.\n")


# ── Step 1: Parse Query ───────────────────────────────────
def parse_query(query: str) -> dict:
    lines = query.split('\n')
    seps  = [i for i, line in enumerate(lines) if line.strip() == '---']

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

    return {
        # Instructions — reused verbatim in system prompt
        "instructions":     '\n'.join(lines[:idx['financial']]).strip(),

        # 3 financial tables
        "balance_sheet":    get_lines(idx['financial'], seps[0]),
        "cash_flow":        get_lines(seps[0],          seps[1]),
        "income_statement": get_lines(seps[1],          seps[2]),
       
        "english_news":     get_lines(idx['english'],  idx['chinese']),
        "chinese_news":     get_lines(idx['chinese'],  idx['japanese']),
        "japanese_news":    get_lines(idx['japanese'], idx['spanish']),
        "spanish_news":     get_lines(idx['spanish'],  idx['greek']),
        "greek_news":       get_lines(idx['greek'],    idx['question']),
    }


# ── Step 2: Build Chunks ──────────────────────────────────
def build_chunks(sections: dict) -> list[dict]:
    """
    Financial tables  → chunked (they are long markdown tables)
    News articles     → kept whole (only ~3 lines each,
                        chunking would destroy the full quote)
    """
    all_chunks = []

    # Chunk financial tables
    for source in ["balance_sheet", "cash_flow", "income_statement"]:
        text = sections[source]
        if not text.strip():
            continue
        words = text.split()
        step  = CHUNK_SIZE - CHUNK_OVERLAP
        for i in range(0, len(words), step):
            chunk = " ".join(words[i:i + CHUNK_SIZE])
            if chunk.strip():
                all_chunks.append({"source": source, "text": chunk})

    # Keep news articles whole
    for source in ["english_news", "chinese_news", "japanese_news",
                   "spanish_news", "greek_news"]:
        text = sections[source]
        if text.strip():
            all_chunks.append({"source": source, "text": text.strip()})

    sources = list(set(c['source'] for c in all_chunks))
    print(f"  {len(all_chunks)} chunks | sections: {sources}")
    return all_chunks


# ── Step 3: Retrieve ──────────────────────────────────────
def retrieve_chunks(chunks: list[dict], question: str, top_k: int = TOP_K) -> tuple[str, list[str]]:
    """
    multilingual-e5-small maps all languages into the same vector
    space — English question can match Chinese/Japanese/Greek chunks
    by semantic proximity, not keyword overlap.
    """
    if not chunks:
        return "", []

    prefixed_q      = f"query: {question}"
    prefixed_chunks = [f"passage: {c['text']}" for c in chunks]

    q_emb  = embedder.encode([prefixed_q])
    c_embs = embedder.encode(prefixed_chunks, batch_size=32)
    sims   = cosine_similarity(q_emb, c_embs)[0]

    # Preserve original document order after selecting top_k
    top_idxs = sorted(np.argsort(sims)[-top_k:].tolist())

    retrieved = [chunks[i] for i in top_idxs]
    sources   = list(set(c['source'] for c in retrieved))

    formatted = "\n\n---\n\n".join([
        f"[{c['source'].replace('_', ' ').title()}]\n{c['text']}"
        for c in retrieved
    ])

    return formatted, sources


# ── Step 4: Load Dataset & Build Event Map ────────────────
def build_event_map(dataset):
    event_map   = {}
    ordered_ids = []

    print("Parsing and chunking all 19 events...")
    for row in dataset:
        task_id  = row['task_id']
        question = row['question']
        answer   = row['answer']

        if task_id not in event_map:
            print(f"\n  {task_id}")
            sections = parse_query(row['query'])
            chunks   = build_chunks(sections)
            event_map[task_id] = {
                "instructions": sections["instructions"],
                "chunks":       chunks,
                "questions":    [],
                "answers":      [],
            }
            ordered_ids.append(task_id)

        event_map[task_id]["questions"].append(question)
        event_map[task_id]["answers"].append(answer)

    print(f"\nTotal events:   {len(event_map)}")
    print(f"Total questions:{sum(len(v['questions']) for v in event_map.values())}")
    return event_map, ordered_ids


# ── Step 5: Prompt Builder ────────────────────────────────
def get_prompt(question: str, retrieved: str, instructions: str) -> tuple[str, str]:
    """
    Combines:
    - Original dataset instructions (exact format expected)
    - Task-specific reasoning hint
    - Retrieved chunks (financial table chunks + full news articles)
    """
    q = question.lower()

    if "revenue" in q and "trend" in q:
        hint = "Extract revenue figures across years and identify the trend direction and magnitude."
    elif "balance sheet" in q or "current assets" in q or "liability" in q:
        hint = "Extract current assets, total liabilities, total equity. Calculate liabilities/equity. Show only final numbers — no calculation steps in the answer."
    elif "cash flow" in q or "operating" in q or "investing" in q:
        hint = "Extract operating, investing, and financing cash flows. Identify irregularities."
    elif "r&d" in q or "research" in q:
        hint = "Extract R&D expenditure and revenue. Calculate R&D/revenue ratio. Show only the final ratio — no calculation steps in the answer."
    else:
        hint = "Answer based on the financial data and news provided."

    system = f"""{instructions}

Task guidance: {hint}

Step 1: Reason through the numbers inside <scratchpad> tags.
Step 2: Write your final response inside <answer> tags using this exact format:
    Answer: [under 100 words]
    News Evidence: [quote the specific multilingual news that supports your answer, or "None"]

Retrieved Context:
{retrieved}"""

    return system, question


# ── Step 6: Groq Call With Backoff ────────────────────────
async def call_groq(client: AsyncGroq, system: str, user: str) -> str:
    for attempt in range(MAX_RETRIES):
        try:
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
            err = str(e).lower()
            print(f"    Full error: {e}")
            if any(x in err for x in ["rate limit", "429", "413"]):
                wait = BASE_BACKOFF ** (attempt + 1)
                print(f"    Rate limit → waiting {wait}s (attempt {attempt+1}/{MAX_RETRIES})")
                await asyncio.sleep(wait)
            else:
                print(f"    API error: {e}")
                return ""

    print("    Max retries exceeded")
    return ""


# ── Step 7: Extract Answer ────────────────────────────────
def extract_answer(raw: str) -> str:
    # Priority 1: <answer> tags
    match = re.search(r'<answer>(.*?)</answer>', raw, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # Priority 2: "Answer:" format (matches dataset's own format)
    match = re.search(r'(Answer:.*)', raw, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # Fallback: strip scratchpad, return rest
    return re.sub(r'<scratchpad>.*?</scratchpad>', '', raw, flags=re.DOTALL).strip()


# ── Step 8: Process One Event ─────────────────────────────
async def process_event(
    client:    AsyncGroq,
    task_id:   str,
    data:      dict,
    semaphore: asyncio.Semaphore
) -> list[dict]:

    results = []
    async with semaphore:
        for q_idx, (question, reference) in enumerate(
            zip(data["questions"], data["answers"])
        ):
            retrieved, sources = retrieve_chunks(data["chunks"], question, TOP_K)
            tokens_est = len(retrieved.split()) / 0.75

            print(f"  {task_id} | Q{q_idx+1} | ~{tokens_est:.0f} tokens | {sources}")

            system, user = get_prompt(question, retrieved, data["instructions"])
            raw          = await call_groq(client, system, user)
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
            await asyncio.sleep(20) 

    return results


# ── Step 9: ROUGE Scoring ─────────────────────────────────
def compute_rouge(results: list[dict]) -> dict:
    scorer = rouge_scorer.RougeScorer(
        ['rouge1', 'rouge2', 'rougeL'], use_stemmer=True
    )
    r1, r2, rL, failed = [], [], [], []

    for r in results:
        if not r['prediction']:
            failed.append(f"{r['task_id']} Q{r['q_idx']}")
            r1.append(0.0); r2.append(0.0); rL.append(0.0)
            continue
        s = scorer.score(r['reference'], r['prediction'])
        r1.append(s['rouge1'].fmeasure)
        r2.append(s['rouge2'].fmeasure)
        rL.append(s['rougeL'].fmeasure)

    if failed:
        print(f"\n⚠️  {len(failed)} failed (scored as 0):")
        for f in failed:
            print(f"    - {f}")

    return {
        "ROUGE-1":  round(sum(r1)/len(r1), 4),
        "ROUGE-2":  round(sum(r2)/len(r2), 4),
        "ROUGE-L":  round(sum(rL)/len(rL), 4),
        "n_total":  len(results),
        "n_failed": len(failed),
        "n_scored": len(results) - len(failed)
    }


# ── Step 10: Main ─────────────────────────────────────────
async def main():
    print("── Loading PolyFiQA-Easy ──")
    dataset = load_dataset("TheFinAI/PolyFiQA-Expert", split="test")
    event_map, ordered_ids = build_event_map(dataset)

    client    = AsyncGroq(api_key=GROQ_API_KEY)
    semaphore = asyncio.Semaphore(1)           # FIX 1: was 3

    print(f"\n── Running RAG with {MODEL} ──\n")
    tasks = [
        process_event(client, tid, event_map[tid], semaphore)
        for tid in ordered_ids
    ]

    all_nested  = await asyncio.gather(*tasks)
    all_results = [r for event in all_nested for r in event]

    # FIX 3: retry failures
    failed_results = [r for r in all_results if not r['prediction']]
    if failed_results:
        print(f"\n── Retrying {len(failed_results)} failed rows ──")
        await asyncio.sleep(30)

        for r in failed_results:
            print(f"  Retrying {r['task_id']} Q{r['q_idx']}...")
            chunks             = event_map[r['task_id']]['chunks']
            retrieved, sources = retrieve_chunks(chunks, r['question'], TOP_K)
            system, user       = get_prompt(
                                     r['question'], retrieved,
                                     event_map[r['task_id']]['instructions']
                                 )
            raw        = await call_groq(client, system, user)
            prediction = extract_answer(raw)

            r['prediction']          = prediction
            r['retrieved_context']   = retrieved
            r['sources_retrieved']   = sources
            r['raw_output']          = raw
            await asyncio.sleep(5)

    print("\n── Final ROUGE Scores ──────────────────")
    metrics = compute_rouge(all_results)
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "model":    MODEL,
            "approach": "structure_aware_rag_multilingual_e5_small",
            "config":   {
                "chunk_size":    CHUNK_SIZE,
                "chunk_overlap": CHUNK_OVERLAP,
                "top_k":         TOP_K
            },
            "metrics":  metrics,
            "results":  all_results
        }, f, indent=2, ensure_ascii=False)

    print(f"\n── Saved to {OUTPUT_FILE} ──")


if __name__ == "__main__":
    asyncio.run(main())