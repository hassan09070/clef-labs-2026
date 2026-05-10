import os
import re
from typing import Dict, List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field
import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

app = FastAPI(title="Qwen Trading API", version="1.0.0")

# ========================================================================= #
# 1. Initialize FinBERT and Qwen
# ========================================================================= #
print("Loading FinBERT model...")
device_num = 0 if torch.cuda.is_available() else -1
finbert = pipeline(
    "text-classification",
    model="ProsusAI/finbert",
    device=device_num
)

print("Configuring 4-bit quantization for Qwen...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

model_id = "Qwen/Qwen2.5-7B-Instruct"
print(f"Loading tokenizer for {model_id}...")
tokenizer = AutoTokenizer.from_pretrained(model_id)

print(f"Loading {model_id}...")
if torch.cuda.is_available():
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto"
    )
else:
    print("WARNING: CUDA is not available. Loading on CPU which will be very slow.")
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cpu")


# ========================================================================= #
# 2. Helper Functions
# ========================================================================= #
def get_chunked_sentiment(news_data) -> float:
    if isinstance(news_data, (list, tuple)):
        news_text = " ".join([str(item) for item in news_data if item])
    else:
        news_text = str(news_data)

    if not news_text or len(news_text.strip()) == 0 or news_text.strip().lower() in ['nan', 'none']:
        return 0.0

    words = news_text.split()
    chunk_size = 350
    chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    chunk_scores = []

    for chunk in chunks:
        try:
            result = finbert(chunk, padding=True, truncation=True, max_length=512)[0]
            label = result['label']
            if label == 'positive':
                chunk_scores.append(1.0)
            elif label == 'negative':
                chunk_scores.append(-1.0)
            else:
                chunk_scores.append(0.0)
        except Exception as e:
            print(f"Error on chunk: {e}")
            continue

    active_scores = [score for score in chunk_scores if score != 0.0]
    if not active_scores:
        return 0.0
    return sum(active_scores) / len(active_scores)


def run_qwen_inference(prompt: str, temperature: float = 0.1) -> str:
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs, max_new_tokens=150, temperature=temperature, do_sample=True
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    decision_match = re.search(r'Decision:\s*(BUY|SELL|HOLD)', response, re.IGNORECASE)
    return decision_match.group(1).upper() if decision_match else "HOLD"


def get_tsla_decision(asset_name, momentum, trend_5d, sentiment, sentiment_3d, price):
    prompt = f"""You are a contrarian quantitative trader specializing in Tesla (TSLA).
TSLA is driven by retail emotion, hype cycles, and short-squeezes. Your strategy is to identify extreme overextensions or divergences between price action and news sentiment to fade the crowd.

Evaluation Framework:

[EXAMPLE 1: Exhaustion / Divergence Buy]
- Current Price: $165.00
- 5-Day Trend: -8.50%
- Daily Momentum: -0.05
- Today's News Sentiment: -0.1
- 3-Day Average Sentiment: 0.2
Reasoning: Price has aggressively dumped (-8.5%), but recent sentiment is actually mildly positive, showing a divergence where sellers are exhausted.
Decision: BUY

[EXAMPLE 2: Euphoria / Selling the Hype]
- Current Price: $240.00
- 5-Day Trend: 12.20%
- Daily Momentum: 0.08
- Today's News Sentiment: 0.95
- 3-Day Average Sentiment: 0.85
Reasoning: Massive 5-day run-up combined with extreme euphoric sentiment indicates a crowded, overextended trade ripe for a sharp pullback.
Decision: SELL

[EXAMPLE 3: No Edge / Choppy]
- Current Price: $205.00
- 5-Day Trend: 1.20%
- Daily Momentum: 0.01
- Today's News Sentiment: 0.3
- 3-Day Average Sentiment: 0.2
Reasoning: Both price action and sentiment are mildly positive but lack the extremes or divergences required for a high-probability contrarian setup.
Decision: HOLD

[CURRENT DATA FOR {asset_name}]
- Current Price: ${price}
- 5-Day Trend: {trend_5d:.2%}
- Daily Momentum: {momentum}
- Today's News Sentiment: {sentiment}
- 3-Day Average Sentiment: {sentiment_3d:.2f}

Based on the current data, evaluate the setup.
Provide your response strictly in the following format:
Reasoning: [Analyze if there is extreme overextension or a divergence between price and sentiment]
Decision: [BUY, SELL, or HOLD]
"""
    return run_qwen_inference(prompt, temperature=0.2)


def get_generic_decision(asset_name, momentum, trend_5d, sentiment, sentiment_3d, price):
    prompt = f"""You are a quantitative financial agent specializing in Bitcoin (BTC).
BTC is a momentum-driven asset. Your strategy is to ride the trend when mathematical momentum and news sentiment align, and to HOLD when signals conflict in order to preserve capital.

Here are examples of your logic:

[EXAMPLE 1: Bullish Confluence]
- Momentum: 0.04
- Daily News Sentiment Score: 0.8
Decision: BUY
Reasoning: Positive momentum perfectly aligns with strong positive sentiment, indicating a high-probability trend continuation.

[EXAMPLE 2: Bearish Confluence]
- Momentum: -0.05
- Daily News Sentiment Score: -0.7
Decision: SELL
Reasoning: Negative momentum is confirmed by highly negative news sentiment, signaling a high-probability sell-off.

[EXAMPLE 3: Conflicting Signals (Bullish News, Bearish Price)]
- Momentum: -0.03
- Daily News Sentiment Score: 0.6
Decision: HOLD
Reasoning: Positive news is being ignored by negative price action; conflicting signals dictate preserving capital.

[EXAMPLE 4: Weak/Neutral Signals]
- Momentum: 0.005
- Daily News Sentiment Score: 0.1
Decision: HOLD
Reasoning: Both momentum and sentiment are flat, providing no directional edge.

[CURRENT DATA]
- Current Price: {price}
- Momentum: {momentum}
- Daily News Sentiment Score: {sentiment}

Based on this, make a trading decision for tomorrow.
Provide your response strictly in the following format:
Decision: [BUY, SELL, or HOLD]
Reasoning: [One brief sentence explaining why]
"""
    return run_qwen_inference(prompt, temperature=0.1)


# ========================================================================= #
# 3. Pydantic Models for FastAPI
# ========================================================================= #
class HistoricalPrice(BaseModel):
    date: str
    price: float


class TradingRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    date: str
    price: Dict[str, float]
    news: Dict[str, List[str]]
    symbol: List[str]
    momentum: Optional[Dict[str, str]] = None
    history_price: Dict[str, List[HistoricalPrice]] = Field(default_factory=dict)
    ten_k: Optional[Dict[str, List[str]]] = Field(default=None, alias="10k")
    ten_q: Optional[Dict[str, List[str]]] = Field(default=None, alias="10q")


class TradingResponse(BaseModel):
    recommended_action: str


# ========================================================================= #
# 4. API Endpoints
# ========================================================================= #
@app.get("/")
async def home():
    return {"message": "Qwen Trading API for FinMMEval Task 3 is running."}


@app.get("/health")
async def health():
    return {"status": "healthy", "model": "Qwen/Qwen2.5-7B-Instruct"}


@app.post("/trading_action/", response_model=TradingResponse)
async def get_trading_decision(request: TradingRequest):
    try:
        if not request.symbol:
            raise HTTPException(status_code=400, detail="No symbol provided")

        symbol = request.symbol[0]
        if symbol not in request.price:
            raise HTTPException(status_code=400, detail=f"No price for symbol {symbol}")

        price = request.price[symbol]

        # Calculate momentum
        momentum_raw = (request.momentum or {}).get(symbol, "0.0")
        try:
            momentum = float(momentum_raw)
        except ValueError:
            momentum = 0.0

        # Calculate 5-Day Trend
        history = request.history_price.get(symbol, [])
        history_prices = [item.price for item in history]
        trend_5d = 0.0
        if history_prices:
            idx = -5 if len(history_prices) >= 5 else 0
            old_price = history_prices[idx]
            if old_price > 0:
                trend_5d = (price - old_price) / old_price

        # Calculate sentiment
        news_items = request.news.get(symbol, [])
        sentiment = get_chunked_sentiment(news_items)
        sentiment_3d = sentiment  # Fallback to current sentiment

        # Retrieve logic based on symbol
        if symbol.upper() == "TSLA":
            decision = get_tsla_decision(symbol, momentum, trend_5d, sentiment, sentiment_3d, price)
        else:
            decision = get_generic_decision(symbol, momentum, trend_5d, sentiment, sentiment_3d, price)

        return TradingResponse(recommended_action=decision)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to get trading decision: {exc}") from exc


if __name__ == "__main__":
    import uvicorn

    print("Starting Qwen Trading API...")
    print("API will be available at: http://0.0.0.0:62237")
    uvicorn.run(app, host="0.0.0.0", port=62237, log_level="info")
