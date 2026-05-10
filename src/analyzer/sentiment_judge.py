"""LLM-as-judge: scores sentiment for each brand mention using OpenAI."""
import asyncio
import json
from openai import AsyncOpenAI
from src.utils.config import OPENAI_API_KEY
from src.utils.models import Sentiment

_client = None

JUDGE_SYSTEM = """\
You are a sentiment analyst. Given a text snippet from an AI assistant's response and a brand name,
classify the sentiment toward that brand as one of: positive, neutral, negative.

Rules:
- positive: recommended, praised, described with advantages
- negative: criticized, warned against, described with significant drawbacks
- neutral: mentioned factually without clear positive or negative framing

Respond ONLY with valid JSON: {"sentiment": "positive"|"neutral"|"negative", "reason": "<10 words max>"}
"""


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    return _client


async def judge_sentiment(brand: str, snippet: str) -> tuple[Sentiment, str]:
    """Call OpenAI to judge sentiment. Returns (Sentiment, reason)."""
    prompt = f'Brand: "{brand}"\nSnippet: "{snippet}"'
    try:
        resp = await _get_client().chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        data = json.loads(resp.choices[0].message.content or "{}")
        sentiment = Sentiment(data.get("sentiment", "neutral"))
        reason = data.get("reason", "")
        return sentiment, reason
    except Exception:
        return Sentiment.NEUTRAL, "judge_error"


async def judge_batch(
    items: list[tuple[str, str]]  # [(brand, snippet), ...]
) -> list[tuple[Sentiment, str]]:
    """Judge multiple (brand, snippet) pairs concurrently."""
    return await asyncio.gather(*[judge_sentiment(b, s) for b, s in items])
