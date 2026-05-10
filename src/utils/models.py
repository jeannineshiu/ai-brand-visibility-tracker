from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional
from enum import Enum


class LLMProvider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"


class Sentiment(str, Enum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"


class LLMResponse(BaseModel):
    run_id: str
    prompt_id: str
    prompt_text: str
    provider: LLMProvider
    model: str
    response_text: str
    latency_ms: int
    created_at: datetime = Field(default_factory=datetime.utcnow)
    error: Optional[str] = None


class BrandMention(BaseModel):
    run_id: str
    prompt_id: str
    provider: LLMProvider
    brand: str
    position: int  # 1st, 2nd, 3rd mention order
    sentiment: Sentiment
    snippet: str   # surrounding context
    created_at: datetime = Field(default_factory=datetime.utcnow)


class CitationSource(BaseModel):
    run_id: str
    prompt_id: str
    provider: LLMProvider
    url: Optional[str] = None
    domain: Optional[str] = None
    domain_type: Optional[str] = None  # reddit, editorial, brand, news, etc.
    created_at: datetime = Field(default_factory=datetime.utcnow)


class PromptConfig(BaseModel):
    prompt_id: str
    prompt_text: str
    category: str           # e.g. "project_management", "crm"
    target_brands: list[str]
