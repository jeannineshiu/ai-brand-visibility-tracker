import asyncio
import time
import uuid
from typing import Optional
from tenacity import retry, stop_after_attempt, wait_exponential

from src.utils.config import OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY, DEFAULT_MODEL_MAP
from src.utils.models import LLMResponse, LLMProvider


class OpenAIClient:
    def __init__(self, model: Optional[str] = None):
        from openai import AsyncOpenAI
        self.client = AsyncOpenAI(api_key=OPENAI_API_KEY)
        self.model = model or DEFAULT_MODEL_MAP["openai"]

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def query(self, prompt_id: str, prompt_text: str, run_id: str) -> LLMResponse:
        start = time.time()
        try:
            resp = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt_text}],
                temperature=0.3,
            )
            latency = int((time.time() - start) * 1000)
            return LLMResponse(
                run_id=run_id,
                prompt_id=prompt_id,
                prompt_text=prompt_text,
                provider=LLMProvider.OPENAI,
                model=self.model,
                response_text=resp.choices[0].message.content or "",
                latency_ms=latency,
            )
        except Exception as e:
            return LLMResponse(
                run_id=run_id,
                prompt_id=prompt_id,
                prompt_text=prompt_text,
                provider=LLMProvider.OPENAI,
                model=self.model,
                response_text="",
                latency_ms=int((time.time() - start) * 1000),
                error=str(e),
            )


class AnthropicClient:
    def __init__(self, model: Optional[str] = None):
        import anthropic
        self.client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
        self.model = model or DEFAULT_MODEL_MAP["anthropic"]

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def query(self, prompt_id: str, prompt_text: str, run_id: str) -> LLMResponse:
        start = time.time()
        try:
            resp = await self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt_text}],
            )
            latency = int((time.time() - start) * 1000)
            return LLMResponse(
                run_id=run_id,
                prompt_id=prompt_id,
                prompt_text=prompt_text,
                provider=LLMProvider.ANTHROPIC,
                model=self.model,
                response_text=resp.content[0].text,
                latency_ms=latency,
            )
        except Exception as e:
            return LLMResponse(
                run_id=run_id,
                prompt_id=prompt_id,
                prompt_text=prompt_text,
                provider=LLMProvider.ANTHROPIC,
                model=self.model,
                response_text="",
                latency_ms=int((time.time() - start) * 1000),
                error=str(e),
            )


class GeminiClient:
    def __init__(self, model: Optional[str] = None):
        from google import genai
        self.client = genai.Client(api_key=GOOGLE_API_KEY)
        self.model = model or DEFAULT_MODEL_MAP["gemini"]

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def query(self, prompt_id: str, prompt_text: str, run_id: str) -> LLMResponse:
        start = time.time()
        try:
            resp = await asyncio.to_thread(
                self.client.models.generate_content,
                model=self.model,
                contents=prompt_text,
            )
            latency = int((time.time() - start) * 1000)
            return LLMResponse(
                run_id=run_id,
                prompt_id=prompt_id,
                prompt_text=prompt_text,
                provider=LLMProvider.GEMINI,
                model=self.model,
                response_text=resp.text,
                latency_ms=latency,
            )
        except Exception as e:
            return LLMResponse(
                run_id=run_id,
                prompt_id=prompt_id,
                prompt_text=prompt_text,
                provider=LLMProvider.GEMINI,
                model=self.model,
                response_text="",
                latency_ms=int((time.time() - start) * 1000),
                error=str(e),
            )


class MockClient:
    """Used when API key not available — returns realistic fake responses."""
    MOCK_RESPONSES = {
        "openai": (
            "The best project management tools include Asana, Monday.com, and Jira. "
            "Asana is great for teams, Monday.com offers flexible workflows, and "
            "Jira is preferred by engineering teams. Notion is also rising. "
            "Sources: https://www.g2.com/categories/project-management, "
            "https://www.reddit.com/r/projectmanagement"
        ),
        "anthropic": (
            "Top project management tools: 1) Asana — excellent for cross-team visibility. "
            "2) Jira — deep engineering integrations. 3) Linear — modern, fast. "
            "4) Monday.com — visual and customizable. "
            "Reference: https://www.capterra.com/project-management-software/"
        ),
        "gemini": (
            "Popular PM tools include Trello, Asana, and ClickUp. Trello uses Kanban boards, "
            "Asana supports complex workflows, and ClickUp is an all-in-one option. "
            "See: https://www.techradar.com/best/best-project-management-software"
        ),
    }

    def __init__(self, provider: str):
        self.provider = provider
        self.model = f"mock-{provider}"

    async def query(self, prompt_id: str, prompt_text: str, run_id: str) -> LLMResponse:
        await asyncio.sleep(0.1)  # simulate latency
        return LLMResponse(
            run_id=run_id,
            prompt_id=prompt_id,
            prompt_text=prompt_text,
            provider=LLMProvider(self.provider),
            model=self.model,
            response_text=self.MOCK_RESPONSES.get(self.provider, "Mock response."),
            latency_ms=100,
        )
