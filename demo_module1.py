"""
Module 1 Demo: Prompt Runner
Run: python demo_module1.py
"""
import asyncio
from src.prompt_runner.runner import run_prompts
from src.utils.models import PromptConfig

PROMPTS = [
    PromptConfig(
        prompt_id="pm_001",
        prompt_text="What are the best project management tools for software teams in 2024? List the top 5 with pros and cons. Please cite your sources with URLs.",
        category="project_management",
        target_brands=["Asana", "Jira", "Linear", "Monday.com", "Notion", "ClickUp"],
    ),
    PromptConfig(
        prompt_id="pm_002",
        prompt_text="Which project management software is best for remote teams? Compare the most popular options. Please cite your sources with URLs.",
        category="project_management",
        target_brands=["Asana", "Jira", "Linear", "Monday.com", "Notion", "ClickUp"],
    ),
]


async def main():
    results = await run_prompts(
        prompts=PROMPTS,
        providers=["openai", "anthropic", "gemini"],  # no key → auto mock
    )

    print("\n--- Sample Response (OpenAI, first prompt) ---")
    for r in results:
        if r.provider.value == "openai" and r.prompt_id == "pm_001":
            print(r.response_text[:500])
            break


if __name__ == "__main__":
    asyncio.run(main())
