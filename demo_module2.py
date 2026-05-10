"""
Module 2 Demo: Brand & Source Analyzer
Replays the latest raw JSON from Module 1 — no extra API cost for LLM queries,
only OpenAI is called for sentiment judging.

Run: python demo_module2.py
"""
import asyncio
import json
import glob
from src.utils.models import LLMResponse, PromptConfig
from src.analyzer.pipeline import analyze_responses

PROMPTS = [
    PromptConfig(
        prompt_id="pm_001",
        prompt_text="",
        category="project_management",
        target_brands=["Asana", "Jira", "Linear", "Monday.com", "Notion", "ClickUp", "Trello"],
    ),
    PromptConfig(
        prompt_id="pm_002",
        prompt_text="",
        category="project_management",
        target_brands=["Asana", "Jira", "Linear", "Monday.com", "Notion", "ClickUp", "Trello"],
    ),
]


def load_latest_run() -> list[LLMResponse]:
    """Load the most recently modified raw run JSON from data/raw/."""
    import os
    files = glob.glob("data/raw/run_*.json")
    if not files:
        raise FileNotFoundError("No raw run files found. Run demo_module1.py first.")
    latest = max(files, key=os.path.getmtime)  # newest by modification time
    print(f"Loading: {latest}\n")
    with open(latest, encoding="utf-8") as f:
        raw = json.load(f)
    return [LLMResponse(**r) for r in raw]


async def main():
    """Run full analysis pipeline on the latest raw query results."""
    responses = load_latest_run()
    # Only analyze successful responses
    ok_responses = [r for r in responses if not r.error and r.response_text]
    print(f"Analyzing {len(ok_responses)}/{len(responses)} successful responses...\n")

    mentions, citations = await analyze_responses(ok_responses, PROMPTS)

    print(f"\nTotal brand mentions: {len(mentions)}")
    print(f"Total citations found: {len(citations)}")

    # Show per-brand visibility summary
    from collections import Counter
    brand_counts = Counter(m.brand for m in mentions)
    print("\n--- Brand Visibility (mention count across all providers/prompts) ---")
    for brand, count in brand_counts.most_common():
        sentiments = [m.sentiment.value for m in mentions if m.brand == brand]
        pos = sentiments.count("positive")
        neg = sentiments.count("negative")
        print(f"  {brand:<15} mentions={count}  pos={pos}  neg={neg}")


if __name__ == "__main__":
    asyncio.run(main())
