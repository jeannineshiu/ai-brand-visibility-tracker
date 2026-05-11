"""
Module 3 Demo: Visibility Metrics + Dashboard
Saves Module 1+2 results to DB (skips if run already saved), then launches Dash.

Run: python demo_module3.py
Then open: http://localhost:8050
"""
import asyncio
import json
import glob
import os
import socket
from src.utils.models import LLMResponse, PromptConfig
from src.analyzer.pipeline import analyze_responses
from src.storage import store
from src.metrics.calculator import visibility_summary, competitor_gap

BRANDS = ["Asana", "Jira", "Linear", "Monday.com", "Notion", "ClickUp", "Trello"]
TARGET_BRAND = "Asana"

PROMPTS = [
    PromptConfig(prompt_id="pm_001", prompt_text="", category="project_management", target_brands=BRANDS),
    PromptConfig(prompt_id="pm_002", prompt_text="", category="project_management", target_brands=BRANDS),
]


def load_latest_run() -> list[LLMResponse]:
    """Load the most recently modified raw run JSON."""
    files = glob.glob("data/raw/run_*.json")
    if not files:
        raise FileNotFoundError("Run demo_module1.py first.")
    latest = max(files, key=os.path.getmtime)
    print(f"Loading: {latest}")
    with open(latest, encoding="utf-8") as f:
        return [LLMResponse(**r) for r in json.load(f)]


def find_free_port(start: int = 8050) -> int:
    """Return the first available port starting from start."""
    for port in range(start, start + 10):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("localhost", port)) != 0:
                return port
    return start


async def main():
    """Run analysis, persist to DB (idempotent), print metrics, launch dashboard."""
    # 1. Load responses
    responses = load_latest_run()
    ok = [r for r in responses if not r.error and r.response_text]

    # 2. Persist — skip if run_id already in DB
    store.init()
    run_id = ok[0].run_id if ok else None
    if run_id and store.run_exists(run_id):
        print(f"\nRun {run_id} already in DB — skipping insert.")
    else:
        mentions, citations, _ = await analyze_responses(ok, PROMPTS)
        store.save_responses(ok)
        store.save_mentions(mentions)
        store.save_citations(citations)
        print(f"\nSaved {len(ok)} responses, {len(mentions)} mentions, {len(citations)} citations.")

    # 3. Print metrics
    print("\n=== Visibility Summary ===")
    print(visibility_summary(BRANDS).to_string(index=False))

    print(f"\n=== Competitor Gap for '{TARGET_BRAND}' ===")
    gap = competitor_gap(TARGET_BRAND)
    print(gap.to_string(index=False) if not gap.empty else "(no data — run demo_module1.py a few more times)")

    # 4. Launch dashboard on first available port
    port = find_free_port(8050)
    print(f"\nLaunching dashboard at http://localhost:{port} ...")
    from src.metrics.dashboard import build_dash_app
    app = build_dash_app(BRANDS, TARGET_BRAND)
    app.run(debug=False, port=port)


if __name__ == "__main__":
    asyncio.run(main())
