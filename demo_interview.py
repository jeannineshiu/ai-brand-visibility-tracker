"""
AI Brand Visibility Tracker — Interview Demo
One command. All 4 modules. ~45 seconds.

Usage:
    python demo_interview.py            # uses configured backend (BigQuery or SQLite)
    python demo_interview.py --offline  # forces SQLite + seeds demo data

What this shows:
    1. 85 unit tests passing (pytest)
    2. Real data: brand visibility table + citation source breakdown
    3. LightGBM opportunity scores for a target brand
    4. Live FastAPI: /recommendations and /visibility endpoints
    5. Architecture decision summary
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from rich import box
from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table

console = Console()
ROOT = Path(__file__).parent


# ── helpers ───────────────────────────────────────────────────────────────────

def _step(n: int, label: str):
    console.print()
    console.print(Rule(f"[bold cyan]{n}/5  {label}[/]"))


def _ok(msg: str):
    console.print(f"  [bold green]✓[/]  {msg}")


def _warn(msg: str):
    console.print(f"  [yellow]⚠[/]  {msg}")


# ── optional: seed SQLite with demo data ──────────────────────────────────────

def _seed_demo_data():
    """Populate SQLite with realistic demo records so offline mode looks good."""
    import random
    from datetime import datetime, timedelta
    from src.utils.models import BrandMention, CitationSource, LLMProvider, Sentiment
    from src.storage import sqlite_store

    with sqlite_store.get_connection() as conn:
        n = conn.execute("SELECT COUNT(*) FROM brand_mentions").fetchone()[0]
    if n >= 50:
        return  # already has enough data

    console.print("  [dim]Seeding demo data into SQLite...[/]", end=" ")

    random.seed(42)
    brands = {
        "project_management": ["Asana", "Jira", "Linear", "Monday.com", "Notion", "ClickUp"],
        "crm":                ["HubSpot", "Salesforce", "Pipedrive", "Attio", "Zoho CRM"],
        "ai_writing":         ["Jasper", "Copy.ai", "Grammarly", "Writer", "Notion AI"],
        "developer_tools":    ["GitHub", "GitLab", "Linear", "Jira", "Shortcut"],
    }
    cite_urls = [
        ("g2.com", "review_site"), ("capterra.com", "review_site"),
        ("techcrunch.com", "tech_media"), ("venturebeat.com", "tech_media"),
        ("reddit.com", "community"), ("news.ycombinator.com", "community"),
        ("stackoverflow.com", "developer"), ("dev.to", "developer"),
        ("forbes.com", "business_media"), ("medium.com", "blog_platform"),
    ]
    providers = [LLMProvider.OPENAI, LLMProvider.ANTHROPIC, LLMProvider.GEMINI]
    sentiments = [Sentiment.POSITIVE] * 6 + [Sentiment.NEUTRAL] * 3 + [Sentiment.NEGATIVE]

    mentions = []
    citations = []
    base_date = datetime(2026, 4, 1)

    for day_offset in range(40):
        dt = base_date + timedelta(days=day_offset)
        run_id = f"demo-run-{day_offset:02d}"
        for cat, brand_list in brands.items():
            prompt_id = f"{cat[:3]}_demo_{day_offset % 10:02d}"
            for provider in providers:
                # 2–4 brand mentions per (prompt, provider)
                n_brands = random.randint(2, 4)
                selected = random.sample(brand_list, min(n_brands, len(brand_list)))
                for pos, brand in enumerate(selected, 1):
                    m = BrandMention(
                        run_id=run_id, prompt_id=prompt_id,
                        provider=provider, brand=brand, position=pos,
                        sentiment=random.choice(sentiments),
                        snippet=f"{brand} is recommended for {cat.replace('_', ' ')}.",
                        created_at=dt,
                    )
                    mentions.append(m)

                # 0–2 citations per (prompt, provider)
                for _ in range(random.randint(0, 2)):
                    domain, dtype = random.choice(cite_urls)
                    c = CitationSource(
                        run_id=run_id, prompt_id=prompt_id,
                        provider=provider,
                        url=f"https://www.{domain}/some-article",
                        domain=domain, domain_type=dtype,
                        created_at=dt,
                    )
                    citations.append(c)

    sqlite_store.save_mentions(mentions)
    sqlite_store.save_citations(citations)

    # Seed llm_responses so visibility % denominator is correct
    with sqlite_store.get_connection() as conn:
        for day_offset in range(40):
            for cat in brands:
                prompt_id = f"{cat[:3]}_demo_{day_offset % 10:02d}"
                run_id = f"demo-run-{day_offset:02d}"
                dt = (base_date + timedelta(days=day_offset)).isoformat()
                for provider in providers:
                    conn.execute(
                        "INSERT OR REPLACE INTO llm_responses VALUES (?,?,?,?,?,?,?,?,?)",
                        (run_id, prompt_id, "", provider.value, "mock", "", 100, None, dt),
                    )

    console.print(f"[green]{len(mentions)} mentions, {len(citations)} citations[/]")


# ── step 1: tests ─────────────────────────────────────────────────────────────

def step_tests():
    _step(1, "Test Suite")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--tb=no", "--no-header"],
        capture_output=True, text=True,
    )
    lines = [l for l in result.stdout.strip().splitlines() if l.strip()]
    summary = lines[-1] if lines else "no output"
    if result.returncode == 0:
        _ok(summary)
    else:
        _warn(f"Some tests failed: {summary}")


# ── step 2: data ──────────────────────────────────────────────────────────────

def step_data():
    _step(2, "Stored Data")
    from src.metrics.calculator import visibility_summary, citation_type_breakdown
    from src.storage.sqlite_store import get_connection

    with get_connection() as conn:
        n_m = conn.execute("SELECT COUNT(*) FROM brand_mentions").fetchone()[0]
        n_c = conn.execute("SELECT COUNT(*) FROM citation_sources").fetchone()[0]
        n_brands = conn.execute("SELECT COUNT(DISTINCT brand) FROM brand_mentions").fetchone()[0]

    if n_m < 10:
        _warn("Very little data. Run: python run_batch_pipeline.py --category crm")
        _warn("Or use: python demo_interview.py --offline  for seeded demo data")
        return False

    _ok(f"{n_m:,} brand mentions  ·  {n_c:,} citations  ·  {n_brands} brands")

    # Visibility table
    df = visibility_summary()
    vtable = Table(title="Brand Visibility (Top 8)", box=box.SIMPLE_HEAVY, show_lines=False)
    vtable.add_column("Brand", style="bold")
    vtable.add_column("Mentions", justify="right")
    vtable.add_column("Avg Position", justify="right")
    vtable.add_column("Positive %", justify="right", style="green")
    vtable.add_column("Visibility %", justify="right", style="cyan")
    for _, row in df.head(8).iterrows():
        vtable.add_row(
            row["brand"], str(int(row["mentions"])),
            f"{row['avg_position']:.1f}",
            f"{row['pos_rate']:.0f}%",
            f"{row['visibility_pct']:.1f}%",
        )
    console.print(vtable)

    # Citation table
    cdf = citation_type_breakdown()
    ctable = Table(title="Citation Source Types", box=box.SIMPLE_HEAVY, show_lines=False)
    ctable.add_column("Domain Type")
    ctable.add_column("Citations", justify="right", style="cyan")
    ctable.add_column("Unique Domains", justify="right")
    for _, row in cdf.iterrows():
        ctable.add_row(row["domain_type"], str(int(row["count"])), str(int(row["unique_domains"])))
    console.print(ctable)

    return True


# ── step 3: LightGBM ──────────────────────────────────────────────────────────

def step_lgbm():
    _step(3, "LightGBM Opportunity Model")
    from src.recommender.train_lgbm import load_model, load_features_df, build_training_data, train, save_model, predict_opportunities

    model, features, le = load_model()
    df = load_features_df()

    if model is None:
        console.print("  [dim]No saved model — training now...[/]", end=" ")
        try:
            df = build_training_data()
            model, le, features = train(df)
            save_model(model, le, features, df)
            console.print("[green]done[/]")
        except Exception as e:
            _warn(f"Cannot train: {e}")
            return

    _ok(f"Model loaded  ·  features: {len(features)}  ·  training rows: {len(df) if df is not None else '?'}")

    if df is not None and not df.empty:
        # Pick the brand with most data
        target = df["brand"].value_counts().index[0]
        result = predict_opportunities(model, features, le, target, df)
        if not result.empty:
            rtable = Table(title=f"Opportunity Scores — {target}", box=box.SIMPLE_HEAVY, show_lines=False)
            rtable.add_column("Domain Type")
            rtable.add_column("Score", justify="right", style="cyan")
            rtable.add_column("Gap vs Competitors", justify="right")
            for _, row in result.iterrows():
                gap = float(row["competitor_avg_cooccurrence"]) - float(row["cooccurrence"])
                gap_str = f"+{gap:.1f}" if gap > 0 else f"{gap:.1f}"
                gap_style = "red" if gap > 0 else "green"
                rtable.add_row(
                    row["domain_type"],
                    f"{row['opportunity_score']:.1f}",
                    f"[{gap_style}]{gap_str}[/]",
                )
            console.print(rtable)
            console.print(f"  [dim]Positive gap = competitors outrank {target} here → high opportunity[/]")


# ── step 4: live API ───────────────────────────────────────────────────────────

def step_api(offline: bool):
    _step(4, "Live FastAPI Demo")

    # Build subprocess env: use SQLite if offline
    env = os.environ.copy()
    if offline:
        env["GCP_PROJECT_ID"] = ""
        env["GOOGLE_APPLICATION_CREDENTIALS"] = ""

    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "src.recommender.api:app",
         "--port", "8765", "--log-level", "error"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    try:
        import httpx

        # Wait for server to be ready (up to 10s)
        for i in range(20):
            try:
                r = httpx.get("http://localhost:8765/health", timeout=1)
                if r.status_code == 200:
                    break
            except Exception:
                time.sleep(0.5)
        else:
            _warn("API did not start in time — skip")
            proc.terminate()
            return

        health = httpx.get("http://localhost:8765/health").json()
        model_tag = "[green]LightGBM[/]" if health.get("lgbm_model_loaded") else "[yellow]rule-based fallback[/]"
        _ok(f"API ready at [cyan]http://localhost:8765[/cyan]  ·  model: {model_tag}")

        # ── POST /recommendations ─────────────────────────────────────────────
        r = httpx.post("http://localhost:8765/recommendations", json={
            "target_brand": "Asana",
            "competitors": ["Jira", "Linear", "Monday.com", "Notion"],
            "top_n": 5,
        }, timeout=10)
        recs = r.json()

        rec_table = Table(
            title="POST /recommendations  {target_brand: Asana}",
            box=box.SIMPLE_HEAVY, show_lines=False,
        )
        rec_table.add_column("Domain Type")
        rec_table.add_column("Score", justify="right", style="cyan")
        rec_table.add_column("Priority")
        rec_table.add_column("Action")
        rec_table.add_column("Engine", style="dim")
        colors = {"HIGH": "red", "MEDIUM": "yellow", "LOW": "green"}
        for rec in recs:
            c = colors.get(rec["priority"], "white")
            action = rec["action"][:55] + "…" if len(rec["action"]) > 55 else rec["action"]
            rec_table.add_row(
                rec["domain_type"],
                f"{rec['opportunity_score']:.1f}",
                f"[{c}]{rec['priority']}[/]",
                action,
                rec["model_used"],
            )
        console.print(rec_table)

        # ── GET /visibility ───────────────────────────────────────────────────
        r2 = httpx.get("http://localhost:8765/visibility?brands=Asana,Jira,Linear", timeout=10)
        vis = r2.json()
        if vis:
            v_table = Table(
                title="GET /visibility?brands=Asana,Jira,Linear",
                box=box.SIMPLE_HEAVY, show_lines=False,
            )
            v_table.add_column("Brand", style="bold")
            v_table.add_column("Mentions", justify="right")
            v_table.add_column("Avg Position", justify="right")
            v_table.add_column("Positive %", justify="right", style="green")
            v_table.add_column("Visibility %", justify="right", style="cyan")
            for v in vis:
                v_table.add_row(
                    v["brand"], str(v["mentions"]),
                    f"{v['avg_position']:.1f}",
                    f"{v['pos_rate']:.0f}%",
                    f"{v['visibility_pct']:.1f}%",
                )
            console.print(v_table)

    finally:
        proc.terminate()
        proc.wait()


# ── step 5: highlights ────────────────────────────────────────────────────────

def step_highlights():
    _step(5, "Architecture Decision Points")

    points = [
        ("asyncio.gather",   "3 LLMs queried in parallel — not sequentially"),
        ("Majority voting",  "Run N trials → keep brand only if in >50% → filters LLM noise"),
        ("LLM-as-judge",     "gpt-4o-mini scores each mention as pos/neutral/neg (batched)"),
        ("spaCy NER",        "extract_entities() surfaces unknown competitors organically"),
        ("LightGBM",         "R²=0.80  target = competitor_cooccurrence - brand_cooccurrence"),
        ("Dual backend",     "Same interface routes to BigQuery (prod) or SQLite (local/test)"),
        ("Dynamic prompts",  "POST /prompts — add prompts at runtime without touching Python"),
        ("Auth",             "POST /retrain requires X-API-Key header (FastAPI Depends)"),
        ("Tests",            "85 unit tests — all LLM calls mocked with AsyncMock"),
        ("CI",               "GitHub Actions runs pytest on every push to main"),
        ("Docker",           "Model loaded via volume mount — survives container restart"),
    ]

    table = Table(box=box.SIMPLE_HEAVY, show_header=False, show_lines=False, padding=(0, 2))
    table.add_column(style="cyan bold", no_wrap=True)
    table.add_column()
    for label, desc in points:
        table.add_row(label, desc)
    console.print(table)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Interview demo")
    parser.add_argument("--offline", action="store_true",
                        help="Force SQLite + seed demo data (no BigQuery / internet needed)")
    args = parser.parse_args()

    console.print()
    console.print(Panel(
        "[bold cyan]AI Brand Visibility Tracker[/bold cyan]\n"
        "[dim]Tracks brand mentions in LLM responses · "
        "Sentiment · Citation co-occurrence · LightGBM opportunity scoring[/dim]\n\n"
        "[dim]Problem:  ChatGPT & Gemini are becoming primary B2B discovery surfaces.\n"
        "          Traditional SEO tools are blind to AI-generated recommendations.\n"
        "Solution: Query 3 LLMs with 40 prompts, detect brands, model opportunity gaps.[/dim]",
        border_style="cyan",
        title="[bold]Interview Demo[/bold]",
        subtitle="[dim]python demo_interview.py[/dim]",
    ))

    if args.offline:
        import src.storage.store as store_mod
        store_mod._USE_BIGQUERY = False
        from src.storage.sqlite_store import init_db
        init_db()
        _seed_demo_data()

    step_tests()
    has_data = step_data()
    if has_data:
        step_lgbm()
        step_api(offline=args.offline)
    step_highlights()

    console.print()
    console.print(Panel(
        "  [cyan]python demo_module3.py[/cyan]           → Plotly Dash dashboard  "
        "[dim](localhost:8050)[/dim]\n"
        "  [cyan]python demo_module4.py[/cyan]           → FastAPI + Swagger docs  "
        "[dim](localhost:8000/docs)[/dim]\n"
        "  [cyan]cat src/recommender/train_lgbm.py[/cyan] → LightGBM feature engineering\n"
        "  [cyan]cat src/analyzer/pipeline.py[/cyan]     → Full async pipeline orchestration",
        title="[bold green]✓ Demo complete — explore more[/bold green]",
        border_style="green",
    ))


if __name__ == "__main__":
    main()
