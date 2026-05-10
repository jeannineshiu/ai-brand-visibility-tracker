"""
Module 4 Demo: Recommendation Engine + FastAPI

Step 1 — print recommendations directly (no server needed)
Step 2 — start FastAPI server, then test with curl

Run: python demo_module4.py
"""
from rich.console import Console
from rich.table import Table

from src.recommender.feature_engineer import build_source_features, build_brand_features
from src.recommender.scorer import compute_opportunity_scores

console = Console()

TARGET_BRAND = "Asana"
COMPETITORS = ["Jira", "Linear", "Monday.com", "Notion", "ClickUp", "Trello"]


def print_recommendations():
    """Print ranked recommendations to terminal."""
    source_features = build_source_features()
    brand_features = build_brand_features(TARGET_BRAND)
    recs = compute_opportunity_scores(source_features, brand_features, TARGET_BRAND, COMPETITORS)

    table = Table(title=f"Source Recommendations for '{TARGET_BRAND}'")
    table.add_column("Rank", justify="right", style="dim")
    table.add_column("Domain Type", style="bold")
    table.add_column("Score", justify="right")
    table.add_column("Priority")
    table.add_column("Action")

    priority_style = {"HIGH": "red", "MEDIUM": "yellow", "LOW": "green"}
    for i, r in enumerate(recs[:8], 1):
        style = priority_style.get(r.priority, "white")
        table.add_row(
            str(i),
            r.domain_type,
            str(r.opportunity_score),
            f"[{style}]{r.priority}[/]",
            r.action[:60] + "..." if len(r.action) > 60 else r.action,
        )
    console.print(table)

    console.print("\n[bold]Top recommendation reasoning:[/]")
    for r in recs[:3]:
        console.print(f"  [cyan]{r.domain_type}[/] — {r.reason}")


def start_api():
    """Launch FastAPI server on first available port."""
    import socket
    import uvicorn

    port = 8000
    for p in range(8000, 8010):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("localhost", p)) != 0:  # port is free
                port = p
                break

    comp_json = '", "'.join(COMPETITORS)
    console.print(f"\n[bold green]Starting FastAPI server at http://localhost:{port}[/]")
    console.print(f"Docs: http://localhost:{port}/docs")
    console.print("\nTest with:")
    console.print(f'  curl -X POST http://localhost:{port}/recommendations \\')
    console.print('    -H "Content-Type: application/json" \\')
    body = f'{{"target_brand": "{TARGET_BRAND}", "competitors": ["{comp_json}"], "top_n": 5}}'
    console.print(f"    -d '{body}'")
    console.print(f"\n  curl http://localhost:{port}/visibility?brands=Asana,Jira,Monday.com\n")
    uvicorn.run("src.recommender.api:app", host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    print_recommendations()
    start_api()
