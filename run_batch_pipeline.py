"""
Batch pipeline: run 40 prompts × 3 LLMs → analyze → save to BigQuery.

Usage:
    python run_batch_pipeline.py            # run all 40 prompts
    python run_batch_pipeline.py --dry-run  # show what would run, no API calls
    python run_batch_pipeline.py --category crm  # run one category only

Cost estimate: ~$0.05–0.10 per full run of 40 prompts × 3 LLMs
Run twice to get 240 rows for LightGBM training.
"""
import asyncio
import argparse
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

from data.prompts_batch import BATCH_PROMPTS
from src.prompt_runner.runner import run_prompts
from src.analyzer.pipeline import analyze_responses
from src.storage import store

console = Console()

BATCH_SIZE = 10      # prompts per batch (avoids rate limits)
BATCH_PAUSE = 3      # seconds between batches


async def run_pipeline(prompts, dry_run: bool = False):
    """Run prompts in batches, analyze, and save each batch to DB."""
    store.init()

    total = len(prompts)
    all_mentions = []
    all_citations = []

    console.print(f"\n[bold]Starting batch pipeline[/]")
    console.print(f"Prompts: [cyan]{total}[/]  Batch size: [cyan]{BATCH_SIZE}[/]  "
                  f"Estimated queries: [cyan]{total * 3}[/]")

    if dry_run:
        console.print("\n[yellow]DRY RUN — listing prompts only:[/]")
        for p in prompts:
            console.print(f"  [{p.category}] {p.prompt_id}: {p.prompt_text[:70]}...")
        return

    batches = [prompts[i:i + BATCH_SIZE] for i in range(0, len(prompts), BATCH_SIZE)]

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total} batches"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Running batches...", total=len(batches))

        for i, batch in enumerate(batches):
            progress.update(task, description=f"Batch {i+1}/{len(batches)} ({batch[0].category})")

            # 1. Query all LLMs
            responses = await run_prompts(batch, providers=["openai", "anthropic", "gemini"])
            ok = [r for r in responses if not r.error and r.response_text]

            if not ok:
                console.print(f"  [yellow]Batch {i+1}: no successful responses, skipping[/]")
                progress.advance(task)
                continue

            # 2. Skip if already saved
            run_id = ok[0].run_id
            if store.run_exists(run_id):
                console.print(f"  [dim]Batch {i+1}: run {run_id} already saved, skipping[/]")
                progress.advance(task)
                continue

            # 3. Analyze
            mentions, citations = await analyze_responses(ok, batch)
            all_mentions.extend(mentions)
            all_citations.extend(citations)

            # 4. Save
            store.save_responses(ok)
            store.save_mentions(mentions)
            store.save_citations(citations)

            progress.advance(task)

            # Pause between batches to avoid rate limits (except last batch)
            if i < len(batches) - 1:
                await asyncio.sleep(BATCH_PAUSE)

    console.print(f"\n[bold green]Done![/]")
    console.print(f"Total mentions saved: [cyan]{len(all_mentions)}[/]")
    console.print(f"Total citations saved: [cyan]{len(all_citations)}[/]")

    # Summary table
    if all_mentions:
        from collections import Counter
        counts = Counter(m.brand for m in all_mentions)
        console.print("\n[bold]Brand mention counts:[/]")
        for brand, count in counts.most_common(10):
            console.print(f"  {brand:<20} {count}")


def parse_args():
    parser = argparse.ArgumentParser(description="Batch pipeline for brand visibility data generation")
    parser.add_argument("--dry-run", action="store_true", help="List prompts without making API calls")
    parser.add_argument("--category", choices=["project_management", "crm", "ai_writing", "developer_tools"],
                        help="Run only one category")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    prompts = BATCH_PROMPTS
    if args.category:
        prompts = [p for p in BATCH_PROMPTS if p.category == args.category]
        console.print(f"[dim]Filtered to category: {args.category} ({len(prompts)} prompts)[/]")

    asyncio.run(run_pipeline(prompts, dry_run=args.dry_run))
