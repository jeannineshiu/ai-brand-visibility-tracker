"""Orchestrates brand detection + sentiment + citation extraction for a batch of LLM responses."""
from rich.console import Console
from rich.table import Table

from src.utils.models import LLMResponse, BrandMention, CitationSource, PromptConfig
from src.analyzer.brand_detector import detect_brands
from src.analyzer.sentiment_judge import judge_batch
from src.analyzer.citation_extractor import extract_citations

console = Console()


async def analyze_responses(
    responses: list[LLMResponse],
    prompt_configs: list[PromptConfig],
) -> tuple[list[BrandMention], list[CitationSource]]:
    """
    Full analysis pipeline for a list of LLM responses.
    Returns (brand_mentions, citation_sources).
    """
    prompt_map = {p.prompt_id: p for p in prompt_configs}
    all_mentions: list[BrandMention] = []
    all_citations: list[CitationSource] = []

    # --- Phase 1: Brand detection + Citation extraction (CPU, fast) ---
    raw_mentions_per_response: list[tuple] = []
    for resp in responses:
        if resp.error or not resp.response_text:
            raw_mentions_per_response.append((resp, []))
            continue

        target_brands = prompt_map.get(resp.prompt_id, PromptConfig(
            prompt_id="", prompt_text="", category="", target_brands=[]
        )).target_brands

        raw = detect_brands(resp.response_text, target_brands)
        raw_mentions_per_response.append((resp, raw))

        # Citations — no async needed
        for cite in extract_citations(resp.response_text):
            all_citations.append(CitationSource(
                run_id=resp.run_id,
                prompt_id=resp.prompt_id,
                provider=resp.provider,
                url=cite.url,
                domain=cite.domain,
                domain_type=cite.domain_type,
            ))

    # --- Phase 2: Sentiment (async LLM-as-judge, batch all at once) ---
    judge_inputs: list[tuple[str, str]] = []   # (brand, snippet)
    judge_meta: list[tuple[LLMResponse, object]] = []   # parallel tracking

    for resp, raw_list in raw_mentions_per_response:
        if not raw_list:
            continue
        for raw in raw_list:
            judge_inputs.append((raw.brand, raw.snippet))
            judge_meta.append((resp, raw))

    sentiments = await judge_batch(judge_inputs) if judge_inputs else []

    for (resp, raw), (sentiment, _reason) in zip(judge_meta, sentiments):
        all_mentions.append(BrandMention(
            run_id=resp.run_id,
            prompt_id=resp.prompt_id,
            provider=resp.provider,
            brand=raw.brand,
            position=raw.position,
            sentiment=sentiment,
            snippet=raw.snippet,
        ))

    _print_analysis_summary(all_mentions, all_citations)
    return all_mentions, all_citations


def _print_analysis_summary(mentions: list[BrandMention], citations: list[CitationSource]):
    # Brand mentions table
    table = Table(title="Brand Mentions")
    table.add_column("Brand", style="bold")
    table.add_column("Provider", style="cyan")
    table.add_column("Prompt")
    table.add_column("Position", justify="right")
    table.add_column("Sentiment")

    sentiment_style = {"positive": "green", "neutral": "yellow", "negative": "red"}
    for m in sorted(mentions, key=lambda x: (x.brand, x.provider.value)):
        style = sentiment_style.get(m.sentiment.value, "white")
        table.add_row(
            m.brand, m.provider.value, m.prompt_id,
            str(m.position),
            f"[{style}]{m.sentiment.value}[/]",
        )
    console.print(table)

    # Citation table
    ctable = Table(title="Citation Sources")
    ctable.add_column("Domain")
    ctable.add_column("Type", style="cyan")
    ctable.add_column("Provider")
    for c in citations:
        ctable.add_row(c.domain or c.url[:40], c.domain_type, c.provider.value)
    console.print(ctable)
