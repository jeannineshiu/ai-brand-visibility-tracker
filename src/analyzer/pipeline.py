"""Orchestrates brand detection + sentiment + citation extraction for a batch of LLM responses."""
from rich.console import Console
from rich.table import Table

from src.utils.models import LLMResponse, BrandMention, CitationSource, PromptConfig
from src.analyzer.brand_detector import detect_brands, extract_entities
from src.analyzer.sentiment_judge import judge_batch
from src.analyzer.citation_extractor import extract_citations
from src.analyzer.vote import VotedMention

console = Console()


async def analyze_responses(
    responses: list[LLMResponse],
    prompt_configs: list[PromptConfig],
) -> tuple[list[BrandMention], list[CitationSource], list[str]]:
    """
    Full analysis pipeline for a list of LLM responses.
    Returns (brand_mentions, citation_sources, discovered_competitors).
    discovered_competitors: ORG entities found by spaCy not in any target brand list.
    """
    prompt_map = {p.prompt_id: p for p in prompt_configs}
    all_known = {b.lower() for p in prompt_configs for b in p.target_brands}
    all_mentions: list[BrandMention] = []
    all_citations: list[CitationSource] = []
    discovered: set[str] = set()

    # --- Phase 1: Brand detection + Citation extraction + Entity discovery ---
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

        for cite in extract_citations(resp.response_text):
            all_citations.append(CitationSource(
                run_id=resp.run_id,
                prompt_id=resp.prompt_id,
                provider=resp.provider,
                url=cite.url,
                domain=cite.domain,
                domain_type=cite.domain_type,
            ))

        for ent in extract_entities(resp.response_text):
            if ent.lower() not in all_known:
                discovered.add(ent)

    # --- Phase 2: Sentiment (async LLM-as-judge, batch all at once) ---
    judge_inputs: list[tuple[str, str]] = []
    judge_meta: list[tuple[LLMResponse, object]] = []

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

    discovered_list = sorted(discovered)
    _print_analysis_summary(all_mentions, all_citations, discovered_list)
    return all_mentions, all_citations, discovered_list


async def analyze_voted_responses(
    voted: list[tuple[LLMResponse, list[VotedMention]]],
    all_responses: list[LLMResponse],
) -> tuple[list[BrandMention], list[CitationSource], list[str]]:
    """
    Pipeline variant for majority-voted brand mentions.

    - voted: output of majority_vote() — one (rep_response, voted_mentions) per (prompt, provider)
    - all_responses: every response across all N trials, used for citation aggregation
    """
    all_mentions: list[BrandMention] = []
    all_citations: list[CitationSource] = []
    discovered: set[str] = set()
    voted_brands = {vm.brand.lower() for _, vms in voted for vm in vms}

    # Citations + entity discovery from ALL trial responses
    seen_citations: set[tuple] = set()
    for resp in all_responses:
        if resp.error or not resp.response_text:
            continue
        for cite in extract_citations(resp.response_text):
            key = (resp.prompt_id, resp.provider, cite.url)
            if key not in seen_citations:
                seen_citations.add(key)
                all_citations.append(CitationSource(
                    run_id=resp.run_id,
                    prompt_id=resp.prompt_id,
                    provider=resp.provider,
                    url=cite.url,
                    domain=cite.domain,
                    domain_type=cite.domain_type,
                ))
        for ent in extract_entities(resp.response_text):
            if ent.lower() not in voted_brands:
                discovered.add(ent)

    # Sentiment judging on voted brands only
    judge_inputs: list[tuple[str, str]] = []
    judge_meta: list[tuple[LLMResponse, VotedMention]] = []

    for rep_resp, voted_mentions in voted:
        for vm in voted_mentions:
            judge_inputs.append((vm.brand, vm.snippet))
            judge_meta.append((rep_resp, vm))

    sentiments = await judge_batch(judge_inputs) if judge_inputs else []

    for (resp, vm), (sentiment, _reason) in zip(judge_meta, sentiments):
        all_mentions.append(BrandMention(
            run_id=resp.run_id,
            prompt_id=resp.prompt_id,
            provider=resp.provider,
            brand=vm.brand,
            position=vm.position,
            sentiment=sentiment,
            snippet=vm.snippet,
        ))

    discovered_list = sorted(discovered)
    _print_analysis_summary(all_mentions, all_citations, discovered_list)
    return all_mentions, all_citations, discovered_list


def _print_analysis_summary(
    mentions: list[BrandMention],
    citations: list[CitationSource],
    discovered: list[str] | None = None,
):
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

    if discovered:
        dtable = Table(title="Discovered Competitors (not in target list)")
        dtable.add_column("Entity", style="magenta")
        for ent in discovered:
            dtable.add_row(ent)
        console.print(dtable)
