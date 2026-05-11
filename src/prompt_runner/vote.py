"""Majority voting over N LLM responses for stable brand detection."""
from collections import defaultdict
from dataclasses import dataclass
from math import ceil
from statistics import median

from src.utils.models import LLMResponse, PromptConfig
from src.analyzer.brand_detector import detect_brands, RawMention


@dataclass
class VotedMention:
    brand: str
    position: int          # median position re-ranked after voting
    char_offset: int       # from the first run where brand appeared
    snippet: str           # from the first run where brand appeared
    appearance_rate: float # fraction of runs that mentioned this brand (e.g. 0.67)


def majority_vote(
    responses: list[LLMResponse],
    prompt_configs: list[PromptConfig],
    n_runs: int,
    threshold: float = 0.5,
) -> list[tuple[LLMResponse, list[VotedMention]]]:
    """
    Aggregate N runs per (prompt_id, provider) into one stable brand detection result.

    A brand is kept only if it appears in more than `threshold` fraction of runs
    (default: >50%, so 2/3 for n_runs=3).

    Returns one (representative_response, voted_mentions) per (prompt_id, provider).
    The representative response is the middle trial — used downstream for
    sentiment judging and citation extraction.
    """
    prompt_map = {p.prompt_id: p for p in prompt_configs}
    min_count = ceil(n_runs * threshold)

    # Group valid responses by (prompt_id, provider)
    groups: dict[tuple, list[LLMResponse]] = defaultdict(list)
    for r in responses:
        if not r.error and r.response_text:
            groups[(r.prompt_id, r.provider)].append(r)

    result: list[tuple[LLMResponse, list[VotedMention]]] = []

    for (prompt_id, provider), group in groups.items():
        target_brands = prompt_map.get(
            prompt_id, PromptConfig(prompt_id="", prompt_text="", category="", target_brands=[])
        ).target_brands

        # Detect brands in every trial response for this (prompt, provider)
        # brand → [(trial_index, RawMention)]
        brand_runs: dict[str, list[tuple[int, RawMention]]] = defaultdict(list)
        for i, resp in enumerate(group):
            for m in detect_brands(resp.response_text, target_brands):
                brand_runs[m.brand].append((i, m))

        # Keep brands that clear the majority threshold
        voted: list[VotedMention] = []
        for brand, run_mentions in brand_runs.items():
            if len(run_mentions) >= min_count:
                positions = [m.position for _, m in run_mentions]
                _, first = run_mentions[0]
                voted.append(VotedMention(
                    brand=brand,
                    position=round(median(positions)),
                    char_offset=first.char_offset,
                    snippet=first.snippet,
                    appearance_rate=len(run_mentions) / len(group),
                ))

        # Re-sort and assign final 1-based positions
        voted.sort(key=lambda m: m.position)
        voted = [
            VotedMention(
                brand=m.brand, position=i + 1,
                char_offset=m.char_offset, snippet=m.snippet,
                appearance_rate=m.appearance_rate,
            )
            for i, m in enumerate(voted)
        ]

        # Representative response: middle trial (avoids first/last bias)
        rep = group[len(group) // 2]
        result.append((rep, voted))

    return result
