"""Detects brand mentions in LLM responses: position, context snippet."""
import re
import spacy
from dataclasses import dataclass

_nlp = None


def _get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
    return _nlp


@dataclass
class RawMention:
    brand: str
    position: int       # 1-based order of first mention
    char_offset: int    # character position in response
    snippet: str        # ±100 chars around mention


def detect_brands(response_text: str, target_brands: list[str]) -> list[RawMention]:
    """
    Find all target_brands in response_text.
    Returns list sorted by first appearance (position 1 = mentioned first).
    Uses case-insensitive whole-word matching to avoid partial hits
    (e.g. 'Linear' not matching 'Nonlinear').
    """
    mentions: list[RawMention] = []
    seen: set[str] = set()

    # Build per-brand regex: word boundary, case-insensitive
    for brand in target_brands:
        pattern = re.compile(r'\b' + re.escape(brand) + r'\b', re.IGNORECASE)
        match = pattern.search(response_text)
        if match and brand.lower() not in seen:
            seen.add(brand.lower())
            start = match.start()
            snippet = response_text[max(0, start - 100): start + len(brand) + 100]
            mentions.append(RawMention(
                brand=brand,
                position=0,        # filled below after sorting
                char_offset=start,
                snippet=snippet.strip(),
            ))

    # Sort by appearance order, assign 1-based positions
    mentions.sort(key=lambda m: m.char_offset)
    for i, m in enumerate(mentions):
        m.position = i + 1

    return mentions


def extract_entities(response_text: str) -> list[str]:
    """Use spaCy NER to find ORG entities not in target list (competitor discovery)."""
    doc = _get_nlp()(response_text)
    return list({ent.text for ent in doc.ents if ent.label_ == "ORG"})
