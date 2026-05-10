"""Extracts and classifies URLs cited in LLM responses."""
import re
from urllib.parse import urlparse
from dataclasses import dataclass

URL_PATTERN = re.compile(
    r'https?://[^\s\)\]\,\"\'<>]+',
    re.IGNORECASE,
)

# Domain classification rules (checked in order, first match wins)
DOMAIN_TYPE_RULES: list[tuple[re.Pattern, str]] = [
    (re.compile(r'reddit\.com', re.I),                    "community"),
    (re.compile(r'stackoverflow\.com|dev\.to|hackernews', re.I), "developer"),
    (re.compile(r'g2\.com|capterra\.com|trustradius\.com|getapp\.com', re.I), "review_site"),
    (re.compile(r'techcrunch|wired|theverge|zdnet|techradar|infoworld', re.I), "tech_media"),
    (re.compile(r'forbes|businessinsider|entrepreneur|inc\.com', re.I), "business_media"),
    (re.compile(r'medium\.com|substack\.com|ghost\.io', re.I), "blog_platform"),
    (re.compile(r'youtube\.com|youtu\.be', re.I),         "video"),
    (re.compile(r'wikipedia\.org', re.I),                 "reference"),
    (re.compile(r'linkedin\.com', re.I),                  "professional_social"),
    (re.compile(r'twitter\.com|x\.com', re.I),            "social"),
]


@dataclass
class ExtractedCitation:
    url: str
    domain: str
    domain_type: str   # one of the categories above, or "brand_owned" / "other"


def classify_domain(domain: str) -> str:
    for pattern, dtype in DOMAIN_TYPE_RULES:
        if pattern.search(domain):
            return dtype
    # Simple heuristic: single-segment domain (e.g. asana.com) → likely brand owned
    parts = domain.lstrip("www.").split(".")
    if len(parts) == 2:
        return "brand_owned"
    return "other"


def extract_citations(response_text: str) -> list[ExtractedCitation]:
    """Find all URLs in response_text and classify their domain type."""
    seen: set[str] = set()
    citations: list[ExtractedCitation] = []

    for match in URL_PATTERN.finditer(response_text):
        url = match.group().rstrip(".,;)")   # strip trailing punctuation
        if url in seen:
            continue
        seen.add(url)

        parsed = urlparse(url)
        domain = parsed.netloc.lower().lstrip("www.")
        citations.append(ExtractedCitation(
            url=url,
            domain=domain,
            domain_type=classify_domain(domain),
        ))

    return citations
