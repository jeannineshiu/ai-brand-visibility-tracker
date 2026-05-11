"""Unit tests for citation_extractor.py"""
import pytest
from src.analyzer.citation_extractor import extract_citations, classify_domain


class TestClassifyDomain:
    def test_reddit(self):
        assert classify_domain("reddit.com") == "community"

    def test_stackoverflow(self):
        assert classify_domain("stackoverflow.com") == "developer"

    def test_g2(self):
        assert classify_domain("g2.com") == "review_site"

    def test_capterra(self):
        assert classify_domain("capterra.com") == "review_site"

    def test_techcrunch(self):
        assert classify_domain("techcrunch.com") == "tech_media"

    def test_forbes(self):
        assert classify_domain("forbes.com") == "business_media"

    def test_medium(self):
        assert classify_domain("medium.com") == "blog_platform"

    def test_youtube(self):
        assert classify_domain("youtube.com") == "video"

    def test_wikipedia(self):
        assert classify_domain("wikipedia.org") == "reference"

    def test_linkedin(self):
        assert classify_domain("linkedin.com") == "professional_social"

    def test_twitter(self):
        assert classify_domain("twitter.com") == "social"

    def test_brand_owned_two_segment(self):
        # asana.com has 2 parts after stripping www → brand_owned
        assert classify_domain("asana.com") == "brand_owned"

    def test_brand_owned_with_www(self):
        assert classify_domain("www.notion.so") == "brand_owned"

    def test_other_multi_segment(self):
        # some.unknown.site.xyz has 4 parts → other
        assert classify_domain("some.unknown.site.xyz") == "other"


class TestExtractCitations:
    def test_single_url(self):
        text = "See https://www.g2.com/categories/project-management for reviews."
        citations = extract_citations(text)
        assert len(citations) == 1
        assert citations[0].domain_type == "review_site"

    def test_multiple_urls(self):
        text = (
            "Sources: https://www.reddit.com/r/pm and "
            "https://techcrunch.com/2024/asana-review"
        )
        citations = extract_citations(text)
        assert len(citations) == 2
        types = {c.domain_type for c in citations}
        assert types == {"community", "tech_media"}

    def test_deduplication(self):
        url = "https://www.g2.com/categories/project-management"
        text = f"See {url} and also {url}"
        citations = extract_citations(text)
        assert len(citations) == 1

    def test_trailing_punctuation_stripped(self):
        text = "See https://www.g2.com/pm."
        citations = extract_citations(text)
        assert not citations[0].url.endswith(".")

    def test_trailing_parenthesis_stripped(self):
        text = "Source (https://capterra.com/pm)"
        citations = extract_citations(text)
        assert not citations[0].url.endswith(")")

    def test_no_urls(self):
        citations = extract_citations("No links here at all.")
        assert citations == []

    def test_domain_extracted_correctly(self):
        text = "Visit https://www.reddit.com/r/projectmanagement"
        citations = extract_citations(text)
        assert citations[0].domain == "reddit.com"

    def test_http_and_https(self):
        text = "http://stackoverflow.com/q/123 and https://dev.to/post"
        citations = extract_citations(text)
        assert len(citations) == 2
