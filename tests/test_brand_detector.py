"""Unit tests for brand_detector.py"""
import pytest
from src.analyzer.brand_detector import detect_brands, RawMention


class TestDetectBrands:
    def test_single_brand_detected(self):
        mentions = detect_brands("We recommend Asana for team projects.", ["Asana"])
        assert len(mentions) == 1
        assert mentions[0].brand == "Asana"
        assert mentions[0].position == 1

    def test_case_insensitive(self):
        mentions = detect_brands("asana is great, ASANA is popular.", ["Asana"])
        # Only first occurrence captured (seen set deduplicates)
        assert len(mentions) == 1
        assert mentions[0].brand == "Asana"

    def test_word_boundary_no_partial_match(self):
        # "Linear" should not match inside "Nonlinear"
        mentions = detect_brands("Nonlinear thinking is key.", ["Linear"])
        assert len(mentions) == 0

    def test_word_boundary_matches_standalone(self):
        mentions = detect_brands("We use Linear for sprint tracking.", ["Linear"])
        assert len(mentions) == 1

    def test_multiple_brands_position_order(self):
        text = "Jira is popular, then Asana, then Linear."
        mentions = detect_brands(text, ["Asana", "Jira", "Linear"])
        brands_in_order = [m.brand for m in mentions]
        assert brands_in_order == ["Jira", "Asana", "Linear"]
        assert [m.position for m in mentions] == [1, 2, 3]

    def test_brand_not_in_text(self):
        mentions = detect_brands("Nothing relevant here.", ["Asana"])
        assert mentions == []

    def test_empty_brand_list(self):
        mentions = detect_brands("Asana is great.", [])
        assert mentions == []

    def test_empty_text(self):
        mentions = detect_brands("", ["Asana"])
        assert mentions == []

    def test_snippet_contains_brand(self):
        text = "The best tool is Asana for managing projects across teams."
        mentions = detect_brands(text, ["Asana"])
        assert "Asana" in mentions[0].snippet

    def test_snippet_length_bounded(self):
        # snippet = ±100 chars around match
        prefix = "x" * 200
        text = f"{prefix} Asana is here."
        mentions = detect_brands(text, ["Asana"])
        assert len(mentions[0].snippet) <= 215  # brand(5) + 100 each side

    def test_brand_with_special_regex_chars(self):
        # Monday.com has a dot — re.escape should handle it
        mentions = detect_brands("We use Monday.com daily.", ["Monday.com"])
        assert len(mentions) == 1
        assert mentions[0].brand == "Monday.com"

    def test_dot_in_brand_no_wildcard_match(self):
        # "MondayXcom" should NOT match "Monday.com"
        mentions = detect_brands("MondayXcom is a fake tool.", ["Monday.com"])
        assert len(mentions) == 0

    def test_char_offset_correct(self):
        text = "Start. Then Asana appears."
        mentions = detect_brands(text, ["Asana"])
        assert text[mentions[0].char_offset:mentions[0].char_offset + 5] == "Asana"
