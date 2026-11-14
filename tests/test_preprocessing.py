from src.preprocessing import clean_text


def test_clean_text_lowercases_and_collapses_whitespace():
    assert clean_text("  BREAKING   News  ") == "breaking news"


def test_clean_text_removes_html_and_normalizes_url():
    result = clean_text("<p>Read https://example.com/story now!</p>")
    assert "<" not in result and "example" not in result
    assert "url" in result


def test_clean_text_handles_missing_values():
    assert clean_text(None) == ""


def test_clean_text_preserves_negation_words():
    assert "not" in clean_text("This is not verified.")
