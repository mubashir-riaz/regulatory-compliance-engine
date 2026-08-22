"""
Unit Tests for Regulatory Text Chunker (Phase 2, Step 3.3).
"""

from app.services.text_chunker import chunk_regulatory_text


def test_chunk_empty_text():
    assert chunk_regulatory_text("") == []
    assert chunk_regulatory_text("   \n\t  ") == []


def test_chunk_small_text_single_chunk():
    text = "Article 1. This is a short regulatory text."
    chunks = chunk_regulatory_text(text, chunk_size=3000)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_chunk_multiple_paragraphs():
    paragraphs = [
        f"Article {i}. The entity shall implement control {i} to protect information assets."
        for i in range(1, 20)
    ]
    full_text = "\n\n".join(paragraphs)
    chunks = chunk_regulatory_text(full_text, chunk_size=200, chunk_overlap=50)

    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) > 0
        assert "Article" in chunk


def test_chunk_overlap_preservation():
    p1 = "Paragraph 1: Policy requirements regarding user access control and authentication."
    p2 = "Paragraph 2: Policy requirements regarding password complexity and rotation."
    p3 = "Paragraph 3: Policy requirements regarding multi-factor authentication enforcement."

    full_text = f"{p1}\n\n{p2}\n\n{p3}"
    chunks = chunk_regulatory_text(full_text, chunk_size=120, chunk_overlap=60)

    assert len(chunks) >= 2
