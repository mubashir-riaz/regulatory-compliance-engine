"""
Text Chunking Utilities for Regulatory and Compliance Documents.

Provides intelligent paragraph- and section-aware text chunking to ensure
coherent chunks are sent to LLMs without cutting across clauses mid-sentence.
"""

import re
from typing import List


def chunk_regulatory_text(
    text: str,
    chunk_size: int = 3000,
    chunk_overlap: int = 200,
) -> List[str]:
    """
    Split a large regulatory text document into coherent chunks for LLM processing.

    :param text: Full raw or extracted text of the regulatory document
    :param chunk_size: Target maximum character size per chunk (default: 3000)
    :param chunk_overlap: Character overlap between consecutive chunks (default: 200)
    :return: List of text chunk strings
    """
    if not text or not text.strip():
        return []

    cleaned_text = text.strip()
    if len(cleaned_text) <= chunk_size:
        return [cleaned_text]

    # Split primarily along natural paragraph or section breaks (double newlines or section headers)
    raw_paragraphs = re.split(r"\n\s*\n", cleaned_text)
    paragraphs: List[str] = []
    for p in raw_paragraphs:
        p_str = p.strip()
        if not p_str:
            continue
        # If a single paragraph is excessively large, split it by line or sentence
        if len(p_str) > chunk_size:
            sub_lines = p_str.split("\n")
            for sub in sub_lines:
                sub_str = sub.strip()
                if len(sub_str) > chunk_size:
                    # Split sentences by period followed by space/capital letter
                    sentences = re.split(r"(?<=[.?!])\s+", sub_str)
                    paragraphs.extend([s.strip() for s in sentences if s.strip()])
                elif sub_str:
                    paragraphs.append(sub_str)
        else:
            paragraphs.append(p_str)

    chunks: List[str] = []
    current_chunk_parts: List[str] = []
    current_length = 0

    for paragraph in paragraphs:
        p_len = len(paragraph)

        if current_length + p_len + 2 > chunk_size and current_chunk_parts:
            # Finalize current chunk
            chunk_content = "\n\n".join(current_chunk_parts).strip()
            chunks.append(chunk_content)

            # Build overlap from recent paragraphs if possible
            overlap_parts: List[str] = []
            overlap_len = 0
            for part in reversed(current_chunk_parts):
                if overlap_len + len(part) <= chunk_overlap:
                    overlap_parts.insert(0, part)
                    overlap_len += len(part) + 2
                else:
                    break

            current_chunk_parts = overlap_parts + [paragraph]
            current_length = sum(len(p) + 2 for p in current_chunk_parts)
        else:
            current_chunk_parts.append(paragraph)
            current_length += p_len + 2

    if current_chunk_parts:
        final_chunk = "\n\n".join(current_chunk_parts).strip()
        if not chunks or final_chunk != chunks[-1]:
            chunks.append(final_chunk)

    return chunks
