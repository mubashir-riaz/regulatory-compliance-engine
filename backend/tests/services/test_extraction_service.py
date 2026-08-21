"""
Unit and Integration Tests for Regulatory Obligation Extraction Service (Phase 2, Step 3.2).
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from app.schemas.extraction import ExtractedObligation
from app.services.extraction_service import (
    DEFAULT_PROMPT_PATH,
    SAMPLE_GDPR_ARTICLE_5_TEXT,
    ExtractionService,
    InvalidResponseError,
    LLMProviderError,
)

# Sample parsed output for GDPR Article 5
MOCK_GDPR_ARTICLE_5_OBLIGATIONS = [
    {
        "clause": "Article 5(1)(a)",
        "text": "Personal data shall be processed lawfully, fairly and in a transparent manner in relation to the data subject.",
        "category": "Lawfulness, Fairness & Transparency",
        "mandatory": True,
        "keywords": ["personal data", "lawfulness", "fairness", "transparency"],
    },
    {
        "clause": "Article 5(1)(b)",
        "text": "Personal data shall be collected for specified, explicit and legitimate purposes and not further processed incompatibly.",
        "category": "Purpose Limitation",
        "mandatory": True,
        "keywords": ["purpose limitation", "specified purposes", "legitimate purposes"],
    },
    {
        "clause": "Article 5(1)(c)",
        "text": "Personal data shall be adequate, relevant and limited to what is necessary in relation to processing purposes.",
        "category": "Data Minimisation",
        "mandatory": True,
        "keywords": ["data minimisation", "adequate", "relevant", "necessary"],
    },
    {
        "clause": "Article 5(1)(d)",
        "text": "Personal data shall be accurate and, where necessary, kept up to date with reasonable steps taken to rectify inaccuracies.",
        "category": "Accuracy",
        "mandatory": True,
        "keywords": ["accuracy", "rectification", "erasure", "data quality"],
    },
    {
        "clause": "Article 5(1)(e)",
        "text": "Personal data shall be kept in a form which permits identification of data subjects for no longer than is necessary.",
        "category": "Storage Limitation",
        "mandatory": True,
        "keywords": ["storage limitation", "retention", "data subjects"],
    },
    {
        "clause": "Article 5(1)(f)",
        "text": "Personal data shall be processed in a manner that ensures appropriate security using technical and organisational measures.",
        "category": "Integrity and Confidentiality",
        "mandatory": True,
        "keywords": ["security", "integrity", "confidentiality", "technical measures"],
    },
    {
        "clause": "Article 5(2)",
        "text": "The controller shall be responsible for, and be able to demonstrate compliance with, the principles in paragraph 1.",
        "category": "Accountability & Governance",
        "mandatory": True,
        "keywords": ["accountability", "demonstrate compliance", "controller responsibility"],
    },
]


def test_load_prompt_template():
    """Test that extraction prompt template loads from file correctly."""
    service = ExtractionService()
    template = service.load_prompt_template()
    assert template is not None
    assert "{text}" in template
    assert "Few-Shot Examples" in template
    assert "clause" in template
    assert "mandatory" in template


def test_format_prompt():
    """Test prompt formatting with sample regulatory text."""
    service = ExtractionService()
    formatted = service.format_prompt(SAMPLE_GDPR_ARTICLE_5_TEXT)
    assert "Article 5 - Principles relating to processing of personal data" in formatted
    assert "{text}" not in formatted


def test_parse_clean_json_array():
    """Test parsing a clean, raw JSON array of obligations."""
    service = ExtractionService()
    json_str = json.dumps(MOCK_GDPR_ARTICLE_5_OBLIGATIONS)
    obligations = service.parse_and_validate_response(json_str)

    assert len(obligations) == 7
    assert isinstance(obligations[0], ExtractedObligation)
    assert obligations[0].clause == "Article 5(1)(a)"
    assert obligations[0].mandatory is True
    assert "transparency" in obligations[0].keywords


def test_parse_markdown_code_fences():
    """Test parsing LLM response wrapped in markdown code fences."""
    service = ExtractionService()
    json_str = json.dumps(MOCK_GDPR_ARTICLE_5_OBLIGATIONS[:2])
    fenced_str = f"```json\n{json_str}\n```"

    obligations = service.parse_and_validate_response(fenced_str)
    assert len(obligations) == 2
    assert obligations[0].clause == "Article 5(1)(a)"
    assert obligations[1].clause == "Article 5(1)(b)"


def test_parse_dict_wrapper_response():
    """Test parsing LLM response wrapped in a root dictionary (e.g. {'obligations': [...]})."""
    service = ExtractionService()
    wrapped_data = {"obligations": MOCK_GDPR_ARTICLE_5_OBLIGATIONS[:3]}
    json_str = json.dumps(wrapped_data)

    obligations = service.parse_and_validate_response(json_str)
    assert len(obligations) == 3
    assert obligations[0].clause == "Article 5(1)(a)"
    assert obligations[2].clause == "Article 5(1)(c)"


def test_parse_single_obligation_dict():
    """Test parsing LLM response returning a single obligation dict."""
    service = ExtractionService()
    single_data = MOCK_GDPR_ARTICLE_5_OBLIGATIONS[0]
    json_str = json.dumps(single_data)

    obligations = service.parse_and_validate_response(json_str)
    assert len(obligations) == 1
    assert obligations[0].clause == "Article 5(1)(a)"


def test_parse_with_surrounding_text_fallback():
    """Test regex extraction fallback when LLM outputs surrounding text."""
    service = ExtractionService()
    json_str = json.dumps(MOCK_GDPR_ARTICLE_5_OBLIGATIONS[:2])
    noisy_str = f"Here are the extracted obligations from the text chunk:\n{json_str}\nHope this helps!"

    obligations = service.parse_and_validate_response(noisy_str)
    assert len(obligations) == 2
    assert obligations[0].clause == "Article 5(1)(a)"


def test_parse_malformed_json_raises_error():
    """Test that completely unparseable text raises InvalidResponseError."""
    service = ExtractionService()
    with pytest.raises(InvalidResponseError):
        service.parse_and_validate_response("This is not JSON at all, no brackets or braces.")


def test_extracted_obligation_schema_type_coercion():
    """Test schema type coercion for strings, boolean strings, and keyword comma-strings."""
    raw_item = {
        "clause": "  CC6.1  ",
        "text": "  Implement access control.  ",
        "category": "Access Control",
        "mandatory": "true",
        "keywords": "access, security, firewall",
    }
    obligation = ExtractedObligation.model_validate(raw_item)
    assert obligation.clause == "CC6.1"
    assert obligation.text == "Implement access control."
    assert obligation.mandatory is True
    assert obligation.keywords == ["access", "security", "firewall"]


@pytest.mark.asyncio
async def test_extract_obligations_empty_input():
    """Test that empty text chunk returns an empty list without making LLM calls."""
    service = ExtractionService()
    result = await service.extract_obligations("")
    assert result == []

    result_whitespace = await service.extract_obligations("   \n\t  ")
    assert result_whitespace == []


@pytest.mark.asyncio
async def test_extract_obligations_groq_mocked():
    """Test full extraction flow with mocked Groq SDK / API call."""
    service = ExtractionService(
        provider="groq",
        groq_api_key="mock-groq-key",
    )

    mock_llm_response = json.dumps(MOCK_GDPR_ARTICLE_5_OBLIGATIONS)
    with patch.object(service, "_call_groq", new_callable=AsyncMock) as mock_call:
        mock_call.return_value = mock_llm_response

        obligations = await service.extract_obligations(SAMPLE_GDPR_ARTICLE_5_TEXT)

        assert len(obligations) == 7
        assert obligations[0].clause == "Article 5(1)(a)"
        assert obligations[5].clause == "Article 5(1)(f)"
        assert obligations[6].clause == "Article 5(2)"
        mock_call.assert_called_once()


@pytest.mark.asyncio
async def test_extract_obligations_gemini_mocked():
    """Test full extraction flow with mocked Gemini REST call."""
    service = ExtractionService(
        provider="gemini",
        gemini_api_key="mock-gemini-key",
    )

    mock_llm_response = json.dumps(MOCK_GDPR_ARTICLE_5_OBLIGATIONS[:4])
    with patch.object(service, "_call_gemini", new_callable=AsyncMock) as mock_call:
        mock_call.return_value = mock_llm_response

        obligations = await service.extract_obligations(SAMPLE_GDPR_ARTICLE_5_TEXT, provider="gemini")

        assert len(obligations) == 4
        assert obligations[0].clause == "Article 5(1)(a)"
        assert obligations[3].clause == "Article 5(1)(d)"
        mock_call.assert_called_once()


@pytest.mark.asyncio
async def test_missing_api_key_raises_llm_provider_error():
    """Test that calling provider without API key raises LLMProviderError."""
    service_groq = ExtractionService(provider="groq", groq_api_key="")
    with pytest.raises(LLMProviderError) as exc_groq:
        await service_groq._call_groq("test prompt")
    assert "GROQ_API_KEY" in str(exc_groq.value)

    service_gemini = ExtractionService(provider="gemini", gemini_api_key="")
    with pytest.raises(LLMProviderError) as exc_gemini:
        await service_gemini._call_gemini("test prompt")
    assert "GEMINI_API_KEY" in str(exc_gemini.value)
