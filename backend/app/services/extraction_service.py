"""
Regulatory Obligation Extraction Service.

Extracts discrete regulatory obligations, compliance requirements, and control statements
from regulatory text chunks using Groq or Gemini LLMs.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import httpx

from app.core.config import settings
from app.schemas.extraction import ExtractedObligation

logger = logging.getLogger(__name__)

# Default prompt path
DEFAULT_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "extract_obligations.txt"

# Default models
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"
DEFAULT_GEMINI_MODEL = "gemini-1.5-flash"

# Sample text for validation and testing (GDPR Article 5)
SAMPLE_GDPR_ARTICLE_5_TEXT = """
Article 5 - Principles relating to processing of personal data
1. Personal data shall be:
(a) processed lawfully, fairly and in a transparent manner in relation to the data subject ('lawfulness, fairness and transparency');
(b) collected for specified, explicit and legitimate purposes and not further processed in a manner that is incompatible with those purposes; further processing for archiving purposes in the public interest, scientific or historical research purposes or statistical purposes shall, in accordance with Article 89(1), not be considered to be incompatible with the initial purposes ('purpose limitation');
(c) adequate, relevant and limited to what is necessary in relation to the purposes for which they are processed ('data minimisation');
(d) accurate and, where necessary, kept up to date; every reasonable step must be taken to ensure that personal data that are inaccurate, having regard to the purposes for which they are processed, are erased or rectified without delay ('accuracy');
(e) kept in a form which permits identification of data subjects for no longer than is necessary for the purposes for which the personal data are processed; personal data may be stored for longer periods insofar as the personal data will be processed solely for archiving purposes in the public interest, scientific or historical research purposes or statistical purposes in accordance with Article 89(1) subject to implementation of the appropriate technical and organisational measures required by this Regulation in order to safeguard the rights and freedoms of the data subject ('storage limitation');
(f) processed in a manner that ensures appropriate security of the personal data, including protection against unauthorised or unlawful processing and against accidental loss, destruction or damage, using appropriate technical or organisational measures ('integrity and confidentiality').
2. The controller shall be responsible for, and be able to demonstrate compliance with, paragraph 1 ('accountability').
""".strip()


class ExtractionServiceError(Exception):
    """Base exception for extraction service errors."""
    pass


class LLMProviderError(ExtractionServiceError):
    """Raised when an LLM provider request fails or authentication fails."""
    pass


class InvalidResponseError(ExtractionServiceError):
    """Raised when an LLM response cannot be parsed or validated as obligations."""
    pass


class ExtractionService:
    """
    Service for extracting structured regulatory obligations from text chunks via LLM.
    """

    def __init__(
        self,
        provider: Optional[str] = None,
        groq_api_key: Optional[str] = None,
        gemini_api_key: Optional[str] = None,
        prompt_path: Optional[Union[str, Path]] = None,
        http_client: Optional[httpx.AsyncClient] = None,
    ):
        self.provider = (provider or getattr(settings, "LLM_PROVIDER", "groq")).lower().strip()
        self.groq_api_key = (
            groq_api_key
            if groq_api_key is not None
            else getattr(settings, "GROQ_API_KEY", None)
        )
        self.gemini_api_key = (
            gemini_api_key
            if gemini_api_key is not None
            else getattr(settings, "GEMINI_API_KEY", None)
        )
        self.prompt_path = Path(prompt_path) if prompt_path else DEFAULT_PROMPT_PATH
        self._http_client = http_client
        self._prompt_template: Optional[str] = None

    def load_prompt_template(self) -> str:
        """
        Load the extraction prompt template from file.
        """
        if self._prompt_template is None:
            if not self.prompt_path.exists():
                raise FileNotFoundError(f"Extraction prompt template not found at {self.prompt_path}")
            self._prompt_template = self.prompt_path.read_text(encoding="utf-8")
        return self._prompt_template

    def format_prompt(self, text: str) -> str:
        """
        Format the extraction prompt with the provided regulatory text chunk.
        """
        template = self.load_prompt_template()
        return template.replace("{text}", text.strip())

    async def extract_obligations(
        self,
        text: str,
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ) -> List[ExtractedObligation]:
        """
        Extract structured obligations from a regulatory text chunk.

        :param text: Regulatory text chunk to extract obligations from
        :param provider: Optional provider override ("groq" or "gemini")
        :param model: Optional model identifier override
        :return: List of validated ExtractedObligation objects
        """
        if not text or not text.strip():
            logger.warning("Empty regulatory text provided to extract_obligations.")
            return []

        formatted_prompt = self.format_prompt(text)
        target_provider = (provider or self.provider).lower().strip()

        logger.info(f"Sending extraction request to LLM provider '{target_provider}'...")
        raw_response = await self._call_llm(
            prompt=formatted_prompt,
            provider=target_provider,
            model=model,
        )

        obligations = self.parse_and_validate_response(raw_response)
        logger.info(f"Successfully extracted and validated {len(obligations)} obligations.")
        return obligations

    async def _call_llm(
        self,
        prompt: str,
        provider: str,
        model: Optional[str] = None,
    ) -> str:
        """
        Dispatch prompt to the requested LLM provider.
        """
        if provider == "groq":
            return await self._call_groq(prompt=prompt, model=model)
        elif provider in ("gemini", "google"):
            return await self._call_gemini(prompt=prompt, model=model)
        else:
            raise LLMProviderError(f"Unsupported LLM provider '{provider}'. Must be 'groq' or 'gemini'.")

    async def _call_groq(self, prompt: str, model: Optional[str] = None) -> str:
        """
        Call Groq API via official SDK if available, or async HTTP request.
        """
        api_key = self.groq_api_key
        if not api_key or not api_key.strip() or api_key.startswith("your-"):
            raise LLMProviderError("GROQ_API_KEY is not configured in settings or environment.")

        target_model = model or DEFAULT_GROQ_MODEL

        # Attempt using groq SDK if installed
        try:
            from groq import AsyncGroq
            client = AsyncGroq(api_key=api_key)
            completion = await client.chat.completions.create(
                model=target_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a regulatory compliance analysis system that outputs strictly valid JSON arrays.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=4096,
            )
            content = completion.choices[0].message.content
            if not content:
                raise InvalidResponseError("Groq returned an empty response.")
            return content
        except ImportError:
            # Fallback to direct HTTP request via httpx
            logger.debug("groq SDK not found, using direct HTTP request to Groq API.")
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": target_model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a regulatory compliance analysis system that outputs strictly valid JSON arrays.",
                    },
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.1,
                "max_tokens": 4096,
            }

            async with (self._http_client or httpx.AsyncClient(timeout=60.0)) as client:
                response = await client.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers=headers,
                    json=payload,
                )
                if response.status_code != 200:
                    raise LLMProviderError(
                        f"Groq API returned HTTP {response.status_code}: {response.text}"
                    )
                data = response.json()
                content = data["choices"][0]["message"]["content"]
                if not content:
                    raise InvalidResponseError("Groq returned an empty response.")
                return content
        except Exception as e:
            if isinstance(e, (LLMProviderError, InvalidResponseError)):
                raise
            raise LLMProviderError(f"Groq API call failed: {e}") from e

    async def _call_gemini(self, prompt: str, model: Optional[str] = None) -> str:
        """
        Call Google Gemini API via REST endpoint.
        """
        api_key = self.gemini_api_key
        if not api_key or not api_key.strip() or api_key.startswith("your-"):
            raise LLMProviderError("GEMINI_API_KEY is not configured in settings or environment.")

        target_model = model or DEFAULT_GEMINI_MODEL
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{target_model}:generateContent?key={api_key}"

        payload = {
            "contents": [
                {
                    "parts": [{"text": prompt}]
                }
            ],
            "generationConfig": {
                "temperature": 0.1,
                "responseMimeType": "application/json",
            },
        }

        try:
            async with (self._http_client or httpx.AsyncClient(timeout=60.0)) as client:
                response = await client.post(url, json=payload)
                if response.status_code != 200:
                    raise LLMProviderError(
                        f"Gemini API returned HTTP {response.status_code}: {response.text}"
                    )
                data = response.json()
                candidates = data.get("candidates", [])
                if not candidates:
                    raise InvalidResponseError(f"Gemini returned no candidates: {data}")
                content = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                if not content:
                    raise InvalidResponseError("Gemini returned empty text content.")
                return content
        except Exception as e:
            if isinstance(e, (LLMProviderError, InvalidResponseError)):
                raise
            raise LLMProviderError(f"Gemini API call failed: {e}") from e

    def parse_and_validate_response(self, response_text: str) -> List[ExtractedObligation]:
        """
        Parse raw LLM response text, extract JSON payload, and validate against ExtractedObligation schema.

        :param response_text: Raw string response from LLM
        :return: List of validated ExtractedObligation models
        :raises InvalidResponseError: If response contains invalid or unparseable JSON
        """
        if not response_text or not response_text.strip():
            logger.warning("Received empty LLM response text.")
            return []

        cleaned_text = self._clean_json_text(response_text)

        try:
            parsed_data = json.loads(cleaned_text)
        except json.JSONDecodeError as exc:
            # Fallback: attempt regex extraction of JSON array or object
            logger.debug(f"Direct JSON parsing failed ({exc}), attempting regex fallback extraction...")
            extracted_block = self._extract_json_substring(cleaned_text)
            if extracted_block:
                try:
                    parsed_data = json.loads(extracted_block)
                except json.JSONDecodeError as inner_exc:
                    raise InvalidResponseError(
                        f"Failed to parse LLM response as JSON: {inner_exc}. Raw text:\n{response_text[:300]}"
                    ) from inner_exc
            else:
                raise InvalidResponseError(
                    f"Failed to parse LLM response as JSON: {exc}. Raw text:\n{response_text[:300]}"
                ) from exc

        # Handle object wrapper variations (e.g. {"obligations": [...]} or {"items": [...]})
        items_list: List[Any] = []
        if isinstance(parsed_data, list):
            items_list = parsed_data
        elif isinstance(parsed_data, dict):
            for key in ("obligations", "requirements", "items", "data", "results", "rules"):
                if key in parsed_data and isinstance(parsed_data[key], list):
                    items_list = parsed_data[key]
                    break
            else:
                # Check if the dict itself is a single obligation object
                if "clause" in parsed_data and "text" in parsed_data:
                    items_list = [parsed_data]
                else:
                    raise InvalidResponseError(
                        f"Parsed JSON object does not contain an obligation list: {list(parsed_data.keys())}"
                    )
        else:
            raise InvalidResponseError(f"Unexpected JSON root type: {type(parsed_data).__name__}")

        # Validate each item against Pydantic schema
        validated_obligations: List[ExtractedObligation] = []
        for idx, item in enumerate(items_list):
            if not isinstance(item, dict):
                logger.warning(f"Skipping non-dict item at index {idx}: {item}")
                continue
            try:
                obligation = ExtractedObligation.model_validate(item)
                validated_obligations.append(obligation)
            except Exception as val_err:
                logger.warning(f"Validation error for item {idx} ({item.get('clause', 'unknown')}): {val_err}")

        return validated_obligations

    @staticmethod
    def _clean_json_text(text: str) -> str:
        """
        Clean markdown code fences (```json ... ```) and leading/trailing whitespace.
        """
        text = text.strip()
        # Remove markdown code blocks if present
        if text.startswith("```"):
            lines = text.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        return text

    @staticmethod
    def _extract_json_substring(text: str) -> Optional[str]:
        """
        Extract bracketed JSON substring ([ ... ] or { ... }) using regex search.
        """
        # Try array first
        array_match = re.search(r"\[\s*\{.*\}\s*\]", text, re.DOTALL)
        if array_match:
            return array_match.group(0)

        # Try object next
        obj_match = re.search(r"\{.*\}", text, re.DOTALL)
        if obj_match:
            return obj_match.group(0)

        return None


# Global singleton instance
extraction_service = ExtractionService()


async def run_sample_gdpr_article_5_demo() -> List[ExtractedObligation]:
    """
    Convenience demo function extracting obligations from sample GDPR Article 5 text.
    """
    service = ExtractionService()
    return await service.extract_obligations(SAMPLE_GDPR_ARTICLE_5_TEXT)
