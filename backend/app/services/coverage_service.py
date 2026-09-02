"""
Regulatory Compliance Coverage Assessment Service (Phase 2, Step 5.1).

Responsible for evaluating whether an audit evidence text satisfies a specific
regulatory obligation:
1. Compares evidence text against regulatory obligation text and metadata (clause/category).
2. Determines coverage status as one of: FULL, PARTIAL, NONE.
3. Uses configured LLM provider (Groq or Gemini) with structured JSON prompting.
4. Provides a robust rule-based / keyword fallback for offline testing or LLM downtime.
5. Returns structured output validated with CoverageAssessmentResult Pydantic schema:
   - status (FULL, PARTIAL, NONE)
   - confidence score (0.0 to 1.0)
   - reasoning
   - relevant evidence snippet
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from uuid import UUID

import httpx

from app.core.config import settings
from app.schemas.coverage import (
    CoverageAssessmentRequest,
    CoverageAssessmentResult,
    CoverageStatus,
)

logger = logging.getLogger(__name__)

# Default prompt path
DEFAULT_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "assess_coverage.txt"

# Default LLM models
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"
DEFAULT_GEMINI_MODEL = "gemini-1.5-flash"

# Embedded prompt fallback if template file is not found
EMBEDDED_PROMPT_TEMPLATE = """
You are an expert regulatory compliance auditor and legal engineering AI.
Your task is to evaluate whether the provided evidence text demonstrates compliance with a specific regulatory obligation.

### Regulatory Obligation:
- Clause / Control ID: {clause}
- Category / Domain: {category}
- Obligation Statement:
{obligation_text}

### Audit Evidence Text:
{evidence_text}

### Evaluation Criteria:
Determine the coverage status as exactly one of:
1. "FULL": The evidence completely satisfies all mandatory criteria, requirements, and controls specified in the obligation.
2. "PARTIAL": The evidence addresses some aspects or parts of the obligation, but leaves gaps, lacks complete enforcement, or only partially implements the required controls.
3. "NONE": The evidence does not address the obligation, is irrelevant, contradicts the requirement, or shows no implementation of the required controls.

### Instructions:
- Assess the match objectively based ONLY on the evidence text provided.
- Provide a confidence score from 0.0 to 1.0.
- Provide clear, auditor-grade reasoning explaining specifically what requirements are met or missing.
- Extract the exact relevant quote or snippet from the evidence text that supports your determination. If status is NONE, the snippet should be null or empty.
- Return ONLY a valid JSON object with no markdown formatting.

### JSON Output Format:
{{
  "status": "FULL" | "PARTIAL" | "NONE",
  "confidence": 0.95,
  "reasoning": "Explanation of why the evidence fully, partially, or does not meet the obligation...",
  "relevant_snippet": "Exact quote from evidence..."
}}
""".strip()


class CoverageServiceError(Exception):
    """Base exception for CoverageService errors."""
    pass


class CoverageLLMError(CoverageServiceError):
    """Raised when LLM provider request fails or authentication fails."""
    pass


class CoverageParseError(CoverageServiceError):
    """Raised when LLM response cannot be parsed or validated."""
    pass


class CoverageService:
    """
    Service for assessing whether an audit evidence statement satisfies a regulatory obligation.
    """

    def __init__(
        self,
        provider: Optional[str] = None,
        groq_api_key: Optional[str] = None,
        gemini_api_key: Optional[str] = None,
        prompt_path: Optional[Union[str, Path]] = None,
        http_client: Optional[httpx.AsyncClient] = None,
        fallback_enabled: bool = True,
        graph_service: Optional[Any] = None,
    ):
        """
        Initialize CoverageService.

        :param provider: LLM provider ("groq" or "gemini", defaults to settings.LLM_PROVIDER)
        :param groq_api_key: Groq API key
        :param gemini_api_key: Google Gemini API key
        :param prompt_path: Custom path to prompt template
        :param http_client: Reusable httpx.AsyncClient
        :param fallback_enabled: If True, uses rule-based fallback when LLM is unavailable
        :param graph_service: Optional GraphService instance for Neo4j operations
        """
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
        self.fallback_enabled = fallback_enabled
        self._prompt_template: Optional[str] = None
        self._graph_service = graph_service

    @property
    def graph_service(self) -> Any:
        """
        Lazily resolved GraphService instance for Neo4j graph operations.
        """
        if self._graph_service is None:
            from app.services.graph_service import graph_service
            self._graph_service = graph_service
        return self._graph_service

    # -------------------------------------------------------------------------
    # Prompt Template Management
    # -------------------------------------------------------------------------

    def load_prompt_template(self) -> str:
        """
        Load the coverage assessment prompt template from file or fallback to embedded template.
        """
        if self._prompt_template is None:
            if self.prompt_path.exists():
                self._prompt_template = self.prompt_path.read_text(encoding="utf-8")
            else:
                logger.warning(
                    f"Prompt template file not found at {self.prompt_path}, using embedded template."
                )
                self._prompt_template = EMBEDDED_PROMPT_TEMPLATE
        return self._prompt_template

    def format_prompt(
        self,
        evidence_text: str,
        obligation_text: str,
        clause: Optional[str] = None,
        category: Optional[str] = None,
    ) -> str:
        """
        Format the coverage assessment prompt with evidence, obligation, and metadata.
        """
        template = self.load_prompt_template()
        clause_str = str(clause).strip() if clause else "Not Specified"
        category_str = str(category).strip() if category else "General Compliance"

        formatted = template.replace("{evidence_text}", evidence_text.strip())
        formatted = formatted.replace("{obligation_text}", obligation_text.strip())
        formatted = formatted.replace("{clause}", clause_str)
        formatted = formatted.replace("{category}", category_str)
        return formatted

    # -------------------------------------------------------------------------
    # Main Assessment Pipeline
    # -------------------------------------------------------------------------

    async def assess_coverage(
        self,
        evidence_text: str,
        obligation_text: Union[str, Any],
        clause: Optional[str] = None,
        category: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        use_fallback: bool = True,
    ) -> CoverageAssessmentResult:
        """
        Assess coverage of a regulatory obligation by audit evidence text.

        :param evidence_text: Statement or excerpt of audit evidence / policy / configuration
        :param obligation_text: Obligation requirement text, or an obligation object
        :param clause: Optional clause or control ID (e.g. 'CC6.1', 'Article 5(1)(a)')
        :param category: Optional control category or domain (e.g. 'Access Control')
        :param metadata: Optional additional obligation metadata dictionary
        :param provider: Optional provider override ('groq' or 'gemini')
        :param model: Optional model override
        :param use_fallback: Whether to use keyword/rule-based fallback on LLM failure
        :return: Validated CoverageAssessmentResult Pydantic model
        """
        # Normalize input obligation if passed as an object or dictionary
        norm_text, norm_clause, norm_cat, norm_meta = self._normalize_inputs(
            obligation_text=obligation_text,
            clause=clause,
            category=category,
            metadata=metadata,
        )

        if not evidence_text or not evidence_text.strip():
            return CoverageAssessmentResult(
                status=CoverageStatus.NONE,
                confidence=1.0,
                reasoning="No evidence text was provided for evaluation.",
                relevant_snippet=None,
            )

        if not norm_text or not norm_text.strip():
            return CoverageAssessmentResult(
                status=CoverageStatus.NONE,
                confidence=1.0,
                reasoning="No obligation text was provided for evaluation.",
                relevant_snippet=None,
            )

        target_provider = (provider or self.provider).lower().strip()

        # Check if API keys are present for the chosen provider
        has_key = False
        if target_provider == "groq" and self.groq_api_key and not self.groq_api_key.startswith("your-"):
            has_key = True
        elif target_provider in ("gemini", "google") and self.gemini_api_key and not self.gemini_api_key.startswith("your-"):
            has_key = True

        # If no valid API key is present and fallback is allowed, run rule-based assessment
        if not has_key:
            if use_fallback and self.fallback_enabled:
                logger.info(
                    f"No API key configured for provider '{target_provider}'. "
                    "Executing rule-based / keyword coverage assessment fallback."
                )
                return self.assess_coverage_rule_based(
                    evidence_text=evidence_text,
                    obligation_text=norm_text,
                    clause=norm_clause,
                    category=norm_cat,
                    metadata=norm_meta,
                )
            raise CoverageLLMError(f"API key not configured for LLM provider '{target_provider}'.")

        # Execute LLM coverage assessment
        formatted_prompt = self.format_prompt(
            evidence_text=evidence_text,
            obligation_text=norm_text,
            clause=norm_clause,
            category=norm_cat,
        )

        try:
            logger.info(
                f"Dispatching coverage assessment to LLM provider '{target_provider}' "
                f"for clause '{norm_clause or 'N/A'}'..."
            )
            raw_response = await self._call_llm(
                prompt=formatted_prompt,
                provider=target_provider,
                model=model,
            )
            result = self.parse_and_validate_response(raw_response)
            logger.info(
                f"Coverage assessment completed: status={result.status.value}, "
                f"confidence={result.confidence:.2f}"
            )
            return result

        except Exception as err:
            logger.warning(f"LLM coverage assessment failed ({err}).", exc_info=True)
            if use_fallback and self.fallback_enabled:
                logger.info("Falling back to rule-based coverage assessment.")
                return self.assess_coverage_rule_based(
                    evidence_text=evidence_text,
                    obligation_text=norm_text,
                    clause=norm_clause,
                    category=norm_cat,
                    metadata=norm_meta,
                )
            if isinstance(err, CoverageServiceError):
                raise
            raise CoverageServiceError(f"Coverage assessment failed: {err}") from err

    # -------------------------------------------------------------------------
    # Input Normalization
    # -------------------------------------------------------------------------

    @staticmethod
    def _normalize_inputs(
        obligation_text: Union[str, Any],
        clause: Optional[str] = None,
        category: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, Optional[str], Optional[str], Dict[str, Any]]:
        """
        Normalize obligation input arguments from string, model, or dictionary.
        """
        meta = dict(metadata or {})
        text = ""
        resolved_clause = clause
        resolved_category = category

        if isinstance(obligation_text, str):
            text = obligation_text.strip()
        elif isinstance(obligation_text, dict):
            text = str(obligation_text.get("text") or obligation_text.get("description") or "").strip()
            resolved_clause = resolved_clause or obligation_text.get("clause") or obligation_text.get("code")
            resolved_category = resolved_category or obligation_text.get("category")
            meta.update(obligation_text)
        else:
            # Handle Pydantic or SQLAlchemy model
            text = str(
                getattr(obligation_text, "text", None)
                or getattr(obligation_text, "description", None)
                or ""
            ).strip()
            resolved_clause = (
                resolved_clause
                or getattr(obligation_text, "clause", None)
                or getattr(obligation_text, "code", None)
            )
            resolved_category = resolved_category or getattr(obligation_text, "category", None)

        if not resolved_clause and "clause" in meta:
            resolved_clause = str(meta["clause"])
        if not resolved_category and "category" in meta:
            resolved_category = str(meta["category"])

        return text, resolved_clause, resolved_category, meta

    # -------------------------------------------------------------------------
    # LLM Provider Dispatch
    # -------------------------------------------------------------------------

    async def _call_llm(
        self,
        prompt: str,
        provider: str,
        model: Optional[str] = None,
    ) -> str:
        """
        Dispatch prompt to the configured LLM provider.
        """
        if provider == "groq":
            return await self._call_groq(prompt=prompt, model=model)
        elif provider in ("gemini", "google"):
            return await self._call_gemini(prompt=prompt, model=model)
        else:
            raise CoverageLLMError(f"Unsupported LLM provider '{provider}'. Must be 'groq' or 'gemini'.")

    async def _call_groq(self, prompt: str, model: Optional[str] = None) -> str:
        """
        Call Groq API via SDK or direct HTTP request with candidate model fallback.
        """
        api_key = self.groq_api_key
        if not api_key or not api_key.strip() or api_key.startswith("your-"):
            raise CoverageLLMError("GROQ_API_KEY is not configured.")

        # Candidate models to try in order of preference
        candidate_models = (
            [model] if model else ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "qwen/qwen3.6-27b", "qwen/qwen3.8-27b", "openai/gpt-oss-20b"]
        )

        last_err: Optional[Exception] = None
        for target_model in candidate_models:
            try:
                try:
                    from groq import AsyncGroq
                    client = AsyncGroq(api_key=api_key)
                    completion = await client.chat.completions.create(
                        model=target_model,
                        messages=[
                            {
                                "role": "system",
                                "content": "You are a regulatory compliance auditor. You output strictly valid JSON objects.",
                            },
                            {"role": "user", "content": prompt},
                        ],
                        temperature=0.1,
                        max_tokens=2048,
                    )
                    content = completion.choices[0].message.content
                    if not content:
                        raise CoverageParseError("Groq returned an empty response.")
                    return content
                except ImportError:
                    headers = {
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    }
                    payload = {
                        "model": target_model,
                        "messages": [
                            {
                                "role": "system",
                                "content": "You are a regulatory compliance auditor. You output strictly valid JSON objects.",
                            },
                            {"role": "user", "content": prompt},
                        ],
                        "temperature": 0.1,
                        "max_tokens": 2048,
                    }

                    async with (self._http_client or httpx.AsyncClient(timeout=60.0)) as client:
                        response = await client.post(
                            "https://api.groq.com/openai/v1/chat/completions",
                            headers=headers,
                            json=payload,
                        )
                        if response.status_code != 200:
                            raise CoverageLLMError(
                                f"Groq API returned HTTP {response.status_code}: {response.text}"
                            )
                        data = response.json()
                        content = data["choices"][0]["message"]["content"]
                        if not content:
                            raise CoverageParseError("Groq returned an empty response.")
                        return content
            except Exception as e:
                err_msg = str(e).lower()
                last_err = e
                # If model is decommissioned, not found, or not accessible, try next candidate
                if "model_not_found" in err_msg or "decommissioned" in err_msg or "does not exist" in err_msg or "404" in err_msg or "400" in err_msg:
                    logger.debug(f"Groq model '{target_model}' unavailable ({e}), trying next candidate...")
                    continue
                if isinstance(e, (CoverageLLMError, CoverageParseError)):
                    raise
                raise CoverageLLMError(f"Groq API call failed: {e}") from e

        if isinstance(last_err, (CoverageLLMError, CoverageParseError)):
            raise last_err
        raise CoverageLLMError(f"All Groq candidate models failed: {last_err}") from last_err

    async def _call_gemini(self, prompt: str, model: Optional[str] = None) -> str:
        """
        Call Google Gemini API via REST endpoint.
        """
        api_key = self.gemini_api_key
        if not api_key or not api_key.strip() or api_key.startswith("your-"):
            raise CoverageLLMError("GEMINI_API_KEY is not configured.")

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
                    raise CoverageLLMError(
                        f"Gemini API returned HTTP {response.status_code}: {response.text}"
                    )
                data = response.json()
                candidates = data.get("candidates", [])
                if not candidates:
                    raise CoverageParseError(f"Gemini returned no candidates: {data}")
                content = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                if not content:
                    raise CoverageParseError("Gemini returned empty text content.")
                return content
        except Exception as e:
            if isinstance(e, (CoverageLLMError, CoverageParseError)):
                raise
            raise CoverageLLMError(f"Gemini API call failed: {e}") from e

    # -------------------------------------------------------------------------
    # Response Parsing & Validation
    # -------------------------------------------------------------------------

    def parse_and_validate_response(self, response_text: str) -> CoverageAssessmentResult:
        """
        Parse raw LLM response text into a validated CoverageAssessmentResult model.
        """
        if not response_text or not response_text.strip():
            raise CoverageParseError("Received empty response text from LLM.")

        cleaned = self._clean_json_text(response_text)

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            data = self._extract_first_json_object(cleaned)
            if data is None:
                raise CoverageParseError(f"No valid JSON object found in response: {response_text[:300]}")

        # If data is a list of results, take the first element
        if isinstance(data, list) and data:
            data = data[0]

        if not isinstance(data, dict):
            raise CoverageParseError(f"Expected JSON object, got {type(data).__name__}")

        try:
            return CoverageAssessmentResult.model_validate(data)
        except Exception as val_err:
            raise CoverageParseError(f"Validation error for coverage result: {val_err}") from val_err

    @staticmethod
    def _clean_json_text(text: str) -> str:
        """Strip reasoning <think>...</think> tags and markdown code block wrappers (```json ... ```)."""
        text = text.strip()
        # Remove reasoning think tags from models like Qwen
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        return text

    @classmethod
    def _extract_first_json_object(cls, text: str) -> Optional[Dict[str, Any]]:
        """
        Extract the first complete JSON object using JSONDecoder.raw_decode.
        Robustly handles extra trailing commentary, notes, or subsequent JSON blocks.
        """
        idx = text.find("{")
        while idx != -1:
            try:
                obj, _ = json.JSONDecoder().raw_decode(text[idx:])
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                pass
            idx = text.find("{", idx + 1)
        return None

    # -------------------------------------------------------------------------
    # Rule-Based / Keyword Fallback Engine
    # -------------------------------------------------------------------------

    def assess_coverage_rule_based(
        self,
        evidence_text: str,
        obligation_text: str,
        clause: Optional[str] = None,
        category: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> CoverageAssessmentResult:
        """
        Rule-based keyword, stemming, and domain concept overlap fallback for coverage assessment.

        Evaluates:
        1. Term and concept overlap using stemmed roots between obligation and evidence.
        2. Domain concept clusters (Access Control, Cryptography, Logging, Data Protection, etc.).
        3. Operational implementation vs gap/planned indicators.
        4. Extracts the strongest supporting evidence snippet.
        """
        clean_evidence = evidence_text.strip()
        clean_obligation = obligation_text.strip()

        # Domain concept clusters for cybersecurity and regulatory compliance
        domain_clusters: Dict[str, Set[str]] = {
            "access_control": {
                "acces", "acce", "authe", "authent", "mfa", "passw", "crede", "login", "sso",
                "biome", "permi", "privi", "role", "rbac", "ident", "token", "author", "keys",
            },
            "cryptography": {
                "encry", "ciphe", "crypt", "tls", "ssl", "aes", "hash", "key", "certi", "protect",
            },
            "audit_logging": {
                "audit", "log", "loggi", "monit", "trace", "alert", "siem", "recor", "event",
            },
            "data_protection": {
                "priva", "conse", "subje", "perso", "gdpr", "recti", "eras", "minim", "data",
            },
            "incident_response": {
                "incid", "breac", "notif", "respo", "escal", "discov", "vulne", "patch",
            },
            "business_continuity": {
                "backu", "disas", "recov", "resto", "failo", "redun", "avail", "resil",
            },
        }

        # Split evidence into individual sentences for pinpoint snippet extraction
        sentences = [
            s.strip() for s in re.split(r"(?<=[.!?])\s+", clean_evidence) if len(s.strip()) > 15
        ]
        if not sentences:
            sentences = [clean_evidence]

        stop_words = {
            "the", "and", "that", "this", "with", "from", "shall", "must", "should", "will",
            "have", "been", "they", "their", "each", "other", "such", "than", "more", "also",
            "into", "over", "under", "both", "only", "then", "when", "where", "which", "while",
            "for", "are", "can", "our", "all", "per", "its", "has", "had",
        }

        def get_stems(text: str) -> Set[str]:
            words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
            stems = set()
            for w in words:
                if w not in stop_words:
                    stems.add(w)
                    if len(w) >= 4:
                        stems.add(w[:4])
                        stems.add(w[:5])
            return stems

        ob_context = clean_obligation
        if category:
            ob_context += f" {category}"
        if metadata and "keywords" in metadata:
            kw = metadata["keywords"]
            if isinstance(kw, (list, tuple, set)):
                ob_context += " " + " ".join(str(k) for k in kw)

        ob_stems = get_stems(ob_context)
        ev_stems = get_stems(clean_evidence)

        # Matched stems
        matched_stems = ob_stems.intersection(ev_stems)

        # Domain cluster overlap
        domain_matched: Set[str] = set()
        matched_domains: Set[str] = set()
        for dom_name, dom_tokens in domain_clusters.items():
            if ob_stems.intersection(dom_tokens):
                intersection = ev_stems.intersection(dom_tokens)
                if intersection:
                    domain_matched.update(intersection)
                    matched_domains.add(dom_name)

        # Enforcement & gap indicator regexes
        enforcement_pattern = r"\b(?:must|enforced|configured|required|active|mandatory|reviewed|automated|revoked|implements?|authenticated?|verified)\b"
        has_enforcement = bool(re.search(enforcement_pattern, clean_evidence, re.IGNORECASE))

        gap_pattern = r"\b(?:planned|planning|in progress|partially|draft|future|not yet|gap|roadmap|voluntary|optional|some users)\b"
        has_gap = bool(re.search(gap_pattern, clean_evidence, re.IGNORECASE))

        # Best sentence selection based on matched stems count
        best_sentence = sentences[0]
        max_sentence_score = 0
        for s in sentences:
            s_stems = get_stems(s)
            score = len(s_stems.intersection(ob_stems)) + len(s_stems.intersection(domain_matched))
            if score > max_sentence_score:
                max_sentence_score = score
                best_sentence = s

        # Combined relevance metrics
        total_matched_concepts = len(matched_stems) + (len(domain_matched) * 2)

        # Coverage Determination
        if total_matched_concepts >= 4 and has_enforcement and not has_gap:
            status = CoverageStatus.FULL
            confidence = min(0.95, 0.75 + (min(total_matched_concepts, 10) * 0.02))
            reasoning = (
                f"Rule-based assessment determined FULL coverage: Evidence demonstrates operational implementation "
                f"aligning with obligation requirement '{clause or 'control'}' across key concepts "
                f"({', '.join(sorted(list(domain_matched | matched_stems))[:5])})."
            )
            snippet = best_sentence

        elif (total_matched_concepts >= 2 or has_gap) and (domain_matched or matched_stems):
            status = CoverageStatus.PARTIAL
            confidence = min(0.88, 0.65 + (min(total_matched_concepts, 8) * 0.02))
            if has_gap:
                reasoning = (
                    f"Rule-based assessment determined PARTIAL coverage: Evidence touches relevant controls "
                    f"({', '.join(sorted(list(domain_matched | matched_stems))[:4])}), but includes qualifiers "
                    f"indicating planned, roadmap, or incomplete implementation."
                )
            else:
                reasoning = (
                    f"Rule-based assessment determined PARTIAL coverage: Evidence partially implements the required "
                    f"controls, but does not provide complete verification across all aspects of '{clause or 'the obligation'}'."
                )
            snippet = best_sentence

        else:
            status = CoverageStatus.NONE
            confidence = 0.90
            reasoning = (
                f"Rule-based assessment determined NO coverage: Evidence text does not substantively address "
                f"the domain or controls mandated by obligation '{clause or 'specified'}'."
            )
            snippet = None

        return CoverageAssessmentResult(
            status=status,
            confidence=round(confidence, 4),
            reasoning=reasoning,
            relevant_snippet=snippet,
        )

    # -------------------------------------------------------------------------
    # Neo4j Graph Integration (Phase 2, Step 5.2)
    # -------------------------------------------------------------------------

    async def create_satisfies_relationship(
        self,
        evidence_id: Union[str, UUID],
        obligation_id: Union[str, UUID],
        coverage: Union[str, CoverageStatus],
        confidence: float,
        reasoning: str,
        evidence_text: Optional[str] = None,
        properties: Optional[Dict[str, Any]] = None,
        create_nodes_if_missing: bool = False,
    ) -> Dict[str, Any]:
        """
        Create or update the SATISFIES relationship in Neo4j between EvidenceArtifact
        and RegulatoryObligation (Phase 2, Step 5.2).

        Properties stored on the relationship edge:
        - coverage: FULL, PARTIAL, or NONE
        - coverage_status: Alias for coverage
        - confidence: Confidence score (0.0 to 1.0)
        - reasoning: Auditor-grade rationale
        - evidence_text: Assessed evidence text or snippet
        - updated_at: ISO 8601 timestamp

        Uses Cypher MERGE for idempotent updates (prevents duplicate relationships on re-runs).

        :param evidence_id: UUID or string ID of the EvidenceArtifact node
        :param obligation_id: UUID or string ID of the RegulatoryObligation node
        :param coverage: Coverage status (FULL, PARTIAL, NONE)
        :param confidence: Assessment confidence score (0.0 to 1.0)
        :param reasoning: Explanatory reasoning
        :param evidence_text: Relevant quote or evidence statement
        :param properties: Optional additional relationship properties
        :param create_nodes_if_missing: If True, MERGE nodes if not present
        :return: Relationship edge dict with rel_type, properties, source_id, target_id
        """
        return await self.graph_service.create_satisfies_relationship(
            evidence_id=evidence_id,
            obligation_id=obligation_id,
            coverage=coverage,
            confidence=confidence,
            reasoning=reasoning,
            evidence_text=evidence_text,
            properties=properties,
            create_nodes_if_missing=create_nodes_if_missing,
        )

    async def store_coverage_assessment(
        self,
        evidence_id: Union[str, UUID],
        obligation_id: Union[str, UUID],
        assessment: Union[CoverageAssessmentResult, Dict[str, Any]],
        evidence_text: Optional[str] = None,
        properties: Optional[Dict[str, Any]] = None,
        create_nodes_if_missing: bool = False,
    ) -> Dict[str, Any]:
        """
        Store a successful coverage assessment in Neo4j as a SATISFIES relationship edge (Phase 2, Step 5.2).

        :param evidence_id: UUID or string ID of the EvidenceArtifact node
        :param obligation_id: UUID or string ID of the RegulatoryObligation node
        :param assessment: CoverageAssessmentResult instance or dictionary
        :param evidence_text: Optional evidence text override (defaults to assessment.relevant_snippet)
        :param properties: Optional extra properties dictionary
        :param create_nodes_if_missing: If True, MERGE endpoints if not already in graph
        :return: Relationship edge dictionary with rel_type, properties, source_id, target_id
        """
        return await self.graph_service.store_coverage_assessment(
            evidence_id=evidence_id,
            obligation_id=obligation_id,
            assessment=assessment,
            evidence_text=evidence_text,
            properties=properties,
            create_nodes_if_missing=create_nodes_if_missing,
        )

    async def store_assessment_in_graph(
        self,
        evidence_id: Union[str, UUID],
        obligation_id: Union[str, UUID],
        assessment: Union[CoverageAssessmentResult, Dict[str, Any]],
        evidence_text: Optional[str] = None,
        properties: Optional[Dict[str, Any]] = None,
        create_nodes_if_missing: bool = False,
    ) -> Dict[str, Any]:
        """
        Convenience alias for store_coverage_assessment.
        """
        return await self.store_coverage_assessment(
            evidence_id=evidence_id,
            obligation_id=obligation_id,
            assessment=assessment,
            evidence_text=evidence_text,
            properties=properties,
            create_nodes_if_missing=create_nodes_if_missing,
        )

    async def assess_and_store(
        self,
        evidence_id: Union[str, UUID],
        obligation_id: Union[str, UUID],
        evidence_text: str,
        obligation_text: Union[str, Any],
        clause: Optional[str] = None,
        category: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        use_fallback: bool = True,
        properties: Optional[Dict[str, Any]] = None,
        create_nodes_if_missing: bool = False,
    ) -> Tuple[CoverageAssessmentResult, Dict[str, Any]]:
        """
        Assess coverage and store the resulting SATISFIES relationship in Neo4j in a single call.

        :param evidence_id: UUID or string ID of the EvidenceArtifact node
        :param obligation_id: UUID or string ID of the RegulatoryObligation node
        :param evidence_text: Statement or excerpt of audit evidence
        :param obligation_text: Obligation requirement text or model
        :param clause: Optional clause/control ID (e.g. 'CC6.1')
        :param category: Optional control category (e.g. 'Access Control')
        :param metadata: Optional metadata dictionary
        :param provider: LLM provider override ('groq' or 'gemini')
        :param model: LLM model override
        :param use_fallback: Whether to fallback to rule-based evaluation
        :param properties: Optional extra properties for the graph relationship
        :param create_nodes_if_missing: Whether to create graph nodes if not present
        :return: Tuple of (CoverageAssessmentResult, relationship dict)
        """
        assessment = await self.assess_coverage(
            evidence_text=evidence_text,
            obligation_text=obligation_text,
            clause=clause,
            category=category,
            metadata=metadata,
            provider=provider,
            model=model,
            use_fallback=use_fallback,
        )

        rel = await self.store_coverage_assessment(
            evidence_id=evidence_id,
            obligation_id=obligation_id,
            assessment=assessment,
            evidence_text=evidence_text,
            properties=properties,
            create_nodes_if_missing=create_nodes_if_missing,
        )

        return assessment, rel

    async def get_satisfies_relationship(
        self,
        evidence_id: Union[str, UUID],
        obligation_id: Union[str, UUID],
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve existing SATISFIES relationship between EvidenceArtifact and RegulatoryObligation from Neo4j.

        :param evidence_id: ID of the EvidenceArtifact node
        :param obligation_id: ID of the RegulatoryObligation node
        :return: Relationship dictionary if found, else None
        """
        return await self.graph_service.get_satisfies_relationship(
            evidence_id=evidence_id,
            obligation_id=obligation_id,
        )


# Global singleton instance
coverage_service = CoverageService()
