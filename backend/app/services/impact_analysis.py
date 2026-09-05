"""
Change Impact Analysis Service (Phase 2, Step 7.1).

Accepts new or draft regulatory text, extracts discrete structured obligations
using the existing ExtractionService (Phase 2, Step 3), validates them, and preserves
key identifiers (clause, text, category, mandatory, keywords, IDs) for downstream
change comparison (Step 7.2), graph traversal (Step 7.3), and invalidation reporting (Step 7.4).
"""

import logging
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from uuid import UUID, uuid4

from app.schemas.extraction import ExtractedObligation
from app.schemas.impact import (
    DraftExtractionResult,
    DraftObligation,
    DraftRegulationInput,
)
from app.services.extraction_service import (
    ExtractionService,
    extraction_service as default_extraction_service,
)
from app.services.text_chunker import chunk_regulatory_text

logger = logging.getLogger(__name__)

# Sample text for testing change impact analysis (Modified GDPR Article 5 storage limitation)
SAMPLE_GDPR_DRAFT_ARTICLE_5_TEXT = """
Article 5 - Principles relating to processing of personal data (Amended Draft)
1. Personal data shall be:
(a) processed lawfully, fairly and in a transparent manner in relation to the data subject ('lawfulness, fairness and transparency');
(b) collected for specified, explicit and legitimate purposes and not further processed in a manner that is incompatible with those purposes ('purpose limitation');
(c) adequate, relevant and limited to what is strictly necessary in relation to the purposes for which they are processed ('data minimisation');
(d) accurate and, where necessary, kept up to date; every reasonable step must be taken to ensure inaccurate data are rectified without delay ('accuracy');
(e) kept in a form which permits identification of data subjects for a maximum of 12 months from the date of collection, after which data must be erased or irreversibly anonymised ('storage limitation');
(f) processed in a manner that ensures appropriate security of the personal data, including protection against unauthorised or unlawful processing and against accidental loss, destruction or damage, using appropriate technical or organisational measures ('integrity and confidentiality').
2. The controller shall be responsible for, and be able to demonstrate compliance with, paragraph 1 ('accountability').
""".strip()


class ImpactAnalysisError(Exception):
    """Base exception for change impact analysis operations."""
    pass


class DraftExtractionError(ImpactAnalysisError):
    """Raised when an unrecoverable error occurs during draft obligation extraction."""
    pass


class ImpactAnalysisService:
    """
    Change Impact Analysis Service (Phase 2, Step 7).

    Accepts new or draft regulatory text, extracts discrete structured obligations
    using the existing ExtractionService (Phase 2, Step 3), validates them, and preserves
    key identifiers (clause, text, category, mandatory, keywords, IDs) for downstream
    change comparison (Step 7.2), graph traversal (Step 7.3), and invalidation reporting (Step 7.4).
    """

    def __init__(
        self,
        extraction_service: Optional[ExtractionService] = None,
        chunk_size: int = 3000,
        chunk_overlap: int = 200,
    ):
        """
        Initialize ImpactAnalysisService with an underlying ExtractionService.

        :param extraction_service: Reusable ExtractionService instance (defaults to global singleton)
        :param chunk_size: Maximum character length per chunk for large draft texts
        :param chunk_overlap: Overlap character count between consecutive chunks
        """
        self.extraction_service = extraction_service or default_extraction_service
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    async def extract_draft_obligations(
        self,
        draft_input: Union[str, DraftRegulationInput, Dict[str, Any]],
        framework: Optional[str] = None,
        version: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        raise_on_error: bool = False,
    ) -> DraftExtractionResult:
        """
        Accept draft regulatory text and framework/version identifiers, extract and validate
        structured obligations reusing ExtractionService, and prepare them for comparison.

        :param draft_input: Raw draft regulatory text string, a DraftRegulationInput model, or dict
        :param framework: Framework name/identifier override (e.g. 'GDPR', 'SOC 2')
        :param version: Version/draft slug override (e.g. '2024-draft', 'v2.0')
        :param provider: LLM provider override ('groq' or 'gemini')
        :param model: Optional LLM model identifier
        :param metadata: Optional extra metadata dictionary
        :param raise_on_error: If True, raises DraftExtractionError when extraction fails
        :return: DraftExtractionResult containing list of validated DraftObligation objects
        """
        # 1. Normalize and unpack inputs
        raw_text, fw, ver, meta = self._normalize_input(
            draft_input=draft_input,
            framework=framework,
            version=version,
            metadata=metadata,
        )

        # 2. Gracefully handle empty or invalid draft text
        if not raw_text or not raw_text.strip():
            logger.warning("Empty or blank regulatory draft text provided to ImpactAnalysisService.")
            return DraftExtractionResult(
                framework=fw,
                version=ver,
                obligations=[],
                total_obligations=0,
                raw_text_length=0,
                metadata={
                    "status": "empty_or_invalid_text",
                    **meta,
                },
            )

        cleaned_text = raw_text.strip()
        text_length = len(cleaned_text)

        # 3. Intelligent text chunking
        if text_length > self.chunk_size:
            chunks = chunk_regulatory_text(
                cleaned_text,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )
            logger.info(f"Split draft regulatory text ({text_length} chars) into {len(chunks)} chunk(s).")
        else:
            chunks = [cleaned_text]

        # 4. Extract obligations chunk-by-chunk using ExtractionService
        validated_obligations: List[DraftObligation] = []
        seen_keys: Set[Tuple[str, str]] = set()
        failed_chunks: int = 0
        chunk_errors: List[str] = []

        for chunk_idx, chunk in enumerate(chunks):
            logger.info(
                f"Extracting draft obligations from chunk {chunk_idx + 1}/{len(chunks)} "
                f"({len(chunk)} chars) for framework='{fw}', version='{ver}'..."
            )
            try:
                extracted_list: List[ExtractedObligation] = await self.extraction_service.extract_obligations(
                    text=chunk,
                    provider=provider,
                    model=model,
                )
            except Exception as chunk_exc:
                err_msg = f"Extraction failed for chunk {chunk_idx + 1}: {chunk_exc}"
                logger.error(err_msg, exc_info=True)
                chunk_errors.append(err_msg)
                failed_chunks += 1
                if raise_on_error:
                    raise DraftExtractionError(err_msg) from chunk_exc
                continue

            # 5. Transform and validate ExtractedObligation into DraftObligation
            for item in extracted_list:
                clause_clean = (item.clause or "").strip()
                text_clean = (item.text or "").strip()

                # Deduplicate identical clause + text extracted from overlapping chunks
                dedup_key = (clause_clean.lower(), text_clean.lower())
                if dedup_key in seen_keys:
                    logger.debug(f"Skipping duplicate extracted obligation: {clause_clean}")
                    continue
                seen_keys.add(dedup_key)

                draft_ob = DraftObligation.from_extracted(
                    extracted=item,
                    framework=fw,
                    version=ver,
                    source_text=chunk if len(chunks) > 1 else cleaned_text,
                    metadata={
                        "chunk_index": chunk_idx,
                        "total_chunks": len(chunks),
                        **(meta or {}),
                    },
                )
                validated_obligations.append(draft_ob)

        logger.info(
            f"Extracted and validated {len(validated_obligations)} draft obligations "
            f"for framework='{fw}', version='{ver}' ({failed_chunks} failed chunks)."
        )

        result_meta = {
            "total_chunks": len(chunks),
            "failed_chunks": failed_chunks,
            **(meta or {}),
        }
        if chunk_errors:
            result_meta["extraction_errors"] = chunk_errors

        return DraftExtractionResult(
            framework=fw,
            version=ver,
            obligations=validated_obligations,
            total_obligations=len(validated_obligations),
            raw_text_length=text_length,
            metadata=result_meta,
        )

    def _normalize_input(
        self,
        draft_input: Union[str, DraftRegulationInput, Dict[str, Any]],
        framework: Optional[str] = None,
        version: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, Optional[str], Optional[str], Dict[str, Any]]:
        """
        Normalize various input types (string, Pydantic model, or dict) into
        (raw_text, framework, version, metadata).
        """
        meta: Dict[str, Any] = dict(metadata or {})

        if isinstance(draft_input, DraftRegulationInput):
            raw_text = draft_input.draft_text or ""
            fw = framework or draft_input.framework
            ver = version or draft_input.version
            meta = {**draft_input.metadata, **meta}
            if draft_input.existing_version_id:
                meta["existing_version_id"] = str(draft_input.existing_version_id)
        elif isinstance(draft_input, dict):
            raw_text = str(
                draft_input.get("draft_text")
                or draft_input.get("text")
                or draft_input.get("content")
                or ""
            )
            fw = framework or draft_input.get("framework")
            ver = version or draft_input.get("version")
            input_meta = draft_input.get("metadata")
            if isinstance(input_meta, dict):
                meta = {**input_meta, **meta}
            if "existing_version_id" in draft_input and draft_input["existing_version_id"]:
                meta["existing_version_id"] = str(draft_input["existing_version_id"])
        elif isinstance(draft_input, str):
            raw_text = draft_input
            fw = framework
            ver = version
        else:
            # Fallback for unexpected or invalid types
            logger.warning(f"Unexpected draft_input type: {type(draft_input).__name__}")
            raw_text = ""
            fw = framework
            ver = version

        return raw_text, fw, ver, meta

    # Step 7.2 hook placeholder
    # async def compare_with_existing_obligations(...):
    #     """Compare draft obligations with baseline obligations (Step 7.2)."""
    #     pass

    # Step 7.3 hook placeholder
    # async def find_connected_evidence(...):
    #     """Traverse graph to find evidence connected to impacted obligations (Step 7.3)."""
    #     pass

    # Step 7.4 hook placeholder
    # async def mark_evidence_potentially_invalid(...):
    #     """Flag evidence artifacts needing re-certification (Step 7.4)."""
    #     pass

    # Step 7.5 hook placeholder
    # async def generate_impact_report(...):
    #     """Synthesize audit-ready change impact report (Step 7.5)."""
    #     pass


# Global singleton instance for impact analysis
impact_analysis_service = ImpactAnalysisService()


async def extract_draft_obligations(
    draft_input: Union[str, DraftRegulationInput, Dict[str, Any]],
    framework: Optional[str] = None,
    version: Optional[str] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    raise_on_error: bool = False,
) -> DraftExtractionResult:
    """
    Convenience function extracting and validating obligations from a draft regulation
    using the global impact_analysis_service.
    """
    return await impact_analysis_service.extract_draft_obligations(
        draft_input=draft_input,
        framework=framework,
        version=version,
        provider=provider,
        model=model,
        metadata=metadata,
        raise_on_error=raise_on_error,
    )
