"""
Change Impact Analysis Service (Phase 2, Step 7.1 & 7.2).

Accepts new or draft regulatory text, extracts discrete structured obligations
using the existing ExtractionService (Phase 2, Step 3), validates them, and preserves
key identifiers (clause, text, category, mandatory, keywords, IDs) for downstream
change comparison (Step 7.2), graph traversal (Step 7.3), and invalidation reporting (Step 7.4).

Step 7.2 compares existing baseline obligations against draft obligations:
- Loads existing obligations for selected framework/version from Neo4j/SQL.
- Matches obligations deterministically by clause or stable ID where available.
- Where exact matching is unavailable, uses semantic similarity (embeddings/lexical overlap).
- Classifies changes as ADDED, MODIFIED, REMOVED, or UNCHANGED.
- Avoids marking minor wording/stylistic variations as major modifications unless
  regulatory meaning/burdens changed.
- Includes auditor-grade reasoning and confidence scores.
- Preserves read-only safety (does not mutate the production regulatory graph).
"""

import difflib
import json
import logging
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union
from uuid import UUID, uuid4

import httpx

from app.core.config import settings
from app.integrations.qdrant_client import (
    QdrantClient,
    qdrant_client as default_qdrant_client,
)
from app.schemas.extraction import ExtractedObligation
from app.schemas.impact import (
    CHANGE_ADDED,
    CHANGE_MODIFIED,
    CHANGE_REMOVED,
    CHANGE_UNCHANGED,
    CompareObligationsRequest,
    DraftExtractionResult,
    DraftObligation,
    DraftRegulationInput,
    ObligationChangeType,
    ObligationComparisonItem,
    ObligationComparisonResult,
)
from app.services.extraction_service import (
    ExtractionService,
    extraction_service as default_extraction_service,
)
from app.services.graph_service import (
    GraphService,
    graph_service as default_graph_service,
)
from app.services.text_chunker import chunk_regulatory_text

try:
    from sqlalchemy.ext.asyncio import AsyncSession
except ImportError:
    AsyncSession = Any

try:
    from app.repositories.framework_repo import (
        RegulatoryFrameworkRepository,
        RegulatoryRequirementRepository,
        RegulatoryVersionRepository,
    )
except ImportError:
    RegulatoryFrameworkRepository = None
    RegulatoryRequirementRepository = None
    RegulatoryVersionRepository = None

logger = logging.getLogger(__name__)

# Default prompt paths
DEFAULT_COMPARE_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "compare_obligations.txt"

# Default LLM models
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"
DEFAULT_GEMINI_MODEL = "gemini-1.5-flash"

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

# Sample baseline obligations for testing change impact analysis
SAMPLE_GDPR_BASELINE_ARTICLE_5_OBLIGATIONS: List[Dict[str, Any]] = [
    {
        "id": "GDPR_2016_ART5_1_A",
        "clause": "Article 5(1)(a)",
        "code": "Art. 5(1)(a)",
        "title": "Lawfulness, fairness and transparency",
        "text": "Personal data shall be processed lawfully, fairly and in a transparent manner in relation to the data subject ('lawfulness, fairness and transparency').",
        "category": "Lawfulness & Transparency",
        "mandatory": True,
        "keywords": ["personal data", "lawfulness", "fairness", "transparency"],
        "framework": "GDPR",
        "version": "2016",
    },
    {
        "id": "GDPR_2016_ART5_1_B",
        "clause": "Article 5(1)(b)",
        "code": "Art. 5(1)(b)",
        "title": "Purpose limitation",
        "text": "Personal data shall be collected for specified, explicit and legitimate purposes and not further processed in a manner that is incompatible with those purposes ('purpose limitation').",
        "category": "Purpose Limitation",
        "mandatory": True,
        "keywords": ["purpose limitation", "specified purposes"],
        "framework": "GDPR",
        "version": "2016",
    },
    {
        "id": "GDPR_2016_ART5_1_C",
        "clause": "Article 5(1)(c)",
        "code": "Art. 5(1)(c)",
        "title": "Data minimisation",
        "text": "Personal data shall be adequate, relevant and limited to what is strictly necessary in relation to the purposes for which they are processed ('data minimisation').",
        "category": "Data Minimisation",
        "mandatory": True,
        "keywords": ["data minimisation", "adequate", "relevant", "strictly necessary"],
        "framework": "GDPR",
        "version": "2016",
    },
    {
        "id": "GDPR_2024_ART5",
        "clause": "Article 5(1)(e)",
        "code": "Art. 5(1)(e)",
        "title": "Storage limitation",
        "text": "Data should only be retained as long as necessary for the purposes for which the personal data are processed.",
        "category": "Data Retention",
        "mandatory": True,
        "keywords": ["retention", "storage limitation", "necessary"],
        "framework": "GDPR",
        "version": "2016",
    },
    {
        "id": "GDPR_2016_LEGACY_AUDIT",
        "clause": "Article 5(3)-legacy",
        "code": "Art. 5(3)-legacy",
        "title": "Audit Logs",
        "text": "Organizations must maintain audit logs for all data subject access operations.",
        "category": "Audit Logging",
        "mandatory": True,
        "keywords": ["audit logs", "logging"],
        "framework": "GDPR",
        "version": "2016",
    },
]


def normalize_clause_key(clause: Optional[str]) -> str:
    """
    Normalize clause or article identifiers for deterministic matching.
    Strips punctuation, whitespace, and unifies common abbreviations.
    Example: 'Article 5(1)(e)' -> 'art51e', 'Art. 5' -> 'art5'.
    """
    if not clause:
        return ""
    c = str(clause).lower().strip()
    c = re.sub(r'^(?:article|art\.?|section|sec\.?|clause|control|rule|criterion|para\.?|paragraph)\s*', 'art_', c)
    c = re.sub(r'[^a-z0-9]', '', c)
    return c


def normalize_id_key(val: Optional[Union[str, UUID]]) -> str:
    """Normalize obligation ID for matching."""
    if val is None:
        return ""
    return str(val).strip().lower()


class ImpactAnalysisError(Exception):
    """Base exception for change impact analysis operations."""
    pass


class DraftExtractionError(ImpactAnalysisError):
    """Raised when an unrecoverable error occurs during draft obligation extraction."""
    pass


class ObligationComparisonError(ImpactAnalysisError):
    """Raised when an unrecoverable error occurs during obligation comparison."""
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
        graph_service: Optional[GraphService] = None,
        qdrant_client: Optional[QdrantClient] = None,
        chunk_size: int = 3000,
        chunk_overlap: int = 200,
        provider: Optional[str] = None,
        groq_api_key: Optional[str] = None,
        gemini_api_key: Optional[str] = None,
        prompt_path: Optional[Union[str, Path]] = None,
        http_client: Optional[httpx.AsyncClient] = None,
    ):
        """
        Initialize ImpactAnalysisService with underlying services and configurations.

        :param extraction_service: Reusable ExtractionService instance (defaults to global singleton)
        :param graph_service: Reusable GraphService instance for reading baseline obligations from Neo4j
        :param qdrant_client: Optional Qdrant client for embedding similarity
        :param chunk_size: Maximum character length per chunk for large draft texts
        :param chunk_overlap: Overlap character count between consecutive chunks
        :param provider: LLM provider override ('groq' or 'gemini')
        :param groq_api_key: Optional Groq API key
        :param gemini_api_key: Optional Google Gemini API key
        :param prompt_path: Optional path to prompt template for comparison
        :param http_client: Optional httpx.AsyncClient for LLM network requests
        """
        self.extraction_service = extraction_service or default_extraction_service
        self.graph_service = graph_service or default_graph_service
        self.qdrant_client = qdrant_client or default_qdrant_client
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

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
        self.prompt_path = Path(prompt_path) if prompt_path else DEFAULT_COMPARE_PROMPT_PATH
        self._http_client = http_client
        self._prompt_template: Optional[str] = None

    # -------------------------------------------------------------------------
    # Step 7.1: Extraction of Draft Obligations
    # -------------------------------------------------------------------------

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
            logger.warning(f"Unexpected draft_input type: {type(draft_input).__name__}")
            raw_text = ""
            fw = framework
            ver = version

        return raw_text, fw, ver, meta

    # -------------------------------------------------------------------------
    # Step 7.2: Loading Baseline Obligations
    # -------------------------------------------------------------------------

    async def load_existing_obligations(
        self,
        framework: Optional[str] = None,
        version: Optional[str] = None,
        version_id: Optional[Union[str, UUID]] = None,
        db_session: Optional[AsyncSession] = None,
    ) -> List[Dict[str, Any]]:
        """
        Load baseline regulatory obligations from Neo4j knowledge graph or SQL repository.
        Does not mutate the graph (read-only query).

        :param framework: Framework name/identifier (e.g. 'GDPR', 'SOC 2')
        :param version: Baseline version identifier (e.g. '2016', 'v1')
        :param version_id: Optional UUID of existing regulatory version
        :param db_session: Optional SQLAlchemy AsyncSession for relational fallback
        :return: List of normalized obligation dictionaries
        """
        obligations: List[Dict[str, Any]] = []

        # 1. Query Neo4j knowledge graph if client/service available
        if self.graph_service:
            try:
                query = """
                MATCH (f:RegulatoryFramework)-[:HAS_VERSION]->(v:RegulatoryVersion)-[:CONTAINS]->(o:RegulatoryObligation)
                WHERE ($version_id IS NOT NULL AND (v.id = $version_id OR o.version_id = $version_id))
                   OR (
                       ($framework IS NULL OR toLower(f.name) = toLower($framework))
                       AND ($version IS NULL OR toLower(v.version_slug) = toLower($version))
                   )
                RETURN o.id AS id,
                       o.code AS code,
                       o.clause AS clause,
                       o.title AS title,
                       coalesce(o.description, o.text, o.source_text) AS text,
                       o.category AS category,
                       o.mandatory AS mandatory,
                       o.keywords AS keywords,
                       o.source_text AS source_text,
                       f.name AS framework,
                       v.version_slug AS version,
                       v.id AS version_id
                """
                params = {
                    "version_id": str(version_id) if version_id else None,
                    "framework": framework,
                    "version": version,
                }
                records = await self.graph_service.execute_query(query, parameters=params)
                if records:
                    for rec in records:
                        norm = self._normalize_obligation_item(rec)
                        if norm.get("text") or norm.get("clause") or norm.get("code"):
                            obligations.append(norm)
                    if obligations:
                        logger.info(
                            f"Loaded {len(obligations)} baseline obligations from Neo4j graph "
                            f"(framework='{framework}', version='{version}')."
                        )
                        return obligations
            except Exception as graph_err:
                logger.debug(f"Neo4j query for baseline obligations did not return records ({graph_err}). Trying SQL.")

        # 2. Relational SQL fallback if session provided
        if db_session is not None and RegulatoryRequirementRepository is not None:
            try:
                req_repo = RegulatoryRequirementRepository(db_session)
                ver_repo = RegulatoryVersionRepository(db_session)
                target_ver_id = version_id

                if not target_ver_id and framework and version and RegulatoryFrameworkRepository is not None:
                    fw_repo = RegulatoryFrameworkRepository(db_session)
                    fw_obj = await fw_repo.get_by_name(framework)
                    if fw_obj:
                        ver_obj = await ver_repo.get_by_slug(fw_obj.id, version)
                        if ver_obj:
                            target_ver_id = ver_obj.id

                if target_ver_id:
                    v_uuid = UUID(str(target_ver_id)) if not isinstance(target_ver_id, UUID) else target_ver_id
                    req_list = await req_repo.list_by_version(v_uuid)
                    for r in req_list:
                        obligations.append(self._normalize_obligation_item({
                            "id": str(r.id),
                            "code": r.code,
                            "clause": r.code,
                            "title": r.title,
                            "text": r.description or r.title,
                            "framework": framework,
                            "version": version,
                        }))
                    if obligations:
                        logger.info(f"Loaded {len(obligations)} baseline obligations from SQL repository.")
                        return obligations
            except Exception as sql_err:
                logger.warning(f"Failed to query SQL repository for baseline obligations: {sql_err}")

        return obligations

    # -------------------------------------------------------------------------
    # Step 7.2: Comparing Draft Obligations With Existing Obligations
    # -------------------------------------------------------------------------

    async def compare_obligations(
        self,
        draft_obligations: Sequence[Union[DraftObligation, Dict[str, Any], ExtractedObligation]],
        existing_obligations: Optional[Sequence[Union[Dict[str, Any], DraftObligation, Any]]] = None,
        framework: Optional[str] = None,
        baseline_version: Optional[str] = None,
        draft_version: Optional[str] = None,
        existing_version_id: Optional[Union[str, UUID]] = None,
        db_session: Optional[AsyncSession] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        similarity_threshold: float = 0.65,
        use_llm: bool = True,
    ) -> ObligationComparisonResult:
        """
        Compare baseline regulatory obligations with draft obligations (Phase 2, Step 7.2).

        Detects:
        - ADDED: New obligation in draft with no equivalent baseline obligation
        - MODIFIED: Matched obligation whose regulatory meaning or requirements changed
        - REMOVED: Baseline obligation absent from draft regulation
        - UNCHANGED: Matched obligation with identical or minor wording differences (no regulatory change)

        Priority matching:
        1. Deterministic match by normalized clause identifier or obligation ID.
        2. Semantic similarity matching (embeddings & lexical token overlap) where clauses changed.

        :param draft_obligations: Newly extracted draft obligations
        :param existing_obligations: Baseline obligations (if None, loaded from graph/SQL)
        :param framework: Framework name (e.g. 'GDPR', 'SOC 2')
        :param baseline_version: Baseline version identifier (e.g. '2016')
        :param draft_version: Draft version identifier (e.g. '2024-draft')
        :param existing_version_id: UUID of baseline version
        :param db_session: Optional SQLAlchemy session for loading baseline obligations
        :param provider: LLM provider override ('groq' or 'gemini')
        :param model: LLM model identifier
        :param similarity_threshold: Minimum semantic score to pair unmatched obligations
        :param use_llm: If True, uses configured LLM for subtle meaning comparisons
        :return: ObligationComparisonResult containing list of classified ObligationComparisonItem objects
        """
        # 1. Normalize draft obligations
        norm_draft = [self._normalize_obligation_item(d) for d in draft_obligations]

        # 2. Resolve baseline obligations
        if existing_obligations is not None:
            norm_existing = [self._normalize_obligation_item(e) for e in existing_obligations]
        else:
            loaded = await self.load_existing_obligations(
                framework=framework,
                version=baseline_version,
                version_id=existing_version_id,
                db_session=db_session,
            )
            norm_existing = loaded

        # Resolve framework and version metadata if missing
        if not framework:
            framework = (
                (norm_draft[0].get("framework") if norm_draft else None)
                or (norm_existing[0].get("framework") if norm_existing else None)
            )
        if not draft_version and norm_draft:
            draft_version = norm_draft[0].get("version")
        if not baseline_version and norm_existing:
            baseline_version = norm_existing[0].get("version")

        # 3. Initialize tracking structures
        unmatched_existing_indices = set(range(len(norm_existing)))
        unmatched_draft_indices = set(range(len(norm_draft)))

        # Tuple: (existing_index, draft_index, match_method, similarity_score)
        matched_pairs: List[Tuple[int, int, str, float]] = []

        # 4. Priority 1: Deterministic Match by Clause or ID
        # First index existing obligations by normalized clause and id
        clause_to_existing: Dict[str, List[int]] = {}
        id_to_existing: Dict[str, int] = {}

        for e_idx, e_ob in enumerate(norm_existing):
            c_key = normalize_clause_key(e_ob.get("clause") or e_ob.get("code"))
            if c_key:
                clause_to_existing.setdefault(c_key, []).append(e_idx)
            id_key = normalize_id_key(e_ob.get("id"))
            if id_key:
                id_to_existing[id_key] = e_idx

        # Attempt deterministic matching for each draft obligation
        for d_idx in sorted(list(unmatched_draft_indices)):
            d_ob = norm_draft[d_idx]
            d_clause_key = normalize_clause_key(d_ob.get("clause") or d_ob.get("code"))
            d_id_key = normalize_id_key(d_ob.get("id"))

            matched_e_idx: Optional[int] = None
            match_method = "deterministic_clause"

            # Check ID match first
            if d_id_key and d_id_key in id_to_existing:
                cand_e_idx = id_to_existing[d_id_key]
                if cand_e_idx in unmatched_existing_indices:
                    matched_e_idx = cand_e_idx
                    match_method = "deterministic_id"

            # Check Clause match
            if matched_e_idx is None and d_clause_key and d_clause_key in clause_to_existing:
                available_candidates = [
                    idx for idx in clause_to_existing[d_clause_key]
                    if idx in unmatched_existing_indices
                ]
                if available_candidates:
                    # If multiple candidates share clause, pick highest text similarity
                    if len(available_candidates) == 1:
                        matched_e_idx = available_candidates[0]
                    else:
                        best_sim = -1.0
                        best_idx = available_candidates[0]
                        for c_idx in available_candidates:
                            sim = self._compute_text_similarity(
                                norm_existing[c_idx]["text"],
                                d_ob["text"],
                            )
                            if sim > best_sim:
                                best_sim = sim
                                best_idx = c_idx
                        matched_e_idx = best_idx
                    match_method = "deterministic_clause"

            if matched_e_idx is not None:
                matched_pairs.append((matched_e_idx, d_idx, match_method, 1.0))
                unmatched_existing_indices.remove(matched_e_idx)
                unmatched_draft_indices.remove(d_idx)

        # 5. Priority 2: Semantic Similarity Matching for remaining unmatched
        if unmatched_existing_indices and unmatched_draft_indices:
            candidate_similarities: List[Tuple[float, int, int]] = []

            for e_idx in unmatched_existing_indices:
                for d_idx in unmatched_draft_indices:
                    sim = await self.compute_obligation_similarity(
                        norm_existing[e_idx],
                        norm_draft[d_idx],
                    )
                    if sim >= similarity_threshold:
                        candidate_similarities.append((sim, e_idx, d_idx))

            # Sort candidate pairs by similarity descending (greedy bipartite matching)
            candidate_similarities.sort(key=lambda x: x[0], reverse=True)

            for sim_score, e_idx, d_idx in candidate_similarities:
                if e_idx in unmatched_existing_indices and d_idx in unmatched_draft_indices:
                    matched_pairs.append((e_idx, d_idx, "semantic_similarity", sim_score))
                    unmatched_existing_indices.remove(e_idx)
                    unmatched_draft_indices.remove(d_idx)

        # 6. Change Classification for Matched Pairs (MODIFIED vs UNCHANGED)
        changes: List[ObligationComparisonItem] = []

        for e_idx, d_idx, match_method, sim_score in matched_pairs:
            old_ob = norm_existing[e_idx]
            new_ob = norm_draft[d_idx]
            old_text = old_ob.get("text") or ""
            new_text = new_ob.get("text") or ""

            # Check if LLM evaluation should be used
            llm_result = None
            if use_llm:
                llm_result = await self._evaluate_change_with_llm(
                    old_ob=old_ob,
                    new_ob=new_ob,
                    provider=provider,
                    model=model,
                )

            if llm_result is not None:
                is_modified, reason, confidence = llm_result
            else:
                # Deterministic fallback evaluation
                is_modified, reason, confidence = self._detect_substantive_differences(
                    old_text=old_text,
                    new_text=new_text,
                    old_ob=old_ob,
                    new_ob=new_ob,
                )

            change_type = ObligationChangeType.MODIFIED if is_modified else ObligationChangeType.UNCHANGED

            changes.append(
                ObligationComparisonItem(
                    change_type=change_type,
                    old_obligation_id=old_ob.get("id"),
                    old_clause=old_ob.get("clause") or old_ob.get("code"),
                    old_text=old_text,
                    new_obligation_id=new_ob.get("id"),
                    new_clause=new_ob.get("clause") or new_ob.get("code"),
                    new_text=new_text,
                    category=new_ob.get("category") or old_ob.get("category"),
                    similarity_score=sim_score,
                    reason=reason,
                    confidence=confidence,
                    metadata={
                        "matching_method": match_method,
                        "similarity_score": sim_score,
                        "baseline_version": baseline_version,
                        "draft_version": draft_version,
                    },
                )
            )

        # 7. Unmatched Draft Obligations -> ADDED
        for d_idx in sorted(list(unmatched_draft_indices)):
            new_ob = norm_draft[d_idx]
            clause_name = new_ob.get("clause") or new_ob.get("code") or "draft requirement"
            changes.append(
                ObligationComparisonItem(
                    change_type=ObligationChangeType.ADDED,
                    old_obligation_id=None,
                    old_clause=None,
                    old_text=None,
                    new_obligation_id=new_ob.get("id"),
                    new_clause=new_ob.get("clause") or new_ob.get("code"),
                    new_text=new_ob.get("text"),
                    category=new_ob.get("category"),
                    similarity_score=0.0,
                    reason=f"New obligation introduced for '{clause_name}' with no matching baseline requirement.",
                    confidence=0.95,
                    metadata={
                        "matching_method": "unmatched_draft",
                        "draft_version": draft_version,
                    },
                )
            )

        # 8. Unmatched Baseline Obligations -> REMOVED
        for e_idx in sorted(list(unmatched_existing_indices)):
            old_ob = norm_existing[e_idx]
            clause_name = old_ob.get("clause") or old_ob.get("code") or "baseline requirement"
            changes.append(
                ObligationComparisonItem(
                    change_type=ObligationChangeType.REMOVED,
                    old_obligation_id=old_ob.get("id"),
                    old_clause=old_ob.get("clause") or old_ob.get("code"),
                    old_text=old_ob.get("text"),
                    new_obligation_id=None,
                    new_clause=None,
                    new_text=None,
                    category=old_ob.get("category"),
                    similarity_score=0.0,
                    reason=f"Baseline obligation '{clause_name}' has no matching requirement in draft regulation and was removed.",
                    confidence=0.95,
                    metadata={
                        "matching_method": "unmatched_baseline",
                        "baseline_version": baseline_version,
                    },
                )
            )

        # 9. Build Summary Metrics
        summary = {
            "ADDED": sum(1 for c in changes if c.change_type == ObligationChangeType.ADDED),
            "MODIFIED": sum(1 for c in changes if c.change_type == ObligationChangeType.MODIFIED),
            "REMOVED": sum(1 for c in changes if c.change_type == ObligationChangeType.REMOVED),
            "UNCHANGED": sum(1 for c in changes if c.change_type == ObligationChangeType.UNCHANGED),
            "TOTAL": len(changes),
        }

        logger.info(
            f"Comparison completed: {summary['TOTAL']} total changes "
            f"({summary['ADDED']} added, {summary['MODIFIED']} modified, "
            f"{summary['REMOVED']} removed, {summary['UNCHANGED']} unchanged)."
        )

        return ObligationComparisonResult(
            framework=framework,
            baseline_version=baseline_version,
            draft_version=draft_version,
            changes=changes,
            total_changes=len(changes),
            summary=summary,
            metadata={
                "total_draft_obligations": len(norm_draft),
                "total_baseline_obligations": len(norm_existing),
                "matched_pairs_count": len(matched_pairs),
                "deterministic_matches": sum(1 for _, _, m, _ in matched_pairs if m.startswith("deterministic")),
                "semantic_matches": sum(1 for _, _, m, _ in matched_pairs if m == "semantic_similarity"),
                "similarity_threshold": similarity_threshold,
            },
        )

    # -------------------------------------------------------------------------
    # High-level Step 7.2 Entrypoint: compare_with_existing_obligations
    # -------------------------------------------------------------------------

    async def compare_with_existing_obligations(
        self,
        draft_input: Union[str, DraftRegulationInput, DraftExtractionResult, Sequence[Union[DraftObligation, Dict[str, Any]]]],
        existing_obligations: Optional[Sequence[Union[Dict[str, Any], DraftObligation, Any]]] = None,
        framework: Optional[str] = None,
        baseline_version: Optional[str] = None,
        draft_version: Optional[str] = None,
        existing_version_id: Optional[Union[str, UUID]] = None,
        db_session: Optional[AsyncSession] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        similarity_threshold: float = 0.65,
        use_llm: bool = True,
    ) -> ObligationComparisonResult:
        """
        High-level entrypoint for Phase 2 Step 7.2.
        Accepts raw draft text, a DraftRegulationInput, a DraftExtractionResult, or obligation list.
        Extracts draft obligations if necessary, retrieves baseline obligations if not supplied,
        and performs comparison and change classification.

        :param draft_input: Raw draft text, DraftRegulationInput, DraftExtractionResult, or obligation list
        :param existing_obligations: Optional baseline obligations list
        :param framework: Framework name (e.g. 'GDPR', 'SOC 2')
        :param baseline_version: Baseline version identifier (e.g. '2016')
        :param draft_version: Draft version identifier (e.g. '2024-draft')
        :param existing_version_id: UUID of baseline version
        :param db_session: Optional SQLAlchemy session
        :param provider: LLM provider override
        :param model: LLM model override
        :param similarity_threshold: Minimum semantic matching threshold
        :param use_llm: Whether to invoke LLM for subtle meaning comparison
        :return: ObligationComparisonResult
        """
        # Resolve draft obligations from draft_input
        draft_obs: List[Union[DraftObligation, Dict[str, Any]]] = []

        if isinstance(draft_input, DraftExtractionResult):
            draft_obs = list(draft_input.obligations)
            framework = framework or draft_input.framework
            draft_version = draft_version or draft_input.version
        elif isinstance(draft_input, (str, DraftRegulationInput)):
            extraction_result = await self.extract_draft_obligations(
                draft_input=draft_input,
                framework=framework,
                version=draft_version,
                provider=provider,
                model=model,
            )
            draft_obs = list(extraction_result.obligations)
            framework = framework or extraction_result.framework
            draft_version = draft_version or extraction_result.version
        elif isinstance(draft_input, (list, tuple)):
            draft_obs = list(draft_input)
        else:
            logger.warning(f"Unsupported draft_input type: {type(draft_input).__name__}")
            draft_obs = []

        return await self.compare_obligations(
            draft_obligations=draft_obs,
            existing_obligations=existing_obligations,
            framework=framework,
            baseline_version=baseline_version,
            draft_version=draft_version,
            existing_version_id=existing_version_id,
            db_session=db_session,
            provider=provider,
            model=model,
            similarity_threshold=similarity_threshold,
            use_llm=use_llm,
        )

    # -------------------------------------------------------------------------
    # Evaluation Helpers & Scoring
    # -------------------------------------------------------------------------

    def _normalize_obligation_item(self, item: Any) -> Dict[str, Any]:
        """Normalize various obligation models/dicts into a unified dictionary structure."""
        if isinstance(item, dict):
            ob_id = item.get("id") or item.get("obligation_id") or item.get("code")
            clause = item.get("clause") or item.get("code") or ""
            code = item.get("code") or item.get("clause") or ""
            title = item.get("title") or ""
            text = (
                item.get("text")
                or item.get("description")
                or item.get("source_text")
                or item.get("statement")
                or title
                or ""
            )
            category = item.get("category") or item.get("domain") or "General"
            mandatory = item.get("mandatory", True)
            keywords = item.get("keywords") or []
            framework = item.get("framework")
            version = item.get("version")
        else:
            ob_id = getattr(item, "id", None) or getattr(item, "code", None)
            clause = getattr(item, "clause", None) or getattr(item, "code", None) or ""
            code = getattr(item, "code", None) or getattr(item, "clause", None) or ""
            title = getattr(item, "title", None) or ""
            text = (
                getattr(item, "text", None)
                or getattr(item, "description", None)
                or getattr(item, "source_text", None)
                or title
                or ""
            )
            category = getattr(item, "category", None) or "General"
            mandatory = getattr(item, "mandatory", True)
            keywords = getattr(item, "keywords", None) or []
            framework = getattr(item, "framework", None)
            version = getattr(item, "version", None)

        if isinstance(keywords, str):
            keywords = [k.strip() for k in keywords.split(",") if k.strip()]
        elif not isinstance(keywords, list):
            keywords = list(keywords or [])

        return {
            "id": str(ob_id) if ob_id is not None else None,
            "clause": str(clause).strip() if clause else "",
            "code": str(code).strip() if code else "",
            "title": str(title).strip() if title else "",
            "text": str(text).strip(),
            "category": str(category).strip() if category else "General",
            "mandatory": bool(mandatory) if mandatory is not None else True,
            "keywords": [str(k).strip() for k in keywords if str(k).strip()],
            "framework": str(framework) if framework else None,
            "version": str(version) if version else None,
            "raw": item,
        }

    def _compute_text_similarity(self, text_a: str, text_b: str) -> float:
        """Compute lexical SequenceMatcher ratio between two texts."""
        if not text_a or not text_b:
            return 0.0
        return difflib.SequenceMatcher(None, text_a.lower().strip(), text_b.lower().strip()).ratio()

    def _compute_token_jaccard(self, text_a: str, text_b: str) -> float:
        """Compute token-level Jaccard overlap similarity excluding common stopwords."""
        stopwords = {
            "a", "an", "the", "and", "or", "of", "in", "to", "for", "with", "on", "at",
            "from", "by", "is", "are", "was", "were", "be", "been", "being", "that",
            "this", "these", "those", "which", "shall", "must", "should", "as", "it",
        }
        tokens_a = set(re.findall(r"\b[a-z0-9]+\b", text_a.lower())) - stopwords
        tokens_b = set(re.findall(r"\b[a-z0-9]+\b", text_b.lower())) - stopwords
        if not tokens_a or not tokens_b:
            return 0.0
        intersection = tokens_a.intersection(tokens_b)
        union = tokens_a.union(tokens_b)
        return len(intersection) / len(union)

    async def _compute_embedding_similarity(self, text_a: str, text_b: str) -> Optional[float]:
        """Compute cosine similarity using Qdrant client embedding generation."""
        if not self.qdrant_client:
            return None
        try:
            vecs = await self.qdrant_client.generate_embeddings_batch([text_a, text_b])
            if len(vecs) == 2:
                v1, v2 = vecs[0], vecs[1]
                dot = sum(x * y for x, y in zip(v1, v2))
                norm1 = math.sqrt(sum(x * x for x in v1))
                norm2 = math.sqrt(sum(y * y for y in v2))
                if norm1 > 0 and norm2 > 0:
                    return max(0.0, min(1.0, dot / (norm1 * norm2)))
        except Exception as e:
            logger.debug(f"Embedding similarity computation skipped: {e}")
        return None

    async def compute_obligation_similarity(
        self,
        ob_a: Dict[str, Any],
        ob_b: Dict[str, Any],
    ) -> float:
        """
        Compute composite semantic and lexical similarity score between two obligations.
        Used when exact clause/ID matching is unavailable.
        """
        text_a = ob_a.get("text") or ob_a.get("title") or ""
        text_b = ob_b.get("text") or ob_b.get("title") or ""

        seq_sim = self._compute_text_similarity(text_a, text_b)
        tok_sim = self._compute_token_jaccard(text_a, text_b)

        emb_sim = await self._compute_embedding_similarity(text_a, text_b)
        if emb_sim is not None:
            text_score = 0.5 * emb_sim + 0.3 * seq_sim + 0.2 * tok_sim
        else:
            text_score = 0.6 * seq_sim + 0.4 * tok_sim

        # Metadata boosts
        boost = 0.0
        cat_a = (ob_a.get("category") or "").strip().lower()
        cat_b = (ob_b.get("category") or "").strip().lower()
        if cat_a and cat_b and cat_a == cat_b:
            boost += 0.10

        kw_a = set(k.lower() for k in ob_a.get("keywords") or [])
        kw_b = set(k.lower() for k in ob_b.get("keywords") or [])
        if kw_a and kw_b:
            kw_overlap = len(kw_a.intersection(kw_b)) / max(len(kw_a.union(kw_b)), 1)
            boost += 0.10 * kw_overlap

        c_a = normalize_clause_key(ob_a.get("clause") or ob_a.get("code"))
        c_b = normalize_clause_key(ob_b.get("clause") or ob_b.get("code"))
        if c_a and c_b and (c_a in c_b or c_b in c_a):
            boost += 0.10

        return min(1.0, round(text_score + boost, 4))

    def _detect_substantive_differences(
        self,
        old_text: str,
        new_text: str,
        old_ob: Dict[str, Any],
        new_ob: Dict[str, Any],
    ) -> Tuple[bool, str, float]:
        """
        Evaluate substantive differences between baseline and draft obligation texts.
        Avoids marking minor wording or stylistic updates as major modifications.

        Returns (is_modified, reason, confidence).
        """
        ot = (old_text or "").strip()
        nt = (new_text or "").strip()

        # 1. Exact string match
        if ot.lower() == nt.lower():
            return (
                False,
                "Obligation text and regulatory requirements are identical.",
                1.0,
            )

        # 2. Punctuation and whitespace cleaned match
        clean_ot = re.sub(r"[^a-z0-9]", "", ot.lower())
        clean_nt = re.sub(r"[^a-z0-9]", "", nt.lower())
        if clean_ot == clean_nt:
            return (
                False,
                "Identical obligation requirement with minor formatting or punctuation differences.",
                0.99,
            )

        # 3. Timeframe / Retention change (e.g. general necessity -> 12 months)
        has_old_necessity = bool(
            re.search(
                r"\b(?:as long as necessary|no longer than is? necessary|only be retained as long as|for necessary period)\b",
                ot,
                re.I,
            )
        )
        new_time_match = re.search(
            r"\b(?:within|maximum of|up to|no longer than)?\s*(\d+\s*(?:months?|days?|hours?|years?))\b",
            nt,
            re.I,
        )
        if has_old_necessity and new_time_match:
            time_str = new_time_match.group(1).strip()
            return (
                True,
                f"Retention requirement changed from general necessity to a {time_str} maximum.",
                0.94,
            )

        # 4. Notification timeframe change (e.g. general expediency -> 48 hours)
        has_old_undue_delay = bool(
            re.search(r"\b(?:without undue delay|promptly|timely|expeditiously)\b", ot, re.I)
        )
        new_hours_match = re.search(
            r"\b(?:within|no later than)\s*(\d+\s*hours?)\b",
            nt,
            re.I,
        )
        if (has_old_undue_delay or "notif" in ot.lower()) and new_hours_match:
            hours_str = new_hours_match.group(1).strip()
            return (
                True,
                f"Notification requirement changed from general expediency to a strict {hours_str} deadline.",
                0.94,
            )

        # 5. General timeframe differences
        old_timeframes = set(
            re.findall(r"\b\d+\s*(?:days?|hours?|months?|years?|weeks?)\b", ot.lower())
        )
        new_timeframes = set(
            re.findall(r"\b\d+\s*(?:days?|hours?|months?|years?|weeks?)\b", nt.lower())
        )
        if old_timeframes != new_timeframes and (old_timeframes or new_timeframes):
            added_tf = new_timeframes - old_timeframes
            removed_tf = old_timeframes - new_timeframes
            if added_tf and removed_tf:
                return (
                    True,
                    f"Compliance timeframe changed from {', '.join(sorted(removed_tf))} to {', '.join(sorted(added_tf))}.",
                    0.94,
                )
            elif added_tf:
                return (
                    True,
                    f"Strict compliance deadline ({', '.join(sorted(added_tf))}) introduced into requirement.",
                    0.94,
                )
            else:
                return (
                    True,
                    f"Specific timeframe constraint ({', '.join(sorted(removed_tf))}) removed from requirement.",
                    0.92,
                )

        # 6. Specific technical controls added
        control_phrases = [
            ("automated purging", "Automated purging procedures added to data lifecycle requirements."),
            ("anonymised", "Irreversible anonymisation requirement introduced upon retention expiry."),
            ("anonymized", "Irreversible anonymization requirement introduced upon retention expiry."),
            ("encrypted", "Encryption requirement introduced for protected data."),
            ("independent audit", "Independent third-party audit requirement added."),
            ("multi-factor", "Multi-factor authentication mandate introduced."),
        ]
        for phrase, reason_desc in control_phrases:
            if phrase in nt.lower() and phrase not in ot.lower():
                return (True, reason_desc, 0.93)

        # 7. Modality changes (mandatory vs advisory)
        old_modality = self._extract_modality(ot, old_ob.get("mandatory"))
        new_modality = self._extract_modality(nt, new_ob.get("mandatory"))
        if old_modality != new_modality:
            if old_modality == "advisory" and new_modality == "mandatory":
                return (
                    True,
                    "Requirement elevated from advisory recommendation to mandatory regulatory obligation.",
                    0.93,
                )
            elif old_modality == "mandatory" and new_modality == "advisory":
                return (
                    True,
                    "Requirement relaxed from mandatory obligation to optional guidance.",
                    0.93,
                )
            elif new_modality == "prohibition":
                return (
                    True,
                    "Strict prohibition constraint introduced into obligation.",
                    0.93,
                )

        # 8. High lexical similarity (minor stylistic / wording differences)
        seq_ratio = difflib.SequenceMatcher(None, ot.lower(), nt.lower()).ratio()
        if seq_ratio >= 0.85:
            tokens_old = set(re.findall(r"\b[a-z0-9]+\b", ot.lower()))
            tokens_new = set(re.findall(r"\b[a-z0-9]+\b", nt.lower()))
            diff_tokens = (tokens_old ^ tokens_new)
            stylistic_words = {
                "shall", "must", "be", "in", "a", "manner", "transparent", "transparently",
                "organisational", "organizational", "whilst", "while", "that", "which", "the",
                "an", "such", "this", "these", "subject", "every", "all", "reasonable", "measures",
            }
            if diff_tokens.issubset(stylistic_words) or len(diff_tokens) <= 3:
                return (
                    False,
                    "Minor wording/stylistic update with no substantive change to regulatory requirement.",
                    0.94,
                )

        # 9. Meaning changed significantly
        if seq_ratio < 0.85:
            ot_snippet = ot[:60] + "..." if len(ot) > 60 else ot
            nt_snippet = nt[:60] + "..." if len(nt) > 60 else nt
            return (
                True,
                f"Requirement text modified from '{ot_snippet}' to '{nt_snippet}' altering compliance criteria.",
                0.90,
            )

        return (
            False,
            "Minor wording/stylistic variation with no substantive change to regulatory requirement.",
            0.92,
        )

    def _extract_modality(self, text: str, mandatory_flag: Optional[bool]) -> str:
        """Helper to extract obligation modality (mandatory, advisory, prohibition)."""
        t = text.lower()
        if re.search(r"\b(?:shall not|must not|prohibited|forbidden|never)\b", t):
            return "prohibition"
        if re.search(r"\b(?:shall|must|required|mandatory|obliged)\b", t):
            return "mandatory"
        if re.search(r"\b(?:should|may|recommended|optional|guidance)\b", t):
            return "advisory"
        if mandatory_flag is True:
            return "mandatory"
        if mandatory_flag is False:
            return "advisory"
        return "mandatory"

    # -------------------------------------------------------------------------
    # LLM Meaning Comparison
    # -------------------------------------------------------------------------

    def load_compare_prompt_template(self) -> str:
        """Load obligation comparison prompt template from file or embedded fallback."""
        if self._prompt_template is None:
            if self.prompt_path.exists():
                self._prompt_template = self.prompt_path.read_text(encoding="utf-8")
            else:
                self._prompt_template = """
You are an expert regulatory compliance auditor and legal engineering AI.
Compare the baseline obligation with the draft obligation:

Baseline Obligation:
- ID: {old_id}
- Clause: {old_clause}
- Category: {old_category}
- Requirement: {old_text}

Draft Obligation:
- ID: {new_id}
- Clause: {new_clause}
- Category: {new_category}
- Requirement: {new_text}

Determine if the regulatory meaning has substantively changed:
- If only minor wording, punctuation, or stylistic changes occurred with NO change to regulatory obligations, classify as "UNCHANGED".
- If requirements, deadlines, scope, mandatory nature, or technical controls changed, classify as "MODIFIED".

Return valid JSON:
{
  "change_type": "MODIFIED" | "UNCHANGED",
  "reason": "Specific explanation of what changed or why it is functionally unchanged",
  "confidence": 0.94
}
""".strip()
        return self._prompt_template

    async def _evaluate_change_with_llm(
        self,
        old_ob: Dict[str, Any],
        new_ob: Dict[str, Any],
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ) -> Optional[Tuple[bool, str, float]]:
        """
        Use the configured LLM to carefully evaluate if the regulatory meaning changed.
        Returns (is_modified, reason, confidence) if successful, None otherwise.
        """
        target_provider = (provider or self.provider).lower().strip()
        has_key = (
            (target_provider == "groq" and self.groq_api_key and not self.groq_api_key.startswith("your-"))
            or (target_provider in ("gemini", "google") and self.gemini_api_key and not self.gemini_api_key.startswith("your-"))
        )
        if not has_key:
            return None

        template = self.load_compare_prompt_template()
        prompt = (
            template.replace("{old_id}", str(old_ob.get("id") or ""))
            .replace("{old_clause}", str(old_ob.get("clause") or old_ob.get("code") or ""))
            .replace("{old_category}", str(old_ob.get("category") or ""))
            .replace("{old_text}", str(old_ob.get("text") or ""))
            .replace("{new_id}", str(new_ob.get("id") or ""))
            .replace("{new_clause}", str(new_ob.get("clause") or ""))
            .replace("{new_category}", str(new_ob.get("category") or ""))
            .replace("{new_text}", str(new_ob.get("text") or ""))
        )

        try:
            if target_provider == "groq":
                raw_response = await self._call_groq(prompt, model=model or DEFAULT_GROQ_MODEL)
            else:
                raw_response = await self._call_gemini(prompt, model=model or DEFAULT_GEMINI_MODEL)

            parsed = self._extract_json_block(raw_response)
            if not isinstance(parsed, dict):
                return None

            change_type_str = str(parsed.get("change_type", "")).strip().upper()
            if change_type_str not in ("MODIFIED", "UNCHANGED"):
                return None

            is_mod = (change_type_str == "MODIFIED")
            reason = str(parsed.get("reason") or parsed.get("reasoning") or "").strip()
            confidence = float(parsed.get("confidence") or 0.90)
            confidence = max(0.0, min(1.0, confidence))

            return (is_mod, reason, confidence)
        except Exception as e:
            logger.debug(f"LLM comparison failed, will use deterministic evaluation: {e}")
            return None

    async def _call_groq(self, prompt: str, model: str) -> str:
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.groq_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 500,
        }
        client = self._http_client or httpx.AsyncClient(timeout=30.0)
        close_client = self._http_client is None
        try:
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        finally:
            if close_client:
                await client.aclose()

    async def _call_gemini(self, prompt: str, model: str) -> str:
        clean_model = model if model.startswith("models/") else f"models/{model}"
        url = f"https://generativelanguage.googleapis.com/v1beta/{clean_model}:generateContent?key={self.gemini_api_key}"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.1, "maxOutputTokens": 500},
        }
        client = self._http_client or httpx.AsyncClient(timeout=30.0)
        close_client = self._http_client is None
        try:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data["candidates"][0]["content"]["parts"][0]["text"]
        finally:
            if close_client:
                await client.aclose()

    def _extract_json_block(self, text: str) -> Any:
        cleaned = text.strip()
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned, re.IGNORECASE)
        if match:
            cleaned = match.group(1).strip()
        return json.loads(cleaned)

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


async def compare_obligations(
    draft_obligations: Sequence[Union[DraftObligation, Dict[str, Any], ExtractedObligation]],
    existing_obligations: Optional[Sequence[Union[Dict[str, Any], DraftObligation, Any]]] = None,
    framework: Optional[str] = None,
    baseline_version: Optional[str] = None,
    draft_version: Optional[str] = None,
    existing_version_id: Optional[Union[str, UUID]] = None,
    db_session: Optional[AsyncSession] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    similarity_threshold: float = 0.65,
    use_llm: bool = True,
) -> ObligationComparisonResult:
    """
    Convenience function comparing draft obligations with baseline obligations
    using the global impact_analysis_service (Phase 2, Step 7.2).
    """
    return await impact_analysis_service.compare_obligations(
        draft_obligations=draft_obligations,
        existing_obligations=existing_obligations,
        framework=framework,
        baseline_version=baseline_version,
        draft_version=draft_version,
        existing_version_id=existing_version_id,
        db_session=db_session,
        provider=provider,
        model=model,
        similarity_threshold=similarity_threshold,
        use_llm=use_llm,
    )


async def compare_with_existing_obligations(
    draft_input: Union[str, DraftRegulationInput, DraftExtractionResult, Sequence[Union[DraftObligation, Dict[str, Any]]]],
    existing_obligations: Optional[Sequence[Union[Dict[str, Any], DraftObligation, Any]]] = None,
    framework: Optional[str] = None,
    baseline_version: Optional[str] = None,
    draft_version: Optional[str] = None,
    existing_version_id: Optional[Union[str, UUID]] = None,
    db_session: Optional[AsyncSession] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    similarity_threshold: float = 0.65,
    use_llm: bool = True,
) -> ObligationComparisonResult:
    """
    High-level convenience entrypoint comparing draft regulations with baseline obligations
    using the global impact_analysis_service (Phase 2, Step 7.2).
    """
    return await impact_analysis_service.compare_with_existing_obligations(
        draft_input=draft_input,
        existing_obligations=existing_obligations,
        framework=framework,
        baseline_version=baseline_version,
        draft_version=draft_version,
        existing_version_id=existing_version_id,
        db_session=db_session,
        provider=provider,
        model=model,
        similarity_threshold=similarity_threshold,
        use_llm=use_llm,
    )


async def load_existing_obligations(
    framework: Optional[str] = None,
    version: Optional[str] = None,
    version_id: Optional[Union[str, UUID]] = None,
    db_session: Optional[AsyncSession] = None,
) -> List[Dict[str, Any]]:
    """
    Convenience function loading existing obligations from Neo4j/SQL
    using the global impact_analysis_service.
    """
    return await impact_analysis_service.load_existing_obligations(
        framework=framework,
        version=version,
        version_id=version_id,
        db_session=db_session,
    )
