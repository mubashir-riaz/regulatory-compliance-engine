"""
Qdrant Vector Database Integration Client (Phase 2, Step 4.2).

Provides:
1. Connection management to the configured Qdrant vector database instance.
2. Automated collection creation and schema initialization (vectors and payload indexes).
3. Text embedding generation for regulatory obligations using configured providers
   (Google Gemini text-embedding-004, FastEmbed, or deterministic local fallback).
4. Vector and structured metadata payload storage for regulatory obligations
   (obligation ID, framework, version, clause, category, title, text, keywords, etc.).
5. Semantic similarity search with optional metadata filtering (by framework, version, category).
"""

import hashlib
import logging
import math
import os
import uuid
from typing import Any, Dict, List, Optional, Sequence, Union
from uuid import UUID

import httpx
from qdrant_client import AsyncQdrantClient, models

from app.core.config import settings

logger = logging.getLogger(__name__)

# Default configurations
DEFAULT_COLLECTION_NAME = getattr(settings, "QDRANT_COLLECTION", "regulatory_obligations")
DEFAULT_EMBEDDING_MODEL = getattr(settings, "EMBEDDING_MODEL", "text-embedding-004")
DEFAULT_VECTOR_SIZE = 768  # Standard dimension for Google Gemini text-embedding-004
DEFAULT_DISTANCE = models.Distance.COSINE


class QdrantIntegrationError(Exception):
    """Base exception for Qdrant integration and embedding errors."""
    pass


class EmbeddingGenerationError(QdrantIntegrationError):
    """Raised when embedding generation fails."""
    pass


class QdrantClient:
    """
    Client for managing Qdrant vector database operations, embedding generation,
    vector storage, and similarity search for regulatory obligations.
    """

    def __init__(
        self,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        collection_name: Optional[str] = None,
        embedding_provider: Optional[str] = None,
        embedding_model: Optional[str] = None,
        client: Optional[AsyncQdrantClient] = None,
        http_client: Optional[httpx.AsyncClient] = None,
    ):
        """
        Initialize Qdrant client integration.

        :param url: Qdrant server URL (defaults to settings.QDRANT_URL or env)
        :param api_key: Qdrant API key (defaults to settings.QDRANT_API_KEY or env)
        :param collection_name: Target vector collection name
        :param embedding_provider: Embedding provider ("gemini", "fastembed", "local")
        :param embedding_model: Target embedding model identifier
        :param client: Optional pre-configured AsyncQdrantClient instance
        :param http_client: Optional pre-configured httpx.AsyncClient instance
        """
        self.url = (
            url
            or os.getenv("QDRANT_URL")
            or getattr(settings, "QDRANT_URL", "http://qdrant:6333")
        )
        self.api_key = (
            api_key
            or os.getenv("QDRANT_API_KEY")
            or getattr(settings, "QDRANT_API_KEY", None)
        )
        self.collection_name = (
            collection_name
            or getattr(settings, "QDRANT_COLLECTION", DEFAULT_COLLECTION_NAME)
        )
        self.embedding_provider = (
            embedding_provider
            or getattr(settings, "EMBEDDING_PROVIDER", None)
            or getattr(settings, "LLM_PROVIDER", "gemini")
        ).lower().strip()
        self.embedding_model = (
            embedding_model
            or getattr(settings, "EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
        )
        self.gemini_api_key = getattr(settings, "GEMINI_API_KEY", None)

        self._client = client
        self._http_client = http_client

    # -------------------------------------------------------------------------
    # Connection Lifecycle
    # -------------------------------------------------------------------------

    def get_client(self) -> AsyncQdrantClient:
        """
        Get or initialize the AsyncQdrantClient instance.
        """
        if not self._client:
            logger.info(f"Connecting to Qdrant at {self.url}...")
            self._client = AsyncQdrantClient(
                url=self.url,
                api_key=self.api_key,
            )
        return self._client

    async def close(self) -> None:
        """
        Close the underlying AsyncQdrantClient connection.
        """
        if self._client:
            logger.info("Closing Qdrant client connection...")
            await self._client.close()
            self._client = None

    async def test_connection(self) -> bool:
        """
        Test connectivity to the Qdrant instance.

        :return: True if connection is successful, raises exception otherwise.
        """
        try:
            client = self.get_client()
            collections = await client.get_collections()
            logger.info(f"Qdrant connection test successful. Collections count: {len(collections.collections)}")
            return True
        except Exception as e:
            logger.error(f"Qdrant connection test failed: {e}")
            raise QdrantIntegrationError(f"Failed to connect to Qdrant at {self.url}: {e}") from e

    # -------------------------------------------------------------------------
    # Collection Management
    # -------------------------------------------------------------------------

    async def ensure_collection(
        self,
        collection_name: Optional[str] = None,
        vector_size: int = DEFAULT_VECTOR_SIZE,
        distance: models.Distance = DEFAULT_DISTANCE,
    ) -> bool:
        """
        Check if the target Qdrant collection exists, and create it if it does not exist.
        Also creates payload indexes for structured metadata querying.

        :param collection_name: Optional collection name override
        :param vector_size: Dimension size of the embedding vector (default: 768)
        :param distance: Vector distance metric (default: Cosine)
        :return: True if collection is ready
        """
        target_collection = collection_name or self.collection_name
        client = self.get_client()

        try:
            exists = await client.collection_exists(collection_name=target_collection)
            if not exists:
                logger.info(
                    f"Creating Qdrant collection '{target_collection}' "
                    f"(size={vector_size}, distance={distance.name})..."
                )
                await client.create_collection(
                    collection_name=target_collection,
                    vectors_config=models.VectorParams(
                        size=vector_size,
                        distance=distance,
                    ),
                )
                logger.info(f"Collection '{target_collection}' successfully created.")

                # Initialize payload indexes for metadata fields to accelerate filtering
                await self._create_payload_indexes(target_collection)
            else:
                logger.debug(f"Qdrant collection '{target_collection}' already exists.")
            return True
        except Exception as e:
            logger.error(f"Error ensuring Qdrant collection '{target_collection}': {e}", exc_info=True)
            raise QdrantIntegrationError(f"Failed to ensure collection '{target_collection}': {e}") from e

    async def _create_payload_indexes(self, collection_name: str) -> None:
        """
        Create payload indexes for frequent filter fields.
        """
        client = self.get_client()
        index_fields = ["obligation_id", "framework", "version", "clause", "category"]
        for field in index_fields:
            try:
                await client.create_payload_index(
                    collection_name=collection_name,
                    field_name=field,
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )
                logger.debug(f"Created payload index for '{field}' in collection '{collection_name}'")
            except Exception as idx_err:
                logger.warning(f"Could not create payload index for '{field}': {idx_err}")

    # -------------------------------------------------------------------------
    # Embedding Generation
    # -------------------------------------------------------------------------

    async def generate_embedding(
        self,
        text: str,
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ) -> List[float]:
        """
        Generate a vector embedding for a single text using the configured provider.

        :param text: Text string to embed
        :param provider: Optional provider override ("gemini", "fastembed", "local")
        :param model: Optional model identifier override
        :return: List of floats representing the embedding vector
        """
        if not text or not text.strip():
            return [0.0] * DEFAULT_VECTOR_SIZE

        results = await self.generate_embeddings_batch(
            texts=[text],
            provider=provider,
            model=model,
        )
        return results[0]

    async def generate_embeddings_batch(
        self,
        texts: List[str],
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ) -> List[List[float]]:
        """
        Generate vector embeddings for a list of texts in batch.

        :param texts: List of text strings to embed
        :param provider: Optional provider override
        :param model: Optional model identifier override
        :return: List of vector embedding lists
        """
        if not texts:
            return []

        target_provider = (provider or self.embedding_provider).lower().strip()
        target_model = model or self.embedding_model

        # 1. Attempt Google Gemini Embeddings if configured
        if target_provider in ("gemini", "google") or self.gemini_api_key:
            if self.gemini_api_key and not self.gemini_api_key.startswith("your-"):
                try:
                    return await self._generate_gemini_embeddings(texts, model=target_model)
                except Exception as gemini_err:
                    logger.warning(
                        f"Gemini embedding generation failed ({gemini_err}). Falling back to local embedding.",
                        exc_info=True,
                    )

        # 2. Attempt FastEmbed if installed and requested
        if target_provider == "fastembed":
            try:
                return self._generate_fastembed_embeddings(texts, model=target_model)
            except Exception as fe_err:
                logger.warning(f"FastEmbed generation failed ({fe_err}). Falling back to local embedding.")

        # 3. Deterministic Local Normalization Fallback
        logger.debug(f"Generating deterministic local embeddings for {len(texts)} item(s).")
        return [self._generate_local_deterministic_embedding(t, dim=DEFAULT_VECTOR_SIZE) for t in texts]

    async def _generate_gemini_embeddings(
        self,
        texts: List[str],
        model: Optional[str] = None,
    ) -> List[List[float]]:
        """
        Generate embeddings via Google Gemini REST API.
        """
        api_key = self.gemini_api_key
        if not api_key:
            raise EmbeddingGenerationError("GEMINI_API_KEY is not set in configuration.")

        target_model = model or DEFAULT_EMBEDDING_MODEL
        # Normalize model name for Gemini API endpoint
        clean_model = target_model if target_model.startswith("models/") else f"models/{target_model}"

        url = f"https://generativelanguage.googleapis.com/v1beta/{clean_model}:batchEmbedContents?key={api_key}"

        requests_payload = [
            {
                "model": clean_model,
                "content": {
                    "parts": [{"text": text[:8000]}]  # truncate overly long chunks to stay within limits
                },
            }
            for text in texts
        ]

        payload = {"requests": requests_payload}

        async with (self._http_client or httpx.AsyncClient(timeout=60.0)) as client:
            response = await client.post(url, json=payload)
            if response.status_code != 200:
                raise EmbeddingGenerationError(
                    f"Gemini Embedding API returned HTTP {response.status_code}: {response.text}"
                )
            data = response.json()
            embeddings_data = data.get("embeddings", [])
            if len(embeddings_data) != len(texts):
                raise EmbeddingGenerationError(
                    f"Expected {len(texts)} embeddings from Gemini, but received {len(embeddings_data)}."
                )
            return [item.get("values", []) for item in embeddings_data]

    def _generate_fastembed_embeddings(
        self,
        texts: List[str],
        model: Optional[str] = None,
    ) -> List[List[float]]:
        """
        Generate embeddings using fastembed library if installed.
        """
        try:
            from fastembed import TextEmbedding
            embedding_model = TextEmbedding(model_name=model or "BAAI/bge-small-en-v1.5")
            embeddings = list(embedding_model.embed(texts))
            return [e.tolist() for e in embeddings]
        except ImportError as err:
            raise EmbeddingGenerationError(f"fastembed package is not installed: {err}") from err

    @staticmethod
    def _generate_local_deterministic_embedding(text: str, dim: int = DEFAULT_VECTOR_SIZE) -> List[float]:
        """
        Generate a deterministic, L2-normalized pseudo-embedding vector for offline/testing scenarios.
        """
        if not text:
            return [0.0] * dim

        vector = [0.0] * dim
        words = text.lower().split()
        for idx, word in enumerate(words):
            h = hashlib.sha256(f"{word}:{idx % 32}".encode("utf-8")).digest()
            for i in range(min(len(h), dim)):
                bucket = (int.from_bytes(h[i : i + 2], "big") if i + 2 <= len(h) else h[i]) % dim
                val = (h[i] - 128) / 128.0
                vector[bucket] += val

        # Normalize vector to unit length (L2 norm) for cosine distance
        norm = math.sqrt(sum(x * x for x in vector))
        if norm > 0:
            vector = [x / norm for x in vector]
        else:
            vector[0] = 1.0
        return vector

    # -------------------------------------------------------------------------
    # Obligation Vector Upsert
    # -------------------------------------------------------------------------

    async def upsert_obligation(
        self,
        obligation_id: Union[str, UUID],
        text: str,
        framework: str,
        version: str,
        clause: str,
        category: Optional[str] = None,
        title: Optional[str] = None,
        mandatory: Optional[bool] = None,
        keywords: Optional[List[str]] = None,
        collection_name: Optional[str] = None,
        vector: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """
        Generate embedding (if not provided) and store an obligation vector with structured metadata in Qdrant.

        Metadata payload stored:
        - obligation_id: Unique string/UUID of the obligation
        - framework: Framework name (e.g. 'SOC 2', 'GDPR')
        - version: Version slug (e.g. '2017', 'v1')
        - clause: Clause identifier (e.g. 'CC6.1', 'Article 5(1)(a)')
        - category: Control or domain category (e.g. 'Access Control')
        - title: Obligation title or requirement header
        - text: Complete requirement text
        - mandatory: Boolean flag indicating if requirement is mandatory
        - keywords: List of keywords

        :return: Result dictionary with point ID and collection info
        """
        target_collection = collection_name or self.collection_name
        await self.ensure_collection(collection_name=target_collection)

        # Generate embedding vector if not supplied
        if vector is None:
            vector = await self.generate_embedding(text)

        point_id = str(obligation_id)
        payload = {
            "obligation_id": point_id,
            "framework": str(framework).strip(),
            "version": str(version).strip(),
            "clause": str(clause).strip(),
            "category": str(category).strip() if category else None,
            "title": str(title).strip() if title else None,
            "text": str(text).strip() if text else None,
            "mandatory": bool(mandatory) if mandatory is not None else True,
            "keywords": list(keywords) if keywords else [],
        }

        point = models.PointStruct(
            id=point_id,
            vector=vector,
            payload=payload,
        )

        client = self.get_client()
        await client.upsert(
            collection_name=target_collection,
            points=[point],
        )

        logger.info(
            f"Stored obligation vector in Qdrant '{target_collection}': id={point_id}, "
            f"framework='{framework}', version='{version}', clause='{clause}'"
        )

        return {
            "success": True,
            "collection": target_collection,
            "obligation_id": point_id,
            "vector_dimension": len(vector),
            "payload": payload,
        }

    async def upsert_obligations_batch(
        self,
        obligations: Sequence[Union[Dict[str, Any], Any]],
        default_framework: Optional[str] = None,
        default_version: Optional[str] = None,
        collection_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Batch generate embeddings and upsert multiple obligations with payload metadata in Qdrant.

        :param obligations: List of obligations (models or dictionaries)
        :param default_framework: Default framework name fallback
        :param default_version: Default version slug fallback
        :param collection_name: Optional collection name override
        :return: Summary dictionary with count of points upserted
        """
        if not obligations:
            return {"success": True, "count": 0, "points": []}

        target_collection = collection_name or self.collection_name
        await self.ensure_collection(collection_name=target_collection)

        # 1. Parse and prepare metadata
        parsed_items: List[Dict[str, Any]] = []
        texts_to_embed: List[str] = []

        for item in obligations:
            if isinstance(item, dict):
                ob_id = item.get("obligation_id") or item.get("id") or str(uuid.uuid4())
                clause = item.get("clause") or item.get("code") or ""
                text = item.get("text") or item.get("description") or ""
                category = item.get("category")
                title = item.get("title") or (f"[{category}] {clause}" if category else clause)
                fw = item.get("framework") or default_framework or ""
                ver = item.get("version") or default_version or ""
                mandatory = item.get("mandatory", True)
                keywords = item.get("keywords", [])
                vector = item.get("vector")
            else:
                ob_id = getattr(item, "id", None) or getattr(item, "obligation_id", str(uuid.uuid4()))
                clause = getattr(item, "clause", None) or getattr(item, "code", "")
                text = getattr(item, "text", None) or getattr(item, "description", "")
                category = getattr(item, "category", None)
                title = getattr(item, "title", None) or (f"[{category}] {clause}" if category else clause)
                fw = getattr(item, "framework", default_framework or "")
                ver = getattr(item, "version", default_version or "")
                mandatory = getattr(item, "mandatory", True)
                keywords = getattr(item, "keywords", [])
                vector = getattr(item, "vector", None)

            parsed_items.append({
                "obligation_id": str(ob_id),
                "framework": str(fw).strip(),
                "version": str(ver).strip(),
                "clause": str(clause).strip(),
                "category": str(category).strip() if category else None,
                "title": str(title).strip() if title else None,
                "text": str(text).strip() if text else None,
                "mandatory": bool(mandatory) if mandatory is not None else True,
                "keywords": list(keywords) if keywords else [],
                "vector": vector,
            })
            texts_to_embed.append(str(text).strip() if text else str(title))

        # 2. Batch generate embeddings for items without pre-computed vectors
        indices_needing_embeddings = [i for i, item in enumerate(parsed_items) if item["vector"] is None]
        if indices_needing_embeddings:
            sub_texts = [texts_to_embed[i] for i in indices_needing_embeddings]
            generated_vectors = await self.generate_embeddings_batch(sub_texts)
            for idx, vec in zip(indices_needing_embeddings, generated_vectors):
                parsed_items[idx]["vector"] = vec

        # 3. Construct Qdrant PointStruct instances
        points: List[models.PointStruct] = []
        for item in parsed_items:
            payload = {
                "obligation_id": item["obligation_id"],
                "framework": item["framework"],
                "version": item["version"],
                "clause": item["clause"],
                "category": item["category"],
                "title": item["title"],
                "text": item["text"],
                "mandatory": item["mandatory"],
                "keywords": item["keywords"],
            }
            points.append(
                models.PointStruct(
                    id=item["obligation_id"],
                    vector=item["vector"],
                    payload=payload,
                )
            )

        # 4. Upsert batch into Qdrant
        client = self.get_client()
        await client.upsert(
            collection_name=target_collection,
            points=points,
        )

        logger.info(f"Batch upserted {len(points)} obligation vectors into Qdrant collection '{target_collection}'")

        return {
            "success": True,
            "collection": target_collection,
            "count": len(points),
            "obligation_ids": [p.id for p in points],
        }

    # -------------------------------------------------------------------------
    # Similarity Search
    # -------------------------------------------------------------------------

    async def search_similar_obligations(
        self,
        query_text: Optional[str] = None,
        query_vector: Optional[List[float]] = None,
        framework: Optional[str] = None,
        version: Optional[str] = None,
        category: Optional[str] = None,
        limit: int = 5,
        score_threshold: Optional[float] = None,
        collection_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Perform semantic similarity vector search for regulatory obligations in Qdrant.

        Supports:
        - Query by raw text (automatically converted to embedding vector) or pre-computed vector.
        - Metadata filtering by framework, version, and category.
        - Result ranking with similarity scores.

        :param query_text: Query text string to search for
        :param query_vector: Optional direct embedding vector
        :param framework: Optional filter by framework name (e.g. 'SOC 2')
        :param version: Optional filter by version slug (e.g. '2017')
        :param category: Optional filter by category name (e.g. 'Access Control')
        :param limit: Maximum number of results to return (default: 5)
        :param score_threshold: Minimum similarity score threshold
        :param collection_name: Optional collection name override
        :return: List of search result dictionaries containing score and metadata
        """
        target_collection = collection_name or self.collection_name
        await self.ensure_collection(collection_name=target_collection)

        # 1. Resolve query vector
        if query_vector is None:
            if not query_text or not query_text.strip():
                raise ValueError("Either query_text or query_vector must be provided for similarity search.")
            query_vector = await self.generate_embedding(query_text)

        # 2. Build metadata filter conditions
        must_conditions = []
        if framework:
            must_conditions.append(
                models.FieldCondition(
                    key="framework",
                    match=models.MatchValue(value=framework.strip()),
                )
            )
        if version:
            must_conditions.append(
                models.FieldCondition(
                    key="version",
                    match=models.MatchValue(value=version.strip()),
                )
            )
        if category:
            must_conditions.append(
                models.FieldCondition(
                    key="category",
                    match=models.MatchValue(value=category.strip()),
                )
            )

        query_filter = models.Filter(must=must_conditions) if must_conditions else None

        client = self.get_client()

        # 3. Execute vector search query
        # Support modern query_points API with fallback to legacy search if mocked/custom
        if hasattr(client, "query_points"):
            response = await client.query_points(
                collection_name=target_collection,
                query=query_vector,
                query_filter=query_filter,
                limit=limit,
                score_threshold=score_threshold,
                with_payload=True,
            )
            points = response.points if hasattr(response, "points") else response
        elif hasattr(client, "search"):
            points = await client.search(
                collection_name=target_collection,
                query_vector=query_vector,
                query_filter=query_filter,
                limit=limit,
                score_threshold=score_threshold,
                with_payload=True,
            )
        else:
            raise QdrantIntegrationError("AsyncQdrantClient does not support query_points or search methods.")

        # 4. Format and return results
        results: List[Dict[str, Any]] = []
        for pt in points:
            payload = pt.payload or {}
            results.append({
                "obligation_id": payload.get("obligation_id", str(pt.id)),
                "score": float(pt.score) if hasattr(pt, "score") and pt.score is not None else 1.0,
                "framework": payload.get("framework"),
                "version": payload.get("version"),
                "clause": payload.get("clause"),
                "category": payload.get("category"),
                "title": payload.get("title"),
                "text": payload.get("text"),
                "mandatory": payload.get("mandatory"),
                "keywords": payload.get("keywords", []),
                "payload": payload,
            })

        logger.info(f"Similarity search in '{target_collection}' returned {len(results)} matches.")
        return results

    # -------------------------------------------------------------------------
    # Utility Operations
    # -------------------------------------------------------------------------

    async def get_obligation_vector(
        self,
        obligation_id: Union[str, UUID],
        collection_name: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve stored vector and payload for a specific obligation by its ID.
        """
        target_collection = collection_name or self.collection_name
        client = self.get_client()
        point_id = str(obligation_id)

        try:
            records = await client.retrieve(
                collection_name=target_collection,
                ids=[point_id],
                with_payload=True,
                with_vectors=True,
            )
            if records:
                pt = records[0]
                return {
                    "obligation_id": point_id,
                    "vector": pt.vector,
                    "payload": pt.payload,
                }
            return None
        except Exception as e:
            logger.warning(f"Failed to retrieve obligation vector for {point_id}: {e}")
            return None

    async def delete_obligation(
        self,
        obligation_id: Union[str, UUID],
        collection_name: Optional[str] = None,
    ) -> bool:
        """
        Delete an obligation vector from Qdrant by its ID.
        """
        target_collection = collection_name or self.collection_name
        client = self.get_client()
        point_id = str(obligation_id)

        try:
            await client.delete(
                collection_name=target_collection,
                points_selector=models.PointIdsList(points=[point_id]),
            )
            logger.info(f"Deleted obligation vector '{point_id}' from '{target_collection}'.")
            return True
        except Exception as e:
            logger.error(f"Failed to delete obligation vector '{point_id}': {e}")
            return False


# Global singleton instance for application usage
qdrant_client = QdrantClient()
qdrant_service = qdrant_client  # Alias for service naming convention


async def get_qdrant_client() -> QdrantClient:
    """
    FastAPI dependency returning the global QdrantClient instance.
    """
    return qdrant_client
