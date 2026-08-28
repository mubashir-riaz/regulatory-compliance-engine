"""
Neo4j Database Integration Client.

Provides driver initialization, connection/session handling for FastAPI,
query execution capabilities, and connection testing.
"""

import os
import logging
from typing import Any, Dict, List, Optional, AsyncGenerator
from neo4j import AsyncGraphDatabase, AsyncDriver, AsyncSession
from app.core.config import settings

logger = logging.getLogger(__name__)


class Neo4jClient:
    """
    Reusable Neo4j client for managing driver connection, sessions,
    and executing Cypher queries within FastAPI application context.
    """

    def __init__(
        self,
        uri: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
    ):
        self.uri = (
            uri
            or os.getenv("NEO4J_URI")
            or os.getenv("NEO4J_URL")
            or getattr(settings, "NEO4J_URI", "bolt://neo4j:7687")
        )
        self.user = (
            user
            or os.getenv("NEO4J_USER")
            or os.getenv("NEO4J_USERNAME")
            or getattr(settings, "NEO4J_USER", "neo4j")
        )
        self.password = (
            password
            or os.getenv("NEO4J_PASSWORD")
            or getattr(settings, "NEO4J_PASSWORD", "password")
        )

        neo4j_auth = os.getenv("NEO4J_AUTH")
        if neo4j_auth and "/" in neo4j_auth and not (user or password or os.getenv("NEO4J_USER") or os.getenv("NEO4J_PASSWORD")):
            auth_user, auth_pass = neo4j_auth.split("/", 1)
            self.user = auth_user
            self.password = auth_pass

        self._driver: Optional[AsyncDriver] = None

    def connect(self) -> AsyncDriver:
        """
        Initialize and return the async Neo4j driver connection pool.
        """
        if not self._driver:
            logger.info(f"Connecting to Neo4j database at {self.uri}")
            self._driver = AsyncGraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password),
            )
        return self._driver

    async def close(self) -> None:
        """
        Close the Neo4j driver connection pool.
        """
        if self._driver:
            logger.info("Closing Neo4j connection pool")
            await self._driver.close()
            self._driver = None

    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """
        FastAPI dependency yielding an async Neo4j session.
        """
        driver = self.connect()
        async with driver.session() as session:
            yield session

    async def execute_query(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
        db: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Execute a Cypher query and return the results as a list of dictionaries.

        :param query: Cypher query string
        :param parameters: Optional dictionary of query parameters
        :param db: Optional target database name
        :return: List of result records formatted as dicts
        """
        driver = self.connect()
        params = parameters or {}
        async with driver.session(database=db) as session:
            result = await session.run(query, params)
            records = await result.data()
            return records

    async def test_connection(self) -> bool:
        """
        Executes 'RETURN 1' query to verify backend communication with Neo4j.
        Returns True if communication is successful.
        """
        try:
            records = await self.execute_query("RETURN 1 AS result")
            if records and records[0].get("result") == 1:
                logger.info("Neo4j connection test successful (RETURN 1).")
                return True
            logger.warning(f"Unexpected connection test result: {records}")
            return False
        except Exception as e:
            logger.error(f"Neo4j connection test failed: {e}")
            raise e


# Global singleton instance for application usage
neo4j_client = Neo4jClient()


async def get_neo4j_client() -> Neo4jClient:
    """
    FastAPI dependency returning the global Neo4jClient instance.
    """
    return neo4j_client
