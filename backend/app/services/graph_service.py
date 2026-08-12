"""
Graph Service.

Generic high-level service for interacting with Neo4j database.
Provides query execution and test node creation/retrieval methods.
"""

import logging
from typing import Any, Dict, List, Optional
from app.integrations.neo4j_client import Neo4jClient, neo4j_client

logger = logging.getLogger(__name__)


class GraphService:
    """
    Generic service wrapping Neo4j graph database operations.
    """

    def __init__(self, client: Optional[Neo4jClient] = None):
        self.client = client or neo4j_client

    async def execute_query(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
        db: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Execute a raw Cypher query through the Neo4j client.

        :param query: Cypher query string
        :param parameters: Optional dictionary of query parameters
        :param db: Optional target database name
        :return: List of record dictionaries returned by query
        """
        return await self.client.execute_query(query, parameters=parameters, db=db)

    async def create_test_node(self, name: str = "graph-test") -> Dict[str, Any]:
        """
        Create or merge a simple test node (:Test {name: $name}).

        :param name: Name property for test node (default: 'graph-test')
        :return: Properties dictionary of created test node
        """
        query = """
        MERGE (n:Test {name: $name})
        RETURN n.name AS name
        """
        results = await self.execute_query(query, parameters={"name": name})
        if results:
            logger.info(f"Test node created/verified with name: {name}")
            return results[0]
        return {"name": name}

    async def get_test_node(self, name: str = "graph-test") -> Optional[Dict[str, Any]]:
        """
        Retrieve a test node by name (:Test {name: $name}).

        :param name: Name property to search for (default: 'graph-test')
        :return: Properties dictionary if found, else None
        """
        query = """
        MATCH (n:Test {name: $name})
        RETURN n.name AS name
        """
        results = await self.execute_query(query, parameters={"name": name})
        if results:
            return results[0]
        return None
