"""Neo4j Persistence Layer for Semantic Graph.

This module provides storage and retrieval of SemanticGraph in Neo4j,
enabling persistent graph storage with native traversal and vector search.

Key Features:
- Save/load SemanticGraph to/from Neo4j
- Node labels map to NodeType (Instance, Class, Event, etc.)
- Relationship types map to edge relation_type
- Bi-temporal edge properties preserved
- Vector index support for similarity search

Example:
    >>> from neo4j import GraphDatabase
    >>> from draagon_ai.cognition.decomposition.graph import SemanticGraph
    >>> from draagon_ai.cognition.decomposition.graph.neo4j_store import Neo4jGraphStore
    >>>
    >>> driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))
    >>> store = Neo4jGraphStore(driver)
    >>>
    >>> # Save a graph
    >>> graph = SemanticGraph()
    >>> # ... build graph ...
    >>> await store.save(graph, instance_id="my-project")
    >>>
    >>> # Load a graph
    >>> loaded = await store.load(instance_id="my-project")
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from neo4j import AsyncGraphDatabase, AsyncDriver, AsyncSession

from .models import GraphNode, GraphEdge, NodeType, EdgeRelationType
from .semantic_graph import SemanticGraph


@dataclass
class Neo4jConfig:
    """Configuration for Neo4j connection."""

    uri: str = "bolt://localhost:7687"
    username: str = "neo4j"
    password: str = "neo4j"
    database: str = "neo4j"

    # Vector index settings
    embedding_dimensions: int = 1536
    similarity_function: str = "cosine"


class Neo4jGraphStore:
    """Persistent storage for SemanticGraph in Neo4j.

    This class handles:
    - Saving SemanticGraph nodes and edges to Neo4j
    - Loading graphs back from Neo4j
    - Creating and managing vector indexes
    - Instance-based graph partitioning

    Nodes are stored with labels matching their NodeType:
        :Instance, :Class, :Event, :Attribute, :Collection

    All nodes also have an :Entity label for cross-type queries.

    Edges are stored as Neo4j relationships with properties
    for bi-temporal tracking (valid_from, valid_to).
    """

    def __init__(self, driver: AsyncDriver, config: Neo4jConfig | None = None):
        """Initialize the store.

        Args:
            driver: Neo4j async driver instance
            config: Optional configuration (uses defaults if not provided)
        """
        self.driver = driver
        self.config = config or Neo4jConfig()

    @classmethod
    async def connect(cls, config: Neo4jConfig | None = None) -> "Neo4jGraphStore":
        """Create a store with a new connection.

        Args:
            config: Connection configuration

        Returns:
            Connected Neo4jGraphStore
        """
        config = config or Neo4jConfig()
        driver = AsyncGraphDatabase.driver(
            config.uri,
            auth=(config.username, config.password),
        )
        return cls(driver, config)

    async def close(self) -> None:
        """Close the driver connection."""
        await self.driver.close()

    async def initialize(self) -> None:
        """Initialize the database schema and indexes.

        Creates:
        - Constraints for node_id uniqueness
        - Vector indexes for similarity search
        - Indexes for common query patterns
        """
        async with self.driver.session(database=self.config.database) as session:
            # Create uniqueness constraints
            await session.run("""
                CREATE CONSTRAINT node_id_unique IF NOT EXISTS
                FOR (n:Entity) REQUIRE n.node_id IS UNIQUE
            """)

            # Create indexes for common queries
            await session.run("""
                CREATE INDEX instance_id_idx IF NOT EXISTS
                FOR (n:Entity) ON (n.instance_id)
            """)

            await session.run("""
                CREATE INDEX canonical_name_idx IF NOT EXISTS
                FOR (n:Entity) ON (n.canonical_name)
            """)

            await session.run("""
                CREATE INDEX synset_id_idx IF NOT EXISTS
                FOR (n:Class) ON (n.synset_id)
            """)

            # Create vector index for embeddings
            try:
                await session.run(f"""
                    CREATE VECTOR INDEX entity_embeddings IF NOT EXISTS
                    FOR (n:Entity) ON (n.embedding)
                    OPTIONS {{indexConfig: {{
                        `vector.dimensions`: {self.config.embedding_dimensions},
                        `vector.similarity_function`: '{self.config.similarity_function}'
                    }}}}
                """)
            except Exception:
                # Vector index may already exist or not be supported
                pass

    async def save(
        self,
        graph: SemanticGraph,
        instance_id: str,
        clear_existing: bool = False,
    ) -> dict[str, int]:
        """Save a SemanticGraph to Neo4j.

        Args:
            graph: The graph to save
            instance_id: Identifier for this data instance (e.g., "doug-personal")
            clear_existing: If True, delete existing nodes for this instance first

        Returns:
            Dict with counts: {"nodes": N, "edges": M}
        """
        async with self.driver.session(database=self.config.database) as session:
            if clear_existing:
                await session.run(
                    "MATCH (n:Entity {instance_id: $instance_id}) DETACH DELETE n",
                    instance_id=instance_id,
                )

            # Save nodes
            nodes_saved = 0
            for node in graph.iter_nodes():
                await self._save_node(session, node, instance_id)
                nodes_saved += 1

            # Save edges (including historical ones)
            edges_saved = 0
            for edge in graph.iter_edges(current_only=False):
                await self._save_edge(session, edge, instance_id)
                edges_saved += 1

            return {"nodes": nodes_saved, "edges": edges_saved}

    async def _save_node(
        self,
        session: AsyncSession,
        node: GraphNode,
        instance_id: str,
    ) -> None:
        """Save a single node to Neo4j."""
        # Determine labels based on node type
        labels = ["Entity", self._node_type_to_label(node.node_type)]
        labels_str = ":".join(labels)

        # Build properties
        props = {
            "node_id": node.node_id,
            "instance_id": instance_id,
            "canonical_name": node.canonical_name,
            "node_type": node.node_type.value,
            "entity_type": node.entity_type.value if node.entity_type else None,
            "synset_id": node.synset_id,
            "wikidata_qid": node.wikidata_qid,
            "created_at": node.created_at.isoformat(),
            "updated_at": node.updated_at.isoformat(),
            "confidence": node.confidence,
            "source_ids": node.source_ids,
        }

        # Add custom properties (flattened with prefix)
        for key, value in node.properties.items():
            if isinstance(value, (str, int, float, bool, list)):
                props[f"prop_{key}"] = value

        # Add embedding if present
        if node.embedding:
            props["embedding"] = node.embedding

        # Use MERGE to upsert
        await session.run(
            f"""
            MERGE (n:{labels_str} {{node_id: $node_id}})
            SET n += $props
            """,
            node_id=node.node_id,
            props=props,
        )

    async def _save_edge(
        self,
        session: AsyncSession,
        edge: GraphEdge,
        instance_id: str,
    ) -> None:
        """Save a single edge to Neo4j."""
        # Normalize relation type for Neo4j (uppercase, no spaces)
        rel_type = edge.relation_type.upper().replace(" ", "_").replace("-", "_")

        # Build properties
        props = {
            "edge_id": edge.edge_id,
            "instance_id": instance_id,
            "relation_type": edge.relation_type,  # Keep original for display
            "valid_from": edge.valid_from.isoformat(),
            "valid_to": edge.valid_to.isoformat() if edge.valid_to else None,
            "created_at": edge.created_at.isoformat(),
            "confidence": edge.confidence,
            "source_decomposition_id": edge.source_decomposition_id,
        }

        # Add custom properties
        for key, value in edge.properties.items():
            if isinstance(value, (str, int, float, bool, list)):
                props[f"prop_{key}"] = value

        # Create relationship
        await session.run(
            f"""
            MATCH (source:Entity {{node_id: $source_id}})
            MATCH (target:Entity {{node_id: $target_id}})
            MERGE (source)-[r:{rel_type} {{edge_id: $edge_id}}]->(target)
            SET r += $props
            """,
            source_id=edge.source_node_id,
            target_id=edge.target_node_id,
            edge_id=edge.edge_id,
            props=props,
        )

    async def load(
        self,
        instance_id: str,
        current_only: bool = True,
    ) -> SemanticGraph:
        """Load a SemanticGraph from Neo4j.

        Args:
            instance_id: Identifier for the data instance to load
            current_only: If True, only load current (non-expired) edges

        Returns:
            Reconstructed SemanticGraph
        """
        graph = SemanticGraph()

        async with self.driver.session(database=self.config.database) as session:
            # Load nodes
            result = await session.run(
                "MATCH (n:Entity {instance_id: $instance_id}) RETURN n",
                instance_id=instance_id,
            )

            async for record in result:
                node = self._record_to_node(record["n"])
                graph.add_node(node)

            # Load edges
            if current_only:
                edge_query = """
                    MATCH (source:Entity {instance_id: $instance_id})
                          -[r]->(target:Entity {instance_id: $instance_id})
                    WHERE r.valid_to IS NULL
                    RETURN source.node_id AS source_id,
                           target.node_id AS target_id,
                           type(r) AS rel_type,
                           properties(r) AS props
                """
            else:
                edge_query = """
                    MATCH (source:Entity {instance_id: $instance_id})
                          -[r]->(target:Entity {instance_id: $instance_id})
                    RETURN source.node_id AS source_id,
                           target.node_id AS target_id,
                           type(r) AS rel_type,
                           properties(r) AS props
                """

            result = await session.run(edge_query, instance_id=instance_id)

            async for record in result:
                edge = self._record_to_edge(
                    record["source_id"],
                    record["target_id"],
                    record["rel_type"],
                    record["props"],
                )
                graph.add_edge(edge)

        return graph

    async def vector_search(
        self,
        instance_id: str,
        query_embedding: list[float],
        limit: int = 10,
        node_types: list[NodeType] | None = None,
        min_score: float = 0.0,
    ) -> list[tuple[GraphNode, float]]:
        """Find nodes similar to a query embedding.

        Args:
            instance_id: Data instance to search within
            query_embedding: Query vector
            limit: Maximum results to return
            node_types: Optional filter by node types
            min_score: Minimum similarity score (0-1)

        Returns:
            List of (node, score) tuples ordered by similarity
        """
        async with self.driver.session(database=self.config.database) as session:
            # Build type filter
            type_filter = ""
            if node_types:
                labels = [self._node_type_to_label(nt) for nt in node_types]
                type_filter = "AND (" + " OR ".join(f"n:{label}" for label in labels) + ")"

            result = await session.run(
                f"""
                CALL db.index.vector.queryNodes('entity_embeddings', $limit, $embedding)
                YIELD node AS n, score
                WHERE n.instance_id = $instance_id {type_filter}
                  AND score >= $min_score
                RETURN n, score
                ORDER BY score DESC
                """,
                instance_id=instance_id,
                embedding=query_embedding,
                limit=limit * 2,  # Over-fetch to account for filtering
                min_score=min_score,
            )

            results = []
            async for record in result:
                node = self._record_to_node(record["n"])
                results.append((node, record["score"]))
                if len(results) >= limit:
                    break

            return results

    async def traverse(
        self,
        instance_id: str,
        start_node_ids: list[str],
        max_depth: int = 3,
        relation_types: list[str] | None = None,
        current_only: bool = True,
    ) -> SemanticGraph:
        """Traverse the graph from starting nodes.

        Args:
            instance_id: Data instance
            start_node_ids: Node IDs to start from
            max_depth: Maximum traversal depth
            relation_types: Optional filter by relation types
            current_only: Only follow current (non-expired) edges

        Returns:
            Subgraph containing traversed nodes and edges
        """
        subgraph = SemanticGraph()

        async with self.driver.session(database=self.config.database) as session:
            # Build relation filter
            rel_filter = ""
            if relation_types:
                rel_types = [rt.upper().replace(" ", "_").replace("-", "_")
                            for rt in relation_types]
                rel_filter = ":" + "|".join(rel_types)

            # Build temporal filter
            temporal_filter = "WHERE ALL(r IN relationships(path) WHERE r.valid_to IS NULL)" if current_only else ""

            result = await session.run(
                f"""
                MATCH (start:Entity)
                WHERE start.node_id IN $start_ids
                  AND start.instance_id = $instance_id
                CALL apoc.path.subgraphAll(start, {{
                    maxLevel: $max_depth,
                    relationshipFilter: '{rel_filter}>'
                }})
                YIELD nodes, relationships
                RETURN nodes, relationships
                """,
                start_ids=start_node_ids,
                instance_id=instance_id,
                max_depth=max_depth,
            )

            # Fallback if APOC not available - use variable length path
            try:
                async for record in result:
                    for node_data in record["nodes"]:
                        node = self._record_to_node(node_data)
                        if node.node_id not in subgraph.nodes:
                            subgraph.add_node(node)

                    for rel_data in record["relationships"]:
                        # Extract edge from relationship
                        pass  # Would need to parse relationship data
            except Exception:
                # APOC not available, use simpler query
                result = await session.run(
                    f"""
                    MATCH path = (start:Entity)-[{rel_filter}*1..{max_depth}]-(connected:Entity)
                    WHERE start.node_id IN $start_ids
                      AND start.instance_id = $instance_id
                      AND connected.instance_id = $instance_id
                    {temporal_filter}
                    UNWIND nodes(path) AS n
                    WITH DISTINCT n
                    RETURN n
                    """,
                    start_ids=start_node_ids,
                    instance_id=instance_id,
                )

                async for record in result:
                    node = self._record_to_node(record["n"])
                    if node.node_id not in subgraph.nodes:
                        subgraph.add_node(node)

                # Load edges between found nodes
                node_ids = list(subgraph.nodes.keys())
                if node_ids:
                    edge_result = await session.run(
                        f"""
                        MATCH (source:Entity)-[r{rel_filter}]->(target:Entity)
                        WHERE source.node_id IN $node_ids
                          AND target.node_id IN $node_ids
                          AND source.instance_id = $instance_id
                          {"AND r.valid_to IS NULL" if current_only else ""}
                        RETURN source.node_id AS source_id,
                               target.node_id AS target_id,
                               type(r) AS rel_type,
                               properties(r) AS props
                        """,
                        node_ids=node_ids,
                        instance_id=instance_id,
                    )

                    async for record in edge_result:
                        edge = self._record_to_edge(
                            record["source_id"],
                            record["target_id"],
                            record["rel_type"],
                            record["props"],
                        )
                        subgraph.add_edge(edge)

        return subgraph

    async def get_statistics(self, instance_id: str) -> dict[str, Any]:
        """Get statistics about a stored graph.

        Args:
            instance_id: Data instance

        Returns:
            Dict with node counts by type, edge counts, etc.
        """
        async with self.driver.session(database=self.config.database) as session:
            # Node counts by type
            result = await session.run(
                """
                MATCH (n:Entity {instance_id: $instance_id})
                RETURN n.node_type AS type, count(*) AS count
                """,
                instance_id=instance_id,
            )

            node_counts = {}
            async for record in result:
                node_counts[record["type"]] = record["count"]

            # Edge counts
            result = await session.run(
                """
                MATCH (s:Entity {instance_id: $instance_id})-[r]->(t:Entity {instance_id: $instance_id})
                RETURN r.relation_type AS type, count(*) AS count
                """,
                instance_id=instance_id,
            )

            edge_counts = {}
            async for record in result:
                edge_counts[record["type"]] = record["count"]

            return {
                "instance_id": instance_id,
                "total_nodes": sum(node_counts.values()),
                "total_edges": sum(edge_counts.values()),
                "nodes_by_type": node_counts,
                "edges_by_type": edge_counts,
            }

    async def delete_instance(self, instance_id: str) -> int:
        """Delete all data for an instance.

        Args:
            instance_id: Data instance to delete

        Returns:
            Number of nodes deleted
        """
        async with self.driver.session(database=self.config.database) as session:
            result = await session.run(
                """
                MATCH (n:Entity {instance_id: $instance_id})
                WITH n, count(*) AS cnt
                DETACH DELETE n
                RETURN sum(cnt) AS deleted
                """,
                instance_id=instance_id,
            )
            record = await result.single()
            return record["deleted"] if record else 0

    def _node_type_to_label(self, node_type: NodeType) -> str:
        """Convert NodeType to Neo4j label."""
        return node_type.value.capitalize()

    def _record_to_node(self, node_data: dict[str, Any]) -> GraphNode:
        """Convert Neo4j node record to GraphNode."""
        # Extract custom properties (those with prop_ prefix)
        properties = {}
        for key, value in node_data.items():
            if key.startswith("prop_"):
                properties[key[5:]] = value  # Remove prefix

        # Handle entity_type
        entity_type = None
        if node_data.get("entity_type"):
            from ..identifiers import EntityType
            try:
                entity_type = EntityType(node_data["entity_type"])
            except ValueError:
                pass

        return GraphNode(
            node_id=node_data["node_id"],
            node_type=NodeType(node_data.get("node_type", "instance")),
            canonical_name=node_data.get("canonical_name", ""),
            entity_type=entity_type,
            properties=properties,
            synset_id=node_data.get("synset_id"),
            wikidata_qid=node_data.get("wikidata_qid"),
            created_at=datetime.fromisoformat(node_data["created_at"]) if node_data.get("created_at") else datetime.now(timezone.utc),
            updated_at=datetime.fromisoformat(node_data["updated_at"]) if node_data.get("updated_at") else datetime.now(timezone.utc),
            source_ids=node_data.get("source_ids", []),
            confidence=node_data.get("confidence", 1.0),
            embedding=node_data.get("embedding"),
        )

    def _record_to_edge(
        self,
        source_id: str,
        target_id: str,
        rel_type: str,
        props: dict[str, Any],
    ) -> GraphEdge:
        """Convert Neo4j relationship record to GraphEdge."""
        # Extract custom properties
        properties = {}
        for key, value in props.items():
            if key.startswith("prop_"):
                properties[key[5:]] = value

        return GraphEdge(
            edge_id=props.get("edge_id", ""),
            source_node_id=source_id,
            target_node_id=target_id,
            relation_type=props.get("relation_type", rel_type.lower()),
            properties=properties,
            valid_from=datetime.fromisoformat(props["valid_from"]) if props.get("valid_from") else datetime.now(timezone.utc),
            valid_to=datetime.fromisoformat(props["valid_to"]) if props.get("valid_to") else None,
            source_decomposition_id=props.get("source_decomposition_id"),
            confidence=props.get("confidence", 1.0),
            created_at=datetime.fromisoformat(props["created_at"]) if props.get("created_at") else datetime.now(timezone.utc),
        )


# Synchronous wrapper for non-async contexts
class Neo4jGraphStoreSync:
    """Synchronous wrapper for Neo4jGraphStore.

    Use this when you need to work with Neo4j in synchronous code.
    Internally runs async operations in an event loop.
    """

    def __init__(self, uri: str, username: str, password: str, database: str = "neo4j"):
        """Initialize with connection parameters."""
        from neo4j import GraphDatabase
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        self.database = database
        self.config = Neo4jConfig(
            uri=uri,
            username=username,
            password=password,
            database=database,
        )

    def close(self) -> None:
        """Close the driver."""
        self.driver.close()

    def save(
        self,
        graph: SemanticGraph,
        instance_id: str,
        clear_existing: bool = False,
    ) -> dict[str, int]:
        """Save a SemanticGraph to Neo4j."""
        with self.driver.session(database=self.database) as session:
            if clear_existing:
                session.run(
                    "MATCH (n:Entity {instance_id: $instance_id}) DETACH DELETE n",
                    instance_id=instance_id,
                )

            nodes_saved = 0
            for node in graph.iter_nodes():
                self._save_node_sync(session, node, instance_id)
                nodes_saved += 1

            edges_saved = 0
            for edge in graph.iter_edges(current_only=False):
                self._save_edge_sync(session, edge, instance_id)
                edges_saved += 1

            return {"nodes": nodes_saved, "edges": edges_saved}

    def _save_node_sync(self, session, node: GraphNode, instance_id: str) -> None:
        """Save a node synchronously."""
        labels = ["Entity", node.node_type.value.capitalize()]
        labels_str = ":".join(labels)

        props = {
            "node_id": node.node_id,
            "instance_id": instance_id,
            "canonical_name": node.canonical_name,
            "node_type": node.node_type.value,
            "entity_type": node.entity_type.value if node.entity_type else None,
            "synset_id": node.synset_id,
            "wikidata_qid": node.wikidata_qid,
            "created_at": node.created_at.isoformat(),
            "updated_at": node.updated_at.isoformat(),
            "confidence": node.confidence,
            "source_ids": node.source_ids,
        }

        for key, value in node.properties.items():
            if isinstance(value, (str, int, float, bool, list)):
                props[f"prop_{key}"] = value

        if node.embedding:
            props["embedding"] = node.embedding

        session.run(
            f"""
            MERGE (n:{labels_str} {{node_id: $node_id}})
            SET n += $props
            """,
            node_id=node.node_id,
            props=props,
        )

    def _save_edge_sync(self, session, edge: GraphEdge, instance_id: str) -> None:
        """Save an edge synchronously."""
        rel_type = edge.relation_type.upper().replace(" ", "_").replace("-", "_")

        props = {
            "edge_id": edge.edge_id,
            "instance_id": instance_id,
            "relation_type": edge.relation_type,
            "valid_from": edge.valid_from.isoformat(),
            "valid_to": edge.valid_to.isoformat() if edge.valid_to else None,
            "created_at": edge.created_at.isoformat(),
            "confidence": edge.confidence,
            "source_decomposition_id": edge.source_decomposition_id,
        }

        for key, value in edge.properties.items():
            if isinstance(value, (str, int, float, bool, list)):
                props[f"prop_{key}"] = value

        session.run(
            f"""
            MATCH (source:Entity {{node_id: $source_id}})
            MATCH (target:Entity {{node_id: $target_id}})
            MERGE (source)-[r:{rel_type} {{edge_id: $edge_id}}]->(target)
            SET r += $props
            """,
            source_id=edge.source_node_id,
            target_id=edge.target_node_id,
            edge_id=edge.edge_id,
            props=props,
        )

    def load(self, instance_id: str, current_only: bool = True) -> SemanticGraph:
        """Load a SemanticGraph from Neo4j."""
        graph = SemanticGraph()

        with self.driver.session(database=self.database) as session:
            # Load nodes
            result = session.run(
                "MATCH (n:Entity {instance_id: $instance_id}) RETURN n",
                instance_id=instance_id,
            )

            for record in result:
                node = self._record_to_node_sync(dict(record["n"]))
                graph.add_node(node)

            # Load edges
            edge_query = """
                MATCH (source:Entity {instance_id: $instance_id})
                      -[r]->(target:Entity {instance_id: $instance_id})
            """
            if current_only:
                edge_query += " WHERE r.valid_to IS NULL"
            edge_query += """
                RETURN source.node_id AS source_id,
                       target.node_id AS target_id,
                       type(r) AS rel_type,
                       properties(r) AS props
            """

            result = session.run(edge_query, instance_id=instance_id)

            for record in result:
                edge = self._record_to_edge_sync(
                    record["source_id"],
                    record["target_id"],
                    record["rel_type"],
                    dict(record["props"]),
                )
                graph.add_edge(edge)

        return graph

    def _record_to_node_sync(self, node_data: dict[str, Any]) -> GraphNode:
        """Convert Neo4j node record to GraphNode."""
        properties = {}
        for key, value in node_data.items():
            if key.startswith("prop_"):
                properties[key[5:]] = value

        entity_type = None
        if node_data.get("entity_type"):
            from ..identifiers import EntityType
            try:
                entity_type = EntityType(node_data["entity_type"])
            except ValueError:
                pass

        return GraphNode(
            node_id=node_data["node_id"],
            node_type=NodeType(node_data.get("node_type", "instance")),
            canonical_name=node_data.get("canonical_name", ""),
            entity_type=entity_type,
            properties=properties,
            synset_id=node_data.get("synset_id"),
            wikidata_qid=node_data.get("wikidata_qid"),
            created_at=datetime.fromisoformat(node_data["created_at"]) if node_data.get("created_at") else datetime.now(timezone.utc),
            updated_at=datetime.fromisoformat(node_data["updated_at"]) if node_data.get("updated_at") else datetime.now(timezone.utc),
            source_ids=node_data.get("source_ids", []),
            confidence=node_data.get("confidence", 1.0),
            embedding=node_data.get("embedding"),
        )

    def _record_to_edge_sync(
        self,
        source_id: str,
        target_id: str,
        rel_type: str,
        props: dict[str, Any],
    ) -> GraphEdge:
        """Convert Neo4j relationship record to GraphEdge."""
        properties = {}
        for key, value in props.items():
            if key.startswith("prop_"):
                properties[key[5:]] = value

        return GraphEdge(
            edge_id=props.get("edge_id", ""),
            source_node_id=source_id,
            target_node_id=target_id,
            relation_type=props.get("relation_type", rel_type.lower()),
            properties=properties,
            valid_from=datetime.fromisoformat(props["valid_from"]) if props.get("valid_from") else datetime.now(timezone.utc),
            valid_to=datetime.fromisoformat(props["valid_to"]) if props.get("valid_to") else None,
            source_decomposition_id=props.get("source_decomposition_id"),
            confidence=props.get("confidence", 1.0),
            created_at=datetime.fromisoformat(props["created_at"]) if props.get("created_at") else datetime.now(timezone.utc),
        )

    def get_statistics(self, instance_id: str) -> dict[str, Any]:
        """Get statistics about a stored graph."""
        with self.driver.session(database=self.database) as session:
            result = session.run(
                """
                MATCH (n:Entity {instance_id: $instance_id})
                RETURN n.node_type AS type, count(*) AS count
                """,
                instance_id=instance_id,
            )

            node_counts = {}
            for record in result:
                node_counts[record["type"]] = record["count"]

            result = session.run(
                """
                MATCH (s:Entity {instance_id: $instance_id})-[r]->(t:Entity {instance_id: $instance_id})
                RETURN r.relation_type AS type, count(*) AS count
                """,
                instance_id=instance_id,
            )

            edge_counts = {}
            for record in result:
                edge_counts[record["type"]] = record["count"]

            return {
                "instance_id": instance_id,
                "total_nodes": sum(node_counts.values()),
                "total_edges": sum(edge_counts.values()),
                "nodes_by_type": node_counts,
                "edges_by_type": edge_counts,
            }
