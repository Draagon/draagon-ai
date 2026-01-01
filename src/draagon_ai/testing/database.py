"""
Neo4j test database lifecycle manager.

TestDatabase is LIFECYCLE-ONLY. It manages:
- Connection/initialization
- Cleanup between tests
- Configuration for creating real providers

It does NOT wrap MemoryProvider or provide helper methods.
Tests use the REAL MemoryProvider directly.

Critical Design Decision: Lifecycle-Only
-----------------------------------------
Why this matters:
1. No leaky abstractions - tests use production APIs
2. Real bugs surface in tests - no mock hiding issues
3. If tests pass, production will work

Cleanup Policy:
- Clear BEFORE each test (ensures clean slate)
- Do NOT clear AFTER test (preserves state for debugging)
- Use http://localhost:7474 to inspect failed test data

Example:
    @pytest.fixture(scope="session")
    async def test_database():
        db = TestDatabase()
        await db.initialize()
        yield db
        await db.close()

    @pytest.fixture
    async def clean_database(test_database):
        await test_database.clear()
        await test_database.verify_connection()
        yield test_database

    @pytest.fixture
    async def memory_provider(clean_database, embedding_provider, llm_provider):
        config = clean_database.get_config()
        provider = Neo4jMemoryProvider(config, embedding_provider, llm_provider)
        await provider.initialize()
        return provider
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

try:
    from neo4j import AsyncGraphDatabase
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    AsyncGraphDatabase = None

if TYPE_CHECKING:
    from neo4j import AsyncDriver


@dataclass
class TestDatabaseConfig:
    """Configuration for test database.

    Can be customized via environment variables for CI/CD.
    """

    neo4j_uri: str = field(
        default_factory=lambda: os.environ.get(
            "NEO4J_TEST_URI", "bolt://localhost:7687"
        )
    )
    neo4j_username: str = field(
        default_factory=lambda: os.environ.get("NEO4J_TEST_USER", "neo4j")
    )
    neo4j_password: str = field(
        default_factory=lambda: os.environ.get(
            "NEO4J_TEST_PASSWORD", "draagon-ai-2025"
        )
    )
    database_name: str = field(
        default_factory=lambda: os.environ.get("NEO4J_TEST_DATABASE", "neo4j")
    )


class TestDatabase:
    """Neo4j test database lifecycle manager ONLY.

    Does NOT wrap or provide helper methods for MemoryProvider.
    Tests use the REAL MemoryProvider directly.

    Lifecycle methods:
    - initialize(): Create driver connection
    - clear(): Delete all nodes and relationships
    - verify_connection(): Health check with helpful errors
    - close(): Close driver connection
    - get_config(): Return config dict for provider creation

    Example:
        db = TestDatabase()
        await db.initialize()

        # Get config for real provider
        config = db.get_config()
        provider = Neo4jMemoryProvider(config, embedding, llm)
        await provider.initialize()

        # Use real provider in tests...

        await db.close()
    """

    def __init__(self, config: TestDatabaseConfig | None = None):
        """Initialize test database manager.

        Args:
            config: Optional custom configuration. If not provided,
                   uses TestDatabaseConfig defaults (with env vars).
        """
        if not NEO4J_AVAILABLE:
            raise ImportError(
                "neo4j package required for TestDatabase. "
                "Install with: pip install neo4j"
            )

        self.config = config or TestDatabaseConfig()
        self._driver: AsyncDriver | None = None
        self._initialized = False

    @property
    def neo4j_uri(self) -> str:
        """Get Neo4j URI."""
        return self.config.neo4j_uri

    @property
    def database_name(self) -> str:
        """Get database name."""
        return self.config.database_name

    async def initialize(self) -> None:
        """Initialize driver connection.

        Creates the async driver but does NOT create a separate test database.
        Uses the configured database name directly.
        """
        if self._initialized:
            return

        self._driver = AsyncGraphDatabase.driver(
            self.config.neo4j_uri,
            auth=(self.config.neo4j_username, self.config.neo4j_password),
        )
        self._initialized = True

    async def verify_connection(self) -> None:
        """Verify Neo4j connection is healthy.

        Raises:
            RuntimeError: If connection fails, with helpful debugging info.
        """
        if not self._driver:
            raise RuntimeError(
                "TestDatabase not initialized. Call initialize() first."
            )

        try:
            async with self._driver.session(database=self.config.database_name) as session:
                result = await session.run("RETURN 1 as n")
                record = await result.single()
                if record["n"] != 1:
                    raise RuntimeError("Unexpected query result")
        except Exception as e:
            error_msg = str(e)
            raise RuntimeError(
                f"Neo4j connection failed.\n"
                f"\n"
                f"Connection details:\n"
                f"  URI: {self.config.neo4j_uri}\n"
                f"  Database: {self.config.database_name}\n"
                f"  Username: {self.config.neo4j_username}\n"
                f"\n"
                f"Troubleshooting:\n"
                f"  1. Is Neo4j running? Try: docker ps | grep neo4j\n"
                f"  2. Check connection: cypher-shell -a {self.config.neo4j_uri}\n"
                f"  3. View browser: http://localhost:7474\n"
                f"\n"
                f"Environment variables:\n"
                f"  NEO4J_TEST_URI={os.environ.get('NEO4J_TEST_URI', '(not set)')}\n"
                f"  NEO4J_TEST_DATABASE={os.environ.get('NEO4J_TEST_DATABASE', '(not set)')}\n"
                f"\n"
                f"Error: {error_msg}"
            ) from e

    async def clear(self) -> None:
        """Delete all nodes and relationships from test database.

        CLEANUP POLICY:
        - Clears BEFORE each test (ensures clean slate)
        - Does NOT clear AFTER test (preserves state for debugging)
        - For debugging: http://localhost:7474 (database: {database_name})
        """
        if not self._driver:
            raise RuntimeError(
                "TestDatabase not initialized. Call initialize() first."
            )

        async with self._driver.session(database=self.config.database_name) as session:
            # Delete all nodes and relationships
            await session.run("MATCH (n) DETACH DELETE n")

    async def close(self) -> None:
        """Close driver connection."""
        if self._driver:
            await self._driver.close()
            self._driver = None
            self._initialized = False

    def get_config(self) -> dict:
        """Return configuration dict for creating real providers.

        Returns:
            Dict with uri, username, password, database keys.
            Can be unpacked into Neo4jMemoryConfig.

        Example:
            config_dict = test_db.get_config()
            provider_config = Neo4jMemoryConfig(**config_dict)
        """
        return {
            "uri": self.config.neo4j_uri,
            "username": self.config.neo4j_username,
            "password": self.config.neo4j_password,
            "database": self.config.database_name,
        }

    async def node_count(self) -> int:
        """Get count of all nodes in database.

        Useful for verifying cleanup worked.
        """
        if not self._driver:
            raise RuntimeError(
                "TestDatabase not initialized. Call initialize() first."
            )

        async with self._driver.session(database=self.config.database_name) as session:
            result = await session.run("MATCH (n) RETURN count(n) as count")
            record = await result.single()
            return record["count"]

    async def __aenter__(self) -> "TestDatabase":
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()
