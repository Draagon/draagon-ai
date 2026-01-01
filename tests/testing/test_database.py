"""Tests for TestDatabase lifecycle manager (TASK-002)."""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from draagon_ai.testing.database import (
    TestDatabase,
    TestDatabaseConfig,
)


# =============================================================================
# Test TestDatabaseConfig
# =============================================================================


class TestTestDatabaseConfig:
    """Tests for TestDatabaseConfig dataclass."""

    def test_default_values(self):
        """Config uses sensible defaults."""
        config = TestDatabaseConfig()

        assert config.neo4j_uri == "bolt://localhost:7687"
        assert config.neo4j_username == "neo4j"
        assert config.database_name == "neo4j"

    def test_env_var_override(self):
        """Config reads from environment variables."""
        with patch.dict(os.environ, {
            "NEO4J_TEST_URI": "bolt://custom:7687",
            "NEO4J_TEST_USER": "custom_user",
            "NEO4J_TEST_PASSWORD": "custom_pass",
            "NEO4J_TEST_DATABASE": "custom_db",
        }):
            config = TestDatabaseConfig()

            assert config.neo4j_uri == "bolt://custom:7687"
            assert config.neo4j_username == "custom_user"
            assert config.neo4j_password == "custom_pass"
            assert config.database_name == "custom_db"

    def test_explicit_values_override_env(self):
        """Explicit values in constructor override env vars."""
        with patch.dict(os.environ, {"NEO4J_TEST_URI": "bolt://env:7687"}):
            # Note: dataclass field defaults use factory functions that read env,
            # so we need to pass explicit values to override
            config = TestDatabaseConfig(neo4j_uri="bolt://explicit:7687")

            assert config.neo4j_uri == "bolt://explicit:7687"


# =============================================================================
# Test TestDatabase Initialization
# =============================================================================


class TestTestDatabaseInit:
    """Tests for TestDatabase initialization."""

    def test_requires_neo4j_package(self):
        """TestDatabase requires neo4j package."""
        with patch("draagon_ai.testing.database.NEO4J_AVAILABLE", False):
            with pytest.raises(ImportError, match="neo4j package required"):
                TestDatabase()

    def test_creates_with_default_config(self):
        """TestDatabase uses default config when none provided."""
        db = TestDatabase()
        assert db.config.neo4j_uri == "bolt://localhost:7687"

    def test_creates_with_custom_config(self):
        """TestDatabase accepts custom config."""
        config = TestDatabaseConfig(
            neo4j_uri="bolt://custom:7687",
            database_name="custom_db",
        )
        db = TestDatabase(config)

        assert db.neo4j_uri == "bolt://custom:7687"
        assert db.database_name == "custom_db"

    def test_not_initialized_until_called(self):
        """Database isn't initialized until initialize() is called."""
        db = TestDatabase()
        assert not db._initialized
        assert db._driver is None


# =============================================================================
# Test get_config()
# =============================================================================


class TestGetConfig:
    """Tests for get_config() method."""

    def test_returns_dict(self):
        """get_config() returns a dict."""
        db = TestDatabase()
        config = db.get_config()

        assert isinstance(config, dict)

    def test_config_has_required_keys(self):
        """Config dict has all required keys for provider."""
        db = TestDatabase()
        config = db.get_config()

        assert "uri" in config
        assert "username" in config
        assert "password" in config
        assert "database" in config

    def test_config_matches_database_settings(self):
        """Config values match database settings."""
        custom_config = TestDatabaseConfig(
            neo4j_uri="bolt://test:7687",
            neo4j_username="test_user",
            neo4j_password="test_pass",
            database_name="test_db",
        )
        db = TestDatabase(custom_config)
        config = db.get_config()

        assert config["uri"] == "bolt://test:7687"
        assert config["username"] == "test_user"
        assert config["password"] == "test_pass"
        assert config["database"] == "test_db"


# =============================================================================
# Test Lifecycle with Mocked Neo4j
# =============================================================================


class TestLifecycleMocked:
    """Tests for lifecycle methods using mocked Neo4j driver."""

    @pytest.fixture
    def mock_driver(self):
        """Create a mock Neo4j async driver."""
        driver = MagicMock()
        driver.close = AsyncMock()

        # Mock session context manager
        session = MagicMock()
        session.run = AsyncMock()
        session.__aenter__ = AsyncMock(return_value=session)
        session.__aexit__ = AsyncMock(return_value=None)

        driver.session = MagicMock(return_value=session)

        return driver, session

    @pytest.fixture
    def db_with_mock(self, mock_driver):
        """Create TestDatabase with mocked driver."""
        driver, session = mock_driver
        db = TestDatabase()

        # Patch the driver creation
        with patch("draagon_ai.testing.database.AsyncGraphDatabase") as mock_gdb:
            mock_gdb.driver = MagicMock(return_value=driver)
            yield db, driver, session

    @pytest.mark.asyncio
    async def test_initialize_creates_driver(self, db_with_mock):
        """initialize() creates driver connection."""
        db, driver, session = db_with_mock

        with patch("draagon_ai.testing.database.AsyncGraphDatabase") as mock_gdb:
            mock_gdb.driver = MagicMock(return_value=driver)
            await db.initialize()

        assert db._initialized
        assert db._driver is driver

    @pytest.mark.asyncio
    async def test_initialize_idempotent(self, db_with_mock):
        """initialize() only runs once."""
        db, driver, session = db_with_mock

        with patch("draagon_ai.testing.database.AsyncGraphDatabase") as mock_gdb:
            mock_gdb.driver = MagicMock(return_value=driver)

            await db.initialize()
            await db.initialize()  # Second call should be no-op

            # Driver should only be created once
            assert mock_gdb.driver.call_count == 1

    @pytest.mark.asyncio
    async def test_clear_deletes_all_nodes(self, db_with_mock):
        """clear() runs DETACH DELETE query."""
        db, driver, session = db_with_mock

        with patch("draagon_ai.testing.database.AsyncGraphDatabase") as mock_gdb:
            mock_gdb.driver = MagicMock(return_value=driver)
            await db.initialize()

        await db.clear()

        # Verify the delete query was run
        session.run.assert_called_with("MATCH (n) DETACH DELETE n")

    @pytest.mark.asyncio
    async def test_clear_requires_initialization(self):
        """clear() fails if not initialized."""
        db = TestDatabase()

        with pytest.raises(RuntimeError, match="not initialized"):
            await db.clear()

    @pytest.mark.asyncio
    async def test_close_closes_driver(self, db_with_mock):
        """close() closes the driver connection."""
        db, driver, session = db_with_mock

        with patch("draagon_ai.testing.database.AsyncGraphDatabase") as mock_gdb:
            mock_gdb.driver = MagicMock(return_value=driver)
            await db.initialize()

        await db.close()

        driver.close.assert_called_once()
        assert db._driver is None
        assert not db._initialized

    @pytest.mark.asyncio
    async def test_verify_connection_requires_initialization(self):
        """verify_connection() fails if not initialized."""
        db = TestDatabase()

        with pytest.raises(RuntimeError, match="not initialized"):
            await db.verify_connection()

    @pytest.mark.asyncio
    async def test_verify_connection_runs_test_query(self, db_with_mock):
        """verify_connection() runs a test query."""
        db, driver, session = db_with_mock

        # Mock successful query result
        mock_result = MagicMock()
        mock_result.single = AsyncMock(return_value={"n": 1})
        session.run = AsyncMock(return_value=mock_result)

        with patch("draagon_ai.testing.database.AsyncGraphDatabase") as mock_gdb:
            mock_gdb.driver = MagicMock(return_value=driver)
            await db.initialize()

        await db.verify_connection()  # Should not raise

        session.run.assert_called_with("RETURN 1 as n")


# =============================================================================
# Test Error Messages
# =============================================================================


class TestErrorMessages:
    """Tests for helpful error messages."""

    @pytest.mark.asyncio
    async def test_verify_connection_helpful_error(self):
        """verify_connection() provides helpful error message."""
        db = TestDatabase()

        # Mock driver that fails on query
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_session.run = AsyncMock(side_effect=Exception("Connection refused"))
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_driver.session = MagicMock(return_value=mock_session)

        with patch("draagon_ai.testing.database.AsyncGraphDatabase") as mock_gdb:
            mock_gdb.driver = MagicMock(return_value=mock_driver)
            await db.initialize()

        with pytest.raises(RuntimeError) as exc_info:
            await db.verify_connection()

        error_msg = str(exc_info.value)

        # Check for helpful debugging info
        assert "Neo4j connection failed" in error_msg
        assert db.config.neo4j_uri in error_msg
        assert db.config.database_name in error_msg
        assert "Is Neo4j running?" in error_msg
        assert "docker ps" in error_msg


# =============================================================================
# Test Context Manager
# =============================================================================


class TestContextManager:
    """Tests for async context manager support."""

    @pytest.mark.asyncio
    async def test_context_manager_initializes_and_closes(self):
        """Context manager initializes on enter and closes on exit."""
        mock_driver = MagicMock()
        mock_driver.close = AsyncMock()

        with patch("draagon_ai.testing.database.AsyncGraphDatabase") as mock_gdb:
            mock_gdb.driver = MagicMock(return_value=mock_driver)

            async with TestDatabase() as db:
                assert db._initialized
                assert db._driver is mock_driver

            # After context exit
            mock_driver.close.assert_called_once()


# =============================================================================
# Integration Tests (Skip if Neo4j not available)
# =============================================================================


@pytest.mark.integration
class TestIntegration:
    """Integration tests requiring real Neo4j.

    These tests are skipped if Neo4j is not available.
    To run: ensure Neo4j is running and run with:
        pytest tests/testing/test_database.py -m integration -v
    """

    @pytest.fixture
    async def real_database(self):
        """Create a real TestDatabase connection."""
        db = TestDatabase()
        try:
            await db.initialize()
            await db.verify_connection()
            yield db
        except Exception as e:
            pytest.skip(f"Neo4j not available: {e}")
        finally:
            await db.close()

    @pytest.mark.asyncio
    async def test_real_connection(self, real_database):
        """Can connect to real Neo4j instance."""
        # If we get here, connection worked
        assert real_database._initialized

    @pytest.mark.asyncio
    async def test_real_clear(self, real_database):
        """Can clear real database."""
        # Add a test node
        async with real_database._driver.session(
            database=real_database.database_name
        ) as session:
            await session.run("CREATE (n:TestNode {name: 'test'})")

        # Verify node exists
        count_before = await real_database.node_count()
        assert count_before > 0

        # Clear
        await real_database.clear()

        # Verify empty
        count_after = await real_database.node_count()
        assert count_after == 0

    @pytest.mark.asyncio
    async def test_get_config_works_with_real_provider(self, real_database):
        """get_config() returns values compatible with Neo4jMemoryProvider."""
        config = real_database.get_config()

        # These are the keys Neo4jMemoryConfig expects
        assert "uri" in config
        assert "username" in config
        assert "password" in config
        assert "database" in config
