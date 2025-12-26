"""Tests for auth module (credentials and scopes)."""

import os
import pytest
from datetime import datetime, timedelta
from unittest.mock import patch

from draagon_ai.auth import (
    CredentialScope,
    Credential,
    CredentialStore,
    EnvCredentialStore,
    InMemoryCredentialStore,
)


# =============================================================================
# Credential Tests
# =============================================================================


class TestCredential:
    """Tests for Credential dataclass."""

    def test_basic_credential(self):
        """Test creating a basic credential."""
        cred = Credential(
            type="api_key",
            value="secret",
            name="test_api",
            scope=CredentialScope.GLOBAL,
        )
        assert cred.type == "api_key"
        assert cred.value == "secret"
        assert cred.name == "test_api"
        assert cred.scope == CredentialScope.GLOBAL

    def test_is_expired_no_expiry(self):
        """Test is_expired with no expiry date."""
        cred = Credential(
            type="api_key",
            value="secret",
            name="test",
            scope=CredentialScope.GLOBAL,
        )
        assert cred.is_expired is False

    def test_is_expired_future(self):
        """Test is_expired with future expiry."""
        cred = Credential(
            type="oauth",
            value="token",
            name="test",
            scope=CredentialScope.USER,
            expires_at=datetime.now() + timedelta(hours=1),
        )
        assert cred.is_expired is False

    def test_is_expired_past(self):
        """Test is_expired with past expiry."""
        cred = Credential(
            type="oauth",
            value="token",
            name="test",
            scope=CredentialScope.USER,
            expires_at=datetime.now() - timedelta(hours=1),
        )
        assert cred.is_expired is True

    def test_is_oauth(self):
        """Test is_oauth property."""
        oauth = Credential(
            type="oauth", value="token", name="test", scope=CredentialScope.USER
        )
        api_key = Credential(
            type="api_key", value="key", name="test", scope=CredentialScope.GLOBAL
        )

        assert oauth.is_oauth is True
        assert api_key.is_oauth is False

    def test_can_refresh(self):
        """Test can_refresh property."""
        with_refresh = Credential(
            type="oauth",
            value="token",
            name="test",
            scope=CredentialScope.USER,
            refresh_token="refresh",
        )
        without_refresh = Credential(
            type="oauth",
            value="token",
            name="test",
            scope=CredentialScope.USER,
        )
        not_oauth = Credential(
            type="api_key",
            value="key",
            name="test",
            scope=CredentialScope.GLOBAL,
            refresh_token="ignored",
        )

        assert with_refresh.can_refresh is True
        assert without_refresh.can_refresh is False
        assert not_oauth.can_refresh is False


# =============================================================================
# EnvCredentialStore Tests
# =============================================================================


class TestEnvCredentialStore:
    """Tests for EnvCredentialStore."""

    @pytest.mark.asyncio
    async def test_get_existing_credential(self):
        """Test getting an existing credential from env."""
        env = {"DRAAGON_CRED_TEST_API": "secret_value"}
        with patch.dict(os.environ, env, clear=True):
            store = EnvCredentialStore()
            cred = await store.get_credential(
                "test_api", CredentialScope.GLOBAL, {}
            )

        assert cred is not None
        assert cred.value == "secret_value"
        assert cred.type == "api_key"
        assert cred.name == "test_api"

    @pytest.mark.asyncio
    async def test_get_credential_with_type(self):
        """Test getting credential with type override."""
        env = {
            "DRAAGON_CRED_GOOGLE": "oauth_token",
            "DRAAGON_CRED_GOOGLE_TYPE": "oauth",
        }
        with patch.dict(os.environ, env, clear=True):
            store = EnvCredentialStore()
            cred = await store.get_credential("google", CredentialScope.GLOBAL, {})

        assert cred is not None
        assert cred.type == "oauth"

    @pytest.mark.asyncio
    async def test_get_nonexistent_credential(self):
        """Test getting a credential that doesn't exist."""
        with patch.dict(os.environ, {}, clear=True):
            store = EnvCredentialStore()
            cred = await store.get_credential(
                "nonexistent", CredentialScope.GLOBAL, {}
            )

        assert cred is None

    @pytest.mark.asyncio
    async def test_only_global_scope(self):
        """Test that only global scope is supported."""
        env = {"DRAAGON_CRED_TEST": "value"}
        with patch.dict(os.environ, env, clear=True):
            store = EnvCredentialStore()

            # Global should work
            global_cred = await store.get_credential(
                "test", CredentialScope.GLOBAL, {}
            )
            assert global_cred is not None

            # User should not
            user_cred = await store.get_credential(
                "test", CredentialScope.USER, {"user_id": "doug"}
            )
            assert user_cred is None

    @pytest.mark.asyncio
    async def test_store_raises(self):
        """Test that store raises NotImplementedError."""
        store = EnvCredentialStore()
        cred = Credential(
            type="api_key", value="x", name="test", scope=CredentialScope.GLOBAL
        )
        with pytest.raises(NotImplementedError):
            await store.store_credential(cred, {})

    @pytest.mark.asyncio
    async def test_delete_raises(self):
        """Test that delete raises NotImplementedError."""
        store = EnvCredentialStore()
        with pytest.raises(NotImplementedError):
            await store.delete_credential("test", CredentialScope.GLOBAL, {})

    @pytest.mark.asyncio
    async def test_list_credentials(self):
        """Test listing credentials from environment."""
        env = {
            "DRAAGON_CRED_ONE": "value1",
            "DRAAGON_CRED_TWO": "value2",
            "OTHER_VAR": "ignored",
        }
        with patch.dict(os.environ, env, clear=True):
            store = EnvCredentialStore()
            creds = await store.list_credentials()

        assert len(creds) == 2
        names = {c.name for c in creds}
        assert names == {"one", "two"}

    @pytest.mark.asyncio
    async def test_custom_prefix(self):
        """Test custom environment variable prefix."""
        env = {"MYAPP_CRED_API": "custom"}
        with patch.dict(os.environ, env, clear=True):
            store = EnvCredentialStore(prefix="MYAPP_CRED_")
            cred = await store.get_credential("api", CredentialScope.GLOBAL, {})

        assert cred is not None
        assert cred.value == "custom"


# =============================================================================
# InMemoryCredentialStore Tests
# =============================================================================


class TestInMemoryCredentialStore:
    """Tests for InMemoryCredentialStore."""

    @pytest.fixture
    def store(self):
        """Create a fresh in-memory store."""
        return InMemoryCredentialStore()

    @pytest.mark.asyncio
    async def test_store_and_get(self, store):
        """Test storing and retrieving a credential."""
        cred = Credential(
            type="api_key",
            value="secret",
            name="test",
            scope=CredentialScope.GLOBAL,
        )
        await store.store_credential(cred, {})

        retrieved = await store.get_credential("test", CredentialScope.GLOBAL, {})
        assert retrieved is not None
        assert retrieved.value == "secret"

    @pytest.mark.asyncio
    async def test_user_scoped_credentials(self, store):
        """Test user-scoped credentials are isolated."""
        cred_doug = Credential(
            type="oauth",
            value="doug_token",
            name="calendar",
            scope=CredentialScope.USER,
        )
        cred_sarah = Credential(
            type="oauth",
            value="sarah_token",
            name="calendar",
            scope=CredentialScope.USER,
        )

        await store.store_credential(cred_doug, {"user_id": "doug"})
        await store.store_credential(cred_sarah, {"user_id": "sarah"})

        # Each user gets their own
        doug_retrieved = await store.get_credential(
            "calendar", CredentialScope.USER, {"user_id": "doug"}
        )
        sarah_retrieved = await store.get_credential(
            "calendar", CredentialScope.USER, {"user_id": "sarah"}
        )

        assert doug_retrieved.value == "doug_token"
        assert sarah_retrieved.value == "sarah_token"

    @pytest.mark.asyncio
    async def test_delete_credential(self, store):
        """Test deleting a credential."""
        cred = Credential(
            type="api_key", value="x", name="test", scope=CredentialScope.GLOBAL
        )
        await store.store_credential(cred, {})

        # Should exist
        assert await store.get_credential("test", CredentialScope.GLOBAL, {}) is not None

        # Delete
        result = await store.delete_credential("test", CredentialScope.GLOBAL, {})
        assert result is True

        # Should be gone
        assert await store.get_credential("test", CredentialScope.GLOBAL, {}) is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, store):
        """Test deleting a credential that doesn't exist."""
        result = await store.delete_credential(
            "nonexistent", CredentialScope.GLOBAL, {}
        )
        assert result is False

    @pytest.mark.asyncio
    async def test_list_all(self, store):
        """Test listing all credentials."""
        await store.store_credential(
            Credential(type="api_key", value="1", name="a", scope=CredentialScope.GLOBAL),
            {},
        )
        await store.store_credential(
            Credential(type="api_key", value="2", name="b", scope=CredentialScope.USER),
            {"user_id": "doug"},
        )

        all_creds = await store.list_credentials()
        assert len(all_creds) == 2

    @pytest.mark.asyncio
    async def test_list_by_scope(self, store):
        """Test listing credentials filtered by scope."""
        await store.store_credential(
            Credential(type="api_key", value="1", name="a", scope=CredentialScope.GLOBAL),
            {},
        )
        await store.store_credential(
            Credential(type="api_key", value="2", name="b", scope=CredentialScope.USER),
            {"user_id": "doug"},
        )

        global_creds = await store.list_credentials(scope=CredentialScope.GLOBAL)
        user_creds = await store.list_credentials(scope=CredentialScope.USER)

        assert len(global_creds) == 1
        assert len(user_creds) == 1

    @pytest.mark.asyncio
    async def test_clear(self, store):
        """Test clearing all credentials."""
        await store.store_credential(
            Credential(type="api_key", value="x", name="a", scope=CredentialScope.GLOBAL),
            {},
        )
        await store.store_credential(
            Credential(type="api_key", value="y", name="b", scope=CredentialScope.GLOBAL),
            {},
        )

        store.clear()
        assert len(await store.list_credentials()) == 0

    @pytest.mark.asyncio
    async def test_cascading_lookup(self, store):
        """Test cascading credential lookup."""
        # Store at different scopes
        await store.store_credential(
            Credential(
                type="api_key", value="global", name="api", scope=CredentialScope.GLOBAL
            ),
            {},
        )
        await store.store_credential(
            Credential(
                type="api_key", value="user", name="api", scope=CredentialScope.USER
            ),
            {"user_id": "doug"},
        )

        # User scope should win when user_id is provided
        cred = await store.get_credential_cascading("api", {"user_id": "doug"})
        assert cred.value == "user"

        # Global should be returned for other users
        cred = await store.get_credential_cascading("api", {"user_id": "sarah"})
        assert cred.value == "global"

        # Global should be returned with no context
        cred = await store.get_credential_cascading("api", {})
        assert cred.value == "global"
