"""Tests for draagon_ai exception hierarchy."""

import pytest

from draagon_ai.exceptions import (
    DraagonError,
    ConfigurationError,
    ProviderError,
    LLMError,
    MemoryError,
    EmbeddingError,
    CognitionError,
    BeliefError,
    LearningError,
    CuriosityError,
    OpinionError,
    MCPError,
    MCPConnectionError,
    MCPToolError,
    AuthError,
    CredentialNotFoundError,
    CredentialExpiredError,
)


class TestDraagonError:
    """Tests for base DraagonError."""

    def test_basic_error(self):
        """Test creating a basic error."""
        err = DraagonError("Something went wrong")
        assert str(err) == "Something went wrong"
        assert err.cause is None

    def test_error_with_cause(self):
        """Test error with underlying cause."""
        cause = ValueError("inner error")
        err = DraagonError("Outer error", cause=cause)
        assert "Outer error" in str(err)
        assert "inner error" in str(err)
        assert err.cause is cause

    def test_all_errors_inherit_from_base(self):
        """Test that all exceptions inherit from DraagonError."""
        exceptions = [
            ConfigurationError("test"),
            ProviderError("test"),
            LLMError("test"),
            MemoryError("test"),
            EmbeddingError("test"),
            CognitionError("test"),
            BeliefError("test"),
            LearningError("test"),
            CuriosityError("test"),
            OpinionError("test"),
            MCPError("test"),
            MCPConnectionError("test"),
            MCPToolError("test"),
            AuthError("test"),
            CredentialNotFoundError("cred"),
            CredentialExpiredError("cred"),
        ]
        for exc in exceptions:
            assert isinstance(exc, DraagonError), f"{type(exc).__name__} should inherit from DraagonError"


class TestProviderErrors:
    """Tests for provider error hierarchy."""

    def test_llm_error_is_provider_error(self):
        """Test LLMError is a ProviderError."""
        err = LLMError("LLM failed")
        assert isinstance(err, ProviderError)
        assert isinstance(err, DraagonError)

    def test_memory_error_is_provider_error(self):
        """Test MemoryError is a ProviderError."""
        err = MemoryError("Memory failed")
        assert isinstance(err, ProviderError)

    def test_embedding_error_is_provider_error(self):
        """Test EmbeddingError is a ProviderError."""
        err = EmbeddingError("Embedding failed")
        assert isinstance(err, ProviderError)


class TestCognitionErrors:
    """Tests for cognition error hierarchy."""

    def test_belief_error_is_cognition_error(self):
        """Test BeliefError is a CognitionError."""
        err = BeliefError("Belief reconciliation failed")
        assert isinstance(err, CognitionError)
        assert isinstance(err, DraagonError)

    def test_learning_error_is_cognition_error(self):
        """Test LearningError is a CognitionError."""
        err = LearningError("Learning failed")
        assert isinstance(err, CognitionError)

    def test_curiosity_error_is_cognition_error(self):
        """Test CuriosityError is a CognitionError."""
        err = CuriosityError("Curiosity failed")
        assert isinstance(err, CognitionError)

    def test_opinion_error_is_cognition_error(self):
        """Test OpinionError is a CognitionError."""
        err = OpinionError("Opinion formation failed")
        assert isinstance(err, CognitionError)


class TestMCPErrors:
    """Tests for MCP error hierarchy."""

    def test_mcp_connection_error(self):
        """Test MCPConnectionError with server name."""
        err = MCPConnectionError("Connection failed", server_name="calendar")
        assert isinstance(err, MCPError)
        assert isinstance(err, DraagonError)
        assert err.server_name == "calendar"

    def test_mcp_tool_error(self):
        """Test MCPToolError with tool name."""
        err = MCPToolError("Tool failed", tool_name="list-events")
        assert isinstance(err, MCPError)
        assert err.tool_name == "list-events"

    def test_mcp_error_with_cause(self):
        """Test MCP error with underlying cause."""
        cause = IOError("socket closed")
        err = MCPConnectionError("Connection lost", server_name="fetch", cause=cause)
        assert err.cause is cause
        assert "socket closed" in str(err)


class TestAuthErrors:
    """Tests for auth error hierarchy."""

    def test_credential_not_found_basic(self):
        """Test CredentialNotFoundError without scope."""
        err = CredentialNotFoundError("api_key")
        assert isinstance(err, AuthError)
        assert isinstance(err, DraagonError)
        assert "api_key" in str(err)
        assert err.credential_name == "api_key"
        assert err.scope is None

    def test_credential_not_found_with_scope(self):
        """Test CredentialNotFoundError with scope."""
        err = CredentialNotFoundError("oauth_token", scope="user:doug")
        assert "oauth_token" in str(err)
        assert "user:doug" in str(err)
        assert err.scope == "user:doug"

    def test_credential_expired(self):
        """Test CredentialExpiredError."""
        err = CredentialExpiredError("google_token")
        assert isinstance(err, AuthError)
        assert "expired" in str(err).lower()
        assert err.credential_name == "google_token"


class TestExceptionCatching:
    """Test that exception hierarchy enables proper catching."""

    def test_catch_all_draagon_errors(self):
        """Test catching all library errors with base class."""
        errors = [
            LLMError("llm"),
            MemoryError("memory"),
            BeliefError("belief"),
            MCPToolError("tool"),
            CredentialNotFoundError("cred"),
        ]

        for err in errors:
            with pytest.raises(DraagonError):
                raise err

    def test_catch_provider_errors_only(self):
        """Test catching only provider errors."""
        # These should be caught
        with pytest.raises(ProviderError):
            raise LLMError("llm failed")

        with pytest.raises(ProviderError):
            raise MemoryError("memory failed")

        # This should NOT be caught as ProviderError
        with pytest.raises(CognitionError):
            raise BeliefError("belief failed")

    def test_catch_cognition_errors_only(self):
        """Test catching only cognition errors."""
        with pytest.raises(CognitionError):
            raise LearningError("learning failed")

        # This should NOT be caught as CognitionError
        with pytest.raises(AuthError):
            raise CredentialExpiredError("token")
