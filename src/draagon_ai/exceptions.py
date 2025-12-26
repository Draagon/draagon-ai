"""Standard exception hierarchy for Draagon AI.

All draagon-ai exceptions inherit from DraagonError, making it easy
to catch all library-specific errors.

Exception Hierarchy:
    DraagonError (base)
    ├── ConfigurationError - Invalid configuration
    ├── ProviderError - Base for provider errors
    │   ├── LLMError - LLM provider errors
    │   ├── MemoryError - Memory provider errors
    │   └── EmbeddingError - Embedding provider errors
    ├── CognitionError - Base for cognitive service errors
    │   ├── BeliefError - Belief reconciliation errors
    │   ├── LearningError - Learning service errors
    │   └── CuriosityError - Curiosity engine errors
    ├── MCPError - MCP-related errors
    │   ├── MCPConnectionError - Failed to connect to MCP server
    │   └── MCPToolError - Tool execution failed
    └── AuthError - Authentication/credential errors
"""


class DraagonError(Exception):
    """Base exception for all draagon-ai errors.

    Catch this to handle any library-specific exception:
        try:
            result = await service.process(...)
        except DraagonError as e:
            logger.error(f"Draagon AI error: {e}")
    """

    def __init__(self, message: str, cause: Exception | None = None):
        super().__init__(message)
        self.cause = cause

    def __str__(self) -> str:
        if self.cause:
            return f"{super().__str__()} (caused by: {self.cause})"
        return super().__str__()


# =============================================================================
# Configuration Errors
# =============================================================================


class ConfigurationError(DraagonError):
    """Invalid configuration.

    Raised when DraagonConfig has invalid settings, missing required
    values, or incompatible options.
    """

    pass


# =============================================================================
# Provider Errors
# =============================================================================


class ProviderError(DraagonError):
    """Base exception for provider-related errors."""

    pass


class LLMError(ProviderError):
    """LLM provider error.

    Raised when:
    - LLM API call fails
    - Response parsing fails
    - Rate limits exceeded
    - Invalid model specified
    """

    pass


class MemoryError(ProviderError):
    """Memory provider error.

    Raised when:
    - Memory storage fails
    - Search fails
    - Connection to vector DB lost
    - Invalid query
    """

    pass


class EmbeddingError(ProviderError):
    """Embedding provider error.

    Raised when:
    - Embedding generation fails
    - Invalid input text
    - Model not available
    """

    pass


# =============================================================================
# Cognition Errors
# =============================================================================


class CognitionError(DraagonError):
    """Base exception for cognitive service errors."""

    pass


class BeliefError(CognitionError):
    """Belief reconciliation error.

    Raised when:
    - Observation creation fails
    - Belief reconciliation fails
    - Conflict resolution fails
    """

    pass


class LearningError(CognitionError):
    """Learning service error.

    Raised when:
    - Learning detection fails
    - Memory modification fails
    - Verification fails
    """

    pass


class CuriosityError(CognitionError):
    """Curiosity engine error.

    Raised when:
    - Question generation fails
    - Knowledge gap detection fails
    """

    pass


class OpinionError(CognitionError):
    """Opinion formation error.

    Raised when:
    - Opinion formation fails
    - Identity update fails
    """

    pass


# =============================================================================
# MCP Errors
# =============================================================================


class MCPError(DraagonError):
    """Base exception for MCP-related errors."""

    pass


class MCPConnectionError(MCPError):
    """Failed to connect to MCP server.

    Raised when:
    - Server process fails to start
    - Connection timeout
    - Server exits unexpectedly
    """

    def __init__(
        self,
        message: str,
        server_name: str | None = None,
        cause: Exception | None = None,
    ):
        super().__init__(message, cause)
        self.server_name = server_name


class MCPToolError(MCPError):
    """MCP tool execution failed.

    Raised when:
    - Tool not found
    - Invalid arguments
    - Tool execution error
    """

    def __init__(
        self,
        message: str,
        tool_name: str | None = None,
        cause: Exception | None = None,
    ):
        super().__init__(message, cause)
        self.tool_name = tool_name


# =============================================================================
# Auth Errors
# =============================================================================


class AuthError(DraagonError):
    """Authentication/credential error.

    Raised when:
    - Credential not found
    - Credential expired
    - Invalid scope
    - OAuth token refresh failed
    """

    pass


class CredentialNotFoundError(AuthError):
    """Credential not found in store."""

    def __init__(self, credential_name: str, scope: str | None = None):
        message = f"Credential '{credential_name}' not found"
        if scope:
            message += f" in scope '{scope}'"
        super().__init__(message)
        self.credential_name = credential_name
        self.scope = scope


class CredentialExpiredError(AuthError):
    """Credential has expired and cannot be refreshed."""

    def __init__(self, credential_name: str):
        super().__init__(f"Credential '{credential_name}' has expired")
        self.credential_name = credential_name
