"""Draagon AI - Cognitive architecture for AI with soul.

Draagon AI is a cognitive engine that enables AI agents to have:
- Persistent personality and identity
- Beliefs, values, and opinions that can evolve
- Memory with proper scoping (world/context/agent/user/session)
- Curiosity-driven knowledge seeking
- Multi-agent support with isolation

Example:
    from draagon_ai import CognitiveEngine, AgentContext
    from draagon_ai.personality import AgentIdentity

    engine = CognitiveEngine(config)
    agent = await engine.create_agent(
        agent_id="assistant",
        archetype="helpful_assistant",
    )

    response = await agent.process(
        query="Hello!",
        user_id="user_1",
    )
"""

__version__ = "0.1.0"

# Core types and context
from draagon_ai.core import (
    AgentContext,
    AgentIdentity,
    SessionContext,
    MemoryScope as CoreMemoryScope,  # Alias to distinguish from memory.MemoryScope
)

# LLM abstraction
from draagon_ai.llm import (
    LLMProvider,
    EmbeddingProvider,
    ModelTier,
    ChatMessage,
    ChatResponse,
)

# Memory abstraction
from draagon_ai.memory import (
    MemoryProvider,
    MemoryType,
    MemoryScope,
    Memory,
    SearchResult,
)

# Personality
from draagon_ai.personality import (
    Archetype,
    HELPFUL_ASSISTANT,
    get_archetype,
    list_archetypes,
)

# Cognition
from draagon_ai.cognition import (
    BeliefReconciliationService,
    ReconciliationResult,
    CredibilityProvider,
    OpinionFormationService,
    IdentityManager,  # Protocol
    IdentityManagerImpl,  # Implementation
    IdentityStorage,
    serialize_identity,
    deserialize_identity,
    CuriosityEngine,
    TraitProvider,
    ProactiveQuestionTimingService,
    QuestionOpportunity,
    ConversationMoment,
    LearningService,
    LearningResult,
    LearningType,
    VerificationResult,
    LearningExtension,  # Protocol for domain-specific hooks
)

# Persona system
from draagon_ai.persona import (
    Persona,
    PersonaTraits,
    PersonaRelationship,
    PersonaManager,
    SinglePersonaManager,
    MultiPersonaManager,
)

# Configuration
from draagon_ai.config import (
    DraagonConfig,
    LLMConfig,
    MemoryConfig,
    EmbeddingConfig,
    MCPConfig,
    CognitiveConfig,
)

# Auth & Credentials
from draagon_ai.auth import (
    CredentialScope,
    Credential,
    CredentialStore,
    EnvCredentialStore,
    InMemoryCredentialStore,
)

# Exceptions
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

# Tools (MCP integration - optional dependency)
from draagon_ai.tools import (
    MCP_AVAILABLE,
    MCPClient,
    MCPServerConfig,
    MCPTool,
    create_mcp_client,
)

__all__ = [
    # Version
    "__version__",
    # Core
    "AgentContext",
    "AgentIdentity",
    "SessionContext",
    "CoreMemoryScope",
    # LLM
    "LLMProvider",
    "EmbeddingProvider",
    "ModelTier",
    "ChatMessage",
    "ChatResponse",
    # Memory
    "MemoryProvider",
    "MemoryType",
    "MemoryScope",
    "Memory",
    "SearchResult",
    # Personality
    "Archetype",
    "HELPFUL_ASSISTANT",
    "get_archetype",
    "list_archetypes",
    # Cognition
    "BeliefReconciliationService",
    "ReconciliationResult",
    "CredibilityProvider",
    "OpinionFormationService",
    "IdentityManager",  # Protocol
    "IdentityManagerImpl",  # Implementation
    "IdentityStorage",
    "serialize_identity",
    "deserialize_identity",
    "CuriosityEngine",
    "TraitProvider",
    "ProactiveQuestionTimingService",
    "QuestionOpportunity",
    "ConversationMoment",
    "LearningService",
    "LearningResult",
    "LearningType",
    "VerificationResult",
    "LearningExtension",
    # Persona system
    "Persona",
    "PersonaTraits",
    "PersonaRelationship",
    "PersonaManager",
    "SinglePersonaManager",
    "MultiPersonaManager",
    # Configuration
    "DraagonConfig",
    "LLMConfig",
    "MemoryConfig",
    "EmbeddingConfig",
    "MCPConfig",
    "CognitiveConfig",
    # Auth & Credentials
    "CredentialScope",
    "Credential",
    "CredentialStore",
    "EnvCredentialStore",
    "InMemoryCredentialStore",
    # Exceptions
    "DraagonError",
    "ConfigurationError",
    "ProviderError",
    "LLMError",
    "MemoryError",
    "EmbeddingError",
    "CognitionError",
    "BeliefError",
    "LearningError",
    "CuriosityError",
    "OpinionError",
    "MCPError",
    "MCPConnectionError",
    "MCPToolError",
    "AuthError",
    "CredentialNotFoundError",
    "CredentialExpiredError",
    # Tools (optional MCP dependency)
    "MCP_AVAILABLE",
    "MCPClient",
    "MCPServerConfig",
    "MCPTool",
    "create_mcp_client",
]
