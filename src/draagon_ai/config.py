"""Unified configuration for Draagon AI.

DraagonConfig provides a clean way to configure all components:
- LLM provider and model
- Memory provider and connection
- Embedding provider
- MCP settings
- Cognitive features
"""

from dataclasses import dataclass, field
from typing import Literal
import os


@dataclass
class LLMConfig:
    """Configuration for the LLM provider."""

    provider: Literal["groq", "ollama", "openai", "anthropic"] = "groq"
    model: str = "llama-3.3-70b-versatile"
    api_key: str | None = None
    base_url: str | None = None

    # Model tier settings
    fast_model: str | None = None  # For quick routing decisions
    complex_model: str | None = None  # For complex reasoning


@dataclass
class MemoryConfig:
    """Configuration for the memory provider."""

    provider: Literal["qdrant", "in_memory"] = "qdrant"
    url: str = "http://localhost:6333"
    api_key: str | None = None
    collection: str = "draagon_memories"


@dataclass
class EmbeddingConfig:
    """Configuration for embeddings."""

    provider: Literal["ollama", "openai", "sentence_transformers"] = "ollama"
    model: str = "nomic-embed-text"
    url: str = "http://localhost:11434"
    dimensions: int = 768


@dataclass
class MCPConfig:
    """Configuration for MCP (Model Context Protocol)."""

    enabled: bool = False
    servers: list[dict] = field(default_factory=list)


@dataclass
class CognitiveConfig:
    """Configuration for cognitive features."""

    # Personality evolution
    personality_evolution_enabled: bool = True

    # Curiosity engine
    curiosity_enabled: bool = True
    max_curiosity_questions_per_day: int = 3
    min_gap_between_questions_minutes: int = 30

    # Learning
    learning_enabled: bool = True
    verification_enabled: bool = True


@dataclass
class DraagonConfig:
    """Main configuration for draagon-ai.

    Create from environment variables:
        config = DraagonConfig.from_env()

    Or specify directly:
        config = DraagonConfig(
            llm=LLMConfig(provider="groq", model="llama-3.3-70b-versatile"),
            memory=MemoryConfig(url="http://qdrant:6333"),
        )
    """

    llm: LLMConfig = field(default_factory=LLMConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    mcp: MCPConfig = field(default_factory=MCPConfig)
    cognitive: CognitiveConfig = field(default_factory=CognitiveConfig)

    @classmethod
    def from_env(cls) -> "DraagonConfig":
        """Load configuration from environment variables.

        Environment variables:
        - DRAAGON_LLM_PROVIDER: groq, ollama, openai, anthropic
        - DRAAGON_LLM_MODEL: Model name
        - DRAAGON_LLM_API_KEY: API key (or provider-specific like GROQ_API_KEY)
        - DRAAGON_LLM_BASE_URL: Custom base URL
        - DRAAGON_MEMORY_PROVIDER: qdrant, in_memory
        - DRAAGON_MEMORY_URL: Qdrant URL
        - DRAAGON_MEMORY_COLLECTION: Collection name
        - DRAAGON_EMBEDDING_PROVIDER: ollama, openai
        - DRAAGON_EMBEDDING_MODEL: Embedding model name
        - DRAAGON_EMBEDDING_URL: Embedding service URL
        - DRAAGON_MCP_ENABLED: true/false
        - DRAAGON_CURIOSITY_ENABLED: true/false
        - DRAAGON_LEARNING_ENABLED: true/false
        """
        # Determine LLM API key (try provider-specific first)
        llm_provider = os.getenv("DRAAGON_LLM_PROVIDER", "groq")
        api_key = os.getenv("DRAAGON_LLM_API_KEY")
        if not api_key:
            # Try provider-specific keys
            if llm_provider == "groq":
                api_key = os.getenv("GROQ_API_KEY")
            elif llm_provider == "openai":
                api_key = os.getenv("OPENAI_API_KEY")
            elif llm_provider == "anthropic":
                api_key = os.getenv("ANTHROPIC_API_KEY")

        return cls(
            llm=LLMConfig(
                provider=llm_provider,  # type: ignore
                model=os.getenv("DRAAGON_LLM_MODEL", "llama-3.3-70b-versatile"),
                api_key=api_key,
                base_url=os.getenv("DRAAGON_LLM_BASE_URL"),
                fast_model=os.getenv("DRAAGON_LLM_FAST_MODEL"),
                complex_model=os.getenv("DRAAGON_LLM_COMPLEX_MODEL"),
            ),
            memory=MemoryConfig(
                provider=os.getenv("DRAAGON_MEMORY_PROVIDER", "qdrant"),  # type: ignore
                url=os.getenv("DRAAGON_MEMORY_URL", "http://localhost:6333"),
                api_key=os.getenv("DRAAGON_MEMORY_API_KEY"),
                collection=os.getenv("DRAAGON_MEMORY_COLLECTION", "draagon_memories"),
            ),
            embedding=EmbeddingConfig(
                provider=os.getenv("DRAAGON_EMBEDDING_PROVIDER", "ollama"),  # type: ignore
                model=os.getenv("DRAAGON_EMBEDDING_MODEL", "nomic-embed-text"),
                url=os.getenv("DRAAGON_EMBEDDING_URL", "http://localhost:11434"),
                dimensions=int(os.getenv("DRAAGON_EMBEDDING_DIMENSIONS", "768")),
            ),
            mcp=MCPConfig(
                enabled=os.getenv("DRAAGON_MCP_ENABLED", "false").lower() == "true",
            ),
            cognitive=CognitiveConfig(
                personality_evolution_enabled=os.getenv(
                    "DRAAGON_PERSONALITY_EVOLUTION", "true"
                ).lower() == "true",
                curiosity_enabled=os.getenv(
                    "DRAAGON_CURIOSITY_ENABLED", "true"
                ).lower() == "true",
                max_curiosity_questions_per_day=int(
                    os.getenv("DRAAGON_MAX_QUESTIONS_PER_DAY", "3")
                ),
                learning_enabled=os.getenv(
                    "DRAAGON_LEARNING_ENABLED", "true"
                ).lower() == "true",
                verification_enabled=os.getenv(
                    "DRAAGON_VERIFICATION_ENABLED", "true"
                ).lower() == "true",
            ),
        )

    @classmethod
    def default(cls) -> "DraagonConfig":
        """Create a default configuration (same as no-arg constructor)."""
        return cls()
