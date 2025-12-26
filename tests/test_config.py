"""Tests for DraagonConfig."""

import os
import pytest
from unittest.mock import patch

from draagon_ai.config import (
    DraagonConfig,
    LLMConfig,
    MemoryConfig,
    EmbeddingConfig,
    MCPConfig,
    CognitiveConfig,
)


class TestDraagonConfig:
    """Tests for DraagonConfig."""

    def test_default_config(self):
        """Test creating default configuration."""
        config = DraagonConfig()

        assert config.llm.provider == "groq"
        assert config.llm.model == "llama-3.3-70b-versatile"
        assert config.memory.provider == "qdrant"
        assert config.embedding.provider == "ollama"
        assert config.mcp.enabled is False
        assert config.cognitive.learning_enabled is True

    def test_custom_config(self):
        """Test creating custom configuration."""
        config = DraagonConfig(
            llm=LLMConfig(provider="openai", model="gpt-4o"),
            memory=MemoryConfig(url="http://qdrant:6333"),
            mcp=MCPConfig(enabled=True),
        )

        assert config.llm.provider == "openai"
        assert config.llm.model == "gpt-4o"
        assert config.memory.url == "http://qdrant:6333"
        assert config.mcp.enabled is True

    def test_from_env_defaults(self):
        """Test loading from environment with no vars set."""
        with patch.dict(os.environ, {}, clear=True):
            config = DraagonConfig.from_env()

        assert config.llm.provider == "groq"
        assert config.memory.provider == "qdrant"

    def test_from_env_with_vars(self):
        """Test loading from environment variables."""
        env = {
            "DRAAGON_LLM_PROVIDER": "openai",
            "DRAAGON_LLM_MODEL": "gpt-4o",
            "OPENAI_API_KEY": "sk-test",
            "DRAAGON_MEMORY_URL": "http://qdrant:6333",
            "DRAAGON_MCP_ENABLED": "true",
            "DRAAGON_CURIOSITY_ENABLED": "false",
        }
        with patch.dict(os.environ, env, clear=True):
            config = DraagonConfig.from_env()

        assert config.llm.provider == "openai"
        assert config.llm.model == "gpt-4o"
        assert config.llm.api_key == "sk-test"
        assert config.memory.url == "http://qdrant:6333"
        assert config.mcp.enabled is True
        assert config.cognitive.curiosity_enabled is False

    def test_from_env_groq_api_key(self):
        """Test Groq API key detection."""
        env = {
            "GROQ_API_KEY": "gsk-test",
        }
        with patch.dict(os.environ, env, clear=True):
            config = DraagonConfig.from_env()

        assert config.llm.api_key == "gsk-test"

    def test_from_env_anthropic_api_key(self):
        """Test Anthropic API key detection."""
        env = {
            "DRAAGON_LLM_PROVIDER": "anthropic",
            "ANTHROPIC_API_KEY": "sk-ant-test",
        }
        with patch.dict(os.environ, env, clear=True):
            config = DraagonConfig.from_env()

        assert config.llm.provider == "anthropic"
        assert config.llm.api_key == "sk-ant-test"

    def test_default_method(self):
        """Test the default() class method."""
        config = DraagonConfig.default()
        assert config == DraagonConfig()


class TestLLMConfig:
    """Tests for LLMConfig."""

    def test_default_values(self):
        """Test default LLM config."""
        config = LLMConfig()
        assert config.provider == "groq"
        assert config.model == "llama-3.3-70b-versatile"
        assert config.api_key is None
        assert config.fast_model is None
        assert config.complex_model is None


class TestMemoryConfig:
    """Tests for MemoryConfig."""

    def test_default_values(self):
        """Test default memory config."""
        config = MemoryConfig()
        assert config.provider == "qdrant"
        assert config.url == "http://localhost:6333"
        assert config.collection == "draagon_memories"


class TestCognitiveConfig:
    """Tests for CognitiveConfig."""

    def test_default_values(self):
        """Test default cognitive config."""
        config = CognitiveConfig()
        assert config.personality_evolution_enabled is True
        assert config.curiosity_enabled is True
        assert config.max_curiosity_questions_per_day == 3
        assert config.min_gap_between_questions_minutes == 30
        assert config.learning_enabled is True
        assert config.verification_enabled is True
