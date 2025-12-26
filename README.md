# Draagon AI

**Cognitive architecture for AI agents with personality, memory, and beliefs.**

Draagon AI is a Python framework for building AI agents that have genuine personality, persistent memory, and evolving beliefs. Unlike simple chatbots, Draagon-powered agents can:

- **Remember** - Long-term memory with beliefs, facts, skills, and episodic recall
- **Learn** - Automatically extract learnings from interactions
- **Reason** - Form opinions, reconcile contradictions, identify knowledge gaps
- **Evolve** - Personality traits adjust over time based on experience
- **Multi-persona** - Support for single assistants or multiple NPCs/characters

## Installation

```bash
pip install draagon-ai

# With specific LLM providers
pip install draagon-ai[groq]        # Groq (fast inference)
pip install draagon-ai[openai]      # OpenAI
pip install draagon-ai[anthropic]   # Anthropic Claude
pip install draagon-ai[ollama]      # Local Ollama

# With all optional dependencies
pip install draagon-ai[all]
```

## Quick Start

```python
from draagon_ai import Persona, PersonaTraits, SinglePersonaManager

# Create a persona
assistant = Persona(
    id="assistant",
    name="Ada",
    description="A helpful AI assistant",
    traits=PersonaTraits.from_dict({
        "helpful": 0.95,
        "curious": 0.7,
        "formal": 0.3,
    }),
)

# Create a persona manager
manager = SinglePersonaManager(assistant)

# Get the active persona
persona = await manager.get_active_persona({})
print(f"Active: {persona.name}")  # "Ada"
```

## Multi-Persona Example (RPG NPCs)

```python
from draagon_ai import Persona, MultiPersonaManager, PersonaTraits

# Create NPCs
blacksmith = Persona(
    id="grumak",
    name="Grumak",
    description="Aging orc blacksmith",
    traits=PersonaTraits.from_dict({"gruff": 0.8, "honest": 0.9}),
    voice_notes="Short sentences. Drops articles.",
)

sage = Persona(
    id="elara",
    name="Elara",
    description="Elven scholar",
    traits=PersonaTraits.from_dict({"mysterious": 0.9, "verbose": 0.7}),
)

# Manage multiple personas
manager = MultiPersonaManager(
    personas=[blacksmith, sage],
    default_id="grumak",
)

# Switch between personas
await manager.switch_to("elara")
current = await manager.get_active_persona({})
print(f"Now speaking as: {current.name}")  # "Elara"
```

## Core Concepts

### Persona
A coherent identity the AI can embody, with traits, knowledge filters, and voice style.

### Memory
Long-term storage with beliefs, facts, skills, and episodic memories. Supports:
- Contradiction detection
- Confidence propagation
- Memory decay and consolidation

### Cognition
Higher-level reasoning including:
- **Beliefs** - Reconcile conflicting information
- **Opinions** - Form and update preferences
- **Curiosity** - Identify knowledge gaps
- **Learning** - Extract facts and skills from conversations

### MCP Support
Model Context Protocol integration for extensible tool use.

## Configuration

Environment variables:
```bash
# LLM
DRAAGON_LLM_PROVIDER=groq          # groq, openai, anthropic, ollama
DRAAGON_LLM_MODEL=llama-3.3-70b-versatile
GROQ_API_KEY=your-key

# Memory
DRAAGON_MEMORY_PROVIDER=qdrant     # qdrant, in_memory
DRAAGON_MEMORY_URL=http://localhost:6333

# Embedding
DRAAGON_EMBEDDING_PROVIDER=ollama
DRAAGON_EMBEDDING_MODEL=nomic-embed-text
```

## License

Draagon AI is dual-licensed:

- **AGPL-3.0** - Free for open source projects
- **Commercial License** - Available for proprietary use

See [LICENSE](LICENSE) for AGPL terms. Contact doug@draagon.ai for commercial licensing.

## Contributing

Contributions are welcome! Please read our contributing guidelines before submitting PRs.

## Links

- [Documentation](https://github.com/Draagon/draagon-ai#readme)
- [GitHub](https://github.com/Draagon/draagon-ai)
- [PyPI](https://pypi.org/project/draagon-ai/)
