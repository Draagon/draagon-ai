"""Memory Layers - 4-layer cognitive memory architecture.

This module implements a biologically-inspired memory system with four layers:

1. **Working Memory** (seconds to minutes)
   - Session-scoped, limited capacity (7±2 items)
   - Attention weights and activation decay
   - Used for current context and goals

2. **Episodic Memory** (hours to days)
   - Autobiographical experiences
   - Chronologically linked episodes
   - Entity and event tracking

3. **Semantic Memory** (days to months)
   - Factual knowledge, entities, relationships
   - Entity resolution and deduplication
   - Community detection

4. **Metacognitive Memory** (weeks to permanent)
   - Skills, strategies, insights
   - Effectiveness tracking
   - Self-improvement capability

Example:
    from draagon_ai.memory import TemporalCognitiveGraph
    from draagon_ai.memory.layers import (
        WorkingMemory,
        EpisodicMemory,
        SemanticMemory,
        MetacognitiveMemory,
        MemoryPromotion,
    )

    # Initialize graph
    graph = TemporalCognitiveGraph()

    # Create layers
    working = WorkingMemory(graph, session_id="session_123")
    episodic = EpisodicMemory(graph)
    semantic = SemanticMemory(graph)
    metacognitive = MetacognitiveMemory(graph)

    # Set up promotion service
    promotion = MemoryPromotion(
        graph=graph,
        working=working,
        episodic=episodic,
        semantic=semantic,
        metacognitive=metacognitive,
    )

    # Use the layers
    await working.add("User asked about weather", attention_weight=0.8)
    episode = await episodic.start_episode(episode_type="conversation")
    entity = await semantic.create_entity("Doug", "person")
    skill = await metacognitive.add_skill("restart_plex", "command", "docker restart plex")

    # Run promotion cycle
    stats = await promotion.promote_all()
"""

# Base classes
from .base import LayerConfig, MemoryLayerBase, PromotionResult

# Working Memory
from .working import (
    WorkingMemory,
    WorkingMemoryItem,
    DEFAULT_CAPACITY,
    DEFAULT_TTL as WORKING_DEFAULT_TTL,
)

# Episodic Memory
from .episodic import (
    EpisodicMemory,
    Episode,
    Event,
    DEFAULT_TTL as EPISODIC_DEFAULT_TTL,
)

# Semantic Memory
from .semantic import (
    SemanticMemory,
    Entity,
    Fact,
    Relationship,
    EntityMatch,
    DEFAULT_TTL as SEMANTIC_DEFAULT_TTL,
)

# Metacognitive Memory
from .metacognitive import (
    MetacognitiveMemory,
    Skill,
    Strategy,
    Insight,
    BehaviorNode,
    DEFAULT_TTL as METACOGNITIVE_DEFAULT_TTL,
)

# Promotion Service
from .promotion import (
    MemoryPromotion,
    MemoryConsolidator,
    PromotionConfig,
    PromotionStats,
)

__all__ = [
    # Base
    "LayerConfig",
    "MemoryLayerBase",
    "PromotionResult",
    # Working
    "WorkingMemory",
    "WorkingMemoryItem",
    "DEFAULT_CAPACITY",
    "WORKING_DEFAULT_TTL",
    # Episodic
    "EpisodicMemory",
    "Episode",
    "Event",
    "EPISODIC_DEFAULT_TTL",
    # Semantic
    "SemanticMemory",
    "Entity",
    "Fact",
    "Relationship",
    "EntityMatch",
    "SEMANTIC_DEFAULT_TTL",
    # Metacognitive
    "MetacognitiveMemory",
    "Skill",
    "Strategy",
    "Insight",
    "BehaviorNode",
    "METACOGNITIVE_DEFAULT_TTL",
    # Promotion
    "MemoryPromotion",
    "MemoryConsolidator",
    "PromotionConfig",
    "PromotionStats",
]
