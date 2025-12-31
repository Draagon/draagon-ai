# Evolving Synset Database Architecture

## Overview

This document describes a future architecture where the synset database is not static but evolves through usage, learning, and reinforcement. Instead of maintaining a separate WordNet database, synset definitions become part of the semantic memory system itself.

## Current State (Phase 0)

Currently, we use NLTK WordNet as the synset database:
- **Bootstrap**: Load ~117,000 synsets from WordNet
- **Limitation**: Static, English-focused, academic vocabulary bias
- **Limitation**: No domain-specific terms (kubernetes, API, microservices)
- **Limitation**: Can't learn new senses from context

## Vision: Synsets as Semantic Knowledge

**Key Insight**: A synset definition IS semantic knowledge. The statement "The word 'bank' in financial contexts means 'a financial institution'" is just a special kind of fact.

This means synset definitions belong in the same memory system as:
- User facts ("Doug has 3 cats")
- Skills ("To restart Plex, run `docker restart plex`")
- Preferences ("Doug prefers Celsius")

## Proposed Architecture

### Data Model

```python
@dataclass
class LearnedSynset:
    """A synset definition learned from context."""
    synset_id: str              # e.g., "kubernetes.n.01"
    word: str                   # e.g., "kubernetes"
    pos: str                    # e.g., "n"
    definition: str             # e.g., "container orchestration platform"
    examples: list[str]         # Example sentences
    hypernyms: list[str]        # Parent concepts
    hyponyms: list[str]         # Child concepts

    # Learning metadata
    source: str                 # "wordnet", "llm", "user"
    learned_from_context: str   # Original context that taught this
    confidence: float           # How confident we are in this definition
    usage_count: int            # How often this sense has been used
    success_rate: float         # How often it led to good disambiguations
```

### Memory Integration

Synsets are stored as semantic memories with special metadata:

```python
# Store a learned synset
memory_id = await memory.store(
    content=f"Synset {synset_id}: {definition}",
    metadata={
        "type": "synset_definition",
        "word": "kubernetes",
        "synset_id": "kubernetes.n.01",
        "pos": "n",
        "definition": "container orchestration platform for automating deployment",
        "hypernyms": ["software.n.01", "platform.n.03"],
        "source": "llm",
        "learned_from_context": "We deploy our services to kubernetes clusters",
    }
)
```

### Hybrid Database Class

```python
class EvolvingSynsetDatabase:
    """Synset database that grows from usage.

    Priority order for sense selection:
    1. User-corrected definitions (highest trust)
    2. LLM-generated definitions that proved useful (verified by good outcomes)
    3. WordNet definitions (baseline, academically validated)
    4. LLM-generated definitions (unverified, exploratory)
    """

    PRIORITY = {
        "user_corrected": 1.0,
        "llm_verified": 0.9,
        "wordnet": 0.8,
        "llm_unverified": 0.5,
    }

    def __init__(self, memory_provider: MemoryProvider):
        self.memory = memory_provider
        self.wordnet = self._try_load_wordnet()

    async def get_synsets(self, word: str, pos: str | None = None) -> list[SynsetInfo]:
        """Get synsets for a word, merging learned and WordNet definitions."""

        # 1. Check semantic memory for learned definitions
        learned = await self.memory.search(
            query=f"synset definition for '{word}'",
            memory_type=MemoryType.SEMANTIC,
            metadata_filter={"type": "synset_definition", "word": word},
        )

        # 2. Get WordNet synsets if available
        wordnet_synsets = []
        if self.wordnet:
            wordnet_synsets = self._get_from_wordnet(word, pos)

        # 3. Merge, handling duplicates and conflicts
        return self._merge_synsets(learned, wordnet_synsets)

    async def learn_synset_from_context(
        self,
        word: str,
        context: str,
        llm: LLMProvider,
    ) -> SynsetInfo:
        """Learn a new synset definition when word is unknown."""

        # Ask LLM to generate definition
        prompt = f"""
        The word "{word}" appears in this context:
        "{context}"

        Generate a dictionary-style definition for this word as used here.

        Output XML:
        <synset>
            <definition>...</definition>
            <part_of_speech>n|v|adj|adv</part_of_speech>
            <hypernyms>comma-separated parent concepts</hypernyms>
            <example>example sentence using this sense</example>
        </synset>
        """

        response = await llm.chat([{"role": "user", "content": prompt}])
        synset_info = self._parse_llm_synset(word, response)

        # Store as unverified learned synset
        await self._store_learned_synset(synset_info, source="llm_unverified", context=context)

        return synset_info

    async def reinforce_synset(self, synset_id: str, outcome: str):
        """Reinforce or demote a synset based on disambiguation outcome.

        Called after disambiguation when we know if the result was good or bad.
        """
        memories = await self.memory.search(
            query=synset_id,
            metadata_filter={"synset_id": synset_id, "type": "synset_definition"},
        )

        if not memories:
            return  # WordNet synset, can't reinforce

        memory = memories[0]

        if outcome == "success":
            # Boost confidence
            await self.memory.boost_memory(memory.id)

            # If it's unverified LLM, upgrade to verified
            if memory.metadata.get("source") == "llm_unverified":
                await self.memory.update(memory.id, {"source": "llm_verified"})

        elif outcome == "failure":
            await self.memory.demote_memory(memory.id)

    async def user_correction(
        self,
        word: str,
        context: str,
        correct_definition: str,
    ) -> SynsetInfo:
        """User explicitly corrects a definition.

        This creates a highest-priority synset that will be preferred
        over both WordNet and LLM definitions.
        """
        synset_info = SynsetInfo(
            synset_id=self._generate_synset_id(word),
            word=word,
            definition=correct_definition,
            # ... other fields
        )

        await self._store_learned_synset(
            synset_info,
            source="user_corrected",
            context=context,
        )

        return synset_info
```

## Evolution Lifecycle

### 1. Bootstrap Phase
- Load WordNet as baseline (~117k synsets)
- These have `source="wordnet"` and moderate priority

### 2. Learning Phase
When encountering unknown words:
```
User: "We need to scale our k8s deployment"
       ↓
System: "k8s" not in WordNet
       ↓
LLM generates: "k8s - abbreviation for Kubernetes, a container orchestration platform"
       ↓
Store as learned synset (source="llm_unverified")
```

### 3. Verification Phase
When learned synsets lead to good outcomes:
```
Disambiguation: "k8s" → "kubernetes.n.01"
       ↓
User confirms response was helpful
       ↓
Boost synset, upgrade to "llm_verified"
```

### 4. User Correction Phase
When synsets are wrong:
```
System: Uses wrong sense of "python" (snake)
       ↓
User: "No, I meant the programming language"
       ↓
Store correction as "user_corrected" (highest priority)
```

## Benefits

1. **Grows with Usage**: System learns domain vocabulary naturally
2. **Self-Correcting**: Bad definitions get demoted through reinforcement
3. **User-Adaptive**: Corrections are remembered permanently
4. **No Manual Maintenance**: Database evolves without human curation
5. **Domain-Specific**: Tech, medical, legal terms learned from context

## Implementation Phases

### Phase 1: WordNet Only (Current)
- Use NLTK WordNet
- Fail explicitly if not available
- Baseline 40-60% accuracy with Lesk

### Phase 2: LLM Fallback for Unknown Words
- When word not in WordNet, ask LLM
- Store as unverified learned synset
- Track success/failure rates

### Phase 3: Memory Integration
- Store synsets in semantic memory
- Apply reinforcement learning
- Enable user corrections

### Phase 4: Active Learning
- Identify low-confidence disambiguations
- Queue clarifying questions
- Learn from corrections proactively

## Memory Type Integration

Learned synsets fit naturally into the 4-layer memory architecture:

| Layer | Synset Storage |
|-------|----------------|
| **Working** | Current disambiguation context |
| **Episodic** | Recent word sense usages |
| **Semantic** | Learned synset definitions |
| **Metacognitive** | Disambiguation success patterns |

## Example: Tech Domain Learning

Starting state: WordNet only (no "kubernetes", "docker", "microservices")

```
Day 1: User mentions "kubernetes"
→ LLM generates definition
→ Stored as unverified

Day 2: "kubernetes" used again, disambiguation succeeds
→ Boosted, upgraded to verified

Day 5: User corrects "pod" (meant k8s pod, not legume)
→ User correction stored with highest priority

Day 30: System has learned 50+ tech terms from context
→ Tech vocabulary now part of semantic memory
```

## Relation to Implicit Knowledge Graphs

This evolving synset database is a natural extension of the implicit knowledge graph concept:

- **Traditional**: "Doug has 3 cats" → stored fact
- **Extended**: "'cat' means 'small domesticated feline'" → stored synset
- **Connection**: Both are semantic knowledge extracted from context

The synset database becomes just another facet of the knowledge graph, subject to the same:
- Confidence weighting
- Reinforcement learning
- Evolutionary optimization
- Multi-source reconciliation

## Future: Cross-Document Sense Learning

Eventually, the system could learn word senses across documents:

```
Document 1: "The bank approved our loan application"
Document 2: "We walked along the river bank at sunset"
Document 3: "Bank on me to finish this project"

→ System learns three senses of "bank" from context
→ No external dictionary required
→ Senses emerge from usage patterns
```

This is the ultimate vision: a self-bootstrapping semantic system that learns word meanings the way humans do - from context, usage, and correction.
