# Semantic Expansion: Presupposition Extraction for Deep Understanding

**Version:** 0.1.0-CONCEPT
**Status:** Exploration / Pre-Specification
**Last Updated:** 2025-12-30
**Author:** draagon-ai team

---

## Executive Summary

This document captures a breakthrough architectural concept that addresses a fundamental limitation in current multi-agent systems: **short phrases carry implicit meaning that surface-level processing cannot capture**.

Current approaches (including our existing SharedWorkingMemory) compare text at face value. This concept proposes using LLMs to **expand short statements into full semantic frames**, generate **multiple interpretation variations**, and **explore them in parallel** using draagon-ai's cognitive infrastructure.

**Key Insight:** This is where graph structures become genuinely necessary—not as an alternative storage format, but as the natural representation of semantic relationships extracted from expanded meaning.

---

## Table of Contents

1. [The Problem](#1-the-problem)
2. [The Solution: Semantic Frame Expansion](#2-the-solution-semantic-frame-expansion)
3. [Research Foundation](#3-research-foundation)
4. [Architecture Concept](#4-architecture-concept)
5. [Integration with draagon-ai](#5-integration-with-draagon-ai)
6. [Graph Structure Necessity](#6-graph-structure-necessity)
7. [Example Walkthrough](#7-example-walkthrough)
8. [Open Questions](#8-open-questions)
9. [Relationship to FR-001-005](#9-relationship-to-fr-001-005)
10. [Next Steps](#10-next-steps)

---

## 1. The Problem

### 1.1 Surface-Level Comparison Fails

Current approaches to conflict detection and semantic understanding compare text at surface level:

```python
# Current: Compare "The meeting is at 3pm" vs "The meeting is at 4pm"
# Detection: Find "3pm" != "4pm" → conflict
# Problem: Works for this case, but...

# What about:
# "Doug likes coffee" vs "He prefers tea in the morning"
#
# Current approach: No obvious conflict
# Reality: These might conflict if "he" = Doug and "morning" = default context
```

### 1.2 Implicit Knowledge Problem

Short phrases carry enormous implicit meaning:

| Statement | Surface Meaning | Implicit Presuppositions |
|-----------|-----------------|--------------------------|
| "Turn off the lights" | Set lights to off | There are lights. They're on. Speaker has authority. Lights are controllable. |
| "The meeting moved" | Meeting location changed | There was a meeting. It had a time/place. Something caused the change. |
| "Doug's favorite color" | Color preference | Doug exists. Doug has preferences. Colors have subjective value. |
| "It's cold in here" | Temperature is low | There's an enclosed space. Normal temperature expected. Discomfort implied. Possibly a request to change it. |

### 1.3 Context Dependency

The same statement means different things based on:
- **Who said it** (user identity, authority level)
- **When it was said** (recency, temporal context)
- **What we already believe** (existing facts, preferences)
- **Conversation history** (recent topics, implied references)
- **Relationship knowledge** (who is "he", what is "the usual")

---

## 2. The Solution: Semantic Frame Expansion

### 2.1 Core Concept

Instead of comparing short phrases directly, use an LLM to **expand each statement into a full semantic frame** that makes all implicit knowledge explicit.

```
Input: "Doug likes coffee"

Expanded Semantic Frame:
{
  "subject": {
    "entity": "Doug",
    "type": "PERSON",
    "resolution": "primary_user | household_member | unknown"
  },
  "predicate": {
    "relation": "likes",
    "type": "PREFERENCE",
    "strength": "positive_moderate",
    "temporal": "habitual | general"
  },
  "object": {
    "entity": "coffee",
    "type": "BEVERAGE",
    "specificity": "category | brand_unknown"
  },
  "presuppositions": [
    "Doug is capable of having preferences",
    "Doug has experience with coffee",
    "Coffee is available in Doug's context",
    "This is a stable preference (not momentary)"
  ],
  "implications": [
    "Doug would likely accept coffee if offered",
    "Coffee could be a good gift for Doug",
    "Doug may have coffee-making supplies"
  ],
  "negations": [
    "Doug does NOT dislike coffee",
    "Coffee is NOT avoided by Doug"
  ],
  "open_questions": [
    "Does Doug like all types of coffee?",
    "When does Doug drink coffee?",
    "How does Doug take their coffee?"
  ]
}
```

### 2.2 Multi-Variant Generation

Different contexts yield different expansions. Generate **multiple interpretation variants**:

```
Input: "He prefers tea in the morning"

Variant A (if "he" = Doug, based on recent conversation):
{
  "subject": "Doug",
  "potential_conflict": "Doug likes coffee" (established belief)
  "resolution": "Morning-specific exception to general coffee preference"
  "confidence": 0.8
}

Variant B (if "he" = unknown, based on ambiguous reference):
{
  "subject": "Unknown male reference",
  "requires_clarification": true,
  "possible_referents": ["Doug", "other_household_member", "external_person"]
  "confidence": 0.5
}

Variant C (if morning is emphasized, based on temporal framing):
{
  "subject": "Doug",
  "interpretation": "Doug has time-of-day beverage preferences",
  "inference": "Possibly tea AM, coffee PM"
  "confidence": 0.6
}
```

### 2.3 Parallel Exploration

Spin off agents to **explore each variant independently**, then reconcile:

```
┌─────────────────────────────────────────────────────────────────┐
│                    SEMANTIC EXPANSION ENGINE                     │
│                                                                  │
│  Input: "He prefers tea in the morning"                         │
│                          │                                       │
│                          ▼                                       │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              LLM EXPANSION (Frame Semantics)             │    │
│  │  - Extract presuppositions                               │    │
│  │  - Identify ambiguous references                         │    │
│  │  - Generate interpretation variants                       │    │
│  └─────────────────────────────────────────────────────────┘    │
│                          │                                       │
│            ┌─────────────┼─────────────┐                        │
│            ▼             ▼             ▼                        │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐            │
│  │  Variant A   │ │  Variant B   │ │  Variant C   │            │
│  │  Agent       │ │  Agent       │ │  Agent       │            │
│  │              │ │              │ │              │            │
│  │ Explore:     │ │ Explore:     │ │ Explore:     │            │
│  │ "Doug has    │ │ "Reference   │ │ "Time-based  │            │
│  │ morning tea  │ │ unclear,     │ │ preferences  │            │
│  │ exception"   │ │ clarify"     │ │ pattern"     │            │
│  └──────┬───────┘ └──────┬───────┘ └──────┬───────┘            │
│         │                │                │                     │
│         └────────────────┼────────────────┘                     │
│                          ▼                                       │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                 VARIANT RECONCILIATION                   │    │
│  │  - Weight by cognitive factors                           │    │
│  │  - Select most likely interpretation                     │    │
│  │  - Flag alternatives for curiosity                       │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Research Foundation

### 3.1 Presupposition Extraction

Presuppositions are implicit assumptions that must be true for a statement to make sense.

| Statement | Presuppositions |
|-----------|-----------------|
| "John stopped smoking" | John smoked before. John no longer smokes. |
| "The king of France is bald" | There is a king of France. (Classic example: presupposition failure) |
| "Turn on the lights again" | Lights were on before. They're currently off. |

**Linguistic research:** Presupposition triggers include definite descriptions, change-of-state verbs, temporal clauses, cleft sentences, etc.

### 3.2 Frame Semantics (FrameNet)

Frames are conceptual structures that organize background knowledge:

```
Frame: COMMERCIAL_TRANSACTION
  - Buyer: Entity acquiring goods
  - Seller: Entity providing goods
  - Goods: Thing being exchanged
  - Money: Payment given
  - Presuppositions: Ownership transfer, value agreement

"I bought a book" evokes this frame:
  - Buyer: "I"
  - Goods: "a book"
  - Seller: implied (unstated)
  - Money: implied (unstated)
```

### 3.3 Commonsense Knowledge Bases

| Resource | What It Provides |
|----------|------------------|
| **ATOMIC** | If-then inference tuples (X intends Y → X will Z) |
| **COMET** | Neural commonsense inference model trained on ATOMIC |
| **ConceptNet** | Semantic network of commonsense knowledge |
| **GenericsKB** | Generic statements ("birds fly", "coffee is hot") |

Example ATOMIC inferences:
```
Event: "PersonX drinks coffee"
→ PersonX wants: to feel awake, to enjoy the taste
→ PersonX is: a coffee drinker, awake
→ As a result: PersonX feels more alert
→ PersonX needs: coffee, a cup
```

### 3.4 Implicit Knowledge Extraction

Recent LLM research shows models can extract implicit information:

```python
# From GPT-4/Claude studies

prompt = """
Statement: "She finally passed her driving test"

Extract all implicit information:
"""

# Model output:
# - She took the test before and failed
# - She has been trying for some time
# - She can now drive legally
# - She likely practiced between attempts
# - There's relief/celebration implied
# - "Finally" indicates multiple attempts or long wait
```

---

## 4. Architecture Concept

### 4.1 Semantic Expansion Service

```python
@dataclass
class SemanticFrame:
    """A fully expanded semantic representation of a statement."""

    original_text: str

    # Core elements
    entities: list[Entity]  # Named entities with types and resolutions
    relations: list[Relation]  # Semantic relations between entities
    temporal_context: TemporalContext  # When this applies

    # Implicit knowledge
    presuppositions: list[Presupposition]  # Must-be-true assumptions
    implications: list[Implication]  # Likely consequences
    negations: list[str]  # What this explicitly rules out

    # Uncertainty
    ambiguities: list[Ambiguity]  # Unresolved references
    open_questions: list[str]  # What we'd like to know

    # Metadata
    confidence: float  # 0-1, how confident in this expansion
    frame_type: str  # The conceptual frame evoked


@dataclass
class ExpansionVariant:
    """One possible interpretation of a statement."""

    variant_id: str
    frame: SemanticFrame

    # What makes this variant different
    resolution_choices: dict[str, str]  # ambiguity -> chosen resolution
    context_assumptions: list[str]  # Assumptions made

    # Weighting factors
    confidence: float  # Base confidence
    recency_weight: float  # Based on temporal factors
    memory_weight: float  # Based on memory layer support
    belief_weight: float  # Based on existing belief alignment
    relationship_weight: float  # Based on known relationships

    # Combined score
    @property
    def combined_score(self) -> float:
        return (
            self.confidence * 0.3 +
            self.recency_weight * 0.2 +
            self.memory_weight * 0.2 +
            self.belief_weight * 0.2 +
            self.relationship_weight * 0.1
        )


class SemanticExpansionService:
    """Expand short statements into full semantic frames."""

    def __init__(
        self,
        llm: LLMProvider,
        belief_service: BeliefService,
        memory_provider: MemoryProvider,
    ):
        self.llm = llm
        self.belief_service = belief_service
        self.memory_provider = memory_provider

    async def expand(
        self,
        statement: str,
        context: TaskContext,
    ) -> list[ExpansionVariant]:
        """
        Expand a statement into multiple interpretation variants.

        Steps:
        1. Use LLM to extract semantic frame
        2. Identify ambiguities and resolution options
        3. Generate variants for each resolution combination
        4. Weight variants by cognitive factors
        5. Return ranked list of variants
        """

        # Step 1: Initial expansion
        base_frame = await self._extract_frame(statement)

        # Step 2: Identify ambiguities
        ambiguities = await self._find_ambiguities(base_frame, context)

        # Step 3: Generate variants
        variants = await self._generate_variants(base_frame, ambiguities, context)

        # Step 4: Weight by cognitive factors
        weighted_variants = await self._apply_cognitive_weights(variants, context)

        # Step 5: Return sorted
        return sorted(weighted_variants, key=lambda v: v.combined_score, reverse=True)

    async def _extract_frame(self, statement: str) -> SemanticFrame:
        """Use LLM to extract initial semantic frame."""

        prompt = f"""Analyze this statement and extract its full semantic meaning:

Statement: "{statement}"

Extract:
1. ENTITIES: Named entities with types (PERSON, PLACE, THING, TIME, etc.)
2. RELATIONS: How entities relate to each other
3. PRESUPPOSITIONS: What must be true for this to make sense
4. IMPLICATIONS: What this implies or suggests
5. NEGATIONS: What this explicitly rules out
6. AMBIGUITIES: Unclear references or multiple interpretations
7. OPEN QUESTIONS: What information would help clarify

Respond in XML:
<semantic_frame>
    <entities>
        <entity type="PERSON" text="Doug" resolved="false"/>
    </entities>
    <relations>
        <relation subject="Doug" predicate="likes" object="coffee" type="PREFERENCE"/>
    </relations>
    <presuppositions>
        <presupposition>Doug has experience with coffee</presupposition>
    </presuppositions>
    <implications>
        <implication>Doug would accept coffee if offered</implication>
    </implications>
    <negations>
        <negation>Doug does not dislike coffee</negation>
    </negations>
    <ambiguities>
        <ambiguity text="Doug" possibilities="primary_user,household_member,external"/>
    </ambiguities>
    <open_questions>
        <question>What type of coffee does Doug prefer?</question>
    </open_questions>
    <frame_type>PREFERENCE_STATEMENT</frame_type>
    <confidence>0.85</confidence>
</semantic_frame>
"""

        response = await self.llm.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )

        return self._parse_frame_response(response, statement)

    async def _apply_cognitive_weights(
        self,
        variants: list[ExpansionVariant],
        context: TaskContext,
    ) -> list[ExpansionVariant]:
        """Apply cognitive weighting factors to variants."""

        for variant in variants:
            # Recency: Recent references get higher weight
            variant.recency_weight = await self._compute_recency_weight(
                variant, context
            )

            # Memory: Check which memory layers support this interpretation
            variant.memory_weight = await self._compute_memory_weight(
                variant, context
            )

            # Beliefs: Does this align with existing beliefs?
            variant.belief_weight = await self._compute_belief_weight(
                variant, context
            )

            # Relationships: Does entity resolution match known relationships?
            variant.relationship_weight = await self._compute_relationship_weight(
                variant, context
            )

        return variants
```

### 4.2 Parallel Variant Exploration

```python
class VariantExplorationOrchestrator:
    """Explore multiple interpretation variants in parallel."""

    def __init__(
        self,
        expansion_service: SemanticExpansionService,
        parallel_orchestrator: ParallelCognitiveOrchestrator,
    ):
        self.expansion_service = expansion_service
        self.parallel_orchestrator = parallel_orchestrator

    async def explore_interpretations(
        self,
        statement: str,
        context: TaskContext,
        max_variants: int = 3,
    ) -> InterpretationResult:
        """
        Explore multiple interpretations of a statement in parallel.

        Returns the most likely interpretation with alternatives.
        """

        # Get expansion variants
        variants = await self.expansion_service.expand(statement, context)

        # Take top N variants worth exploring
        explore_variants = [v for v in variants[:max_variants] if v.combined_score > 0.3]

        if len(explore_variants) == 1:
            # Only one viable interpretation - use it
            return InterpretationResult(
                primary=explore_variants[0],
                alternatives=[],
                confidence=explore_variants[0].confidence,
                needs_clarification=False,
            )

        # Multiple viable interpretations - explore in parallel
        agents = [
            AgentSpec(
                agent_id=f"variant_explorer_{v.variant_id}",
                role=AgentRole.RESEARCHER,
                context={"variant": v, "original": statement},
            )
            for v in explore_variants
        ]

        # Explore each variant
        result = await self.parallel_orchestrator.orchestrate_parallel(
            agents=agents,
            context=context,
            agent_executor=self._variant_explorer_executor,
        )

        # Reconcile results
        return await self._reconcile_explorations(result, explore_variants)

    async def _variant_explorer_executor(
        self,
        agent: AgentSpec,
        context: TaskContext,
    ) -> AgentResult:
        """Execute exploration of a single variant."""

        variant: ExpansionVariant = agent.context["variant"]
        original: str = agent.context["original"]

        # Check if this variant is consistent with beliefs
        belief_conflicts = await self._check_belief_consistency(variant, context)

        # Check if this variant is consistent with memory
        memory_support = await self._check_memory_support(variant, context)

        # Generate follow-up questions if needed
        clarifications = await self._generate_clarifications(variant, context)

        return AgentResult(
            agent_id=agent.agent_id,
            success=True,
            output={
                "variant_id": variant.variant_id,
                "belief_conflicts": belief_conflicts,
                "memory_support": memory_support,
                "clarifications": clarifications,
                "adjusted_confidence": self._compute_adjusted_confidence(
                    variant, belief_conflicts, memory_support
                ),
            },
        )
```

---

## 5. Integration with draagon-ai

### 5.1 Memory Layers as Weighting Factors

Each memory layer provides different weighting signals:

| Memory Layer | Weighting Role | Example |
|--------------|----------------|---------|
| **Working Memory** | Recency / immediate context | "He" → recent male reference |
| **Episodic Memory** | Conversation patterns | Previous discussions about tea |
| **Semantic Memory** | Established facts | "Doug likes coffee" (stored preference) |
| **Metacognitive Memory** | Self-knowledge | "I'm often wrong about temporal references" |

```python
async def _compute_memory_weight(
    self,
    variant: ExpansionVariant,
    context: TaskContext,
) -> float:
    """Weight variant by support from memory layers."""

    weights = []

    # Working memory: Recent context
    if variant.uses_recent_reference:
        recent_support = await self._check_working_memory(variant, context)
        weights.append(("working", recent_support, 0.3))

    # Episodic memory: Conversation patterns
    episode_support = await self._check_episodic_memory(variant, context)
    weights.append(("episodic", episode_support, 0.2))

    # Semantic memory: Established facts
    fact_support = await self._check_semantic_memory(variant, context)
    weights.append(("semantic", fact_support, 0.35))

    # Metacognitive: How reliable is this interpretation type?
    meta_confidence = await self._check_metacognitive(variant)
    weights.append(("meta", meta_confidence, 0.15))

    return sum(score * weight for _, score, weight in weights)
```

### 5.2 Beliefs as Consistency Checks

Existing beliefs constrain viable interpretations:

```python
async def _compute_belief_weight(
    self,
    variant: ExpansionVariant,
    context: TaskContext,
) -> float:
    """Weight variant by alignment with existing beliefs."""

    # Get relevant beliefs
    beliefs = await self.belief_service.get_beliefs_for_entities(
        [e.text for e in variant.frame.entities]
    )

    conflicts = 0
    supports = 0

    for belief in beliefs:
        # Check if variant aligns with or contradicts belief
        alignment = await self._check_belief_alignment(variant, belief)
        if alignment > 0.5:
            supports += alignment
        elif alignment < -0.5:
            conflicts += abs(alignment)

    # Return score: +1 for strong support, -1 for strong conflict
    if conflicts > 0 and supports == 0:
        return 0.1  # Very low weight for conflicting variant
    elif supports > 0 and conflicts == 0:
        return 0.9  # High weight for supported variant
    else:
        # Mixed: reduce confidence
        return 0.5 + (supports - conflicts) / (supports + conflicts) * 0.4
```

### 5.3 Curiosity for Ambiguity Resolution

When variants are ambiguous, trigger curiosity engine:

```python
async def _handle_ambiguous_interpretation(
    self,
    variants: list[ExpansionVariant],
    context: TaskContext,
) -> None:
    """Queue clarifying questions for ambiguous interpretations."""

    # Find what differs between top variants
    differences = self._find_variant_differences(variants)

    for diff in differences:
        # Create curiosity-driven question
        question = CuriosityQuestion(
            topic=diff.ambiguity_topic,
            question=diff.clarifying_question,
            priority=self._compute_question_priority(diff, variants),
            trigger="semantic_ambiguity",
            context={
                "variants": [v.variant_id for v in variants],
                "ambiguity": diff.ambiguity_type,
            },
        )

        await self.curiosity_service.queue_question(question)
```

### 5.4 Transactive Memory for Entity Resolution

Use transactive memory to resolve "who knows about X":

```python
async def _resolve_entity_reference(
    self,
    entity: Entity,
    context: TaskContext,
) -> list[EntityResolution]:
    """Resolve ambiguous entity references using transactive memory."""

    if entity.type == "PERSON" and not entity.resolved:
        # Check who we know about
        known_people = await self.transactive_memory.get_known_entities("PERSON")

        resolutions = []
        for person in known_people:
            similarity = await self._compute_reference_similarity(
                entity.text, person, context
            )
            if similarity > 0.3:
                resolutions.append(EntityResolution(
                    entity=entity,
                    resolved_to=person,
                    confidence=similarity,
                    reason=f"Matches known person '{person.name}'",
                ))

        return sorted(resolutions, key=lambda r: r.confidence, reverse=True)

    return []
```

---

## 6. Graph Structure Necessity

### 6.1 Why Graphs Now (But Not Before)

| Before (Dict-Based) | Now (Graph-Based) | Why Change |
|---------------------|-------------------|------------|
| Store observations as strings | Store semantic triples | Need relationships |
| Compare text for conflicts | Traverse relationship paths | Deeper conflict detection |
| Simple key-value lookup | Query semantic neighbors | Inference chains |
| Flat belief list | Belief dependency graph | Belief revision propagation |

### 6.2 Graph Structure for Semantic Frames

```
┌─────────────────────────────────────────────────────────────────┐
│                    SEMANTIC KNOWLEDGE GRAPH                      │
│                                                                  │
│  ┌─────────┐    LIKES     ┌─────────┐    TYPE_OF    ┌─────────┐│
│  │  Doug   │─────────────→│ Coffee  │──────────────→│Beverage ││
│  │ (PERSON)│              │ (THING) │               │(CATEGORY)││
│  └────┬────┘              └─────────┘               └─────────┘│
│       │                                                         │
│       │ PREFERS                                                 │
│       │ (context: morning)                                      │
│       ▼                                                         │
│  ┌─────────┐              ┌─────────┐                          │
│  │   Tea   │←─────────────│ Time:   │                          │
│  │ (THING) │  APPLIES_IN  │ Morning │                          │
│  └─────────┘              └─────────┘                          │
│                                                                 │
│  Inference Path:                                                │
│  Doug --PREFERS(morning)--> Tea                                │
│  Doug --LIKES(general)--> Coffee                               │
│  ∴ Doug has time-specific beverage preferences                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 6.3 Graph Queries for Interpretation

```python
class SemanticKnowledgeGraph:
    """Graph-based semantic knowledge representation."""

    async def find_conflicts(
        self,
        new_triple: Triple,
    ) -> list[Conflict]:
        """Find existing triples that conflict with new assertion."""

        # Query: Same subject + same relation type + different object
        query = """
        MATCH (s:Entity {id: $subject})-[r:$relation_type]->(o:Entity)
        WHERE o.id != $object
        RETURN s, r, o
        """

        conflicts = []
        for result in await self.execute(query, new_triple):
            # Check if this is a true conflict or contextual variation
            if await self._is_true_conflict(new_triple, result):
                conflicts.append(Conflict(
                    existing=result,
                    new=new_triple,
                    conflict_type="CONTRADICTORY_ASSERTION",
                ))

        return conflicts

    async def infer_implications(
        self,
        triple: Triple,
        max_depth: int = 3,
    ) -> list[Implication]:
        """Traverse graph to infer implications of an assertion."""

        # Find inference chains
        query = """
        MATCH path = (s:Entity {id: $subject})-[*1..$depth]->(target)
        WHERE ALL(r IN relationships(path) WHERE r.inferrable = true)
        RETURN path, target
        """

        implications = []
        for path in await self.execute(query, triple, depth=max_depth):
            implications.append(Implication(
                source=triple,
                inferred=path.target,
                inference_chain=path.relationships,
                confidence=self._compute_chain_confidence(path),
            ))

        return implications
```

---

## 7. Example Walkthrough

### 7.1 Scenario

User says: "He prefers tea in the morning"

Recent conversation included: "Doug mentioned he was tired today"

Existing belief: "Doug likes coffee" (from semantic memory)

### 7.2 Expansion Process

**Step 1: Initial Frame Extraction**

```xml
<semantic_frame>
    <entities>
        <entity type="PERSON" text="He" resolved="false"/>
        <entity type="BEVERAGE" text="tea" resolved="true"/>
        <entity type="TIME" text="morning" resolved="true"/>
    </entities>
    <relations>
        <relation subject="He" predicate="prefers" object="tea"
                  context="morning" type="CONDITIONAL_PREFERENCE"/>
    </relations>
    <ambiguities>
        <ambiguity text="He" possibilities="Doug,unknown_male"/>
    </ambiguities>
    <presuppositions>
        <presupposition>"He" refers to a known male</presupposition>
        <presupposition>The preference is time-specific</presupposition>
    </presuppositions>
</semantic_frame>
```

**Step 2: Variant Generation**

```
Variant A (He = Doug, aligns with recent reference):
- Resolution: He → Doug
- Interpretation: Doug has a morning-specific tea preference
- Potential conflict: Existing belief "Doug likes coffee"
- Resolution of conflict: Time-specific exception
- Recency weight: 0.9 (Doug mentioned recently)
- Belief weight: 0.6 (partial conflict with coffee preference)
- Combined: 0.75

Variant B (He = unknown, needs clarification):
- Resolution: He → unknown_male_reference
- Interpretation: Someone (not Doug) prefers tea
- No belief conflict
- Recency weight: 0.3 (no clear antecedent)
- Belief weight: 1.0 (no conflict)
- Combined: 0.55

Variant C (He = Doug, challenges existing belief):
- Resolution: He → Doug
- Interpretation: Doug's preference has changed to tea
- Belief update required
- Recency weight: 0.9
- Belief weight: 0.3 (directly contradicts)
- Combined: 0.50
```

**Step 3: Parallel Exploration**

Each variant agent explores:
- Does this interpretation have memory support?
- Does this require belief revision?
- What clarifications would help?

**Step 4: Reconciliation**

```
Winner: Variant A (score 0.75)
- Most consistent with recency + partial belief alignment
- Interpretation: "Doug prefers tea in the morning" (exception to general coffee preference)

Action taken:
1. Create new belief: "Doug prefers tea in the morning" (PREFERENCE, confidence 0.8)
2. Update existing belief: "Doug likes coffee" → add context "general/non-morning"
3. Queue curiosity question: "When did Doug start preferring tea in the morning?"
```

---

## 8. Open Questions

### 8.1 Design Questions

1. **How deep should expansion go?**
   - Expanding every statement fully is expensive
   - When to use full expansion vs. surface comparison?
   - Progressive expansion based on initial ambiguity detection?

2. **How many variants to explore?**
   - 3-5 seems reasonable (like Anthropic's agent count)
   - What's the cutoff confidence for exploration?
   - How to handle long-tail variants?

3. **Graph storage implementation?**
   - Embedded graph (NetworkX) for simplicity?
   - Neo4j for production scale?
   - Qdrant with relationship metadata?

4. **Inference chain limits?**
   - Max depth for graph traversal?
   - How to prevent hallucinated implications?
   - Confidence decay per hop?

### 8.2 Integration Questions

1. **When in the pipeline does expansion happen?**
   - On all observations before storage?
   - On demand when conflicts detected?
   - Background async expansion?

2. **How does this interact with current SharedWorkingMemory?**
   - Replace simple observations with SemanticFrames?
   - Dual storage (raw + expanded)?
   - Lazy expansion on conflict detection?

3. **LLM cost management?**
   - Expansion is LLM-intensive
   - Caching strategies for common patterns?
   - Cheaper model for initial expansion, better for reconciliation?

---

## 9. Relationship to FR-001-005

### 9.1 Current Roadmap

| FR | Focus | Graph Need |
|----|-------|------------|
| FR-001 | Shared Working Memory | Dict-based ✓ |
| FR-002 | Parallel Orchestration | Dict-based ✓ |
| FR-003 | Belief Reconciliation | **Could benefit from graphs** |
| FR-004 | Transactive Memory | Dict-based (expertise routing) |
| FR-005 | Metacognitive Reflection | Dict-based |

### 9.2 Proposed Integration

**Option A: Add as FR-006/007**
- FR-006: Semantic Expansion Service
- FR-007: Knowledge Graph Integration

**Option B: Integrate into FR-003**
- Belief reconciliation naturally benefits from semantic expansion
- Could make FR-003 more sophisticated but larger scope

**Option C: Create Parallel Track**
- Semantic expansion as an enhancement layer
- Can be added to any FR as an optional capability

### 9.3 Recommendation

**Start with Option A** (FR-006/007):
1. Keeps current FR-001-005 focused and achievable
2. Allows prototyping semantic expansion independently
3. Can retrofit to earlier FRs once proven
4. Graph structure becomes natural when needed, not forced

---

## 10. Next Steps

### 10.1 Immediate (This Session)

1. ✅ Document this concept (this file)
2. Run the value-proving tests that were added to FR-002
3. Optionally: Quick prototype of frame extraction prompt

### 10.2 Short-Term (Next Session)

1. Prototype SemanticExpansionService with simple frame extraction
2. Test on challenging examples (the "Doug likes coffee" vs "He prefers tea" case)
3. Evaluate LLM performance and cost

### 10.3 Medium-Term

1. Design graph schema for semantic triples
2. Integrate with one memory layer as proof of concept
3. Write formal requirements for FR-006

### 10.4 Decision Point

After prototyping, decide:
- Is the value worth the complexity?
- Where in roadmap should this live?
- What's the minimal viable implementation?

---

## Appendix A: Related Research

1. **FrameNet Project** - Berkeley frame semantics database
2. **ATOMIC 2020** - Commonsense knowledge graph
3. **COMET** - Neural commonsense reasoning
4. **ConceptNet 5** - Semantic network
5. **PropBank** - Semantic role labeling
6. **AMR (Abstract Meaning Representation)** - Sentence meaning graphs
7. **SRL (Semantic Role Labeling)** - Who did what to whom

---

## Appendix B: Prompt Templates

### Frame Extraction Prompt

```xml
<system>
You are a semantic analysis expert. Given a statement, extract its full
semantic meaning including entities, relations, presuppositions, and
implications.
</system>

<user>
Analyze: "{statement}"

Context:
- Recent conversation: {recent_context}
- Known entities: {known_entities}
- Existing beliefs: {relevant_beliefs}

Extract semantic frame in XML format.
</user>
```

### Variant Comparison Prompt

```xml
<system>
You are evaluating multiple interpretations of a statement.
</system>

<user>
Statement: "{statement}"

Interpretation A: {variant_a_summary}
Interpretation B: {variant_b_summary}

Given context: {context}

Which interpretation is most likely? Why?
What would help resolve the ambiguity?

Respond in XML:
<comparison>
    <preferred>A or B</preferred>
    <confidence>0.0-1.0</confidence>
    <reasoning>Why this interpretation</reasoning>
    <clarifying_question>What to ask to be sure</clarifying_question>
</comparison>
</user>
```

---

**End of Concept Document**

*This document captures the breakthrough insight about semantic expansion and presupposition extraction. It is intended to preserve the concept before prototyping and formalize into requirements.*
