# Runtime Synset Learning Architecture

## Overview

This document describes the architecture for learning new synsets (word definitions) at runtime through agent interactions, curiosity-driven research, and user explanations.

## Problem Statement

The evolving synset database currently:
1. Loads from static JSON files at startup
2. Has an `add_synset()` method but no automated triggers
3. Doesn't integrate with the cognitive memory system
4. Can't discover unknown terms during processing

We need a system that:
1. Detects unknown terms during WSD processing
2. Queues them for later resolution (via research or user inquiry)
3. Learns definitions from agent outputs and user explanations
4. Persists learned synsets for future use
5. Integrates with the curiosity engine for proactive learning

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Query Processing                                │
│                                                                              │
│  "Set up a k8s cluster with ArgoCD"                                         │
│              │                                                               │
│              ▼                                                               │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │  WSD Processing                                                        │  │
│  │  ├─ "k8s" → kubernetes.tech.01 ✓ (alias resolved)                     │  │
│  │  ├─ "cluster" → cluster.n.01 ✓ (WordNet)                              │  │
│  │  └─ "ArgoCD" → ??? (NOT FOUND)                                        │  │
│  │                   │                                                    │  │
│  │                   └─► UnknownTermEvent("ArgoCD", context="GitOps...")  │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          UnknownTermBuffer                                   │
│                     (Working Memory - Layer 1)                               │
│                                                                              │
│  Stores: term, context, timestamp, attempts, resolved                        │
│  TTL: Session duration (5 minutes)                                          │
│  Purpose: Batch collection, deduplication, context gathering                │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
                    ▼                               ▼
┌─────────────────────────────┐   ┌─────────────────────────────────────────┐
│   End-of-Loop Resolution    │   │   Curiosity Engine Integration          │
│                             │   │                                         │
│  After ReAct loop completes:│   │  For terms not resolved in session:     │
│  1. Check agent outputs     │   │  1. Queue as KNOWLEDGE_GAP question     │
│  2. Look for definitions    │   │  2. Priority based on usage frequency   │
│  3. Extract synset data     │   │  3. Ask user OR trigger research agent  │
│  4. Add to EvolvingSynsetDB │   │                                         │
└─────────────────────────────┘   └─────────────────────────────────────────┘
                    │                               │
                    └───────────────┬───────────────┘
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         SynsetLearningService                                │
│                                                                              │
│  Methods:                                                                    │
│  - record_unknown_term(term, context)                                       │
│  - extract_definition_from_output(agent_output, unknown_terms)              │
│  - learn_from_user_explanation(term, explanation, user_id)                  │
│  - research_unknown_term(term, context) → triggers research agent           │
│  - persist_learned_synset(synset, source)                                   │
│                                                                              │
│  Storage Strategy:                                                           │
│  - Working memory: Unknown term buffer (temporary)                          │
│  - Qdrant: Learned synsets (persistent, searchable)                         │
│  - JSON export: Periodic backup, version control                            │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Storage Strategy: Hybrid Approach

### Why Not Just JSON Files?

JSON files are good for:
- Version control (can see what was learned)
- Bootstrap data (curated vocabulary)
- Offline editing
- Portability

But problematic for:
- Concurrent writes from multiple agents
- Searching by embedding (semantic similarity)
- Real-time updates
- Scaling to large vocabularies

### Why Not Just Qdrant?

Qdrant is good for:
- Semantic search (find similar definitions)
- Concurrent access
- Real-time updates
- Integration with existing memory system

But problematic for:
- Version control
- Human curation
- Debugging (can't easily inspect)
- Portability

### The Hybrid Solution

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Synset Storage Layers                               │
│                                                                              │
│  Layer 1: Bootstrap (JSON Files) - Read-only at runtime                     │
│  ├─ data/synsets/cloud_infrastructure.json                                  │
│  ├─ data/synsets/ai_ml.json                                                 │
│  └─ ... (298 curated terms)                                                 │
│  Source: BOOTSTRAP, confidence: 1.0                                          │
│                                                                              │
│  Layer 2: Qdrant Collection "learned_synsets" - Read/Write                  │
│  ├─ Runtime-learned definitions                                             │
│  ├─ User-provided definitions                                               │
│  ├─ Research-discovered definitions                                         │
│  └─ Searchable by embedding                                                 │
│  Source: LLM | USER, confidence: varies                                      │
│                                                                              │
│  Layer 3: Export Queue - Periodic JSON export                               │
│  └─ High-confidence learned synsets → JSON for version control              │
│                                                                              │
│  Priority: USER > BOOTSTRAP > LLM_VERIFIED > LLM_UNVERIFIED                 │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Qdrant Schema for Learned Synsets

```python
# Collection: "learned_synsets"
# Vector: definition embedding (768 dimensions)

payload = {
    "synset_id": "argocd.tech.01",
    "word": "argocd",
    "pos": "n",
    "definition": "A declarative GitOps continuous delivery tool for Kubernetes",
    "examples": ["ArgoCD syncs cluster state with Git repos"],
    "hypernyms": ["gitops_tool.tech.01", "cd_tool.tech.01"],
    "hyponyms": [],
    "aliases": ["argo-cd", "argo_cd"],
    "domain": "DEVOPS_CICD",

    # Learning metadata
    "source": "llm",  # bootstrap | user | llm
    "confidence": 0.85,
    "verified": false,
    "learned_from_context": "User asked about setting up GitOps with ArgoCD",
    "learned_at": "2025-12-31T12:00:00Z",
    "learned_by_agent": "research_agent_01",

    # Reinforcement
    "usage_count": 0,
    "success_count": 0,
    "failure_count": 0,

    # Indexing
    "word_variants": ["argocd", "argo-cd", "argo_cd"],  # For exact match
}
```

## Integration Points

### 1. WSD Processing (wsd.py)

```python
class WordSenseDisambiguator:
    def __init__(
        self,
        wordnet: WordNetInterface,
        llm: LLMProvider,
        synset_learner: SynsetLearningService | None = None,  # NEW
    ):
        self._synset_learner = synset_learner

    async def disambiguate(self, text: str, context: str = "") -> DisambiguationResult:
        result = await self._process_tokens(text, context)

        # NEW: Record unknown terms for later learning
        if self._synset_learner:
            for token in result.tokens:
                if token.synset_id is None and token.is_content_word:
                    await self._synset_learner.record_unknown_term(
                        term=token.word,
                        context=context,
                        pos=token.pos,
                    )

        return result
```

### 2. End-of-Loop Resolution (AgentLoop)

```python
class AgentLoop:
    async def _post_loop_processing(self, result: AgentResponse):
        """Called after ReAct loop completes."""

        # Check for definitions in agent output
        if self._synset_learner:
            unknown_terms = await self._synset_learner.get_pending_terms()

            if unknown_terms:
                learned = await self._synset_learner.extract_definitions_from_output(
                    output=result.response,
                    tool_results=result.tool_results,
                    unknown_terms=unknown_terms,
                )

                for synset in learned:
                    await self._synset_learner.persist_learned_synset(synset)
```

### 3. Curiosity Engine Integration

```python
class CuriosityEngine:
    async def detect_gaps(self, context: ConversationContext) -> list[CuriousQuestion]:
        questions = []

        # NEW: Check for unresolved unknown terms
        if self._synset_learner:
            unresolved = await self._synset_learner.get_unresolved_terms()

            for term in unresolved:
                if term.attempts >= 2:  # Tried twice, still unknown
                    questions.append(CuriousQuestion(
                        question_id=uuid.uuid4().hex,
                        question=f"What does '{term.word}' mean in this context?",
                        question_type=QuestionType.KNOWLEDGE_GAP,
                        priority=QuestionPriority.MEDIUM,
                        context=term.context,
                        purpose=QuestionPurpose.DEEPEN_UNDERSTANDING,
                        why_asking=f"I encountered '{term.word}' but couldn't find a definition",
                        follow_up_plan="I'll remember this for future conversations",
                    ))

        return questions
```

### 4. User Explanation Learning

```python
class LearningService:
    async def process_conversation(
        self,
        messages: list[Message],
        result: AgentResponse,
    ) -> list[LearningOutcome]:
        outcomes = []

        # NEW: Detect definition patterns
        definition_result = await self._detect_user_definitions(messages)

        if definition_result.found_definitions:
            for defn in definition_result.definitions:
                synset = LearnedSynset(
                    synset_id=f"{defn.term}.user.01",
                    word=defn.term,
                    pos=defn.pos or "n",
                    definition=defn.explanation,
                    source=SynsetSource.USER,
                    confidence=0.95,  # High confidence for user-provided
                )
                await self._synset_learner.persist_learned_synset(synset)
                outcomes.append(LearningOutcome(
                    type=LearningType.FACT,
                    content=f"Learned definition of '{defn.term}'",
                ))

        return outcomes
```

### 5. Research Agent Trigger

```python
class SynsetLearningService:
    async def research_unknown_term(
        self,
        term: str,
        context: str,
    ) -> LearnedSynset | None:
        """Trigger background research agent to find definition."""

        # Use web search or documentation lookup
        research_prompt = f"""
        Research the term "{term}" in the context: {context}

        Provide:
        1. A concise definition (1-2 sentences)
        2. Part of speech
        3. 1-2 example sentences
        4. Related/parent concepts (hypernyms)
        5. Domain/category

        Format as structured data.
        """

        # This would trigger a background agent task
        result = await self._research_agent.execute(research_prompt)

        if result.success:
            return self._parse_research_result(term, result.output)

        return None
```

## New Data Types

```python
@dataclass
class UnknownTermRecord:
    """A term that couldn't be resolved during WSD."""
    term: str
    context: str
    pos: str | None
    timestamp: datetime
    session_id: str
    attempts: int = 1
    resolved: bool = False
    resolution_source: str | None = None  # "agent_output" | "user" | "research"

@dataclass
class DefinitionExtraction:
    """A definition extracted from agent output or user explanation."""
    term: str
    explanation: str
    pos: str | None
    examples: list[str]
    confidence: float
    source_type: str  # "agent_output" | "user_explanation" | "research"

class SynsetLearningService:
    """Service for runtime synset learning."""

    def __init__(
        self,
        memory: MemoryProvider,
        evolving_db: EvolvingSynsetDatabase,
        qdrant_store: QdrantSynsetStore | None = None,
        llm: LLMProvider | None = None,
        curiosity_engine: CuriosityEngine | None = None,
    ):
        self._memory = memory
        self._evolving_db = evolving_db
        self._qdrant = qdrant_store
        self._llm = llm
        self._curiosity = curiosity_engine

        # In-memory buffer for current session
        self._unknown_terms: dict[str, UnknownTermRecord] = {}

    async def record_unknown_term(
        self,
        term: str,
        context: str,
        pos: str | None = None,
        session_id: str = "",
    ) -> None:
        """Record a term that couldn't be resolved."""
        key = term.lower()

        if key in self._unknown_terms:
            self._unknown_terms[key].attempts += 1
            return

        self._unknown_terms[key] = UnknownTermRecord(
            term=term,
            context=context,
            pos=pos,
            timestamp=datetime.now(),
            session_id=session_id,
        )

        # Also store in working memory for cross-agent visibility
        await self._memory.store(
            content=f"Unknown term: {term}",
            memory_type=MemoryType.OBSERVATION,
            scope=MemoryScope.SESSION,
            metadata={
                "term": term,
                "context": context,
                "type": "unknown_synset",
            },
        )

    async def extract_definitions_from_output(
        self,
        output: str,
        tool_results: list[dict],
        unknown_terms: list[str],
    ) -> list[LearnedSynset]:
        """Use LLM to extract definitions from agent output."""

        if not self._llm or not unknown_terms:
            return []

        prompt = f"""
        The following terms were unknown during processing: {unknown_terms}

        Agent output:
        {output}

        Tool results:
        {tool_results}

        For each unknown term that was explained or defined in the output,
        extract the definition in this format:

        <definitions>
          <definition>
            <term>the term</term>
            <explanation>concise definition</explanation>
            <pos>n|v|adj|adv</pos>
            <examples>
              <example>Example sentence 1</example>
            </examples>
            <hypernyms>parent_concept_1, parent_concept_2</hypernyms>
            <domain>TECHNOLOGY|AI_ML|etc</domain>
            <confidence>0.0-1.0</confidence>
          </definition>
        </definitions>

        Only include terms that were actually explained. If a term wasn't
        defined, don't include it.
        """

        response = await self._llm.chat([{"role": "user", "content": prompt}])
        return self._parse_definition_response(response)

    async def persist_learned_synset(
        self,
        synset: LearnedSynset,
        to_qdrant: bool = True,
    ) -> str:
        """Persist a learned synset to storage."""

        # Add to in-memory evolving database
        self._evolving_db.add_synset(synset)

        # Store in Qdrant for persistence and search
        if to_qdrant and self._qdrant:
            await self._qdrant.store_synset(synset)

        # Also store as a memory for the learning record
        await self._memory.store(
            content=f"Learned: {synset.word} = {synset.definition}",
            memory_type=MemoryType.KNOWLEDGE,
            scope=MemoryScope.AGENT,
            confidence=synset.confidence,
            metadata={
                "synset_id": synset.synset_id,
                "type": "learned_synset",
                "source": synset.source.value,
            },
        )

        # Mark as resolved
        key = synset.word.lower()
        if key in self._unknown_terms:
            self._unknown_terms[key].resolved = True
            self._unknown_terms[key].resolution_source = synset.source.value

        return synset.synset_id
```

## QdrantSynsetStore

```python
class QdrantSynsetStore:
    """Qdrant-backed storage for learned synsets."""

    COLLECTION_NAME = "learned_synsets"

    def __init__(
        self,
        client: QdrantClient,
        embedding_provider: EmbeddingProvider,
    ):
        self._client = client
        self._embedder = embedding_provider

    async def store_synset(self, synset: LearnedSynset) -> str:
        """Store a synset with its definition embedding."""

        # Embed the definition for semantic search
        embedding = await self._embedder.embed(synset.definition)

        # Build payload
        payload = {
            "synset_id": synset.synset_id,
            "word": synset.word,
            "pos": synset.pos,
            "definition": synset.definition,
            "examples": synset.examples,
            "hypernyms": synset.hypernyms,
            "hyponyms": synset.hyponyms,
            "aliases": synset.aliases,
            "domain": synset.domain,
            "source": synset.source.value,
            "confidence": synset.confidence,
            "usage_count": synset.usage_count,
            "success_count": synset.success_count,
            "failure_count": synset.failure_count,
            "word_variants": [synset.word.lower()] + [a.lower() for a in synset.aliases],
            "learned_at": datetime.now().isoformat(),
        }

        # Upsert to Qdrant
        self._client.upsert(
            collection_name=self.COLLECTION_NAME,
            points=[{
                "id": synset.synset_id,
                "vector": embedding,
                "payload": payload,
            }],
        )

        return synset.synset_id

    async def search_by_word(self, word: str) -> list[LearnedSynset]:
        """Exact match search by word or alias."""

        results = self._client.scroll(
            collection_name=self.COLLECTION_NAME,
            scroll_filter=models.Filter(
                should=[
                    models.FieldCondition(
                        key="word_variants",
                        match=models.MatchValue(value=word.lower()),
                    ),
                ],
            ),
        )

        return [self._payload_to_synset(r.payload) for r in results[0]]

    async def search_similar(
        self,
        definition: str,
        limit: int = 5,
    ) -> list[tuple[LearnedSynset, float]]:
        """Semantic search for similar definitions."""

        embedding = await self._embedder.embed(definition)

        results = self._client.search(
            collection_name=self.COLLECTION_NAME,
            query_vector=embedding,
            limit=limit,
        )

        return [(self._payload_to_synset(r.payload), r.score) for r in results]

    async def load_all(self) -> list[LearnedSynset]:
        """Load all synsets from Qdrant (for initializing EvolvingSynsetDatabase)."""

        synsets = []
        offset = None

        while True:
            results, offset = self._client.scroll(
                collection_name=self.COLLECTION_NAME,
                offset=offset,
                limit=100,
            )

            synsets.extend(self._payload_to_synset(r.payload) for r in results)

            if offset is None:
                break

        return synsets
```

## Testing Strategy

### Unit Tests

```python
class TestSynsetLearningService:
    """Tests for runtime synset learning."""

    async def test_record_unknown_term(self):
        """Unknown terms are recorded in buffer and memory."""
        service = SynsetLearningService(
            memory=MockMemoryProvider(),
            evolving_db=EvolvingSynsetDatabase(),
        )

        await service.record_unknown_term(
            term="ArgoCD",
            context="Setting up GitOps pipeline",
        )

        assert "argocd" in service._unknown_terms
        assert service._unknown_terms["argocd"].attempts == 1

    async def test_extract_definition_from_output(self):
        """Definitions are extracted from agent output."""
        service = SynsetLearningService(
            memory=MockMemoryProvider(),
            evolving_db=EvolvingSynsetDatabase(),
            llm=MockLLMProvider(response=MOCK_DEFINITION_RESPONSE),
        )

        synsets = await service.extract_definitions_from_output(
            output="ArgoCD is a GitOps tool that...",
            tool_results=[],
            unknown_terms=["ArgoCD"],
        )

        assert len(synsets) == 1
        assert synsets[0].word == "argocd"

    async def test_persist_to_qdrant_and_evolving_db(self):
        """Learned synsets go to both Qdrant and evolving DB."""
        evolving_db = EvolvingSynsetDatabase()
        qdrant_store = MockQdrantSynsetStore()

        service = SynsetLearningService(
            memory=MockMemoryProvider(),
            evolving_db=evolving_db,
            qdrant_store=qdrant_store,
        )

        synset = LearnedSynset(
            synset_id="argocd.tech.01",
            word="argocd",
            definition="GitOps CD tool",
            source=SynsetSource.LLM,
        )

        await service.persist_learned_synset(synset)

        assert evolving_db.has_word("argocd")
        assert qdrant_store.stored_count == 1
```

### Integration Tests

```python
class TestSynsetLearningIntegration:
    """Integration tests with real Qdrant."""

    @pytest.fixture
    async def qdrant_store(self):
        """Real Qdrant connection for integration tests."""
        client = QdrantClient(":memory:")  # In-memory for tests

        # Create collection
        client.create_collection(
            collection_name="learned_synsets",
            vectors_config=models.VectorParams(
                size=768,
                distance=models.Distance.COSINE,
            ),
        )

        return QdrantSynsetStore(client, MockEmbedder())

    async def test_full_learning_flow(self, qdrant_store):
        """Test complete flow: unknown → extract → persist → retrieve."""

        evolving_db = EvolvingSynsetDatabase()
        service = SynsetLearningService(
            memory=MockMemoryProvider(),
            evolving_db=evolving_db,
            qdrant_store=qdrant_store,
            llm=RealLLMProvider(),  # Use real LLM for extraction
        )

        # 1. Record unknown term
        await service.record_unknown_term(
            term="Pulumi",
            context="Infrastructure as code using Python",
        )

        # 2. Simulate agent output that explains it
        synsets = await service.extract_definitions_from_output(
            output="""
            Pulumi is an infrastructure as code platform that allows you to
            define cloud resources using general-purpose programming languages
            like Python, TypeScript, and Go instead of domain-specific languages.
            """,
            tool_results=[],
            unknown_terms=["Pulumi"],
        )

        assert len(synsets) == 1

        # 3. Persist
        await service.persist_learned_synset(synsets[0])

        # 4. Verify in evolving DB
        assert evolving_db.has_word("pulumi")

        # 5. Verify in Qdrant
        qdrant_results = await qdrant_store.search_by_word("pulumi")
        assert len(qdrant_results) == 1

        # 6. Verify semantic search works
        similar = await qdrant_store.search_similar(
            "tool for defining infrastructure using code",
        )
        assert any(s.word == "pulumi" for s, _ in similar)
```

### End-to-End Tests

```python
class TestCuriosityDrivenLearning:
    """Test curiosity engine → research → learning flow."""

    async def test_curiosity_triggers_research(self):
        """Unresolved terms trigger curiosity questions."""

        curiosity = CuriosityEngine(...)
        service = SynsetLearningService(
            memory=MockMemoryProvider(),
            evolving_db=EvolvingSynsetDatabase(),
            curiosity_engine=curiosity,
        )

        # Record term that won't be resolved
        await service.record_unknown_term("FluxCD", "GitOps deployment")
        await service.record_unknown_term("FluxCD", "GitOps deployment")  # 2nd attempt

        # Get curiosity questions
        questions = await curiosity.detect_gaps(MockContext())

        flux_questions = [q for q in questions if "FluxCD" in q.question]
        assert len(flux_questions) == 1
        assert flux_questions[0].question_type == QuestionType.KNOWLEDGE_GAP
```

## File Structure

```
prototypes/implicit_knowledge_graphs/
├── src/
│   ├── synset_learning.py          # NEW: SynsetLearningService
│   ├── qdrant_synset_store.py      # NEW: Qdrant storage adapter
│   ├── evolving_synsets.py         # MODIFIED: Add load_from_qdrant()
│   └── wsd.py                      # MODIFIED: Integration point
├── tests/
│   ├── test_synset_learning.py     # NEW: Unit tests
│   ├── test_qdrant_synsets.py      # NEW: Qdrant integration tests
│   └── test_curiosity_learning.py  # NEW: E2E tests
└── docs/
    └── specs/
        └── RUNTIME_SYNSET_LEARNING.md  # This document
```

## Migration Path

1. **Phase 1**: Implement SynsetLearningService with in-memory buffer only
2. **Phase 2**: Add QdrantSynsetStore for persistence
3. **Phase 3**: Integrate with WSD for unknown term detection
4. **Phase 4**: Integrate with CuriosityEngine for proactive research
5. **Phase 5**: Add export mechanism for high-confidence synsets → JSON

## Open Questions

1. **Deduplication**: How do we handle when user provides a definition that conflicts with bootstrap data?
   - Proposal: User definitions get stored separately, preference setting determines which is used

2. **Verification**: Should LLM-learned definitions be verified before use?
   - Proposal: Start with low confidence (0.6), boost on successful usage

3. **Cross-agent learning**: If one agent learns a term, should others see it?
   - Proposal: Qdrant storage is shared, but confidence may vary by agent

4. **Garbage collection**: How do we remove low-quality learned synsets?
   - Proposal: Periodic cleanup of synsets with success_rate < 0.3 after 10+ uses
