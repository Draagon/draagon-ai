#!/usr/bin/env python3
"""Pipeline Optimization Experiments.

This benchmark systematically tests different configurations of the retrieval pipeline
to identify which optimizations actually improve performance.

Experimental Variables:
1. Phase 1/0 Semantic Extraction:
   - ALL: All approaches use semantic extraction (normalized terminology)
   - NONE: No approaches use semantic extraction (raw text matching)
   - MIXED: Current state (Graph + Vector use extraction, Raw does not)

2. Query Expansion:
   - NONE: Use query as-is
   - SERIAL: Expand query, then search
   - PARALLEL: Generate multiple expansions, search all in parallel

3. SharedMemory Normalization:
   - RAW: Store observations as raw text
   - EXTRACTED: Store observations with semantic extraction (Phase 1/0)

4. Document Pre-processing:
   - RAW: Index documents as-is
   - EXTRACTED: Pre-extract entities/concepts from documents

Run with:
    GROQ_API_KEY=your_key python3.11 tests/integration/agents/benchmark_pipeline_experiments.py
"""

import asyncio
import os
import sys
import time
import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

# Add project to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))


# =============================================================================
# Experiment Configuration
# =============================================================================

class SemanticExtractionMode(str, Enum):
    """How semantic extraction is applied across approaches."""
    ALL = "all"        # All approaches use Phase 1/0 extraction
    NONE = "none"      # No approaches use extraction
    MIXED = "mixed"    # Current: Graph + Vector use it, Raw does not


class QueryExpansionMode(str, Enum):
    """How query expansion is performed."""
    NONE = "none"          # No expansion
    SERIAL = "serial"      # Expand then search
    PARALLEL = "parallel"  # Multiple expansions searched in parallel


class SharedMemoryMode(str, Enum):
    """How shared memory stores observations."""
    RAW = "raw"            # Store raw text
    EXTRACTED = "extracted"  # Store with semantic extraction


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""
    name: str
    semantic_mode: SemanticExtractionMode
    expansion_mode: QueryExpansionMode
    memory_mode: SharedMemoryMode
    description: str


# Define experiment matrix
EXPERIMENTS = [
    # Baseline (current implementation)
    ExperimentConfig(
        name="baseline_mixed",
        semantic_mode=SemanticExtractionMode.MIXED,
        expansion_mode=QueryExpansionMode.NONE,
        memory_mode=SharedMemoryMode.RAW,
        description="Current state: Graph/Vector extracted, Raw not",
    ),

    # Semantic extraction experiments
    ExperimentConfig(
        name="all_extracted",
        semantic_mode=SemanticExtractionMode.ALL,
        expansion_mode=QueryExpansionMode.NONE,
        memory_mode=SharedMemoryMode.RAW,
        description="All approaches use Phase 1/0 extraction",
    ),
    ExperimentConfig(
        name="none_extracted",
        semantic_mode=SemanticExtractionMode.NONE,
        expansion_mode=QueryExpansionMode.NONE,
        memory_mode=SharedMemoryMode.RAW,
        description="No approaches use extraction (raw matching)",
    ),

    # Query expansion experiments
    ExperimentConfig(
        name="serial_expansion",
        semantic_mode=SemanticExtractionMode.MIXED,
        expansion_mode=QueryExpansionMode.SERIAL,
        memory_mode=SharedMemoryMode.RAW,
        description="Expand query before searching",
    ),
    ExperimentConfig(
        name="parallel_expansion",
        semantic_mode=SemanticExtractionMode.MIXED,
        expansion_mode=QueryExpansionMode.PARALLEL,
        memory_mode=SharedMemoryMode.RAW,
        description="Multiple query expansions in parallel",
    ),

    # Shared memory normalization
    ExperimentConfig(
        name="extracted_memory",
        semantic_mode=SemanticExtractionMode.MIXED,
        expansion_mode=QueryExpansionMode.NONE,
        memory_mode=SharedMemoryMode.EXTRACTED,
        description="SharedMemory stores extracted observations",
    ),

    # Combined: All extracted + parallel expansion
    ExperimentConfig(
        name="full_optimization",
        semantic_mode=SemanticExtractionMode.ALL,
        expansion_mode=QueryExpansionMode.PARALLEL,
        memory_mode=SharedMemoryMode.EXTRACTED,
        description="All optimizations enabled",
    ),
]


# =============================================================================
# Shared Types
# =============================================================================

class QueryDifficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    EXPERT = "expert"


@dataclass
class Document:
    """A document in the corpus."""
    id: str
    content: str
    source: str
    doc_type: str
    entities: list[str] = field(default_factory=list)
    # Extracted data (populated when semantic_mode != NONE)
    extracted_entities: list[str] = field(default_factory=list)
    extracted_concepts: list[str] = field(default_factory=list)


@dataclass
class TestCase:
    """Test case for benchmarking."""
    query: str
    expected_sources: list[str]
    expected_content: list[str]
    difficulty: QueryDifficulty
    description: str


@dataclass
class ExperimentResult:
    """Result from running an experiment."""
    config: ExperimentConfig
    total_recall: float
    total_source_accuracy: float
    avg_latency_ms: float
    wins_by_approach: dict[str, int]
    recall_by_difficulty: dict[str, float]
    per_query_results: list[dict]


# =============================================================================
# Semantic Extractor (Phase 1/0)
# =============================================================================

class SemanticExtractor:
    """Extracts normalized entities and concepts from text."""

    def __init__(self, llm):
        self.llm = llm
        self._cache = {}  # Cache extractions to avoid redundant LLM calls

    def _parse_section(self, content: str, prefix: str) -> list[str]:
        """Parse a section that may be comma-separated or bullet-point format.

        Handles both:
            PREFIX: item1, item2, item3
        And:
            PREFIX:
            - item1
            - item2
        And:
            PREFIX:
            1. item1
            2. item2
        """
        lines = content.split("\n")
        items = []
        in_section = False

        for i, line in enumerate(lines):
            line = line.strip()

            if line.upper().startswith(prefix.upper()):
                in_section = True
                # Check if items are on same line (comma-separated)
                after_prefix = line[len(prefix):].strip()
                if after_prefix.startswith(":"):
                    after_prefix = after_prefix[1:].strip()
                if after_prefix:
                    # Comma-separated on same line
                    items.extend([x.strip() for x in after_prefix.split(",") if x.strip()])
                continue

            if in_section:
                # Check if we've hit another section
                if any(line.upper().startswith(p) for p in ["ENTITIES", "CONCEPTS", "NORMALIZED", "EXPANDED"]):
                    break

                # Parse bullet or numbered items
                if line.startswith("-") or line.startswith("*"):
                    item = line[1:].strip()
                    if item:
                        items.append(item)
                elif line and line[0].isdigit() and "." in line[:3]:
                    # Numbered list (e.g., "1. item")
                    item = line.split(".", 1)[1].strip() if "." in line else line
                    if item:
                        items.append(item)
                elif line and not line.startswith("#"):
                    # Plain text continuation (not a header)
                    items.append(line)

        return items

    async def extract(self, text: str, use_cache: bool = True) -> dict:
        """Extract entities and concepts from text.

        Returns:
            {
                "entities": ["entity1", "entity2", ...],
                "concepts": ["concept1", "concept2", ...],
                "normalized_text": "text with normalized terminology"
            }
        """
        cache_key = hash(text[:500])  # Use first 500 chars as key
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]

        prompt = f"""Extract key entities and concepts from this text. Output in this exact format:

ENTITIES: entity1, entity2, entity3
CONCEPTS: concept1, concept2, concept3
NORMALIZED: [rewrite the key terms using consistent terminology]

Text: {text[:2000]}

Remember:
- ENTITIES are specific things (systems, features, components, people, projects)
- CONCEPTS are abstract ideas (patterns, architectures, approaches)
- NORMALIZED rewrites ambiguous terms to standard forms (e.g., "SMS system" -> "dual-channel SMS")
"""

        response = await self.llm.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=300,
        )
        content = response.content if hasattr(response, "content") else str(response)

        # Parse response using robust multi-format parser
        result = {
            "entities": self._parse_section(content, "ENTITIES"),
            "concepts": self._parse_section(content, "CONCEPTS"),
            "normalized_text": text[:500],
        }

        # Get normalized text (single line after NORMALIZED:)
        for line in content.split("\n"):
            line = line.strip()
            if line.startswith("NORMALIZED:"):
                result["normalized_text"] = line[11:].strip() or text[:500]
                break

        if use_cache:
            self._cache[cache_key] = result

        return result

    async def extract_query(self, query: str) -> dict:
        """Extract from a query (more focused extraction)."""
        prompt = f"""Extract the key search terms from this question:

Question: {query}

Output format:
ENTITIES: specific things being asked about
CONCEPTS: abstract ideas being asked about
EXPANDED: alternative phrasings of the same question (2-3 variations)
"""

        response = await self.llm.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=200,
        )
        content = response.content if hasattr(response, "content") else str(response)

        # Parse response using robust multi-format parser
        result = {
            "entities": self._parse_section(content, "ENTITIES"),
            "concepts": self._parse_section(content, "CONCEPTS"),
            "expansions": self._parse_section(content, "EXPANDED"),
        }

        return result


# =============================================================================
# Configurable Retrieval Approaches
# =============================================================================

class ConfigurableRawContext:
    """Raw context approach with optional semantic extraction."""

    def __init__(self, llm, documents: list[Document], extractor: SemanticExtractor | None):
        self.llm = llm
        self.documents = documents
        self.extractor = extractor
        self.use_extraction = extractor is not None

    async def retrieve(self, query: str, query_extraction: dict | None, k: int = 5) -> list[tuple[Document, float]]:
        """Retrieve documents with optional extraction-based matching."""

        if self.use_extraction and query_extraction:
            # Use extracted entities/concepts for matching
            query_terms = set(
                [e.lower() for e in query_extraction.get("entities", [])] +
                [c.lower() for c in query_extraction.get("concepts", [])]
            )
        else:
            # Use raw word matching
            query_terms = set(query.lower().split())

        scored = []
        for doc in self.documents:
            if self.use_extraction and doc.extracted_entities:
                # Match against extracted entities
                doc_terms = set(
                    [e.lower() for e in doc.extracted_entities] +
                    [c.lower() for c in doc.extracted_concepts]
                )
            else:
                # Match against raw content
                doc_terms = set(doc.content[:2000].lower().split())

            overlap = len(query_terms & doc_terms)
            if overlap > 0:
                scored.append((doc, overlap / max(len(query_terms), 1)))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]


class ConfigurableVectorRAG:
    """Vector RAG with optional semantic extraction."""

    def __init__(self, llm, embedder, documents: list[Document], extractor: SemanticExtractor | None):
        self.llm = llm
        self.embedder = embedder
        self.documents = documents
        self.extractor = extractor
        self.use_extraction = extractor is not None
        self.embedded_docs: list[tuple[Document, list[float]]] = []

    async def setup(self):
        """Pre-embed all documents."""
        print("    Embedding documents for Vector/RAG...")
        for i, doc in enumerate(self.documents):
            # Use extracted text if available, otherwise raw
            if self.use_extraction and doc.extracted_concepts:
                text = " ".join(doc.extracted_entities + doc.extracted_concepts)[:1500]
            else:
                text = doc.content[:1500]

            try:
                embedding = await self.embedder.embed(text)
                self.embedded_docs.append((doc, embedding))
            except Exception as e:
                print(f"      Warning: Failed to embed {doc.id}: {e}")

            if (i + 1) % 10 == 0:
                print(f"      Embedded {i + 1}/{len(self.documents)}")

        print(f"    Embedded {len(self.embedded_docs)} documents")

    async def retrieve(self, query: str, query_extraction: dict | None, k: int = 5) -> list[tuple[Document, float]]:
        """Retrieve documents via vector similarity."""
        import math

        # Embed query (use extracted terms if available)
        if self.use_extraction and query_extraction:
            query_text = " ".join(
                query_extraction.get("entities", []) +
                query_extraction.get("concepts", [])
            ) or query
        else:
            query_text = query

        query_embedding = await self.embedder.embed(query_text)

        def cosine_sim(a, b):
            dot = sum(x * y for x, y in zip(a, b))
            norm_a = math.sqrt(sum(x * x for x in a))
            norm_b = math.sqrt(sum(x * x for x in b))
            return dot / (norm_a * norm_b) if norm_a > 0 and norm_b > 0 else 0

        scored = [(doc, cosine_sim(query_embedding, emb)) for doc, emb in self.embedded_docs]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]


class ConfigurableSemanticGraph:
    """Semantic graph approach with optional extraction normalization."""

    def __init__(self, llm, documents: list[Document], extractor: SemanticExtractor | None):
        self.llm = llm
        self.documents = documents
        self.extractor = extractor
        self.use_extraction = extractor is not None

        # Build entity index
        self.entity_to_docs: dict[str, list[Document]] = {}
        for doc in documents:
            entities = doc.extracted_entities if self.use_extraction else doc.entities
            for entity in entities:
                key = entity.lower()
                if key not in self.entity_to_docs:
                    self.entity_to_docs[key] = []
                self.entity_to_docs[key].append(doc)

    async def retrieve(self, query: str, query_extraction: dict | None, k: int = 5) -> list[tuple[Document, float]]:
        """Retrieve documents via entity matching."""

        # Get query entities
        if self.use_extraction and query_extraction:
            query_entities = query_extraction.get("entities", [])
        else:
            # Extract entities using LLM
            query_entities = await self._extract_entities_llm(query)

        # Find matching documents
        doc_scores: dict[str, float] = {}
        for entity in query_entities:
            key = entity.lower()
            # Exact match
            if key in self.entity_to_docs:
                for doc in self.entity_to_docs[key]:
                    doc_scores[doc.id] = doc_scores.get(doc.id, 0) + 1.0
            # Fuzzy match
            for stored_key in self.entity_to_docs:
                if key in stored_key or stored_key in key:
                    for doc in self.entity_to_docs[stored_key]:
                        doc_scores[doc.id] = doc_scores.get(doc.id, 0) + 0.5

        # Get documents and scores
        scored = []
        for doc in self.documents:
            if doc.id in doc_scores:
                scored.append((doc, doc_scores[doc.id] / max(len(query_entities), 1)))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

    async def _extract_entities_llm(self, query: str) -> list[str]:
        """Extract entities from query using LLM."""
        prompt = f"""Extract key entities from this query. One per line, no explanations.

Query: {query}

Entities:"""

        response = await self.llm.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=100,
        )
        content = response.content if hasattr(response, "content") else str(response)

        entities = []
        for line in content.strip().split("\n"):
            line = line.strip().lstrip("-*• ")
            if line and len(line) < 50:
                entities.append(line)

        return entities


# =============================================================================
# Configurable Hybrid Pipeline
# =============================================================================

class ConfigurableHybridPipeline:
    """Hybrid pipeline with configurable extraction and expansion."""

    def __init__(
        self,
        llm,
        embedder,
        documents: list[Document],
        config: ExperimentConfig,
        extractor: SemanticExtractor | None,
    ):
        self.llm = llm
        self.embedder = embedder
        self.documents = documents
        self.config = config
        self.extractor = extractor

        # Create configurable approaches
        use_extraction = config.semantic_mode != SemanticExtractionMode.NONE
        ext = extractor if use_extraction else None

        self.raw = ConfigurableRawContext(llm, documents,
            ext if config.semantic_mode == SemanticExtractionMode.ALL else None)
        self.vector = ConfigurableVectorRAG(llm, embedder, documents, ext)
        self.graph = ConfigurableSemanticGraph(llm, documents, ext)

    async def setup(self):
        """Initialize approaches."""
        await self.vector.setup()

    async def retrieve(
        self,
        query: str,
        tc: TestCase,
    ) -> tuple[list[Document], float, str]:
        """Run retrieval with configured pipeline.

        Returns:
            (retrieved_docs, recall, answer)
        """
        from draagon_ai.orchestration.shared_memory import (
            SharedWorkingMemory,
            SharedWorkingMemoryConfig,
        )
        from draagon_ai.orchestration.multi_agent_orchestrator import AgentRole

        start = time.perf_counter()

        # Step 1: Extract query (if enabled)
        query_extraction = None
        if self.extractor and self.config.semantic_mode != SemanticExtractionMode.NONE:
            query_extraction = await self.extractor.extract_query(query)

        # Step 2: Expand query (if enabled)
        queries_to_search = [query]
        if self.config.expansion_mode == QueryExpansionMode.SERIAL:
            # Generate one expansion, add to list
            if query_extraction and query_extraction.get("expansions"):
                queries_to_search.extend(query_extraction["expansions"][:2])
        elif self.config.expansion_mode == QueryExpansionMode.PARALLEL:
            # Generate multiple expansions
            if query_extraction and query_extraction.get("expansions"):
                queries_to_search.extend(query_extraction["expansions"][:3])

        # Step 3: Create shared memory
        memory_config = SharedWorkingMemoryConfig(
            max_items_per_agent=10,
            max_total_items=30,
        )
        shared_memory = SharedWorkingMemory(
            task_id=f"exp_{hash(query)}",
            config=memory_config,
        )

        # Step 4: Run all approaches (potentially with multiple queries)
        all_results: dict[str, list[tuple[Document, float]]] = {
            "raw": [],
            "vector": [],
            "graph": [],
        }

        async def search_with_query(q: str, q_ext: dict | None):
            """Run all approaches for a single query."""
            raw_r = await self.raw.retrieve(q, q_ext)
            vec_r = await self.vector.retrieve(q, q_ext)
            graph_r = await self.graph.retrieve(q, q_ext)
            return raw_r, vec_r, graph_r

        if self.config.expansion_mode == QueryExpansionMode.PARALLEL and len(queries_to_search) > 1:
            # Run all queries in parallel
            tasks = [search_with_query(q, query_extraction) for q in queries_to_search]
            results = await asyncio.gather(*tasks)
            for raw_r, vec_r, graph_r in results:
                all_results["raw"].extend(raw_r)
                all_results["vector"].extend(vec_r)
                all_results["graph"].extend(graph_r)
        else:
            # Run single query (or serial expansion)
            for q in queries_to_search:
                raw_r, vec_r, graph_r = await search_with_query(q, query_extraction)
                all_results["raw"].extend(raw_r)
                all_results["vector"].extend(vec_r)
                all_results["graph"].extend(graph_r)

        # Step 5: Add observations to shared memory
        for approach, results in all_results.items():
            for doc, score in results[:5]:
                # Optionally extract observation content
                if self.config.memory_mode == SharedMemoryMode.EXTRACTED and self.extractor:
                    obs_content = f"[doc:{doc.id}] Entities: {', '.join(doc.extracted_entities[:3])}"
                else:
                    obs_content = f"[doc:{doc.id}] {approach} found {doc.doc_type} from {doc.source}"

                await shared_memory.add_observation(
                    content=obs_content,
                    source_agent_id=approach,
                    attention_weight=min(0.9, 0.5 + score * 0.4),
                    is_belief_candidate=True,
                    belief_type="EVIDENCE",
                )

        # Step 6: Aggregate with multi-source boosting
        doc_scores: dict[str, tuple[float, set]] = {}
        for approach, results in all_results.items():
            for doc, score in results:
                if doc.id not in doc_scores:
                    doc_scores[doc.id] = (0.0, set())
                current_score, sources = doc_scores[doc.id]
                doc_scores[doc.id] = (max(current_score, score), sources | {approach})

        # Apply multi-source bonus
        final_scores = []
        for doc_id, (score, sources) in doc_scores.items():
            doc = next((d for d in self.documents if d.id == doc_id), None)
            if doc:
                bonus = 0.2 * (len(sources) - 1)
                final_scores.append((doc, score + bonus, sources))

        final_scores.sort(key=lambda x: x[1], reverse=True)
        top_docs = [doc for doc, _, _ in final_scores[:7]]

        # Step 7: Generate answer
        context = "\n\n".join([
            f"=== [{doc.id}] ===\n{doc.content[:2500]}"
            for doc in top_docs
        ])

        prompt = f"""Answer this question using the documents below.

Documents:
{context}

Question: {query}

Answer concisely, citing document IDs."""

        response = await self.llm.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=500,
        )
        answer = response.content if hasattr(response, "content") else str(response)

        # Calculate recall
        found = sum(1 for e in tc.expected_content if e.lower() in answer.lower())
        recall = found / len(tc.expected_content) if tc.expected_content else 1.0

        latency = (time.perf_counter() - start) * 1000

        return top_docs, recall, answer, latency


# =============================================================================
# Document Loader
# =============================================================================

async def load_documents(base_path: Path, extractor: SemanticExtractor | None) -> list[Document]:
    """Load and optionally extract from documents."""
    documents = []

    # Party Lore main project
    party_lore = base_path / "party-lore"
    if party_lore.exists():
        # CLAUDE.md
        claude_md = party_lore / "CLAUDE.md"
        if claude_md.exists():
            content = claude_md.read_text()[:50000]
            doc = Document(
                id="party-lore-claude-md",
                content=content,
                source="party-lore",
                doc_type="context",
                entities=["Party Lore", "Spring Boot", "Twilio", "Claude", "SMS"],
            )
            if extractor:
                extracted = await extractor.extract(content[:3000])
                doc.extracted_entities = extracted["entities"]
                doc.extracted_concepts = extracted["concepts"]
            documents.append(doc)

        # Key requirements
        req_path = party_lore / ".specify" / "requirements"
        if req_path.exists():
            for req_file in list(req_path.glob("fr-*.md"))[:10]:
                content = req_file.read_text()[:20000]
                doc = Document(
                    id=f"req-{req_file.stem}",
                    content=content,
                    source="party-lore",
                    doc_type="requirement",
                    entities=[],
                )
                if extractor:
                    extracted = await extractor.extract(content[:2000])
                    doc.extracted_entities = extracted["entities"]
                    doc.extracted_concepts = extracted["concepts"]
                documents.append(doc)

    # Party Lore Content
    content_hub = base_path / "party-lore-content"
    if content_hub.exists():
        claude_md = content_hub / "CLAUDE.md"
        if claude_md.exists():
            content = claude_md.read_text()[:30000]
            doc = Document(
                id="content-claude-md",
                content=content,
                source="party-lore-content",
                doc_type="context",
                entities=["Content Hub", "realm", "deploy"],
            )
            if extractor:
                extracted = await extractor.extract(content[:2000])
                doc.extracted_entities = extracted["entities"]
                doc.extracted_concepts = extracted["concepts"]
            documents.append(doc)

        # Realm files
        realms_path = content_hub / "realms"
        if realms_path.exists():
            for realm_dir in list(realms_path.iterdir())[:5]:
                if realm_dir.is_dir():
                    realm_json = realm_dir / "realm.json"
                    if realm_json.exists():
                        content = realm_json.read_text()
                        doc = Document(
                            id=f"realm-{realm_dir.name}",
                            content=content,
                            source="party-lore-content",
                            doc_type="realm",
                            entities=[realm_dir.name],
                        )
                        if extractor:
                            extracted = await extractor.extract(content[:2000])
                            doc.extracted_entities = extracted["entities"]
                            doc.extracted_concepts = extracted["concepts"]
                        documents.append(doc)

    print(f"    Loaded {len(documents)} documents")
    return documents


# =============================================================================
# Test Cases
# =============================================================================

TEST_CASES = [
    # EASY - Direct lookups
    TestCase(
        query="What is the dual-channel SMS system in Party Lore?",
        expected_sources=["req-fr-001-dual-channel-sms", "party-lore-claude-md"],
        expected_content=["group", "private", "SMS"],
        difficulty=QueryDifficulty.EASY,
        description="Direct requirement lookup",
    ),
    TestCase(
        query="What classes are available in the fantasy realm?",
        expected_sources=["realm-fantasy"],
        expected_content=["Swordmaster", "Shadow", "Battlemage"],
        difficulty=QueryDifficulty.EASY,
        description="Direct realm content lookup",
    ),
    TestCase(
        query="How do I run the autonomous test suite?",
        expected_sources=["party-lore-claude-md"],
        expected_content=["GROQ", "test", "autonomous"],
        difficulty=QueryDifficulty.EASY,
        description="Direct CLAUDE.md lookup",
    ),

    # MEDIUM - Semantic understanding
    TestCase(
        query="How does Party Lore handle player timeouts and inactivity?",
        expected_sources=["req-fr-003-ai-autopilot-system"],
        expected_content=["autopilot", "timeout", "AI"],
        difficulty=QueryDifficulty.MEDIUM,
        description="Requires connecting autopilot to timeout",
    ),
    TestCase(
        query="What happens when the AI generates narrative without scene context?",
        expected_sources=["party-lore-claude-md"],
        expected_content=["narrative", "scene", "context"],
        difficulty=QueryDifficulty.MEDIUM,
        description="Bug analysis understanding",
    ),
    TestCase(
        query="How do players coordinate group decisions in combat?",
        expected_sources=["req-fr-002-intelligent-scene-resolution", "req-fr-044-muca-framework-group-coordination-system"],
        expected_content=["MUCA", "consensus", "group"],
        difficulty=QueryDifficulty.MEDIUM,
        description="Cross-requirement understanding",
    ),

    # HARD - Multi-hop reasoning
    TestCase(
        query="What's the relationship between the Content Hub and game realms?",
        expected_sources=["content-claude-md"],
        expected_content=["deploy", "realm", "validate"],
        difficulty=QueryDifficulty.HARD,
        description="Cross-project connection",
    ),
    TestCase(
        query="How does the queue-first architecture prevent race conditions?",
        expected_sources=["party-lore-claude-md"],
        expected_content=["queue", "FIFO", "sequential"],
        difficulty=QueryDifficulty.HARD,
        description="Architecture deep dive",
    ),
    TestCase(
        query="What caused test failures and how were they fixed?",
        expected_sources=["party-lore-claude-md"],
        expected_content=["test", "fix", "failure"],
        difficulty=QueryDifficulty.HARD,
        description="Root cause analysis",
    ),

    # EXPERT - Deep domain knowledge
    TestCase(
        query="If I modify a Flyway migration after deployment, what's the emergency fix?",
        expected_sources=["party-lore-claude-md"],
        expected_content=["checksum", "flyway", "migration"],
        difficulty=QueryDifficulty.EXPERT,
        description="Specific operational procedure",
    ),
    TestCase(
        query="How does the tiered LLM system handle combat mechanics vs narrative?",
        expected_sources=["req-fr-002-intelligent-scene-resolution"],
        expected_content=["Tier", "mechanics", "narrative"],
        difficulty=QueryDifficulty.EXPERT,
        description="Complex system understanding",
    ),
    TestCase(
        query="What's the complete flow from player SMS to narrative response?",
        expected_sources=["party-lore-claude-md"],
        expected_content=["queue", "SMS", "narrative"],
        difficulty=QueryDifficulty.EXPERT,
        description="End-to-end architecture",
    ),
]


# =============================================================================
# Main Benchmark Runner
# =============================================================================

async def run_experiment(
    config: ExperimentConfig,
    documents: list[Document],
    llm,
    embedder,
    extractor: SemanticExtractor,
) -> ExperimentResult:
    """Run a single experiment configuration."""
    print(f"\n  Running: {config.name}")
    print(f"    {config.description}")

    # Create pipeline with this config
    use_extraction = config.semantic_mode != SemanticExtractionMode.NONE
    pipeline = ConfigurableHybridPipeline(
        llm=llm,
        embedder=embedder,
        documents=documents,
        config=config,
        extractor=extractor if use_extraction else None,
    )
    await pipeline.setup()

    # Run all test cases
    per_query_results = []
    total_recall = 0.0
    total_source_acc = 0.0
    total_latency = 0.0

    for tc in TEST_CASES:
        docs, recall, answer, latency = await pipeline.retrieve(tc.query, tc)

        # Calculate source accuracy
        retrieved_ids = {d.id for d in docs}
        source_acc = sum(1 for s in tc.expected_sources if s in retrieved_ids) / len(tc.expected_sources)

        per_query_results.append({
            "query": tc.query[:50],
            "difficulty": tc.difficulty.value,
            "recall": recall,
            "source_accuracy": source_acc,
            "latency_ms": latency,
        })

        total_recall += recall
        total_source_acc += source_acc
        total_latency += latency

        print(f"    [{tc.difficulty.value}] R={recall:.2f} Src={source_acc:.2f} {latency:.0f}ms")

    n = len(TEST_CASES)

    # Calculate by difficulty
    recall_by_diff = {}
    for diff in QueryDifficulty:
        diff_results = [r for r in per_query_results if r["difficulty"] == diff.value]
        if diff_results:
            recall_by_diff[diff.value] = sum(r["recall"] for r in diff_results) / len(diff_results)

    return ExperimentResult(
        config=config,
        total_recall=total_recall / n,
        total_source_accuracy=total_source_acc / n,
        avg_latency_ms=total_latency / n,
        wins_by_approach={},  # Not tracked in hybrid mode
        recall_by_difficulty=recall_by_diff,
        per_query_results=per_query_results,
    )


async def main():
    """Run all experiments and compare results."""
    print("\n" + "=" * 80)
    print("PIPELINE OPTIMIZATION EXPERIMENTS")
    print("=" * 80)

    # Check API key
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("ERROR: GROQ_API_KEY not set")
        return

    # Initialize providers
    from draagon_ai.llm.groq import GroqLLM
    from draagon_ai.memory.embedding import OllamaEmbeddingProvider

    print("\n[1] Initializing providers...")
    llm = GroqLLM(api_key=api_key)
    embedder = OllamaEmbeddingProvider(
        base_url="http://localhost:11434",
        model="mxbai-embed-large",
        dimension=1024,
    )

    # Create semantic extractor
    extractor = SemanticExtractor(llm)

    # Load documents with extraction
    print("\n[2] Loading and extracting documents...")
    dev_path = Path("/home/doug/Development")
    documents = await load_documents(dev_path, extractor)

    # Run experiments
    print("\n[3] Running Experiments...")
    print("-" * 80)

    results: list[ExperimentResult] = []
    for config in EXPERIMENTS:
        result = await run_experiment(config, documents, llm, embedder, extractor)
        results.append(result)

    # Summary
    print("\n" + "=" * 80)
    print("EXPERIMENT RESULTS SUMMARY")
    print("=" * 80)

    print("\n  Overall Performance:")
    print(f"  {'Experiment':<25} {'Recall':>10} {'Source Acc':>12} {'Latency':>10}")
    print("  " + "-" * 60)

    for r in sorted(results, key=lambda x: x.total_recall, reverse=True):
        print(f"  {r.config.name:<25} {r.total_recall:>10.1%} {r.total_source_accuracy:>12.1%} {r.avg_latency_ms:>8.0f}ms")

    print("\n  Performance by Difficulty:")
    for diff in QueryDifficulty:
        print(f"\n    {diff.value.upper()}:")
        for r in results:
            if diff.value in r.recall_by_difficulty:
                print(f"      {r.config.name:<25} {r.recall_by_difficulty[diff.value]:.1%}")

    # Find best config
    best = max(results, key=lambda x: x.total_recall)
    print(f"\n  BEST CONFIGURATION: {best.config.name}")
    print(f"    {best.config.description}")
    print(f"    Recall: {best.total_recall:.1%}")

    print("\n" + "=" * 80)
    print("EXPERIMENTS COMPLETE")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
