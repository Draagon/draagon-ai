#!/usr/bin/env python3
"""Real-World Retrieval Benchmark: Party Lore Projects.

This benchmark tests the 3 retrieval approaches (Raw Context, Vector/RAG, Semantic Graph)
on REAL project documentation from the Party Lore ecosystem:
- party-lore: Main game backend (Spring Boot, specifications, requirements)
- party-lore-content: Realm definitions, game content, marketing
- partylore.com: Marketing website

This provides realistic test cases for:
1. Cross-project information retrieval
2. Finding needles in large haystacks (specific details in verbose docs)
3. Connecting related concepts across different files
4. Disambiguating similar terms (e.g., "Atlas" in different contexts)

Run with:
    GROQ_API_KEY=your_key python3.11 tests/integration/agents/benchmark_real_world_retrieval.py
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


class RetrievalApproach(str, Enum):
    """The four retrieval approaches (3 base + 1 hybrid)."""
    RAW_CONTEXT = "raw_context"
    VECTOR_RAG = "vector_rag"
    SEMANTIC_GRAPH = "semantic_graph"
    HYBRID = "hybrid"  # Combined approach using shared working memory


class QueryDifficulty(str, Enum):
    """Query difficulty levels."""
    EASY = "easy"          # Direct keyword match
    MEDIUM = "medium"      # Requires semantic understanding
    HARD = "hard"          # Multi-hop or cross-document
    EXPERT = "expert"      # Requires deep domain knowledge


@dataclass
class RealWorldDocument:
    """A real document from the project."""
    id: str
    source_project: str  # party-lore, party-lore-content, partylore.com
    file_path: str
    content: str
    doc_type: str  # requirement, readme, analysis, realm, config
    entities: list[str] = field(default_factory=list)


@dataclass
class RealWorldTestCase:
    """Test case with expected outcomes from real documents."""
    query: str
    expected_sources: list[str]  # Document IDs that should be found
    expected_content: list[str]  # Keywords that should appear in answer
    difficulty: QueryDifficulty
    description: str
    category: str  # architecture, gameplay, requirements, cross-project


@dataclass
class ApproachResult:
    """Result from a single approach."""
    approach: RetrievalApproach
    recall: float
    precision: float
    source_accuracy: float  # Did we find the right source documents?
    latency_ms: float
    context_size: int
    answer: str = ""


@dataclass
class BenchmarkResult:
    """Full benchmark result for a test case."""
    test_case: RealWorldTestCase
    results: dict[RetrievalApproach, ApproachResult]
    winner: RetrievalApproach
    all_failed: bool


# =============================================================================
# Document Loader - Load Real Files from Party Lore Projects
# =============================================================================

class RealWorldDocumentLoader:
    """Loads real documents from Party Lore projects."""

    def __init__(self, base_path: Path):
        self.base_path = base_path
        self.documents: list[RealWorldDocument] = []

    async def load_all(self) -> list[RealWorldDocument]:
        """Load all relevant documents from the projects."""
        print("\n    Loading real-world documents...")

        # Party Lore main project
        party_lore = self.base_path / "party-lore"
        if party_lore.exists():
            await self._load_party_lore(party_lore)

        # Party Lore Content
        content = self.base_path / "party-lore-content"
        if content.exists():
            await self._load_content_hub(content)

        # Partylore.com
        website = self.base_path / "partylore.com"
        if website.exists():
            await self._load_website(website)

        print(f"    Loaded {len(self.documents)} documents")
        return self.documents

    async def _load_party_lore(self, path: Path):
        """Load documents from main party-lore project."""
        # CLAUDE.md - Main project context
        claude_md = path / "CLAUDE.md"
        if claude_md.exists():
            self.documents.append(RealWorldDocument(
                id="party-lore-claude-md",
                source_project="party-lore",
                file_path=str(claude_md),
                content=self._read_file(claude_md),
                doc_type="context",
                entities=["Party Lore", "Spring Boot", "Twilio", "Claude", "SMS", "TCPA",
                          "Flyway", "PostgreSQL", "Redis", "autonomous test", "queue-first"],
            ))

        # README.md
        readme = path / "README.md"
        if readme.exists():
            self.documents.append(RealWorldDocument(
                id="party-lore-readme",
                source_project="party-lore",
                file_path=str(readme),
                content=self._read_file(readme),
                doc_type="readme",
                entities=["Party Lore", "SMS RPG", "async gameplay", "AI Dungeon Master",
                          "console mode", "dual-channel", "Twilio"],
            ))

        # Key requirements files
        req_path = path / ".specify" / "requirements"
        if req_path.exists():
            key_reqs = [
                "fr-001-dual-channel-sms.md",
                "fr-002-intelligent-scene-resolution.md",
                "fr-003-ai-autopilot-system.md",
                "fr-004-deep-narrative-memory-system.md",
                "fr-005-multi-realm-system.md",
                "fr-044-muca-framework-group-coordination-system.md",
                "fr-045-core-game-loop-orchestration.md",
                "fr-156-message-queue-architecture.md",
                "fr-198-game-mode-transition-system.md",
            ]
            for req_file in key_reqs:
                fp = req_path / req_file
                if fp.exists():
                    content = self._read_file(fp)
                    # Extract FR number for entities
                    fr_num = req_file.split("-")[0] + "-" + req_file.split("-")[1]
                    self.documents.append(RealWorldDocument(
                        id=f"req-{req_file.replace('.md', '')}",
                        source_project="party-lore",
                        file_path=str(fp),
                        content=content,
                        doc_type="requirement",
                        entities=self._extract_entities_from_content(content),
                    ))

        # Analysis documents
        analysis_path = path / ".specify" / "analysis"
        if analysis_path.exists():
            for analysis_file in analysis_path.glob("*.md"):
                content = self._read_file(analysis_file)
                self.documents.append(RealWorldDocument(
                    id=f"analysis-{analysis_file.stem}",
                    source_project="party-lore",
                    file_path=str(analysis_file),
                    content=content,
                    doc_type="analysis",
                    entities=self._extract_entities_from_content(content),
                ))

    async def _load_content_hub(self, path: Path):
        """Load documents from party-lore-content."""
        # CLAUDE.md
        claude_md = path / "CLAUDE.md"
        if claude_md.exists():
            self.documents.append(RealWorldDocument(
                id="content-claude-md",
                source_project="party-lore-content",
                file_path=str(claude_md),
                content=self._read_file(claude_md),
                doc_type="context",
                entities=["Content Hub", "realm", "marketing", "deploy", "validate"],
            ))

        # Realm files
        realms_path = path / "realms"
        if realms_path.exists():
            for realm_dir in realms_path.iterdir():
                if realm_dir.is_dir() and not realm_dir.name.startswith("_"):
                    realm_json = realm_dir / "realm.json"
                    if realm_json.exists():
                        content = self._read_file(realm_json)
                        try:
                            realm_data = json.loads(content)
                            realm_name = realm_data.get("name", realm_dir.name)
                            entities = [realm_name, realm_data.get("realmType", "")]
                            entities.extend(realm_data.get("availableClasses", []))
                        except json.JSONDecodeError:
                            entities = [realm_dir.name]

                        self.documents.append(RealWorldDocument(
                            id=f"realm-{realm_dir.name}",
                            source_project="party-lore-content",
                            file_path=str(realm_json),
                            content=content,
                            doc_type="realm",
                            entities=entities,
                        ))

    async def _load_website(self, path: Path):
        """Load documents from partylore.com."""
        readme = path / "README.md"
        if readme.exists():
            self.documents.append(RealWorldDocument(
                id="website-readme",
                source_project="partylore.com",
                file_path=str(readme),
                content=self._read_file(readme),
                doc_type="readme",
                entities=["landing page", "Twilio", "privacy policy", "SMS consent"],
            ))

        # Docs folder
        docs_path = path / "docs"
        if docs_path.exists():
            for doc_file in docs_path.glob("*.md"):
                content = self._read_file(doc_file)
                self.documents.append(RealWorldDocument(
                    id=f"website-{doc_file.stem}",
                    source_project="partylore.com",
                    file_path=str(doc_file),
                    content=content,
                    doc_type="documentation",
                    entities=self._extract_entities_from_content(content),
                ))

    def _read_file(self, path: Path, max_chars: int = 50000) -> str:
        """Read file content, truncating if too large."""
        try:
            content = path.read_text(encoding="utf-8")
            if len(content) > max_chars:
                content = content[:max_chars] + "\n\n[... truncated ...]"
            return content
        except Exception as e:
            return f"[Error reading file: {e}]"

    def _extract_entities_from_content(self, content: str) -> list[str]:
        """Extract meaningful entity names from content."""
        import re
        entities = []

        # FR numbers (e.g., FR-001, FR-156)
        fr_matches = re.findall(r"FR-\d+[A-Z]?", content)
        entities.extend(fr_matches[:10])

        # TASK numbers (e.g., TASK-206)
        task_matches = re.findall(r"TASK-\d+", content)
        entities.extend(task_matches[:10])

        # Domain-specific terms we know are important for Party Lore
        domain_terms = [
            # Core systems
            "dual-channel", "dual channel", "SMS", "autopilot", "AI autopilot",
            "scene resolution", "narrative", "MUCA", "consensus",
            "queue-first", "message queue", "QueueOrchestrator",
            # Technical
            "Twilio", "Claude", "Spring Boot", "PostgreSQL", "Redis", "Flyway",
            "GameSession", "GameContext", "PromptEngineering",
            # Gameplay
            "realm", "fantasy", "cyberpunk", "cosmic horror",
            "Swordmaster", "Shadow", "Battlemage", "Healer", "Archer",
            "Iron Company", "Neon Underground",
            # Analysis terms
            "narrative vacuum", "scene context", "test failure",
            "root cause", "autonomous test",
            # Content hub
            "Content Hub", "deploy", "validate", "realm.json",
        ]

        content_lower = content.lower()
        for term in domain_terms:
            if term.lower() in content_lower:
                entities.append(term)

        # Extract header terms (## Something Important)
        header_matches = re.findall(r"^#+\s+(.+)$", content, re.MULTILINE)
        for header in header_matches[:10]:
            # Clean and add if reasonable length
            header = header.strip()
            if 3 < len(header) < 50 and not header.lower().startswith(("the ", "a ", "an ")):
                entities.append(header)

        # Technical compound terms (kebab-case or CamelCase)
        kebab_matches = re.findall(r"\b([a-z]+-[a-z]+(?:-[a-z]+)*)\b", content_lower)
        for match in kebab_matches[:10]:
            if len(match) > 5:  # Skip short ones like "to-do"
                entities.append(match)

        camel_matches = re.findall(r"\b([A-Z][a-z]+(?:[A-Z][a-z]+)+)\b", content)
        entities.extend(camel_matches[:10])

        # Dedupe while preserving some variation for matching
        seen = set()
        unique = []
        for e in entities:
            key = e.lower().replace("-", " ").replace("_", " ")
            if key not in seen:
                seen.add(key)
                unique.append(e)

        return unique


# =============================================================================
# Test Cases - Realistic Queries Against Real Documents
# =============================================================================

REAL_WORLD_TEST_CASES = [
    # =========================================================================
    # EASY: Direct lookups
    # =========================================================================
    RealWorldTestCase(
        query="What is the dual-channel SMS system in Party Lore?",
        expected_sources=["req-fr-001-dual-channel-sms", "party-lore-readme"],
        expected_content=["group channel", "private channel", "vCard", "3-6 players"],
        difficulty=QueryDifficulty.EASY,
        description="Direct requirement lookup",
        category="requirements",
    ),
    RealWorldTestCase(
        query="What classes are available in the Iron Company fantasy realm?",
        expected_sources=["realm-fantasy"],
        expected_content=["Swordmaster", "Shadow", "Battlemage", "Healer", "Archer"],
        difficulty=QueryDifficulty.EASY,
        description="Direct realm content lookup",
        category="gameplay",
    ),
    RealWorldTestCase(
        query="How do I run the autonomous test suite?",
        expected_sources=["party-lore-claude-md"],
        expected_content=["GROQ_API_KEY", ".env.local", "autonomous-test", "suite=core"],
        difficulty=QueryDifficulty.EASY,
        description="Direct CLAUDE.md lookup",
        category="development",
    ),

    # =========================================================================
    # MEDIUM: Semantic understanding required
    # =========================================================================
    RealWorldTestCase(
        query="How does Party Lore handle player timeouts and inactivity?",
        expected_sources=["req-fr-003-ai-autopilot-system", "party-lore-readme"],
        expected_content=["autopilot", "8-hour", "timeout", "AI", "character"],
        difficulty=QueryDifficulty.MEDIUM,
        description="Requires connecting autopilot to timeout concept",
        category="gameplay",
    ),
    RealWorldTestCase(
        query="What happens when the AI generates narrative without scene context?",
        expected_sources=["analysis-narrative-vacuum-root-cause-analysis"],
        expected_content=["narrative vacuum", "scene context", "setupPrompt", "GameContext"],
        difficulty=QueryDifficulty.MEDIUM,
        description="Bug analysis understanding",
        category="architecture",
    ),
    RealWorldTestCase(
        query="How do players coordinate group decisions in combat?",
        expected_sources=["req-fr-002-intelligent-scene-resolution", "req-fr-044-muca-framework-group-coordination-system"],
        expected_content=["MUCA", "consensus", "tactical", "intent"],
        difficulty=QueryDifficulty.MEDIUM,
        description="Cross-requirement understanding",
        category="gameplay",
    ),

    # =========================================================================
    # HARD: Multi-hop or cross-document reasoning
    # =========================================================================
    RealWorldTestCase(
        query="What's the relationship between the Content Hub and game realms deployment?",
        expected_sources=["content-claude-md", "realm-fantasy"],
        expected_content=["deploy-game.sh", "validate", "qat", "prod", "realm.json"],
        difficulty=QueryDifficulty.HARD,
        description="Cross-project connection",
        category="cross-project",
    ),
    RealWorldTestCase(
        query="How does the queue-first architecture prevent race conditions in game sessions?",
        expected_sources=["party-lore-claude-md", "req-fr-156-message-queue-architecture"],
        expected_content=["QueueOrchestrator", "FIFO", "sequential", "GameSessionWorker"],
        difficulty=QueryDifficulty.HARD,
        description="Architecture deep dive",
        category="architecture",
    ),
    RealWorldTestCase(
        query="What caused the 100% test failure rate and how was it fixed?",
        expected_sources=["analysis-narrative-vacuum-root-cause-analysis"],
        expected_content=["TASK-206", "scene context", "GameContextAssembler", "75%"],
        difficulty=QueryDifficulty.HARD,
        description="Root cause analysis comprehension",
        category="architecture",
    ),

    # =========================================================================
    # EXPERT: Deep domain knowledge + multi-hop
    # =========================================================================
    RealWorldTestCase(
        query="If I modify a Flyway migration after deployment, what's the emergency fix process?",
        expected_sources=["party-lore-claude-md"],
        expected_content=["cksum", "flyway_schema_history", "railway connect", "checksum"],
        difficulty=QueryDifficulty.EXPERT,
        description="Specific operational procedure",
        category="development",
    ),
    RealWorldTestCase(
        query="How does the tiered LLM system handle combat mechanics vs narrative generation?",
        expected_sources=["req-fr-002-intelligent-scene-resolution"],
        expected_content=["Tier 1", "Tier 2", "mechanics", "narrative", "intent"],
        difficulty=QueryDifficulty.EXPERT,
        description="Complex system understanding",
        category="architecture",
    ),
    RealWorldTestCase(
        query="What's the complete flow from player SMS to narrative response?",
        expected_sources=["party-lore-claude-md", "req-fr-045-core-game-loop-orchestration"],
        expected_content=["queue", "GameServiceLoopProcessor", "narrative", "SMS"],
        difficulty=QueryDifficulty.EXPERT,
        description="End-to-end architecture",
        category="architecture",
    ),
]


# =============================================================================
# Approach Implementations
# =============================================================================

class RawContextApproach:
    """Load documents into LLM context, ask directly."""

    def __init__(self, llm, documents: list[RealWorldDocument]):
        self.llm = llm
        self.documents = documents

    async def retrieve(self, query: str, tc: RealWorldTestCase) -> ApproachResult:
        """Load all docs as context, ask LLM to find answer."""
        start = time.perf_counter()

        # Build context from all documents (truncated to fit context window)
        context_parts = []
        total_chars = 0
        max_chars = 100000  # ~25k tokens

        for doc in self.documents:
            doc_content = f"\n\n=== [{doc.id}] ({doc.source_project}) ===\n{doc.content[:5000]}"
            if total_chars + len(doc_content) > max_chars:
                break
            context_parts.append(doc_content)
            total_chars += len(doc_content)

        context = "".join(context_parts)

        prompt = f"""Given this documentation from the Party Lore project ecosystem, answer the question.

Documentation:
{context}

Question: {query}

Instructions:
1. Find the relevant information in the documentation
2. Cite which documents you found the information in
3. Provide a comprehensive answer
4. If the answer isn't in the documentation, say so

Answer:"""

        response = await self.llm.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=500,
        )
        content = response.content if hasattr(response, "content") else str(response)

        latency = (time.perf_counter() - start) * 1000

        # Score based on expected content
        recall = self._compute_recall(content, tc.expected_content)
        source_acc = self._compute_source_accuracy(content, tc.expected_sources)

        return ApproachResult(
            approach=RetrievalApproach.RAW_CONTEXT,
            recall=recall,
            precision=recall,
            source_accuracy=source_acc,
            latency_ms=latency,
            context_size=len(context),
            answer=content[:500],
        )

    def _compute_recall(self, answer: str, expected: list[str]) -> float:
        found = sum(1 for e in expected if e.lower() in answer.lower())
        return found / len(expected) if expected else 1.0

    def _compute_source_accuracy(self, answer: str, expected_sources: list[str]) -> float:
        found = sum(1 for s in expected_sources if s.lower() in answer.lower())
        return found / len(expected_sources) if expected_sources else 0.0


class VectorRAGApproach:
    """Embed query, search, return top-k."""

    def __init__(self, llm, embedder, documents: list[RealWorldDocument]):
        self.llm = llm
        self.embedder = embedder
        self.documents = documents
        self.embedded_docs: list[tuple[RealWorldDocument, list[float]]] = []

    async def setup(self):
        """Embed all documents."""
        print("    Embedding documents for Vector/RAG...")
        for i, doc in enumerate(self.documents):
            # Embed a summary of the document
            # mxbai-embed-large has 512 token context, so ~1500 chars is safe
            text_to_embed = doc.content[:1500]
            try:
                embedding = await self.embedder.embed(text_to_embed)
                self.embedded_docs.append((doc, embedding))
            except Exception as e:
                print(f"      Warning: Failed to embed {doc.id}: {e}")
                continue
            if (i + 1) % 10 == 0:
                print(f"      Embedded {i + 1}/{len(self.documents)}")
        print(f"    Embedded {len(self.embedded_docs)} documents")

    async def retrieve(self, query: str, tc: RealWorldTestCase, use_hyde: bool = True) -> ApproachResult:
        """Search with optional HyDE expansion."""
        import math
        start = time.perf_counter()

        # Optionally expand with HyDE
        if use_hyde and tc.difficulty in [QueryDifficulty.HARD, QueryDifficulty.EXPERT]:
            query_text = await self._hyde_expand(query)
        else:
            query_text = query

        # Embed query
        query_embedding = await self.embedder.embed(query_text)

        # Search
        def cosine_sim(a, b):
            dot = sum(x * y for x, y in zip(a, b))
            norm_a = math.sqrt(sum(x * x for x in a))
            norm_b = math.sqrt(sum(x * x for x in b))
            return dot / (norm_a * norm_b) if norm_a > 0 and norm_b > 0 else 0

        scores = [(doc, cosine_sim(query_embedding, emb)) for doc, emb in self.embedded_docs]
        scores.sort(key=lambda x: x[1], reverse=True)
        top_docs = [doc for doc, _ in scores[:5]]

        # Build context from top docs
        context = "\n\n".join([
            f"=== [{doc.id}] ({doc.source_project}) ===\n{doc.content[:3000]}"
            for doc in top_docs
        ])

        # Ask LLM
        prompt = f"""Given these retrieved documents from the Party Lore project, answer the question.

Retrieved Documents:
{context}

Question: {query}

Answer with specific facts from the documents. Cite which documents you used."""

        response = await self.llm.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=500,
        )
        content = response.content if hasattr(response, "content") else str(response)

        latency = (time.perf_counter() - start) * 1000

        recall = self._compute_recall(content, tc.expected_content)
        source_acc = self._compute_source_accuracy(content, tc.expected_sources, top_docs)

        return ApproachResult(
            approach=RetrievalApproach.VECTOR_RAG,
            recall=recall,
            precision=recall,
            source_accuracy=source_acc,
            latency_ms=latency,
            context_size=len(top_docs),
            answer=content[:500],
        )

    async def _hyde_expand(self, query: str) -> str:
        """Generate hypothetical document for better embedding."""
        prompt = f"""Write a detailed technical documentation paragraph that would answer this question about the Party Lore SMS RPG platform:

Question: {query}

Write as if you are documenting the system architecture or gameplay mechanics."""

        response = await self.llm.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=200,
        )
        return response.content if hasattr(response, "content") else str(response)

    def _compute_recall(self, answer: str, expected: list[str]) -> float:
        found = sum(1 for e in expected if e.lower() in answer.lower())
        return found / len(expected) if expected else 1.0

    def _compute_source_accuracy(self, answer: str, expected_sources: list[str],
                                  retrieved_docs: list[RealWorldDocument]) -> float:
        retrieved_ids = {doc.id for doc in retrieved_docs}
        found = sum(1 for s in expected_sources if s in retrieved_ids)
        return found / len(expected_sources) if expected_sources else 0.0


class HybridPipelineApproach:
    """Combined approach using shared working memory.

    Runs all approaches in parallel, shares observations via working memory,
    and synthesizes a combined answer from the best evidence.
    """

    def __init__(
        self,
        llm,
        embedder,
        documents: list[RealWorldDocument],
        raw_approach: "RawContextApproach",
        vector_approach: "VectorRAGApproach",
        graph_approach: "SemanticGraphApproach",
    ):
        self.llm = llm
        self.embedder = embedder
        self.documents = documents
        self.raw_approach = raw_approach
        self.vector_approach = vector_approach
        self.graph_approach = graph_approach

    async def retrieve(self, query: str, tc: RealWorldTestCase) -> ApproachResult:
        """Run all approaches, share evidence, synthesize answer."""
        from draagon_ai.orchestration.shared_memory import (
            SharedWorkingMemory,
            SharedWorkingMemoryConfig,
        )
        from draagon_ai.orchestration.multi_agent_orchestrator import AgentRole
        import math
        import re

        start = time.perf_counter()

        # Create shared working memory for this query
        config = SharedWorkingMemoryConfig(
            max_items_per_agent=10,
            max_total_items=30,
            attention_decay_factor=0.95,
        )
        shared_memory = SharedWorkingMemory(
            task_id=f"query_{hash(query)}",
            config=config,
        )

        # Run all approaches in parallel to gather evidence
        raw_task = asyncio.create_task(self._gather_raw_evidence(query, tc, shared_memory))
        vector_task = asyncio.create_task(self._gather_vector_evidence(query, tc, shared_memory))
        graph_task = asyncio.create_task(self._gather_graph_evidence(query, tc, shared_memory))

        await asyncio.gather(raw_task, vector_task, graph_task)

        # Get all observations sorted by attention
        all_obs = await shared_memory.get_context_for_agent(
            agent_id="synthesizer",
            role=AgentRole.RESEARCHER,
            max_items=20,
        )

        # Track doc_id -> (count, max_attention, sources)
        # Documents found by multiple approaches get boosted
        doc_scores: dict[str, tuple[int, float, list[str]]] = {}

        for obs in all_obs:
            if "[doc:" in obs.content:
                match = re.search(r"\[doc:([^\]]+)\]", obs.content)
                if match:
                    doc_id = match.group(1)
                    source = obs.source_agent_id
                    if doc_id not in doc_scores:
                        doc_scores[doc_id] = (0, 0.0, [])
                    count, max_att, sources = doc_scores[doc_id]
                    doc_scores[doc_id] = (
                        count + 1,
                        max(max_att, obs.attention_weight),
                        sources + [source] if source not in sources else sources,
                    )

        # Score docs: multi-source bonus + attention
        scored_docs = []
        for doc_id, (count, max_att, sources) in doc_scores.items():
            doc = next((d for d in self.documents if d.id == doc_id), None)
            if doc:
                # Bonus for being found by multiple approaches
                multi_source_bonus = 0.2 * (len(set(sources)) - 1)
                final_score = max_att + multi_source_bonus
                scored_docs.append((doc, final_score, sources))

        # Sort by combined score
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        relevant_docs = [doc for doc, _, _ in scored_docs[:7]]

        # Collect findings, prioritizing multi-source docs
        findings = []
        for doc, score, sources in scored_docs[:5]:
            sources_str = ", ".join(set(sources))
            findings.append(f"- [{doc.id}] found by {sources_str} (score: {score:.2f})")

        findings_text = "\n".join(findings) if findings else "(No specific findings)"

        # Build combined context
        context = "\n\n".join([
            f"=== [{doc.id}] ({doc.source_project}) ===\n{doc.content[:2500]}"
            for doc in relevant_docs
        ])

        # Synthesize answer with cross-approach evidence
        prompt = f"""You have evidence from multiple retrieval approaches analyzing Party Lore documentation.

Multi-Source Document Ranking:
{findings_text}

Relevant Documents:
{context}

Question: {query}

Synthesize a comprehensive answer using the evidence above. Focus on documents found by multiple approaches as they are most likely relevant."""

        response = await self.llm.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=600,
        )
        content = response.content if hasattr(response, "content") else str(response)

        latency = (time.perf_counter() - start) * 1000

        recall = self._compute_recall(content, tc.expected_content)
        source_acc = self._compute_source_accuracy(content, tc.expected_sources, relevant_docs)

        return ApproachResult(
            approach=RetrievalApproach.HYBRID,
            recall=recall,
            precision=recall,
            source_accuracy=source_acc,
            latency_ms=latency,
            context_size=len(relevant_docs),
            answer=content[:500],
        )

    async def _gather_raw_evidence(self, query: str, tc: RealWorldTestCase, memory) -> None:
        """Gather evidence from raw context approach."""
        # Get documents from raw context (top matches by keyword overlap)
        query_words = set(query.lower().split())
        scored_docs = []
        for doc in self.documents:
            doc_words = set(doc.content[:2000].lower().split())
            overlap = len(query_words & doc_words)
            scored_docs.append((doc, overlap))

        scored_docs.sort(key=lambda x: x[1], reverse=True)
        top_docs = [doc for doc, _ in scored_docs[:5]]

        for doc in top_docs:
            # Add as observation with moderate attention
            await memory.add_observation(
                content=f"[doc:{doc.id}] Raw context found relevant document from {doc.source_project}",
                source_agent_id="raw_context",
                attention_weight=0.6,
                is_belief_candidate=True,
                belief_type="EVIDENCE",
            )

    async def _gather_vector_evidence(self, query: str, tc: RealWorldTestCase, memory) -> None:
        """Gather evidence from vector similarity."""
        import math

        if not self.vector_approach or not self.vector_approach.embedded_docs:
            return

        # Embed query
        query_embedding = await self.embedder.embed(query)

        def cosine_sim(a, b):
            dot = sum(x * y for x, y in zip(a, b))
            norm_a = math.sqrt(sum(x * x for x in a))
            norm_b = math.sqrt(sum(x * x for x in b))
            return dot / (norm_a * norm_b) if norm_a > 0 and norm_b > 0 else 0

        scores = [(doc, cosine_sim(query_embedding, emb))
                  for doc, emb in self.vector_approach.embedded_docs]
        scores.sort(key=lambda x: x[1], reverse=True)

        for doc, sim in scores[:5]:
            # Higher similarity = higher attention weight
            await memory.add_observation(
                content=f"[doc:{doc.id}] Vector similarity {sim:.2f} - {doc.doc_type} from {doc.source_project}",
                source_agent_id="vector_rag",
                attention_weight=min(0.9, 0.5 + sim * 0.4),  # Scale similarity to attention
                is_belief_candidate=True,
                belief_type="EVIDENCE",
            )

    async def _gather_graph_evidence(self, query: str, tc: RealWorldTestCase, memory) -> None:
        """Gather evidence from semantic graph (entity matching)."""
        # Extract entities from query
        entities = await self.graph_approach._extract_entities(query)

        # Find docs via entity matching
        found_docs = set()
        entity_matches = {}

        for entity in entities:
            normalized = self.graph_approach._normalize_entity(entity)

            # Check all normalized entity keys
            for norm_key, orig_keys in self.graph_approach.normalized_to_original.items():
                if normalized in norm_key or norm_key in normalized:
                    for orig_key in orig_keys:
                        for doc in self.graph_approach.entity_to_docs.get(orig_key, []):
                            found_docs.add(doc.id)
                            if doc.id not in entity_matches:
                                entity_matches[doc.id] = []
                            entity_matches[doc.id].append(entity)

        for doc_id in list(found_docs)[:5]:
            doc = next((d for d in self.documents if d.id == doc_id), None)
            if doc:
                matched = entity_matches.get(doc_id, [])
                # More entity matches = higher attention
                attention = min(0.9, 0.5 + len(matched) * 0.1)
                await memory.add_observation(
                    content=f"[doc:{doc.id}] Entity match: {', '.join(matched[:3])} in {doc.doc_type}",
                    source_agent_id="semantic_graph",
                    attention_weight=attention,
                    is_belief_candidate=True,
                    belief_type="EVIDENCE",
                )

    def _compute_recall(self, answer: str, expected: list[str]) -> float:
        found = sum(1 for e in expected if e.lower() in answer.lower())
        return found / len(expected) if expected else 1.0

    def _compute_source_accuracy(self, answer: str, expected_sources: list[str],
                                  retrieved_docs: list[RealWorldDocument]) -> float:
        retrieved_ids = {doc.id for doc in retrieved_docs}
        found = sum(1 for s in expected_sources if s in retrieved_ids)
        return found / len(expected_sources) if expected_sources else 0.0


class SemanticGraphApproach:
    """Query knowledge graph for entities and relationships."""

    def __init__(self, llm, documents: list[RealWorldDocument]):
        self.llm = llm
        self.documents = documents

        # Build entity index with normalized keys for better matching
        self.entity_to_docs: dict[str, list[RealWorldDocument]] = {}
        self.normalized_to_original: dict[str, list[str]] = {}  # normalized -> original keys

        for doc in documents:
            for entity in doc.entities:
                # Add original entity
                if entity not in self.entity_to_docs:
                    self.entity_to_docs[entity] = []
                self.entity_to_docs[entity].append(doc)

                # Track normalized form for fuzzy matching
                normalized = self._normalize_entity(entity)
                if normalized not in self.normalized_to_original:
                    self.normalized_to_original[normalized] = []
                if entity not in self.normalized_to_original[normalized]:
                    self.normalized_to_original[normalized].append(entity)

    def _normalize_entity(self, entity: str) -> str:
        """Normalize entity for fuzzy matching."""
        return entity.lower().replace("-", " ").replace("_", " ").strip()

    async def retrieve(self, query: str, tc: RealWorldTestCase) -> ApproachResult:
        """Extract entities, traverse graph, find relevant facts."""
        start = time.perf_counter()

        # Extract entities from query using LLM
        entities = await self._extract_entities(query)

        # Find relevant docs via entity traversal
        relevant_docs: set[str] = set()
        for entity in entities:
            normalized_query = self._normalize_entity(entity)

            # Direct match on original keys
            if entity in self.entity_to_docs:
                for doc in self.entity_to_docs[entity]:
                    relevant_docs.add(doc.id)

            # Normalized exact match
            if normalized_query in self.normalized_to_original:
                for original_key in self.normalized_to_original[normalized_query]:
                    for doc in self.entity_to_docs.get(original_key, []):
                        relevant_docs.add(doc.id)

            # Fuzzy substring match on normalized forms
            for normalized_key, original_keys in self.normalized_to_original.items():
                # Check for substring match in either direction
                if normalized_query in normalized_key or normalized_key in normalized_query:
                    for original_key in original_keys:
                        for doc in self.entity_to_docs.get(original_key, []):
                            relevant_docs.add(doc.id)
                # Also check word overlap (for multi-word entities)
                query_words = set(normalized_query.split())
                key_words = set(normalized_key.split())
                if query_words and key_words and len(query_words & key_words) >= 1:
                    # At least one meaningful word overlap
                    overlap_word = list(query_words & key_words)[0]
                    if len(overlap_word) >= 4:  # Skip short common words
                        for original_key in original_keys:
                            for doc in self.entity_to_docs.get(original_key, []):
                                relevant_docs.add(doc.id)

        # For hard queries, expand to related entities (multi-hop)
        if tc.difficulty in [QueryDifficulty.HARD, QueryDifficulty.EXPERT]:
            expanded_docs = set()
            for doc_id in relevant_docs:
                doc = next((d for d in self.documents if d.id == doc_id), None)
                if doc:
                    for entity in doc.entities:
                        if entity in self.entity_to_docs:
                            for related_doc in self.entity_to_docs[entity]:
                                expanded_docs.add(related_doc.id)
            relevant_docs.update(expanded_docs)

        # Get full docs
        docs = [d for d in self.documents if d.id in relevant_docs][:10]

        if not docs:
            # Fallback: return docs with most entities
            docs = sorted(self.documents, key=lambda d: len(d.entities), reverse=True)[:5]

        # Build context
        context = "\n\n".join([
            f"=== [{doc.id}] ({doc.source_project}) ===\n{doc.content[:3000]}"
            for doc in docs
        ])

        # Ask LLM
        prompt = f"""Given this knowledge graph context from the Party Lore project, answer the question.

Entities found: {', '.join(entities)}

Knowledge Graph Facts:
{context}

Question: {query}

Answer with specific facts. Trace multi-hop relationships if needed."""

        response = await self.llm.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=500,
        )
        content = response.content if hasattr(response, "content") else str(response)

        latency = (time.perf_counter() - start) * 1000

        recall = self._compute_recall(content, tc.expected_content)
        source_acc = self._compute_source_accuracy(content, tc.expected_sources, docs)

        return ApproachResult(
            approach=RetrievalApproach.SEMANTIC_GRAPH,
            recall=recall,
            precision=recall,
            source_accuracy=source_acc,
            latency_ms=latency,
            context_size=len(docs),
            answer=content[:500],
        )

    async def _extract_entities(self, query: str) -> list[str]:
        """Extract entity names from query using LLM."""
        prompt = f"""Extract ONLY the key entities from this query. Output each entity on its own line.
Do NOT include explanations, numbering, or commentary. Just the entities.

Query: {query}

Entities to extract:
- System/feature names (e.g., "dual-channel SMS", "autopilot", "MUCA")
- Technical terms (e.g., "queue-first", "scene resolution")
- Project names (e.g., "Party Lore", "Content Hub")
- FR/TASK identifiers (e.g., "FR-001", "TASK-206")

Output format - one entity per line, nothing else:"""

        response = await self.llm.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=150,
        )
        content = response.content if hasattr(response, "content") else str(response)

        # Parse entities - filter out lines that look like explanations
        entities = []
        for line in content.strip().split("\n"):
            line = line.strip()
            # Skip empty lines, numbered items, bullet points with colons, long sentences
            if not line:
                continue
            if line.startswith(("-", "*", "•")) and ":" in line:
                continue
            if len(line) > 50:  # Skip long explanatory text
                continue
            if line.lower().startswith(("unfortunately", "here", "the query", "please")):
                continue
            # Clean up bullets/numbers
            line = line.lstrip("-*•0123456789. ")
            if line:
                entities.append(line)

        return entities

    def _compute_recall(self, answer: str, expected: list[str]) -> float:
        found = sum(1 for e in expected if e.lower() in answer.lower())
        return found / len(expected) if expected else 1.0

    def _compute_source_accuracy(self, answer: str, expected_sources: list[str],
                                  retrieved_docs: list[RealWorldDocument]) -> float:
        retrieved_ids = {doc.id for doc in retrieved_docs}
        found = sum(1 for s in expected_sources if s in retrieved_ids)
        return found / len(expected_sources) if expected_sources else 0.0


# =============================================================================
# Main Benchmark
# =============================================================================

async def main():
    """Run the real-world benchmark."""
    print("\n" + "=" * 80)
    print("REAL-WORLD RETRIEVAL BENCHMARK")
    print("Testing on Party Lore Project Documentation")
    print("=" * 80)

    # Check API key
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("ERROR: GROQ_API_KEY not set")
        return

    # Import providers
    from draagon_ai.llm.groq import GroqLLM
    from draagon_ai.memory.embedding import OllamaEmbeddingProvider

    print("\n[1] Initializing providers...")
    llm = GroqLLM(api_key=api_key)

    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    # Using mxbai-embed-large: higher quality embeddings (MTEB 64.68 vs nomic's 53.01)
    # 1024 dimensions vs nomic's 768
    embedder = OllamaEmbeddingProvider(
        base_url=ollama_url,
        model="mxbai-embed-large",
        dimension=1024,
    )

    try:
        await embedder.embed("test")
        print("    Providers ready")
    except Exception as e:
        print(f"    Embedding error: {e}")
        print("    Continuing without Vector/RAG approach...")
        embedder = None

    # Load real documents
    print("\n[2] Loading real-world documents...")
    dev_path = Path("/home/doug/Development")
    loader = RealWorldDocumentLoader(dev_path)
    documents = await loader.load_all()

    if not documents:
        print("ERROR: No documents found. Check paths.")
        return

    # Show document summary
    by_project = {}
    for doc in documents:
        by_project.setdefault(doc.source_project, []).append(doc)

    print("\n    Document Summary:")
    for project, docs in by_project.items():
        print(f"      {project}: {len(docs)} documents")

    # Initialize approaches
    print("\n[3] Initializing approaches...")
    raw_approach = RawContextApproach(llm, documents)
    graph_approach = SemanticGraphApproach(llm, documents)

    rag_approach = None
    hybrid_approach = None
    if embedder:
        rag_approach = VectorRAGApproach(llm, embedder, documents)
        await rag_approach.setup()

        # Hybrid approach combines all three with shared working memory
        hybrid_approach = HybridPipelineApproach(
            llm=llm,
            embedder=embedder,
            documents=documents,
            raw_approach=raw_approach,
            vector_approach=rag_approach,
            graph_approach=graph_approach,
        )

    print("    All approaches ready")

    # Run benchmark
    print("\n" + "-" * 80)
    print("[4] Running Benchmark")
    print("-" * 80)

    results: list[BenchmarkResult] = []

    for tc in REAL_WORLD_TEST_CASES:
        print(f"\n  Query: \"{tc.query[:70]}...\"" if len(tc.query) > 70 else f"\n  Query: \"{tc.query}\"")
        print(f"  Difficulty: {tc.difficulty.value} | Category: {tc.category}")

        approach_results = {}

        # Raw Context
        raw_result = await raw_approach.retrieve(tc.query, tc)
        approach_results[RetrievalApproach.RAW_CONTEXT] = raw_result

        # Vector RAG (if available)
        if rag_approach:
            rag_result = await rag_approach.retrieve(tc.query, tc)
            approach_results[RetrievalApproach.VECTOR_RAG] = rag_result

        # Semantic Graph
        graph_result = await graph_approach.retrieve(tc.query, tc)
        approach_results[RetrievalApproach.SEMANTIC_GRAPH] = graph_result

        # Hybrid (if available)
        if hybrid_approach:
            hybrid_result = await hybrid_approach.retrieve(tc.query, tc)
            approach_results[RetrievalApproach.HYBRID] = hybrid_result

        # Determine winner (by recall, then source accuracy)
        winner = max(
            approach_results.keys(),
            key=lambda a: (approach_results[a].recall, approach_results[a].source_accuracy)
        )

        all_failed = all(r.recall < 0.5 for r in approach_results.values())

        results.append(BenchmarkResult(
            test_case=tc,
            results=approach_results,
            winner=winner,
            all_failed=all_failed,
        ))

        # Print results
        status = "" if not all_failed else ""
        print(f"  {status} Winner: {winner.value}")

        for approach, result in approach_results.items():
            marker = "" if approach == winner else " "
            print(f"    {marker} {approach.value:15} R={result.recall:.2f} "
                  f"Src={result.source_accuracy:.2f} {result.latency_ms:.0f}ms")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    # Wins by approach
    wins = {a: 0 for a in RetrievalApproach}
    for r in results:
        wins[r.winner] += 1

    print("\n  Wins by Approach:")
    for approach, count in sorted(wins.items(), key=lambda x: x[1], reverse=True):
        pct = count / len(results) * 100
        print(f"    {approach.value:15} {count}/{len(results)} ({pct:.0f}%)")

    # By difficulty
    print("\n  Performance by Difficulty:")
    for diff in QueryDifficulty:
        diff_results = [r for r in results if r.test_case.difficulty == diff]
        if not diff_results:
            continue
        print(f"\n    {diff.value.upper()}:")
        for approach in RetrievalApproach:
            recalls = [r.results[approach].recall for r in diff_results if approach in r.results]
            if recalls:
                avg = sum(recalls) / len(recalls)
                print(f"      {approach.value:15} avg_recall={avg:.1%}")

    # By category
    print("\n  Performance by Category:")
    categories = set(r.test_case.category for r in results)
    for cat in categories:
        cat_results = [r for r in results if r.test_case.category == cat]
        print(f"\n    {cat.upper()}:")
        for approach in RetrievalApproach:
            recalls = [r.results[approach].recall for r in cat_results if approach in r.results]
            if recalls:
                avg = sum(recalls) / len(recalls)
                wins_cat = sum(1 for r in cat_results if r.winner == approach)
                print(f"      {approach.value:15} avg={avg:.1%} wins={wins_cat}/{len(cat_results)}")

    # Hardest cases
    print("\n  Hardest Cases (all approaches <50% recall):")
    hard_cases = [r for r in results if r.all_failed]
    if hard_cases:
        for r in hard_cases:
            print(f"    - {r.test_case.description}")
            best_recall = max(res.recall for res in r.results.values())
            print(f"      Best recall: {best_recall:.1%}")
    else:
        print("    (none - all test cases had at least one approach with >50% recall)")

    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETE")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
