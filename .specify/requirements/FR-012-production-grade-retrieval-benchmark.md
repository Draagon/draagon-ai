# FR-012: Production-Grade Retrieval Pipeline Benchmark

**Status:** Specification
**Priority:** Critical
**Complexity:** High
**Created:** 2026-01-02
**Updated:** 2026-01-02

---

## Overview

A comprehensive benchmark suite that validates draagon-ai's retrieval pipeline at production scale using industry-standard methodologies (BEIR, RAGAS, MTEB, HotpotQA) with 500+ real documents, multi-hop queries, distractor documents, and zero-result cases.

**Core Goal:** Prove the retrieval pipeline works at scale and leads the industry, not just at toy scale.

**Research Foundation:**
- [BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models](https://arxiv.org/abs/2104.08663)
- [RAGAS: Automated Evaluation of Retrieval Augmented Generation](https://aclanthology.org/2024.eacl-demo.16/)
- [MTEB: Massive Text Embedding Benchmark](https://github.com/embeddings-benchmark/mteb)
- [HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering](https://hotpotqa.github.io/)

---

## Motivation

### Current State

The retrieval pipeline has been validated with 8-12 document experiments showing:
- **Hybrid approach**: 88.9% retrieval, 72.2% answer quality
- **Query expansion**: 300% improvement (22.2% → 88.9%)
- **Memory merge strategies**: HYBRID > EAGER/LAZY

**Problem:** These are **training wheels benchmarks**. They prove the concept works at toy scale but don't validate production readiness.

### Industry Gap Analysis

| Metric | Current Experiments | Industry Standard (BEIR) | Gap |
|--------|---------------------|--------------------------|-----|
| **Corpus Size** | 8-12 documents | 500K-5M documents | 41,666× smaller |
| **Query Complexity** | Single-hop, explicit | Multi-hop, implicit reasoning | Not tested |
| **Evaluation** | Keyword matching | RAGAS metrics (Faithfulness, Context Relevance) | Mock vs Real |
| **Ground Truth** | Hand-crafted expected keywords | Human-annotated relevance judgments | Subjective vs Objective |
| **Distractor Ratio** | ~1:3 (relevant:irrelevant) | ~1:1000+ | 300× easier |
| **Statistical Validity** | 1 run per test | 5+ runs with variance | No confidence intervals |
| **Embedding Quality** | Mock (keyword matching) | Real embeddings (MTEB benchmarked) | Not semantic |

**Honest Assessment:**
- Current 88.9% retrieval rate: In 8-doc corpus, random chance = 33%. Not impressive.
- Current tests: **Easy Mode**. Finding "Luna, Mochi, Pepper" in 3 statements is trivial.
- Production validation: **Missing**. Untested on real corpus scale, multi-hop reasoning, adversarial cases.

### What Production-Ready Means

Based on [Google Cloud RAG Best Practices](https://cloud.google.com/blog/products/ai-machine-learning/optimizing-rag-retrieval) and [Pinecone Production RAG](https://www.pinecone.io/learn/series/vector-databases-in-production-for-busy-engineers/rag-evaluation/):

| Requirement | Current Status | Production Target |
|-------------|---------------|-------------------|
| **Corpus Scale** | 8 docs | 500+ minimum, 10K+ ideal |
| **Multi-Hop Queries** | None | 20%+ of test queries |
| **Distractor Documents** | Minimal | 70%+ corpus are distractors |
| **Zero-Result Queries** | None | 10%+ of test queries |
| **Adversarial Cases** | None | Paraphrased, misleading, contradictory |
| **Statistical Testing** | 1 run | 5+ runs with p-values |
| **Real Embeddings** | Mock keyword | MTEB-benchmarked models |
| **Evaluation Metrics** | Keyword matching | RAGAS: Faithfulness (0.7+), Context Relevance (0.8+) |

---

## Requirements

### FR-012.1: Large-Scale Document Corpus (500+ Documents)

**Description:** Assemble a diverse corpus of 500+ documents representing real-world content variety - not just technical documentation. A production RAG system must handle narrative, legal, conversational, and academic content.

**Corpus Distribution:**

| Category | Count | % | Purpose |
|----------|-------|---|---------|
| **Technical Documentation** | 125 | 25% | Current strength, code/API docs |
| **Narrative/Creative** | 75 | 15% | Story comprehension, long-form prose |
| **Knowledge Base/FAQ** | 75 | 15% | Q&A retrieval, structured answers |
| **Legal/Contracts** | 50 | 10% | Dense formal language, precise terminology |
| **Conversational** | 50 | 10% | Informal language, chat/email style |
| **Academic/Research** | 50 | 10% | Formal/dense text, citations |
| **News/Blog** | 50 | 10% | Current events style, mixed formality |
| **Synthetic Distractors** | 25 | 5% | Robustness testing across ALL categories |
| **Total** | **500** | **100%** | **Production-realistic diversity** |

**Sources by Category:**

1. **Technical Documentation** (125 documents)
   - Local `~/Development`: draagon-ai, metaobjects-core, roxy-voice-assistant
   - Online: Python stdlib, FastAPI, Neo4j, Anthropic Claude docs
   - Filter: 1KB-500KB, exclude node_modules/target/.archive

2. **Narrative/Creative** (75 documents)
   - Local `party-lore/`: Game stories, character descriptions, world-building
   - Online: Wikipedia articles (history, science, culture)
   - Online: Project Gutenberg excerpts (public domain fiction)
   - Purpose: Test long-form prose comprehension, story retrieval

3. **Knowledge Base/FAQ** (75 documents)
   - Online: Stack Overflow top answers (Python, JavaScript tags)
   - Online: Product FAQ pages (GitHub, Notion, Slack help centers)
   - Online: How-to guides and tutorials
   - Purpose: Test Q&A style retrieval, structured content

4. **Legal/Contracts** (50 documents)
   - Online: Terms of Service (Apple, Google, GitHub, Microsoft)
   - Online: Privacy Policies (GDPR-compliant examples)
   - Online: Open-source licenses (MIT, Apache 2.0, GPL full text)
   - Online: Court opinions from CourtListener.com (US federal, free)
   - Online: SEC filings from EDGAR (10-K excerpts, public)
   - Online: EU regulations from eur-lex.europa.eu (GDPR, AI Act)
   - Online: Contract templates (NDA, SLA samples from LawInsider)
   - Purpose: Test dense terminology, cross-references, negation complexity

5. **Conversational** (50 documents)
   - Synthetic: LLM-generated chat transcripts, support ticket threads
   - Synthetic: Email thread simulations (professional, casual)
   - Online: Reddit discussions (technical subreddits, anonymized)
   - Purpose: Test informal language, incomplete sentences, context-dependent meaning

6. **Academic/Research** (50 documents)
   - Online: arXiv abstracts (cs.AI, cs.CL, cs.LG categories)
   - Online: Research paper summaries and introductions
   - Online: Wikipedia scientific articles (physics, biology, chemistry)
   - Purpose: Test formal academic writing, citation-heavy text, technical vocabulary

7. **News/Blog** (50 documents)
   - Online: Tech blog posts (Hacker News top articles, dev.to)
   - Online: News articles (tech, science sections from major outlets)
   - Synthetic: LLM-generated blog posts on various topics
   - Purpose: Test journalistic style, mixed formality, current events

8. **Synthetic Distractors** (25 documents)
   - Generated across ALL categories (not just technical)
   - Topics: Random mix matching each category's style
   - Distribution: 50% very different, 30% somewhat similar, 20% very similar (hard negatives)
   - Purpose: Test robustness against misleading but irrelevant content

**Why Legal Documents Are Particularly Challenging:**

Legal text tests retrieval in unique ways that prove semantic understanding:
- **Dense terminology**: "indemnification", "force majeure", "severability"
- **Long sentences**: Single sentences spanning 100+ words
- **Cross-references**: "Subject to Section 4.2(b)(iii) above..."
- **Negation complexity**: "shall not be liable except where..."
- **Precise definitions**: Same word means different things in different contexts
- **Formal structure**: Numbered sections, subsections, clauses

If the pipeline handles legal docs well, it proves semantic understanding beyond keyword matching.

**Document Metadata:**

```python
class DocumentCategory(str, Enum):
    """Content category for corpus diversity tracking."""
    TECHNICAL = "technical"
    NARRATIVE = "narrative"
    KNOWLEDGE_BASE = "knowledge_base"
    LEGAL = "legal"
    CONVERSATIONAL = "conversational"
    ACADEMIC = "academic"
    NEWS_BLOG = "news_blog"
    SYNTHETIC = "synthetic"

@dataclass
class BenchmarkDocument:
    """Document in benchmark corpus."""
    doc_id: str                    # Unique identifier
    source: str                    # "local", "online", "synthetic"
    category: DocumentCategory     # Content category (technical, legal, narrative, etc.)
    domain: str                    # Specific domain (python, contract_law, fiction, etc.)
    file_path: str                 # Original file path or URL
    content: str                   # Full document text
    chunk_ids: list[str]           # IDs of chunks created from this doc
    metadata: dict                 # Source-specific metadata
    is_distractor: bool            # True if synthetic distractor
    semantic_tags: list[str]       # For relevance judgment
```

**Implementation:**

```python
from draagon_ai.testing.benchmarks import CorpusBuilder, DocumentCategory

# Build corpus with diverse content
builder = CorpusBuilder(target_size=500, cache_dir="/tmp/benchmark_corpus")

# 1. Technical Documentation (125 docs, 25%)
await builder.add_local_documents(
    root_path="~/Development",
    patterns=["**/*.md", "**/*.py", "**/*.java"],
    category=DocumentCategory.TECHNICAL,
    exclude_patterns=["**/target/**", "**/node_modules/**", "**/party-lore/**"],
    max_docs=75,
)
await builder.add_online_documentation(
    category=DocumentCategory.TECHNICAL,
    sources=[
        ("python", "https://docs.python.org/3.11/"),
        ("fastapi", "https://fastapi.tiangolo.com/"),
        ("anthropic", "https://docs.anthropic.com/claude/"),
    ],
    max_docs=50,
)

# 2. Narrative/Creative (75 docs, 15%)
await builder.add_local_documents(
    root_path="~/Development/party-lore",
    patterns=["**/*.md"],
    category=DocumentCategory.NARRATIVE,
    max_docs=25,
)
await builder.add_online_content(
    category=DocumentCategory.NARRATIVE,
    sources=[
        ("wikipedia", "https://en.wikipedia.org/wiki/", topics=["History", "Science"]),
        ("gutenberg", "https://www.gutenberg.org/", max_excerpts=25),
    ],
    max_docs=50,
)

# 3. Knowledge Base/FAQ (75 docs, 15%)
await builder.add_online_content(
    category=DocumentCategory.KNOWLEDGE_BASE,
    sources=[
        ("stackoverflow", "top_answers", tags=["python", "javascript"]),
        ("github_help", "https://docs.github.com/"),
    ],
    max_docs=75,
)

# 4. Legal/Contracts (50 docs, 10%)
await builder.add_legal_documents(
    sources=[
        ("tos", ["apple", "google", "github", "microsoft"]),
        ("licenses", ["MIT", "Apache-2.0", "GPL-3.0"]),
        ("courtlistener", "federal_opinions", max_docs=15),
        ("sec_edgar", "10k_excerpts", max_docs=10),
        ("eur_lex", ["GDPR", "AI_Act"]),
    ],
    max_docs=50,
)

# 5. Conversational (50 docs, 10%)
await builder.add_synthetic_content(
    category=DocumentCategory.CONVERSATIONAL,
    templates=["chat_transcript", "support_ticket", "email_thread"],
    max_docs=50,
)

# 6. Academic/Research (50 docs, 10%)
await builder.add_online_content(
    category=DocumentCategory.ACADEMIC,
    sources=[
        ("arxiv", "abstracts", categories=["cs.AI", "cs.CL", "cs.LG"]),
        ("wikipedia", "scientific_articles"),
    ],
    max_docs=50,
)

# 7. News/Blog (50 docs, 10%)
await builder.add_online_content(
    category=DocumentCategory.NEWS_BLOG,
    sources=[
        ("hackernews", "top_articles"),
        ("devto", "featured_posts"),
    ],
    max_docs=50,
)

# 8. Synthetic Distractors (25 docs, 5%)
await builder.add_synthetic_distractors(
    categories=[cat for cat in DocumentCategory],  # All categories
    count=25,
    similarity_distribution={"very_different": 0.5, "somewhat_similar": 0.3, "very_similar": 0.2},
)

# Build and cache corpus
corpus = await builder.build()
corpus.save("/tmp/benchmark_corpus/corpus_v1.json")

# Validate diversity
assert corpus.get_category_distribution() == {
    DocumentCategory.TECHNICAL: 125,
    DocumentCategory.NARRATIVE: 75,
    DocumentCategory.KNOWLEDGE_BASE: 75,
    DocumentCategory.LEGAL: 50,
    DocumentCategory.CONVERSATIONAL: 50,
    DocumentCategory.ACADEMIC: 50,
    DocumentCategory.NEWS_BLOG: 50,
    DocumentCategory.SYNTHETIC: 25,
}
```

**Success Criteria:**
- [ ] 500+ documents total
- [ ] Category distribution matches target (±10% tolerance)
- [ ] 8 distinct content categories represented
- [ ] Legal documents include: ToS, licenses, court opinions, regulations
- [ ] Narrative documents include: stories, Wikipedia, fiction excerpts
- [ ] Conversational documents include: chat, email, support tickets
- [ ] Size distribution: 1KB-500KB per document
- [ ] No duplicate content (hash-based deduplication)
- [ ] Each category has queries that specifically test it

---

### FR-012.2: Multi-Hop Query Test Suite

**Description:** Create test queries requiring multi-hop reasoning across 2+ documents, following HotpotQA methodology.

**Query Types:**

1. **Bridge Queries** (Requires finding entity A, then using A to find entity B)
   - "What cognitive service uses the LayeredMemoryProvider?"
   - "Which framework feature does Roxy's learning service implement?"

2. **Comparison Queries** (Requires retrieving facts about A and B, then comparing)
   - "What's the difference between episodic and semantic memory TTL?"
   - "Compare Python's asyncio.gather vs TaskGroup for concurrent tasks"

3. **Aggregation Queries** (Requires collecting facts from 3+ documents)
   - "List all memory types across draagon-ai, Roxy, and metaobjects"
   - "What are all the testing frameworks mentioned in the repos?"

4. **Temporal Reasoning Queries** (Requires understanding chronology)
   - "Which features were added to draagon-ai after the Roxy integration?"
   - "What changed in Neo4j drivers between version 4 and 5?"

5. **Negation Queries** (Requires proving absence)
   - "Which cognitive services DON'T use Neo4j?"
   - "What Python features are NOT used in the agent loop?"

**Query Construction:**

```python
from draagon_ai.testing.benchmarks import MultiHopQuery, QueryType, HopDescription

# Example: Bridge query
query = MultiHopQuery(
    query_id="mh_001",
    question="What cognitive service uses the LayeredMemoryProvider?",
    query_type=QueryType.BRIDGE,
    hops=[
        HopDescription(
            step=1,
            reasoning="Find LayeredMemoryProvider usage",
            required_document_ids=["draagon_ai_claude_md"],
            required_facts=["LayeredMemoryProvider is the 4-layer memory implementation"],
        ),
        HopDescription(
            step=2,
            reasoning="Identify which cognitive service references it",
            required_document_ids=["draagon_ai_learning_py", "draagon_ai_beliefs_py"],
            required_facts=["Learning service stores facts in semantic memory"],
        ),
    ],
    expected_answer_contains=["learning service", "semantic memory"],
    minimum_documents_required=2,
    maximum_documents_sufficient=4,
    difficulty=QueryDifficulty.MEDIUM,
)
```

**Difficulty Distribution:**

| Difficulty | % of Queries | Characteristics |
|------------|-------------|-----------------|
| **EASY** | 20% | 2-hop, keywords present, explicit relationships |
| **MEDIUM** | 50% | 2-3 hop, paraphrased keywords, implicit relationships |
| **HARD** | 25% | 3+ hop, no keyword overlap, requires inference |
| **EXPERT** | 5% | 4+ hop, contradictory sources, requires deep understanding |

**Success Criteria:**
- [ ] 50+ multi-hop queries
- [ ] All 5 query types represented
- [ ] Difficulty distribution: 20% EASY, 50% MEDIUM, 25% HARD, 5% EXPERT
- [ ] Average hops per query ≥ 2.5
- [ ] 100% queries have annotated ground truth (required documents + expected facts)
- [ ] 100% queries validated by human reviewers

---

### FR-012.3: Zero-Result Query Handling

**Description:** Test queries that have NO relevant documents in corpus (following [ARES framework](https://aclanthology.org/2024.naacl-long.20/) stress-testing methodology).

**Query Categories:**

1. **Out-of-Domain Queries**
   - "How do I configure Azure Functions?" (Azure not in corpus)
   - "What's the best strategy for Starcraft?" (Gaming not in corpus)

2. **Temporally Invalid Queries**
   - "What features were added in draagon-ai 2.0?" (Version doesn't exist)
   - "How does Python 3.13 handle async?" (Not documented yet)

3. **Nonsensical Queries**
   - "How many colors does the memory provider taste like?"
   - "When will the agent loop become sentient?"

4. **Contradictory Premise Queries**
   - "Why does draagon-ai require regex for semantic understanding?" (Violates constitution)
   - "How do I enable JSON output format?" (XML-only architecture)

**Expected Behavior:**

```python
from draagon_ai.testing.benchmarks import ZeroResultQuery, ExpectedBehavior

query = ZeroResultQuery(
    query_id="zr_001",
    question="How do I configure Azure Functions?",
    category=ZeroResultCategory.OUT_OF_DOMAIN,
    expected_behavior=ExpectedBehavior(
        should_return_answer=False,
        should_indicate_uncertainty=True,
        acceptable_responses=[
            "I don't have information about Azure Functions in my knowledge base",
            "No relevant documents found for Azure Functions",
            "This topic is outside my current knowledge domain",
        ],
        unacceptable_responses=[
            "Azure Functions are configured by..." (hallucination),
            "Based on the documents..." (false confidence),
        ],
        max_confidence_threshold=0.3,  # Should be very uncertain
    ),
)
```

**Success Criteria:**
- [ ] 25+ zero-result queries (10% of total)
- [ ] All 4 categories represented
- [ ] 100% queries correctly identified as unanswerable (confidence < 0.3)
- [ ] 0% hallucinated answers (faithfulness score = 1.0 for "I don't know" responses)
- [ ] Average response time ≤ 2x normal queries (should fail fast, not search forever)

---

### FR-012.4: Adversarial Query Suite

**Description:** Queries designed to challenge the retrieval pipeline with misleading keywords, paraphrasing, and contradictory information.

**Attack Vectors:**

1. **Keyword Stuffing** (Irrelevant docs with matching keywords)
   - Query: "How does memory consolidation work?"
   - Distractor doc: "Memory consolidation in PostgreSQL involves checkpoint writes..."
   - Relevant doc: "LayeredMemoryProvider consolidates working memory to episodic..."

2. **Semantic Paraphrasing** (Zero keyword overlap)
   - Query: "What's the lifespan of temporary conversation context?"
   - Relevant doc: "Working memory has a TTL of 5 minutes"
   - Challenge: "lifespan" → "TTL", "temporary conversation context" → "working memory"

3. **Contradictory Sources** (Multiple docs with conflicting claims)
   - Query: "How many layers does the memory architecture have?"
   - Doc A (correct): "4-layer memory: working, episodic, semantic, metacognitive"
   - Doc B (distractor): "3-tier caching: L1, L2, L3" (different context)
   - Doc C (outdated): "2-layer memory: short-term and long-term" (old design doc)

4. **Misleading Context** (Partially relevant but wrong conclusions)
   - Query: "Does draagon-ai use regex for semantic understanding?"
   - Doc A (correct): "NEVER use regex for semantic understanding"
   - Doc B (misleading): "Regex is allowed for security blocklists, TTS transforms..." (correct but misleading if misread)

**Evaluation:**

```python
from draagon_ai.testing.benchmarks import AdversarialQuery, DistractorDocument

query = AdversarialQuery(
    query_id="adv_001",
    question="How does memory consolidation work?",
    attack_vector=AttackVector.KEYWORD_STUFFING,
    distractor_documents=[
        DistractorDocument(
            doc_id="postgresql_internals",
            content="Memory consolidation in PostgreSQL involves checkpoint writes to disk...",
            keyword_overlap=0.8,  # "memory", "consolidation" both present
            semantic_relevance=0.1,  # Not relevant to draagon-ai
            is_hard_negative=True,
        ),
    ],
    correct_document_ids=["draagon_ai_memory_layered"],
    expected_retrieved_docs=["draagon_ai_memory_layered"],
    max_acceptable_false_positives=0,  # Should NOT retrieve PostgreSQL doc
)
```

**Success Criteria:**
- [ ] 40+ adversarial queries (15% of total)
- [ ] All 4 attack vectors represented
- [ ] 100% correct document retrieval despite distractors
- [ ] 0% false positives from keyword stuffing
- [ ] 80%+ correct retrieval for zero-keyword-overlap paraphrasing
- [ ] 90%+ correct source selection from contradictory documents

---

### FR-012.5: RAGAS Evaluation Metrics

**Description:** Implement [RAGAS framework](https://docs.ragas.io/en/latest/concepts/metrics/) metrics for reference-free RAG evaluation.

**Core Metrics:**

1. **Faithfulness** (Does answer come from retrieved context?)
   - Measures factual accuracy of generated response
   - Formula: `(# verified claims) / (# total claims in answer)`
   - Target: ≥ 0.70 (industry production standard)
   - Calculation: LLM judges whether each claim in answer is supported by retrieved documents

2. **Answer Relevance** (Does answer address the query?)
   - Measures how relevant the answer is to the question
   - Formula: `cosine_similarity(query_embedding, answer_embedding)`
   - Target: ≥ 0.80
   - Calculation: Embed query and answer, compute semantic similarity

3. **Context Precision** (Are retrieved docs relevant?)
   - Measures precision of retrieval (low noise)
   - Formula: `mean(precision@k for k in [1..K])`
   - Target: ≥ 0.75
   - Calculation: For each position k, check if document is relevant

4. **Context Recall** (Are all relevant docs retrieved?)
   - Measures recall of retrieval (comprehensive)
   - Formula: `(# retrieved relevant docs) / (# total relevant docs in corpus)`
   - Target: ≥ 0.80
   - Calculation: Check if ground truth documents were retrieved

5. **Context Relevance** (Overall retrieval quality)
   - Measures signal-to-noise ratio in retrieved context
   - Formula: `(# relevant chunks) / (# total retrieved chunks)`
   - Target: ≥ 0.70
   - Calculation: LLM judges relevance of each chunk

**Implementation:**

```python
from draagon_ai.testing.benchmarks import RAGASEvaluator
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

evaluator = RAGASEvaluator(
    llm_provider=llm,
    embedding_provider=embedder,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
)

# Evaluate single query
result = await evaluator.evaluate_single(
    query="How does memory consolidation work?",
    retrieved_contexts=[doc1.content, doc2.content, doc3.content],
    generated_answer="Memory consolidation in draagon-ai moves items from working to episodic...",
    ground_truth_contexts=[doc1, doc2],  # Known relevant docs
)

# Result breakdown
assert result.faithfulness >= 0.70  # 70%+ claims verified
assert result.answer_relevancy >= 0.80  # 80%+ relevant to query
assert result.context_precision >= 0.75  # 75%+ precision@k
assert result.context_recall >= 0.80  # 80%+ relevant docs retrieved
```

**Success Criteria:**
- [ ] All 5 RAGAS metrics implemented
- [ ] Faithfulness ≥ 0.70 (70% of claims verifiable from context)
- [ ] Answer Relevance ≥ 0.80 (80% semantic similarity to query)
- [ ] Context Precision ≥ 0.75 (75% of retrieved docs are relevant)
- [ ] Context Recall ≥ 0.80 (80% of relevant docs are retrieved)
- [ ] Context Relevance ≥ 0.70 (70% signal-to-noise ratio)
- [ ] Evaluation latency ≤ 5s per query (for production viability)

---

### FR-012.6: Statistical Validity and Variance

**Description:** Ensure benchmark results are statistically significant, not random noise.

**Requirements:**

1. **Multiple Runs per Query**
   - Run each query 5 times minimum
   - Capture variance due to:
     - LLM non-determinism (temperature, sampling)
     - Embedding model variations
     - Neo4j query planner differences
     - Concurrent load effects

2. **Statistical Reporting**
   - Report mean ± std deviation for all metrics
   - Calculate 95% confidence intervals
   - Use bootstrapping for small sample sizes
   - Report p-values for approach comparisons

3. **Variance Analysis**
   - Identify high-variance queries (std dev > 20%)
   - Investigate root causes (flaky, ambiguous, or genuinely uncertain)
   - Flag unstable metrics for replication studies

**Implementation:**

```python
from draagon_ai.testing.benchmarks import StatisticalValidator
import numpy as np
from scipy import stats

validator = StatisticalValidator(runs_per_query=5, confidence_level=0.95)

# Run benchmark multiple times
results = []
for run in range(5):
    run_result = await benchmark.run_all_queries(approach="hybrid")
    results.append(run_result)

# Statistical analysis
stats_report = validator.analyze(results)

print(f"Faithfulness: {stats_report.faithfulness.mean:.3f} ± {stats_report.faithfulness.std:.3f}")
print(f"95% CI: [{stats_report.faithfulness.ci_lower:.3f}, {stats_report.faithfulness.ci_upper:.3f}]")
print(f"Context Recall: {stats_report.context_recall.mean:.3f} ± {stats_report.context_recall.std:.3f}")

# Compare approaches
comparison = validator.compare_approaches(
    baseline_results=results_raw_context,
    treatment_results=results_hybrid,
)

print(f"Hybrid vs Raw Context:")
print(f"  Recall improvement: {comparison.recall_delta:.1%} (p={comparison.recall_pvalue:.4f})")
print(f"  Faithfulness improvement: {comparison.faithfulness_delta:.1%} (p={comparison.faithfulness_pvalue:.4f})")

# Flag unstable queries
unstable = validator.find_unstable_queries(threshold=0.20)
for query_id, variance in unstable:
    print(f"Warning: Query {query_id} has high variance (std={variance:.2%})")
```

**Success Criteria:**
- [ ] 5+ runs per query for all benchmarks
- [ ] Mean ± std deviation reported for all metrics
- [ ] 95% confidence intervals calculated
- [ ] P-values < 0.05 for claimed performance improvements
- [ ] High-variance queries (std > 20%) flagged and investigated
- [ ] Bootstrapping used for comparisons with n < 30

---

### FR-012.7: Real Embedding Quality (MTEB-Benchmarked)

**Description:** Replace mock keyword-based "embeddings" with real, MTEB-benchmarked embedding models.

**Model Selection Criteria:**

Based on [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard):

| Model | MTEB Score | Dimensions | Use Case |
|-------|-----------|-----------|----------|
| **mxbai-embed-large** | 64.68 | 1024 | Production (Ollama-compatible) |
| **all-MiniLM-L6-v2** | 58.80 | 384 | Fast baseline (local, no API) |
| **text-embedding-3-large** | 64.59 | 3072 | OpenAI (API required) |
| **gte-Qwen2-7B-instruct** | 72.71 | 3584 | State-of-the-art (requires GPU) |

**Implementation:**

```python
from draagon_ai.memory.embedding import (
    OllamaEmbeddingProvider,
    SentenceTransformerEmbeddingProvider,
    OpenAIEmbeddingProvider,
)

# Default: Ollama mxbai-embed-large (MTEB 64.68)
embedder = OllamaEmbeddingProvider(
    base_url="http://localhost:11434",
    model="mxbai-embed-large",
    dimension=1024,
)

# Fallback: SentenceTransformer (local, no API)
embedder_local = SentenceTransformerEmbeddingProvider(
    model_name="all-MiniLM-L6-v2",
    dimension=384,
)

# Premium: OpenAI (best quality, costs money)
embedder_premium = OpenAIEmbeddingProvider(
    model="text-embedding-3-large",
    dimension=3072,
)

# Benchmark all models
benchmark = RetrievalBenchmark(corpus=corpus, queries=queries)
results = {}

for name, embedder in [("mxbai", embedder), ("minilm", embedder_local)]:
    results[name] = await benchmark.run(
        embedding_provider=embedder,
        approach="hybrid",
    )

# Compare embedding quality
print(f"mxbai-embed-large: Context Recall = {results['mxbai'].context_recall:.2%}")
print(f"all-MiniLM-L6-v2: Context Recall = {results['minilm'].context_recall:.2%}")
```

**Success Criteria:**
- [ ] MTEB-benchmarked models used (score ≥ 58.0)
- [ ] Embedding quality validation: semantic similarity tests
- [ ] Model comparison: mxbai vs MiniLM vs baseline
- [ ] Embedding dimension ≥ 384 (semantic richness)
- [ ] Vector search integration validated (cosine similarity, approximate nearest neighbors)
- [ ] No mock embeddings in production benchmarks

---

### FR-012.8: Benchmark Execution Infrastructure

**Description:** Automated infrastructure for running benchmarks, collecting results, and generating reports.

**Components:**

1. **Benchmark Runner**
   - Orchestrates corpus loading, query execution, metric collection
   - Supports parallel execution (multi-process for CPU-bound tasks)
   - Checkpointing for long-running benchmarks (resume on failure)
   - Progress tracking with ETA

2. **Results Database**
   - Store all benchmark runs in structured format
   - Schema: `benchmark_runs(id, timestamp, approach, corpus_version, metrics)`
   - Enable historical comparison (is performance improving over time?)
   - Export to CSV, JSON, and markdown

3. **Report Generator**
   - Generate comprehensive markdown reports
   - Include: metrics table, statistical significance, variance analysis, failure cases
   - Visualizations: retrieval recall over time, metric distributions, approach comparisons
   - Publish to `.specify/benchmarks/results/`

**Implementation:**

```python
from draagon_ai.testing.benchmarks import BenchmarkRunner, BenchmarkConfig

config = BenchmarkConfig(
    corpus_path="/tmp/benchmark_corpus/corpus_v1.json",
    query_suite_path="/tmp/benchmark_corpus/queries_v1.json",
    approaches=["raw_context", "vector_rag", "semantic_graph", "hybrid"],
    runs_per_query=5,
    parallel_workers=4,
    checkpoint_dir="/tmp/benchmark_checkpoints",
)

runner = BenchmarkRunner(config)

# Run all benchmarks
results = await runner.run_all(
    llm_provider=llm,
    memory_provider=memory,
    embedding_provider=embedder,
)

# Generate report
report = runner.generate_report(results)
report.save(".specify/benchmarks/results/run_2026-01-02.md")

# Export for analysis
results.to_csv(".specify/benchmarks/results/run_2026-01-02.csv")
```

**Success Criteria:**
- [ ] Automated benchmark execution (single command)
- [ ] Parallel execution (4+ workers for CPU-bound tasks)
- [ ] Checkpointing (resume on failure)
- [ ] Progress tracking with ETA
- [ ] Results database (historical tracking)
- [ ] Markdown report generation (metrics, variance, failures)
- [ ] CSV export for external analysis
- [ ] Benchmark runtime ≤ 30 minutes for full suite (500 docs, 250 queries, 5 runs)

---

### FR-012.9: Industry Standard Comparison

**Description:** Compare draagon-ai retrieval performance to published baselines from BEIR, RAGAS, and academic research.

**Baseline Models:**

| Approach | BEIR nDCG@10 | RAGAS Faithfulness | Notes |
|----------|--------------|-------------------|-------|
| **BM25 (Lexical)** | 0.428 | 0.65 | Baseline sparse retrieval |
| **Contriever (Dense)** | 0.491 | 0.72 | Unsupervised dense retrieval |
| **ColBERT (Late Interaction)** | 0.537 | 0.76 | State-of-the-art zero-shot |
| **GPT-3.5 RAG (Naive)** | N/A | 0.68 | Simple vector retrieval |
| **GPT-3.5 RAG (Optimized)** | N/A | 0.82 | Query expansion + reranking |

**draagon-ai Targets:**

| Approach | Target nDCG@10 | Target Faithfulness | Rationale |
|----------|---------------|---------------------|-----------|
| **Raw Context** | 0.45 | 0.70 | Match BM25 baseline |
| **Vector/RAG** | 0.50 | 0.75 | Match Contriever |
| **Semantic Graph** | 0.48 | 0.73 | Unique approach, aim for competitive |
| **Hybrid** | **0.55+** | **0.80+** | **Beat GPT-3.5 optimized RAG** |

**Reporting:**

```python
from draagon_ai.testing.benchmarks import IndustryComparison

comparison = IndustryComparison()
comparison.add_baseline("BM25", ndcg=0.428, faithfulness=0.65)
comparison.add_baseline("Contriever", ndcg=0.491, faithfulness=0.72)
comparison.add_baseline("ColBERT", ndcg=0.537, faithfulness=0.76)

comparison.add_draagon_result("Hybrid", ndcg=results.ndcg, faithfulness=results.faithfulness)

report = comparison.generate_report()
print(report)
# Output:
# draagon-ai Hybrid vs Industry Baselines:
#   nDCG@10: 0.552 (vs BM25: +28.9%, vs Contriever: +12.4%, vs ColBERT: +2.8%)
#   Faithfulness: 0.815 (vs BM25: +25.4%, vs Contriever: +13.2%, vs ColBERT: +7.2%)
#   Rank: 1st of 4 approaches (LEADING INDUSTRY)
```

**Success Criteria:**
- [ ] Hybrid approach beats BM25 baseline (+20% nDCG@10)
- [ ] Hybrid approach beats Contriever (+10% nDCG@10)
- [ ] Hybrid approach competitive with ColBERT (±5% nDCG@10)
- [ ] Faithfulness ≥ 0.80 (optimized RAG standard)
- [ ] Results published with methodology for replication
- [ ] Confidence intervals overlap with industry baselines (statistical validity)

---

### FR-012.10: Continuous Benchmark Integration

**Description:** Integrate benchmarks into CI/CD pipeline to prevent performance regressions.

**CI/CD Integration:**

1. **Smoke Tests** (Run on every PR)
   - Subset: 50 queries, 100 documents (10% of full benchmark)
   - Metrics: Context Recall, Faithfulness
   - Threshold: -5% regression tolerance
   - Runtime: ≤ 5 minutes

2. **Nightly Full Benchmark**
   - Full suite: 250 queries, 500 documents, 5 runs
   - All RAGAS metrics
   - Historical tracking
   - Runtime: ≤ 30 minutes

3. **Weekly Comparison Benchmark**
   - Run all 4 approaches (raw, vector, graph, hybrid)
   - Statistical comparison with p-values
   - Generate trend reports
   - Runtime: ≤ 2 hours

**GitHub Actions Workflow:**

```yaml
name: Retrieval Benchmark

on:
  pull_request:
    paths:
      - 'src/draagon_ai/memory/**'
      - 'src/draagon_ai/orchestration/**'
  schedule:
    - cron: '0 2 * * *'  # 2 AM UTC daily

jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install -e ".[test]"
          pip install ragas sentence-transformers

      - name: Start Ollama
        run: |
          docker run -d -p 11434:11434 ollama/ollama
          docker exec ollama ollama pull mxbai-embed-large

      - name: Run smoke benchmark (PR only)
        if: github.event_name == 'pull_request'
        run: pytest tests/benchmarks/test_retrieval_smoke.py -v
        env:
          GROQ_API_KEY: ${{ secrets.GROQ_API_KEY }}

      - name: Run full benchmark (Nightly)
        if: github.event_name == 'schedule'
        run: pytest tests/benchmarks/test_retrieval_full.py -v
        env:
          GROQ_API_KEY: ${{ secrets.GROQ_API_KEY }}

      - name: Upload results
        uses: actions/upload-artifact@v3
        with:
          name: benchmark-results
          path: .specify/benchmarks/results/
```

**Success Criteria:**
- [ ] Smoke tests run on all PRs (≤ 5 minutes)
- [ ] Nightly full benchmarks with historical tracking
- [ ] Weekly comparison benchmarks (all approaches)
- [ ] Regression detection: -5% threshold triggers failure
- [ ] Results published to `.specify/benchmarks/results/`
- [ ] Trend visualization (performance over time)

---

## Technical Architecture

### Class Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                      BenchmarkRunner                         │
├─────────────────────────────────────────────────────────────┤
│ + config: BenchmarkConfig                                    │
│ + corpus: DocumentCorpus                                     │
│ + queries: QuerySuite                                        │
│ + evaluator: RAGASEvaluator                                  │
├─────────────────────────────────────────────────────────────┤
│ + run_all(approaches, runs) → BenchmarkResults               │
│ + run_approach(approach, queries) → ApproachResults          │
│ + generate_report(results) → MarkdownReport                  │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                      DocumentCorpus                          │
├─────────────────────────────────────────────────────────────┤
│ + documents: list[BenchmarkDocument]                         │
│ + metadata: CorpusMetadata                                   │
├─────────────────────────────────────────────────────────────┤
│ + load_from_cache(path) → DocumentCorpus                     │
│ + save_to_cache(path)                                        │
│ + get_document(doc_id) → BenchmarkDocument                   │
│ + get_distractors(domain, count) → list[BenchmarkDocument]  │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                      QuerySuite                              │
├─────────────────────────────────────────────────────────────┤
│ + queries: list[BenchmarkQuery]                              │
│ + multi_hop_queries: list[MultiHopQuery]                     │
│ + zero_result_queries: list[ZeroResultQuery]                 │
│ + adversarial_queries: list[AdversarialQuery]                │
├─────────────────────────────────────────────────────────────┤
│ + get_by_difficulty(difficulty) → list[BenchmarkQuery]       │
│ + get_by_type(query_type) → list[BenchmarkQuery]             │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                      RAGASEvaluator                          │
├─────────────────────────────────────────────────────────────┤
│ + llm_provider: LLMProvider                                  │
│ + embedding_provider: EmbeddingProvider                      │
│ + metrics: list[Metric]                                      │
├─────────────────────────────────────────────────────────────┤
│ + evaluate_single(query, contexts, answer) → RAGASResult     │
│ + evaluate_batch(items) → list[RAGASResult]                  │
└─────────────────────────────────────────────────────────────┘
```

### Data Models

```python
@dataclass
class BenchmarkDocument:
    doc_id: str
    source: str  # "local", "online", "synthetic"
    domain: str
    content: str
    chunk_ids: list[str]
    is_distractor: bool
    semantic_tags: list[str]
    metadata: dict


@dataclass
class MultiHopQuery:
    query_id: str
    question: str
    query_type: QueryType  # BRIDGE, COMPARISON, AGGREGATION, etc.
    hops: list[HopDescription]
    expected_answer_contains: list[str]
    minimum_documents_required: int
    difficulty: QueryDifficulty


@dataclass
class RAGASResult:
    query_id: str
    faithfulness: float  # 0.0-1.0
    answer_relevancy: float
    context_precision: float
    context_recall: float
    context_relevance: float
    retrieved_doc_ids: list[str]
    generated_answer: str
    evaluation_time_ms: float


@dataclass
class BenchmarkResults:
    approach: str
    corpus_version: str
    timestamp: datetime
    runs: list[RunResults]
    aggregate_metrics: AggregateMetrics
    statistical_report: StatisticalReport
```

---

## Implementation Plan

### Phase 1: Corpus Assembly (Week 1)
1. Implement `CorpusBuilder` for local document scanning
2. Add online documentation downloader with caching
3. Implement synthetic distractor generator using LLM
4. Build corpus deduplication and validation
5. Validate: 500+ documents, 70%+ distractors

### Phase 2: Query Suite Creation (Week 1-2)
1. Design multi-hop query templates
2. Generate 50+ multi-hop queries with ground truth
3. Create 25+ zero-result queries
4. Build 40+ adversarial queries
5. Human validation of all queries

### Phase 3: RAGAS Integration (Week 2)
1. Integrate RAGAS library
2. Implement faithfulness, answer relevancy, context precision/recall metrics
3. Build LLM-as-judge evaluation harness
4. Validate against RAGAS published baselines

### Phase 4: Embedding Quality (Week 2-3)
1. Integrate Ollama mxbai-embed-large
2. Add SentenceTransformer local fallback
3. Build embedding quality validation tests
4. Compare embedding models on test queries

### Phase 5: Statistical Framework (Week 3)
1. Implement multiple-run harness (5+ runs per query)
2. Add statistical reporting (mean, std, CI)
3. Build variance analysis tools
4. Implement p-value calculation for approach comparisons

### Phase 6: Benchmark Infrastructure (Week 3-4)
1. Build `BenchmarkRunner` orchestrator
2. Add checkpointing and progress tracking
3. Implement results database and historical tracking
4. Build markdown report generator

### Phase 7: Validation and Tuning (Week 4)
1. Run full benchmark suite (all approaches, 5 runs)
2. Compare to industry baselines (BEIR, RAGAS)
3. Identify failure modes and iterate
4. Publish final results and methodology

### Phase 8: CI/CD Integration (Week 4)
1. Build smoke test suite (50 queries)
2. Add GitHub Actions workflow
3. Configure nightly full benchmarks
4. Set up regression detection

---

## Success Criteria

### Production-Ready Validation
- [ ] 500+ document corpus assembled
- [ ] 250+ test queries created (50 multi-hop, 25 zero-result, 40 adversarial, 135 standard)
- [ ] RAGAS metrics implemented and validated
- [ ] Real MTEB-benchmarked embeddings used (no mocks)
- [ ] Statistical validity: 5+ runs, p-values < 0.05 for claims
- [ ] Hybrid approach beats BM25 baseline (+20% nDCG@10)
- [ ] Hybrid approach beats Contriever (+10% nDCG@10)
- [ ] Faithfulness ≥ 0.80 (optimized RAG standard)
- [ ] Context Recall ≥ 0.80 (comprehensive retrieval)
- [ ] Benchmark runtime ≤ 30 minutes (full suite)
- [ ] CI/CD smoke tests running on all PRs
- [ ] Results published with replication methodology

### Industry Leadership
- [ ] nDCG@10 ≥ 0.55 (beat GPT-3.5 optimized RAG)
- [ ] Faithfulness ≥ 0.80 (production standard)
- [ ] Multi-hop query success rate ≥ 70%
- [ ] Zero-result query accuracy = 100% (no hallucinations)
- [ ] Adversarial query robustness ≥ 90%
- [ ] Published comparison to BEIR/RAGAS baselines
- [ ] Methodology documented for community replication

---

## Open Questions

1. **Corpus Licensing:** Are all online documentation sources permissible for benchmark use? (Need to verify licenses)
2. **Compute Requirements:** Full benchmark with 500 docs × 250 queries × 5 runs = 625K retrieval operations. Need to estimate GPU/CPU hours.
3. **Ground Truth Annotation:** Multi-hop queries need human validation. Budget for annotation hours?
4. **Embedding Model Hosting:** Should we bundle Ollama in Docker for CI, or use cloud API? (Cost vs speed tradeoff)

---

## References

- **BEIR Benchmark:** [BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models](https://arxiv.org/abs/2104.08663)
- **RAGAS Framework:** [RAGAS: Automated Evaluation of Retrieval Augmented Generation](https://aclanthology.org/2024.eacl-demo.16/)
- **MTEB Leaderboard:** [MTEB: Massive Text Embedding Benchmark](https://huggingface.co/spaces/mteb/leaderboard)
- **HotpotQA:** [HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering](https://hotpotqa.github.io/)
- **ARES Framework:** [ARES: An Automated Evaluation Framework for Retrieval-Augmented Generation Systems](https://aclanthology.org/2024.naacl-long.20/)
- **Google Cloud RAG:** [Optimizing RAG: Retrieval Best Practices](https://cloud.google.com/blog/products/ai-machine-learning/optimizing-rag-retrieval)
- **Pinecone RAG Production:** [RAG Evaluation: Don't let customers tell you first](https://www.pinecone.io/learn/series/vector-databases-in-production-for-busy-engineers/rag-evaluation/)
- **HyDE: Precise Zero-Shot Dense Retrieval:** [Precise Zero-Shot Dense Retrieval without Relevance Labels](https://arxiv.org/abs/2212.10496)

---

**Document Status:** Specification Complete
**Next Step:** Review and approve specification, then proceed to planning phase
**Estimated Implementation:** 4 weeks (1 engineer)
**Dependencies:** FR-009 (Integration Testing Framework), Ollama, RAGAS library
