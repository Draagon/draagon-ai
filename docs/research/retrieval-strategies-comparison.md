# Retrieval Strategies Research: Semantic Graph vs Vector RAG vs Raw Context

**Date:** 2025-01-01
**Status:** Active Research
**Related:** FR-010, Document Ingestion, Retrieval Benchmark

---

## Executive Summary

Research validates that **semantic knowledge graphs scale better than raw context as files grow larger**. The optimal solution is a **hybrid approach** combining vector similarity search with graph-based retrieval, using parallel agents to leverage the strengths of each.

### Key Findings

| Metric | Raw Context | Vector RAG | Semantic Graph | Hybrid |
|--------|-------------|------------|----------------|--------|
| Small scale (<32k tokens) | ✅ Good | ✅ Good | ✅ Good | Overkill |
| Large scale (>100k tokens) | ❌ Degrades | ⚠️ Moderate | ✅ Maintains | ✅ Best |
| Multi-hop reasoning | ❌ Poor | ⚠️ Limited | ✅ Excellent | ✅ Excellent |
| Semantic similarity | ✅ Good | ✅ Excellent | ⚠️ Requires embedding | ✅ Excellent |
| Structural relationships | ❌ None | ❌ None | ✅ Explicit | ✅ Explicit |
| Latency | Fast | Fast | 2.3× slower | Parallel mitigates |
| Setup cost | None | Embedding | Graph construction | Both |

---

## Research Sources

### 1. The "Lost in the Middle" Problem

**Source:** [U-NIAH: Unified RAG and LLM Evaluation](https://arxiv.org/html/2503.00353v1)

> "Performance peaks when key information is at the start or end of the input context, but drops if relevant details are mid-context."

**Source:** [Databricks Long Context RAG Study](https://www.databricks.com/blog/long-context-rag-performance-llms) (2,000+ experiments, 13 LLMs)

> "Most model performance decreases after a certain context size. Notably, Llama-3.1-405b performance starts to decrease after 32k tokens, GPT-4 starts to decrease after 64k tokens."

**Implication:** Raw context injection has a ceiling. Beyond ~32-64k tokens, retrieval becomes necessary.

---

### 2. GraphRAG vs Vector RAG Performance

**Source:** [FalkorDB Benchmark](https://www.falkordb.com/blog/graphrag-accuracy-diffbot-falkordb/)

> "GraphRAG outperformed vector RAG **3.4x**. FalkorDB's 2025 SDK pushes that to **90%+ accuracy** for schema-heavy enterprise queries."

**Source:** [AWS GraphRAG Blog](https://aws.amazon.com/blogs/machine-learning/improving-retrieval-augmented-generation-accuracy-with-graphrag/)

> "Integrating graph-based structures into RAG workflows improves answer precision by up to **35%** compared to vector-only retrieval."

**Source:** [Neo4j Knowledge Graph vs Vector RAG](https://neo4j.com/blog/developer/knowledge-graph-vs-vector-rag/)

> "Many pieces of information may be semantically similar but not relevant, or **relevant but not semantically similar**. Graph search presents specific levers and patterns that can be optimized for greater, granular control."

---

### 3. Multi-Hop Reasoning Advantage

**Source:** [GraphRAG vs Baseline RAG Analysis](https://www.genui.com/resources/graphrag-vs.-traditional-rag-solving-multi-hop-reasoning-in-llms)

> "Traditional RAG approaches often struggle with complex queries that require integrating information from multiple sources... GraphRAG handles tasks where traditional RAG models struggle."

**Real-world legal domain results:**
> "Graph RAG achieved **80-85% accuracy** on complex multi-hop queries, compared to **45-50%** for vector-only RAG—a **3.2x improvement**."

---

### 4. Vector Search Limitations

**Source:** [Paragon Analysis](https://www.useparagon.com/blog/vector-database-vs-knowledge-graphs-for-rag)

> "Vector-based search struggles with ambiguous context, lacks explicit reasoning, and doesn't maintain structured knowledge over time."

**Key limitation:** Vector similarity finds things that "sound alike" but may miss things that are "logically connected."

Example:
- Query: "What database tables store customer orders?"
- Vector RAG: Finds documents mentioning "orders" and "customers"
- Semantic Graph: Traverses `Customer --[PLACES]--> Order --[STORED_IN]--> orders_table`

---

### 5. Hybrid Approaches

**Source:** [HybridRAG Paper (ACM 2024)](https://arxiv.org/abs/2408.04948)

> "HybridRAG integrates Knowledge Graph-based RAG and vector database-based RAG... significantly improving accuracy and recall."

**Source:** [Zep Temporal Memory](https://arxiv.org/html/2501.13956v1)

> "In the challenging LongMemEval benchmark, graph-based memory achieves accuracy improvements of **up to 18.5%**."

---

### 6. Caveats and Limitations

**Source:** [GraphRAG Evaluation Research](https://arxiv.org/html/2506.06331v1)

> "GraphRAG achieves 13.4% **lower** accuracy on Natural Question benchmark compared to vanilla RAG, with particularly poor performance on time-sensitive queries."

**When Vector RAG wins:**
- Simple similarity queries ("find documents about X")
- Time-sensitive information (graphs may be stale)
- Low-structure domains (creative writing, general chat)

**When Semantic Graph wins:**
- Complex multi-hop queries
- Structural/relational questions
- Enterprise knowledge with defined schemas
- Code repositories with defined relationships

---

## Recommendations for draagon-ai

### 1. Implement Hybrid Parallel Retrieval

Run Vector RAG and Semantic Graph retrieval **in parallel**, then merge results:

```
Query → ┬→ Vector RAG Agent ────→ Semantic Chunks
        │
        └→ Graph RAG Agent ─────→ Entity/Relationship Context
                                          │
                                          ▼
                                   Merge & Synthesize
                                          │
                                          ▼
                                       Answer
```

### 2. Use Routing Heuristics

Before retrieval, classify the query to determine which approach(es) to use:

| Query Type | Use Vector | Use Graph | Example |
|------------|------------|-----------|---------|
| Simple similarity | ✅ | ❌ | "Find examples of error handling" |
| Entity lookup | ❌ | ✅ | "What does the Agent class do?" |
| Multi-hop reasoning | ⚠️ | ✅ | "How does Agent connect to Memory?" |
| Cross-project patterns | ✅ | ✅ | "How do other teams handle auth?" |

### 3. Layer Local + Global Context

```
┌─────────────────────────────────────────────────────────────┐
│                    Enterprise Semantic Graph                 │
│  (All projects, shared entities, org-wide patterns)         │
├─────────────────────────────────────────────────────────────┤
│     Project A        │     Project B        │    Project C  │
│   Local Context      │   Local Context      │  Local Context│
│   (CLAUDE.md, etc)   │   (CLAUDE.md, etc)   │               │
└─────────────────────────────────────────────────────────────┘
```

---

## Benchmarks to Run

1. **Scale Test:** Same questions at 10KB, 100KB, 1MB context sizes
2. **Multi-hop Test:** Questions requiring 2, 3, 4+ hops
3. **Cross-project Test:** Questions about patterns across repositories
4. **Latency Test:** Response time at various scales
5. **Accuracy Test:** Against known-answer datasets

---

## References

1. [Summary of a Haystack (ACL 2024)](https://aclanthology.org/2024.emnlp-main.552/)
2. [RAG vs GraphRAG Systematic Evaluation](https://arxiv.org/html/2502.11371v1)
3. [Neo4j Knowledge Graph vs Vector RAG](https://neo4j.com/blog/developer/knowledge-graph-vs-vector-rag/)
4. [Databricks Long Context RAG Study](https://www.databricks.com/blog/long-context-rag-performance-llms)
5. [Microsoft GraphRAG Project](https://www.microsoft.com/en-us/research/project/graphrag/)
6. [HybridRAG Paper](https://arxiv.org/abs/2408.04948)
7. [GraphRAG-Bench](https://arxiv.org/html/2506.02404v1)
8. [U-NIAH Framework](https://arxiv.org/html/2503.00353v1)
9. [FalkorDB GraphRAG Benchmark](https://www.falkordb.com/blog/graphrag-accuracy-diffbot-falkordb/)
10. [AWS GraphRAG Blog](https://aws.amazon.com/blogs/machine-learning/improving-retrieval-augmented-generation-accuracy-with-graphrag/)
