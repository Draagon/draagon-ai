# Phase 7: Production-Grade Retrieval Benchmark

**Status:** Ready for Implementation
**Priority:** Critical
**Duration:** 4 weeks (1 engineer)
**Dependencies:** FR-009 (Integration Testing Framework)

---

## Overview

Build a comprehensive benchmark suite that validates draagon-ai's retrieval pipeline at production scale using industry-standard methodologies (BEIR, RAGAS, MTEB, HotpotQA).

**Goal:** Prove the retrieval pipeline works at scale and leads the industry, not just at toy scale.

### Corpus Diversity (Updated)

The benchmark uses a **diverse corpus** representing real-world content variety:

| Category | Count | % | Examples |
|----------|-------|---|----------|
| **Technical** | 125 | 25% | Code docs, API refs, READMEs |
| **Narrative** | 75 | 15% | Stories, Wikipedia, fiction |
| **Knowledge Base** | 75 | 15% | FAQs, Stack Overflow, how-to |
| **Legal** | 50 | 10% | ToS, contracts, court opinions, GDPR |
| **Conversational** | 50 | 10% | Chat, email, support tickets |
| **Academic** | 50 | 10% | arXiv, research, scientific articles |
| **News/Blog** | 50 | 10% | Blog posts, news articles |
| **Synthetic** | 25 | 5% | LLM-generated distractors |

**Why diversity matters:** A production RAG system must handle legal jargon, informal chat, academic citations, and narrative prose - not just technical documentation.

---

## Task Breakdown

### Week 1: Corpus Assembly (Days 1-3)
| Task | Priority | Effort | Dependencies | Status |
|------|----------|--------|--------------|--------|
| **TASK-070**: Document Data Models | P0 | 1 day | None | Pending |
| **TASK-071**: Local Document Scanner | P0 | 1 day | TASK-070 | Pending |
| **TASK-072**: Online Documentation Fetcher | P1 | 1 day | TASK-070 | Pending |
| **TASK-073**: Synthetic Distractor Generator | P1 | 1 day | TASK-070 | Pending |
| **TASK-074**: CorpusBuilder Orchestrator | P0 | 1 day | TASK-070, 071, 072, 073 | Pending |

### Week 1-2: Query Suite (Days 4-7)
| Task | Priority | Effort | Dependencies | Status |
|------|----------|--------|--------------|--------|
| **TASK-075**: Query Data Models | P0 | 1 day | None | Pending |
| **TASK-076**: Multi-Hop Query Generator | P0 | 2 days | TASK-075 | Pending |
| **TASK-077**: Zero-Result & Adversarial Queries | P1 | 1 day | TASK-075 | Pending |

### Week 2: RAGAS Integration (Days 8-10)
| Task | Priority | Effort | Dependencies | Status |
|------|----------|--------|--------------|--------|
| **TASK-078**: RAGAS Metrics Integration | P0 | 2 days | TASK-074, 075 | Pending |
| **TASK-079**: Industry Comparison Framework | P1 | 1 day | TASK-078 | Pending |

### Week 2-3: Embedding Quality (Days 11-13)
| Task | Priority | Effort | Dependencies | Status |
|------|----------|--------|--------------|--------|
| **TASK-080**: Ollama mxbai-embed-large Integration | P0 | 1 day | None | Pending |
| **TASK-081**: SentenceTransformer Fallback | P1 | 1 day | None | Pending |
| **TASK-082**: Embedding Quality Validation | P0 | 1 day | TASK-080, 081 | Pending |

### Week 3: Statistical Framework (Days 14-16)
| Task | Priority | Effort | Dependencies | Status |
|------|----------|--------|--------------|--------|
| **TASK-083**: Multiple-Run Harness | P0 | 1 day | TASK-078 | Pending |
| **TASK-084**: Statistical Reporting | P0 | 1 day | TASK-083 | Pending |
| **TASK-085**: Variance Analysis Tools | P1 | 1 day | TASK-084 | Pending |

### Week 3-4: Infrastructure (Days 17-20)
| Task | Priority | Effort | Dependencies | Status |
|------|----------|--------|--------------|--------|
| **TASK-086**: BenchmarkRunner Orchestrator | P0 | 2 days | TASK-074, 078, 083 | Pending |
| **TASK-087**: Checkpointing & Progress Tracking | P1 | 1 day | TASK-086 | Pending |
| **TASK-088**: Report Generation (Markdown/CSV) | P0 | 1 day | TASK-086 | Pending |

### Week 4: Validation (Days 21-23)
| Task | Priority | Effort | Dependencies | Status |
|------|----------|--------|--------------|--------|
| **TASK-089**: Full Benchmark Execution | P0 | 2 days | All previous | Pending |
| **TASK-090**: Industry Baseline Comparison | P0 | 1 day | TASK-089 | Pending |

### Week 4: CI/CD Integration (Days 24-28)
| Task | Priority | Effort | Dependencies | Status |
|------|----------|--------|--------------|--------|
| **TASK-091**: Smoke Test Suite (50 queries) | P0 | 1 day | TASK-086 | Pending |
| **TASK-092**: GitHub Actions Workflow | P0 | 1 day | TASK-091 | Pending |
| **TASK-093**: Regression Detection Setup | P1 | 1 day | TASK-092 | Pending |

---

## Dependency Graph

```
TASK-070 (Document Models)
  ├─> TASK-071 (Local Scanner)
  ├─> TASK-072 (Online Fetcher)
  ├─> TASK-073 (Distractor Generator)
  └─> TASK-074 (CorpusBuilder)
       └─> TASK-078 (RAGAS Integration)
            └─> TASK-083 (Multiple Runs)
                 └─> TASK-086 (BenchmarkRunner)
                      ├─> TASK-089 (Full Benchmark)
                      │    └─> TASK-090 (Industry Comparison)
                      └─> TASK-091 (Smoke Tests)
                           └─> TASK-092 (CI/CD Workflow)

TASK-075 (Query Models)
  ├─> TASK-076 (Multi-Hop Generator)
  ├─> TASK-077 (Zero-Result & Adversarial)
  └─> TASK-078 (RAGAS Integration)

TASK-080 (Ollama Embedding)
  ├─> TASK-081 (SentenceTransformer)
  └─> TASK-082 (Embedding Validation)
       └─> TASK-086 (BenchmarkRunner)
```

---

## Success Criteria

### Production Readiness
- [ ] 500+ document corpus assembled
- [ ] 250+ test queries created (50 multi-hop, 25 zero-result, 40 adversarial)
- [ ] RAGAS metrics implemented (faithfulness, context recall, answer relevance)
- [ ] Real MTEB-benchmarked embeddings (mxbai-embed-large)
- [ ] Statistical validity: 5+ runs, p-values < 0.05
- [ ] Benchmark runtime ≤ 30 minutes (full suite)

### Industry Leadership
- [ ] Hybrid approach beats BM25 baseline (+20% nDCG@10)
- [ ] Hybrid approach beats Contriever (+10% nDCG@10)
- [ ] Faithfulness ≥ 0.80 (production standard)
- [ ] Context Recall ≥ 0.80 (comprehensive retrieval)
- [ ] Multi-hop query success rate ≥ 70%
- [ ] Zero-result query accuracy = 100% (no hallucinations)

### CI/CD Integration
- [ ] Smoke tests run on all PRs (≤ 5 minutes)
- [ ] Nightly full benchmarks with historical tracking
- [ ] Regression detection: -5% threshold triggers failure
- [ ] Results published to `.specify/benchmarks/results/`

---

## Implementation Sequence

**Recommended order for maximum parallelization:**

### Week 1 (3 engineers possible)
- **Engineer 1**: TASK-070, TASK-071, TASK-074 (corpus assembly critical path)
- **Engineer 2**: TASK-072, TASK-073 (online/synthetic parallel work)
- **Engineer 3**: TASK-075, TASK-076 (query suite can start independently)

### Week 2 (2 engineers optimal)
- **Engineer 1**: TASK-078 (RAGAS integration - critical path)
- **Engineer 2**: TASK-080, TASK-081, TASK-082 (embedding quality - independent)

### Week 3 (2 engineers optimal)
- **Engineer 1**: TASK-083, TASK-084 (statistical framework)
- **Engineer 2**: TASK-086, TASK-087, TASK-088 (benchmark runner)

### Week 4 (1 engineer for validation + integration)
- **Engineer 1**: TASK-089, TASK-090, TASK-091, TASK-092, TASK-093 (serial execution required)

---

## Quick Start (Single Engineer)

**First 3 days (minimum viable benchmark):**

1. **Day 1**: TASK-070 (models) → TASK-071 (local scanner) → Scan 100 docs
2. **Day 2**: TASK-075 (query models) → Create 10 test queries manually
3. **Day 3**: TASK-078 (basic RAGAS) → Run simple benchmark

This gives a **working prototype** in 3 days. Then iterate to add:
- More documents (TASK-072, 073, 074)
- More queries (TASK-076, 077)
- Statistical rigor (TASK-083, 084)
- Full infrastructure (TASK-086, 087, 088)

---

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| **Corpus assembly slow** | Start with local scanner only (TASK-071), defer online/synthetic |
| **Query creation labor-intensive** | Use LLM to generate query templates, human validation |
| **RAGAS integration complex** | Use library's default config first, tune later |
| **Benchmark runtime > 30 min** | Implement parallelization (multiprocessing), add checkpointing |
| **CI/CD tests flaky** | Cache corpus/queries, use deterministic seeds, retry logic |

---

## Next Steps

1. **Review task breakdown** (validate estimates, dependencies)
2. **Assign TASK-070** (foundational - must complete first)
3. **Setup development environment** (Ollama, RAGAS library)
4. **Create feature branch** (`feature/FR-012-benchmark`)
5. **Begin implementation** (start with TASK-070)

---

**Document Status:** Active
**Last Updated:** 2026-01-02
**Owner:** TBD
