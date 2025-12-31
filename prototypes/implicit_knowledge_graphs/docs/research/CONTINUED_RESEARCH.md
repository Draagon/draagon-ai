# Continued Research Directions

**Version:** 1.0.0
**Last Updated:** 2025-12-31
**Status:** Research Backlog

---

## Overview

This document captures research directions to explore as the prototype evolves. Each section identifies gaps in current understanding and specific questions to investigate.

---

## 1. Decomposition Optimization

### Open Questions

1. **Decomposition Depth** - How deep should we decompose?
   - Every presupposition? Only high-confidence ones?
   - All ATOMIC relations? Only relevant ones?
   - Does deeper = better, or is there a sweet spot?

2. **Decomposition Order** - Does order matter?
   - WSD first (current approach) vs parallel
   - Does frame identification help WSD?
   - Should presuppositions inform entity resolution?

3. **LLM vs Rule-Based** - When to use which?
   - Lesk is fast but less accurate
   - LLM is accurate but expensive
   - Hybrid thresholds need optimization

### Research Tasks

- [ ] Benchmark decomposition depth vs retrieval quality
- [ ] Test different decomposition orders
- [ ] Compare cost/accuracy tradeoffs for LLM usage
- [ ] Investigate if decomposition can be parallelized effectively

### Relevant Papers to Read

- "How Much Knowledge Can You Pack Into the Parameters of a Language Model?" (Roberts et al., 2020)
- "Improving Language Understanding by Generative Pre-Training" (GPT-1 paper)
- Frame-semantic parsing benchmarks on SemEval

---

## 2. Graph Storage Optimization

### Open Questions

1. **Triple Granularity** - What's the right size?
   - Store full sentences + triples?
   - Triples only?
   - Hierarchical (sentence → frame → triple)?

2. **Embedding Strategy** - What to embed?
   - Embed triples as text: "Doug prefers tea"
   - Embed structured: concatenate subject + predicate + object embeddings
   - Embed with context: include presuppositions in embedding

3. **Graph vs Vector** - When does graph structure help?
   - At what scale does graph traversal outperform vector search?
   - Is hybrid always better?
   - What query types benefit most from graph?

### Research Tasks

- [ ] Benchmark different embedding strategies
- [ ] Test graph traversal vs vector search at different scales
- [ ] Investigate optimal triple text templates
- [ ] Compare Qdrant-only vs Qdrant+NetworkX hybrid

### Relevant Papers to Read

- "GraphRAG: Unlocking LLM Discovery on Narrative Private Data" (Microsoft, 2024)
- "RAG vs GraphRAG: A Systematic Evaluation" (2025)
- Knowledge Graph Embedding surveys

---

## 3. Retrieval Optimization

### Open Questions

1. **Context Packing** - How to maximize value per token?
   - Retrieve triples and reconstruct sentences?
   - Retrieve pre-formed context chunks?
   - Dynamic packing based on query type?

2. **Query Decomposition** - Should queries be decomposed same as storage?
   - Does query WSD improve retrieval?
   - Should we extract query presuppositions?
   - Multi-hop query decomposition strategies?

3. **Relevance Scoring** - What makes context "relevant"?
   - Semantic similarity alone?
   - Entity overlap?
   - Frame match?
   - Synset chain similarity?

### Research Tasks

- [ ] Benchmark different context packing strategies
- [ ] Test query decomposition impact on retrieval
- [ ] Develop relevance scoring that considers synsets
- [ ] Compare token efficiency across strategies

### Relevant Papers to Read

- "Learning to Compress: Unlocking the Potential of LLMs for Text Representation" (2025)
- "Context Compression for Long-horizon LLM Agents" (Acon, 2025)
- "LLMLingua-2: Data Distillation for Efficient Prompt Compression"

---

## 4. Evolution and Optimization

### Open Questions

1. **What's Evolvable?** - Which components benefit most?
   - Prompts (proven by Promptbreeder)
   - Weights (standard genetic approach)
   - Strategies (which algorithm to use)?
   - All of the above?

2. **Fitness Functions** - How to measure success?
   - End-to-end output quality only?
   - Intermediate metrics (WSD accuracy, retrieval precision)?
   - Multi-objective optimization?

3. **Evolution Speed** - How fast can we iterate?
   - LLM calls are expensive
   - Can we use cheaper models for evolution?
   - Surrogate fitness functions?

### Research Tasks

- [ ] Identify highest-impact evolvable components
- [ ] Design multi-objective fitness function
- [ ] Test surrogate fitness models (cheaper approximations)
- [ ] Benchmark evolution strategies (genetic vs Bayesian)

### Relevant Papers to Read

- "Promptbreeder: Self-Referential Self-Improvement Via Prompt Evolution" (2024)
- "Evolutionary Optimization of Neural Network Prompts" surveys
- Multi-objective optimization in ML

---

## 5. Evaluation and Benchmarking

### Open Questions

1. **What to Measure?** - Quality dimensions
   - Factual accuracy (did it get facts right?)
   - Completeness (did it answer fully?)
   - Relevance (was context used well?)
   - Efficiency (tokens used per quality point?)

2. **How to Measure?** - Evaluation methods
   - LLM-as-judge (current plan with Opus 4.5)
   - Human evaluation (gold standard but slow)
   - Automated metrics (BLEU, ROUGE - but are they meaningful?)
   - Task-specific evaluation (QA accuracy, etc.)

3. **What to Compare Against?** - Baselines
   - Vanilla RAG (chunk-based)
   - GraphRAG (Microsoft approach)
   - No retrieval (LLM knowledge only)
   - Human-curated context

### Research Tasks

- [ ] Develop robust multi-dimensional scoring rubric
- [ ] Calibrate LLM-as-judge against human judgment
- [ ] Create benchmark dataset for comparison
- [ ] Implement baseline RAG for fair comparison

### Relevant Papers to Read

- "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena" (2023)
- RAG evaluation benchmarks (BEIR, KILT, etc.)
- GraphRAG-Bench paper

---

## 6. Scale and Performance

### Open Questions

1. **At What Scale Does This Win?** - Crossover point
   - Theory: Graph approach wins at large scale
   - Where exactly is the crossover?
   - What's the memory/compute tradeoff?

2. **Context Window Constraints** - Optimization under limits
   - How does performance degrade with smaller windows?
   - Can we maintain quality with 50% fewer tokens?
   - What's the minimum viable context?

3. **Real-Time Feasibility** - Latency considerations
   - Is decomposition too slow for real-time?
   - Can we pre-decompose and cache?
   - What's acceptable latency for different use cases?

### Research Tasks

- [ ] Test at 100, 1K, 10K, 100K fact scales
- [ ] Measure quality vs context size curves
- [ ] Profile latency and identify bottlenecks
- [ ] Investigate caching strategies for decomposition

### Relevant Papers to Read

- Large-scale knowledge base papers (FB15k, WN18, etc.)
- Context window optimization research
- Caching strategies for ML pipelines

---

## 7. Domain-Specific Considerations

### Open Questions

1. **Domain Adaptation** - How to handle specific domains?
   - Medical terminology has different word senses
   - Legal language has precise meanings
   - Technical domains have jargon

2. **Multi-Domain** - Can one system handle multiple domains?
   - Domain detection as first step?
   - Domain-specific WSD models?
   - Transfer learning across domains?

3. **draagon-ai Use Cases** - Specific applications
   - Home Assistant voice control (Roxy)
   - Personal assistant memory
   - Multi-agent coordination

### Research Tasks

- [ ] Test on domain-specific text (home automation commands)
- [ ] Investigate domain detection approaches
- [ ] Evaluate transfer from general to specific domains
- [ ] Design domain-specific test cases

---

## 8. Commonsense Knowledge Integration

### Open Questions

1. **COMET vs ConceptNet vs LLM** - Which source?
   - COMET generates novel inferences
   - ConceptNet has curated knowledge
   - LLMs have implicit commonsense
   - Hybrid approach?

2. **Inference Selection** - Which inferences to store?
   - All ATOMIC relations? Too many
   - Only high-confidence? Miss useful ones
   - Context-dependent selection?

3. **Inference Quality** - Are inferences correct?
   - COMET can hallucinate
   - How to validate inferences?
   - Should inferences have lower confidence?

### Research Tasks

- [ ] Compare COMET vs ConceptNet vs LLM commonsense
- [ ] Develop inference selection heuristics
- [ ] Test inference quality metrics
- [ ] Investigate inference validation approaches

### Relevant Papers to Read

- "ATOMIC 2020: On Symbolic and Neural Commonsense Knowledge Graphs"
- "Time-aware COMET" (2024)
- ConceptNet 5 papers

---

## 9. Memory Layer Integration

### Open Questions

1. **Which Layer Gets What?** - Storage decisions
   - Presuppositions → Semantic memory?
   - Inferences → Working memory (temporary)?
   - Facts → Which layer based on verification?

2. **Layer Promotion** - How does knowledge harden?
   - What triggers promotion from working to semantic?
   - How do weights affect promotion?
   - Cross-layer linking strategies?

3. **Conflict Across Layers** - Resolution
   - Working memory contradicts semantic?
   - Which layer wins?
   - How to flag for user clarification?

### Research Tasks

- [ ] Map decomposition types to memory layers
- [ ] Design promotion rules based on weights
- [ ] Implement cross-layer conflict detection
- [ ] Test memory layer integration

---

## 10. Advanced Topics (Future)

### Potential Extensions

1. **Temporal Reasoning** - Timeline construction
   - Events linked by temporal relations
   - "Before Doug met Sarah" implies timeline
   - Temporal conflict detection

2. **Causal Reasoning** - Cause-effect chains
   - "X because Y" creates causal link
   - Causal inference from ATOMIC
   - Counterfactual reasoning

3. **Belief Revision** - Updating knowledge
   - New info contradicts old
   - Which to believe?
   - Graceful belief updates

4. **Multi-Agent Knowledge** - Shared graphs
   - Multiple agents contribute knowledge
   - Attribution and trust
   - Conflict resolution across agents

---

## Research Priority Matrix

| Topic | Impact | Effort | Priority |
|-------|--------|--------|----------|
| Decomposition Optimization | High | Medium | P0 |
| Retrieval Optimization | High | Medium | P0 |
| Evolution Framework | High | High | P1 |
| Evaluation Benchmarking | High | Medium | P1 |
| Graph Storage | Medium | High | P2 |
| Scale Testing | Medium | Medium | P2 |
| Commonsense Integration | Medium | Medium | P2 |
| Domain Adaptation | Medium | High | P3 |
| Memory Layer Integration | Medium | Medium | P3 |
| Advanced Topics | Low | Very High | P4 |

---

## How to Use This Document

1. **Before starting a new feature** - Check if research is needed
2. **After completing a phase** - Update with findings
3. **When stuck** - Review relevant papers
4. **During planning** - Prioritize research tasks

---

**End of Continued Research Document**
