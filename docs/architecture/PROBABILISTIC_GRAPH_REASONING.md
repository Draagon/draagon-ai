# Probabilistic Graph Reasoning Architecture

**Version:** 0.1.0
**Status:** Design Phase
**Last Updated:** 2024-12-31

## Overview

This document describes a novel architecture that combines semantic knowledge graphs with probabilistic reasoning and ReAct agent loops. The system expands ambiguous messages into branching interpretation graphs, explores multiple hypotheses in parallel, and uses reinforcement from successful interactions to strengthen the knowledge graph.

## Research Foundation

This design synthesizes several state-of-the-art approaches:

### Graph of Thoughts (GoT)
[Graph of Thoughts](https://arxiv.org/abs/2308.09687) models LLM reasoning as an arbitrary graph where thoughts are vertices and edges are dependencies. Unlike Tree of Thoughts, GoT allows:
- **Branching**: Parallel hypothesis generation
- **Merging**: Aggregation of convergent ideas
- **Non-linear reasoning**: Cross-pollination between hypotheses

### GraphRAG
[Microsoft's GraphRAG](https://www.microsoft.com/en-us/research/blog/graphrag-unlocking-llm-discovery-on-narrative-private-data/) uses knowledge graphs for retrieval, enabling:
- Multi-hop reasoning across relationships
- Community-based summarization
- Improved comprehensiveness over vector-only RAG

### Beam Search for Reasoning
[Self-Evaluation Guided Beam Search](https://arxiv.org/pdf/2305.00633) applies beam search to reasoning, maintaining multiple hypotheses and using self-evaluation to prune bad paths.

---

## Architecture Design

### Core Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         PROBABILISTIC GRAPH REASONING                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   Message ──► Phase 0/1 ──► Expansion ──► Context ──► Beam ──► React ──►    │
│       │      Extraction    (Branching)    Retrieval   Search    Loop   │    │
│       │                                                              │      │
│       │                                                              │      │
│       └──────────────────── Neo4j Graph Store ◄──────────────────────┘      │
│                              (Reinforcement)                                 │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 9-Step Pipeline (Original + Optimized)

| Step | Original Description | Optimized Design |
|------|---------------------|------------------|
| 1 | Convert message to semantic graph via Phase 0/1 | **Graph Extraction**: Use existing pipeline |
| 2 | Extract recent message graphs | **Recency Window**: Last N graphs from working memory |
| 3 | LLM expands message into branches with probabilities | **Probabilistic Expansion**: Generate interpretation DAG |
| 4 | Run expanded graph through Phase 0/1 for depth | **Recursive Extraction**: Deepen promising branches |
| 5 | Query graph for additional context | **GraphRAG Retrieval**: Multi-hop traversal from anchors |
| 6 | Pass full context + branches to ReAct agent | **Beam Search ReAct**: Parallel exploration of top-K interpretations |
| 7 | Update graphs based on results | **Reinforcement**: Strengthen successful paths |
| 8 | Store valuable results back to graph | **Selective Persistence**: Confidence-based storage |
| 9 | Flatten results to user response | **Response Synthesis**: Best-path explanation |

---

## Detailed Design

### Step 1: Graph Extraction (Phase 0/1)

Uses existing `IntegratedPipeline` → `GraphBuilder` to create initial semantic graph.

```python
# Existing code path
pipeline = IntegratedPipeline(llm=llm)
result = await pipeline.process(message)
builder = GraphBuilder()
message_graph = builder.build(result)
```

### Step 2: Recency Context

```python
@dataclass
class RecencyWindow:
    """Recent conversation context."""
    graphs: list[SemanticGraph]  # Last N message graphs
    window_size: int = 10
    time_decay: float = 0.9  # Older messages weighted less

    def get_weighted_nodes(self) -> list[tuple[GraphNode, float]]:
        """Get nodes weighted by recency."""
        weighted = []
        for i, graph in enumerate(reversed(self.graphs)):
            weight = self.time_decay ** i
            for node in graph.iter_nodes():
                weighted.append((node, weight))
        return weighted
```

### Step 3: Probabilistic Expansion (KEY INNOVATION)

This is where we branch into multiple interpretations:

```python
@dataclass
class InterpretationBranch:
    """A single interpretation of ambiguous input."""
    branch_id: str
    interpretation: str  # Natural language description
    probability: float   # 0.0 - 1.0
    graph: SemanticGraph  # Semantic representation
    reasoning: str       # Why this interpretation

@dataclass
class ExpansionResult:
    """Result of probabilistic expansion."""
    original_text: str
    original_graph: SemanticGraph
    branches: list[InterpretationBranch]
    ambiguity_type: str  # "referential", "semantic", "pragmatic"

class ProbabilisticExpander:
    """Expands ambiguous messages into interpretation branches."""

    async def expand(
        self,
        message: str,
        message_graph: SemanticGraph,
        recency_context: RecencyWindow,
    ) -> ExpansionResult:
        """
        Expand message into multiple interpretations with probabilities.

        Example:
            Input: "I got it!"
            Context: Previous message about dropped phone

            Output branches:
            - Branch 1 (80%): "Doug retrieved the phone"
              Graph: Doug -[retrieved]-> phone
            - Branch 2 (15%): "Doug understood something"
              Graph: Doug -[understood]-> [unknown concept]
            - Branch 3 (5%): "Doug received something"
              Graph: Doug -[received]-> [unknown object]
        """
        # Prepare context from recency window
        context_summary = self._summarize_context(recency_context)

        # LLM expansion prompt
        prompt = f"""
Analyze this message and generate possible interpretations with probabilities.

MESSAGE: {message}

RECENT CONTEXT:
{context_summary}

EXTRACTED SEMANTICS:
{self._graph_to_text(message_graph)}

Generate 2-5 interpretations. For each:
1. Describe the interpretation
2. Assign probability (must sum to 1.0)
3. Provide semantic structure (subject, predicate, object, modifiers)
4. Explain reasoning

<response>
<interpretations>
<interpretation probability="0.80">
<description>...</description>
<semantic_structure>
<subject>Doug</subject>
<predicate type="verb" synset="retrieve.v.01">got/retrieved</predicate>
<object>the phone</object>
</semantic_structure>
<reasoning>The recent context mentions dropping a phone...</reasoning>
</interpretation>
...
</interpretations>
</response>
"""
        # Parse LLM response and build graphs for each branch
        ...
```

### Step 4: Recursive Depth Extraction

For high-probability branches, extract more semantic depth:

```python
async def deepen_branch(
    branch: InterpretationBranch,
    depth: int = 2,
) -> InterpretationBranch:
    """Add semantic depth to a branch via Phase 0/1."""

    # Run interpretation text through full pipeline
    result = await pipeline.process(branch.interpretation)
    deeper_graph = builder.build(result)

    # Merge with existing branch graph
    branch.graph.merge(deeper_graph)

    return branch
```

### Step 5: GraphRAG Context Retrieval

Use the Neo4j store for multi-hop retrieval:

```python
async def retrieve_context(
    store: Neo4jGraphStore,
    instance_id: str,
    anchor_nodes: list[GraphNode],
    max_depth: int = 3,
    relation_types: list[str] | None = None,
) -> SemanticGraph:
    """
    Retrieve relevant subgraph from knowledge base.

    Uses GraphRAG-style multi-hop traversal:
    1. Start from anchor nodes (entities in current message)
    2. Traverse outward following relevant relationships
    3. Score and prune based on relevance to query
    """
    # Get subgraph around anchors
    subgraph = await store.traverse(
        instance_id=instance_id,
        start_node_ids=[n.node_id for n in anchor_nodes],
        max_depth=max_depth,
        relation_types=relation_types,
    )

    return subgraph
```

### Step 6: Beam Search ReAct (KEY INNOVATION)

Run multiple interpretation branches through ReAct in parallel:

```python
@dataclass
class BeamState:
    """State of a beam in beam search ReAct."""
    branch: InterpretationBranch
    score: float  # Running score
    trajectory: list[ReActStep]  # Steps taken
    context_graph: SemanticGraph  # Retrieved context
    status: str  # "active", "completed", "pruned"

class BeamSearchReAct:
    """
    Beam search over interpretation branches using ReAct.

    Instead of committing to one interpretation, explore top-K
    interpretations in parallel and let their success determine
    which was correct.
    """

    def __init__(
        self,
        beam_width: int = 3,  # How many beams to maintain
        max_steps: int = 5,   # Max ReAct iterations per beam
        min_probability: float = 0.1,  # Minimum branch probability to consider
    ):
        self.beam_width = beam_width
        self.max_steps = max_steps
        self.min_probability = min_probability

    async def run(
        self,
        expansion: ExpansionResult,
        context_graphs: dict[str, SemanticGraph],  # branch_id -> context
        agent: ReActAgent,
    ) -> BeamSearchResult:
        """
        Run beam search ReAct over interpretation branches.

        Strategy Options:

        OPTION A: Sequential Pruning (Lower Cost)
        - Run all beams for 1 step
        - Score and prune to top-K
        - Repeat until completion

        OPTION B: Parallel Completion (Higher Quality)
        - Run all beams to completion in parallel
        - Select best based on final outcomes

        OPTION C: Adaptive (Best of Both)
        - Start with parallel
        - If beams converge, merge
        - If one clearly wins early, prune others
        """

        # Initialize beams for branches above threshold
        beams = [
            BeamState(
                branch=b,
                score=b.probability,  # Initial score = prior probability
                trajectory=[],
                context_graph=context_graphs[b.branch_id],
                status="active"
            )
            for b in expansion.branches
            if b.probability >= self.min_probability
        ]

        # Run beam search
        for step in range(self.max_steps):
            # Execute one ReAct step for each active beam
            active_beams = [b for b in beams if b.status == "active"]

            # Parallel execution
            results = await asyncio.gather(*[
                self._execute_step(beam, agent)
                for beam in active_beams
            ])

            # Update scores based on step outcomes
            for beam, result in zip(active_beams, results):
                beam.trajectory.append(result)
                beam.score = self._update_score(beam, result)

                if result.is_terminal:
                    beam.status = "completed"

            # Prune to top-K beams
            beams = self._prune_beams(beams)

            # Early termination if all beams complete
            if all(b.status != "active" for b in beams):
                break

        return BeamSearchResult(
            beams=beams,
            best_beam=max(beams, key=lambda b: b.score),
            convergence=self._measure_convergence(beams),
        )

    def _update_score(self, beam: BeamState, step_result: ReActStep) -> float:
        """
        Update beam score based on step outcome.

        Factors:
        - Tool success: Did the action work?
        - Observation coherence: Does result match interpretation?
        - User feedback: (if available) explicit signal
        """
        score = beam.score

        if step_result.action_success:
            score *= 1.2  # Boost for successful actions
        else:
            score *= 0.7  # Penalty for failures

        if step_result.observation_matches_expectation:
            score *= 1.1  # Interpretation was correct

        return min(score, 1.0)  # Cap at 1.0
```

### Step 7: Graph Reinforcement

Strengthen successful paths, weaken failed ones:

```python
class GraphReinforcement:
    """Reinforce graph based on ReAct outcomes."""

    async def reinforce(
        self,
        store: Neo4jGraphStore,
        instance_id: str,
        beam_result: BeamSearchResult,
    ):
        """
        Update graph confidence based on what worked.

        For winning interpretation:
        - Boost confidence on edges that were traversed
        - Strengthen nodes that proved relevant

        For losing interpretations:
        - Don't punish hard (they might be right in other contexts)
        - But don't reinforce either
        """
        best = beam_result.best_beam

        # Boost edges used in winning interpretation
        for edge in best.branch.graph.iter_edges():
            await self._boost_edge(store, instance_id, edge.edge_id)

        # Boost retrieved context that was actually used
        for step in best.trajectory:
            if step.used_context_nodes:
                for node_id in step.used_context_nodes:
                    await self._boost_node(store, instance_id, node_id)
```

### Step 8: Selective Persistence

```python
class SelectivePersistence:
    """Decide what to store back to graph."""

    async def persist(
        self,
        store: Neo4jGraphStore,
        instance_id: str,
        beam_result: BeamSearchResult,
        threshold: float = 0.7,
    ):
        """
        Store valuable learnings back to graph.

        Persist if:
        - High confidence (winning beam score > threshold)
        - Novel information (not already in graph)
        - Factual (not speculative)
        """
        best = beam_result.best_beam

        if best.score < threshold:
            return  # Not confident enough

        # Extract new facts from trajectory
        new_facts = self._extract_facts(best.trajectory)

        for fact in new_facts:
            # Check if already in graph
            if not await self._fact_exists(store, instance_id, fact):
                # Add to graph
                fact_graph = self._fact_to_graph(fact)
                await store.save(fact_graph, instance_id)
```

### Step 9: Response Synthesis

```python
class ResponseSynthesizer:
    """Generate user response from beam results."""

    async def synthesize(
        self,
        beam_result: BeamSearchResult,
        original_message: str,
    ) -> str:
        """
        Generate response explaining what we understood and did.

        If high convergence (beams agreed):
            - Confident response
        If low convergence:
            - Ask clarifying question
        If one beam clearly won:
            - Respond based on that interpretation
        """
        if beam_result.convergence > 0.8:
            # Beams agreed, confident response
            return self._confident_response(beam_result.best_beam)
        elif beam_result.best_beam.score > 0.7:
            # One interpretation clearly won
            return self._explain_interpretation(beam_result.best_beam)
        else:
            # Ambiguous, ask for clarification
            return self._clarification_question(beam_result)
```

---

## Architecture Options for Parallel Beam Exploration

### Option A: Sequential Pruning (Recommended for Cost Efficiency)

```
Step 0:  [Branch1:0.80]  [Branch2:0.15]  [Branch3:0.05]
           ↓                 ↓               (pruned - below threshold)
Step 1:  [Branch1:0.85]  [Branch2:0.10]
           ↓               (pruned - low score)
Step 2:  [Branch1:0.90]  → WINNER
```

**Pros**: Lower LLM calls, faster convergence
**Cons**: May prune correct interpretation early

### Option B: Parallel Completion (Recommended for Quality)

```
Step 0-N:  [Branch1] ────────────────→ Score: 0.85
           [Branch2] ────────────────→ Score: 0.72
           [Branch3] ────────────────→ Score: 0.45
                                       ↓
                                   Best: Branch1
```

**Pros**: Each interpretation gets full exploration
**Cons**: Higher cost (3x+ LLM calls)

### Option C: Adaptive Hybrid (Recommended Overall)

```
Step 0:   [Branch1:0.80]  [Branch2:0.15]  [Branch3:0.05]
            ↓                 ↓               ↓
Step 1:   [Branch1:0.88]  [Branch2:0.12]  [Branch3:0.03]
            ↓                 ↓               (auto-prune)
          (both continue - close scores)
            ↓                 ↓
Step 2:   [Branch1:0.91]  [Branch2:0.08]
            ↓               (auto-prune - gap too large)
Step 3:   [Branch1:0.94]  → WINNER
```

**Decision Logic**:
- If gap between top-2 beams > 0.3: prune lower
- If beams converge to same conclusion: merge
- If stuck (scores plateau): expand beam width

---

## Example Walkthrough

### Input

```
Previous context: "I dropped my phone in the trash"
Current message: "I got it!"
```

### Step 1: Phase 0/1 Extraction

```
Message Graph:
  (Doug) -[subject_of]-> (got [get.v.01]) -[has_object]-> (it [???])

Ambiguity detected: "it" is unresolved anaphora
```

### Step 2: Recency Context

```
Recent Graph (1 message ago):
  (Doug) -[agent_of]-> (dropped [drop.v.01]) -[patient]-> (phone)
  (phone) -[destination]-> (trash)
```

### Step 3: Probabilistic Expansion

```
Branch 1 (80%): "Doug retrieved the phone from the trash"
  Graph: (Doug) -[agent]-> (retrieve.v.01) -[patient]-> (phone)
         (phone) -[source]-> (trash)
  Reasoning: "it" refers to "phone" from context; "got" = retrieved

Branch 2 (12%): "Doug understood something"
  Graph: (Doug) -[experiencer]-> (understand.v.01) -[theme]-> (concept)
  Reasoning: "got it" can mean "understood"

Branch 3 (8%): "Doug received the phone (from someone)"
  Graph: (Doug) -[recipient]-> (receive.v.01) -[theme]-> (phone)
  Reasoning: Alternative meaning of "got"
```

### Step 4: Recursive Depth

```
Branch 1 expanded:
  (Doug) -[agent]-> (retrieve.v.01) -[patient]-> (phone)
  (phone) -[source]-> (trash)
  (retrieve.v.01) -[xReact]-> (relief, satisfaction)
  (retrieve.v.01) -[xIntent]-> (recover_possession)
```

### Step 5: Context Retrieval

```
Query: Nodes related to Doug, phone, trash

Retrieved:
  (Doug) -[owns]-> (phone [iPhone_14])
  (phone) -[has_property]-> (valuable)
  (trash) -[located_in]-> (kitchen)
```

### Step 6: Beam Search ReAct

```
Beam 1 (Branch 1): Interpretation: Retrieved phone
  Step 1: No action needed (statement of fact)
  Step 2: Update context
  Score: 0.92 (high confidence, matches context)

Beam 2 (Branch 2): Interpretation: Understood something
  Step 1: Search for what was being discussed
  Step 2: No match to "understanding" context
  Score: 0.15 (doesn't fit)

Winner: Beam 1
```

### Step 7: Reinforcement

```
Boost edges:
  - Doug -[owns]-> phone (+0.05 confidence)
  - "it" → phone anaphora resolution (+0.1)
```

### Step 8: Persistence

```
Store:
  (Doug) -[retrieved]-> (phone) -[from]-> (trash)
  valid_from: 2024-12-31T12:00:00
```

### Step 9: Response

```
"Great! Good thing you got your phone out of the trash."
```

---

## Implementation Phases

### Phase 1: Minimal Viable Loop (MVP)

Build the basic loop without beam search:

1. Phase 0/1 → Graph
2. Store in Neo4j
3. Single interpretation (no branching)
4. Basic ReAct
5. Store results

**Files to create**:
- `src/draagon_ai/cognition/reasoning/loop.py` - Main processing loop
- `src/draagon_ai/cognition/reasoning/context.py` - Context retrieval

### Phase 2: Probabilistic Expansion

Add branching interpretations:

1. `ProbabilisticExpander` class
2. LLM prompt for generating branches
3. Graph builder for branch graphs

**Files to create**:
- `src/draagon_ai/cognition/reasoning/expander.py`
- `src/draagon_ai/cognition/reasoning/prompts/expansion.py`

### Phase 3: Beam Search ReAct

Add parallel beam exploration:

1. `BeamSearchReAct` class
2. Scoring and pruning logic
3. Adaptive strategy selection

**Files to create**:
- `src/draagon_ai/cognition/reasoning/beam_search.py`

### Phase 4: Reinforcement & Persistence

Add learning loop:

1. `GraphReinforcement` class
2. `SelectivePersistence` class
3. Confidence update logic

**Files to create**:
- `src/draagon_ai/cognition/reasoning/reinforcement.py`

### Phase 5: Comparison Framework

Build comparison with current draagon-ai:

1. Same input messages
2. Run through both systems
3. Compare outputs
4. Measure improvement over time

---

## Metrics for Comparison

| Metric | Description |
|--------|-------------|
| **Comprehension Accuracy** | Did system correctly understand ambiguous messages? |
| **Context Utilization** | How much relevant context was retrieved? |
| **Response Quality** | User satisfaction with responses |
| **Graph Growth** | Knowledge accumulation over time |
| **Retrieval Precision** | Retrieved context that was actually used |
| **Inference Depth** | Hops of reasoning to reach conclusion |

---

## Cost Considerations

| Approach | LLM Calls per Message | Notes |
|----------|----------------------|-------|
| Current (no graph) | 1-3 | Basic ReAct |
| Option A (Sequential) | 3-10 | Prunes early, lower cost |
| Option B (Parallel) | 10-30 | Full exploration, higher quality |
| Option C (Adaptive) | 5-15 | Best balance |

**Optimization strategies**:
- Use smaller models (Haiku) for expansion, larger (Opus) for final response
- Cache common expansions
- Batch graph queries
- Early termination when confident

---

## Sources

- [Graph of Thoughts: Solving Elaborate Problems with Large Language Models](https://arxiv.org/abs/2308.09687)
- [Microsoft GraphRAG](https://www.microsoft.com/en-us/research/blog/graphrag-unlocking-llm-discovery-on-narrative-private-data/)
- [Self-Evaluation Guided Beam Search for Reasoning](https://arxiv.org/pdf/2305.00633)
- [From Local to Global: A Graph RAG Approach](https://arxiv.org/abs/2404.16130)
- [Neo4j Multi-Hop Reasoning with Knowledge Graphs and LLMs](https://neo4j.com/blog/genai/knowledge-graph-llm-multi-hop-reasoning/)
- [Tree of Thoughts Prompting](https://www.promptingguide.ai/techniques/tot)
- [Demystifying Chains, Trees, and Graphs of Thoughts](https://arxiv.org/html/2401.14295v5)
