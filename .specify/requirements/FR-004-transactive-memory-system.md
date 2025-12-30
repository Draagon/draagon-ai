# FR-004: Transactive Memory System

**Feature:** "Who knows what" expertise tracking for agent routing
**Status:** Planned
**Priority:** P1 (Enhancement)
**Complexity:** Medium
**Phase:** Cognitive Swarm Phase 4

---

## Overview

Implement a transactive memory system that tracks which agents are experts at which topics, enabling intelligent query routing, reduced redundant work, and better task decomposition. Inspired by human team dynamics where team members know "who knows what" without needing to ask everyone.

### Research Foundation

- **Wegner (1987)**: Transactive memory in human groups enables efficient knowledge access
- **Memory in LLM-MAS Survey (MongoDB)**: "Who knows what" critical for multi-agent coordination
- **MultiAgentBench**: Expertise routing reduces strategic silence failures
- **Anthropic Multi-Agent Research**: Detailed task descriptions prevent duplication (transactive memory provides this)

---

## Functional Requirements

### FR-004.1: Expertise Entry Tracking

**Requirement:** Track agent expertise per topic with success/failure counts

**Acceptance Criteria:**
- Each expertise entry includes:
  - Topic (string)
  - Confidence (0-1)
  - Success count
  - Failure count
  - Last updated timestamp
- `success_rate` property: success_count / (success_count + failure_count)
- Default confidence: 0.5 (neutral prior)

**Test Approach:**
- Create expertise entry for agent_weather on topic "weather"
- Initialize with confidence=0.5, success=0, failure=0
- Update with success
- Assert confidence increases, success_count=1
- Assert success_rate=1.0

**Constitution Check:** ✅ Research-grounded (Wegner 1987)

---

### FR-004.2: Expertise Updates from Outcomes

**Requirement:** Update expertise based on agent task success/failure

**Acceptance Criteria:**
- `update_expertise(agent_id, topic, success: bool)`
- On success:
  - Increment success_count
  - Increase confidence by 0.1 (capped at 1.0)
- On failure:
  - Increment failure_count
  - Decrease confidence by 0.15 (floored at 0.0)
- Update last_updated timestamp

**Test Approach:**
- Agent A starts with confidence 0.5 on "coding"
- 3 successes: confidence → 0.6, 0.7, 0.8
- 1 failure: confidence → 0.65
- Assert success_rate = 3/4 = 0.75

**Constitution Check:** ✅ Cognitive authenticity (genuine learning from outcomes)

---

### FR-004.3: Query Routing by Expertise

**Requirement:** Route queries to agents ranked by expertise

**Acceptance Criteria:**
- `route_query(query: str, available_agents: list[AgentSpec]) -> list[AgentSpec]`
- Extract topics from query (LLM-based or keyword extraction)
- Score each agent by topic match confidence
- Return agents sorted by score (highest first)
- If no expertise data, return agents unmodified

**Test Approach:**
- Query: "What's the weather for my meeting tomorrow?"
- Topics extracted: ["weather", "meeting"]
- agent_weather has expertise 0.9 in "weather"
- agent_calendar has expertise 0.8 in "meeting"
- agent_generic has no expertise
- Assert ranking: [agent_weather, agent_calendar, agent_generic]

**Constitution Check:** ✅ LLM-First (topic extraction via LLM or semantic, NOT hardcoded keywords)

---

### FR-004.4: Topic Hierarchy for Generalization

**Requirement:** Support topic hierarchies for expertise generalization

**Acceptance Criteria:**
- `_topic_hierarchy: dict[str, list[str]]` maps topic → parent topics
- Example: "python" → ["programming", "software"]
- When updating expertise on "python", also update parent topics (with decay)
- When routing query about "typescript", check expertise in "programming" if no direct match

**Test Approach:**
- Agent A has expertise 0.9 in "python"
- Hierarchy: "python" → ["programming"]
- Query about "typescript"
- Agent A scores 0.63 (0.9 * 0.7 for parent match)
- Assert Agent A ranked above agents with no programming expertise

**Constitution Check:** ✅ Cognitive authenticity (semantic generalization)

---

### FR-004.5: Get Experts for Topic

**Requirement:** Query which agents are experts in a specific topic

**Acceptance Criteria:**
- `get_experts(topic: str, min_confidence: float = 0.6) -> list[tuple[str, float]]`
- Returns list of (agent_id, confidence) pairs
- Filtered by min_confidence threshold
- Sorted by confidence descending

**Test Approach:**
- 3 agents with "weather" expertise: A (0.9), B (0.7), C (0.4)
- `get_experts("weather", min_confidence=0.6)`
- Assert returns [(A, 0.9), (B, 0.7)]
- C excluded due to low confidence

---

### FR-004.6: Natural Language Expert Query

**Requirement:** Answer "who knows about X?" questions

**Acceptance Criteria:**
- `who_knows_about(topic: str) -> str`
- Returns natural language string
- Lists top 3 experts with confidence
- If no experts: "No agents have demonstrated expertise in '{topic}' yet."

**Test Approach:**
- Query: `who_knows_about("weather")`
- agent_weather (0.95), agent_forecast (0.82), agent_climate (0.75)
- Assert response: "For 'weather': agent_weather (confidence: 95%), agent_forecast (confidence: 82%), agent_climate (confidence: 75%)"

**Constitution Check:** ✅ Test outcomes (verifies useful output, not specific format)

---

### FR-004.7: Topic Extraction from Query

**Requirement:** Extract relevant topics from natural language query

**Acceptance Criteria:**
- `_extract_topics(query: str) -> list[str]`
- Phase 1: Simple keyword extraction (filter stopwords, min length 4)
- Phase 2: LLM-based topic extraction (structured output)
- Returns list of relevant topics

**Test Approach:**
- Query: "What's the weather like for my meeting tomorrow?"
- Simple: ["weather", "meeting", "tomorrow"]
- LLM: ["weather", "scheduling", "forecasting"]
- Assert topics are non-empty and relevant

**Constitution Check:** ✅ LLM-First (Phase 2 uses LLM, Phase 1 is placeholder)

---

### FR-004.8: Expertise Score Computation

**Requirement:** Compute agent's expertise score for given topics

**Acceptance Criteria:**
- `_compute_expertise_score(agent_id: str, topics: list[str]) -> float`
- Exact topic match: full confidence
- Partial match (substring): 0.7x confidence
- No match: 0.5 (neutral prior)
- Average across all topics

**Test Approach:**
- Agent A: expertise 0.9 in "weather_api"
- Topics: ["weather", "api"]
- Partial matches: 0.9 * 0.7 = 0.63 for each
- Average: 0.63
- Assert score = 0.63

---

## Data Structures

```python
@dataclass
class ExpertiseEntry:
    topic: str
    confidence: float  # 0-1
    success_count: int = 0
    failure_count: int = 0
    last_updated: datetime = field(default_factory=datetime.now)

    @property
    def success_rate(self) -> float:
        total = self.success_count + self.failure_count
        if total == 0:
            return 0.5  # Prior
        return self.success_count / total

class TransactiveMemory:
    def __init__(self, embedding_provider: Any = None)

    async def route_query(
        self,
        query: str,
        available_agents: list[AgentSpec],
    ) -> list[AgentSpec]

    async def update_expertise(
        self,
        agent_id: str,
        topic: str,
        success: bool,
    ) -> None

    async def get_experts(
        self,
        topic: str,
        min_confidence: float = 0.6,
    ) -> list[tuple[str, float]]

    async def who_knows_about(self, topic: str) -> str

    async def _extract_topics(self, query: str) -> list[str]

    async def _compute_expertise_score(
        self,
        agent_id: str,
        topics: list[str],
    ) -> float

    def get_expertise_summary(self) -> dict[str, dict[str, float]]
```

---

## Integration Points

### Upstream Dependencies
- `AgentSpec` (orchestration)
- LLMProvider (for topic extraction in Phase 2)
- Embedding provider (optional, for semantic topic matching)

### Downstream Consumers
- `ParallelCognitiveOrchestrator` (calls `route_query()`)
- `MetacognitiveReflectionService` (updates expertise from reflection)
- Agent introspection ("What am I good at?")

---

## Success Criteria

### Cognitive Metrics
- **Routing Accuracy**: 85%+ queries routed to highest-expertise agent
- **Expertise Learning Rate**: Agent confidence converges to true ability within 10 tasks
- **Generalization**: Parent topic matches work for 70%+ of related queries

### Performance Metrics
- **Routing Latency**: <50ms to route query
- **Update Latency**: <10ms to update expertise
- **Storage**: O(agents × topics) memory footprint

### Benchmark Targets
- **MultiAgentBench Research Task**: Target 90% (vs 84.13% SOTA) via better routing
- **Novel Transactive Memory Benchmark**: Create and pass at 85%+

---

## Testing Strategy

### Unit Tests
- `test_expertise_entry()`: Success/failure tracking
- `test_update_expertise()`: Confidence adjustments
- `test_route_query()`: Agent ranking
- `test_topic_hierarchy()`: Generalization
- `test_get_experts()`: Filtering and sorting
- `test_who_knows_about()`: Natural language output

### Integration Tests
- `test_routing_with_orchestrator()`: Integration with ParallelCognitiveOrchestrator
- `test_expertise_from_reflection()`: MetacognitiveReflection updates expertise

### Cognitive Tests (Novel Benchmark)
- `test_expertise_routing_benchmark()`: Custom scenarios (see FR spec Appendix)
- `test_expertise_learning_benchmark()`: Success rate convergence
- `test_cross_domain_generalization()`: Python expert helps with TypeScript

---

## Implementation Tasks

1. **Core Data Structures** (1 day)
   - `ExpertiseEntry` dataclass
   - `TransactiveMemory` class skeleton
   - Internal storage: `_expertise: dict[str, dict[str, ExpertiseEntry]]`

2. **Expertise Tracking** (2 days)
   - `update_expertise()` with confidence updates
   - Topic hierarchy support
   - Parent topic propagation

3. **Query Routing** (2 days)
   - `_extract_topics()` (simple keyword version)
   - `_compute_expertise_score()`
   - `route_query()` with ranking

4. **Expert Queries** (1 day)
   - `get_experts()`
   - `who_knows_about()` natural language
   - `get_expertise_summary()`

5. **LLM Integration (Phase 2)** (2 days)
   - LLM-based topic extraction
   - Semantic topic matching via embeddings

6. **Testing** (2 days)
   - Unit tests for all methods
   - Integration tests with orchestrator
   - Novel benchmark creation

**Total Estimate:** 10 days

---

## Novel Benchmark: Transactive Memory Scenarios

```python
TRANSACTIVE_MEMORY_SCENARIOS = [
    {
        "name": "Expertise Routing",
        "setup": [
            ("agent_weather", "weather", "success"),
            ("agent_weather", "weather", "success"),
            ("agent_calendar", "scheduling", "success"),
        ],
        "query": "What's the weather for my meeting tomorrow?",
        "expected_route": ["agent_weather", "agent_calendar"],
        "scoring": "Precision@2 = 1.0 if both in top 2",
    },
    {
        "name": "Expertise Learning",
        "setup": [
            ("agent_a", "coding", "success"),
            ("agent_a", "coding", "success"),
            ("agent_a", "coding", "failure"),
            ("agent_b", "coding", "success"),
            ("agent_b", "coding", "success"),
        ],
        "query": "Help me debug this code",
        "expected_route": ["agent_b", "agent_a"],  # B has better success rate
        "scoring": "MRR = 1.0 if agent_b is rank 1",
    },
    {
        "name": "Cross-Domain Generalization",
        "setup": [
            ("agent_x", "python", "success"),
            ("agent_x", "javascript", "success"),
        ],
        "query": "Help me with TypeScript",
        "expected_route": ["agent_x"],  # Should generalize to programming
        "scoring": "Recall@3 = 1.0 if agent_x in top 3",
    },
]
```

**Scoring:** Average Precision@2, MRR (Mean Reciprocal Rank), Recall@3 across all scenarios. Target: 85%+

---

## Constitution Compliance

- ✅ **LLM-First**: Topic extraction via LLM (Phase 2), semantic matching
- ✅ **Research-Grounded**: Wegner 1987 transactive memory theory
- ✅ **Cognitive Authenticity**: Genuine expertise tracking, not hardcoded rules
- ✅ **Async-First**: All methods are async
- ✅ **Protocol-Based**: Uses AgentSpec protocol
- ✅ **Test Outcomes**: Benchmark validates correct routing, not specific algorithms

---

**Document Status:** Draft
**Last Updated:** 2025-12-30
**Author:** draagon-ai team
