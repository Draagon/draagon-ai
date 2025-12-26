"""Integration tests for Phase C.1 components.

Tests the complete flow:
1. Multi-agent orchestration
2. Learning channel communication
3. Temporal cognitive graph storage
4. Scope-based access control
"""

import pytest
import asyncio
from datetime import datetime

from draagon_ai.memory import (
    TemporalNode,
    TemporalEdge,
    NodeType,
    EdgeType,
    MemoryLayer,
    HierarchicalScope,
    ScopeType,
    Permission,
    ScopeRegistry,
    TemporalCognitiveGraph,
    EmbeddingProvider,
    reset_scope_registry,
)
from draagon_ai.orchestration import (
    MultiAgentOrchestrator,
    OrchestrationMode,
    AgentSpec,
    AgentRole,
    TaskContext,
    AgentResult,
    Learning,
    LearningType,
    LearningScope,
    InMemoryLearningChannel,
    reset_learning_channel,
)


class MockEmbeddingProvider(EmbeddingProvider):
    """Mock embedding provider for testing.

    Uses word-based embeddings that preserve semantic similarity for testing.
    Similar words produce similar embeddings. Uses hashlib for determinism.
    """

    def __init__(self, dimension: int = 768):
        self.dimension = dimension
        self.call_count = 0

    def _stable_hash(self, word: str) -> int:
        """Generate a stable hash independent of PYTHONHASHSEED."""
        import hashlib
        return int(hashlib.md5(word.encode()).hexdigest(), 16)

    async def embed(self, text: str) -> list[float]:
        """Generate a deterministic embedding based on text content."""
        self.call_count += 1

        # Create a simple word-based embedding
        embedding = [0.0] * self.dimension

        # Use word positions to create embedding
        words = text.lower().split()
        for i, word in enumerate(words):
            # Hash each word to positions in the embedding (using stable hash)
            h = self._stable_hash(word)
            for j in range(10):  # Spread across 10 positions
                idx = abs(h + j) % self.dimension
                embedding[idx] += 0.1 * (1 / (i + 1))  # Decay by position

        # Normalize
        magnitude = sum(x * x for x in embedding) ** 0.5
        if magnitude > 0:
            embedding = [x / magnitude for x in embedding]

        return embedding


@pytest.fixture(autouse=True)
def reset_globals():
    """Reset global state before each test."""
    reset_learning_channel()
    reset_scope_registry()
    yield
    reset_learning_channel()
    reset_scope_registry()


@pytest.fixture
def embedding_provider():
    """Fixture for mock embedding provider."""
    return MockEmbeddingProvider()


@pytest.fixture
def scope_registry():
    """Fixture for scope registry with household scopes."""
    registry = ScopeRegistry()

    # Create hierarchy using create_scope
    household = registry.create_scope(
        scope_type=ScopeType.CONTEXT,
        name="Household 1",
        scope_id="context:household_1",
        parent_scope_id="world:global",
    )
    agent_roxy = registry.create_scope(
        scope_type=ScopeType.AGENT,
        name="Roxy Agent",
        scope_id="agent:roxy",
        parent_scope_id="context:household_1",
    )
    agent_researcher = registry.create_scope(
        scope_type=ScopeType.AGENT,
        name="Researcher Agent",
        scope_id="agent:researcher",
        parent_scope_id="context:household_1",
    )
    user_doug = registry.create_scope(
        scope_type=ScopeType.USER,
        name="Doug",
        scope_id="user:doug",
        parent_scope_id="agent:roxy",
    )

    return registry


@pytest.fixture
def cognitive_graph(embedding_provider):
    """Fixture for temporal cognitive graph."""
    return TemporalCognitiveGraph(embedding_provider=embedding_provider)


@pytest.fixture
def learning_channel():
    """Fixture for in-memory learning channel."""
    return InMemoryLearningChannel()


@pytest.fixture
def orchestrator(learning_channel):
    """Fixture for multi-agent orchestrator with learning channel."""
    return MultiAgentOrchestrator(learning_channel=learning_channel)


class TestEndToEndC1Flow:
    """Test complete C.1 integration scenarios."""

    @pytest.mark.anyio
    async def test_multi_agent_with_shared_learning(
        self, orchestrator, learning_channel, cognitive_graph
    ):
        """Test multi-agent execution with cross-agent learning sharing."""
        learnings_received = []

        # Subscribe to learnings from all agents
        async def learning_handler(learning: Learning):
            learnings_received.append(learning)

        await learning_channel.subscribe(
            "observer",
            learning_handler,
            context_id=None,  # No context filter to receive all
        )

        # Define agent executor that uses cognitive graph
        async def agent_executor(agent: AgentSpec, context: TaskContext) -> AgentResult:
            # Simulate agent work and store in graph
            if agent.role == AgentRole.RESEARCHER:
                # Researcher discovers a fact
                node = await cognitive_graph.add_node(
                    content=f"Research finding from {agent.agent_id}",
                    node_type=NodeType.FACT,
                    scope_id="context:household_1",
                    metadata={"source_agent_id": agent.agent_id},
                )

                learning = Learning(
                    learning_type=LearningType.FACT,
                    content="Discovered important research finding",
                    source_agent_id=agent.agent_id,
                    scope=LearningScope.GLOBAL,  # GLOBAL scope reaches all
                )
                return AgentResult(
                    agent_id=agent.agent_id,
                    success=True,
                    output={"finding": node.content},
                    learnings=[learning],
                )

            elif agent.role == AgentRole.PRIMARY:
                # Primary agent uses the research
                results = await cognitive_graph.search(
                    query="research finding",
                    scope_ids=["context:household_1"],
                    node_types=[NodeType.FACT],
                    limit=1,
                )

                found = results[0].node.content if results else "No findings"
                return AgentResult(
                    agent_id=agent.agent_id,
                    success=True,
                    output={"used_finding": found},
                )

            return AgentResult(agent_id=agent.agent_id, success=True, output={})

        # Define agent pipeline
        agents = [
            AgentSpec(
                agent_id="researcher",
                name="Research Agent",
                role=AgentRole.RESEARCHER,
            ),
            AgentSpec(
                agent_id="roxy",
                name="Primary Assistant",
                role=AgentRole.PRIMARY,
            ),
        ]

        context = TaskContext(
            query="Research and summarize findings",
            user_id="doug",
        )

        # Execute pipeline
        result = await orchestrator.orchestrate(
            agents=agents,
            context=context,
            executor=agent_executor,
            mode=OrchestrationMode.SEQUENTIAL,
        )

        # Verify orchestration succeeded
        assert result.success is True
        assert len(result.agent_results) == 2

        # Verify researcher's finding was stored in graph
        graph_results = await cognitive_graph.search(
            query="research",
            scope_ids=["context:household_1"],
            limit=10,
        )
        assert len(graph_results) >= 1
        assert "researcher" in graph_results[0].node.content.lower()

        # Verify primary agent found the research
        primary_result = result.agent_results[1]
        assert "research" in primary_result.output["used_finding"].lower()

        # Verify learnings were collected in result
        assert len(result.learnings) >= 1
        assert result.learnings[0].learning_type == LearningType.FACT

    @pytest.mark.anyio
    async def test_scope_based_memory_isolation(
        self, cognitive_graph, scope_registry
    ):
        """Test that scopes properly isolate memories between agents."""
        # Agent Roxy stores private fact
        await cognitive_graph.add_node(
            content="Roxy's private configuration",
            node_type=NodeType.FACT,
            scope_id="agent:roxy",
            metadata={"source_agent_id": "roxy"},
        )

        # Agent Researcher stores private fact
        await cognitive_graph.add_node(
            content="Researcher's private configuration",
            node_type=NodeType.FACT,
            scope_id="agent:researcher",
            metadata={"source_agent_id": "researcher"},
        )

        # Store household-level shared fact
        await cognitive_graph.add_node(
            content="Household shared information",
            node_type=NodeType.FACT,
            scope_id="context:household_1",
            metadata={"source_agent_id": "roxy"},
        )

        # Roxy searches its own scope - should see its private + shared
        roxy_results = await cognitive_graph.search(
            query="configuration information",
            scope_ids=["agent:roxy", "context:household_1"],
            limit=10,
        )
        roxy_contents = [r.node.content for r in roxy_results]
        assert any("Roxy's private" in c for c in roxy_contents)
        assert any("Household shared" in c for c in roxy_contents)
        # Should NOT see researcher's private data
        assert not any("Researcher's private" in c for c in roxy_contents)

        # Researcher searches its scope - should see its private + shared
        researcher_results = await cognitive_graph.search(
            query="configuration information",
            scope_ids=["agent:researcher", "context:household_1"],
            limit=10,
        )
        researcher_contents = [r.node.content for r in researcher_results]
        assert any("Researcher's private" in c for c in researcher_contents)
        assert any("Household shared" in c for c in researcher_contents)
        # Should NOT see Roxy's private data
        assert not any("Roxy's private" in c for c in researcher_contents)

    @pytest.mark.anyio
    async def test_temporal_supersession_in_pipeline(
        self, orchestrator, cognitive_graph
    ):
        """Test that agents can supersede outdated information."""
        # First agent stores initial fact
        async def first_pass_executor(agent: AgentSpec, context: TaskContext) -> AgentResult:
            if agent.agent_id == "initial":
                node = await cognitive_graph.add_node(
                    content="The WiFi password is oldpass123",
                    node_type=NodeType.FACT,
                    scope_id="context:household",
                    entities=["wifi", "password"],
                    metadata={"source_agent_id": agent.agent_id},
                )
                context.agent_outputs[agent.agent_id] = {"node_id": node.node_id}
                return AgentResult(
                    agent_id=agent.agent_id,
                    success=True,
                    output={"node_id": node.node_id},
                )
            return AgentResult(agent_id=agent.agent_id, success=True, output={})

        agents1 = [AgentSpec(agent_id="initial", role=AgentRole.PRIMARY)]
        context1 = TaskContext(query="Store wifi password", user_id="doug")

        result1 = await orchestrator.orchestrate(
            agents=agents1,
            context=context1,
            executor=first_pass_executor,
        )

        assert result1.success is True
        old_node_id = result1.agent_results[0].output["node_id"]

        # Second agent supersedes the fact
        async def second_pass_executor(agent: AgentSpec, context: TaskContext) -> AgentResult:
            if agent.agent_id == "updater":
                new_node = await cognitive_graph.supersede_node(
                    old_node_id=old_node_id,
                    new_content="The WiFi password is newpass456",
                )
                return AgentResult(
                    agent_id=agent.agent_id,
                    success=True,
                    output={"new_node_id": new_node.node_id},
                )
            return AgentResult(agent_id=agent.agent_id, success=True, output={})

        agents2 = [AgentSpec(agent_id="updater", role=AgentRole.PRIMARY)]
        context2 = TaskContext(query="Update wifi password", user_id="doug")

        await orchestrator.orchestrate(
            agents=agents2,
            context=context2,
            executor=second_pass_executor,
        )

        # Verify old node is superseded
        old_node = await cognitive_graph.get_node(old_node_id)
        assert old_node is not None
        assert old_node.is_current is False
        assert old_node.superseded_by is not None

        # Search should return only current version (include_superseded=False is default)
        results = await cognitive_graph.search(
            query="wifi password",
            scope_ids=["context:household"],
            include_superseded=False,
        )
        assert len(results) == 1
        assert "newpass456" in results[0].node.content

    @pytest.mark.anyio
    async def test_multi_layer_memory_in_conversation(
        self, cognitive_graph
    ):
        """Test storing and retrieving memories across different layers."""
        # Store working memory (current context)
        context_node = await cognitive_graph.add_node(
            content="Current conversation about vacation planning",
            node_type=NodeType.CONTEXT,
            scope_id="session:abc123",
        )

        # Store episodic memory (conversation episode)
        episode_node = await cognitive_graph.add_node(
            content="User asked about Paris travel recommendations",
            node_type=NodeType.EPISODE,
            scope_id="user:doug",
            entities=["paris", "travel", "vacation"],
            metadata={"conversation_id": "conv_123"},
        )

        # Store semantic memory (learned fact)
        fact_node = await cognitive_graph.add_node(
            content="Doug's preferred travel season is spring",
            node_type=NodeType.FACT,
            scope_id="user:doug",
            entities=["doug", "travel", "spring"],
            metadata={"source_agent_id": "roxy"},
        )

        # Store skill (procedural knowledge)
        skill_node = await cognitive_graph.add_node(
            content="To book Paris flights, use Air France or Delta",
            node_type=NodeType.SKILL,
            scope_id="context:household",
            entities=["paris", "flights", "booking"],
            metadata={"source_agent_id": "roxy"},
        )

        # Verify layer assignments
        assert context_node.layer == MemoryLayer.WORKING
        assert episode_node.layer == MemoryLayer.EPISODIC
        assert fact_node.layer == MemoryLayer.SEMANTIC
        assert skill_node.layer == MemoryLayer.METACOGNITIVE

        # Search across layers
        all_paris = await cognitive_graph.search(
            query="paris travel",
            scope_ids=["session:abc123", "user:doug", "context:household"],
            limit=10,
        )
        assert len(all_paris) >= 2  # At least episode and skill

        # Search only semantic layer
        semantic_only = await cognitive_graph.search(
            query="travel preferences",
            scope_ids=["user:doug", "context:household"],
            layers=[MemoryLayer.SEMANTIC],
            limit=10,
        )
        for result in semantic_only:
            assert result.node.layer == MemoryLayer.SEMANTIC

    @pytest.mark.anyio
    async def test_graph_traversal_for_related_knowledge(
        self, cognitive_graph
    ):
        """Test graph traversal to find related knowledge."""
        # Create entity nodes
        doug_entity = await cognitive_graph.add_node(
            content="Doug - household member",
            node_type=NodeType.ENTITY,
            scope_id="context:household",
            entities=["doug"],
        )

        paris_entity = await cognitive_graph.add_node(
            content="Paris - destination city",
            node_type=NodeType.ENTITY,
            scope_id="world",
            entities=["paris"],
        )

        # Create fact node
        trip_fact = await cognitive_graph.add_node(
            content="Doug wants to visit Paris in April",
            node_type=NodeType.FACT,
            scope_id="user:doug",
            entities=["doug", "paris", "april"],
            metadata={"source_agent_id": "roxy"},
        )

        # Create relationships using add_edge with separate parameters
        await cognitive_graph.add_edge(
            source_id=trip_fact.node_id,
            target_id=doug_entity.node_id,
            edge_type=EdgeType.RELATED_TO,
            label="about_person",
        )

        await cognitive_graph.add_edge(
            source_id=trip_fact.node_id,
            target_id=paris_entity.node_id,
            edge_type=EdgeType.RELATED_TO,
            label="about_destination",
        )

        # Traverse from trip fact
        traversal = await cognitive_graph.traverse(
            start_node_id=trip_fact.node_id,
            max_hops=2,
        )

        assert len(traversal.nodes) >= 3  # trip_fact + doug + paris
        assert len(traversal.edges) >= 2  # two relationships

        # Verify we found both related entities
        node_contents = [n.content for n in traversal.nodes]
        assert any("Doug" in c for c in node_contents)
        assert any("Paris" in c for c in node_contents)

    @pytest.mark.anyio
    async def test_conditional_agent_execution(
        self, orchestrator
    ):
        """Test conditional agent execution based on previous results."""
        async def conditional_executor(agent: AgentSpec, context: TaskContext) -> AgentResult:
            if agent.agent_id == "classifier":
                # Classifier determines if query needs research
                needs_research = "complex" in context.query.lower()
                return AgentResult(
                    agent_id=agent.agent_id,
                    success=True,
                    output={"needs_research": needs_research},
                )

            elif agent.agent_id == "researcher":
                # This should only run if classifier said needs_research
                return AgentResult(
                    agent_id=agent.agent_id,
                    success=True,
                    output={"research_done": True},
                )

            elif agent.agent_id == "responder":
                # Check if research was done
                research_done = context.agent_outputs.get("researcher", {}).get("research_done", False)
                return AgentResult(
                    agent_id=agent.agent_id,
                    success=True,
                    output={"used_research": research_done},
                )

            return AgentResult(agent_id=agent.agent_id, success=True, output={})

        agents = [
            AgentSpec(agent_id="classifier", role=AgentRole.PRIMARY),
            AgentSpec(
                agent_id="researcher",
                role=AgentRole.RESEARCHER,
                # Use attribute access syntax (prev.needs_research) instead of
                # dict method call (prev.get()) for safe expression evaluation
                run_condition="prev.needs_research == True",
            ),
            AgentSpec(agent_id="responder", role=AgentRole.PRIMARY),
        ]

        # Test with simple query (no research needed)
        context_simple = TaskContext(query="What time is it?", user_id="doug")
        result_simple = await orchestrator.orchestrate(
            agents=agents,
            context=context_simple,
            executor=conditional_executor,
        )

        assert result_simple.success is True
        assert len(result_simple.agent_results) == 2  # classifier + responder (researcher skipped)
        assert result_simple.final_output["used_research"] is False

        # Test with complex query (research needed)
        context_complex = TaskContext(query="Complex question about history", user_id="doug")
        result_complex = await orchestrator.orchestrate(
            agents=agents,
            context=context_complex,
            executor=conditional_executor,
        )

        assert result_complex.success is True
        assert len(result_complex.agent_results) == 3  # All agents ran
        assert result_complex.final_output["used_research"] is True


class TestC1ComponentInteroperability:
    """Test that C.1 components work together correctly."""

    @pytest.mark.anyio
    async def test_learning_channel_with_graph_storage(
        self, learning_channel, cognitive_graph
    ):
        """Test that learnings can be stored in cognitive graph."""
        stored_learnings = []

        # Handler that stores learnings in graph
        async def store_learning_handler(learning: Learning):
            node = await cognitive_graph.add_node(
                content=learning.content,
                node_type=NodeType.FACT,
                scope_id=f"agent:{learning.source_agent_id}",
                metadata={"source_agent_id": learning.source_agent_id},
            )
            stored_learnings.append(node.node_id)

        await learning_channel.subscribe(
            "storage_agent",
            store_learning_handler,
        )

        # Broadcast a learning
        learning = Learning(
            learning_type=LearningType.SKILL,
            content="To restart services, use systemctl restart",
            source_agent_id="operator",
            scope=LearningScope.GLOBAL,
        )
        await learning_channel.broadcast(learning)

        # Verify it was stored
        assert len(stored_learnings) == 1

        node = await cognitive_graph.get_node(stored_learnings[0])
        assert node is not None
        assert "systemctl" in node.content

    @pytest.mark.anyio
    async def test_orchestrator_context_carries_graph_results(
        self, orchestrator, cognitive_graph
    ):
        """Test that orchestrator context can carry graph search results."""
        # Pre-populate graph - add distinct content to make search work better
        await cognitive_graph.add_node(
            content="Doug's favorite color is blue, he loves the color blue",
            node_type=NodeType.FACT,
            scope_id="user:doug",
            entities=["doug", "color", "blue", "favorite"],
            metadata={"source_agent_id": "memory"},
        )

        async def context_carrying_executor(agent: AgentSpec, context: TaskContext) -> AgentResult:
            if agent.agent_id == "memory_retriever":
                results = await cognitive_graph.search(
                    query="doug favorite color blue",
                    scope_ids=["user:doug"],
                    limit=5,
                )
                if results:
                    # Return memories in output - orchestrator stores this in context.agent_outputs
                    return AgentResult(
                        agent_id=agent.agent_id,
                        success=True,
                        output={
                            "found_memories": len(results),
                            "memories": [r.node.content for r in results],
                        },
                    )
                return AgentResult(
                    agent_id=agent.agent_id,
                    success=True,
                    output={"found_memories": 0, "memories": []},
                )

            elif agent.agent_id == "responder":
                # Orchestrator stores each agent's output in context.agent_outputs[agent_id]
                memories = context.agent_outputs.get("memory_retriever", {}).get("memories", [])
                return AgentResult(
                    agent_id=agent.agent_id,
                    success=True,
                    output={"response": f"Based on memories: {memories}"},
                )

            return AgentResult(agent_id=agent.agent_id, success=True, output={})

        agents = [
            AgentSpec(agent_id="memory_retriever", role=AgentRole.RESEARCHER),
            AgentSpec(agent_id="responder", role=AgentRole.PRIMARY),
        ]

        context = TaskContext(query="What is my favorite color?", user_id="doug")
        result = await orchestrator.orchestrate(
            agents=agents,
            context=context,
            executor=context_carrying_executor,
        )

        assert result.success is True
        # Check that memories were found and contain blue
        assert result.agent_results[0].output["found_memories"] > 0
        assert "blue" in result.final_output["response"]
