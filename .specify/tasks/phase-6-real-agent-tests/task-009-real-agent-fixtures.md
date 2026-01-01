# TASK-009: Implement Real Agent Test Fixtures

**Phase**: 6 (Real Agent Integration Tests - FR-010)
**Priority**: P0 (Critical - foundation for all real agent tests)
**Effort**: 1 day
**Status**: ✅ Completed
**Dependencies**: FR-009 (TASK-001 through TASK-008 completed)

## Description

Implement test fixtures for real agent integration tests using actual LLM providers, Neo4j memory, and real cognitive services. This is the infrastructure layer that enables testing real agent behavior (not mocks).

**Core Principle:** Test real components with real LLM, real database, real embeddings.

## Acceptance Criteria

- [x] `embedding_provider` fixture (session-scoped) using real embedding provider
- [x] `real_llm` fixture (session-scoped) using Groq or OpenAI based on env vars
- [x] `memory_provider` fixture (function-scoped) with semantic decomposition enabled
- [x] `tool_registry` fixture (function-scoped) with clean state per test
- [x] `agent` fixture (function-scoped) with real AgentLoop
- [x] `evaluator` fixture (function-scoped) for LLM-as-judge
- [x] `advance_time` test utility for TTL testing
- [x] Fixtures fail gracefully if API keys missing (pytest.skip)

## Technical Notes

**Fixture File:** `tests/integration/agents/conftest.py`

**Embedding Provider:**
```python
@pytest.fixture(scope="session")
async def embedding_provider():
    """Real embedding provider for vector operations."""
    from draagon_ai.llm import get_embedding_provider
    provider = get_embedding_provider()  # Uses configured provider
    await provider.initialize()
    yield provider
    await provider.close()
```

**Real LLM:**
```python
@pytest.fixture(scope="session")
async def real_llm():
    """Real LLM provider (Groq/OpenAI based on env vars)."""
    import os
    if os.getenv("GROQ_API_KEY"):
        from draagon_ai.llm import GroqProvider
        return GroqProvider(api_key=os.getenv("GROQ_API_KEY"))
    elif os.getenv("OPENAI_API_KEY"):
        from draagon_ai.llm import OpenAIProvider
        return OpenAIProvider(api_key=os.getenv("OPENAI_API_KEY"))
    else:
        pytest.skip("No LLM provider configured (need GROQ_API_KEY or OPENAI_API_KEY)")
```

**Memory Provider:**
```python
@pytest.fixture
async def memory_provider(clean_database, embedding_provider, real_llm):
    """Real Neo4jMemoryProvider with semantic decomposition."""
    config = Neo4jMemoryConfig(
        uri=clean_database.get_config()["uri"],
        username=clean_database.get_config()["username"],
        password=clean_database.get_config()["password"],
    )

    provider = Neo4jMemoryProvider(
        config=config,
        embedding_provider=embedding_provider,
        llm=real_llm,
    )
    await provider.initialize()

    yield provider

    await provider.close()
```

**Agent:**
```python
@pytest.fixture
async def agent(memory_provider, real_llm, tool_registry):
    """Fully configured real agent."""
    config = AgentLoopConfig(
        personality="You are a helpful AI assistant for testing.",
        loop_mode=LoopMode.AUTO,
    )

    agent = AgentLoop(
        llm=real_llm,
        memory=memory_provider,
        tool_registry=tool_registry,
        config=config,
    )

    return agent
```

**Time Advance Utility:**
```python
@pytest.fixture
def advance_time():
    """Utility for fast-forwarding time in tests."""
    async def _advance(minutes: int):
        # Option 1: Mock time (fast)
        # Option 2: Use real delays with @pytest.mark.slow
        pass
    return _advance
```

## Testing Requirements

**Unit Tests:** None (fixtures are integration test infrastructure)

**Integration Tests:**
- [ ] Test that `real_llm` fixture skips if no API key
- [ ] Test that `memory_provider` initializes Neo4j connection
- [ ] Test that `agent` fixture creates working agent
- [ ] Test that `embedding_provider` can generate embeddings
- [ ] Test that fixtures clean up properly (no leaked connections)

**Cognitive Tests:** N/A (infrastructure only)

## Files to Create/Modify

**Create:**
- `tests/integration/agents/conftest.py` - All fixtures

**Modify:**
- `tests/integration/conftest.py` - Import agent fixtures if needed
- `pyproject.toml` - Add integration test dependencies

## Dependencies

**External:**
- Neo4j 5.26+ running (bolt://localhost:7687)
- GROQ_API_KEY or OPENAI_API_KEY environment variable
- Embedding provider configured (Qdrant, Pinecone, or built-in)

**Internal:**
- `draagon_ai.orchestration.AgentLoop`
- `draagon_ai.memory.Neo4jMemoryProvider`
- `draagon_ai.llm.GroqProvider` or `OpenAIProvider`
- `draagon_ai.testing.AgentEvaluator` (from FR-009)
- `draagon_ai.testing.TestDatabase` (from FR-009)

## Success Metrics

- All fixtures initialize without errors (with valid config)
- Fixtures skip gracefully if dependencies missing
- No resource leaks (connections closed properly)
- Fixtures can be used in downstream tests
- Session-scoped fixtures shared across tests

## Notes

**Cost Control:**
- Session-scoped `real_llm` and `embedding_provider` reduce API costs
- Function-scoped `memory_provider` ensures clean state per test
- Consider adding `USE_MOCK_LLM` env var for free local testing

**Error Handling:**
- Fixtures MUST skip (not fail) if dependencies unavailable
- Use `pytest.skip("reason")` for missing API keys
- Use `pytest.fail("reason")` for actual fixture bugs
