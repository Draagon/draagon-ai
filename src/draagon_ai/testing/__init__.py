"""
Draagon AI Testing Framework.

A comprehensive testing infrastructure for AI agent applications built on draagon-ai.
Provides caching, mocking, test mode management, and integration testing tools.

Quick Start: Integration Testing (FR-009)
------------------------------------------

1. Define seeds for test data:

    from draagon_ai.testing import SeedFactory, SeedItem, SeedSet

    @SeedFactory.register("user_doug")
    class DougUserSeed(SeedItem):
        async def create(self, provider):
            return await provider.store(
                content="User: Doug, 3 cats",
                memory_type=MemoryType.FACT,
                scope=MemoryScope.USER,
            )

    USER_WITH_CATS = SeedSet("user_with_cats", ["user_doug"])

2. Write integration tests:

    @pytest.mark.integration
    async def test_recall_cats(seed, memory_provider, evaluator):
        await seed.apply(USER_WITH_CATS, memory_provider)

        response = await agent.process("What are my cats' names?")

        result = await evaluator.evaluate_correctness(
            query="What are my cats' names?",
            expected_outcome="Lists cat names",
            actual_response=response.answer
        )
        assert result.correct

3. Set up fixtures in conftest.py:

    from draagon_ai.testing import TestDatabase, create_test_database

    @pytest.fixture(scope="session")
    async def test_database():
        db = await create_test_database()
        yield db
        await db.close()

    @pytest.fixture
    async def clean_database(test_database):
        await test_database.clear()
        yield test_database


Unit Testing with Mocks
-----------------------

1. Configure test modes in your conftest.py:

    from draagon_ai.testing import TestMode, test_mode

    @pytest.fixture
    def unit_test_mode(my_service_mock):
        with test_mode(TestMode.UNIT, mocks={"my_service": my_service_mock}):
            yield

2. Create service-specific mocks by extending ServiceMock:

    from draagon_ai.testing import ServiceMock

    class MyServiceMock(ServiceMock):
        def my_method(self, arg):
            return self.get_response(arg)

3. Use cache manager for external API caching:

    from draagon_ai.testing import UnifiedCacheManager, CacheMode

    cache = UnifiedCacheManager.get_instance()
    result = await cache.get_or_call("llm", (prompt,), call_fn)

4. Run tests with different modes:

    # Fast unit tests (all mocked)
    pytest tests/ -m unit

    # Integration tests (real LLM, mocked services)
    pytest tests/ -m integration
"""

from draagon_ai.testing.cache import (
    CacheMode,
    CacheConfig,
    UnifiedCacheManager,
    CacheMissError,
)

from draagon_ai.testing.mocks import (
    MockResponse,
    ServiceMock,
    MockNotFoundError,
)

from draagon_ai.testing.modes import (
    TestMode,
    TestContext,
    get_test_context,
    is_test_mode,
    get_mock,
    test_mode,
    with_mocks,
)

from draagon_ai.testing.seeds import (
    CircularDependencyError,
    SeedFactory,
    SeedItem,
    SeedNotFoundError,
    SeedSet,
)

from draagon_ai.testing.database import (
    TestDatabase,
    TestDatabaseConfig,
)

from draagon_ai.testing.fixtures import (
    SeedApplicator,
    create_test_database,
    seed_factory,
    seed,
)

from draagon_ai.testing.sequences import (
    StepDependencyError,
    StepOrderError,
    TestSequence,
    step,
)

from draagon_ai.testing.evaluation import (
    AgentEvaluator,
    EvaluationResult,
    QualityResult,
)

from draagon_ai.testing.profiles import (
    AppProfile,
    ToolSet,
    DEFAULT_PROFILE,
    RESEARCHER_PROFILE,
    ASSISTANT_PROFILE,
    MINIMAL_PROFILE,
)

__all__ = [
    # Cache
    "CacheMode",
    "CacheConfig",
    "UnifiedCacheManager",
    "CacheMissError",
    # Mocks
    "MockResponse",
    "ServiceMock",
    "MockNotFoundError",
    # Modes
    "TestMode",
    "TestContext",
    "get_test_context",
    "is_test_mode",
    "get_mock",
    "test_mode",
    "with_mocks",
    # Seeds (FR-009)
    "SeedFactory",
    "SeedItem",
    "SeedSet",
    "SeedNotFoundError",
    "CircularDependencyError",
    # Database (FR-009)
    "TestDatabase",
    "TestDatabaseConfig",
    # Fixtures (FR-009)
    "SeedApplicator",
    "create_test_database",
    "seed_factory",
    "seed",
    # Sequences (FR-009)
    "TestSequence",
    "step",
    "StepDependencyError",
    "StepOrderError",
    # Evaluation (FR-009)
    "AgentEvaluator",
    "EvaluationResult",
    "QualityResult",
    # Profiles (FR-009)
    "AppProfile",
    "ToolSet",
    "DEFAULT_PROFILE",
    "RESEARCHER_PROFILE",
    "ASSISTANT_PROFILE",
    "MINIMAL_PROFILE",
]
