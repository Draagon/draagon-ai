# TASK-007: Implement App Profiles and Agent Factory

**Phase**: 4 (App Profiles)
**Priority**: P1 (Enables multi-agent testing)
**Effort**: 1 day
**Status**: Pending
**Dependencies**: TASK-001, TASK-002, TASK-003

## Description

Implement AppProfile for configurable agent setups and agent_factory fixture for easy agent creation in tests. This enables testing different agent configurations (personalities, tool sets, memory configs) without manual setup.

## Acceptance Criteria

- [ ] `AppProfile` dataclass with name, personality, tool_set, memory_config, llm_model_tier
- [ ] `ToolSet` enum (MINIMAL, BASIC, FULL)
- [ ] `AgentFactory` class with `create(profile)` method
- [ ] `agent_factory` fixture (function-scoped)
- [ ] `agent` fixture (convenience for default profile)
- [ ] Pre-defined profiles: DEFAULT, RESEARCHER, ASSISTANT
- [ ] Unit test: profile creation
- [ ] Unit test: tool set filtering
- [ ] Integration test: agent with custom profile

## Technical Notes

**AppProfile Pattern:**

```python
@dataclass
class AppProfile:
    """Configuration for agent creation in tests."""
    name: str
    personality: str
    tool_set: ToolSet
    memory_config: dict[str, Any] = field(default_factory=dict)
    llm_model_tier: str = "standard"

class ToolSet(Enum):
    MINIMAL = "minimal"  # Basic answer only
    BASIC = "basic"      # Answer + search
    FULL = "full"        # All tools
```

**AgentFactory:**

```python
class AgentFactory:
    """Creates agents from profiles for testing."""

    def __init__(
        self,
        llm_provider: LLMProvider,
        memory_provider: MemoryProvider,
        tool_registry: ToolRegistry,
    ):
        self.llm = llm_provider
        self.memory = memory_provider
        self.tools = tool_registry

    async def create(self, profile: AppProfile) -> Agent:
        """Create agent from profile."""
        # Filter tools by profile.tool_set
        tools = self._filter_tools(profile.tool_set)

        # Apply memory config overrides
        memory_config = {**DEFAULT_MEMORY_CONFIG, **profile.memory_config}

        # Create agent
        return Agent(
            llm_provider=self.llm,
            memory_provider=self.memory,
            tool_registry=tools,
            personality=profile.personality,
            config=memory_config,
        )

    def _filter_tools(self, tool_set: ToolSet) -> ToolRegistry:
        """Filter tools based on ToolSet."""
        if tool_set == ToolSet.MINIMAL:
            return ToolRegistry()  # Empty registry
        elif tool_set == ToolSet.BASIC:
            return self.tools.filter(["answer", "search_memory"])
        else:
            return self.tools  # All tools
```

**Pre-defined Profiles:**

```python
DEFAULT_PROFILE = AppProfile(
    name="default",
    personality="You are a helpful assistant.",
    tool_set=ToolSet.BASIC,
)

RESEARCHER_PROFILE = AppProfile(
    name="researcher",
    personality="You are a research assistant focused on gathering and analyzing information.",
    tool_set=ToolSet.FULL,
    llm_model_tier="advanced",
)

ASSISTANT_PROFILE = AppProfile(
    name="assistant",
    personality="You are a personal assistant helping with daily tasks.",
    tool_set=ToolSet.BASIC,
    memory_config={"working_ttl": 600},  # 10 minute working memory
)
```

## Testing Requirements

### Unit Tests (`tests/framework/test_app_profiles.py`)

1. **Profile Creation**
   ```python
   def test_create_profile():
       profile = AppProfile(
           name="test",
           personality="Test personality",
           tool_set=ToolSet.MINIMAL,
       )
       assert profile.name == "test"
       assert profile.tool_set == ToolSet.MINIMAL
   ```

2. **Tool Set Filtering**
   ```python
   def test_filter_tools_minimal():
       factory = AgentFactory(llm, memory, tools)
       filtered = factory._filter_tools(ToolSet.MINIMAL)
       assert len(filtered.list_tools()) == 0

   def test_filter_tools_basic():
       factory = AgentFactory(llm, memory, tools)
       filtered = factory._filter_tools(ToolSet.BASIC)
       assert "answer" in filtered.list_tools()
       assert "search_memory" in filtered.list_tools()
   ```

### Integration Test (`tests/integration/test_agent_factory.py`)

```python
@pytest.mark.integration
async def test_agent_with_custom_profile(agent_factory, memory_provider):
    """Test creating agent with custom profile."""
    profile = AppProfile(
        name="custom",
        personality="You are a math tutor.",
        tool_set=ToolSet.MINIMAL,
    )

    agent = await agent_factory.create(profile)
    response = await agent.process("What is 2+2?")

    # Agent should answer without using tools
    assert "4" in response.answer
    assert response.tool_calls == []
```

## Files to Create

- `src/draagon_ai/testing/profiles.py` - NEW
  - `AppProfile` dataclass
  - `ToolSet` enum
  - `AgentFactory` class
  - Pre-defined profiles

- `src/draagon_ai/testing/fixtures.py` - EXTEND
  - `agent_factory` fixture
  - `agent` fixture (default profile)

- `tests/framework/test_app_profiles.py` - NEW
  - Test profile creation
  - Test tool filtering

- `tests/integration/test_agent_factory.py` - NEW
  - Test agent creation with profiles

## Implementation Sequence

1. Define `ToolSet` enum
2. Define `AppProfile` dataclass
3. Implement `AgentFactory.__init__()`
4. Implement `_filter_tools()` method
5. Implement `create()` method
6. Define pre-built profiles (DEFAULT, RESEARCHER, ASSISTANT)
7. Add `agent_factory` fixture
8. Add `agent` fixture (uses DEFAULT_PROFILE)
9. Write unit tests
10. Write integration test

## Cognitive Testing Requirements

**Integration Test Must Verify:**
- [ ] Agent uses correct personality from profile
- [ ] Tool set filtering works (MINIMAL has no tools)
- [ ] Memory config overrides are applied
- [ ] Agent can process queries correctly

## Success Criteria

- Can create agents from profiles easily
- Tool set filtering works correctly
- Pre-defined profiles are usable
- `agent_factory` fixture simplifies test setup
- All tests pass
- Ready for use in integration tests
