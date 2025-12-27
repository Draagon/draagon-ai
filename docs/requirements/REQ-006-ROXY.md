# REQ-006: Roxy Thin Client Refactor

**Priority:** High
**Estimated Effort:** Large (2-3 weeks)
**Dependencies:** REQ-001, REQ-002, REQ-003, REQ-004, REQ-005
**Blocks:** None (Final integration)

---

## 1. Overview

### 1.1 Current State
Roxy has significant duplicate logic that should be in draagon-ai:
- Memory service (instead of LayeredMemoryProvider)
- Orchestrator (instead of Agent/AgentLoop)
- Cognitive services (duplicated from draagon-ai)
- Autonomous agent (instead of core version)

### 1.2 Target State
After migration, Roxy should contain ONLY:
```
┌─────────────────────────────────────────────────────────────────┐
│                    roxy-voice-assistant                          │
│                                                                   │
│  ┌─────────────────────┐ ┌─────────────────────┐               │
│  │    ADAPTERS         │ │  PERSONALITY CONFIG  │               │
│  │                     │ │                      │               │
│  │  RoxyMemoryAdapter  │ │  roxy_persona.yaml   │               │
│  │  RoxyLLMAdapter     │ │  - name: "Roxy"      │               │
│  │  RoxyToolRegistry   │ │  - values: {...}     │               │
│  │                     │ │  - traits: {...}     │               │
│  └─────────────────────┘ └─────────────────────┘               │
│                                                                   │
│  ┌─────────────────────┐ ┌─────────────────────┐               │
│  │  CHANNEL-SPECIFIC   │ │  APPLICATION SHELL  │               │
│  │                     │ │                      │               │
│  │  Voice/TTS          │ │  FastAPI + routes    │               │
│  │  Home Assistant     │ │  main.py             │               │
│  │  Wyoming Protocol   │ │  dashboard.py        │               │
│  │  Notifications      │ │                      │               │
│  └─────────────────────┘ └─────────────────────┘               │
└─────────────────────────────────────────────────────────────────┘
```

### 1.3 Success Metrics
- Roxy codebase significantly smaller (50%+ reduction in services/)
- All functionality comes from draagon-ai
- Voice pipeline works correctly
- No regression in user experience
- Tests pass

---

## 2. Detailed Requirements

### 2.1 All Adapters Using draagon-ai

**ID:** REQ-006-01
**Priority:** Critical

#### Description
All Roxy adapters must use draagon-ai services, not local duplicates.

#### Required Adapters
```python
# src/roxy/adapters/

# Memory
class RoxyMemoryAdapter:
    """Uses LayeredMemoryProvider from draagon-ai."""
    def __init__(self, provider: LayeredMemoryProvider): ...

# LLM
class RoxyLLMAdapter:
    """Implements LLMProvider protocol using Roxy's Groq/Ollama."""
    async def generate(self, prompt: str, ...) -> str: ...
    async def chat(self, messages: list[Message], ...) -> str: ...

# Orchestration
class RoxyOrchestrationAdapter:
    """Uses Agent/AgentLoop from draagon-ai."""
    def __init__(self, agent: Agent): ...
    async def process_message(self, ...) -> Response: ...

# Cognitive
class RoxyCognitiveAdapter:
    """Bundles all cognitive service adapters."""
    beliefs: RoxyBeliefAdapter
    curiosity: RoxyCuriosityAdapter
    opinions: RoxyOpinionAdapter

# Autonomous
class RoxyAutonomousAdapter:
    """Uses AutonomousAgentService from draagon-ai core."""
    def __init__(self, service: AutonomousAgentService): ...
```

#### Acceptance Criteria
- [ ] Each adapter exists and is tested
- [ ] Adapters implement correct protocols
- [ ] No business logic in adapters
- [ ] All adapters use dependency injection

---

### 2.2 Personality Configuration (YAML)

**ID:** REQ-006-02
**Priority:** High

#### Description
Move personality content to YAML configuration, not code.

#### File: roxy_persona.yaml
```yaml
# roxy-voice-assistant/config/roxy_persona.yaml

name: "Roxy"
version: "2.5.0"

# Core Values (stable, rarely change)
values:
  truth_seeking: 0.95
  epistemic_humility: 0.90
  helpfulness: 0.95
  respect_privacy: 0.90
  transparency: 0.85

# Personality Traits (evolvable parameters)
traits:
  verification_threshold: 0.7
  curiosity_intensity: 0.6
  debate_persistence: 0.4
  proactive_helpfulness: 0.7
  reflection_frequency: 0.5
  risk_tolerance: 0.3

# Communication Style
communication:
  formality: 0.3      # 0=casual, 1=formal
  humor: 0.5          # 0=serious, 1=playful
  verbosity: 0.3      # 0=concise, 1=detailed
  empathy: 0.7        # 0=matter-of-fact, 1=emotionally aware

# Voice Settings
voice:
  tts_enabled: true
  response_max_words: 40
  expand_abbreviations: true
  spoken_numbers: true

# Autonomous Settings
autonomous:
  enabled: true
  shadow_mode: false
  cycle_minutes: 30
  daily_budget: 20
  max_tier: 1
```

#### Loader
```python
class PersonalityConfig:
    @classmethod
    def load(cls, path: str) -> "PersonalityConfig":
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_identity(self) -> Identity:
        """Convert to draagon-ai Identity object."""
        return Identity(
            name=self.name,
            values=self.values,
            traits=self.traits,
            communication=self.communication,
        )
```

#### Acceptance Criteria
- [ ] All personality config in YAML
- [ ] YAML is validated on load
- [ ] Changes to YAML take effect on restart
- [ ] Defaults are sensible

---

### 2.3 Voice/TTS Channel Handling

**ID:** REQ-006-03
**Priority:** High

#### Description
Keep voice-specific code in Roxy as channel handling.

#### Voice-Specific Components
```python
# src/roxy/channels/voice/

class TTSOptimizer:
    """Optimizes text for speech output."""
    def optimize(self, text: str) -> str:
        # Time: "9:00 AM" → "nine AM"
        # Abbreviations: "Dr." → "Doctor"
        # Markdown: strip
        ...

class ResponseCondenser:
    """Condenses long responses for voice."""
    def condense(self, text: str, max_words: int = 40) -> CondensedResponse:
        ...

class ResponseVarier:
    """Varies responses to avoid repetition."""
    def vary(self, text: str, history: list[str]) -> str:
        ...

class VoiceHandler:
    """Main handler for voice input/output."""
    def __init__(
        self,
        tts_optimizer: TTSOptimizer,
        condenser: ResponseCondenser,
        varier: ResponseVarier,
    ): ...

    async def handle_voice_input(self, audio: bytes) -> VoiceResponse:
        ...
```

#### Acceptance Criteria
- [ ] TTS optimization works
- [ ] Response condensation works
- [ ] Response variation works
- [ ] Voice handler orchestrates all

---

### 2.4 Home Assistant Integration

**ID:** REQ-006-04
**Priority:** High

#### Description
Keep Home Assistant integration as Roxy-specific channel code.

#### HA-Specific Components
```python
# src/roxy/channels/homeassistant/

class HomeAssistantService:
    """Interacts with Home Assistant API."""
    async def get_entity(self, entity_id: str) -> Entity: ...
    async def call_service(self, domain: str, service: str, data: dict) -> Result: ...
    async def resolve_entity_id(self, name: str, domain: str) -> str | None: ...

class RoomAwareHandler:
    """Handles room-aware commands."""
    async def get_area_from_device(self, device_id: str) -> str | None: ...
    async def resolve_entity_in_area(self, name: str, area: str) -> str | None: ...

class HAToolRegistry:
    """Registers HA-specific tools."""
    tools = [
        "get_entity",
        "search_entities",
        "call_service",
        # etc.
    ]
```

#### Acceptance Criteria
- [ ] All HA functionality works
- [ ] Room awareness works
- [ ] Entity resolution works
- [ ] Tools register correctly

---

### 2.5 Wyoming Protocol Support

**ID:** REQ-006-05
**Priority:** Medium

#### Description
Keep Wyoming protocol handling as Roxy-specific.

#### Wyoming Components
```python
# src/roxy/channels/wyoming/

class WyomingHandler:
    """Handles Wyoming protocol for voice satellites."""
    async def handle_wake_word(self, event: WakeWordEvent) -> None: ...
    async def handle_speech_to_text(self, event: STTEvent) -> str: ...
    async def handle_text_to_speech(self, text: str) -> bytes: ...
```

#### Acceptance Criteria
- [ ] Wyoming protocol works with voice satellites
- [ ] Wake word detection works
- [ ] STT/TTS pipeline works

---

### 2.6 Remove All Duplicated Code

**ID:** REQ-006-06
**Priority:** Critical

#### Description
Archive or remove all code that is now in draagon-ai.

#### Files to Remove/Archive
```
src/roxy/services/
  memory.py                    # → Use draagon-ai LayeredMemoryProvider
  belief_reconciliation.py     # → Use draagon-ai BeliefReconciliationService
  curiosity_engine.py          # → Use draagon-ai CuriosityEngine
  opinion_formation.py         # → Use draagon-ai OpinionFormationService
  learning.py                  # → Use draagon-ai LearningService (with extensions)
  roxy_self.py                 # → Use draagon-ai IdentityManager + YAML config
  proactive_questions.py       # → Use draagon-ai ProactiveQuestionTimingService

src/roxy/agent/
  orchestrator.py              # → Use draagon-ai Agent/AgentLoop
  prompts.py                   # → Move prompts to draagon-ai

src/roxy/models/
  cognitive.py                 # → Use draagon-ai types
```

#### Archive Strategy
```bash
# Create archive directory
mkdir -p src/roxy/_archived

# Move files
mv src/roxy/services/memory.py src/roxy/_archived/
# etc.

# Update imports throughout codebase
# Run tests to verify nothing broken
```

#### Acceptance Criteria
- [ ] Each file reviewed for unique logic
- [ ] Unique logic moved to draagon-ai or kept as extension
- [ ] Duplicate files archived (not deleted initially)
- [ ] All imports updated
- [ ] Tests pass

---

### 2.7 FastAPI Shell Only

**ID:** REQ-006-07
**Priority:** High

#### Description
main.py should be thin shell that wires up components.

#### Target main.py Structure
```python
# src/roxy/main.py

from fastapi import FastAPI
from draagon_ai import Agent, LayeredMemoryProvider, AutonomousAgentService
from roxy.adapters import RoxyMemoryAdapter, RoxyLLMAdapter, RoxyOrchestrationAdapter
from roxy.channels.voice import VoiceHandler
from roxy.channels.homeassistant import HomeAssistantService
from roxy.config import Settings, PersonalityConfig

app = FastAPI(title="Roxy Voice Assistant")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load configuration
    settings = Settings()
    personality = PersonalityConfig.load(settings.persona_path)

    # Initialize draagon-ai components
    memory = LayeredMemoryProvider(settings.memory_config)
    llm_adapter = RoxyLLMAdapter(settings.llm_config)
    agent = Agent(memory=memory, llm=llm_adapter, identity=personality.to_identity())

    # Initialize channels
    ha_service = HomeAssistantService(settings.ha_config)
    voice_handler = VoiceHandler(settings.voice_config)

    # Wire up
    app.state.agent = agent
    app.state.ha = ha_service
    app.state.voice = voice_handler

    yield

    # Cleanup
    await memory.close()

app = FastAPI(lifespan=lifespan)

# Include routes
app.include_router(api_router)
app.include_router(dashboard_router)
```

#### Acceptance Criteria
- [ ] main.py is <200 lines
- [ ] All components from draagon-ai
- [ ] Clean startup/shutdown
- [ ] Configuration via Settings

---

### 2.8 Full Regression Test Suite

**ID:** REQ-006-08
**Priority:** Critical

#### Description
Ensure all existing functionality still works.

#### Test Categories
```
tests/
  suites/
    memory/           # Memory store/recall/delete
    concise/          # Voice response brevity
    knowledge/        # RAG retrieval
    homeassistant/    # HA integration
    multiturn/        # Conversation context
    proactive/        # Proactive suggestions
    security/         # Command security
    tts/              # TTS optimization
    autonomous/       # Cognitive architecture
```

#### Acceptance Criteria
- [ ] All existing tests pass
- [ ] No new test failures
- [ ] Coverage maintained or improved
- [ ] Performance not degraded

---

### 2.9 Performance Comparison

**ID:** REQ-006-09
**Priority:** Medium

#### Description
Verify refactored version performs as well or better.

#### Benchmarks
| Metric | Before | After | Target |
|--------|--------|-------|--------|
| Simple query latency | X ms | ? ms | ≤ X ms |
| Memory search | X ms | ? ms | ≤ X ms |
| Tool execution | X ms | ? ms | ≤ X ms |
| Startup time | X s | ? s | ≤ X+1 s |
| Memory usage | X MB | ? MB | ≤ X+50 MB |

#### Acceptance Criteria
- [ ] Benchmarks run before and after
- [ ] No significant regression (>20%)
- [ ] Any regressions documented with reason

---

### 2.10 Documentation Update

**ID:** REQ-006-10
**Priority:** Medium

#### Description
Update all documentation to reflect new architecture.

#### Documentation Updates
- [ ] CLAUDE.md updated with new architecture
- [ ] README.md updated
- [ ] API docs updated
- [ ] Architecture diagrams updated
- [ ] Migration notes for future reference

---

## 3. Implementation Plan

### 3.1 Sequence
1. Create all adapters (REQ-006-01)
2. Create personality YAML (REQ-006-02)
3. Organize voice channel code (REQ-006-03)
4. Organize HA channel code (REQ-006-04)
5. Organize Wyoming code (REQ-006-05)
6. Archive duplicate code (REQ-006-06)
7. Refactor main.py (REQ-006-07)
8. Run full regression tests (REQ-006-08)
9. Performance comparison (REQ-006-09)
10. Update documentation (REQ-006-10)

### 3.2 Risks
| Risk | Mitigation |
|------|------------|
| Breaking changes | Full regression suite |
| Missing functionality | Compare before/after behavior |
| Performance regression | Benchmark before/after |
| Lost edge cases | Keep archived code for reference |

---

## 4. Review Checklist

### Architectural Purity
- [ ] Roxy contains only adapters, channels, config
- [ ] All logic comes from draagon-ai
- [ ] No duplicate implementations
- [ ] Clear separation of concerns

### Functional Completeness
- [ ] All voice features work
- [ ] All HA features work
- [ ] All memory features work
- [ ] All cognitive features work

### Code Quality
- [ ] Codebase is 50%+ smaller
- [ ] Clean adapter pattern
- [ ] Proper dependency injection
- [ ] No orphaned code

### Test Coverage
- [ ] All regression tests pass
- [ ] Coverage maintained
- [ ] Performance acceptable

---

## 5. God-Level Review Prompt

```
ROXY THIN CLIENT REVIEW - REQ-006

Context: Refactoring Roxy to be a thin client using draagon-ai core,
with only adapters, personality config, and channel-specific code.

Review the implementation against these specific criteria:

1. ARCHITECTURAL PURITY
   - Is Roxy truly thin (adapters + channels + config only)?
   - Is there ANY business logic that should be in draagon-ai?
   - Are adapters truly thin (just protocol adaptation)?
   - Is personality completely configurable via YAML?

2. WHAT REMAINS IN ROXY?
   Walk through each remaining file and justify:
   - Why is this file in Roxy and not draagon-ai?
   - Is this truly channel-specific or implementation-specific?
   - Could this be moved to draagon-ai?

3. WHAT WAS REMOVED?
   Verify all duplicates are gone:
   - Is memory.py gone (using LayeredMemoryProvider)?
   - Is orchestrator.py gone (using Agent)?
   - Are cognitive services gone (using draagon-ai)?
   - Is there any orphaned code?

4. FUNCTIONAL COMPLETENESS
   - Do all voice features still work?
   - Do all HA features still work?
   - Do all cognitive features still work?
   - Is any functionality missing?

5. CODEBASE METRICS
   - How much smaller is the codebase?
   - Are there fewer lines of code in services/?
   - Is main.py actually thin?

6. REGRESSION SAFETY
   - Do all tests pass?
   - Is coverage maintained?
   - Is performance acceptable?

Provide specific code references for any issues found.
Rate each section: PASS / NEEDS_WORK / FAIL
Overall recommendation: READY / NOT_READY
```

