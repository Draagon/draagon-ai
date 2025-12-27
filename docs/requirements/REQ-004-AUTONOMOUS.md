# REQ-004: Autonomous Agent Core Integration

**Priority:** Medium
**Estimated Effort:** Medium (1-2 weeks)
**Dependencies:** REQ-001 (Memory), REQ-002 (Orchestrator)
**Blocks:** REQ-006 (Roxy Refactor)

---

## 1. Overview

### 1.1 Current State
- **Roxy** has `autonomous_agent.py` with background cognitive processing
- We created `draagon_ai_ext_autonomous` extension package
- There's also an adapter/factory pattern in Roxy to use extension
- The extension is NOT part of core draagon-ai

### 1.2 Target State
Per the architecture document, autonomous agent should be **core**, not extension:
> "Autonomous agent should be core, not extension - It's a fundamental assistant capability"

Move autonomous agent to draagon-ai core with:
- Protocol-based dependency injection
- Guardrail system with action tiers
- Self-monitoring capability
- Application-agnostic design

### 1.3 Success Metrics
- Autonomous agent runs in any draagon-ai application
- Guardrails prevent harmful actions
- Self-monitoring detects issues
- Dashboard shows activity logs

---

## 2. Detailed Requirements

### 2.1 Move Autonomous Agent to draagon-ai Core

**ID:** REQ-004-01
**Priority:** Critical

#### Description
Move autonomous agent from extension to `draagon_ai.orchestration.autonomous`.

#### Acceptance Criteria
- [ ] Code moved from extension to core
- [ ] No breaking changes to existing API
- [ ] Works without Roxy-specific dependencies
- [ ] Entry point removed (no longer extension)

#### Target Location
```
src/draagon_ai/
  orchestration/
    autonomous/
      __init__.py
      types.py       # Protocols and data models
      service.py     # AutonomousAgentService
      guardrails.py  # GuardrailChecker
      monitor.py     # SelfMonitor
```

---

### 2.2 Protocol-Based Dependency Injection

**ID:** REQ-004-02
**Priority:** Critical

#### Description
All dependencies via protocols, allowing any application to provide implementations.

#### Protocols
```python
@runtime_checkable
class LLMProvider(Protocol):
    async def generate(self, prompt: str, max_tokens: int = 1000) -> str: ...

@runtime_checkable
class SearchProvider(Protocol):
    async def search(self, query: str, limit: int = 5) -> list[SearchResult]: ...

@runtime_checkable
class MemoryStoreProvider(Protocol):
    async def store(self, content: str, memory_type: str) -> str: ...
    async def search(self, query: str) -> list[Memory]: ...

@runtime_checkable
class ContextProvider(Protocol):
    async def gather_context(self) -> AutonomousContext: ...

@runtime_checkable
class NotificationProvider(Protocol):
    async def notify_user(self, user_id: str, message: str, priority: str) -> None: ...
```

#### Acceptance Criteria
- [ ] All dependencies defined as protocols
- [ ] No direct imports from Roxy or other applications
- [ ] Protocols documented with expected behavior
- [ ] Type checking validates protocol conformance

---

### 2.3 Guardrail System with Tiers

**ID:** REQ-004-03
**Priority:** Critical

#### Description
Implement tiered action system with guardrails preventing harmful actions.

#### Action Tiers
```
Tier 0 (Always Safe):
  - Research topics
  - Reflect on interactions
  - Update internal beliefs
  - Note questions to ask later

Tier 1 (Low Risk):
  - Prepare suggestions (but don't send)
  - Summarize conversations
  - Analyze patterns

Tier 2 (Medium - Notify User):
  - Proactive reminders
  - Suggest calendar events
  - Prepare recommendations

Tier 3 (High - Requires Approval):
  - Send messages
  - Modify calendar
  - Control devices

Tier 4 (Forbidden):
  - Financial transactions
  - Security changes
  - Impersonation
```

#### Guardrail Layers
```python
class GuardrailChecker:
    async def check_action(self, action: ProposedAction) -> GuardrailResult:
        # Layer 1: Tier classification
        if action.tier > self.max_allowed_tier:
            return GuardrailResult(allowed=False, reason="tier_exceeded")

        # Layer 2: Harm check (LLM)
        harm_result = await self._check_harm(action)
        if harm_result.could_harm:
            return GuardrailResult(allowed=False, reason=harm_result.explanation)

        # Layer 3: Privacy check
        privacy_result = await self._check_privacy(action)
        if privacy_result.violates:
            return GuardrailResult(allowed=False, reason="privacy_violation")

        # Layer 4: Rate limiting
        if not await self._check_rate_limit(action):
            return GuardrailResult(allowed=False, reason="rate_limited")

        # Layer 5: Budget check
        if not await self._check_budget(action):
            return GuardrailResult(allowed=False, reason="budget_exceeded")

        return GuardrailResult(allowed=True)
```

#### Acceptance Criteria
- [ ] All 5 tiers implemented
- [ ] Each tier has clear boundaries
- [ ] LLM harm check works
- [ ] Privacy check blocks personal data access
- [ ] Rate limiting prevents spam
- [ ] Daily budget enforced

#### Test Cases
| ID | Input | Expected Output | Type |
|----|-------|-----------------|------|
| T01 | Tier 0 action | Allowed | Unit |
| T02 | Tier 4 action | Blocked | Unit |
| T03 | Harmful Tier 0 action | Blocked by harm check | Unit |
| T04 | Over budget | Blocked | Unit |
| T05 | Rate limited | Blocked | Unit |

---

### 2.4 Self-Monitoring Capability

**ID:** REQ-004-04
**Priority:** High

#### Description
After each cycle, agent reviews its own actions for issues.

#### Self-Monitoring Checks
```python
class SelfMonitor:
    async def review_cycle(self, cycle_result: CycleResult) -> list[Finding]:
        findings = []

        # Check for unexpected results
        for action in cycle_result.actions:
            if action.result.unexpected:
                findings.append(Finding(
                    severity="medium",
                    type="unexpected_result",
                    details=action,
                ))

        # Check for contradictions in learned info
        contradictions = await self._check_contradictions(cycle_result)
        findings.extend(contradictions)

        # Check for low-value activities
        if cycle_result.value_score < 0.3:
            findings.append(Finding(
                severity="low",
                type="low_value_cycle",
            ))

        # Check for important findings needing attention
        important = await self._identify_important(cycle_result)
        findings.extend(important)

        return findings
```

#### Acceptance Criteria
- [ ] Self-monitoring runs after each cycle
- [ ] Findings are persisted for dashboard
- [ ] High-severity findings trigger notifications
- [ ] Low-value cycles are flagged for adjustment

---

### 2.5 Action Logging and Dashboard

**ID:** REQ-004-05
**Priority:** Medium

#### Description
Log all autonomous actions for transparency and debugging.

#### Log Structure
```python
@dataclass
class ActionLog:
    id: str
    timestamp: datetime
    action_type: ActionType
    tier: int
    description: str
    outcome: str  # "executed", "blocked", "pending"
    block_reason: str | None
    result: Any | None
    duration_ms: int
    cycle_id: str  # Groups actions in same cycle
```

#### Dashboard Requirements
- View recent actions with filtering
- See block reasons for blocked actions
- View self-monitoring findings
- Track daily budget usage
- Enable/disable autonomous mode

#### Acceptance Criteria
- [ ] All actions logged
- [ ] Logs queryable by time, type, outcome
- [ ] Dashboard endpoint exists
- [ ] Can view findings and actions together

---

### 2.6 Roxy Adapter Implementation

**ID:** REQ-004-06
**Priority:** High

#### Description
Implement adapters in Roxy to provide protocol implementations.

#### Adapters Needed
```python
# src/roxy/adapters/autonomous/
class RoxyLLMProvider:
    """Implements LLMProvider using Roxy's LLM service."""

class RoxySearchProvider:
    """Implements SearchProvider using Roxy's memory."""

class RoxyMemoryStoreProvider:
    """Implements MemoryStoreProvider using Roxy's memory."""

class RoxyContextProvider:
    """Gathers context from Roxy's services."""

class RoxyNotificationProvider:
    """Queues notifications for voice announcement."""
```

#### Acceptance Criteria
- [ ] All protocols implemented
- [ ] Adapters pass type checking
- [ ] Integration with Roxy services works
- [ ] Notifications queue correctly

---

### 2.7 Remove Extension Version

**ID:** REQ-004-07
**Priority:** Low

#### Description
After core integration works, remove the extension package.

#### Cleanup Steps
1. Remove `draagon-ai/extensions/autonomous/` directory
2. Remove entry point from any pyproject.toml
3. Update Roxy's factory to import from core
4. Remove factory's fallback to extension

#### Acceptance Criteria
- [ ] Extension directory removed
- [ ] No references to extension remain
- [ ] Roxy uses core version only
- [ ] Tests pass with core version

---

### 2.8 Unit Tests

**ID:** REQ-004-08
**Priority:** High

#### Coverage Requirements
- 90% coverage for guardrails
- All tier boundaries tested
- All monitoring checks tested

---

### 2.9 Integration Tests

**ID:** REQ-004-09
**Priority:** High

#### Test Scenarios
1. Full autonomous cycle execution
2. Guardrail blocking at each tier
3. Self-monitoring finding detection
4. Action logging completeness

---

### 2.10 Safety E2E Tests

**ID:** REQ-004-10
**Priority:** Critical

#### Description
End-to-end tests verifying safety guarantees.

#### Test Cases
```python
def test_cannot_send_message_without_approval():
    """Tier 3 action requires explicit approval."""
    agent = AutonomousAgentService(config=Config(max_tier=1))
    action = ProposedAction(
        type=ActionType.SEND_MESSAGE,
        tier=3,
        description="Send reminder to user",
    )
    result = await agent.evaluate_action(action)
    assert not result.allowed
    assert result.reason == "tier_exceeded"

def test_harm_check_blocks_harmful():
    """LLM harm check blocks potentially harmful actions."""
    action = ProposedAction(
        type=ActionType.CONTROL_DEVICE,
        tier=2,
        description="Turn off all security cameras",
    )
    result = await agent.evaluate_action(action)
    assert not result.allowed
    assert "harm" in result.reason.lower() or "security" in result.reason.lower()

def test_budget_prevents_spam():
    """Daily budget prevents excessive actions."""
    agent = AutonomousAgentService(config=Config(daily_budget=5))
    for _ in range(5):
        await agent.execute_cycle()
    # 6th cycle should be blocked
    result = await agent.execute_cycle()
    assert result.blocked
    assert "budget" in result.reason
```

---

## 3. Implementation Plan

### 3.1 Sequence
1. Move code from extension to core (REQ-004-01)
2. Verify protocols are complete (REQ-004-02)
3. Implement guardrail system (REQ-004-03)
4. Implement self-monitoring (REQ-004-04)
5. Add logging and dashboard (REQ-004-05)
6. Create Roxy adapters (REQ-004-06)
7. Remove extension (REQ-004-07)
8. Unit tests (REQ-004-08)
9. Integration tests (REQ-004-09)
10. Safety E2E tests (REQ-004-10)

### 3.2 Risks
| Risk | Mitigation |
|------|------------|
| Safety gaps | Extensive adversarial testing |
| LLM harm check inconsistency | Multiple check layers |
| Budget/rate limit bypass | Server-side enforcement |
| Privacy leaks | Explicit data classification |

---

## 4. Review Checklist

### Safety Completeness
- [ ] All tiers properly enforced
- [ ] Harm check catches obvious harms
- [ ] Privacy violations blocked
- [ ] Rate limiting works
- [ ] Budget tracking accurate

### Code Quality
- [ ] Protocols are complete
- [ ] No application-specific code in core
- [ ] Guardrails are defense in depth
- [ ] Logging is comprehensive

### Test Coverage
- [ ] Unit tests ≥ 90%
- [ ] Adversarial cases tested
- [ ] Edge cases covered

---

## 5. God-Level Review Prompt

```
AUTONOMOUS AGENT REVIEW - REQ-004

Context: Moving autonomous agent to draagon-ai core with
protocol-based dependencies and robust guardrails.

Review the implementation against these specific criteria:

1. SAFETY GUARANTEES
   - Can a Tier 4 action EVER execute? (should be impossible)
   - Can harm check be bypassed? (should be impossible)
   - Are there any paths around guardrails?
   - What happens if LLM harm check fails?
   - Is the budget truly enforced?

2. PROTOCOL COMPLETENESS
   - Are all dependencies defined as protocols?
   - Can any application implement these?
   - Are protocols documented clearly?

3. SELF-MONITORING
   - Are all issue types detected?
   - Are findings properly persisted?
   - Do high-severity findings notify?

4. LOGGING & TRANSPARENCY
   - Is every action logged?
   - Can we reconstruct what happened?
   - Is the dashboard useful?

5. ADVERSARIAL SCENARIOS
   - What if LLM tries to escape constraints?
   - What if malicious prompts are injected?
   - What if rate limits are circumvented?

CRITICAL: Any safety gap is a FAIL for this review.

Provide specific code references for any issues found.
Rate each section: PASS / NEEDS_WORK / FAIL
Overall recommendation: READY / NOT_READY
```

