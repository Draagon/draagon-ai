---
name: implement
description: Execute implementation tasks systematically with proper testing
type: workflow
tools: [Read, Write, Edit, MultiEdit, Bash, TodoWrite, Grep, Glob]
model: claude-sonnet-4-5-20250929
---

# /implement - Task Implementation Executor

## Purpose
Execute specific implementation tasks from the task breakdown, following established patterns, cognitive architecture principles, and testing requirements.

## Usage
```
/implement [TASK-XXX or specific task description]
```

## Process

When this command is invoked:

1. **Read Implementation Context**
   - Read specified task from `.specify/tasks/phase-*/task-*.md`
   - Review related functional requirements
   - Understand existing codebase patterns in `src/draagon_ai/`
   - Check task dependencies and prerequisites

2. **Validate Against Constitution**
   - No semantic regex patterns
   - XML format for LLM prompts
   - Protocol-based integrations
   - Async-first for non-critical operations

3. **Plan Implementation Approach**
   - Use TodoWrite to break task into sub-steps
   - Identify files to create/modify
   - Plan testing approach:
     - Unit tests
     - Integration tests
     - Cognitive tests (if applicable)

4. **Execute Implementation**
   - Follow existing code patterns:
     - Dataclasses for data structures
     - Protocols for interfaces
     - Async functions for I/O
   - Implement with proper type hints
   - Add comprehensive docstrings
   - Include logging at appropriate levels

5. **Implement Tests**
   - Unit tests in `tests/unit/`
   - Integration tests in `tests/integration/`
   - Cognitive tests if touching:
     - Memory layers
     - Beliefs/opinions
     - Multi-agent coordination

6. **Verify Implementation**
   - Run tests: `pytest tests/ -v`
   - Check type hints: `mypy src/`
   - Verify cognitive behavior if applicable

7. **Update Task Completion Status**
   - Mark acceptance criteria as completed [x]
   - Add implementation summary to task file
   - Update related documentation

8. **Stage Changes (DO NOT COMMIT)**
   - Use `git add .` to stage all files
   - **NEVER use `git commit`**
   - Inform user changes are staged for review

## Implementation Standards

### Code Style
```python
from dataclasses import dataclass, field
from typing import Protocol, Any
import asyncio
import logging

logger = logging.getLogger(__name__)


@dataclass
class MyDataClass:
    """Clear docstring explaining purpose."""

    field: str
    optional_field: int = 0


class MyProtocol(Protocol):
    """Protocol for extensibility."""

    async def operation(self, arg: str) -> dict[str, Any]: ...


async def my_function(arg: str) -> Result:
    """Do something.

    Args:
        arg: Description of argument

    Returns:
        Description of return value
    """
    logger.debug(f"Processing: {arg}")
    # Implementation
    return Result(...)
```

### LLM Prompt Style
```python
prompt = f"""Analyze this context:

{context}

Respond in XML:
<response>
    <action>action_name</action>
    <reasoning>Why this action</reasoning>
    <confidence>0.0-1.0</confidence>
</response>
"""
```

### Test Style
```python
import pytest
from draagon_ai.module import MyClass


class TestMyClass:
    """Tests for MyClass."""

    @pytest.fixture
    def instance(self):
        """Create test instance."""
        return MyClass()

    @pytest.mark.asyncio
    async def test_basic_operation(self, instance):
        """Test basic operation works."""
        result = await instance.operation("input")
        assert result.success
        assert result.value == "expected"

    @pytest.mark.asyncio
    async def test_cognitive_behavior(self, instance):
        """Test cognitive properties."""
        # For memory/belief/opinion tests
        await instance.add_observation("fact")
        beliefs = await instance.get_beliefs()
        assert len(beliefs) == 1
        assert beliefs[0].confidence > 0.5
```

## Cognitive Implementation Checklist

For tasks touching cognitive components:

### Memory Tasks
- [ ] Capacity enforcement (7±2 for working memory)
- [ ] Attention weighting implemented
- [ ] Decay behavior correct
- [ ] Promotion logic works

### Belief Tasks
- [ ] Observations create UserObservation objects
- [ ] Reconciliation produces AgentBelief
- [ ] Conflicts detected and flagged
- [ ] Credibility weighting applied

### Multi-Agent Tasks
- [ ] Shared memory has locking
- [ ] No race conditions under load
- [ ] Observations have source attribution
- [ ] Conflicts reconciled correctly

## Git Workflow (CRITICAL)

**NEVER COMMIT AUTOMATICALLY**
- Always stage changes: `git add .`
- NEVER run: `git commit`
- User reviews changes before committing
- End with: "Changes staged for review. Ready for commit when approved."

## Output Format
Provide:
- Task completion summary
- Files created/modified
- Testing results
- Any issues encountered
- Git status showing staged changes
- "Changes staged for review" confirmation
- Next recommended task

Remember: Implement for the cognitive AI framework. Follow LLM-first principles, XML output format, and protocol-based design.
