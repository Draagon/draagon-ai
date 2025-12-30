---
name: tasks
description: Break down implementation plans into actionable development tasks
type: workflow
tools: [Read, Write, Edit, TodoWrite]
model: claude-sonnet-4-5-20250929
---

# /tasks - Implementation Task Breakdown

## Purpose
Convert implementation plans into specific, actionable development tasks with clear acceptance criteria, dependencies, and cognitive testing requirements.

## Usage
```
/tasks [plan reference or feature area]
```

## Process

When this command is invoked:

1. **Read Planning Context**
   - Review relevant implementation plan from `.specify/planning/`
   - Read existing tasks in `.specify/tasks/`
   - Understand current phase objectives and constraints
   - Check cognitive architecture requirements

2. **Create Task Breakdown**
   - Generate TASK-XXX entries following established pattern
   - Define clear acceptance criteria for each task
   - Estimate effort (1-3 day tasks preferred)
   - Identify dependencies between tasks
   - Include cognitive testing requirements:
     - Belief consistency tests
     - Memory layer tests
     - Multi-agent coordination tests

3. **Task Categories**
   - **Infrastructure**: Dataclasses, protocols, base classes
   - **Core Logic**: Main implementation
   - **Integration**: Wiring with existing components
   - **Cognitive**: Belief, memory, learning behavior
   - **Testing**: Unit, integration, cognitive tests
   - **Documentation**: API docs, examples

4. **Update Task Documentation**
   - Create task file in `.specify/tasks/phase-X/task-XXX-*.md`
   - Ensure tasks follow TASK-XXX numbering pattern
   - Include all technical details and acceptance criteria

5. **Create Todo List**
   - Use TodoWrite tool to create actionable todo items
   - Map specification tasks to implementation todos
   - Set appropriate priorities and dependencies

6. **Stage Changes**
   - Use `git add .` to stage task updates
   - DO NOT commit (follow manual commit preference)
   - Provide task summary and next steps

## Task Format

Each task file should include:

```markdown
# TASK-XXX: [Clear Title]

**Phase**: X
**Priority**: P0/P1/P2
**Effort**: 1-3 days
**Status**: Pending | In Progress | Completed
**Dependencies**: TASK-YYY, TASK-ZZZ

## Description
Clear description of what needs to be built.

## Acceptance Criteria
- [ ] Criterion 1 (testable)
- [ ] Criterion 2 (testable)
- [ ] Cognitive test: Belief consistency verified
- [ ] Cognitive test: Memory layer behavior correct

## Technical Notes
- Implementation details
- Key classes/modules
- Integration points

## Testing Requirements
- Unit tests for core logic
- Integration tests for module interaction
- Cognitive tests for belief/memory behavior

## Files to Create/Modify
- `src/draagon_ai/module/file.py`
- `tests/module/test_file.py`
```

## Cognitive Testing Requirements

Every task that touches cognitive components must include:

1. **Memory Tasks**
   - Capacity enforcement (Miller's Law)
   - Attention decay behavior
   - Promotion between layers

2. **Belief Tasks**
   - Observation → belief pipeline
   - Conflict detection
   - Reconciliation logic

3. **Multi-Agent Tasks**
   - Shared memory consistency
   - No race conditions
   - Conflict resolution

4. **Learning Tasks**
   - Post-response extraction
   - Skill confidence tracking
   - Relearning triggers

## Output Format
Provide:
- Number of tasks created
- Task dependency graph
- Implementation sequence recommendation
- Testing strategy per task
- Ready-to-implement task prioritization

Remember: Tasks should target the cognitive AI framework architecture. Include cognitive testing requirements for any task touching memory, beliefs, or multi-agent coordination.
