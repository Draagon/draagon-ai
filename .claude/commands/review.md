---
name: review
description: Comprehensive review of project specifications and documentation
type: analysis
tools: [Read, Grep, Glob, Task]
model: claude-sonnet-4-5-20250929
---

# /review - Specification Review and Validation

## Purpose
Perform comprehensive analysis of project specifications, code, and documentation to ensure completeness, consistency, and alignment with cognitive architecture principles.

## Usage
```
/review [area: specs|code|constitution|cognitive|all]
```

## Process

When this command is invoked:

1. **Scan Documentation Landscape**
   - Read all files in `.specify/` directory
   - Map relationships between specifications
   - Identify reference patterns and dependencies
   - Check for orphaned or incomplete documents

2. **Validate Constitution Compliance**
   - Check for semantic regex patterns (FORBIDDEN)
   - Verify LLM prompts use XML format
   - Ensure protocol-based integrations
   - Validate async-first processing patterns

3. **Assess Cognitive Architecture**
   - Verify 4-layer memory implementation
   - Check belief reconciliation pipeline
   - Validate curiosity and opinion formation
   - Review multi-agent coordination patterns

4. **Technical Architecture Review**
   - Validate module organization
   - Check protocol definitions
   - Verify dataclass structures
   - Assess async patterns

5. **Quality Assessment**
   - Check specification clarity and testability
   - Verify acceptance criteria are measurable
   - Assess implementation feasibility
   - Identify potential technical risks

6. **Generate Report**
   - Provide comprehensive analysis summary
   - List constitution violations found
   - Identify cognitive architecture gaps
   - Suggest priority improvements

## Review Areas

### Specifications (/review specs)
- Functional requirements completeness
- Cognitive component coverage
- Protocol interface definitions
- Testing requirements

### Code (/review code)
- Constitution compliance
- Pattern consistency
- Test coverage
- Cognitive behavior implementation

### Constitution (/review constitution)
- Principle violations in code
- Semantic regex patterns
- JSON LLM outputs
- Synchronous blocking

### Cognitive (/review cognitive)
- Memory layer implementation
- Belief reconciliation
- Curiosity engine
- Opinion formation
- Multi-agent coordination

### Complete (/review all)
- Full ecosystem analysis
- Cross-document consistency
- Code-spec alignment
- Overall project health

## Constitution Violation Checks

### Critical Violations (Must Fix)
```python
# VIOLATION: Semantic regex
if re.match(r"actually|no,|wrong", text):
    # MUST use LLM semantic analysis

# VIOLATION: JSON LLM output
prompt = "Return JSON: {...}"
    # MUST use XML format

# VIOLATION: Synchronous blocking in hot path
result = sync_function()  # In async context
    # MUST use async
```

### Warnings (Should Review)
```python
# WARNING: Hard-coded values that could be beliefs
user_prefers_celsius = True
    # SHOULD be stored in belief system

# WARNING: Missing confidence levels
if should_do_action:
    # SHOULD use graduated confidence
```

## Cognitive Checklist

### Memory Architecture
- [ ] Working memory has capacity limits
- [ ] Attention weighting implemented
- [ ] Decay behavior correct
- [ ] Promotion between layers works
- [ ] Session isolation enforced

### Belief System
- [ ] Observations are immutable records
- [ ] Beliefs are reconciled, not direct storage
- [ ] Conflicts detected and flagged
- [ ] Credibility weighting applied
- [ ] Verification status tracked

### Multi-Agent
- [ ] Shared memory has locking
- [ ] Observations have source attribution
- [ ] Transactive memory routes queries
- [ ] Belief reconciliation across agents
- [ ] Metacognitive reflection after tasks

## Output Format
Provide:

### Health Score
```
Overall: A/B/C/D/F
- Constitution Compliance: X/100
- Cognitive Architecture: X/100
- Test Coverage: X/100
- Documentation: X/100
```

### Critical Issues
- Constitution violations requiring immediate fix
- Cognitive architecture gaps

### Recommendations
- Priority improvements
- Next steps

Remember: Review against cognitive AI framework principles. Check LLM-first architecture, 4-layer memory, belief reconciliation, and protocol-based design.
