---
name: constitution
description: Generate or update project constitution and principles
type: workflow
tools: [Read, Write, Edit, Glob, Grep]
model: claude-sonnet-4-5-20250929
---

# /constitution - Project Foundation Generator

## Purpose
Generate or update the project constitution, design principles, and governance documents to establish and maintain project standards for the cognitive AI framework.

## Usage
```
/constitution [area: principles|scope|testing|full]
```

## Process

When this command is invoked:

1. **Read Current Constitution**
   - Read `.specify/constitution/constitution.md`
   - Review `.specify/constitution/principles.md`
   - Check `.specify/constitution/scope.md`
   - Review `.specify/constitution/testing-principles.md`

2. **Analyze Update Request**
   - Identify what aspect needs updating
   - Check for consistency with existing principles
   - Ensure updates align with cognitive architecture

3. **Core Principles (Non-Negotiable)**
   - **LLM-First**: No semantic regex patterns
   - **XML Output**: All LLM prompts use XML
   - **Protocol-Based**: Extensibility via Protocols
   - **Async-First**: Non-blocking operations
   - **4-Layer Memory**: Cognitive memory architecture
   - **Belief-Based**: Observations → Beliefs pipeline
   - **Research-Grounded**: Backed by cognitive science

4. **Update Documentation**
   - Make requested updates to constitution docs
   - Ensure consistency across all files
   - Update version and date

5. **Stage Changes**
   - Use `git add .` to stage constitution updates
   - DO NOT commit (follow manual commit preference)
   - Provide summary of changes

## Constitution Areas

### Principles (/constitution principles)
Design principles that guide all development:
- LLM-first architecture
- 4-layer cognitive memory
- Belief-based knowledge
- Opinion formation
- Curiosity-driven learning
- Confidence-based actions
- Async-first processing
- Protocol-based design
- Research-grounded development

### Scope (/constitution scope)
Project boundaries:
- What draagon-ai IS (framework capabilities)
- What draagon-ai IS NOT (not provided)
- Boundary examples
- Reference implementation

### Testing (/constitution testing)
Testing philosophy:
- Unit tests (foundation)
- Integration tests (critical)
- Cognitive tests (unique to draagon-ai)
- Stress tests (load)
- Benchmark tests (SOTA comparison)
- Test anti-patterns to avoid

### Full (/constitution full)
Complete constitution review and update:
- All sections
- Cross-document consistency
- Version synchronization

## Key Constitution Rules

### Non-Negotiable Principles

```python
# 1. Never Pattern-Match Semantics
# FORBIDDEN
if re.match(r"actually|no,|wrong", user_input):
    handle_correction()

# REQUIRED
correction = await llm.detect_correction(user_input)
if correction.is_correction:
    handle_correction(correction)

# 2. Always Use XML for LLM Output
# FORBIDDEN
prompt = "Return JSON: {\"action\": \"...\"}"

# REQUIRED
prompt = """Return XML:
<response>
    <action>...</action>
</response>"""

# 3. Beliefs Are Not Memories
# FORBIDDEN
memory.store(user_said)

# REQUIRED
observation = belief_service.create_observation(user_said)
belief = await belief_service.reconcile(observation)

# 4. Confidence-Based Actions
# FORBIDDEN
if should_do_action:
    do_action()

# REQUIRED
if confidence > 0.9:
    do_action()
elif confidence > 0.7:
    do_action_with_monitoring()
elif confidence > 0.5:
    confirm_then_do_action()
else:
    ask_for_clarification()
```

## Output Format
Provide:
- Summary of constitution updates
- Changes made to each document
- Consistency verification results
- Staged files confirmation

Remember: The constitution defines non-negotiable principles for the cognitive AI framework. All changes must align with LLM-first architecture, 4-layer memory, and research-grounded development.
