---
name: plan
description: Generate detailed technical implementation plans from specifications
type: workflow
tools: [Read, Write, Edit, Glob, Grep]
model: claude-sonnet-4-5-20250929
---

# /plan - Technical Implementation Planner

## Purpose
Create detailed technical implementation plans based on existing specifications, considering draagon-ai's cognitive architecture and protocol-based design.

## Usage
```
/plan [specification reference or feature area]
```

## Process

When this command is invoked:

1. **Read Architecture Context**
   - Read `.specify/constitution/principles.md` for design patterns
   - Review existing architecture in `src/draagon_ai/`
   - Check related specifications and requirements
   - Review `docs/specs/COGNITIVE_SWARM_ARCHITECTURE.md` if multi-agent related

2. **Analyze Implementation Scope**
   - Identify functional requirements to implement
   - Map to existing module structure:
     - `orchestration/` - Agent loop, decision, execution
     - `memory/` - 4-layer memory architecture
     - `cognition/` - Learning, beliefs, curiosity, opinions
     - `tools/` - Tool decorator and registry
   - Consider integration with existing protocols

3. **Generate Implementation Plan**
   - Break down into module components
   - Define new dataclasses and protocols needed
   - Plan cognitive service integrations
   - Identify LLM prompt designs (XML format)
   - Consider error handling and edge cases
   - Plan test coverage approach

4. **Validate Against Principles**
   - **LLM-First**: No semantic regex patterns
   - **XML Output**: All LLM prompts use XML
   - **Protocol-Based**: New integrations via Protocols
   - **Async-First**: Non-blocking where possible
   - **4-Layer Memory**: Proper layer usage
   - **Belief-Based**: Observations → Beliefs pipeline

5. **Update Planning Documents**
   - Add to appropriate phase plan in `.specify/planning/`
   - Update implementation strategy if needed
   - Define testing approach for new components

6. **Stage Changes**
   - Use `git add .` to stage planning updates
   - DO NOT commit (follow manual commit preference)
   - Provide implementation roadmap summary

## Output Format
Provide:
- High-level implementation approach
- Key modules and classes to build
- Protocol interfaces required
- Cognitive service integrations
- LLM prompt designs (XML)
- Testing strategy overview
- Estimated complexity

## Architecture Patterns

### Module Organization
```
src/draagon_ai/
├── orchestration/      # Agent loop, decision, execution
├── memory/             # 4-layer cognitive memory
├── cognition/          # Learning, beliefs, curiosity, opinions
├── tools/              # @tool decorator, registry
├── behaviors/          # Agent behavior definitions
└── evolution/          # Prompt evolution (experimental)
```

### Protocol Pattern
```python
class NewProvider(Protocol):
    async def operation(self, args: Args) -> Result: ...
```

### LLM Prompt Pattern
```python
prompt = """Analyze this input:

{context}

Respond in XML:
<response>
    <result>...</result>
    <confidence>0.0-1.0</confidence>
</response>
"""
```

### Cognitive Integration Pattern
```python
# Post-response learning (async, non-blocking)
asyncio.create_task(
    learning_service.process_interaction(query, response)
)
```

Remember: Plan for the cognitive AI framework architecture. Consider 4-layer memory, belief reconciliation, protocol-based extensibility, and LLM-first principles.
