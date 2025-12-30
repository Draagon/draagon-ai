---
name: specify
description: Transform high-level feature ideas into comprehensive specifications
type: workflow
tools: [Read, Write, Edit, Glob, Grep]
model: claude-sonnet-4-5-20250929
---

# /specify - Feature Specification Generator

## Purpose
Transform a high-level feature description into a comprehensive specification document following draagon-ai's established patterns and cognitive architecture principles.

## Usage
```
/specify [feature description]
```

## Process

When this command is invoked:

1. **Parse User Input**
   - Extract feature description from command arguments
   - If empty: ERROR "No feature description provided"
   - Identify: components, cognitive aspects, integration points

2. **Read Existing Context**
   - Read `.specify/constitution/constitution.md` to understand project principles
   - Review `.specify/constitution/principles.md` for design patterns
   - Check existing specifications in `.specify/requirements/`

3. **Validate Against Constitution**
   - Ensure feature uses LLM-first architecture (no regex for semantics)
   - Verify XML output format for any LLM prompts
   - Check protocol-based design patterns
   - Confirm async-first processing where appropriate

4. **Analyze the Request**
   - Break down feature into core requirements
   - Identify which cognitive services are involved:
     - Memory layers (working, episodic, semantic, metacognitive)
     - Beliefs and opinions
     - Curiosity and learning
     - Multi-agent coordination
   - Consider how it fits with existing architecture
   - For unclear aspects:
     - Make informed guesses based on cognitive architecture patterns
     - Mark with [NEEDS CLARIFICATION: specific question] if ambiguous
     - **LIMIT: Maximum 3 [NEEDS CLARIFICATION] markers total**

5. **Generate Specification**
   - Create detailed functional requirements following FR-XXX pattern
   - Each requirement must be testable
   - Define data structures and protocols needed
   - Write usage examples with Python code
   - Define success criteria:
     - Quantitative metrics (latency, accuracy, throughput)
     - Cognitive metrics (belief consistency, learning effectiveness)
   - Identify implementation tasks and complexity

6. **Validate Completeness**
   - Ensure all functional requirements are testable
   - Verify cognitive properties are measurable
   - Check protocol compliance

7. **Update Documentation**
   - Create new FR-XXX file in `.specify/requirements/`
   - Update related specifications if needed
   - Add to phase plans in `.specify/planning/` as appropriate

8. **Stage Changes**
   - Use `git add .` to stage all changes
   - DO NOT commit (follow manual commit preference)
   - Provide summary of what was specified
   - Return: SUCCESS (spec ready for planning)

## Output Format
Provide a concise summary of:
- What functional requirement was added (FR-XXX)
- Key cognitive components defined
- Protocol interfaces created
- Implementation complexity assessment

## Constitution Checks

Before finalizing, verify:

1. **No Semantic Regex**: Feature doesn't rely on pattern matching for meaning
2. **XML Output**: Any LLM prompts use XML format
3. **Protocol-Based**: New integrations use Python Protocols
4. **Async-First**: Non-critical operations are async
5. **Confidence-Based**: Actions use graduated confidence levels
6. **Research-Grounded**: Feature aligns with cognitive science principles

Remember: This is for the draagon-ai cognitive AI framework. All specifications should consider the 4-layer memory architecture, belief reconciliation, and LLM-first principles.
