---
name: clarify
description: Refine and improve existing specifications
type: workflow
tools: [Read, Write, Edit, Glob, Grep]
model: claude-sonnet-4-5-20250929
---

# /clarify - Specification Refinement Tool

## Purpose
Refine and improve existing specifications by addressing gaps, ambiguities, or inconsistencies while maintaining alignment with cognitive architecture principles.

## Usage
```
/clarify [specification reference or area]
```

## Process

When this command is invoked:

1. **Read Current Specification**
   - Locate referenced specification in `.specify/`
   - Read related documents and dependencies
   - Understand current scope and status

2. **Identify Clarification Needs**
   - Find [NEEDS CLARIFICATION] markers
   - Detect ambiguous requirements
   - Identify missing cognitive considerations
   - Check for constitution violations

3. **Gather Context**
   - Review related specifications
   - Check implementation if exists
   - Consider cognitive architecture implications
   - Validate against constitution principles

4. **Refine Specification**
   - Address clarification markers
   - Add missing details
   - Ensure cognitive components are specified:
     - Memory layer interactions
     - Belief reconciliation needs
     - Learning opportunities
   - Improve testability of requirements

5. **Validate Refinement**
   - Check constitution compliance
   - Verify cognitive architecture alignment
   - Ensure requirements are testable
   - Validate consistency with related specs

6. **Update Documentation**
   - Apply refinements to specification
   - Remove resolved [NEEDS CLARIFICATION] markers
   - Update related documents if needed
   - Add clarification notes if decisions were made

7. **Stage Changes**
   - Use `git add .` to stage all changes
   - DO NOT commit (follow manual commit preference)
   - Provide summary of clarifications made

## Clarification Categories

### Cognitive Clarifications
- Which memory layer is appropriate?
- Should this create beliefs or direct memories?
- Is curiosity-driven follow-up needed?
- Does this affect agent opinions?

### Technical Clarifications
- What protocol interface is needed?
- How does this integrate with existing modules?
- What error handling is required?
- What are the performance implications?

### Behavioral Clarifications
- What confidence level triggers action?
- How should conflicts be resolved?
- What happens on failure?
- How is this tested?

## Constitution Compliance Checks

During clarification, verify:

1. **LLM-First**: No semantic regex proposed
2. **XML Output**: Any prompts use XML
3. **Protocol-Based**: New integrations use Protocols
4. **Async-First**: Non-critical ops are async
5. **Belief-Based**: User statements become observations
6. **Confidence-Based**: Graduated action thresholds

## Output Format
Provide:
- Summary of clarifications made
- Decisions documented
- Remaining ambiguities (if any)
- Impact on other specifications
- Staged files confirmation

## Example Clarification

Before:
```markdown
## FR-023: User Preference Storage
The system should store user preferences.
[NEEDS CLARIFICATION: How are preferences stored?]
```

After:
```markdown
## FR-023: User Preference Storage
The system stores user preferences through the belief system.

### Process
1. User states preference (e.g., "I prefer Celsius")
2. Statement becomes UserObservation with:
   - source_user_id: user's ID
   - scope: PERSONAL
   - confidence_expressed: 0.9 (default for explicit statements)
3. Observation reconciles into AgentBelief with:
   - belief_type: USER_PREFERENCE
   - confidence: 0.85 (adjusted by user credibility)
4. Preference stored in semantic memory layer (6-month TTL)

### Retrieval
- Query semantic memory for USER_PREFERENCE beliefs
- Filter by user_id and topic
- Return highest-confidence match
```

Remember: Clarifications should always align with the cognitive architecture. User data becomes observations, which reconcile into beliefs, which persist in appropriate memory layers.
