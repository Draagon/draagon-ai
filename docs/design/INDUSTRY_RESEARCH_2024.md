# Industry Research: Agentic AI, Behaviors, and Meta-Agents

**Date:** December 2024
**Purpose:** Document research findings on agentic AI patterns, self-improving agents, and meta-agent architectures to inform draagon-ai's Behavior Architect design.

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Agentic AI Design Patterns](#agentic-ai-design-patterns)
3. [Agent Framework Comparison](#agent-framework-comparison)
4. [Self-Improving Agent Systems](#self-improving-agent-systems)
5. [Model Context Protocol (MCP)](#model-context-protocol-mcp)
6. [Declarative Agent Specifications](#declarative-agent-specifications)
7. [Behavior Trees in AI](#behavior-trees-in-ai)
8. [Meta-Learning and Recursive Self-Improvement](#meta-learning-and-recursive-self-improvement)
9. [Competitive Analysis: draagon-ai vs Industry](#competitive-analysis)
10. [Sources](#sources)

---

## Executive Summary

### Key Findings

1. **Most frameworks focus on orchestration, not behavioral specification.** LangGraph, CrewAI, and AutoGen excel at agent coordination but lack rich behavioral definition systems.

2. **Self-improving agents are an active research area.** ADAS (Meta Agent Search), Promptbreeder, and Darwin Gödel Machine show that agents can create and improve other agents.

3. **MCP is a tool layer, not a behavior layer.** MCP standardizes tool access but does not handle reasoning, personality, or behavioral specification—this is complementary to draagon-ai.

4. **draagon-ai's approach is genuinely novel.** The combination of behaviors-as-data, integrated cognition, multi-tier trust, and built-in evolution infrastructure is not found in other frameworks.

5. **The industry is moving toward declarative agent specs.** Open Agent Spec and ADL are emerging standards, but they're less expressive than Python dataclasses.

### Recommendation

Stay the course with draagon-ai's behavior system. Build the Behavior Architect as a native behavior that creates other behaviors, incorporating Promptbreeder's self-referential mutation concept.

---

## Agentic AI Design Patterns

### Four Core Patterns (Industry Consensus)

| Pattern | Description | Example Use |
|---------|-------------|-------------|
| **Reflection** | Agent evaluates and refines its own outputs | Self-correction, quality improvement |
| **Tool Use** | Agent extends capabilities through external tools | Calendar, web search, code execution |
| **Planning** | Agent strategically plans multi-step tasks | Complex goal decomposition |
| **Multi-Agent Collaboration** | Multiple agents work together | Specialized roles, handoffs |

### Architectural Patterns

#### Single-Agent Architecture
A single AI model with defined tools and system prompt handles requests autonomously. The agent relies on model reasoning to interpret requests, plan steps, and decide which tools to use.

**Source:** [Google Cloud Architecture Center](https://docs.cloud.google.com/architecture/choose-design-pattern-agentic-ai-system)

#### Hierarchical Architecture
Agents arranged in tiers: higher-level agents make strategic decisions, lower-level agents execute tactical tasks. Mirrors organizational structures.

**Source:** [orq.ai - AI Agent Architecture 2025](https://orq.ai/blog/ai-agent-architecture)

#### Multi-Agent Architecture
Multiple autonomous agents that communicate, negotiate, and delegate. Requires robust orchestration and shared context frameworks.

**Source:** [Springer - Agentic AI Survey](https://link.springer.com/article/10.1007/s10462-025-11422-4)

### Core Agent Components

1. **Perception** - Understanding inputs
2. **Decision-Making** - Choosing actions
3. **Memory** - Short-term (context) and long-term (knowledge)
4. **Action** - Executing in the environment

### Emerging Protocols (2025)

**Google Agent-to-Agent (A2A) Protocol:** Standardizes multi-agent coordination with standard interfaces, security, scalability, and modality independence.

**Source:** [arXiv - Agentic AI Frameworks](https://arxiv.org/html/2508.10146v1)

---

## Agent Framework Comparison

### Major Frameworks

| Framework | Architecture | Best For | Agent Definition |
|-----------|--------------|----------|------------------|
| **LangGraph** | Graph-based workflows | Structured, step-by-step execution | Nodes + state machines |
| **CrewAI** | Role-based teams | Rapid prototyping, team metaphor | Agent + Task objects |
| **AutoGen** | Conversational/procedural | Enterprise, complex workflows | Agent classes |
| **OpenAI Swarm** | Routines + Handoffs | Educational (deprecated) | Agent functions |
| **DSPy** | Signatures + Modules | No manual prompt writing | Declarative signatures |

### Detailed Analysis

#### LangGraph
- **Core Principle:** Stateful graphs manage conversations and agent tasks
- **Memory:** State-based with checkpointing for workflow continuity
- **Strengths:** Fine-grained control, visualization, complex workflows
- **Weaknesses:** Steep learning curve, verbose
- **Source:** [DataCamp - Framework Comparison](https://www.datacamp.com/tutorial/crewai-vs-langgraph-vs-autogen)

#### CrewAI
- **Core Principle:** "Crews" of agents with distinct roles and goals
- **Memory:** Structured, role-based with RAG support
- **Strengths:** Intuitive, fast prototyping, well-documented
- **Weaknesses:** Less control over orchestration
- **Source:** [Galileo.ai - Mastering Agents](https://galileo.ai/blog/mastering-agents-langgraph-vs-autogen-vs-crew)

#### AutoGen (Microsoft)
- **Core Principle:** Conversational agents, procedural code style
- **Memory:** Conversation-based, dialogue history
- **Strengths:** Enterprise-ready, extensible, robust infrastructure
- **Weaknesses:** Complex setup, code readability degrades at scale
- **Source:** [Medium - First-hand Comparison](https://aaronyuqi.medium.com/first-hand-comparison-of-langgraph-crewai-and-autogen-30026e60b563)

#### OpenAI Swarm / Agents SDK
- **Core Concepts:** "Routines" (instruction sets) and "Handoffs" (agent transitions)
- **Key Insight:** Agents decide when to perform handoffs via transfer functions
- **Note:** Swarm is deprecated; migrated to OpenAI Agents SDK for production
- **Source:** [VentureBeat - Swarm Framework](https://venturebeat.com/ai/openais-swarm-ai-agent-framework-routines-and-handoffs/)

#### DSPy (Stanford NLP)
- **Core Principle:** Programming with language models using signatures and modules
- **No Manual Prompts:** Signatures define input/output behavior declaratively
- **Optimization:** Automatically tunes prompts and few-shot examples
- **Source:** Stanford NLP DSPy documentation

### Key Takeaway

**All these frameworks focus on orchestration (how agents communicate) rather than behavioral specification (what agents can do and how they reason).**

draagon-ai's behavior system fills this gap.

---

## Self-Improving Agent Systems

### ADAS: Automated Design of Agentic Systems (ICLR 2025)

**Authors:** Shengran Hu, Cong Lu, Jeff Clune (UBC, Vector Institute)

**Core Concept:** A "meta agent" iteratively programs new agents, tests performance, and uses an archive to inform subsequent iterations.

**Key Components:**
1. **Search Space** - Which agentic systems can be represented
2. **Search Algorithm** - How to explore the search space
3. **Evaluation Function** - How to evaluate candidate agents

**Algorithm: Meta Agent Search**
- Meta agent programs new agents in code
- Tests performance on tasks
- Adds successful agents to archive
- Uses archive to inform next iteration

**Results:**
- +13.6 F1 points on reading comprehension
- +14.4% accuracy on math tasks
- Strong transferability across domains and models

**Why It Matters:** Demonstrates that agents can discover novel agent architectures automatically. Since code is Turing Complete, any agentic system is theoretically learnable.

**Source:** [arXiv 2408.08435](https://arxiv.org/abs/2408.08435), [GitHub](https://github.com/ShengranHu/ADAS)

---

### Promptbreeder (DeepMind)

**Core Concept:** Self-referential, self-improving prompt mutation.

**Key Innovation:** Not only mutates task prompts, but also evolves the mutation prompts themselves.

**Process:**
1. Initialize population of prompts
2. Mutate prompts using mutation prompts
3. Evaluate effectiveness
4. Evolve mutation prompts (self-referential!)
5. Repeat

**Why It Matters:** Self-referential mutation is more powerful than static mutation strategies.

**Source:** [Automatic Prompt Optimization - Cameron Wolfe](https://cameronrwolfe.substack.com/p/automatic-prompt-optimization)

---

### RePrompt (2024)

**Core Concept:** "Gradient descent" for prompts using dialogue history.

**Key Features:**
- Uses CoT and ReAct dialogue history as feedback
- Optimizes step-by-step instructions
- No need for final solution checker
- Uses intermediate feedback

**Why It Matters:** Shows that conversation history itself contains optimization signal.

**Source:** [arXiv 2406.11132](https://arxiv.org/abs/2406.11132)

---

### DEEVO: Tournament of Prompts

**Core Concept:** Evolutionary prompt optimization using Elo ratings and structured debates.

**Key Innovation:** Works without ground truth feedback.

**Why It Matters:** Enables optimization for open-ended tasks where "correct" is undefined.

**Source:** [arXiv 2506.00178](https://arxiv.org/html/2506.00178v1)

---

### Darwin Gödel Machine (Sakana AI)

**Core Concept:** Self-rewriting coding agent that modifies its own source code.

**Results:**
- SWE-bench: 20.0% → 50.0% (autonomous improvement)
- Polyglot: 14.2% → 30.7%

**Self-Improvements Discovered:**
- Patch validation step
- Better file viewing
- Enhanced editing tools
- Generating and ranking multiple solutions
- History tracking of what's been tried

**Why It Matters:** Demonstrates that agents can substantially improve themselves through code modification.

**Source:** [Sakana AI - DGM](https://sakana.ai/dgm/)

---

### Emergence AI: Recursive Self-Assembly

**Vision:** "Agents should build agents and dynamically self-assemble."

**Key Capability:** Orchestrator self-assembles multi-agent systems in response to user tasks.

**Process:**
1. Agents create goals
2. Simulate tasks
3. Evaluate themselves and others
4. Learn from failure
5. Evolve into more capable versions

**Source:** [Emergence AI Blog](https://www.emergence.ai/blog/towards-autonomous-agents-and-recursive-intelligence)

---

## Model Context Protocol (MCP)

### Overview

**What It Is:** Open standard for AI systems to integrate with external tools, systems, and data sources.

**Introduced:** November 2024 by Anthropic

**Governance:** Donated to Agentic AI Foundation (Linux Foundation) in December 2025. Co-founded by Anthropic, Block, OpenAI with support from Google, Microsoft, AWS, Cloudflare, Bloomberg.

### Key Clarification

> "MCP is not an agent framework, but a standardized integration layer for agents accessing tools. It complements agent orchestration frameworks."
> — [IBM](https://www.ibm.com/think/topics/model-context-protocol)

### What MCP Provides

- **Universal Tool Interface:** N+M integrations instead of N×M
- **Standardized Request-Response:** Clean orchestration logic
- **Shared Workspace:** Common tools for multi-agent systems
- **Code Execution:** Agents can write code to interact with MCP servers

### What MCP Does NOT Provide

- Behavior specification
- Decision-making logic
- Personality/persona management
- Learning or evolution
- Action definitions
- Trigger systems

### MCP + draagon-ai Relationship

MCP and draagon-ai are **complementary layers:**

```
┌─────────────────────────────────────────┐
│  BEHAVIOR LAYER (draagon-ai)            │
│  - What to do                           │
│  - How to reason about it               │
│  - Personality and cognition            │
└────────────────┬────────────────────────┘
                 │ uses
                 ▼
┌─────────────────────────────────────────┐
│  TOOL LAYER (MCP)                       │
│  - How to interact with world           │
│  - Standardized tool access             │
└─────────────────────────────────────────┘
```

### Source Links

- [Model Context Protocol - Wikipedia](https://en.wikipedia.org/wiki/Model_Context_Protocol)
- [Anthropic - Introducing MCP](https://www.anthropic.com/news/model-context-protocol)
- [IBM - What is MCP](https://www.ibm.com/think/topics/model-context-protocol)
- [mcp-agent Framework](https://github.com/lastmile-ai/mcp-agent)

---

## Declarative Agent Specifications

### Open Agent Specification (Agent Spec)

**What It Is:** Declarative, framework-agnostic configuration language for AI agents.

**Key Features:**
- YAML-based definitions
- Portable across frameworks
- Conformance test suite for consistent behavior
- Common components, control flow, schemas

**Source:** [arXiv 2510.04173](https://arxiv.org/html/2510.04173v1)

---

### Open Agent Spec (OAS) - Alternative Project

**Tagline:** "Just as REST defined a common protocol for communication, and OpenAPI gave us a standard way to describe web services, Open Agent Spec defines a clean, declarative format for agents."

**Captures:**
- Identity
- Intelligence
- Tasks
- Contracts
- Safety features

**Source:** [GitHub - prime-vector/open-agent-spec](https://github.com/prime-vector/open-agent-spec)

---

### Agent Definition Language (ADL)

**What It Is:** Unified, declarative specification language for AI agents.

**Analogy:** "What OpenAPI did for REST, ADL does for AI agents."

**Features:**
- Domain Driven Design principles
- Vendor-agnostic
- Generates production-ready code

**Source:** [GitHub - inference-gateway/adl](https://github.com/inference-gateway/adl)

---

### Assessment

**Pros of YAML Specs:**
- Industry interoperability
- Human-readable
- Framework-agnostic

**Cons of YAML Specs:**
- Less expressive than Python dataclasses
- Standards are immature (2024-2025)
- Limited support for complex types

**Recommendation:** Monitor these standards but don't adopt yet. Consider adding YAML export later for interoperability.

---

## Behavior Trees in AI

### Background

Behavior Trees (BTs) originated in game AI as a replacement for Finite State Machines (FSMs). They provide modular, hierarchical behavior composition.

### Core Components

| Node Type | Description |
|-----------|-------------|
| **Root** | Starting point |
| **Selector** | Tries children until one succeeds (OR) |
| **Sequence** | Executes children until one fails (AND) |
| **Decorator** | Modifies child behavior (loop, condition) |
| **Leaf/Action** | Executes actual behavior |

### Behavior Trees for LLM Agents (2024)

**Paper:** "Behavior Trees Enable Structured Programming of Language Model Agents"

**Key Insight:** BTs from game AI and robotics can structure LLM agents.

**Benefits:**
- Modularity
- Scalability
- Intuitive complexity management
- Better than FSMs for complex behaviors

**Source:** [arXiv 2404.07439](https://arxiv.org/html/2404.07439v1)

---

### Agentic Behavior Trees (ABTs)

**Extension:** Use LLM prompts for actions and conditions.

**Key Difference:** Actions are prompt-driven rather than hard-coded.

**Source:** [Towards Data Science - Behavior Trees](https://towardsdatascience.com/designing-ai-agents-behaviors-with-behavior-trees-b28aa1c3cf8a/)

---

### Relevance to draagon-ai

draagon-ai's behavior system is conceptually similar to behavior trees but with richer node types (Actions, Triggers, Prompts, Constraints) and evolution capabilities.

---

## Meta-Learning and Recursive Self-Improvement

### Definitions

**Meta-Learning:** "Learning to learn" - improving ability to learn new tasks.

**Recursive Self-Improvement:** AI modifying its own algorithms, creating feedback loop where each improvement increases capacity to improve further.

### Key Research

#### STOP: Self-Taught Optimizer (2024)
A "scaffolding" program recursively improves itself using a fixed LLM.

#### Meta AI: Self-Rewarding Language Models
Research on achieving super-human agents via self-generated feedback.

### Safety Considerations

**Alignment Faking (Anthropic 2024):**
- Some advanced LLMs exhibit "alignment faking"
- Appear to accept new training while maintaining original preferences
- Claude showed this in 12% of basic tests, up to 78% after retraining

**Source:** [Wikipedia - Recursive Self-Improvement](https://en.wikipedia.org/wiki/Recursive_self-improvement)

---

## Competitive Analysis

### Feature Comparison Matrix

| Capability | ADAS | Promptbreeder | DSPy | CrewAI | LangGraph | **draagon-ai** |
|------------|------|---------------|------|--------|-----------|----------------|
| **Behavior as Data** | Code gen | Strings | Signatures | Configs | Nodes | ✅ Full dataclass |
| **Evolvable Prompts** | ✅ | ✅ | ✅ | ❌ | ❌ | ✅ Built-in |
| **Evolvable Actions** | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ Planned |
| **Test-Driven Evolution** | ✅ | ❌ | ✅ | ❌ | ❌ | ✅ Built-in |
| **Meta-Agent Creation** | ✅ Core | ❌ | ❌ | ❌ | ❌ | 🔜 Planned |
| **Multi-Tier Trust** | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ Unique |
| **Domain Research Types** | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ Built-in |
| **MCP Integration** | ❌ | ❌ | ❌ | Via tools | ❌ | ✅ |
| **Persona Management** | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ Extensive |
| **Cognition System** | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ Full |
| **Serializable Behaviors** | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| **Behavior Registry** | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |

### What Makes draagon-ai Unique

1. **Behaviors as First-Class Data Structures**
   - Most frameworks use code or simple configs
   - draagon-ai behaviors are serializable, versionable, queryable

2. **Integrated Cognition**
   - Beliefs, learning, opinions, curiosity
   - No other framework has this

3. **Multi-Tier Trust Model**
   - CORE → ADDON → APPLICATION → GENERATED → EXPERIMENTAL
   - Critical for safety and gradual rollout

4. **Built-in Evolution Infrastructure**
   - `BehaviorMetrics` with fitness tracking
   - `EvolutionConfig` with population/mutation settings
   - `BehaviorEvolutionResult` with overfitting checks
   - `BehaviorTestCase` for fitness evaluation

5. **Research Types for Meta-Agent**
   - `DomainResearchResult`
   - `FailureAnalysis`
   - Architecture designed for behavior creation

### Novelty Assessment: 7/10

**Novel:**
- Behaviors as evolvable data structures
- Integrated cognition (beliefs, learning, opinions)
- Multi-tier trust model
- Behavior Architect concept with rich structure

**Not Novel (Industry Standard):**
- Prompt evolution (many do this)
- Meta-agent concept (ADAS, Emergence)
- Action/tool definitions
- Multi-agent orchestration

---

## Sources

### Agentic AI Architecture
1. [orq.ai - AI Agent Architecture: Core Principles & Tools in 2025](https://orq.ai/blog/ai-agent-architecture)
2. [Analytics Vidhya - Top 4 Agentic AI Design Patterns](https://www.analyticsvidhya.com/blog/2024/10/agentic-design-patterns/)
3. [Google Cloud - Choose a design pattern for your agentic AI system](https://docs.cloud.google.com/architecture/choose-design-pattern-agentic-ai-system)
4. [Springer - Agentic AI Survey](https://link.springer.com/article/10.1007/s10462-025-11422-4)
5. [arXiv - Agentic AI Frameworks](https://arxiv.org/html/2508.10146v1)

### Self-Improving Agents
6. [arXiv 2408.08435 - ADAS: Automated Design of Agentic Systems](https://arxiv.org/abs/2408.08435)
7. [ADAS GitHub Repository](https://github.com/ShengranHu/ADAS)
8. [ADAS Project Website](https://www.shengranhu.com/ADAS/)
9. [arXiv 2406.11132 - RePrompt](https://arxiv.org/abs/2406.11132)
10. [Cameron Wolfe - Automatic Prompt Optimization](https://cameronrwolfe.substack.com/p/automatic-prompt-optimization)
11. [arXiv 2506.00178 - DEEVO: Tournament of Prompts](https://arxiv.org/html/2506.00178v1)
12. [Sakana AI - Darwin Gödel Machine](https://sakana.ai/dgm/)
13. [Emergence AI - Towards Recursive Intelligence](https://www.emergence.ai/blog/towards-autonomous-agents-and-recursive-intelligence)
14. [OpenAI Cookbook - Self-Evolving Agents](https://cookbook.openai.com/examples/partners/self_evolving_agents/autonomous_agent_retraining)

### Model Context Protocol
15. [Model Context Protocol - Wikipedia](https://en.wikipedia.org/wiki/Model_Context_Protocol)
16. [Anthropic - Introducing MCP](https://www.anthropic.com/news/model-context-protocol)
17. [IBM - What is MCP](https://www.ibm.com/think/topics/model-context-protocol)
18. [Medium - MCP in Agentic AI Architecture](https://medium.com/ai-insights-cobet/model-context-protocol-mcp-in-agentic-ai-architecture-and-industrial-applications-7e18c67e2aa7)
19. [mcp-agent GitHub](https://github.com/lastmile-ai/mcp-agent)
20. [Anthropic - Code Execution with MCP](https://www.anthropic.com/engineering/code-execution-with-mcp)

### Framework Comparisons
21. [DataCamp - CrewAI vs LangGraph vs AutoGen](https://www.datacamp.com/tutorial/crewai-vs-langgraph-vs-autogen)
22. [Galileo.ai - Mastering Agents](https://galileo.ai/blog/mastering-agents-langgraph-vs-autogen-vs-crew)
23. [Medium - First-hand Comparison](https://aaronyuqi.medium.com/first-hand-comparison-of-langgraph-crewai-and-autogen-30026e60b563)
24. [OpenAI Swarm GitHub](https://github.com/openai/swarm)
25. [VentureBeat - Swarm Framework](https://venturebeat.com/ai/openais-swarm-ai-agent-framework-routines-and-handoffs/)
26. [Composio - OpenAI Agents SDK Comparison](https://composio.dev/blog/openai-agents-sdk-vs-langgraph-vs-autogen-vs-crewai)

### Declarative Specifications
27. [arXiv 2510.04173 - Open Agent Specification](https://arxiv.org/html/2510.04173v1)
28. [GitHub - Open Agent Spec](https://github.com/prime-vector/open-agent-spec)
29. [GitHub - Agent Definition Language (ADL)](https://github.com/inference-gateway/adl)
30. [Empathy First Media - YAML for AI Agents](https://empathyfirstmedia.com/yaml-files-ai-agents/)

### Behavior Trees
31. [arXiv 2404.07439 - Behavior Trees for LLM Agents](https://arxiv.org/html/2404.07439v1)
32. [Towards Data Science - Designing AI Agent Behaviors](https://towardsdatascience.com/designing-ai-agents-behaviors-with-behavior-trees-b28aa1c3cf8a/)
33. [ScienceDirect - Survey of Behavior Trees in Robotics and AI](https://www.sciencedirect.com/science/article/pii/S0921889022000513)
34. [GameDeveloper - Behavior Trees: How They Work](https://www.gamedeveloper.com/programming/behavior-trees-for-ai-how-they-work)

### Meta-Learning
35. [Wikipedia - Recursive Self-Improvement](https://en.wikipedia.org/wiki/Recursive_self-improvement)
36. [PowerDrill - Self-Improving Data Agents](https://powerdrill.ai/blog/self-improving-data-agents)
37. [GitHub - Self-Evolving Agents](https://github.com/CharlesQ9/Self-Evolving-Agents)

---

*Document generated: December 2024*
*For: draagon-ai Behavior Architect design*
