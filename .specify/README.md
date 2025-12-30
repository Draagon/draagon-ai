# draagon-ai Specification

This specification defines the complete requirements, architecture, and implementation details for draagon-ai, an agentic AI framework for building cognitive assistants using **GitHub Spec Kit** methodology.

## Quick Start

1. Read the [Constitution](constitution/constitution.md) first
2. Review [Design Principles](constitution/principles.md) and [Testing Principles](constitution/testing-principles.md)
3. Understand [Project Scope](constitution/scope.md)
4. Browse specifications by priority
5. Review implementation plans and task breakdowns

## Spec Kit Compliance

This specification follows **GitHub Spec Kit** principles:
- **Constitution First**: Non-negotiable principles and constraints
- **Specification**: What and why before how
- **Planning**: Technical implementation strategy
- **Tasks**: Actionable AI-consumable work units
- **Research-Grounded**: Every decision backed by research

## Directory Structure

```
.specify/                    # Spec Kit root directory
├── constitution/            # Project governance
│   ├── constitution.md     # Project charter and principles
│   ├── principles.md       # Design principles (LLM-first, 4-layer memory, etc.)
│   ├── scope.md            # Project boundaries
│   └── testing-principles.md # Testing philosophy
├── requirements/            # Individual functional requirements (FR-*.md files)
├── specification/           # Core specifications
│   ├── non-functional-requirements.md  # Performance and quality
│   ├── cognitive-architecture.md       # 4-layer memory, beliefs, etc.
│   └── multi-agent-spec.md             # Parallel orchestration
├── planning/                # How we're building it
│   ├── technical-architecture.md       # System design
│   ├── COGNITIVE_SWARM_ARCHITECTURE.md # Phase 2 detailed spec
│   └── implementation-strategy.md      # Development approach
├── tasks/                   # Phase-organized task files
│   ├── phase-0/            # Core framework tasks
│   ├── phase-1/            # Cognitive integration tasks
│   └── phase-2/            # Cognitive swarm tasks
├── scenarios/               # Test scenarios
│   ├── integration/        # Component interaction tests
│   ├── unit/               # Unit test specs
│   └── stress/             # Load and concurrency tests
├── memory/                  # Key project artifacts
├── analysis/                # Research and analysis docs
├── archive/                 # Legacy files
└── spec.yaml               # Project metadata
```

## Key Documents by Role

### For Framework Developers
1. **Start Here**: `constitution/constitution.md`
2. **Principles**: `constitution/principles.md`
3. **Cognitive Spec**: `../docs/specs/COGNITIVE_SWARM_ARCHITECTURE.md`
4. **Testing**: `constitution/testing-principles.md`

### For Application Developers
1. **Scope**: `constitution/scope.md` (what's provided vs. not)
2. **Reference**: `../roxy-voice-assistant/` (production example)
3. **API**: `../src/draagon_ai/` (source code)

### For Researchers
1. **Cognitive Architecture**: `constitution/principles.md`
2. **Research Basis**: Links in specification documents
3. **Benchmarks**: `../docs/specs/COGNITIVE_SWARM_ARCHITECTURE.md#benchmark-targets`

## Core Differentiators

What makes draagon-ai unique:

| Capability | Other Frameworks | draagon-ai |
|------------|-----------------|------------|
| Memory | Flat / retrieval-based | 4-layer cognitive with promotion |
| Beliefs | None | Full observation → belief reconciliation |
| Opinions | None | Authentic formation with evolution |
| Curiosity | None | Knowledge gap detection + question queuing |
| Multi-Agent | Shared dict | Cognitive working memory with attention |
| Transactive | None | "Who knows what" expertise routing |

## Research Foundation

Every major feature is backed by research:

- **4-Layer Memory**: Baddeley's Working Memory Model
- **7±2 Capacity**: Miller's Law (1956)
- **Belief Reconciliation**: Epistemic Logic, BDI Agents
- **Transactive Memory**: Wegner (1987)
- **Metacognitive Reflection**: ICML 2025 Position Paper
- **Parallel Coordination**: Anthropic Multi-Agent Research

## Development Phases

### Phase 0: Core Framework (Completed)
- ReAct loop implementation
- Decision engine
- Action executor
- Tool registry
- 4-layer memory architecture
- Sequential multi-agent orchestration

### Phase 1: Cognitive Integration (In Progress)
- @tool decorator with discovery
- Cognitive services wiring
- Post-response learning
- Belief-aware decisions

### Phase 2: Cognitive Swarm (Planned)
- Shared Cognitive Working Memory
- Parallel Execution with Coordination
- Multi-Agent Belief Reconciliation
- Transactive Memory System
- Metacognitive Reflection

### Phase 3: Production Hardening (Future)
- Comprehensive test suite
- Performance optimization
- PyPI distribution
- Complete documentation

## Success Metrics

### Phase 1 Goals
- Cognitive services integrated into ReAct loop
- Post-response learning extraction working
- Belief-aware decisions functional

### Phase 2 Goals (Benchmarks)
| Benchmark | Current SOTA | Our Target |
|-----------|-------------|------------|
| MultiAgentBench Werewolf | 36.33% | 55%+ |
| MemoryAgentBench Conflict | ~50% | 75%+ |
| GAIA Level 3 | ~40% | 55%+ |

## Getting Started

1. **Review Constitution**: Understand non-negotiable principles
2. **Study Architecture**: Review design principles
3. **Check Scope**: Understand what's provided vs. external
4. **Browse Code**: `../src/draagon_ai/`
5. **See Reference**: `../roxy-voice-assistant/`

## Slash Commands

Use `.claude/commands/` for spec-driven development:
- `/specify` - Create new specifications
- `/plan` - Generate implementation plans
- `/tasks` - Break down into actionable tasks
- `/implement` - Execute implementation
- `/review` - Analyze specifications

---

**Document Status**: Active
**Last Updated**: 2025-12-30
**Spec Kit Version**: 1.0 compatible
