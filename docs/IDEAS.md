# Draagon-AI: Product Ideas & Research

> **Last Updated**: 2025-12-28
> **Status**: Active research and planning document

---

## Table of Contents

1. [VS Code Extensions & Development Agents](#vs-code-extensions--development-agents)
2. [Commercialization Strategy](#commercialization-strategy)
3. [HIPAA-Compliant Enterprise Deployment](#hipaa-compliant-enterprise-deployment)
4. [Identity, Personas & Configuration Model](#identity-personas--configuration-model)
5. [Technical Architecture](#technical-architecture)
6. [Implementation Roadmap](#implementation-roadmap)

---

## Executive Summary

**Core Insight**: The "goldfish memory problem" in AI coding assistants is real and unsolved. Draagon-ai's cognitive architecture (beliefs, curiosity, learning, memory consolidation) is genuinely differentiated.

**Key Decision (from research)**: Use **hierarchical configuration + feature flags**, NOT personas. Industry leaders (GitHub Copilot, Glean, Moveworks) all use contextual customization over personality.

**Architecture**:
```
┌─────────────────────────────────────────────────────────────────┐
│                     CONFIGURATION HIERARCHY                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Enterprise (enforced policies, cannot override)               │
│       │                                                          │
│       ▼                                                          │
│   Organization (company-wide defaults)                          │
│       │                                                          │
│       ▼                                                          │
│   Division/Department (optional layer)                          │
│       │                                                          │
│       ▼                                                          │
│   Team/Project (.draagon/instructions.md)                       │
│       │                                                          │
│       ▼                                                          │
│   Role (engineer, manager, security)                            │
│       │                                                          │
│       ▼                                                          │
│   Individual (personal preferences, synced from personal Roxy)  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## VS Code Extensions & Development Agents

### The Opportunity

| Gap in Market | Draagon-ai Solution |
|---------------|---------------------|
| Static CLAUDE.md files | Dynamic, queryable memory via MCP |
| No persistent memory | Layered memory with consolidation |
| No belief tracking | BeliefReconciliationService |
| No knowledge gap detection | CuriosityEngine |
| Per-user only | Team + Org shared memory |

### Tier 1: High-Impact Extensions

#### 1. Draagon Memory MCP Server

Replace static CLAUDE.md with living, queryable project memory.

```
┌─────────────────────────────────────────────────────────────────┐
│ MCP Server Tools:                                                │
│ - memory_store(content, scope, type)                            │
│ - memory_search(query, scope, limit)                            │
│ - memory_consolidate()                                          │
│ - beliefs_query(topic)                                          │
│ - beliefs_reconcile(claim, source)                              │
│                                                                  │
│ MCP Resources:                                                   │
│ - project://context                                              │
│ - project://architecture                                         │
│ - project://conventions                                          │
│ - team://decisions                                               │
│ - user://preferences                                             │
│                                                                  │
│ MCP Prompts:                                                     │
│ - project-onboarding                                             │
│ - code-review-with-memory                                        │
│ - decision-with-history                                          │
└─────────────────────────────────────────────────────────────────┘
```

**LLM Tier Strategy**:
- Groq 70B: Memory ops, pattern matching (fast, cheap)
- Ollama 8B: Offline fallback (free)
- Claude: Complex reasoning, final reviews (quality)

#### 2. Curiosity-Driven Codebase Explorer

Proactively identifies knowledge gaps and asks questions at appropriate times.

**Learns things like**:
- "In this project, we never use default exports"
- "Error codes above 5000 are reserved for third-party integrations"
- "The /v2 API was started but abandoned, don't use it"

#### 3. Belief-Based Code Reviewer

Builds beliefs about code patterns, detects contradictions in new code.

**Belief Types**:
- `ARCHITECTURAL_PATTERN`: "API routes return {success, data, error}"
- `CONVENTION`: "Test files co-located as *.test.ts"
- `DEPRECATION`: "Don't use LegacyDateFormatter"

### Tier 2: Advanced Agents

| Agent | Purpose | Effort |
|-------|---------|--------|
| Architecture Guardian | Living architecture docs in temporal graph | High |
| Test Oracle | Learns testing patterns, generates suggestions | Medium |
| Dependency Advisor | Tracks package decisions and rationale | Low-Medium |
| Pair Programming Memory | Session context that consolidates | Low-Medium |

---

## Commercialization Strategy

### Market Reality

- **Market size**: $5.5B (2024) → $47.3B (2034), 24% CAGR
- **Competition**: GitHub Copilot (40% share), Cursor, Augment
- **Gap**: No competitor has belief reconciliation, curiosity engines, or team knowledge sync

### Positioning

**Wrong**: "Another AI coding assistant"
**Right**: "Cognitive layer for development teams" or "AI that actually remembers"

### Pricing Tiers

| Tier | Price | Key Features |
|------|-------|--------------|
| **Free/OSS** | $0 | Self-hosted, single user, full cognitive services |
| **Pro** | $15/mo | Hosted, Groq included, cross-device sync |
| **Team** | $25/user/mo | **Shared team memory**, SSO, admin dashboard |
| **Enterprise** | $40+/user/mo | SOC 2, self-hosted option, dedicated instance |

### Revenue Projections (Conservative)

| Milestone | Monthly Revenue | ARR |
|-----------|-----------------|-----|
| Month 6 | $750 | - |
| Month 12 | $4,250 | ~$50K |
| Year 2 | $16,500 | ~$200K |

---

## HIPAA-Compliant Enterprise Deployment

### Core Constraint

```
Personal Roxy (home) ──► CareMetx Instance (AWS)
     │                         │
     │ Push preferences        │ NEVER sends data back
     │ (non-PHI only)          │
     ▼                         ▼
   Allowed                   Blocked
```

### AWS Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    AWS VPC (HIPAA-eligible)                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ALB (WebSocket) ──► ECS Fargate (Draagon) ──► RDS PostgreSQL  │
│                              │                                   │
│           ┌──────────────────┼──────────────────┐               │
│           ▼                  ▼                  ▼               │
│       Qdrant            ElastiCache         Bedrock             │
│       (EFS)             (Redis)             (Claude, Llama)     │
│                                                                  │
│   CloudWatch (HIPAA audit logs)                                 │
│   Secrets Manager (mTLS certs, API keys)                        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Preference Sync Protocol

**What CAN be synced** (not PHI):
- Coding style preferences
- Communication preferences
- Tool preferences
- General beliefs about code quality

**What CANNOT be synced**:
- Work memories, code, decisions
- Project-specific context
- Team member information

---

## Identity, Personas & Configuration Model

### Key Research Finding

> **Industry consensus**: Use **contextual customization**, NOT personas.
>
> GitHub Copilot, Glean, Moveworks all use settings/instructions, not personality.
> Research shows **perceived usefulness** matters more than personality matching.

### The Decision: Configuration Over Personas

| Approach | Use | Don't Use |
|----------|-----|-----------|
| **Terminology** | "instructions", "guidelines", "preferences", "standards" | "persona", "personality", "character" |
| **Customization** | Feature flags, settings inheritance | Distinct AI "personalities" |
| **Identity** | Professional, neutral, consistent | Multiple "characters" users can choose |

### Hierarchical Configuration Model

```yaml
# Enterprise Level (enforced, cannot override)
enterprise:
  security:
    audit_logging: required
    data_retention: 90_days
    no_external_llm_calls: true
  compliance:
    hipaa: true
    pci: false

# Organization Level (company defaults)
organization:
  name: "CareMetx"
  branding:
    assistant_name: "Atlas"  # Optional custom name
    tone: "professional"     # professional | friendly | concise
  coding_standards:
    language_preferences: ["python", "typescript"]
    testing_required: true
    documentation_style: "google"

  # Feature flags
  features:
    curiosity_engine: true
    belief_reconciliation: true
    autonomous_refactoring: false  # Disabled at org level
    code_generation: true

# Division/Department Level (optional)
division:
  name: "Engineering"
  features:
    autonomous_refactoring: true  # Override org setting

# Team/Project Level (.draagon/instructions.md)
team:
  name: "Patient Services"
  instructions: |
    - Use async/await for all I/O operations
    - All database operations must use the repository pattern
    - HIPAA audit logging required for PHI access

  # Team-specific features
  features:
    pr_review_automation: true

# Role Level
role:
  name: "Senior Engineer"
  capabilities:
    - approve_autonomous_changes
    - modify_team_instructions
    - view_all_team_memories

# Individual Level (from personal Roxy sync + local prefs)
individual:
  user_id: "doug"
  preferences:
    coding_style:
      indent: 4
      quotes: "double"
    communication:
      verbosity: "concise"
      explanation_depth: "moderate"

  # Personal preferences synced from home Roxy
  synced_preferences:
    general_beliefs:
      - "prefer composition over inheritance"
      - "explicit is better than implicit"
    tool_preferences:
      editor: "vscode"
      shell: "zsh"
```

### Configuration Inheritance Rules

```
┌─────────────────────────────────────────────────────────────────┐
│                     INHERITANCE RULES                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│ 1. ENFORCED settings (enterprise level) CANNOT be overridden   │
│                                                                  │
│ 2. Default settings INHERIT from parent, can be overridden     │
│                                                                  │
│ 3. ADDITIVE settings (like instructions) MERGE with parent     │
│                                                                  │
│ 4. Feature flags can be ENABLED at lower levels if parent      │
│    allows, but cannot be enabled if parent disables            │
│                                                                  │
│ Example:                                                         │
│   Enterprise: autonomous_refactoring = "allowed"                │
│   Org:        autonomous_refactoring = false (disabled)         │
│   Division:   autonomous_refactoring = true  (re-enabled)       │
│   Team:       autonomous_refactoring = true  (inherited)        │
│   User:       cannot override (team setting applies)            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### What This Means for Your Questions

**Q: Will the work version have a personality like Roxy?**

A: No, and it shouldn't. Research shows enterprise AI assistants should be **professional, neutral, and consistent**. The work version will have:
- Consistent, trustworthy behavior
- Contextual awareness (knows your role, team, project)
- Your personal preferences (synced from home Roxy)
- Team and org knowledge

But NOT:
- A distinct "personality"
- Character traits like Roxy's curiosity or warmth
- Multiple personas users can switch between

**Q: Should team members control the personality?**

A: They shouldn't control "personality" at all. Instead:
- **Enterprise/Org** controls: Branding (name, tone), security policies
- **Team** controls: Coding standards, instructions, conventions
- **Individual** controls: Personal preferences (indent, verbosity)

**Q: What about the hierarchy (company, org, division, role, team, individual)?**

A: Yes, exactly. But it's **settings and feature flags**, not personas:

```
New Team Member Gets:
├── Enterprise policies (security, compliance)
├── Org defaults (coding standards, features)
├── Division settings (if applicable)
├── Team instructions (.draagon/instructions.md)
├── Role capabilities (what they can do)
└── Empty individual preferences (builds over time)
```

### Feature Flag Examples

```yaml
feature_flags:
  # Cognitive Services
  curiosity_engine:
    default: true
    description: "AI asks clarifying questions about codebase"
    controllable_at: [org, team]

  belief_reconciliation:
    default: true
    description: "Detects and resolves conflicting information"
    controllable_at: [org, team]

  autonomous_refactoring:
    default: false
    description: "AI can propose refactoring PRs autonomously"
    controllable_at: [org, division, team]
    requires_role: ["senior_engineer", "tech_lead"]

  # Memory Features
  team_memory_sharing:
    default: true
    description: "Learnings shared across team members"
    controllable_at: [org, team]

  cross_team_search:
    default: false
    description: "Search memories from other teams"
    controllable_at: [org]

  # Integration Features
  pr_auto_review:
    default: false
    description: "Automatically review PRs with beliefs"
    controllable_at: [org, team]

  slack_notifications:
    default: false
    description: "Send curiosity questions to Slack"
    controllable_at: [team]
```

### Single Identity, Multiple Contexts

Instead of personas, the AI has ONE consistent identity that adapts to context:

```
┌─────────────────────────────────────────────────────────────────┐
│                     SINGLE AI IDENTITY                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Core Traits (always present):                                 │
│   - Professional                                                 │
│   - Accurate                                                     │
│   - Helpful                                                      │
│   - Respects hierarchy and policies                             │
│                                                                  │
│   Contextual Adaptation:                                         │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │ At CareMetx (work):                                      │   │
│   │ - Knows HIPAA requirements                               │   │
│   │ - Uses team coding conventions                           │   │
│   │ - Applies Doug's synced preferences                      │   │
│   │ - Formal tone (org setting)                              │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │ Personal Roxy (home):                                    │   │
│   │ - Knows household context                                │   │
│   │ - Casual tone                                            │   │
│   │ - Full curiosity/personality features                    │   │
│   │ - Home automation access                                 │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   SAME underlying cognitive architecture, DIFFERENT context.   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Technical Architecture

### Multi-Tenant Data Model

```python
# Configuration hierarchy in database

class Enterprise(Base):
    id: UUID
    name: str
    enforced_policies: dict    # Cannot be overridden
    default_settings: dict     # Can be overridden
    feature_flags: dict        # Enterprise-level flags

class Organization(Base):
    id: UUID
    enterprise_id: UUID
    name: str
    branding: dict            # assistant_name, tone
    settings: dict            # Overrides enterprise defaults
    feature_flags: dict       # Org-level flags
    qdrant_collection: str    # Dedicated collection

class Team(Base):
    id: UUID
    organization_id: UUID
    name: str
    instructions: str         # .draagon/instructions.md content
    settings: dict            # Team-specific settings
    feature_flags: dict       # Team-level flags

class Role(Base):
    id: UUID
    organization_id: UUID
    name: str                 # "engineer", "senior_engineer", "manager"
    capabilities: list[str]   # What this role can do
    feature_access: list[str] # What features this role can use

class User(Base):
    id: UUID
    organization_id: UUID
    team_ids: list[UUID]
    role_id: UUID
    preferences: dict         # Personal preferences
    synced_preferences: dict  # From personal Roxy
    last_sync: datetime

# Compute effective configuration
def get_effective_config(user: User) -> EffectiveConfig:
    """Merge all levels of configuration for a user."""
    enterprise = get_enterprise(user.organization.enterprise_id)
    org = get_organization(user.organization_id)
    team = get_primary_team(user)
    role = get_role(user.role_id)

    return EffectiveConfig(
        # Policies always from enterprise (enforced)
        policies=enterprise.enforced_policies,

        # Settings merge with override
        settings=merge_with_override(
            enterprise.default_settings,
            org.settings,
            team.settings,
            user.preferences,
        ),

        # Feature flags with hierarchy rules
        features=compute_feature_flags(
            enterprise.feature_flags,
            org.feature_flags,
            team.feature_flags,
            role.feature_access,
        ),

        # Instructions are additive
        instructions=merge_instructions(
            org.instructions,
            team.instructions,
        ),

        # Personal preferences (lowest priority but additive)
        personal=user.synced_preferences,
    )
```

### MCP Server with Context Awareness

```python
@server.tool("memory_store")
async def store_memory(
    content: str,
    scope: str,          # "individual" | "team" | "org" | "enterprise"
    memory_type: str,
    ctx: RequestContext,
) -> dict:
    """Store memory respecting hierarchy."""
    user = ctx.user
    config = get_effective_config(user)

    # Check if user can store at this scope
    if scope == "org" and "modify_org_memory" not in config.capabilities:
        raise PermissionError("Cannot store org-level memories")

    # Check if feature is enabled
    if not config.features.get("team_memory_sharing") and scope == "team":
        raise FeatureDisabled("Team memory sharing is disabled")

    return await memory_provider.store(
        content=content,
        scope=MemoryScope[scope.upper()],
        memory_type=MemoryType[memory_type.upper()],
        user_id=user.id,
        team_id=user.primary_team_id if scope in ["team", "individual"] else None,
        org_id=user.organization_id,
    )
```

---

## Implementation Roadmap

### Phase 1: Core (Weeks 1-4)
- [ ] MCP Memory Server with basic scopes
- [ ] Bedrock integration for HIPAA
- [ ] Basic configuration hierarchy (org → team → user)
- [ ] CircleCI + Terraform deployment

### Phase 2: Memory & Beliefs (Weeks 5-8)
- [ ] Full configuration hierarchy
- [ ] Feature flag system
- [ ] Team memory sharing
- [ ] Belief system for code patterns
- [ ] HIPAA audit logging

### Phase 3: Personal Integration (Weeks 9-10)
- [ ] Preference sync API
- [ ] mTLS authentication
- [ ] Daily sync automation
- [ ] Instructions file support (.draagon/)

### Phase 4: Team Rollout (Weeks 11-12)
- [ ] Onboard CareMetx team
- [ ] Seed with company conventions
- [ ] Role-based access control
- [ ] Admin dashboard (basic)

---

## Open Questions

1. **Branding**: Should each org be able to name their assistant? (e.g., "Atlas" for CareMetx)
   - Recommendation: Yes, but it's just a name, not a personality change

2. **Tone settings**: How many options?
   - Recommendation: 3 options: `professional` | `friendly` | `concise`

3. **Cross-team memory**: Should teams be able to see other teams' memories?
   - Recommendation: Feature flag, default off, org controls

4. **Role hierarchy**: How granular should roles be?
   - Recommendation: Start simple: `engineer`, `senior_engineer`, `tech_lead`, `manager`

---

---

## Enterprise Self-Hosted Deployment Strategy

### Key Research Finding

**Industry consensus**: Docker Compose + Helm Charts are the universal standard. 82% of SaaS vendors now support self-hosted deployments.

### What Other Companies Do

| Company | Docker Compose | Helm/K8s | Notes |
|---------|---------------|----------|-------|
| **GitLab** | Yes (Omnibus) | Yes (Official) | Cloud Native Hybrid for production |
| **Supabase** | Yes (Primary) | Community | Docker Compose is main method |
| **PostHog** | Yes (Only) | Sunset 2024 | Dropped K8s, too complex |
| **Sourcegraph** | Yes (Preferred) | Yes | Docker for single-node |
| **Sentry** | Yes (Official) | Not recommended | Docker on dedicated VM |
| **n8n** | Yes | Yes | Worker mode for scale |
| **Airbyte** | Deprecated | Required | K8s mandatory for Enterprise |
| **Mattermost** | Dev only | Yes (Operator) | K8s required for HA |

### Universal Deployment Stack

```
┌─────────────────────────────────────────────────────────────────┐
│                    DEPLOYMENT TIERS                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   TIER 1: MVP (Required)                                        │
│   ─────────────────────                                         │
│   • Docker Compose                                               │
│   • One-liner install: docker compose up -d                     │
│   • Environment-based config                                     │
│   • Health check endpoints                                       │
│   → Target: Small teams, POCs, development                      │
│                                                                  │
│   TIER 2: Production (Expected)                                  │
│   ─────────────────────────────                                  │
│   • Helm Chart for Kubernetes                                    │
│   • Terraform modules (AWS, GCP, Azure)                         │
│   • External database support                                    │
│   • TLS/HTTPS configuration                                      │
│   • Horizontal scaling                                           │
│   → Target: Production deployments, mid-size companies          │
│                                                                  │
│   TIER 3: Enterprise (Differentiator)                            │
│   ─────────────────────────────────────                          │
│   • Air-gap bundle (offline install)                            │
│   • License key system (Keygen or custom)                       │
│   • SSO/SAML/LDAP integration                                   │
│   • Audit logging                                                │
│   • Cloud Marketplace listings (AWS, Azure, GCP)                │
│   → Target: Regulated industries, large enterprises             │
│                                                                  │
│   TIER 4: Advanced (Future)                                      │
│   ───────────────────────                                        │
│   • Kubernetes Operator                                          │
│   • BYOC (Bring Your Own Cloud)                                 │
│   • Multi-region support                                         │
│   → Target: Hyperscale, complex requirements                    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### The Docker Consistency Model

**One image, many environments** - Docker provides consistency across all deployment targets:

```
┌─────────────────────────────────────────────────────────────────┐
│                 SAME DOCKER IMAGES EVERYWHERE                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│   │ Local Dev       │  │ Railway (SaaS)  │  │ Customer        │ │
│   │ (pop-os-desktop)│  │                 │  │ Self-Hosted     │ │
│   └────────┬────────┘  └────────┬────────┘  └────────┬────────┘ │
│            │                    │                    │          │
│            └────────────────────┼────────────────────┘          │
│                                 │                               │
│                                 ▼                               │
│                    ┌─────────────────────┐                      │
│                    │  Same Docker Images │                      │
│                    │  ─────────────────  │                      │
│                    │  draagon-api:1.0.0  │                      │
│                    │  qdrant:v1.12.0     │                      │
│                    │  postgres:16        │                      │
│                    │  redis:7            │                      │
│                    └─────────────────────┘                      │
│                                 │                               │
│            ┌────────────────────┼────────────────────┐          │
│            │                    │                    │          │
│            ▼                    ▼                    ▼          │
│   ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│   │ .env.local      │  │ .env.railway    │  │ .env.customer   │ │
│   │ LLM=ollama      │  │ LLM=groq        │  │ LLM=bedrock     │ │
│   │ LICENSE=dev     │  │ LICENSE=saas    │  │ LICENSE=ent_xxx │ │
│   └─────────────────┘  └─────────────────┘  └─────────────────┘ │
│                                                                  │
│   "Works on my machine" = "Works everywhere"                    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

#### Environment Matrix

| Environment | Docker Command | LLM Provider | License | Use Case |
|-------------|---------------|--------------|---------|----------|
| **Local (Doug)** | `docker compose up` | Ollama (local) | `dev-unlimited` | Personal Roxy |
| **Local (dev)** | `docker compose up` | Groq | `dev-unlimited` | Development/testing |
| **CareMetx** | Same compose on ECS | Bedrock | `ent_caremetx_xxx` | HIPAA enterprise |
| **Railway** | Railway deploys images | Groq | SaaS (no key) | Pro/Team customers |
| **Customer A** | `docker compose up` | Groq | `team_acme_xxx` | Self-hosted Team |
| **Customer B** | Same on their K8s | Their Bedrock | `ent_bigco_xxx` | Self-hosted Enterprise |

#### Environment Configuration Files

```bash
# Directory structure
deploy/
├── docker-compose.yml          # Base compose (all services)
├── docker-compose.override.yml # Local development defaults
├── .env.example                # Template for customers
├── .env.local                  # Your local development
├── .env.caremetx               # CareMetx production (gitignored)
└── environments/
    ├── local.env
    ├── railway.env
    └── enterprise.env.example
```

```bash
# .env.example (what customers get)
# Draagon AI Configuration
# Copy to .env and fill in your values

# === Required ===
LICENSE_KEY=your_license_key_here
DB_PASSWORD=generate_a_secure_password

# === LLM Provider (choose one) ===
LLM_PROVIDER=groq  # groq | anthropic | bedrock | ollama

# For Groq (recommended for most users)
GROQ_API_KEY=gsk_xxxx

# For Anthropic
# ANTHROPIC_API_KEY=sk-ant-xxxx

# For AWS Bedrock (enterprise)
# AWS_REGION=us-east-1
# AWS_ACCESS_KEY_ID=xxxx
# AWS_SECRET_ACCESS_KEY=xxxx

# For Ollama (local/air-gap)
# OLLAMA_URL=http://ollama:11434

# === Optional ===
# LOG_LEVEL=info
# TELEMETRY_ENABLED=true
```

```bash
# .env.local (Doug's personal Roxy)
LICENSE_KEY=dev-unlimited
DB_PASSWORD=localdev123

LLM_PROVIDER=ollama
OLLAMA_URL=http://ollama:11434

# Personal features
ENABLE_HOME_ASSISTANT=true
HOME_ASSISTANT_URL=http://homeassistant.local:8123
HOME_ASSISTANT_TOKEN=xxxx

# Preference sync to work
PREFERENCE_SYNC_ENABLED=true
PREFERENCE_SYNC_URL=https://draagon.caremetx.internal/api/v1/users/doug/preferences
PREFERENCE_SYNC_KEY=xxxx
```

```bash
# .env.caremetx (CareMetx production)
LICENSE_KEY=ent_caremetx_2025_xxxxxxxxxxxx
DB_PASSWORD=${DB_PASSWORD_FROM_SECRETS_MANAGER}

LLM_PROVIDER=bedrock
AWS_REGION=us-east-1

# Enterprise features enabled by license
# SSO configured via admin dashboard

# HIPAA compliance
AUDIT_LOG_ENABLED=true
AUDIT_LOG_RETENTION_DAYS=2555  # 7 years

# No external telemetry
TELEMETRY_ENABLED=false
```

#### Development Workflow

```bash
# Daily development on pop-os-desktop
cd ~/Development/draagon-ai

# Start all services (uses .env.local automatically)
docker compose up -d

# View logs
docker compose logs -f draagon-api

# Make code changes, rebuild
docker compose up -d --build draagon-api

# Run tests against local stack
pytest tests/

# Test with CareMetx config (different LLM, features)
docker compose --env-file .env.caremetx up -d

# Tear down
docker compose down
```

#### Compose Override Pattern

```yaml
# docker-compose.yml (base - shipped to customers)
services:
  draagon-api:
    image: ghcr.io/draagon-ai/draagon:${VERSION:-latest}
    # ... production defaults ...

# docker-compose.override.yml (local dev - not shipped)
services:
  draagon-api:
    build: .  # Build from local source instead of pulling image
    volumes:
      - ./src:/app/src  # Hot reload
    environment:
      - DEBUG=true
      - LOG_LEVEL=debug

  # Add Ollama for local LLM
  ollama:
    image: ollama/ollama:latest
    volumes:
      - ollama_data:/root/.ollama
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]  # Use GPU if available
```

#### Railway Deployment

Railway uses the same Docker images but configures via their dashboard:

```bash
# railway.json (Railway config)
{
  "$schema": "https://railway.app/railway.schema.json",
  "build": {
    "builder": "DOCKERFILE",
    "dockerfilePath": "Dockerfile"
  },
  "deploy": {
    "healthcheckPath": "/health",
    "healthcheckTimeout": 30,
    "restartPolicyType": "ON_FAILURE"
  }
}
```

```bash
# Deploy to Railway
railway up

# Or via GitHub integration - auto-deploys on push to main
```

#### Multi-Compose for Complex Setups

```bash
# CareMetx with all enterprise features
docker compose \
  -f docker-compose.yml \
  -f docker-compose.enterprise.yml \
  --env-file .env.caremetx \
  up -d

# docker-compose.enterprise.yml adds:
# - SSO sidecar
# - Audit log shipper
# - Metrics exporter
```

### Recommended Architecture for Draagon-AI

```yaml
# docker-compose.yml (Tier 1)
version: '3.8'

services:
  draagon-api:
    image: ghcr.io/draagon-ai/draagon:${VERSION:-latest}
    environment:
      - DATABASE_URL=postgresql://draagon:${DB_PASSWORD}@postgres:5432/draagon
      - QDRANT_URL=http://qdrant:6333
      - REDIS_URL=redis://redis:6379
      - LICENSE_KEY=${LICENSE_KEY:-}
      - LLM_PROVIDER=${LLM_PROVIDER:-groq}
      - GROQ_API_KEY=${GROQ_API_KEY:-}
      # For air-gap: use local Ollama
      - OLLAMA_URL=${OLLAMA_URL:-}
    ports:
      - "8000:8000"
    depends_on:
      postgres:
        condition: service_healthy
      qdrant:
        condition: service_started
      redis:
        condition: service_started
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    restart: unless-stopped

  qdrant:
    image: qdrant/qdrant:v1.12.0
    volumes:
      - qdrant_data:/qdrant/storage
    environment:
      - QDRANT__SERVICE__GRPC_PORT=6334
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:6333/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    restart: unless-stopped

  postgres:
    image: postgres:16-alpine
    environment:
      - POSTGRES_USER=draagon
      - POSTGRES_PASSWORD=${DB_PASSWORD}
      - POSTGRES_DB=draagon
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U draagon"]
      interval: 10s
      timeout: 5s
      retries: 5
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5
    restart: unless-stopped

volumes:
  qdrant_data:
  postgres_data:
  redis_data:
```

### License Key System

Use **Keygen** (open source CE available) or build custom with Ed25519 signatures:

```python
# Offline license validation (no phone-home required)
import json
import base64
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

@dataclass
class License:
    customer_id: str
    customer_name: str
    plan: str                    # "pro" | "team" | "enterprise"
    features: list[str]          # ["sso", "audit_logs", "air_gap"]
    max_users: int
    expires_at: datetime
    signature: bytes

def validate_license(license_key: str, public_key: bytes) -> License:
    """Validate offline license - no network required."""
    parts = license_key.split('.')
    data_b64, sig_b64 = parts[0], parts[1]

    data = base64.b64decode(data_b64)
    signature = base64.b64decode(sig_b64)

    # Verify signature (fails if tampered)
    public_key = Ed25519PublicKey.from_public_bytes(public_key)
    public_key.verify(signature, data)

    license_data = json.loads(data)

    # Check expiration
    expires = datetime.fromisoformat(license_data['expires_at'])
    if expires < datetime.now():
        raise LicenseExpired()

    return License(**license_data)
```

### Feature Gating by License Tier

```python
TIER_FEATURES = {
    "free": {
        "max_users": 1,
        "features": ["memory", "beliefs", "curiosity"],
        "memory_limit_mb": 100,
    },
    "pro": {
        "max_users": 1,
        "features": ["memory", "beliefs", "curiosity", "cross_device_sync"],
        "memory_limit_mb": 1000,
    },
    "team": {
        "max_users": 50,
        "features": [
            "memory", "beliefs", "curiosity",
            "team_memory", "shared_beliefs",
            "sso_google", "sso_github",
            "admin_dashboard",
        ],
        "memory_limit_mb": 10000,
    },
    "enterprise": {
        "max_users": -1,  # Unlimited
        "features": [
            "memory", "beliefs", "curiosity",
            "team_memory", "shared_beliefs",
            "sso_saml", "sso_ldap", "sso_oidc",
            "admin_dashboard", "audit_logs",
            "air_gap", "dedicated_support",
            "custom_llm", "byoc",
        ],
        "memory_limit_mb": -1,  # Unlimited
    },
}
```

### Air-Gap Deployment Bundle

For enterprises with no internet access:

```bash
# Create air-gap bundle
draagon-bundle create \
  --version 1.0.0 \
  --include-ollama \
  --include-models llama3.1:8b,nomic-embed-text \
  --output draagon-airgap-1.0.0.tar.gz

# Contents:
# - All Docker images (draagon-api, qdrant, postgres, redis, ollama)
# - Ollama models (for offline LLM inference)
# - Install script
# - Documentation
# - License validation public key
```

### Cloud Marketplace Strategy

Cloud marketplaces are growing 29% CAGR, reaching $163B by 2030:

| Marketplace | Fee | Why List |
|-------------|-----|----------|
| **AWS Marketplace** | 1.5-3% | Enterprise procurement, committed spend |
| **Azure Marketplace** | 3% | Microsoft shops, enterprise |
| **GCP Marketplace** | 3% | GCP-heavy organizations |

**Benefits**:
- Procurement teams can use existing cloud contracts
- No new vendor approval process
- Consolidated billing
- Trust signal for enterprises

### BYOC (Bring Your Own Cloud) - Future

Emerging pattern where vendor manages control plane, customer owns data plane:

```
┌─────────────────────────────────────────────────────────────────┐
│                     BYOC ARCHITECTURE                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   DRAAGON CONTROL PLANE (our infrastructure)                    │
│   ──────────────────────────────────────────                    │
│   • License management                                           │
│   • Version updates                                              │
│   • Monitoring/alerting                                          │
│   • Support ticketing                                            │
│                                                                  │
│                         │ Secure tunnel (no data)               │
│                         ▼                                        │
│                                                                  │
│   CUSTOMER DATA PLANE (their AWS/GCP/Azure)                     │
│   ──────────────────────────────────────────                    │
│   • Draagon API (runs in their VPC)                             │
│   • Qdrant (their data stays here)                              │
│   • PostgreSQL (their data stays here)                          │
│   • LLM calls (Bedrock in their account)                        │
│                                                                  │
│   DATA NEVER LEAVES CUSTOMER INFRASTRUCTURE                     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Implementation Phases

| Phase | Timeline | Deliverables |
|-------|----------|--------------|
| **1: MVP** | Weeks 1-4 | Docker Compose, install script, basic docs |
| **2: Production** | Weeks 5-8 | Helm chart, Terraform modules, upgrade path |
| **3: Enterprise** | Weeks 9-12 | License system, SSO, audit logs, air-gap bundle |
| **4: Marketplace** | Months 4-6 | AWS/Azure/GCP listings, BYOC exploration |

### Trial & Demo Strategy

Three approaches, from cheapest to most isolated:

```
┌─────────────────────────────────────────────────────────────────┐
│                      TRIAL OPTIONS                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  OPTION A: Shared Multi-Tenant (Cheapest)                       │
│  ─────────────────────────────────────────                      │
│  Cost: ~$0.50-2/trial (marginal)                                │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              Single Railway Instance                     │    │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐    │    │
│  │  │ Trial A │  │ Trial B │  │ Trial C │  │ Paid    │    │    │
│  │  │ Acme Co │  │ BigCorp │  │ StartupX│  │Customer │    │    │
│  │  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘    │    │
│  │       │            │            │            │          │    │
│  │       └────────────┴────────────┴────────────┘          │    │
│  │                         │                                │    │
│  │                         ▼                                │    │
│  │  ┌─────────────────────────────────────────────────┐    │    │
│  │  │ Shared Qdrant (tenant_id filtering)             │    │    │
│  │  │ Shared Postgres (tenant_id column)              │    │    │
│  │  │ Shared Redis (key prefixing)                    │    │    │
│  │  └─────────────────────────────────────────────────┘    │    │
│  │                                                          │    │
│  │  Cleanup: Cron job deletes data where:                  │    │
│  │  - trial_expires_at < now() OR                          │    │
│  │  - last_activity > 14 days                              │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  OPTION B: Isolated Collections (Middle Ground)                 │
│  ──────────────────────────────────────────────                 │
│  Cost: ~$5-10/trial                                             │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              Single Railway Instance                     │    │
│  │                                                          │    │
│  │  Shared API, but per-tenant Qdrant collections:         │    │
│  │  ┌─────────────────────────────────────────────────┐    │    │
│  │  │ Qdrant                                          │    │    │
│  │  │ ├── trial_acme_memories     (auto-delete)      │    │    │
│  │  │ ├── trial_bigcorp_memories  (auto-delete)      │    │    │
│  │  │ ├── prod_customer1_memories (permanent)        │    │    │
│  │  │ └── prod_customer2_memories (permanent)        │    │    │
│  │  └─────────────────────────────────────────────────┘    │    │
│  │                                                          │    │
│  │  Better isolation, easy cleanup (drop collection)       │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  OPTION C: Dedicated Instance (Most Isolated)                   │
│  ─────────────────────────────────────────────                  │
│  Cost: ~$20-50/trial/month                                      │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  Railway Project: trial-acme-co                         │    │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐                 │    │
│  │  │ API     │  │ Qdrant  │  │ Postgres│                 │    │
│  │  └─────────┘  └─────────┘  └─────────┘                 │    │
│  │                                                          │    │
│  │  Auto-teardown after 14 days via Railway API            │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  Best for: Enterprise trials, POCs, security-conscious         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

#### Recommendation: Option B (Isolated Collections)

Best balance of cost, isolation, and simplicity:

```python
# Tenant/Trial Management

@dataclass
class Trial:
    id: str
    company_name: str
    admin_email: str
    created_at: datetime
    expires_at: datetime          # 14 days from creation
    last_activity_at: datetime
    status: str                   # "active" | "expired" | "converted"

    # Resource identifiers
    qdrant_collection: str        # f"trial_{id}_memories"
    api_key: str                  # Limited scope key

    # Limits
    max_users: int = 5
    max_memory_mb: int = 100
    features: list[str] = field(default_factory=lambda: [
        "memory", "beliefs", "curiosity"  # No SSO, no team features
    ])


class TrialManager:
    def __init__(self, db: Database, qdrant: QdrantClient, email: EmailService):
        self.db = db
        self.qdrant = qdrant
        self.email = email

    async def create_trial(self, company: str, admin_email: str) -> Trial:
        """Provision a new trial instantly."""
        trial_id = generate_short_id()  # e.g., "acme-7x9k"

        trial = Trial(
            id=trial_id,
            company_name=company,
            admin_email=admin_email,
            created_at=datetime.utcnow(),
            expires_at=datetime.utcnow() + timedelta(days=14),
            last_activity_at=datetime.utcnow(),
            status="active",
            qdrant_collection=f"trial_{trial_id}_memories",
            api_key=generate_trial_api_key(trial_id),
        )

        # Create isolated Qdrant collection
        await self.qdrant.create_collection(
            collection_name=trial.qdrant_collection,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
        )

        # Store trial metadata
        await self.db.trials.insert(trial)

        # Send welcome email with setup instructions
        await self.email.send_trial_welcome(admin_email, trial)

        return trial

    async def cleanup_expired_trials(self):
        """Run daily via cron to clean up expired/inactive trials."""

        # Find trials to clean up
        expired = await self.db.trials.find({
            "$or": [
                {"expires_at": {"$lt": datetime.utcnow()}},
                {"last_activity_at": {"$lt": datetime.utcnow() - timedelta(days=14)}},
            ],
            "status": "active"
        })

        for trial in expired:
            await self._cleanup_trial(trial)

    async def _cleanup_trial(self, trial: Trial):
        """Delete all trial data."""

        # Delete Qdrant collection (all vectors gone)
        await self.qdrant.delete_collection(trial.qdrant_collection)

        # Delete from Postgres
        await self.db.execute(
            "DELETE FROM memories WHERE tenant_id = :id", {"id": trial.id}
        )
        await self.db.execute(
            "DELETE FROM users WHERE tenant_id = :id", {"id": trial.id}
        )

        # Mark trial as expired
        await self.db.trials.update(
            {"id": trial.id},
            {"$set": {"status": "expired", "cleaned_up_at": datetime.utcnow()}}
        )

        # Send "trial expired" email with conversion offer
        await self.email.send_trial_expired(trial.admin_email, trial)
```

#### Automated Cleanup Cron

```python
# cleanup_job.py - Run daily via Railway cron or separate worker

import asyncio
from apscheduler.schedulers.asyncio import AsyncIOScheduler

scheduler = AsyncIOScheduler()

@scheduler.scheduled_job('cron', hour=3)  # Run at 3 AM daily
async def cleanup_trials():
    """Daily cleanup of expired trials."""
    manager = TrialManager(db, qdrant, email)

    cleaned = await manager.cleanup_expired_trials()
    logger.info(f"Cleaned up {len(cleaned)} expired trials")

@scheduler.scheduled_job('cron', hour=9)  # Run at 9 AM daily
async def send_expiry_warnings():
    """Warn trials expiring in 3 days."""
    expiring_soon = await db.trials.find({
        "expires_at": {
            "$gte": datetime.utcnow(),
            "$lt": datetime.utcnow() + timedelta(days=3)
        },
        "status": "active",
        "warning_sent": {"$ne": True}
    })

    for trial in expiring_soon:
        await email.send_trial_expiring_warning(trial.admin_email, trial)
        await db.trials.update(
            {"id": trial.id},
            {"$set": {"warning_sent": True}}
        )
```

#### Railway Automation for Dedicated Trials (Option C)

If you need fully isolated trials for enterprise POCs:

```python
# railway_trial_manager.py
import httpx

class RailwayTrialManager:
    """Manage dedicated Railway projects for enterprise trials."""

    def __init__(self, railway_token: str, template_id: str):
        self.token = railway_token
        self.template_id = template_id  # Your base project template
        self.client = httpx.AsyncClient(
            base_url="https://backboard.railway.app/graphql/v2",
            headers={"Authorization": f"Bearer {railway_token}"}
        )

    async def create_dedicated_trial(self, company: str) -> dict:
        """Spin up a dedicated Railway project for enterprise trial."""

        # Create project from template
        result = await self.client.post("", json={
            "query": """
                mutation($input: ProjectCreateInput!) {
                    projectCreate(input: $input) {
                        id
                        name
                    }
                }
            """,
            "variables": {
                "input": {
                    "name": f"trial-{company.lower()}-{generate_short_id()}",
                    "teamId": TEAM_ID,
                    # Clone from template
                }
            }
        })

        project = result.json()["data"]["projectCreate"]

        # Set environment variables
        await self._set_env_vars(project["id"], {
            "LICENSE_KEY": generate_trial_license(company, days=14),
            "TRIAL_MODE": "true",
            "TRIAL_EXPIRES_AT": (datetime.utcnow() + timedelta(days=14)).isoformat(),
        })

        # Deploy
        await self._trigger_deploy(project["id"])

        # Schedule teardown
        await self._schedule_teardown(project["id"], days=14)

        return {
            "project_id": project["id"],
            "url": f"https://{project['name']}.up.railway.app",
            "expires_at": datetime.utcnow() + timedelta(days=14),
        }

    async def teardown_trial(self, project_id: str):
        """Delete the entire Railway project."""
        await self.client.post("", json={
            "query": """
                mutation($id: String!) {
                    projectDelete(id: $id)
                }
            """,
            "variables": {"id": project_id}
        })
```

#### Trial Signup Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                     TRIAL SIGNUP FLOW                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. User visits draagon.ai/trial                                │
│     └── Enters: Company name, email, team size                  │
│                                                                  │
│  2. Instant provisioning (<5 seconds)                           │
│     ├── Create tenant record                                    │
│     ├── Create Qdrant collection                                │
│     ├── Generate API key                                        │
│     └── Send welcome email                                      │
│                                                                  │
│  3. Welcome email contains:                                      │
│     ├── API key                                                 │
│     ├── Quick start guide                                       │
│     ├── docker-compose.trial.yml (optional local)              │
│     └── MCP server config for Claude Code                      │
│                                                                  │
│  4. Trial experience:                                           │
│     ├── 14 days full access (limited features)                 │
│     ├── 5 user limit                                            │
│     ├── 100MB memory limit                                      │
│     └── No SSO (team tier feature)                              │
│                                                                  │
│  5. Day 11: "3 days left" warning email                         │
│                                                                  │
│  6. Day 14: Trial expires                                       │
│     ├── API returns 402 Payment Required                        │
│     ├── Data retained for 7 more days                          │
│     └── "Convert now" email with discount                       │
│                                                                  │
│  7. Day 21: Data deleted                                        │
│     └── Final "your data was deleted" email                    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

#### Cost Comparison

| Approach | Cost/Trial | Setup Time | Isolation | Best For |
|----------|-----------|------------|-----------|----------|
| **Shared (A)** | ~$0.50 | Instant | Low (filtering) | High volume, self-serve |
| **Collections (B)** | ~$5 | Instant | Medium | Default recommendation |
| **Dedicated (C)** | ~$30 | 2-5 min | Full | Enterprise POCs |

#### Implementation Priority

```
Phase 1 (MVP): Shared multi-tenant with tenant_id filtering
  - Simple, works on single Railway instance
  - Manual cleanup initially

Phase 2: Isolated Qdrant collections
  - Better isolation
  - Automated cleanup cron

Phase 3: Dedicated instances for enterprise
  - Railway API automation
  - Premium trial option
```

### Minimum Viable Self-Hosted Checklist

**Required for launch**:
- [ ] Docker Compose with all dependencies
- [ ] `docker compose up -d` one-liner
- [ ] Health check endpoints (`/health`, `/ready`)
- [ ] Environment-based configuration (`.env.example`)
- [ ] Persistent volumes for data
- [ ] Non-root container user
- [ ] Clear upgrade documentation
- [ ] Backup/restore procedures

**Required for enterprise**:
- [ ] Helm chart
- [ ] License key validation (offline-capable)
- [ ] SSO/SAML integration
- [ ] Audit logging
- [ ] Security hardening guide
- [ ] SLA documentation

---

## Scalability Architecture & Database Strategy

### Current Architecture Assessment

Draagon-ai currently uses a **Qdrant-centric architecture**:
- All memory storage in Qdrant collections
- Graph relationships emulated via payload metadata (Netflix pattern)
- No PostgreSQL or Redis
- Multi-tenancy via payload filtering (user_id, agent_id, context_id)

This is a **valid starting point** for MVP, but has scaling implications.

### Qdrant Scalability Research

```
┌─────────────────────────────────────────────────────────────────┐
│                    QDRANT SCALING THRESHOLDS                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   SINGLE NODE (Your Current Architecture)                       │
│   ─────────────────────────────────────────                     │
│   • Sweet spot: <10M vectors                                    │
│   • Comfortable: up to 50M with good RAM                        │
│   • Maximum practical: ~100M vectors (with tuning)              │
│   • Memory formula: vectors × dimensions × 4 bytes × 1.5        │
│                                                                  │
│   For 1M 768-dim vectors (your current config):                 │
│   • ~4.6 GB RAM for vectors                                     │
│   • Easily fits on 8GB instance                                 │
│                                                                  │
│   For 10M vectors: ~46 GB RAM needed                            │
│   For 100M vectors: ~460 GB RAM (or on-disk with slower search) │
│                                                                  │
│   CLUSTER MODE (When You Need It)                               │
│   ─────────────────────────────────                             │
│   • Add 1 node per 5-10M vectors                                │
│   • 3+ nodes for high availability                              │
│   • Resharding available in Qdrant Cloud                        │
│   • Network overhead adds ~1-5ms latency                        │
│                                                                  │
│   RECOMMENDATION FOR DRAAGON:                                   │
│   ─────────────────────────────                                 │
│   • Local/Personal Roxy: Single node forever (<<1M vectors)    │
│   • Small team (5-20): Single node (<<5M vectors)              │
│   • Enterprise (100+): Plan for cluster at 10M+                │
│   • SaaS multi-tenant: Monitor per-tenant growth               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Key Insight**: You likely won't hit Qdrant limits for years. A personal Roxy storing 1000 memories/day for 10 years = 3.65M vectors. Still single-node territory.

### When to Add PostgreSQL

**Short Answer**: Add PostgreSQL now. It's in your docker-compose already.

**What PostgreSQL is FOR** (not vector search):

```
┌─────────────────────────────────────────────────────────────────┐
│                    POSTGRESQL USE CASES                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. RELATIONAL DATA (Qdrant can't do this well)                 │
│  ───────────────────────────────────────────────                │
│  • Users, organizations, teams, roles                           │
│  • Subscriptions, billing, licenses                             │
│  • Audit logs (HIPAA requires queryable logs)                   │
│  • Feature flags and configuration                              │
│  • Trial management (your TrialManager needs this!)             │
│                                                                  │
│  2. TRANSACTIONS (Qdrant has eventual consistency)              │
│  ────────────────────────────────────────────────               │
│  • User signup flow (create user + org + team atomically)       │
│  • Billing operations                                           │
│  • Permission changes                                           │
│  • License validation state                                     │
│                                                                  │
│  3. COMPLEX QUERIES (Qdrant filters are limited)                │
│  ─────────────────────────────────────────────                  │
│  • "All users who haven't logged in for 30 days"                │
│  • "Teams with >5 members in org X"                             │
│  • "Revenue by tier by month"                                   │
│  • Admin dashboards and reporting                               │
│                                                                  │
│  4. MIGRATIONS & SCHEMA EVOLUTION                               │
│  ────────────────────────────────                               │
│  • Alembic/Django migrations                                    │
│  • Schema versioning                                            │
│  • Data transformations                                         │
│                                                                  │
│  WHAT TO KEEP IN QDRANT                                         │
│  ───────────────────────                                        │
│  • Vector embeddings (semantic search)                          │
│  • Memory content and metadata                                  │
│  • Beliefs (need semantic similarity)                           │
│  • Graph nodes/edges (your current approach)                    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Recommended Split**:

```python
# PostgreSQL tables (add now)
class User(Base):
    id: UUID
    email: str
    organization_id: UUID
    role_id: UUID
    created_at: datetime
    last_login: datetime
    preferences: dict  # JSONB

class Organization(Base):
    id: UUID
    name: str
    plan: str  # "free", "pro", "team", "enterprise"
    settings: dict  # JSONB
    qdrant_collection: str  # Reference to Qdrant

class Team(Base):
    id: UUID
    organization_id: UUID
    name: str
    instructions: str

class AuditLog(Base):  # HIPAA requirement
    id: UUID
    timestamp: datetime
    user_id: UUID
    action: str
    resource_type: str
    resource_id: str
    details: dict  # JSONB
    ip_address: str

class Trial(Base):  # Your trial management
    id: str
    company_name: str
    admin_email: str
    expires_at: datetime
    status: str
    qdrant_collection: str

# Qdrant collections (keep as-is)
# - draagon_memories (vectors + payload)
# - draagon_prompts (vectors + payload)
# - draagon_graph_nodes (vectors + payload)
```

### When to Add Redis

**Short Answer**: Add Redis when you need caching or real-time features. Can wait until Phase 2.

**What Redis is FOR**:

```
┌─────────────────────────────────────────────────────────────────┐
│                      REDIS USE CASES                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. CACHING (Performance)                                        │
│  ─────────────────────────                                      │
│  • LLM response caching (semantic cache)                        │
│  • Embedding cache (avoid re-embedding same text)               │
│  • User session data                                            │
│  • Frequently accessed memories                                 │
│                                                                  │
│  Redis benchmarks show 9.5x higher QPS than Aurora PostgreSQL   │
│  and up to 14x lower latency than MongoDB for vector ops        │
│                                                                  │
│  2. RATE LIMITING                                                │
│  ────────────────                                               │
│  • API rate limits per user/org                                 │
│  • LLM call quotas                                              │
│  • Trial usage tracking                                         │
│                                                                  │
│  3. REAL-TIME FEATURES                                           │
│  ─────────────────────                                          │
│  • Pub/sub for collaborative features                           │
│  • Live typing indicators                                       │
│  • Cross-instance event broadcasting                            │
│                                                                  │
│  4. JOB QUEUES                                                   │
│  ──────────                                                     │
│  • Background embedding generation                              │
│  • Memory consolidation jobs                                    │
│  • Trial cleanup tasks                                          │
│  • Email sending                                                │
│                                                                  │
│  WHEN TO ADD REDIS                                               │
│  ─────────────────                                              │
│  • Phase 1 (MVP): Not required, can use in-memory               │
│  • Phase 2 (Team): Add for session management, rate limiting    │
│  • Phase 3 (Scale): Essential for caching, queues               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Recommendation: Phased Database Addition

```
┌─────────────────────────────────────────────────────────────────┐
│                    PHASED DATABASE STRATEGY                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  PHASE 1 (NOW): Add PostgreSQL                                  │
│  ────────────────────────────────                               │
│  • Already in your docker-compose.yml                           │
│  • Add: Users, Orgs, Teams, Roles, Trials, AuditLogs            │
│  • Use Alembic for migrations                                   │
│  • Keep all vector/memory data in Qdrant                        │
│  • Effort: 1-2 weeks                                            │
│                                                                  │
│  PHASE 2 (Team tier launch): Add Redis                          │
│  ──────────────────────────────────────                         │
│  • Session management                                           │
│  • Rate limiting                                                │
│  • Simple job queue (cleanup, emails)                           │
│  • Effort: 1 week                                               │
│                                                                  │
│  PHASE 3 (Scale): Optimize                                       │
│  ──────────────────────────                                     │
│  • Semantic caching in Redis                                    │
│  • Embedding cache                                              │
│  • Consider Qdrant cluster if >10M vectors                      │
│  • Effort: Ongoing                                              │
│                                                                  │
│  WHY ADD NOW vs LATER                                            │
│  ─────────────────────                                          │
│  ✓ PostgreSQL is trivial to add now (already in compose)        │
│  ✓ Migrations are easier with less data                         │
│  ✓ Trial management NEEDS relational queries                    │
│  ✓ HIPAA audit logs NEED queryable storage                      │
│  ✓ User/org management is relational by nature                  │
│  ✗ Redis CAN wait - in-memory works for MVP                     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Global Shared Knowledge Architecture

### The Opportunity

What if users could opt-in to contribute (anonymized) learnings to a global knowledge base?

```
┌─────────────────────────────────────────────────────────────────┐
│                    KNOWLEDGE SHARING TIERS                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  TIER 1: INDIVIDUAL (Current)                                   │
│  ─────────────────────────────                                  │
│  • Personal memories, beliefs, preferences                      │
│  • Never shared                                                  │
│  • Fully private                                                │
│                                                                  │
│  TIER 2: TEAM (Implemented in hierarchical config)              │
│  ──────────────────────────────────────────────                 │
│  • Team coding conventions                                      │
│  • Project-specific knowledge                                   │
│  • Shared among team members                                    │
│                                                                  │
│  TIER 3: ORGANIZATION                                            │
│  ────────────────────                                           │
│  • Company-wide standards                                       │
│  • Architectural decisions                                      │
│  • Shared across all teams                                      │
│                                                                  │
│  TIER 4: GLOBAL (New Opportunity)                               │
│  ─────────────────────────────────                              │
│  • Community-contributed knowledge                              │
│  • Anonymous, aggregated patterns                               │
│  • Opt-in only                                                  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Architecture Options

**Option A: Separate Global Qdrant (Recommended)**

```
┌─────────────────────────────────────────────────────────────────┐
│                   GLOBAL KNOWLEDGE ARCHITECTURE                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Customer Instances              Global Knowledge Service      │
│   ────────────────────            ────────────────────────      │
│                                                                  │
│   ┌─────────────────┐                                           │
│   │ Customer A      │                                           │
│   │ (Self-hosted)   │──┐                                        │
│   └─────────────────┘  │                                        │
│                        │  Opt-in                                │
│   ┌─────────────────┐  │  Contribution     ┌─────────────────┐  │
│   │ Customer B      │──┼─────────────────► │ Global Qdrant   │  │
│   │ (Railway)       │  │  (anonymized)     │ (Draagon-hosted)│  │
│   └─────────────────┘  │                   │                 │  │
│                        │                   │ • Language best │  │
│   ┌─────────────────┐  │                   │   practices     │  │
│   │ Personal Roxy   │──┘                   │ • Framework     │  │
│   │ (Local Docker)  │                      │   patterns      │  │
│   └─────────────────┘                      │ • Error         │  │
│         │                                  │   solutions     │  │
│         │ Query                            │ • Security      │  │
│         │ (read-only)                      │   guidelines    │  │
│         └─────────────────────────────────►│                 │  │
│                                            └─────────────────┘  │
│                                                                  │
│   Data Flow:                                                    │
│   • Contribution: Customer → Anonymizer → Global Qdrant        │
│   • Query: Customer ← Global MCP Server ← Global Qdrant        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**What Gets Contributed (with consent)**:
- Language/framework best practices (not company-specific code)
- Common error patterns and solutions
- Security anti-patterns
- Performance optimization tips
- Library recommendations

**What NEVER Gets Contributed**:
- Proprietary code
- Company-specific architecture
- User identities
- PHI/PII (HIPAA/GDPR compliance)

### Implementation as MCP Server

```python
# Global knowledge as an optional MCP server
# Customers enable by adding to their MCP config

# mcp_config.json (customer's config)
{
  "servers": {
    "draagon-local": {
      "command": "draagon-mcp",
      "args": ["--local"]
    },
    "draagon-global": {
      "command": "draagon-mcp-global",
      "args": ["--api-key", "${DRAAGON_GLOBAL_API_KEY}"],
      "enabled": true  # Opt-in
    }
  }
}
```

```python
# Global knowledge MCP server
@server.tool("global_knowledge_search")
async def search_global_knowledge(
    query: str,
    category: str,  # "best_practices", "errors", "security", "performance"
    language: str = None,
    framework: str = None,
) -> list[dict]:
    """Search community-contributed knowledge."""
    # Read-only access to global Qdrant
    results = await global_qdrant.search(
        collection="global_knowledge",
        query_vector=embed(query),
        filter={
            "category": category,
            "language": language,
            "framework": framework,
        },
        limit=10,
    )
    return results

@server.tool("contribute_knowledge")
async def contribute_to_global(
    content: str,
    category: str,
    metadata: dict,
    ctx: RequestContext,
) -> dict:
    """Contribute anonymized knowledge to global pool."""
    # Verify user has opted in
    if not ctx.user.global_contribution_enabled:
        raise PermissionError("Global contribution not enabled")

    # Anonymize content
    anonymized = await anonymizer.process(content)

    # Queue for moderation before adding to global
    await moderation_queue.add(anonymized, ctx.org_id)

    return {"status": "queued_for_review"}
```

### Pricing Model for Global Knowledge

| Tier | Read Access | Contribution | Price |
|------|-------------|--------------|-------|
| **Free** | 10 queries/day | None | $0 |
| **Pro** | Unlimited | Optional | Included |
| **Team** | Unlimited | Encouraged | Included |
| **Enterprise** | Unlimited + Private | Required for discount | -10% if contributing |

---

## Extension & Capability Marketplace

### The Big Opportunity

You're right - this is potentially **huge**. The Salesforce AppExchange generates **$2.49B** (2024) → **$8.92B** (2033 projected). MCP is seeing **97M+ monthly SDK downloads** after just one year.

```
┌─────────────────────────────────────────────────────────────────┐
│                 MARKETPLACE OPPORTUNITY                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   SALESFORCE APPEXCHANGE (reference model)                      │
│   ─────────────────────────────────────────                     │
│   • Market size: $2.49B (2024) → $8.92B (2033)                 │
│   • Revenue share: 15% to Salesforce                            │
│   • Ecosystem multiplier: $5.80 earned for every $1 Salesforce │
│   • 7,000+ apps, 10M+ installs                                  │
│   • 90% of Fortune 500 use AppExchange apps                     │
│                                                                  │
│   MCP ECOSYSTEM (1 year old)                                    │
│   ──────────────────────────                                    │
│   • 97M+ monthly SDK downloads                                  │
│   • 10,000+ active servers                                      │
│   • Adopted by: ChatGPT, Claude, Cursor, Gemini, VS Code       │
│   • MCP Registry launching for discovery                        │
│   • Being donated to Linux Foundation                           │
│                                                                  │
│   DRAAGON MARKETPLACE OPPORTUNITY                               │
│   ─────────────────────────────────                             │
│   • More than just MCP servers                                  │
│   • Cognitive capabilities (beliefs, memory patterns)           │
│   • Domain-specific knowledge packs                             │
│   • Industry compliance modules                                 │
│   • Integration accelerators                                    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### What Makes Draagon Marketplace Different

MCP servers are **tools** (do things). Draagon extensions are **capabilities** (know things, believe things, learn things).

```
┌─────────────────────────────────────────────────────────────────┐
│                  MCP vs DRAAGON EXTENSIONS                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  MCP SERVERS (Tools)                                            │
│  ───────────────────                                            │
│  • filesystem: Read/write files                                 │
│  • github: Create PRs, issues                                   │
│  • slack: Send messages                                         │
│  • database: Query data                                         │
│                                                                  │
│  Limitation: Stateless, no learning, no memory                  │
│                                                                  │
│  DRAAGON EXTENSIONS (Cognitive Capabilities)                    │
│  ───────────────────────────────────────────                    │
│  • healthcare-beliefs: HIPAA-aware coding patterns              │
│  • react-memory: React best practices that REMEMBER your style  │
│  • security-curiosity: Proactively asks about security gaps     │
│  • python-typing: Learns YOUR type annotation preferences       │
│  • terraform-patterns: Infrastructure beliefs + team standards  │
│                                                                  │
│  DRAAGON KNOWLEDGE PACKS (Pre-trained Beliefs)                  │
│  ─────────────────────────────────────────────                  │
│  • aws-well-architected: 1000+ architectural beliefs            │
│  • owasp-security: Security anti-pattern detection              │
│  • clean-code: Uncle Bob's principles as beliefs                │
│  • domain-driven-design: DDD patterns and conventions           │
│                                                                  │
│  DRAAGON INDUSTRY MODULES (Compliance + Domain)                 │
│  ──────────────────────────────────────────────                 │
│  • healthcare-hipaa: PHI detection, audit logging, beliefs      │
│  • finance-sox: SOX compliance patterns                         │
│  • gdpr-privacy: European privacy compliance                    │
│  • pci-dss: Payment card security                               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Marketplace Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                  DRAAGON MARKETPLACE ARCHITECTURE                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   EXTENSION TYPES                                                │
│   ───────────────                                               │
│                                                                  │
│   1. MCP Servers (Draagon-enhanced)                             │
│      ┌─────────────────────────────────────────────────────┐    │
│      │ Standard MCP + Draagon Memory Integration           │    │
│      │ • Tools call Draagon memory APIs                    │    │
│      │ • Learn from usage over time                        │    │
│      │ • Share beliefs with other extensions               │    │
│      └─────────────────────────────────────────────────────┘    │
│                                                                  │
│   2. Belief Packs (New!)                                         │
│      ┌─────────────────────────────────────────────────────┐    │
│      │ Pre-trained belief sets for domains                 │    │
│      │ • Exportable/importable JSON                        │    │
│      │ • Version controlled                                │    │
│      │ • Confidence levels from community usage            │    │
│      └─────────────────────────────────────────────────────┘    │
│                                                                  │
│   3. Memory Templates                                            │
│      ┌─────────────────────────────────────────────────────┐    │
│      │ Starter memory structures for project types         │    │
│      │ • React app template                                │    │
│      │ • Microservices template                            │    │
│      │ • Data pipeline template                            │    │
│      └─────────────────────────────────────────────────────┘    │
│                                                                  │
│   4. Curiosity Modules                                           │
│      ┌─────────────────────────────────────────────────────┐    │
│      │ Domain-specific question generators                 │    │
│      │ • Security curiosity: "Have you considered...?"    │    │
│      │ • Performance curiosity: "This could be slow..."   │    │
│      │ • Accessibility curiosity: "Screen readers..."     │    │
│      └─────────────────────────────────────────────────────┘    │
│                                                                  │
│   5. Integration Accelerators                                    │
│      ┌─────────────────────────────────────────────────────┐    │
│      │ Complete packages for specific tools                │    │
│      │ • Jira + memory sync                                │    │
│      │ • Datadog + belief-based alerting                   │    │
│      │ • PagerDuty + incident memory                       │    │
│      └─────────────────────────────────────────────────────┘    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Marketplace Revenue Model

Following AppExchange's proven model:

```
┌─────────────────────────────────────────────────────────────────┐
│                    MARKETPLACE ECONOMICS                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   REVENUE SHARE                                                  │
│   ─────────────                                                 │
│   • Free extensions: $0 (drive platform adoption)               │
│   • Paid extensions: 15% to Draagon (AppExchange model)         │
│   • Enterprise/compliance: 20% (higher support burden)          │
│   • Stripe fee: +$0.30/transaction for credit card              │
│                                                                  │
│   PRICING EXAMPLES                                               │
│   ────────────────                                              │
│   • react-memory (free): Drives adoption                        │
│   • aws-well-architected ($49/mo): $41.65 to developer         │
│   • hipaa-compliance ($199/mo): $159.20 to developer           │
│   • enterprise-security-pack ($999/mo): $799.20 to developer   │
│                                                                  │
│   ECOSYSTEM MULTIPLIER (if similar to Salesforce)               │
│   ───────────────────────────────────────────────               │
│   • For every $1 Draagon makes, ecosystem makes $5.80           │
│   • At $1M ARR Draagon → $5.8M ecosystem revenue                │
│   • Creates incentive for developers to build                   │
│                                                                  │
│   TRUST & DISCOVERY                                              │
│   ─────────────────                                             │
│   • Verified publisher badges                                   │
│   • Usage-based ratings                                         │
│   • Security audits for enterprise tier                         │
│   • SOC 2 compliance for compliance modules                     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Extension SDK

```python
# draagon_extension_sdk/belief_pack.py

from draagon_sdk import BeliefPack, Belief, ConfidenceLevel

class ReactBeliefPack(BeliefPack):
    """Pre-trained beliefs for React development."""

    name = "react-best-practices"
    version = "1.0.0"
    author = "Draagon Community"

    beliefs = [
        Belief(
            claim="Prefer functional components over class components",
            confidence=ConfidenceLevel.HIGH,
            category="CONVENTION",
            evidence=[
                "React docs recommend functional components",
                "Hooks only work with functional components",
            ],
        ),
        Belief(
            claim="Use useMemo for expensive calculations, not all values",
            confidence=ConfidenceLevel.HIGH,
            category="PERFORMANCE",
            evidence=[
                "useMemo has overhead",
                "Premature optimization is the root of all evil",
            ],
        ),
        Belief(
            claim="Avoid prop drilling more than 2 levels deep",
            confidence=ConfidenceLevel.MEDIUM,
            category="ARCHITECTURAL_PATTERN",
            evidence=[
                "Use Context or state management for deep props",
            ],
        ),
        # ... 100+ more beliefs
    ]

    def on_install(self, context):
        """Called when user installs this belief pack."""
        # Merge with user's existing beliefs
        # Lower confidence for conflicts with user's learned beliefs
        pass

    def on_update(self, context, previous_version):
        """Called on pack update."""
        # Handle belief migrations
        pass
```

```python
# draagon_extension_sdk/curiosity_module.py

from draagon_sdk import CuriosityModule, Question, Trigger

class SecurityCuriosity(CuriosityModule):
    """Proactively asks security-related questions."""

    name = "security-curiosity"
    version = "1.0.0"

    triggers = [
        Trigger(
            pattern=r"password|secret|api[_-]?key|token",
            question=Question(
                text="Is this secret being stored securely?",
                context="I noticed what might be a secret. Consider using environment variables or a secrets manager.",
                severity="high",
            ),
        ),
        Trigger(
            pattern=r"eval\(|exec\(|subprocess\.call",
            question=Question(
                text="Is this user input being sanitized?",
                context="Dynamic code execution can be dangerous. Ensure inputs are validated.",
                severity="critical",
            ),
        ),
        Trigger(
            pattern=r"\.innerHTML\s*=",
            question=Question(
                text="Could this lead to XSS?",
                context="Setting innerHTML with user input can enable XSS attacks.",
                severity="high",
            ),
        ),
    ]
```

### Implementation Roadmap

```
┌─────────────────────────────────────────────────────────────────┐
│                  MARKETPLACE IMPLEMENTATION                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  PHASE 1: Foundation (Month 1-2)                                │
│  ───────────────────────────────                                │
│  • Extension SDK with BeliefPack, CuriosityModule base classes  │
│  • Local extension loading from filesystem                      │
│  • Version compatibility checking                               │
│  • 3-5 first-party extensions as examples                       │
│                                                                  │
│  PHASE 2: Registry (Month 3-4)                                  │
│  ─────────────────────────────                                  │
│  • Public registry API (like npm, PyPI)                         │
│  • Extension manifest format                                    │
│  • draagon install <extension> CLI                              │
│  • Basic web UI for discovery                                   │
│                                                                  │
│  PHASE 3: Marketplace (Month 5-6)                               │
│  ─────────────────────────────────                              │
│  • Paid extensions support                                      │
│  • Stripe integration                                           │
│  • Publisher dashboard                                          │
│  • Usage analytics                                              │
│                                                                  │
│  PHASE 4: Ecosystem (Month 7+)                                   │
│  ──────────────────────────────                                 │
│  • Verified publisher program                                   │
│  • Enterprise security audits                                   │
│  • Partner program (ISV-like)                                   │
│  • Revenue share payouts                                        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Why This Is Bigger Than MCP

| Aspect | MCP Servers | Draagon Extensions |
|--------|-------------|-------------------|
| **Type** | Tools (do things) | Capabilities (know things) |
| **State** | Stateless | Stateful (learns) |
| **Memory** | None | Integrated |
| **Learning** | None | Improves over time |
| **Sharing** | Per-user | Team/org shareable |
| **Revenue** | Hard to monetize | Clear value = paid |

**Key Insight**: MCP servers are commoditizing (every tool will have one). Draagon extensions provide **accumulated intelligence** - something that compounds over time and is much harder to replicate.

---

## Killer First Feature Analysis

> **Analysis Date**: 2025-12-28
> **Target Outcome**: $5-10M revenue, niche but defensible

### Market Reality Check (Research Summary)

```
┌─────────────────────────────────────────────────────────────────┐
│                    KEY MARKET INSIGHTS                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  DEVELOPER PAIN POINTS (2024-2025 Research):                    │
│  ───────────────────────────────────────────                    │
│  • 65% cite "missing context" as #1 issue (not hallucinations)  │
│  • 66% frustrated by "almost right" solutions                   │
│  • 46% don't trust AI output (up from 31% last year)            │
│  • Only 3% report high trust in AI-generated code               │
│  • Developers spend only 16% of time coding (!!)                │
│                                                                  │
│  ONBOARDING ECONOMICS:                                          │
│  ─────────────────────                                          │
│  • 3-9 months to full productivity for new hires                │
│  • $75K-240K cost per developer in lost productivity            │
│  • 23% of tech hires quit within first year                     │
│  • Senior devs lose 30% productivity mentoring without systems  │
│  • Cutting ramp-up time in half = 17 dev-years saved/year       │
│                                                                  │
│  HIPAA/HEALTHCARE:                                               │
│  ─────────────────                                              │
│  • $9.77M average healthcare data breach cost (2024)            │
│  • Few HIPAA-compliant AI coding tools exist                    │
│  • Healthcare software market growing 15%+ CAGR                 │
│  • You have domain expertise at CareMetx                        │
│                                                                  │
│  MCP ECOSYSTEM:                                                  │
│  ──────────────                                                 │
│  • 97M+ monthly SDK downloads (after 1 year)                    │
│  • Native VS Code support (1.102+)                              │
│  • Adopted by Claude, ChatGPT, Cursor, Gemini, Copilot          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Top 10 Killer Feature Candidates

| # | Feature | Pain Point Addressed | Your Advantage | Market Size |
|---|---------|---------------------|----------------|-------------|
| 1 | **Team Knowledge Memory** | Tribal knowledge loss, onboarding | Built-in, unique architecture | $75K-240K/hire |
| 2 | **Belief-Based Code Reviewer** | "Almost right" AI, noise | BeliefReconciliation exists | $6.7B→$25.7B |
| 3 | **HIPAA-Compliant Dev Memory** | Healthcare compliance gap | CareMetx expertise, Bedrock | Under-served |
| 4 | **Context MCP for All Tools** | Context not portable | Works with Cursor+Copilot+Claude | Universal need |
| 5 | **Onboarding Accelerator** | 3-9 month ramp time | Memory + beliefs capture | $75K savings/hire |
| 6 | **Extension Marketplace** | Platform economics | 15% revenue share | $2.49B→$8.92B |
| 7 | **PR Review Memory** | Repeat feedback, noise | Learns reviewer preferences | Time waste |
| 8 | **Architecture Guardian** | Docs always stale | Temporal graph tracks changes | Doc maintenance |
| 9 | **Curiosity-Driven Explorer** | Unknown unknowns | CuriosityEngine exists | Gap detection |
| 10 | **Personal Dev Assistant** | AI forgets everything | Full Roxy capabilities | Consumer market |

### Deep Dive: Top 10 Analysis

#### 1. Team Knowledge Memory (Institutional Brain)
**What it is**: Shared memory across team members that captures conventions, decisions, patterns, and tribal knowledge.

**Pain Point**: 65% cite missing context as #1 AI problem. New hires take 3-9 months to become productive. When senior engineers leave, knowledge walks out the door.

**Your Advantage**: Draagon-ai's LayeredMemory + BeliefReconciliation is genuinely unique. No competitor has team-level knowledge sync with conflict detection.

**Market Validation**:
- $75K-240K lost productivity per new hire
- 17 dev-years saved annually if ramp time cut in half
- Engineering managers desperately want this

**Challenges**: Cold start problem. Value compounds over time - hard to demo quickly.

**Score**: ⭐⭐⭐⭐⭐ (High impact, unique, aligns with architecture)

---

#### 2. Belief-Based Code Reviewer
**What it is**: Code review AI that learns your team's patterns, detects contradictions with established beliefs, reduces noise.

**Pain Point**: 66% frustrated by "almost right" solutions. AI code review tools fail due to excessive noise. 25% of AI suggestions contain errors.

**Your Advantage**: BeliefReconciliationService already built. Can detect when new code contradicts established patterns. Confidence tracking reduces noise.

**Market Validation**:
- AI code review market: $6.7B (2024) → $25.7B (2030)
- Top failure mode of AI review tools is "noise"
- Teams want high-signal, context-aware reviews

**Challenges**: Needs code analysis integration. Competitive market (CodeRabbit, Qodo, Greptile).

**Score**: ⭐⭐⭐⭐ (High impact, competitive market)

---

#### 3. HIPAA-Compliant Dev Memory
**What it is**: Team knowledge memory specifically designed for healthcare development - Bedrock-hosted, audit logging, PHI-safe.

**Pain Point**: Few HIPAA-compliant AI coding tools. $9.77M average breach cost. Healthcare developers are underserved.

**Your Advantage**:
- CareMetx gives you domain expertise
- Bedrock integration planned
- One-way data flow architecture designed
- Built-in pilot customer

**Market Validation**:
- Healthcare IT spend: $280B (2024)
- HIPAA-compliant AI is scarce
- Enterprises will pay premium for compliance

**Challenges**: Longer sales cycle. SOC 2 needed eventually. Smaller initial market.

**Score**: ⭐⭐⭐⭐⭐ (Niche but defensible, YOU have the advantage)

---

#### 4. Context MCP for All Tools
**What it is**: Universal context layer that works with Claude Code, Cursor, Copilot, any MCP client. Same memory, any tool.

**Pain Point**: Developers use 3+ AI tools regularly. Context doesn't transfer between them.

**Your Advantage**: MCP-native design. Position as augmentation, not replacement.

**Market Validation**:
- 59% use 3+ AI tools regularly
- MCP ecosystem exploding (97M downloads)
- VS Code native support

**Challenges**: Depends on other tools adopting MCP. Less control over UX.

**Score**: ⭐⭐⭐⭐ (Strategic, but dependent on ecosystem)

---

#### 5. Onboarding Accelerator
**What it is**: System that captures tribal knowledge from existing team and makes it available to new hires through AI.

**Pain Point**: 3-9 months to full productivity. Senior devs spend 30% time mentoring. New hires produce "negative value" first 3 months.

**Your Advantage**: Memory + beliefs + curiosity engine = perfect for capturing undocumented knowledge.

**Market Validation**:
- Companies spend $75K+ onboarding each developer
- 82% higher retention with good onboarding
- 70% higher productivity with structured onboarding
- Clear ROI story

**Challenges**: This is really #1 (Team Knowledge) with a different positioning. Needs team adoption first.

**Score**: ⭐⭐⭐⭐ (High impact, but same as #1)

---

#### 6. Extension Marketplace
**What it is**: Marketplace for Draagon extensions - Belief Packs, Curiosity Modules, Industry Compliance packs.

**Pain Point**: Platform economics - create ecosystem, capture 15% revenue share.

**Your Advantage**: More than MCP tools - cognitive capabilities that learn. Harder to replicate.

**Market Validation**:
- Salesforce AppExchange: $2.49B → $8.92B
- Ecosystem multiplier: $5.80 for every $1 platform revenue

**Challenges**: Chicken-and-egg. Need users before developers build. Too early.

**Score**: ⭐⭐⭐ (Future opportunity, not first feature)

---

#### 7. PR Review Memory
**What it is**: AI that remembers past code reviews, learns reviewer preferences, reduces repeat feedback.

**Pain Point**: Same feedback given repeatedly. Reviewers waste time on preventable issues.

**Your Advantage**: Memory + beliefs track what was said before and why.

**Market Validation**:
- Senior engineers bottlenecked on reviews
- AI code review growing 25%+ CAGR

**Challenges**: Narrow use case. Part of #2 (Belief-Based Reviewer).

**Score**: ⭐⭐⭐ (Too narrow as standalone)

---

#### 8. Architecture Guardian
**What it is**: Living architecture documentation maintained by AI watching code changes.

**Pain Point**: Architecture docs are always stale. "The code is the documentation" is cope.

**Your Advantage**: Temporal graph tracks changes over time. Can answer "how did this evolve?"

**Market Validation**:
- Universal pain point
- Architecture drift causes bugs

**Challenges**: Hard problem. Requires deep code analysis. Smaller buyer pool.

**Score**: ⭐⭐⭐ (High effort, unclear market)

---

#### 9. Curiosity-Driven Explorer
**What it is**: AI that proactively asks questions to fill knowledge gaps about your codebase.

**Pain Point**: "Unknown unknowns" - things the AI doesn't know it doesn't know.

**Your Advantage**: CuriosityEngine already built. Unique differentiation.

**Market Validation**:
- Researchers found reactive chat assistants frustrating
- Proactive is better but no one does it

**Challenges**: Could be annoying. Hard to get the timing right.

**Score**: ⭐⭐⭐⭐ (Differentiated, but execution-dependent)

---

#### 10. Personal Dev Assistant (Full Roxy)
**What it is**: Your personal AI assistant with full memory, personality, cross-device sync.

**Pain Point**: AI forgets everything between sessions.

**Your Advantage**: Full Roxy capabilities, personal use case validated.

**Market Validation**:
- Consumer AI assistant market is huge
- BUT crowded (ChatGPT, Claude, personal AIs)

**Challenges**: B2C is expensive to acquire. Commoditizing. Not your strength.

**Score**: ⭐⭐ (Build for yourself, not as primary business)

---

### Ranking Summary

| Rank | Feature | Score | Reasoning |
|------|---------|-------|-----------|
| 1 | **Team Knowledge Memory** | ⭐⭐⭐⭐⭐ | Unique architecture, massive pain, clear ROI |
| 2 | **HIPAA-Compliant Dev Memory** | ⭐⭐⭐⭐⭐ | Niche but defensible, YOU have advantage |
| 3 | **Belief-Based Code Reviewer** | ⭐⭐⭐⭐ | High impact, but competitive market |
| 4 | **Curiosity-Driven Explorer** | ⭐⭐⭐⭐ | Differentiated, execution-dependent |
| 5 | **Context MCP for All Tools** | ⭐⭐⭐⭐ | Strategic, ecosystem-dependent |
| 6 | **Onboarding Accelerator** | ⭐⭐⭐⭐ | Same as #1 with positioning |
| 7 | **Extension Marketplace** | ⭐⭐⭐ | Future opportunity |
| 8 | **PR Review Memory** | ⭐⭐⭐ | Part of #3 |
| 9 | **Architecture Guardian** | ⭐⭐⭐ | Hard, unclear market |
| 10 | **Personal Dev Assistant** | ⭐⭐ | Crowded, not business focus |

---

### TOP 3 FINALISTS

## #1: Team Knowledge Memory
**"The Shared Brain for Engineering Teams"**

```
┌─────────────────────────────────────────────────────────────────┐
│                    TEAM KNOWLEDGE MEMORY                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  WHAT IT IS:                                                    │
│  Shared memory layer that captures and syncs team knowledge:    │
│  • Coding conventions (learned from code, not just documented)  │
│  • Architectural decisions (with rationale and temporal context)│
│  • Tribal knowledge (things that never get written down)        │
│  • Pattern preferences (with confidence levels)                 │
│                                                                  │
│  DELIVERED AS:                                                   │
│  MCP server that works with Claude Code, Cursor, any MCP client │
│                                                                  │
│  UNIQUE VALUE:                                                   │
│  • Memory consolidates automatically (working→episodic→semantic)│
│  • Beliefs reconcile when team members disagree                 │
│  • Curiosity engine asks clarifying questions                   │
│  • Same context available to whole team                         │
│                                                                  │
│  TARGET CUSTOMER:                                                │
│  • Engineering teams (5-50 people)                              │
│  • High churn or growth (lots of new hires)                     │
│  • Complex codebases (multiple services, years of history)      │
│                                                                  │
│  PRICING:                                                        │
│  • Free: Self-hosted, single user                               │
│  • Team: $25/user/month (shared memory, admin dashboard)        │
│  • Enterprise: $40+/user (SSO, audit, self-hosted option)       │
│                                                                  │
│  GO-TO-MARKET:                                                   │
│  1. Open source core → build community                          │
│  2. CareMetx pilot → case study                                 │
│  3. "Reduce onboarding time by 50%" positioning                 │
│  4. Target engineering managers (budget holders)                │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

PROS:
  ✓ Unique - no competitor has belief reconciliation + team sync
  ✓ Massive pain point - $75K-240K per new hire
  ✓ Clear ROI story for enterprise sales
  ✓ Leverages your existing architecture (90% built)
  ✓ Works with existing tools (Claude, Cursor, Copilot via MCP)
  ✓ Team tier has higher revenue per customer than individual
  ✓ Network effects within teams

CONS:
  ✗ Cold start problem - needs weeks of usage to show value
  ✗ Requires team adoption (harder than individual)
  ✗ Enterprise sales cycle is long
  ✗ Hard to demo in 15 minutes
  ✗ Dependent on MCP ecosystem adoption

REVENUE PROJECTION ($5-10M target):
  • 100 teams × 20 users × $25/mo = $50K/mo = $600K ARR
  • 500 teams × 20 users × $25/mo = $250K/mo = $3M ARR
  • 1000 teams × 25 users × $30/mo (blended) = $750K/mo = $9M ARR

TIMELINE TO $1M ARR: 18-24 months
```

---

## #2: HIPAA-Compliant Dev Memory
**"AI That Understands Healthcare Development"**

```
┌─────────────────────────────────────────────────────────────────┐
│                 HIPAA-COMPLIANT DEV MEMORY                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  WHAT IT IS:                                                    │
│  Team Knowledge Memory (above) but specifically designed for    │
│  healthcare software development:                                │
│  • Runs on AWS Bedrock (no external API calls)                  │
│  • HIPAA audit logging built-in                                 │
│  • PHI-safe architecture (data never leaves VPC)                │
│  • Healthcare-specific beliefs (HIPAA patterns, HL7, FHIR)     │
│  • Pre-loaded with compliance knowledge                         │
│                                                                  │
│  DELIVERED AS:                                                   │
│  Self-hosted Docker/Helm with Bedrock integration               │
│                                                                  │
│  UNIQUE VALUE:                                                   │
│  • Only HIPAA-compliant AI coding memory on market              │
│  • Pre-trained on healthcare development patterns               │
│  • Audit logging satisfies compliance requirements              │
│  • You understand the domain (CareMetx experience)              │
│                                                                  │
│  TARGET CUSTOMER:                                                │
│  • Healthcare software companies                                 │
│  • Health systems with dev teams                                │
│  • Pharma/biotech software teams                                │
│  • Healthcare-adjacent (insurance, benefits)                    │
│                                                                  │
│  PRICING:                                                        │
│  • Enterprise only: $50-100/user/month                          │
│  • Setup fee: $5K-25K (depending on size)                       │
│  • Annual contracts required                                    │
│                                                                  │
│  GO-TO-MARKET:                                                   │
│  1. CareMetx as first customer and case study                   │
│  2. Healthcare software conferences (HIMSS, etc.)               │
│  3. Partner with healthcare IT consultancies                    │
│  4. Content: "AI Coding Assistants and HIPAA Compliance"        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

PROS:
  ✓ Underserved market - few HIPAA-compliant AI coding tools exist
  ✓ Premium pricing ($50-100/user vs $25)
  ✓ You have domain expertise (CareMetx)
  ✓ Built-in pilot customer
  ✓ Defensible - compliance is hard to fake
  ✓ Healthcare has budget (not price-sensitive)
  ✓ Longer contracts = more predictable revenue
  ✓ Less competition in niche

CONS:
  ✗ Smaller total addressable market
  ✗ Longer enterprise sales cycle
  ✗ Eventually need SOC 2 Type II (~$30K, 6 months)
  ✗ Healthcare IT is conservative/slow to adopt
  ✗ Compliance burden increases with scale
  ✗ CareMetx conflict of interest questions?

REVENUE PROJECTION ($5-10M target):
  • 20 customers × 50 users × $75/mo = $75K/mo = $900K ARR
  • 50 customers × 75 users × $75/mo = $281K/mo = $3.4M ARR
  • 100 customers × 100 users × $75/mo = $750K/mo = $9M ARR

  + Setup fees: 100 × $10K = $1M one-time

TIMELINE TO $1M ARR: 12-18 months (higher ACV, fewer customers needed)
```

---

## #3: Belief-Based Code Reviewer
**"Code Review That Actually Understands Your Codebase"**

```
┌─────────────────────────────────────────────────────────────────┐
│                 BELIEF-BASED CODE REVIEWER                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  WHAT IT IS:                                                    │
│  AI code reviewer that learns your team's patterns over time:   │
│  • Builds beliefs about code conventions (from actual code)     │
│  • Detects when new code contradicts established patterns       │
│  • Tracks confidence levels (reduces noise)                     │
│  • Remembers past reviews (no repeat feedback)                  │
│  • Explains WHY something is wrong (with historical context)    │
│                                                                  │
│  DELIVERED AS:                                                   │
│  GitHub App / GitLab integration + MCP server                   │
│                                                                  │
│  UNIQUE VALUE:                                                   │
│  • Learns from YOUR code, not generic rules                     │
│  • Confidence-based filtering (no noise)                        │
│  • Temporal context ("we decided this in PR #234")              │
│  • Belief reconciliation when patterns conflict                 │
│                                                                  │
│  TARGET CUSTOMER:                                                │
│  • Engineering teams frustrated with noisy AI review tools      │
│  • Teams with complex, evolving codebases                       │
│  • Organizations wanting to preserve code review knowledge      │
│                                                                  │
│  PRICING:                                                        │
│  • Free: Open source, limited features                          │
│  • Team: $15-20/user/month                                      │
│  • Enterprise: $30-40/user (SSO, private hosting)              │
│                                                                  │
│  GO-TO-MARKET:                                                   │
│  1. GitHub Marketplace listing                                  │
│  2. "Replace your noisy AI reviewer" positioning               │
│  3. Developer-led adoption (bottom-up)                          │
│  4. Convert to team/enterprise                                  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

PROS:
  ✓ Large, growing market ($6.7B → $25.7B)
  ✓ Clear pain point (66% frustrated by "almost right")
  ✓ GitHub Marketplace = built-in distribution
  ✓ Easier to demo than team memory (immediate value)
  ✓ BeliefReconciliationService already built
  ✓ Can be used by individuals (easier adoption)
  ✓ Upsell path to full Team Knowledge Memory

CONS:
  ✗ Competitive market (CodeRabbit, Qodo, Greptile, etc.)
  ✗ Lower price point than HIPAA
  ✗ Commoditizing space
  ✗ GitHub/GitLab platform dependency
  ✗ Free tier expectations in developer tools
  ✗ Need to differentiate clearly from existing tools

REVENUE PROJECTION ($5-10M target):
  • 500 teams × 15 users × $18/mo = $135K/mo = $1.6M ARR
  • 2000 teams × 15 users × $18/mo = $540K/mo = $6.5M ARR
  • 4000 teams × 20 users × $20/mo = $1.6M/mo = $19M ARR

  (Need high volume due to lower ACV)

TIMELINE TO $1M ARR: 18-24 months
```

---

### RECOMMENDATION: Which One to Start With?

```
┌─────────────────────────────────────────────────────────────────┐
│                    FINAL RECOMMENDATION                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  START WITH: #2 HIPAA-COMPLIANT DEV MEMORY                      │
│  ═══════════════════════════════════════════                    │
│                                                                  │
│  WHY THIS ONE:                                                   │
│                                                                  │
│  1. YOU HAVE AN UNFAIR ADVANTAGE                                │
│     - CareMetx = instant pilot customer                         │
│     - Healthcare domain expertise from your job                 │
│     - Already planning Bedrock integration                      │
│     - No competitor has this specific combination               │
│                                                                  │
│  2. NICHE = DEFENSIBLE                                          │
│     - GitHub Copilot won't prioritize HIPAA compliance          │
│     - Compliance is hard to fake or fast-follow                 │
│     - Healthcare is sticky (long contracts, switching costs)    │
│     - Premium pricing is expected                               │
│                                                                  │
│  3. FASTER TO REVENUE                                            │
│     - 20 customers at $75/user × 50 users = $900K ARR           │
│     - vs. needing 500+ teams for general market                 │
│     - Healthcare companies have budget                          │
│     - Annual contracts = predictable revenue                    │
│                                                                  │
│  4. EXPANDS TO GENERAL MARKET LATER                              │
│     - Once you have team memory working for healthcare...       │
│     - Remove HIPAA-specific bits = general Team Knowledge       │
│     - Proven architecture, proven value                         │
│     - Healthcare case study validates for enterprise            │
│                                                                  │
│  5. SOLVES YOUR OWN PROBLEM                                      │
│     - You can use it at CareMetx                                │
│     - Personal Roxy + Work instance architecture                │
│     - Dogfooding is the best product development                │
│                                                                  │
│  EXECUTION PLAN:                                                 │
│  ───────────────                                                │
│  Month 1-2: Build MCP Memory Server + Bedrock integration       │
│  Month 3:   Deploy at CareMetx, dogfood                         │
│  Month 4-5: Add HIPAA audit logging, team features              │
│  Month 6:   Launch publicly, start outbound to healthcare       │
│  Month 9:   SOC 2 process begins                                │
│  Month 12:  10+ paying customers, $500K+ ARR run rate           │
│                                                                  │
│  FALLBACK:                                                       │
│  If healthcare is too slow, pivot to general Team Knowledge     │
│  Memory (#1) - same core product, different positioning         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Alternative Paths

**If you want faster feedback loops → Start with #3 (Code Reviewer)**
- Easier to demo
- GitHub Marketplace distribution
- Individual adoption, then upsell to teams
- But: more competitive, lower margins

**If you want maximum addressable market → Start with #1 (Team Knowledge)**
- Broader market
- Not limited to healthcare
- But: colder start, harder to demo, longer adoption

**The Best of Both Worlds:**
```
HIPAA Dev Memory is just Team Knowledge Memory with:
  + Bedrock instead of Groq
  + Audit logging
  + Healthcare-specific beliefs
  + Compliance positioning

Build the core once. Configure for different markets.
Healthcare first (your advantage), then expand.
```

---

### Next Steps

1. **Validate with CareMetx**: Can you actually deploy there? Any IP/conflict issues?
2. **Build MCP Memory Server**: This is the foundation for all three options
3. **Add Bedrock Provider**: Replace Groq with Bedrock for HIPAA path
4. **Create Healthcare Belief Pack**: Pre-loaded HIPAA/HL7/FHIR patterns
5. **Pilot at CareMetx**: Use it yourself for 30-60 days
6. **Document the value**: Time saved, knowledge captured, onboarding improvements
7. **Launch**: Healthcare software companies first, expand from there

---

## Archived: Original IDEAS.txt Content

The original IDEAS.txt content is preserved in `docs/IDEAS.txt` for reference. This includes:
- Detailed VS Code extension designs with ASCII diagrams
- Full competitive analysis
- Deployment cost estimates
- Technical implementation notes
