# REQ-005: Memory MCP Server

**Priority:** Medium
**Estimated Effort:** Medium (1-2 weeks)
**Dependencies:** REQ-001 (Memory System)
**Blocks:** None (Enables integration)

---

## 1. Overview

### 1.1 The Vision
From IDEAS.txt:
> "Can I replace [Claude] with my own version that can do the same things as Claude but it has my assistant's personality and my memories"

This implies:
- Multiple apps (Claude Code, VS Code, mobile) share the same memory
- Memory is the "hub" that connects everything
- Each app can use its own LLM but shares context

### 1.2 Why Memory MCP Server?

Two patterns for MCP servers were considered:

**Pattern A: Memory MCP Server** (Chosen)
```
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│  Claude Code  │   │    VS Code    │   │  Mobile App   │
│  (Claude LLM) │   │  (Copilot)    │   │  (Roxy)       │
└───────┬───────┘   └───────┬───────┘   └───────┬───────┘
        │                   │                   │
        │ MCP               │ MCP               │ MCP
        │                   │                   │
        └───────────────────┼───────────────────┘
                            ▼
                ┌───────────────────────┐
                │  Memory MCP Server    │
                │  (draagon-ai)         │
                └───────────────────────┘
```

**Pattern B: Assistant MCP Server** (Not chosen for this)
- Roxy as full assistant that Claude can delegate to
- Better for task delegation, not shared memory

### 1.3 Target State
- MCP server exposes memory operations as tools
- Claude Code can store/search memories
- All apps share the same knowledge base
- Scopes control what each app can access

### 1.4 Success Metrics
- Claude Code successfully stores/retrieves memories
- Memories created in Claude Code appear in Roxy queries
- Scope isolation works correctly
- Performance acceptable for interactive use

---

## 2. Detailed Requirements

### 2.1 MCP Server Scaffolding

**ID:** REQ-005-01
**Priority:** Critical

#### Description
Implement MCP server base following the Model Context Protocol specification.

#### MCP Server Structure
```python
# draagon_ai/mcp/server.py

class MemoryMCPServer:
    """MCP server exposing memory operations."""

    def __init__(self, memory: LayeredMemoryProvider, config: MCPConfig):
        self.memory = memory
        self.config = config
        self.tools = self._register_tools()

    def _register_tools(self) -> list[MCPTool]:
        return [
            MCPTool(
                name="memory.store",
                description="Store a memory in the shared knowledge base",
                input_schema={...},
            ),
            MCPTool(
                name="memory.search",
                description="Search the shared knowledge base",
                input_schema={...},
            ),
            # ... more tools
        ]

    async def handle_request(self, request: MCPRequest) -> MCPResponse:
        """Handle incoming MCP requests."""
        if request.method == "tools/list":
            return await self._list_tools()
        elif request.method == "tools/call":
            return await self._call_tool(request.params)
        # ... other methods
```

#### MCP Protocol Requirements
- Implements JSON-RPC 2.0 over stdio
- Supports `initialize`, `tools/list`, `tools/call` methods
- Returns proper error responses
- Handles cancellation

#### Acceptance Criteria
- [ ] Server starts and responds to `initialize`
- [ ] `tools/list` returns all available tools
- [ ] `tools/call` executes tool and returns result
- [ ] Errors are properly formatted
- [ ] Server can run as subprocess

---

### 2.2 memory.store Tool

**ID:** REQ-005-02
**Priority:** Critical

#### Description
Tool to store memories in the shared knowledge base.

#### Tool Definition
```json
{
  "name": "memory.store",
  "description": "Store a memory in the shared knowledge base. Use this to remember important information about the user, their projects, or learned facts.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "content": {
        "type": "string",
        "description": "The content to remember"
      },
      "memory_type": {
        "type": "string",
        "enum": ["fact", "skill", "insight", "preference"],
        "description": "Type of memory"
      },
      "scope": {
        "type": "string",
        "enum": ["private", "shared", "system"],
        "description": "Visibility scope"
      },
      "entities": {
        "type": "array",
        "items": {"type": "string"},
        "description": "Related entities (optional)"
      }
    },
    "required": ["content", "memory_type"]
  }
}
```

#### Acceptance Criteria
- [ ] Content is stored in appropriate memory layer
- [ ] Memory type maps to correct layer
- [ ] Scope is enforced on storage
- [ ] Entities are extracted if not provided
- [ ] Returns memory ID on success

#### Test Cases
| ID | Input | Expected Output | Type |
|----|-------|-----------------|------|
| T01 | Store fact | Memory ID returned | Integration |
| T02 | Store with entities | Entities preserved | Unit |
| T03 | Invalid scope | Error returned | Unit |

---

### 2.3 memory.search Tool

**ID:** REQ-005-03
**Priority:** Critical

#### Description
Tool to search the shared knowledge base.

#### Tool Definition
```json
{
  "name": "memory.search",
  "description": "Search the shared knowledge base for relevant memories. Use this to recall information about the user or their projects.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "Search query"
      },
      "limit": {
        "type": "integer",
        "default": 5,
        "description": "Maximum results to return"
      },
      "memory_types": {
        "type": "array",
        "items": {"type": "string"},
        "description": "Filter by memory types (optional)"
      }
    },
    "required": ["query"]
  }
}
```

#### Acceptance Criteria
- [ ] Semantic search works across layers
- [ ] Results are ranked by relevance
- [ ] Scope filtering applied (only visible memories)
- [ ] Memory types filter works
- [ ] Returns formatted results

---

### 2.4 memory.list Tool

**ID:** REQ-005-04
**Priority:** Medium

#### Description
Tool to list recent memories without search.

#### Tool Definition
```json
{
  "name": "memory.list",
  "description": "List recent memories, optionally filtered by type.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "memory_type": {
        "type": "string",
        "description": "Filter by type (optional)"
      },
      "limit": {
        "type": "integer",
        "default": 10
      }
    }
  }
}
```

---

### 2.5 beliefs.reconcile Tool

**ID:** REQ-005-05
**Priority:** Medium

#### Description
Tool to add observations that get reconciled into beliefs.

#### Tool Definition
```json
{
  "name": "beliefs.reconcile",
  "description": "Add an observation that will be reconciled with existing beliefs. Use this when you learn something that might conflict with existing knowledge.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "observation": {
        "type": "string",
        "description": "The observation to reconcile"
      },
      "source": {
        "type": "string",
        "description": "Source of the observation (e.g., 'user', 'web', 'code')"
      },
      "confidence": {
        "type": "number",
        "minimum": 0,
        "maximum": 1,
        "default": 0.8
      }
    },
    "required": ["observation"]
  }
}
```

#### Acceptance Criteria
- [ ] Observation is recorded
- [ ] Conflicts with existing beliefs detected
- [ ] Reconciliation result returned
- [ ] Belief updated or conflict noted

---

### 2.6 Scope-Based Access Control

**ID:** REQ-005-06
**Priority:** High

#### Description
Enforce scopes so MCP clients only access appropriate memories.

#### Scope Mapping
```
MCP Scope → Memory Scope:
- "private" → USER (per-user within agent)
- "shared" → CONTEXT (shared within context/household)
- "system" → WORLD (global facts)
```

#### Access Rules
```python
def check_access(client_id: str, memory_scope: str, operation: str) -> bool:
    client_scopes = get_client_scopes(client_id)

    if operation == "read":
        # Can read own scope and higher
        return memory_scope in allowed_read_scopes(client_scopes)
    elif operation == "write":
        # Can only write to own scope or lower
        return memory_scope in allowed_write_scopes(client_scopes)
```

#### Acceptance Criteria
- [ ] Clients can only read appropriate scopes
- [ ] Clients can only write to appropriate scopes
- [ ] Cross-client isolation works
- [ ] Scope violations logged

---

### 2.7 Authentication/Authorization

**ID:** REQ-005-07
**Priority:** High

#### Description
Secure the MCP server with authentication.

#### Authentication Options
1. **API Key** - Simple, for trusted local apps
2. **JWT** - For more complex scenarios
3. **mTLS** - For high security

#### Initial Implementation (API Key)
```python
class MCPAuthenticator:
    def __init__(self, valid_keys: dict[str, ClientConfig]):
        self.valid_keys = valid_keys

    async def authenticate(self, request: MCPRequest) -> ClientContext:
        api_key = request.headers.get("X-API-Key")
        if not api_key or api_key not in self.valid_keys:
            raise AuthenticationError("Invalid API key")

        client_config = self.valid_keys[api_key]
        return ClientContext(
            client_id=client_config.client_id,
            allowed_scopes=client_config.scopes,
        )
```

#### Acceptance Criteria
- [ ] API key authentication works
- [ ] Invalid keys rejected
- [ ] Client context established on auth
- [ ] Audit log of auth attempts

---

### 2.8 Claude Code Integration Test

**ID:** REQ-005-08
**Priority:** Critical

#### Description
Verify the MCP server works with Claude Code.

#### Test Scenario
```bash
# 1. Start MCP server
python -m draagon_ai.mcp.server --config config.yaml

# 2. Configure Claude Code to use server
# In claude_desktop_config.json:
{
  "mcpServers": {
    "memory": {
      "command": "python",
      "args": ["-m", "draagon_ai.mcp.server"],
      "env": {
        "QDRANT_URL": "http://192.168.168.216:6333"
      }
    }
  }
}

# 3. Test in Claude Code:
# User: "Remember that my favorite color is blue"
# Claude: [Uses memory.store tool]
# User: "What's my favorite color?"
# Claude: [Uses memory.search tool, finds "blue"]
```

#### Acceptance Criteria
- [ ] Claude Code discovers memory tools
- [ ] Tool calls work correctly
- [ ] Results are useful to Claude
- [ ] Performance is acceptable (<2s per call)

---

### 2.9 Unit Tests

**ID:** REQ-005-09
**Priority:** High

#### Coverage Requirements
- All tools tested
- Authentication tested
- Scope enforcement tested
- Error handling tested

---

### 2.10 Integration Tests

**ID:** REQ-005-10
**Priority:** High

#### Test Scenarios
1. Full request/response cycle
2. Tool execution with real Qdrant
3. Multi-client scope isolation
4. Concurrent access handling

---

## 3. Implementation Plan

### 3.1 Sequence
1. MCP server scaffolding (REQ-005-01)
2. memory.store tool (REQ-005-02)
3. memory.search tool (REQ-005-03)
4. memory.list tool (REQ-005-04)
5. beliefs.reconcile tool (REQ-005-05)
6. Scope access control (REQ-005-06)
7. Authentication (REQ-005-07)
8. Claude Code integration test (REQ-005-08)
9. Unit tests (REQ-005-09)
10. Integration tests (REQ-005-10)

### 3.2 Risks
| Risk | Mitigation |
|------|------------|
| MCP spec changes | Pin to stable version |
| Claude Code compatibility | Test early and often |
| Performance issues | Benchmark, optimize queries |
| Security vulnerabilities | Security review, API key rotation |

---

## 4. Review Checklist

### Functional Completeness
- [ ] All tools work correctly
- [ ] Scopes enforce isolation
- [ ] Authentication works
- [ ] Claude Code integration verified

### MCP Compliance
- [ ] JSON-RPC 2.0 correct
- [ ] Tool schemas valid
- [ ] Error responses proper
- [ ] Cancellation works

### Security
- [ ] Authentication required
- [ ] Scope violations blocked
- [ ] No information leakage
- [ ] Audit logging complete

### Performance
- [ ] Tool calls < 2s
- [ ] Search results relevant
- [ ] No memory leaks

---

## 5. God-Level Review Prompt

```
MCP SERVER REVIEW - REQ-005

Context: Memory MCP Server enabling Claude Code and other apps
to share a common knowledge base via Model Context Protocol.

Review the implementation against these specific criteria:

1. MCP PROTOCOL COMPLIANCE
   - Is JSON-RPC 2.0 implemented correctly?
   - Are tool schemas valid and complete?
   - Are error responses properly formatted?
   - Does cancellation work?
   - Can Claude Code actually use this?

2. TOOL FUNCTIONALITY
   - Does memory.store persist correctly?
   - Does memory.search return relevant results?
   - Does memory.list work as expected?
   - Does beliefs.reconcile handle conflicts?

3. SCOPE ISOLATION
   - Can clients only access their scopes?
   - Are write permissions correct?
   - Is cross-client isolation verified?
   - Are scope violations logged?

4. SECURITY
   - Is authentication required?
   - Are API keys validated properly?
   - Is there any path to bypass auth?
   - Are there information leakage risks?

5. CLAUDE CODE INTEGRATION
   - Does Claude Code discover tools?
   - Do tool calls work in practice?
   - Is the UX good?
   - Is performance acceptable?

Provide specific code references for any issues found.
Rate each section: PASS / NEEDS_WORK / FAIL
Overall recommendation: READY / NOT_READY
```

