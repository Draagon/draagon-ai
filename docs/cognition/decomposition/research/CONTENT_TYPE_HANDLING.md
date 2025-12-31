# Content Type Handling for Semantic Processing

## Research Question

When users provide context (code, data, documents), how do we extract semantic knowledge that improves LLM context rather than degrading it?

## The Core Problem

Traditional RAG retrieves raw chunks. Our goal is **semantic compression**: extract the *meaning* so we can provide better context with fewer tokens.

But not all content has the same kind of meaning:

| Content Type | Has Sentence Structure? | Semantic Meaning Location |
|--------------|------------------------|---------------------------|
| Prose | Yes | In the sentences themselves |
| Source Code | Partially (comments/docs) | In identifiers, structure, relationships |
| CSV/Tabular | No | In column names, relationships, patterns |
| JSON/YAML | No | In keys, structure, values |
| Logs | Partially | In patterns, anomalies, sequences |
| API Docs | Yes + structured | In descriptions + schema relationships |
| Math/Formulas | No | In symbolic relationships |

## Content Type Taxonomy

### Category 1: Natural Language (Full WSD)
- **Prose documents** - articles, docs, emails
- **Comments and docstrings** - embedded NL in code
- **Error messages and logs** - often natural language
- **Chat transcripts**

→ **Processing**: Full WSD pipeline, extract implicit knowledge

### Category 2: Structured with NL Annotations (Hybrid)
- **Source code** - NL in comments/docs, structure in code
- **API specifications** (OpenAPI) - descriptions + schemas
- **Database schemas** - column comments + relationships
- **Configuration with comments** - YAML/TOML with annotations

→ **Processing**: Extract NL portions for WSD, extract structure separately

### Category 3: Pure Structure (No WSD)
- **CSV/TSV data** - column names + value patterns
- **JSON data** - key names + structure
- **SQL queries** - table/column relationships
- **File trees** - naming conventions + hierarchy

→ **Processing**: Extract schema, relationships, patterns - not word senses

### Category 4: Symbolic (Domain-Specific)
- **Mathematical notation**
- **Chemical formulas**
- **Musical notation**
- **Regex patterns**

→ **Processing**: Domain-specific parsing, not general WSD

## What "Semantic Knowledge" Means Per Type

### For Prose
- Word senses (WSD)
- Presuppositions and implications
- Entity relationships
- Temporal/causal structure

### For Source Code
```python
# Code carries different kinds of semantic information:

def process_bank_transaction(account: BankAccount, amount: float) -> Receipt:
    """Transfer funds to a financial institution account."""
    # ↑ NL: "financial institution" disambiguates "bank"

    # Code semantics (NOT WordNet):
    # - account has type BankAccount (structural)
    # - function returns Receipt (contract)
    # - amount is float (constraint)

    if amount <= 0:
        raise ValueError("Amount must be positive")
        # ↑ NL: Error message is natural language
```

**Code semantic knowledge includes:**
- Type relationships (BankAccount is-a Account)
- Function contracts (inputs → outputs)
- Control flow patterns (if-then, try-except)
- Naming conventions → domain concepts
- Import relationships → dependencies

### For Tabular Data (CSV)
```csv
customer_id,account_type,balance,last_transaction_date
1001,savings,5000.00,2024-01-15
1002,checking,1250.50,2024-01-20
```

**Tabular semantic knowledge includes:**
- Column semantics (customer_id is identifier, balance is monetary)
- Relationships (customer has account_type)
- Constraints (balance is numeric, date format)
- Statistical patterns (ranges, distributions)
- NOT: word senses of individual values

### For JSON/Config
```json
{
  "database": {
    "host": "localhost",
    "port": 5432,
    "name": "production_db"
  }
}
```

**Config semantic knowledge:**
- Key hierarchies (database contains host)
- Value types and constraints
- Common patterns (this is a DB config)
- NOT: disambiguating "host" as WordNet noun

## The Key Insight

**WSD is for natural language understanding, not all semantic understanding.**

Different content types need different semantic extraction:

| Content | Extract What | Store As |
|---------|--------------|----------|
| Prose | Word senses, implications, entities | Semantic graph |
| Code | Types, contracts, relationships, NL comments | Code knowledge graph |
| Data | Schema, patterns, constraints | Data profile |
| Config | Structure, common patterns | Config schema |

## LLM-Driven Content Type Detection

Instead of regex heuristics, use LLM to classify and extract:

```xml
<prompt>
Analyze this content and identify:
1. Content type (prose, code, data, config, mixed)
2. Programming language (if code)
3. Data format (if structured)
4. Natural language portions (comments, descriptions, strings)
5. Structural relationships worth preserving
6. What semantic knowledge would help understand this?

Content:
{content}

Output XML:
<analysis>
  <content_type>mixed</content_type>
  <components>
    <component type="code" language="python" lines="1-50"/>
    <component type="docstring" lines="2-5">
      <natural_language>Transfer funds to a financial institution...</natural_language>
    </component>
    <component type="comment" lines="10">
      <natural_language>Validate amount before processing</natural_language>
    </component>
  </components>
  <structural_knowledge>
    <relationship>BankAccount is parameter type</relationship>
    <relationship>Receipt is return type</relationship>
    <relationship>ValueError raised when amount invalid</relationship>
  </structural_knowledge>
  <recommended_processing>
    <for type="docstring">full_wsd</for>
    <for type="code">extract_types_and_contracts</for>
  </recommended_processing>
</analysis>
</prompt>
```

## Storage Strategy

### Short-Term (Working Memory)
- **All content types**: Store raw for immediate use
- **Why**: LLM may need exact syntax for code assistance

### Medium-Term (Episodic)
- **Prose**: Store semantic decomposition
- **Code**: Store summarized understanding + key snippets
- **Data**: Store schema + sample + statistics

### Long-Term (Semantic)
- **Prose**: Store extracted knowledge (facts, relationships)
- **Code**: Store API contracts, type hierarchies, patterns
- **Data**: Store schema evolution, constraint knowledge
- **Config**: Store common patterns, valid structures

## Context Enhancement vs Replacement

**Goal**: Provide BETTER context, not replace all context.

### When to Use Raw Content
- User explicitly asks about syntax/formatting
- Debugging specific code
- Data validation tasks
- Recent/changing content

### When to Use Semantic Knowledge
- Answering questions about concepts
- Finding related information
- Understanding intent
- Long-term recall

### Hybrid Approach
```
User: "How does our authentication work?"

Context provided to LLM:
1. Semantic knowledge: "System uses JWT tokens,
   AuthService validates, tokens expire in 24h"
2. Key code snippet: actual auth middleware (100 lines, not 1000)
3. Related facts: "Added OAuth support in v2.1"
```

## Processing Pipeline Design

```
Input Content
      │
      ▼
┌─────────────────────────┐
│ LLM: Classify Content   │
│ - Type detection        │
│ - Component extraction  │
│ - Processing strategy   │
└─────────────────────────┘
      │
      ├─── Prose ──────────► WSD Pipeline ──► Semantic Graph
      │
      ├─── Code ───────────► Code Analyzer ──► Code Knowledge
      │    └── NL portions ─► WSD Pipeline     + NL Semantics
      │
      ├─── Data ───────────► Schema Extractor ──► Data Profile
      │
      └─── Config ─────────► Pattern Matcher ──► Config Schema

All paths ──► Unified Knowledge Store
                    │
                    ▼
            Context Generation
            (blend semantic + raw as needed)
```

## What We Should NOT Do

1. **Force WSD on code syntax** - `bank.deposit()` is not about financial institutions
2. **Lose structure** - Flattening JSON to prose loses the hierarchy
3. **Over-summarize** - Sometimes exact values matter
4. **Single representation** - Different queries need different context

## Open Questions

1. **How to handle mixed content?** (markdown with code blocks)
2. **When is raw better than semantic?** (need heuristics or LLM decision)
3. **How to version semantic knowledge?** (code changes → knowledge updates)
4. **Cross-type relationships?** (code implements concept from docs)

## Recommended Implementation Order

1. **Content type classifier** (LLM-based)
2. **NL extractor for code** (docstrings, comments, strings)
3. **Schema extractor for data** (CSV, JSON)
4. **Unified knowledge store** (different types, one interface)
5. **Smart context generator** (blend semantic + raw)

## Success Metrics

- **Context efficiency**: Same quality answers with fewer tokens
- **Accuracy preservation**: No degradation from semantic extraction
- **Cross-reference**: Can answer "what code implements X concept?"
- **Staleness detection**: Know when semantic knowledge is outdated
