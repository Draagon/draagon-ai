# Draagon AI Prototypes

This folder contains experimental prototypes that explore new cognitive architecture concepts before they're integrated into the core draagon-ai framework.

## Philosophy

Prototypes are **research experiments**, not production code. They allow us to:

1. **Explore ideas freely** without worrying about breaking existing functionality
2. **Compare approaches** side-by-side before committing to one
3. **Iterate quickly** on concepts that may or may not work out
4. **Test hypotheses** with real code before architectural decisions

## Prototype Structure

Each prototype lives in its own folder with this structure:

```
prototypes/
├── your_prototype_name/
│   ├── CLAUDE.md           # REQUIRED: Prototype-specific Claude context
│   ├── README.md           # REQUIRED: Quick overview, status
│   ├── docs/               # Self-contained documentation
│   │   ├── research/       # Background, prior art, concepts
│   │   ├── requirements/   # FR-xxx requirement documents
│   │   ├── specs/          # Technical architecture specs
│   │   └── findings/       # Experiment results, learnings
│   ├── src/                # The prototype implementation
│   │   ├── __init__.py
│   │   └── ...
│   ├── tests/              # Tests proving it works
│   │   ├── conftest.py     # Path setup for imports
│   │   └── test_*.py
│   └── requirements.txt    # Extra dependencies (optional)
```

**Key Files:**
- `CLAUDE.md` - Claude-specific context for working on this prototype
- `docs/` - Self-contained documentation (moves with the prototype)

## Creating a New Prototype

1. **Create the folder structure:**
   ```bash
   mkdir -p prototypes/my_idea/{src,tests,docs/{research,requirements,specs,findings}}
   ```

2. **Create CLAUDE.md** - Claude-specific context:
   ```markdown
   # My Idea Prototype - Claude Context

   **Status:** Experimental
   **Last Updated:** YYYY-MM-DD

   ## Overview
   This prototype explores [your idea here].

   ## Hypothesis
   We believe that [approach] will improve [metric] because [reasoning].

   ## Key Concepts
   [Architecture diagram, key patterns]

   ## File Structure
   [What each file does]

   ## Important Patterns
   [Coding conventions, gotchas]

   ## Integration Readiness
   **Status:** NOT ready
   **Blocking Issues:** [list]
   ```

3. **Write README.md** - quick overview:
   ```markdown
   # My Idea Prototype

   ## Status
   - [ ] Initial implementation
   - [ ] Basic tests passing
   - [ ] Compared against baseline
   - [ ] Ready for integration consideration
   ```

4. **Implement in src/** - keep it self-contained
   - Import stable types from draagon-ai (Memory, MemoryType, etc.)
   - Don't import from core orchestration (not wired in yet!)
   - Use protocols/interfaces that match what draagon-ai expects

5. **Add tests/conftest.py** for path setup:
   ```python
   import sys
   from pathlib import Path

   prototype_root = Path(__file__).parent.parent
   sys.path.insert(0, str(prototype_root / "src"))
   sys.path.insert(0, str(prototype_root.parent.parent / "src"))
   ```

6. **Test in tests/** - prove your concept works
   - Unit tests for components
   - Integration tests for the full flow
   - Comparison tests against baseline (if applicable)

## Running Prototype Tests

```bash
# Run tests for a specific prototype
pytest prototypes/semantic_expansion/tests/ -v

# Run all prototype tests
pytest prototypes/ -v
```

## When a Prototype is Ready for Integration

A prototype is ready when:

1. ✅ Core concept is validated with tests
2. ✅ Performance is acceptable
3. ✅ Clear integration path identified
4. ✅ README documents learnings and decisions
5. ✅ Code reviewed for production quality

Then create an integration plan:
1. Identify which core files need changes
2. Write the integration as a separate PR
3. Ensure existing tests still pass
4. Add new tests for the integrated feature

## Current Prototypes

| Prototype | Status | Description |
|-----------|--------|-------------|
| [semantic_expansion](./semantic_expansion/) | 🧪 Experimental | Two-pass semantic expansion with WSD |

---

**Remember:** Prototypes are experiments. Most will teach us something valuable even if they don't make it to production.
