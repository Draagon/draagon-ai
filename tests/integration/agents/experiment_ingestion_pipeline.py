#!/usr/bin/env python3
"""Experiment: Full Ingestion Pipeline for Lore/Documentation Files

This experiment tests the complete pipeline for ingesting documentation:

1. DISCOVERY: Agentic reading of CLAUDE.md to find related files
2. CHUNKING: Semantic splitting of large documents
3. HYBRID STORAGE: Raw + extracted for each chunk
4. RETRIEVAL: Compare context-file vs RAG approaches

The goal is to validate that we can replace large context files with
intelligent RAG/Neo4j retrieval that uses the same pipeline as everything else.

Run with:
    GROQ_API_KEY=your_key python3.11 tests/integration/agents/experiment_ingestion_pipeline.py
"""

import asyncio
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

# Add project to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class Chunk:
    """A semantic chunk of a document."""
    id: str
    source_file: str
    content: str
    raw_content: str  # Original text
    extracted_entities: list[str] = field(default_factory=list)
    extracted_concepts: list[str] = field(default_factory=list)
    heading_path: list[str] = field(default_factory=list)  # e.g., ["Memory", "4-Layer Architecture"]
    chunk_index: int = 0
    total_chunks: int = 0


@dataclass
class FileReference:
    """A file reference discovered in documentation."""
    path: str
    context: str  # Why this file was referenced
    relevance: float  # 0-1, how relevant to the main topic


@dataclass
class IngestionResult:
    """Result of ingesting a set of files."""
    chunks: list[Chunk]
    total_files: int
    total_chunks: int
    extraction_time_ms: float
    files_discovered: list[str]


# =============================================================================
# Semantic Chunking
# =============================================================================

class SemanticChunker:
    """Splits documents into semantic chunks based on structure."""

    def __init__(self,
                 target_chunk_size: int = 500,  # tokens (roughly) - lowered for more chunks
                 min_chunk_size: int = 50,  # Lowered to keep more content
                 overlap_size: int = 50):
        self.target_chunk_size = target_chunk_size
        self.min_chunk_size = min_chunk_size
        self.overlap_size = overlap_size

    def chunk_markdown(self, content: str, source_file: str) -> list[Chunk]:
        """Split markdown content into semantic chunks."""
        chunks = []

        # Split by headers first
        sections = self._split_by_headers(content)

        chunk_index = 0
        for heading_path, section_content in sections:
            section_tokens = self._estimate_tokens(section_content)

            # Skip truly empty sections
            if section_tokens < 10:
                continue

            # If section is small enough, keep as one chunk
            if section_tokens <= self.target_chunk_size:
                chunks.append(Chunk(
                    id=f"{source_file}:{chunk_index}",
                    source_file=source_file,
                    content=section_content,
                    raw_content=section_content,
                    heading_path=heading_path,
                    chunk_index=chunk_index,
                ))
                chunk_index += 1
            else:
                # Split large sections by paragraphs
                sub_chunks = self._split_large_section(section_content, heading_path)
                for sub in sub_chunks:
                    sub.id = f"{source_file}:{chunk_index}"
                    sub.source_file = source_file
                    sub.chunk_index = chunk_index
                    chunks.append(sub)
                    chunk_index += 1

        # Update total_chunks
        for chunk in chunks:
            chunk.total_chunks = len(chunks)

        return chunks

    def _split_by_headers(self, content: str) -> list[tuple[list[str], str]]:
        """Split content by markdown headers, tracking heading hierarchy."""
        sections = []
        current_headings = []
        current_content = []

        lines = content.split('\n')

        for line in lines:
            # Check for markdown header
            header_match = re.match(r'^(#{1,6})\s+(.+)$', line)
            if header_match:
                # Save previous section if it has content
                if current_content:
                    text = '\n'.join(current_content).strip()
                    if text:
                        sections.append((current_headings.copy(), text))
                    current_content = []

                # Update heading hierarchy
                level = len(header_match.group(1))
                heading_text = header_match.group(2)

                # Trim headings to current level
                current_headings = current_headings[:level-1]
                current_headings.append(heading_text)
            else:
                current_content.append(line)

        # Don't forget the last section
        if current_content:
            text = '\n'.join(current_content).strip()
            if text:
                sections.append((current_headings.copy(), text))

        return sections

    def _split_large_section(self, content: str, heading_path: list[str]) -> list[Chunk]:
        """Split a large section into smaller chunks by paragraphs."""
        chunks = []
        paragraphs = re.split(r'\n\s*\n', content)

        current_chunk = []
        current_size = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            para_size = self._estimate_tokens(para)

            # If adding this paragraph exceeds target, start new chunk
            if current_size + para_size > self.target_chunk_size and current_chunk:
                chunk_text = '\n\n'.join(current_chunk)
                chunks.append(Chunk(
                    id="",  # Will be set later
                    source_file="",
                    content=chunk_text,
                    raw_content=chunk_text,
                    heading_path=heading_path.copy(),
                    chunk_index=0,
                ))

                # Start new chunk (no overlap for simplicity)
                current_chunk = []
                current_size = 0

            current_chunk.append(para)
            current_size += para_size

        # Don't forget the last chunk - keep even if small
        if current_chunk:
            chunk_text = '\n\n'.join(current_chunk)
            chunks.append(Chunk(
                id="",
                source_file="",
                content=chunk_text,
                raw_content=chunk_text,
                heading_path=heading_path.copy(),
                chunk_index=0,
            ))

        return chunks

    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimate (words * 1.3)."""
        return int(len(text.split()) * 1.3)


# =============================================================================
# Agentic File Discovery
# =============================================================================

class AgenticFileDiscovery:
    """Discovers relevant files by reading documentation."""

    def __init__(self, llm, base_path: Path):
        self.llm = llm
        self.base_path = base_path

    async def discover_from_claude_md(self, claude_md_content: str) -> list[FileReference]:
        """Extract file references from CLAUDE.md content."""
        prompt = f"""Analyze this CLAUDE.md documentation and extract all file/path references.
For each reference, note:
1. The file path (relative or pattern like "src/**/*.py")
2. Why it's referenced (what it contains/does)
3. How relevant it seems (0.0-1.0) to the core functionality

CLAUDE.md content:
{claude_md_content[:8000]}

Output format (one per line):
PATH: path/to/file | CONTEXT: what it contains | RELEVANCE: 0.X

Only include actual file paths, not conceptual references."""

        response = await self.llm.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=1000,
        )

        references = []
        for line in response.content.split('\n'):
            if line.startswith('PATH:'):
                parts = line.split('|')
                if len(parts) >= 3:
                    try:
                        path = parts[0].replace('PATH:', '').strip()
                        context = parts[1].replace('CONTEXT:', '').strip()
                        relevance = float(parts[2].replace('RELEVANCE:', '').strip())
                        references.append(FileReference(path=path, context=context, relevance=relevance))
                    except:
                        pass

        return references

    async def expand_file_patterns(self, references: list[FileReference]) -> list[str]:
        """Expand glob patterns to actual files."""
        files = []
        for ref in references:
            path = ref.path

            # Handle glob patterns
            if '*' in path:
                pattern_path = self.base_path / path
                matching = list(self.base_path.glob(path))
                files.extend([str(f.relative_to(self.base_path)) for f in matching if f.is_file()])
            else:
                full_path = self.base_path / path
                if full_path.exists() and full_path.is_file():
                    files.append(path)

        return list(set(files))  # Dedupe


# =============================================================================
# Hybrid Storage
# =============================================================================

def parse_section(content: str, prefix: str) -> list[str]:
    """Parse extraction output."""
    lines = content.split("\n")
    items = []
    in_section = False

    for line in lines:
        line = line.strip()
        if line.upper().startswith(prefix.upper()):
            in_section = True
            after_prefix = line[len(prefix):].strip()
            if after_prefix.startswith(":"):
                after_prefix = after_prefix[1:].strip()
            if after_prefix:
                items.extend([x.strip() for x in after_prefix.split(",") if x.strip()])
            continue

        if in_section:
            if any(line.upper().startswith(p) for p in ["ENTITIES", "CONCEPTS"]):
                break
            if line.startswith("-") or line.startswith("*"):
                item = line[1:].strip()
                if item:
                    items.append(item)

    return items


class HybridStore:
    """Stores chunks with both raw and extracted content."""

    def __init__(self, llm):
        self.llm = llm
        self.chunks: list[Chunk] = []

    async def add_chunk(self, chunk: Chunk) -> Chunk:
        """Add chunk with extraction."""
        extraction = await self._extract(chunk.content)
        chunk.extracted_entities = extraction["entities"]
        chunk.extracted_concepts = extraction["concepts"]
        self.chunks.append(chunk)
        return chunk

    async def _extract(self, text: str) -> dict:
        """Extract entities and concepts."""
        prompt = f"""Extract key entities and concepts from this text.

Text: {text[:1500]}

Output format (comma-separated):
ENTITIES: entity1, entity2, entity3
CONCEPTS: concept1, concept2, concept3"""

        response = await self.llm.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=200,
        )

        return {
            "entities": parse_section(response.content, "ENTITIES"),
            "concepts": parse_section(response.content, "CONCEPTS"),
        }

    def search_raw(self, query_words: set[str], k: int = 5) -> list[tuple[Chunk, float]]:
        """Search using raw content."""
        results = []
        for chunk in self.chunks:
            content_words = set(chunk.raw_content.lower().split())
            overlap = len(query_words & content_words)
            if overlap > 0:
                results.append((chunk, overlap / len(query_words)))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]

    def search_extracted(self, query_terms: set[str], k: int = 5) -> list[tuple[Chunk, float]]:
        """Search using extracted entities/concepts."""
        results = []
        for chunk in self.chunks:
            chunk_terms = set(
                [e.lower() for e in chunk.extracted_entities] +
                [c.lower() for c in chunk.extracted_concepts]
            )
            if chunk_terms:
                # Exact matches
                overlap = len(query_terms & chunk_terms)
                # Partial matches
                partial = sum(0.5 for qt in query_terms for ct in chunk_terms
                             if qt in ct or ct in qt and qt != ct)
                score = (overlap + partial) / max(len(query_terms), 1)
                if score > 0:
                    results.append((chunk, score))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]

    def search_hybrid(self, query: str, query_extraction: dict, k: int = 5) -> list[tuple[Chunk, float]]:
        """Search using both raw and extracted, merge results."""
        query_words = set(query.lower().split())
        query_terms = set(
            [e.lower() for e in query_extraction.get("entities", [])] +
            [c.lower() for c in query_extraction.get("concepts", [])]
        )

        raw_results = self.search_raw(query_words, k * 2)
        ext_results = self.search_extracted(query_terms, k * 2) if query_terms else []

        # Merge with consensus boost
        all_results = {}
        for chunk, score in raw_results:
            all_results[chunk.id] = {"chunk": chunk, "raw": score, "ext": 0}
        for chunk, score in ext_results:
            if chunk.id in all_results:
                all_results[chunk.id]["ext"] = score
            else:
                all_results[chunk.id] = {"chunk": chunk, "raw": 0, "ext": score}

        merged = []
        for data in all_results.values():
            score = max(data["raw"], data["ext"])
            if data["raw"] > 0 and data["ext"] > 0:
                score += 0.2  # Consensus boost
            merged.append((data["chunk"], score))

        merged.sort(key=lambda x: x[1], reverse=True)
        return merged[:k]


# =============================================================================
# Main Experiment
# =============================================================================

async def main():
    """Run the ingestion pipeline experiment."""
    from groq import AsyncGroq

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("ERROR: GROQ_API_KEY not set")
        sys.exit(1)

    client = AsyncGroq(api_key=api_key)

    class GroqLLM:
        async def chat(self, messages, temperature=0.7, max_tokens=1000):
            resp = await client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message

    llm = GroqLLM()

    print("=" * 80)
    print("EXPERIMENT: Full Ingestion Pipeline")
    print("=" * 80)
    print()
    print("Testing the complete pipeline for ingesting documentation into RAG/Neo4j")
    print()

    # Use the actual CLAUDE.md from this project
    claude_md_path = project_root / "CLAUDE.md"
    if not claude_md_path.exists():
        print(f"ERROR: {claude_md_path} not found")
        sys.exit(1)

    claude_md_content = claude_md_path.read_text()
    print(f"Loaded CLAUDE.md: {len(claude_md_content)} chars")

    # ==========================================================================
    # Phase 1: Agentic File Discovery
    # ==========================================================================
    print()
    print("-" * 80)
    print("PHASE 1: Agentic File Discovery")
    print("-" * 80)

    discovery = AgenticFileDiscovery(llm, project_root)

    start = time.time()
    references = await discovery.discover_from_claude_md(claude_md_content)
    discovery_time = (time.time() - start) * 1000

    print(f"\nDiscovered {len(references)} file references in {discovery_time:.0f}ms:")
    for ref in references[:10]:
        print(f"  {ref.path} (relevance: {ref.relevance:.1f})")
        print(f"    → {ref.context[:60]}...")

    if len(references) > 10:
        print(f"  ... and {len(references) - 10} more")

    # ==========================================================================
    # Phase 2: Semantic Chunking
    # ==========================================================================
    print()
    print("-" * 80)
    print("PHASE 2: Semantic Chunking")
    print("-" * 80)

    chunker = SemanticChunker(target_chunk_size=600, min_chunk_size=150)

    start = time.time()
    chunks = chunker.chunk_markdown(claude_md_content, "CLAUDE.md")
    chunking_time = (time.time() - start) * 1000

    print(f"\nCreated {len(chunks)} chunks in {chunking_time:.0f}ms:")
    for i, chunk in enumerate(chunks[:5]):
        heading = " > ".join(chunk.heading_path) if chunk.heading_path else "(root)"
        print(f"  [{i}] {heading}")
        print(f"      {chunk.content[:80]}...")

    if len(chunks) > 5:
        print(f"  ... and {len(chunks) - 5} more chunks")

    # ==========================================================================
    # Phase 3: Hybrid Storage with Extraction
    # ==========================================================================
    print()
    print("-" * 80)
    print("PHASE 3: Hybrid Storage (Raw + Extracted)")
    print("-" * 80)

    store = HybridStore(llm)

    start = time.time()
    # Process more chunks (up to 25) for better coverage
    chunks_to_process = min(25, len(chunks))
    for i, chunk in enumerate(chunks[:chunks_to_process]):
        await store.add_chunk(chunk)
        if (i + 1) % 5 == 0:
            print(f"  Stored {i + 1}/{chunks_to_process} chunks...")
    storage_time = (time.time() - start) * 1000

    print(f"\nStored {len(store.chunks)} chunks in {storage_time:.0f}ms")
    print(f"Sample extractions:")
    for chunk in store.chunks[:3]:
        print(f"  [{chunk.chunk_index}] Entities: {chunk.extracted_entities[:3]}...")
        print(f"      Concepts: {chunk.extracted_concepts[:3]}...")

    # ==========================================================================
    # Phase 4: Retrieval Comparison
    # ==========================================================================
    print()
    print("-" * 80)
    print("PHASE 4: Retrieval Comparison")
    print("-" * 80)

    test_queries = [
        {
            "query": "How does the 4-layer memory architecture work?",
            "expected_terms": ["working", "episodic", "semantic", "metacognitive"],
        },
        {
            "query": "What is the @tool decorator used for?",
            "expected_terms": ["tool", "decorator", "registry", "handler"],
        },
        {
            "query": "How do beliefs get reconciled?",
            "expected_terms": ["belief", "reconciliation", "conflict", "observation"],
        },
    ]

    results = {"raw": [], "extracted": [], "hybrid": [], "hybrid_expanded": [], "full_context": []}

    # Query expansion helper
    async def expand_query(query: str) -> list[str]:
        """Generate query variations."""
        prompt = f"""Generate 2 alternative phrasings of this question that might help find relevant documentation.

Original: {query}

Output format (one per line):
1. [first alternative]
2. [second alternative]"""

        response = await llm.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=150,
        )

        expansions = [query]  # Always include original
        for line in response.content.split('\n'):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-')):
                # Remove leading number/bullet
                exp = re.sub(r'^[\d\.\-\*]+\s*', '', line).strip()
                if exp and len(exp) > 10:
                    expansions.append(exp)

        return expansions[:3]  # Max 3 total

    for test in test_queries:
        query = test["query"]
        expected = set(t.lower() for t in test["expected_terms"])

        print(f"\nQuery: \"{query}\"")

        # Extract query for search
        query_extraction = await store._extract(query)

        # Method 1: Raw search
        raw_results = store.search_raw(set(query.lower().split()), k=5)
        raw_found = set()
        for chunk, score in raw_results:
            for term in expected:
                if term in chunk.raw_content.lower():
                    raw_found.add(term)
        results["raw"].append(len(raw_found) / len(expected))

        # Method 2: Extracted search
        query_terms = set(
            [e.lower() for e in query_extraction.get("entities", [])] +
            [c.lower() for c in query_extraction.get("concepts", [])]
        )
        ext_results = store.search_extracted(query_terms, k=5)
        ext_found = set()
        for chunk, score in ext_results:
            for term in expected:
                if term in chunk.raw_content.lower():
                    ext_found.add(term)
        results["extracted"].append(len(ext_found) / len(expected))

        # Method 3: Hybrid search
        hybrid_results = store.search_hybrid(query, query_extraction, k=5)
        hybrid_found = set()
        for chunk, score in hybrid_results:
            for term in expected:
                if term in chunk.raw_content.lower():
                    hybrid_found.add(term)
        results["hybrid"].append(len(hybrid_found) / len(expected))

        # Method 4: Hybrid + Query Expansion (the full pipeline)
        expansions = await expand_query(query)
        print(f"  Expansions: {expansions}")

        all_expanded_results = {}
        for exp_query in expansions:
            exp_extraction = await store._extract(exp_query)
            exp_results = store.search_hybrid(exp_query, exp_extraction, k=3)
            for chunk, score in exp_results:
                if chunk.id not in all_expanded_results:
                    all_expanded_results[chunk.id] = {"chunk": chunk, "score": score, "sources": 1}
                else:
                    all_expanded_results[chunk.id]["sources"] += 1
                    all_expanded_results[chunk.id]["score"] = max(
                        all_expanded_results[chunk.id]["score"],
                        score + 0.2  # Consensus boost
                    )

        # Sort by sources first, then score
        sorted_expanded = sorted(
            all_expanded_results.values(),
            key=lambda x: (x["sources"], x["score"]),
            reverse=True
        )[:5]

        expanded_found = set()
        for data in sorted_expanded:
            for term in expected:
                if term in data["chunk"].raw_content.lower():
                    expanded_found.add(term)
        results["hybrid_expanded"].append(len(expanded_found) / len(expected))

        # Method 5: Full context (baseline - just search whole doc)
        full_found = set()
        for term in expected:
            if term in claude_md_content.lower():
                full_found.add(term)
        results["full_context"].append(len(full_found) / len(expected))

        print(f"  Raw:             {len(raw_found)}/{len(expected)} terms ({results['raw'][-1]:.0%})")
        print(f"  Extracted:       {len(ext_found)}/{len(expected)} terms ({results['extracted'][-1]:.0%})")
        print(f"  Hybrid:          {len(hybrid_found)}/{len(expected)} terms ({results['hybrid'][-1]:.0%})")
        print(f"  Hybrid+Expanded: {len(expanded_found)}/{len(expected)} terms ({results['hybrid_expanded'][-1]:.0%})")
        print(f"  Full context:    {len(full_found)}/{len(expected)} terms ({results['full_context'][-1]:.0%})")

    # ==========================================================================
    # Summary
    # ==========================================================================
    print()
    print("=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)
    print()

    print("Average Retrieval Success:")
    print("-" * 50)
    for method in ["raw", "extracted", "hybrid", "hybrid_expanded", "full_context"]:
        avg = sum(results[method]) / len(results[method])
        bar = "█" * int(avg * 20)
        print(f"  {method:<17} {avg:>5.0%} {bar}")

    print()
    print("Pipeline Timing:")
    print("-" * 40)
    print(f"  Discovery:   {discovery_time:>8.0f}ms")
    print(f"  Chunking:    {chunking_time:>8.0f}ms")
    print(f"  Storage:     {storage_time:>8.0f}ms (10 chunks)")
    print(f"  Total:       {discovery_time + chunking_time + storage_time:>8.0f}ms")

    print()
    print("=" * 80)
    print("CONCLUSIONS")
    print("=" * 80)
    print()

    hybrid_avg = sum(results["hybrid"]) / len(results["hybrid"])
    expanded_avg = sum(results["hybrid_expanded"]) / len(results["hybrid_expanded"])
    full_avg = sum(results["full_context"]) / len(results["full_context"])

    if expanded_avg >= full_avg * 0.9:
        print("✅ HYBRID + EXPANSION matches full context!")
        print("   Can replace large CLAUDE.md with chunked RAG/Neo4j storage.")
    elif expanded_avg >= full_avg * 0.8:
        print("🟡 HYBRID + EXPANSION is viable (80%+ of full context)")
        print("   Good enough for most use cases.")
    else:
        print("⚠️  RETRIEVAL needs improvement")
        print(f"   Only achieves {expanded_avg:.0%} vs {full_avg:.0%} for full context")

    if expanded_avg > hybrid_avg:
        improvement = (expanded_avg - hybrid_avg) / hybrid_avg * 100 if hybrid_avg else 0
        print()
        print(f"📈 Query expansion improved retrieval by {improvement:.0f}%")

    print()
    print("RECOMMENDED ARCHITECTURE:")
    print()
    print("1. INGESTION (one-time or on file change):")
    print("   CLAUDE.md → Agentic Discovery → Find related files")
    print("            → Semantic Chunking → 600-token chunks")
    print("            → Hybrid Storage → Raw + Extracted per chunk")
    print("            → Index in RAG (vectors) + Neo4j (entities)")
    print()
    print("2. QUERY TIME:")
    print("   Query → Expand to variations (parallel)")
    print("        → Search HYBRID (raw + extracted matching)")
    print("        → Retrieve top-k chunks (with raw content)")
    print("        → Synthesize answer")
    print()
    print("3. MINIMAL CLAUDE.md:")
    print("   - Project name and purpose (1 paragraph)")
    print("   - Pointer to MCP server for RAG/Neo4j access")
    print("   - Critical safety rules (if any)")
    print("   - Everything else lives in the indexed corpus")


if __name__ == "__main__":
    asyncio.run(main())
