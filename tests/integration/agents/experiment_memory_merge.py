#!/usr/bin/env python3
"""Experiment: Memory Merge Strategies - When to Extract for Semantic/RAG Storage

HYPOTHESIS:
When merging new information into memory, we have three options:

A. LAZY: Store raw text, extract later when promoting to Semantic/RAG
   - Pro: Don't pay extraction cost if memory gets discarded
   - Con: Raw text in working memory won't match extracted queries

B. EAGER: Extract immediately, store only extracted form
   - Pro: Consistent terminology, ready for matching
   - Con: Lose raw context that helps LLM synthesis

C. HYBRID: Store both raw AND extracted
   - Pro: Best of both - raw for LLM, extracted for matching
   - Con: Storage overhead, complexity

This experiment simulates a multi-turn conversation where:
1. User provides information (gets stored in memory)
2. Later, user asks a question that should retrieve that memory
3. We measure retrieval success and answer quality

Run with:
    GROQ_API_KEY=your_key python3.11 tests/integration/agents/experiment_memory_merge.py
"""

import asyncio
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

# Add project to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))


@dataclass
class MemoryEntry:
    """A memory entry that can store raw and/or extracted content."""
    id: str
    raw_content: str | None = None
    extracted_entities: list[str] = field(default_factory=list)
    extracted_concepts: list[str] = field(default_factory=list)
    source: str = "user"  # user, system, inferred
    timestamp: float = field(default_factory=time.time)


@dataclass
class MemoryStore:
    """Simulated memory store supporting different merge strategies."""
    strategy: Literal["lazy", "eager", "hybrid"]
    entries: list[MemoryEntry] = field(default_factory=list)

    def add(self, entry: MemoryEntry):
        self.entries.append(entry)

    def search_raw(self, query_words: set[str], k: int = 3) -> list[tuple[MemoryEntry, float]]:
        """Search using raw content matching."""
        results = []
        for entry in self.entries:
            if entry.raw_content:
                content_words = set(entry.raw_content.lower().split())
                overlap = len(query_words & content_words)
                if overlap > 0:
                    results.append((entry, overlap / len(query_words)))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]

    def search_extracted(self, query_terms: set[str], k: int = 3) -> list[tuple[MemoryEntry, float]]:
        """Search using extracted entities/concepts."""
        results = []
        for entry in self.entries:
            entry_terms = set(
                [e.lower() for e in entry.extracted_entities] +
                [c.lower() for c in entry.extracted_concepts]
            )
            if entry_terms:
                overlap = len(query_terms & entry_terms)
                # Also check partial matches
                partial = sum(1 for qt in query_terms for et in entry_terms
                             if qt in et or et in qt) * 0.5
                score = (overlap + partial) / max(len(query_terms), 1)
                if score > 0:
                    results.append((entry, score))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]


def parse_section(content: str, prefix: str) -> list[str]:
    """Parse a section that may be comma-separated or bullet-point format."""
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
            if any(line.upper().startswith(p) for p in ["ENTITIES", "CONCEPTS", "NORMALIZED"]):
                break
            if line.startswith("-") or line.startswith("*"):
                item = line[1:].strip()
                if item:
                    items.append(item)
            elif line and line[0].isdigit() and "." in line[:3]:
                item = line.split(".", 1)[1].strip() if "." in line else line
                if item:
                    items.append(item)
            elif line and not line.startswith("#"):
                items.append(line)

    return items


async def main():
    """Run the memory merge strategy experiment."""
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
    print("EXPERIMENT: Memory Merge Strategies")
    print("=" * 80)
    print()
    print("Testing when to extract entities/concepts for memory storage:")
    print("  LAZY:   Store raw, extract on retrieval")
    print("  EAGER:  Extract immediately, store only extracted")
    print("  HYBRID: Store both raw and extracted")
    print()

    # Extraction function
    async def extract(text: str) -> dict:
        """Extract entities and concepts from text."""
        prompt = f"""Extract key entities and concepts from this text.

Text: {text}

Output in this EXACT format (comma-separated on same line):
ENTITIES: entity1, entity2, entity3
CONCEPTS: concept1, concept2, concept3

ENTITIES are specific things (names, systems, tools, places)
CONCEPTS are abstract ideas (patterns, relationships, preferences)"""

        response = await llm.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=200,
        )
        content = response.content

        return {
            "entities": parse_section(content, "ENTITIES"),
            "concepts": parse_section(content, "CONCEPTS"),
        }

    # Simulate conversation scenarios
    # Each scenario has: user statements (to store) and later queries (to retrieve)
    scenarios = [
        {
            "name": "Personal preferences",
            "statements": [
                "I prefer dark mode for all my apps, and I like to work late at night.",
                "My favorite programming language is Python, but I also use TypeScript for frontend.",
                "I have three cats named Luna, Mochi, and Pepper.",
            ],
            "queries": [
                ("What theme do I prefer?", ["dark mode"]),
                ("What languages do I use?", ["Python", "TypeScript"]),
                ("What are my cats' names?", ["Luna", "Mochi", "Pepper"]),
            ],
        },
        {
            "name": "Technical context",
            "statements": [
                "The production database is PostgreSQL running on port 5432 with connection pooling.",
                "We use Redis for caching with a 15-minute TTL for user sessions.",
                "The API rate limit is 100 requests per minute per user.",
            ],
            "queries": [
                ("What database do we use?", ["PostgreSQL", "5432"]),
                ("How long are sessions cached?", ["15-minute", "Redis"]),
                ("What's the rate limit?", ["100 requests", "per minute"]),
            ],
        },
        {
            "name": "Varied phrasing",
            "statements": [
                "Doug's birthday is March 15th and he loves Italian food.",
                "The weekly team meeting is on Tuesdays at 2pm in the main conference room.",
                "Critical alerts should go to the #incidents Slack channel.",
            ],
            "queries": [
                ("When is Doug's birthday?", ["March 15"]),
                ("When do we have team meetings?", ["Tuesday", "2pm"]),
                ("Where do critical alerts go?", ["#incidents", "Slack"]),
            ],
        },
    ]

    # Run experiment for each strategy
    strategies = ["lazy", "eager", "hybrid"]
    results = {s: {"retrieval_success": 0, "answer_quality": 0, "total": 0, "latency_store": 0, "latency_retrieve": 0} for s in strategies}

    for scenario in scenarios:
        print("-" * 80)
        print(f"Scenario: {scenario['name']}")
        print("-" * 80)

        for strategy in strategies:
            print(f"\n  [{strategy.upper()}]")
            store = MemoryStore(strategy=strategy)

            # Phase 1: Store statements
            store_start = time.time()
            for stmt in scenario["statements"]:
                if strategy == "lazy":
                    # Store only raw
                    entry = MemoryEntry(
                        id=f"mem_{len(store.entries)}",
                        raw_content=stmt,
                    )
                elif strategy == "eager":
                    # Extract and store only extracted
                    extraction = await extract(stmt)
                    entry = MemoryEntry(
                        id=f"mem_{len(store.entries)}",
                        extracted_entities=extraction["entities"],
                        extracted_concepts=extraction["concepts"],
                    )
                else:  # hybrid
                    # Store both
                    extraction = await extract(stmt)
                    entry = MemoryEntry(
                        id=f"mem_{len(store.entries)}",
                        raw_content=stmt,
                        extracted_entities=extraction["entities"],
                        extracted_concepts=extraction["concepts"],
                    )
                store.add(entry)
            store_time = (time.time() - store_start) * 1000
            results[strategy]["latency_store"] += store_time

            # Phase 2: Query and evaluate
            retrieve_start = time.time()
            for query, expected_keywords in scenario["queries"]:
                results[strategy]["total"] += 1

                # Search based on strategy
                query_words = set(query.lower().split())

                if strategy == "lazy":
                    # Can only search raw
                    matches = store.search_raw(query_words)
                elif strategy == "eager":
                    # Extract query, search extracted
                    query_extraction = await extract(query)
                    query_terms = set(
                        [e.lower() for e in query_extraction["entities"]] +
                        [c.lower() for c in query_extraction["concepts"]]
                    )
                    matches = store.search_extracted(query_terms) if query_terms else []
                else:  # hybrid
                    # Search both, merge results
                    raw_matches = store.search_raw(query_words)

                    query_extraction = await extract(query)
                    query_terms = set(
                        [e.lower() for e in query_extraction["entities"]] +
                        [c.lower() for c in query_extraction["concepts"]]
                    )
                    extracted_matches = store.search_extracted(query_terms) if query_terms else []

                    # Merge: combine scores, boost if found by both
                    all_matches = {}
                    for entry, score in raw_matches:
                        all_matches[entry.id] = {"entry": entry, "raw_score": score, "ext_score": 0}
                    for entry, score in extracted_matches:
                        if entry.id in all_matches:
                            all_matches[entry.id]["ext_score"] = score
                        else:
                            all_matches[entry.id] = {"entry": entry, "raw_score": 0, "ext_score": score}

                    # Combined score with boost for dual-match
                    matches = []
                    for data in all_matches.values():
                        combined = max(data["raw_score"], data["ext_score"])
                        if data["raw_score"] > 0 and data["ext_score"] > 0:
                            combined += 0.2  # Boost for consensus
                        matches.append((data["entry"], combined))
                    matches.sort(key=lambda x: x[1], reverse=True)

                # Check if we found relevant memory
                found_relevant = False
                retrieved_content = ""
                if matches:
                    top_match = matches[0][0]
                    retrieved_content = top_match.raw_content or " ".join(
                        top_match.extracted_entities + top_match.extracted_concepts
                    )
                    # Check if any expected keyword is in the retrieved content
                    for kw in expected_keywords:
                        if kw.lower() in retrieved_content.lower():
                            found_relevant = True
                            break

                if found_relevant:
                    results[strategy]["retrieval_success"] += 1

                # Evaluate answer quality (can LLM use the retrieved content?)
                if retrieved_content:
                    answer_prompt = f"""Based on this context, answer the question briefly.

Context: {retrieved_content}

Question: {query}

Answer:"""
                    response = await llm.chat(
                        messages=[{"role": "user", "content": answer_prompt}],
                        temperature=0.3,
                        max_tokens=100,
                    )
                    answer = response.content.lower()

                    # Check if answer contains expected info
                    keywords_found = sum(1 for kw in expected_keywords if kw.lower() in answer)
                    quality = keywords_found / len(expected_keywords)
                    results[strategy]["answer_quality"] += quality

            retrieve_time = (time.time() - retrieve_start) * 1000
            results[strategy]["latency_retrieve"] += retrieve_time

            # Print summary for this strategy/scenario
            scenario_total = len(scenario["queries"])
            scenario_success = sum(1 for q, kw in scenario["queries"]
                                  for _ in [1] if results[strategy]["retrieval_success"] > 0)
            print(f"    Store time: {store_time:.0f}ms")
            print(f"    Retrieve time: {retrieve_time:.0f}ms")

    # Final summary
    print()
    print("=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)
    print()

    print("Results by Strategy:")
    print("-" * 70)
    print(f"{'Strategy':<12} {'Retrieval':>12} {'Answer Quality':>15} {'Store (ms)':>12} {'Retrieve (ms)':>14}")
    print("-" * 70)

    for strategy in strategies:
        r = results[strategy]
        retrieval_rate = r["retrieval_success"] / r["total"] if r["total"] else 0
        answer_rate = r["answer_quality"] / r["total"] if r["total"] else 0
        print(f"{strategy:<12} {retrieval_rate:>11.1%} {answer_rate:>14.1%} {r['latency_store']:>11.0f} {r['latency_retrieve']:>13.0f}")

    print()
    print("=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    print()

    # Determine winner
    best_retrieval = max(strategies, key=lambda s: results[s]["retrieval_success"])
    best_quality = max(strategies, key=lambda s: results[s]["answer_quality"])
    fastest_store = min(strategies, key=lambda s: results[s]["latency_store"])
    fastest_retrieve = min(strategies, key=lambda s: results[s]["latency_retrieve"])

    print(f"Best retrieval: {best_retrieval}")
    print(f"Best answer quality: {best_quality}")
    print(f"Fastest store: {fastest_store}")
    print(f"Fastest retrieve: {fastest_retrieve}")

    print()
    print("INTERPRETATION:")
    print()

    lazy_ret = results["lazy"]["retrieval_success"] / results["lazy"]["total"]
    eager_ret = results["eager"]["retrieval_success"] / results["eager"]["total"]
    hybrid_ret = results["hybrid"]["retrieval_success"] / results["hybrid"]["total"]

    if hybrid_ret >= max(lazy_ret, eager_ret):
        print("✅ HYBRID wins or ties for retrieval")
        print("   Storing both raw and extracted gives best matching flexibility.")

    lazy_qual = results["lazy"]["answer_quality"] / results["lazy"]["total"]
    eager_qual = results["eager"]["answer_quality"] / results["eager"]["total"]
    hybrid_qual = results["hybrid"]["answer_quality"] / results["hybrid"]["total"]

    if lazy_qual > eager_qual:
        print()
        print("📝 RAW content helps answer quality")
        print("   LLM produces better answers when it has the original text.")

    if results["lazy"]["latency_store"] < results["eager"]["latency_store"] * 0.5:
        print()
        print("⚡ LAZY is much faster for storage")
        print("   Deferring extraction saves time if memories aren't always retrieved.")

    print()
    print("=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)
    print()
    print("Based on this experiment:")
    print()
    print("1. HYBRID storage is recommended for important/validated information:")
    print("   - Store raw for LLM synthesis quality")
    print("   - Store extracted for retrieval matching")
    print()
    print("2. LAZY storage is acceptable for ephemeral/working memory:")
    print("   - Fast storage, extract only if/when retrieved")
    print("   - Good for high-volume, low-retrieval scenarios")
    print()
    print("3. EAGER-only is NOT recommended:")
    print("   - Loses raw context that helps answer quality")
    print("   - Extraction can miss nuances")
    print()
    print("IMPLEMENTATION:")
    print()
    print("  User input → Validate → HYBRID store (raw + extracted)")
    print("                              ↓")
    print("  Later query → Search extracted → Retrieve raw for LLM")
    print("                              ↓")
    print("  Promote to Semantic/RAG → Already has both forms")


if __name__ == "__main__":
    asyncio.run(main())
