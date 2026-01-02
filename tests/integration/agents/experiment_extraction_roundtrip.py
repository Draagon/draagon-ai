#!/usr/bin/env python3
"""Experiment: Does extracted terminology degrade LLM synthesis quality?

HYPOTHESIS:
When we extract entities/concepts from documents (Phase 1/0), we get normalized
terms like "SMS messaging system" → "dual-channel SMS". If we pass these
extracted terms directly to the LLM for answer synthesis, the LLM might:

1. Have WORSE understanding because it sees unnaturally compressed text
2. Have BETTER understanding because terms are normalized/consistent
3. Have SAME understanding because LLMs are robust to phrasing

We also test:
- Converting extracted terms BACK to natural language before LLM synthesis
- Whether this "round-trip" improves answer quality

PIPELINE VARIATIONS:

A. Current (Extracted → LLM):
   Query → Extract → [entities, concepts] → LLM synthesizes answer

B. Round-trip (Extracted → Expand → LLM):
   Query → Extract → [entities, concepts] → Expand to natural language → LLM synthesizes

C. Raw (No extraction):
   Query → Raw text → LLM synthesizes answer

Run with:
    GROQ_API_KEY=your_key python3.11 tests/integration/agents/experiment_extraction_roundtrip.py
"""

import asyncio
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))


@dataclass
class SynthesisResult:
    """Result from LLM synthesis."""
    answer: str
    relevance_score: float  # LLM self-assessed
    confidence: float
    latency_ms: float


@dataclass
class ExperimentResult:
    """Result from one experiment trial."""
    query: str
    mode: str  # "raw", "extracted", "roundtrip"

    # Input to LLM
    context_given: str

    # Output
    answer: str

    # Quality metrics (LLM-judged)
    coherence: float  # Is the answer well-formed?
    relevance: float  # Does it answer the question?
    completeness: float  # Is it thorough?

    latency_ms: float


def parse_section(content: str, prefix: str) -> list[str]:
    """Parse a section that may be comma-separated or bullet-point format.

    Handles both:
        PREFIX: item1, item2, item3
    And:
        PREFIX:
        - item1
        - item2
    """
    lines = content.split("\n")
    items = []
    in_section = False

    for i, line in enumerate(lines):
        line = line.strip()

        if line.upper().startswith(prefix.upper()):
            in_section = True
            # Check if items are on same line (comma-separated)
            after_prefix = line[len(prefix):].strip()
            if after_prefix.startswith(":"):
                after_prefix = after_prefix[1:].strip()
            if after_prefix:
                # Comma-separated on same line
                items.extend([x.strip() for x in after_prefix.split(",") if x.strip()])
            continue

        if in_section:
            # Check if we've hit another section
            if any(line.upper().startswith(p) for p in ["ENTITIES", "CONCEPTS", "NORMALIZED", "EXPANDED", "TEXT"]):
                break

            # Parse bullet or numbered items
            if line.startswith("-") or line.startswith("*"):
                item = line[1:].strip()
                if item:
                    items.append(item)
            elif line and line[0].isdigit() and "." in line[:3]:
                # Numbered list (e.g., "1. item")
                item = line.split(".", 1)[1].strip() if "." in line else line
                if item:
                    items.append(item)
            elif line and not line.startswith("#"):
                # Plain text continuation (not a header)
                items.append(line)

    return items


async def main():
    """Run the extraction round-trip experiment."""
    from groq import AsyncGroq

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("ERROR: GROQ_API_KEY not set")
        sys.exit(1)

    client = AsyncGroq(api_key=api_key)

    class GroqLLM:
        async def chat(self, messages, temperature=0.7, max_tokens=1000, model="llama-3.1-8b-instant"):
            resp = await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message

    llm = GroqLLM()

    print("=" * 80)
    print("EXPERIMENT: Extracted vs Natural Language for LLM Synthesis")
    print("=" * 80)
    print()
    print("Testing whether extracted/normalized terminology degrades LLM performance")
    print("compared to natural language context.")
    print()

    # Test documents with varying complexity
    test_cases = [
        {
            "query": "How does the SMS messaging system handle delivery failures?",
            "document": """# SMS Messaging Specification

The SMS messaging system uses a dual-channel approach with both primary and backup carriers.
Messages are queued in a persistent message queue and retry logic ensures delivery.

When a message fails to deliver:
1. The DeliveryTracker detects the failure via carrier webhook
2. The message is re-queued with exponential backoff (1s, 2s, 4s, 8s, max 5 retries)
3. If primary carrier fails, the system switches to backup carrier
4. After all retries exhausted, the message is marked as failed and logged

The system supports templates with variable substitution for personalized messages.
Carrier adapters abstract the differences between Twilio, Vonage, and other providers.
""",
            "expected_key_points": ["retry logic", "exponential backoff", "backup carrier", "5 retries"],
        },
        {
            "query": "What happens when a user contradicts their previous statement?",
            "document": """# Belief Reconciliation System

When users provide conflicting information, the belief reconciliation module handles it:

1. Accept Latest - If the new statement is clearly a correction ("Actually, I have 3 cats,
   not 2"), the system accepts the latest value and updates the belief.

2. Weight by Credibility - The system considers the user's track record. If they frequently
   correct themselves, newer statements get higher weight.

3. Ask for Clarification - If the conflict is ambiguous, the system queues a clarification
   question for an appropriate moment (not immediately, to avoid seeming pedantic).

4. Flag Conflict - For critical conflicts (e.g., birthdate inconsistencies), the system
   marks the belief as needing resolution and doesn't act on either value.

Observations are immutable - the system never modifies what the user actually said.
Only agent beliefs (reconciled understanding) can be updated.
""",
            "expected_key_points": ["accept latest", "credibility weight", "clarification", "immutable observations"],
        },
    ]

    # Semantic extractor with robust parsing
    async def extract(text: str) -> dict:
        """Extract entities and concepts from text."""
        prompt = f"""Extract key entities and concepts from this text.

Text: {text[:2000]}

Output in this EXACT format (comma-separated on same line):
ENTITIES: entity1, entity2, entity3
CONCEPTS: concept1, concept2, concept3

Remember:
- ENTITIES are specific things (systems, components, features, tools)
- CONCEPTS are abstract ideas (patterns, approaches, mechanisms, strategies)"""

        response = await llm.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=300,
        )
        content = response.content

        result = {
            "entities": parse_section(content, "ENTITIES"),
            "concepts": parse_section(content, "CONCEPTS"),
        }

        return result

    async def expand_to_natural(entities: list, concepts: list, query: str) -> str:
        """Expand extracted terms back to natural language context."""
        if not entities and not concepts:
            return "No relevant information available."

        prompt = f"""Given these extracted terms related to a query, expand them into
natural language sentences that explain the concepts clearly.

Query: {query}

Entities: {', '.join(entities) if entities else 'None'}
Concepts: {', '.join(concepts) if concepts else 'None'}

Write 2-3 natural language sentences that connect these terms to answer the query.
Focus on clarity and natural phrasing. Do not say you don't have enough information."""

        response = await llm.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=300,
        )
        return response.content

    async def synthesize_answer(query: str, context: str, mode: str) -> str:
        """Have the LLM synthesize an answer from context."""
        prompt = f"""Based on the following context, answer the question.

Context:
{context}

Question: {query}

Provide a clear, concise answer based only on the context provided."""

        response = await llm.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=400,
        )
        return response.content

    async def evaluate_answer(query: str, answer: str, expected_points: list) -> dict:
        """Have an LLM evaluate the quality of an answer."""
        prompt = f"""Evaluate this answer on a scale of 0.0 to 1.0 for each criterion.

Question: {query}

Answer: {answer}

Expected key points that should be mentioned: {', '.join(expected_points)}

Rate the answer on:
1. COHERENCE: Is the answer well-formed and grammatically correct? (0.0-1.0)
2. RELEVANCE: Does it actually answer the question? (0.0-1.0)
3. COMPLETENESS: Does it cover the expected key points? (0.0-1.0)
4. POINT_COVERAGE: What fraction of expected points were mentioned? (0.0-1.0)

Output EXACTLY (numbers only after colon):
COHERENCE: 0.X
RELEVANCE: 0.X
COMPLETENESS: 0.X
POINT_COVERAGE: 0.X"""

        response = await llm.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=200,
        )
        content = response.content

        scores = {"coherence": 0.5, "relevance": 0.5, "completeness": 0.5, "point_coverage": 0.5}
        for line in content.split("\n"):
            line = line.strip().upper()
            for key in scores:
                if line.startswith(key.upper() + ":"):
                    try:
                        val = line.split(":")[1].strip()
                        # Handle "0.8" or "0.8 - explanation"
                        val = val.split()[0] if " " in val else val
                        scores[key] = float(val)
                    except:
                        pass

        return scores

    # Run experiments
    results = []

    for case in test_cases:
        query = case["query"]
        document = case["document"]
        expected = case["expected_key_points"]

        print("-" * 80)
        print(f"Query: {query}")
        print("-" * 80)

        # First extract (shared across modes that need it)
        extraction = await extract(document)
        print(f"\n[Extraction Result]")
        print(f"  Entities: {extraction['entities']}")
        print(f"  Concepts: {extraction['concepts']}")

        # Mode A: Raw (full document)
        print("\n[Mode A: RAW - Full document to LLM]")
        start = time.time()
        answer_raw = await synthesize_answer(query, document, "raw")
        latency_raw = (time.time() - start) * 1000
        scores_raw = await evaluate_answer(query, answer_raw, expected)
        print(f"  Answer: {answer_raw[:200]}...")
        print(f"  Scores: {scores_raw}")
        print(f"  Latency: {latency_raw:.0f}ms")

        # Mode B: Extracted (entities + concepts only)
        print("\n[Mode B: EXTRACTED - Only entities/concepts to LLM]")
        extracted_context = f"""Entities: {', '.join(extraction['entities']) if extraction['entities'] else 'None listed'}
Concepts: {', '.join(extraction['concepts']) if extraction['concepts'] else 'None listed'}"""
        print(f"  Context given: {extracted_context[:200]}...")

        start = time.time()
        answer_extracted = await synthesize_answer(query, extracted_context, "extracted")
        latency_extracted = (time.time() - start) * 1000
        scores_extracted = await evaluate_answer(query, answer_extracted, expected)
        print(f"  Answer: {answer_extracted[:200]}...")
        print(f"  Scores: {scores_extracted}")
        print(f"  Latency: {latency_extracted:.0f}ms")

        # Mode C: Round-trip (extract → expand back → LLM)
        print("\n[Mode C: ROUND-TRIP - Extract, expand back to natural language]")
        expanded_context = await expand_to_natural(
            extraction["entities"],
            extraction["concepts"],
            query
        )
        print(f"  Expanded: {expanded_context[:200]}...")

        start = time.time()
        answer_roundtrip = await synthesize_answer(query, expanded_context, "roundtrip")
        latency_roundtrip = (time.time() - start) * 1000
        scores_roundtrip = await evaluate_answer(query, answer_roundtrip, expected)
        print(f"  Answer: {answer_roundtrip[:200]}...")
        print(f"  Scores: {scores_roundtrip}")
        print(f"  Latency: {latency_roundtrip:.0f}ms")

        # Mode D: Hybrid (extracted + raw summary)
        print("\n[Mode D: HYBRID - Extracted terms + raw document summary]")
        hybrid_context = f"""Key Terms:
{extracted_context}

Full Context:
{document[:1000]}"""

        start = time.time()
        answer_hybrid = await synthesize_answer(query, hybrid_context, "hybrid")
        latency_hybrid = (time.time() - start) * 1000
        scores_hybrid = await evaluate_answer(query, answer_hybrid, expected)
        print(f"  Answer: {answer_hybrid[:200]}...")
        print(f"  Scores: {scores_hybrid}")
        print(f"  Latency: {latency_hybrid:.0f}ms")

        results.append({
            "query": query,
            "raw": {"scores": scores_raw, "latency": latency_raw},
            "extracted": {"scores": scores_extracted, "latency": latency_extracted},
            "roundtrip": {"scores": scores_roundtrip, "latency": latency_roundtrip},
            "hybrid": {"scores": scores_hybrid, "latency": latency_hybrid},
        })

    # Summary
    print()
    print("=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)
    print()

    # Aggregate scores
    modes = ["raw", "extracted", "roundtrip", "hybrid"]
    metrics = ["coherence", "relevance", "completeness", "point_coverage"]

    print("Average Scores by Mode:")
    print("-" * 60)
    print(f"{'Mode':<15} {'Coherence':>10} {'Relevance':>10} {'Complete':>10} {'Coverage':>10}")
    print("-" * 60)

    mode_totals = {}
    for mode in modes:
        avg_scores = {}
        for metric in metrics:
            total = sum(r[mode]["scores"][metric] for r in results)
            avg_scores[metric] = total / len(results)

        mode_totals[mode] = sum(avg_scores.values()) / len(avg_scores)
        print(f"{mode:<15} {avg_scores['coherence']:>10.2f} {avg_scores['relevance']:>10.2f} "
              f"{avg_scores['completeness']:>10.2f} {avg_scores['point_coverage']:>10.2f}")

    print()
    print("Average Latency by Mode:")
    print("-" * 40)
    for mode in modes:
        avg_latency = sum(r[mode]["latency"] for r in results) / len(results)
        print(f"  {mode:<15} {avg_latency:>8.0f}ms")

    print()
    print("=" * 80)
    print("CONCLUSIONS")
    print("=" * 80)
    print()
    print("Key findings from this experiment:")
    print()

    # Calculate which mode won
    mode_wins = {mode: 0 for mode in modes}
    for metric in metrics:
        best_mode = max(modes, key=lambda m: sum(r[m]["scores"][metric] for r in results))
        mode_wins[best_mode] += 1
        avg = sum(r[best_mode]["scores"][metric] for r in results) / len(results)
        print(f"  Best {metric}: {best_mode} ({avg:.2f})")

    overall_winner = max(mode_wins.items(), key=lambda x: x[1])
    print()
    print(f"Overall winner: {overall_winner[0]} (won {overall_winner[1]}/{len(metrics)} metrics)")

    print()
    print("INTERPRETATION:")
    print()

    # Compare extracted vs raw
    raw_avg = mode_totals["raw"]
    extracted_avg = mode_totals["extracted"]
    roundtrip_avg = mode_totals["roundtrip"]
    hybrid_avg = mode_totals["hybrid"]

    if extracted_avg < raw_avg * 0.7:
        print("⚠️  EXTRACTED SIGNIFICANTLY WORSE than RAW")
        print("   Passing extracted terms alone loses too much context.")
    elif extracted_avg < raw_avg * 0.9:
        print("📉 EXTRACTED somewhat worse than RAW")
        print("   Extraction causes moderate information loss.")
    else:
        print("✅ EXTRACTED comparable to RAW")
        print("   LLM handles extracted terminology well.")

    if roundtrip_avg > extracted_avg:
        print()
        print("✅ ROUND-TRIP helps! Expanding back to natural language improves quality.")
    else:
        print()
        print("❌ ROUND-TRIP doesn't help. The expansion step doesn't add value.")

    if hybrid_avg >= max(raw_avg, extracted_avg, roundtrip_avg):
        print()
        print("🏆 HYBRID is BEST! Combining extracted terms with raw context is optimal.")


if __name__ == "__main__":
    asyncio.run(main())
