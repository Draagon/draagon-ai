#!/usr/bin/env python3
"""Test script to verify party-lore document ingestion and graph merging.

This script:
1. Processes a few markdown files from party-lore through Phase 0/1 decomposition
2. Stores them in Neo4j semantic graph
3. Verifies graph merging is working correctly
4. Calculates storage ratio (file size vs Neo4j space)

Run with:
    cd /home/doug/Development/draagon-ai
    python tests/integration/agents/test_party_lore_ingestion.py
"""

import asyncio
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime

# Add project to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

# Load .env
try:
    from dotenv import load_dotenv
    load_dotenv(project_root / ".env")
except ImportError:
    pass

from neo4j import AsyncGraphDatabase


@dataclass
class IngestionStats:
    """Statistics from document ingestion."""
    files_processed: int = 0
    total_file_size_bytes: int = 0
    total_nodes: int = 0
    total_edges: int = 0
    nodes_by_type: dict = field(default_factory=dict)
    edges_by_type: dict = field(default_factory=dict)
    entity_names: list = field(default_factory=list)
    processing_time_seconds: float = 0.0


async def get_neo4j_stats(driver, database: str = "neo4j") -> dict:
    """Get statistics about the Neo4j database."""
    async with driver.session(database=database) as session:
        # Node counts by label
        result = await session.run("""
            MATCH (n)
            RETURN labels(n) AS labels, count(n) AS count
        """)
        node_counts = {}
        total_nodes = 0
        async for record in result:
            labels_str = ":".join(sorted(record["labels"]))
            node_counts[labels_str] = record["count"]
            total_nodes += record["count"]

        # Edge counts by type
        result = await session.run("""
            MATCH ()-[r]->()
            RETURN type(r) AS type, count(r) AS count
        """)
        edge_counts = {}
        total_edges = 0
        async for record in result:
            edge_counts[record["type"]] = record["count"]
            total_edges += record["count"]

        # Database size estimation (node properties size)
        result = await session.run("""
            MATCH (n)
            WITH n, size(keys(properties(n))) AS prop_count
            RETURN sum(prop_count) AS total_properties, count(n) AS total_nodes
        """)
        record = await result.single()
        total_properties = record["total_properties"] if record else 0

        # Get all canonical names for entity deduplication check
        result = await session.run("""
            MATCH (n:Entity)
            WHERE n.canonical_name IS NOT NULL
            RETURN n.canonical_name AS name, n.node_type AS type, count(*) AS count
            ORDER BY count DESC
            LIMIT 50
        """)
        entity_names = []
        duplicates = []
        async for record in result:
            entity_names.append({
                "name": record["name"],
                "type": record["type"],
                "count": record["count"]
            })
            if record["count"] > 1:
                duplicates.append(record["name"])

        return {
            "total_nodes": total_nodes,
            "total_edges": total_edges,
            "total_properties": total_properties,
            "nodes_by_label": node_counts,
            "edges_by_type": edge_counts,
            "entity_names": entity_names,
            "duplicates": duplicates,
        }


async def export_graph_sample(driver, database: str = "neo4j", limit: int = 100) -> dict:
    """Export a sample of the graph for inspection."""
    async with driver.session(database=database) as session:
        # Get sample nodes
        result = await session.run("""
            MATCH (n:Entity)
            RETURN n.node_id AS id, n.canonical_name AS name, n.node_type AS type,
                   n.entity_type AS entity_type, n.synset_id AS synset_id,
                   n.instance_id AS instance_id
            LIMIT $limit
        """, limit=limit)
        nodes = []
        async for record in result:
            nodes.append(dict(record))

        # Get sample edges
        result = await session.run("""
            MATCH (s:Entity)-[r]->(t:Entity)
            RETURN s.canonical_name AS source, type(r) AS rel_type,
                   t.canonical_name AS target, r.confidence AS confidence
            LIMIT $limit
        """, limit=limit)
        edges = []
        async for record in result:
            edges.append(dict(record))

        return {"nodes": nodes, "edges": edges}


async def clear_database(driver, database: str = "neo4j"):
    """Clear all data from the database."""
    async with driver.session(database=database) as session:
        await session.run("MATCH (n) DETACH DELETE n")


async def process_files_with_decomposition(
    files: list[Path],
    driver,
    llm,
    embedding_provider,
    database: str = "neo4j",
    instance_id: str = "party-lore-test"
) -> IngestionStats:
    """Process markdown files through Phase 0/1 and store in Neo4j."""
    from draagon_ai.cognition.decomposition.extractors.integrated_pipeline import IntegratedPipeline
    from draagon_ai.cognition.decomposition.graph.builder import GraphBuilder
    from draagon_ai.cognition.decomposition.graph.neo4j_store import Neo4jGraphStore, Neo4jConfig
    from draagon_ai.cognition.decomposition.graph.semantic_graph import SemanticGraph

    stats = IngestionStats()
    start_time = datetime.now()

    # Initialize components
    pipeline = IntegratedPipeline(llm=llm)
    builder = GraphBuilder()

    # Create graph store
    config = Neo4jConfig(
        uri=os.getenv("NEO4J_TEST_URI", "bolt://localhost:7687"),
        username=os.getenv("NEO4J_TEST_USER", "neo4j"),
        password=os.getenv("NEO4J_TEST_PASSWORD", "draagon-ai-2025"),
        database=database,
    )
    store = Neo4jGraphStore(driver, config)
    await store.initialize()

    # Process each file
    cumulative_graph = SemanticGraph()

    for file_path in files:
        print(f"\n  Processing: {file_path.name}")

        # Read file
        content = file_path.read_text(encoding="utf-8")
        stats.total_file_size_bytes += len(content.encode("utf-8"))

        # Split into chunks (sentences/paragraphs) for processing
        # Process just first 3 sentences for quick testing
        sentences = extract_sentences(content, max_sentences=3)

        for i, sentence in enumerate(sentences):
            if not sentence.strip():
                continue

            print(f"    [{i+1}/{len(sentences)}] {sentence[:60]}...")

            try:
                # Run Phase 0/1 decomposition
                result = await pipeline.process(sentence)

                # Debug: Check what Phase 0 returned
                num_entities = len(result.phase0.entity_identifiers) if result.phase0.entity_identifiers else 0
                num_wsd = len(result.phase0.disambiguation_results) if result.phase0.disambiguation_results else 0

                # Build graph from decomposition
                build_result = builder.build_from_integrated(
                    result,
                    existing_graph=cumulative_graph
                )

                # Debug: Check graph state after build
                print(f"      → Phase0: {num_entities} entities, {num_wsd} WSD | "
                      f"Built: {build_result.stats.get('entities_created', 0)} nodes, "
                      f"{build_result.stats.get('edges_created', 0)} edges | "
                      f"Graph total: {len(cumulative_graph.nodes)} nodes")

            except Exception as e:
                print(f"      → Error: {e}")
                continue

        stats.files_processed += 1

    # Save to Neo4j
    print(f"\n  Saving graph to Neo4j (instance: {instance_id})...")
    save_result = await store.save(cumulative_graph, instance_id, clear_existing=True)
    print(f"    → Saved {save_result['nodes']} nodes, {save_result['edges']} edges")

    stats.total_nodes = save_result["nodes"]
    stats.total_edges = save_result["edges"]
    stats.processing_time_seconds = (datetime.now() - start_time).total_seconds()

    return stats


def extract_sentences(text: str, max_sentences: int = 50) -> list[str]:
    """Extract sentences from markdown text."""
    import re

    # Remove code blocks
    text = re.sub(r'```[\s\S]*?```', '', text)
    text = re.sub(r'`[^`]+`', '', text)

    # Remove markdown headers but keep content
    text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)

    # Remove markdown links, keep text
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)

    # Remove bullet points
    text = re.sub(r'^[\*\-]\s*', '', text, flags=re.MULTILINE)

    # Split into sentences (simple approach)
    sentences = re.split(r'(?<=[.!?])\s+', text)

    # Filter and clean
    clean_sentences = []
    for s in sentences:
        s = s.strip()
        # Skip very short or very long sentences
        if len(s) > 20 and len(s) < 500:
            # Skip sentences that are mostly non-text
            alpha_ratio = sum(c.isalpha() for c in s) / len(s) if s else 0
            if alpha_ratio > 0.5:
                clean_sentences.append(s)

    return clean_sentences[:max_sentences]


async def main():
    """Main entry point."""
    print("\n" + "=" * 80)
    print("PARTY-LORE DOCUMENT INGESTION TEST")
    print("Testing Phase 0/1 decomposition + Neo4j storage + Graph merging")
    print("=" * 80)

    # Check API key
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        print("ERROR: GROQ_API_KEY not set")
        return

    # Party-lore path
    party_lore_path = Path("/home/doug/Development/party-lore")
    if not party_lore_path.exists():
        print(f"ERROR: party-lore not found at {party_lore_path}")
        return

    # Select test files
    test_files = [
        party_lore_path / "CLAUDE.md",
        party_lore_path / "README.md",
        party_lore_path / ".specify" / "requirements" / "fr-001-dual-channel-sms.md",
        party_lore_path / ".specify" / "requirements" / "fr-002-intelligent-scene-resolution.md",
    ]

    # Filter to existing files
    test_files = [f for f in test_files if f.exists()]

    if not test_files:
        print("ERROR: No test files found")
        return

    print(f"\n[1] Selected {len(test_files)} files for testing:")
    for f in test_files:
        size = f.stat().st_size
        print(f"    - {f.name} ({size:,} bytes)")

    # Initialize providers
    print("\n[2] Initializing providers...")

    from draagon_ai.llm.groq import GroqLLM
    llm = GroqLLM(api_key=groq_key)
    print("    ✓ LLM (Groq)")

    # Check Ollama for embeddings
    try:
        from draagon_ai.memory.embedding import OllamaEmbeddingProvider
        ollama_url = os.getenv("OLLAMA_BASE_URL", "http://192.168.168.200:11434")
        embedding_provider = OllamaEmbeddingProvider(
            base_url=ollama_url,
            model="nomic-embed-text",
            dimension=768,
        )
        await embedding_provider.embed("test")
        print(f"    ✓ Embeddings (Ollama @ {ollama_url})")
    except Exception as e:
        print(f"    ✗ Embeddings unavailable: {e}")
        embedding_provider = None

    # Connect to Neo4j
    print("\n[3] Connecting to Neo4j...")
    neo4j_uri = os.getenv("NEO4J_TEST_URI", "bolt://localhost:7687")
    neo4j_user = os.getenv("NEO4J_TEST_USER", "neo4j")
    neo4j_pass = os.getenv("NEO4J_TEST_PASSWORD", "draagon-ai-2025")

    try:
        driver = AsyncGraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_pass))
        # Test connection
        async with driver.session(database="neo4j") as session:
            result = await session.run("RETURN 1 as n")
            await result.single()
        print(f"    ✓ Connected to {neo4j_uri}")
    except Exception as e:
        print(f"    ✗ Neo4j connection failed: {e}")
        print("\n    To start Neo4j:")
        print("    docker run -d --name neo4j -p 7474:7474 -p 7687:7687 \\")
        print("        -e NEO4J_AUTH=neo4j/draagon-ai-2025 neo4j:latest")
        return

    try:
        # Clear existing data
        print("\n[4] Clearing existing test data...")
        await clear_database(driver)
        print("    ✓ Database cleared")

        # Get initial stats
        initial_stats = await get_neo4j_stats(driver)
        print(f"    Initial: {initial_stats['total_nodes']} nodes, {initial_stats['total_edges']} edges")

        # Process files
        print("\n[5] Processing files through Phase 0/1 pipeline...")
        stats = await process_files_with_decomposition(
            test_files, driver, llm, embedding_provider,
            instance_id="party-lore-test"
        )

        # Get final stats
        print("\n[6] Analyzing results...")
        final_stats = await get_neo4j_stats(driver)

        # Export sample for inspection
        sample = await export_graph_sample(driver, limit=30)

        # Print results
        print("\n" + "=" * 80)
        print("RESULTS")
        print("=" * 80)

        print(f"\n  Files processed: {stats.files_processed}")
        print(f"  Total file size: {stats.total_file_size_bytes:,} bytes ({stats.total_file_size_bytes / 1024:.1f} KB)")
        print(f"  Processing time: {stats.processing_time_seconds:.1f} seconds")

        print(f"\n  Graph size:")
        print(f"    Nodes: {final_stats['total_nodes']}")
        print(f"    Edges: {final_stats['total_edges']}")
        print(f"    Properties: {final_stats['total_properties']}")

        print(f"\n  Nodes by label:")
        for label, count in sorted(final_stats["nodes_by_label"].items()):
            print(f"    {label}: {count}")

        print(f"\n  Edges by type:")
        for etype, count in sorted(final_stats["edges_by_type"].items(), key=lambda x: -x[1])[:15]:
            print(f"    {etype}: {count}")

        # Check for duplicates (merging verification)
        print(f"\n  Entity merging check:")
        if final_stats["duplicates"]:
            print(f"    ⚠ Found {len(final_stats['duplicates'])} entities with duplicate canonical names:")
            for dup in final_stats["duplicates"][:5]:
                print(f"      - {dup}")
        else:
            print(f"    ✓ No duplicate entities found - merging appears to be working")

        # Top entities
        print(f"\n  Top entities by occurrence:")
        for entity in final_stats["entity_names"][:10]:
            print(f"    {entity['name']}: type={entity['type']}, count={entity['count']}")

        # Storage ratio calculation
        print("\n" + "-" * 80)
        print("STORAGE RATIO ANALYSIS")
        print("-" * 80)

        # Estimate Neo4j storage (rough calculation)
        # Each node ~200 bytes average (ID, properties, indexes)
        # Each relationship ~100 bytes
        # Each property ~50 bytes average
        estimated_neo4j_bytes = (
            final_stats["total_nodes"] * 200 +
            final_stats["total_edges"] * 100 +
            final_stats["total_properties"] * 50
        )

        if stats.total_file_size_bytes > 0:
            expansion_ratio = estimated_neo4j_bytes / stats.total_file_size_bytes
            print(f"\n  Source file size: {stats.total_file_size_bytes:,} bytes")
            print(f"  Estimated Neo4j size: {estimated_neo4j_bytes:,} bytes")
            print(f"  Expansion ratio: {expansion_ratio:.2f}x")

            print(f"\n  Projected for 500 large files (avg 50KB each):")
            projected_source = 500 * 50 * 1024  # 25 MB
            projected_neo4j = projected_source * expansion_ratio
            print(f"    Source files: {projected_source / 1024 / 1024:.1f} MB")
            print(f"    Neo4j storage: {projected_neo4j / 1024 / 1024:.1f} MB")

            if projected_neo4j < 500 * 1024 * 1024:  # Less than 500MB
                print(f"    → Should be fine for most Neo4j deployments")
            elif projected_neo4j < 2 * 1024 * 1024 * 1024:  # Less than 2GB
                print(f"    → Moderate size, may need dedicated instance")
            else:
                print(f"    → Large dataset, consider Neo4j Enterprise or chunking strategy")

        # Print sample for visual inspection
        print("\n" + "-" * 80)
        print("SAMPLE GRAPH DATA (for visual inspection)")
        print("-" * 80)

        print("\n  Sample nodes:")
        for node in sample["nodes"][:10]:
            print(f"    [{node.get('type', '?')}] {node.get('name', '?')} "
                  f"(synset: {node.get('synset_id', '-')})")

        print("\n  Sample edges:")
        for edge in sample["edges"][:10]:
            print(f"    {edge.get('source', '?')} --[{edge.get('rel_type', '?')}]--> "
                  f"{edge.get('target', '?')} (conf: {edge.get('confidence', '?')})")

        print("\n  To explore the full graph:")
        print("    Open http://localhost:7474 in your browser")
        print("    Run: MATCH (n) RETURN n LIMIT 100")

    finally:
        await driver.close()

    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
