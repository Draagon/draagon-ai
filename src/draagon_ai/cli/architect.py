"""CLI for Behavior Architect.

Provides command-line interface for creating behaviors.

Usage:
    # Create a behavior interactively
    python -m draagon_ai.cli.architect create "A behavior for managing kitchen timers"

    # Create with evolution
    python -m draagon_ai.cli.architect create "A behavior for home automation" --evolve

    # Research only
    python -m draagon_ai.cli.architect research "smart home management"
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any


def create_mock_llm():
    """Create a mock LLM for testing when no real LLM is configured."""
    class MockLLM:
        async def generate(self, prompt: str, system_prompt: str | None = None, temperature: float = 0.7) -> str:
            # Return minimal valid responses for testing
            if "research" in prompt.lower():
                return """
                <research>
                  <core_tasks><task>Handle user requests</task></core_tasks>
                  <suggested_actions>
                    <action name="handle_request" description="Handle a request"/>
                  </suggested_actions>
                  <triggers><trigger>user request</trigger></triggers>
                  <constraints/>
                  <domain_knowledge>Basic handling</domain_knowledge>
                  <sources/>
                </research>
                """
            elif "design" in prompt.lower():
                return """
                <design>
                  <behavior_id>new_behavior</behavior_id>
                  <name>New Behavior</name>
                  <description>Handles requests</description>
                  <actions>
                    <action name="handle"><description>Handle it</description></action>
                  </actions>
                  <triggers><trigger name="main" priority="50"><semantic>request</semantic></trigger></triggers>
                  <constraints/>
                </design>
                """
            elif "prompt" in prompt.lower():
                return """
                <prompts>
                  <decision_prompt>Decide what to do</decision_prompt>
                  <synthesis_prompt>Format response</synthesis_prompt>
                </prompts>
                """
            elif "test" in prompt.lower():
                return """
                <test_cases>
                  <test id="t1" name="Test"><user_query>test</user_query></test>
                </test_cases>
                """
            else:
                return '<action name="handle"/>'

    return MockLLM()


async def research_command(args: argparse.Namespace) -> None:
    """Research a domain."""
    from ..services import BehaviorArchitectService

    print(f"\n🔍 Researching: {args.description}\n")

    llm = create_mock_llm()
    service = BehaviorArchitectService(llm=llm)

    research = await service.research_domain(
        args.description,
        search_web=not args.no_web,
    )

    print("📋 Research Results:")
    print(f"   Domain: {research.domain}")
    print(f"\n   Core Tasks:")
    for task in research.core_tasks:
        print(f"   - {task}")

    print(f"\n   Suggested Actions:")
    for action in research.suggested_actions:
        name = action.get("name", "unknown")
        desc = action.get("description", "")
        print(f"   - {name}: {desc}")

    print(f"\n   Triggers:")
    for trigger in research.suggested_triggers:
        print(f"   - {trigger}")

    if research.domain_knowledge:
        print(f"\n   Domain Knowledge:")
        print(f"   {research.domain_knowledge[:200]}...")

    if args.output:
        output_path = Path(args.output)
        with open(output_path, "w") as f:
            json.dump({
                "domain": research.domain,
                "core_tasks": research.core_tasks,
                "suggested_actions": research.suggested_actions,
                "triggers": research.suggested_triggers,
                "constraints": research.constraints,
                "domain_knowledge": research.domain_knowledge,
                "sources": research.sources,
            }, f, indent=2)
        print(f"\n💾 Saved to: {output_path}")


async def create_command(args: argparse.Namespace) -> None:
    """Create a behavior."""
    from ..services import BehaviorArchitectService
    from ..behaviors import BehaviorRegistry

    print(f"\n🏗️  Creating behavior: {args.description}\n")

    llm = create_mock_llm()

    # Create registry for storage
    storage_path = Path(args.storage) if args.storage else None
    registry = BehaviorRegistry(storage_path=storage_path) if storage_path else None

    service = BehaviorArchitectService(
        llm=llm,
        registry=registry,
    )

    print("📝 Phase 1: Research...")
    research = await service.research_domain(args.description)
    print(f"   Found {len(research.core_tasks)} core tasks")

    print("📐 Phase 2: Design...")
    design = await service.design_behavior(research)
    print(f"   Designed {len(design.actions)} actions")

    print("🔨 Phase 3: Build...")
    behavior, test_cases = await service.build_behavior(design)
    print(f"   Built with {len(test_cases)} test cases")

    print("🧪 Phase 4: Test & Iterate...")
    behavior = await service.test_and_iterate(behavior, test_cases)
    pass_rate = behavior.test_results.pass_rate if behavior.test_results else 0
    print(f"   Pass rate: {pass_rate:.1%}")

    if args.evolve:
        from ..behaviors.types import EvolutionConfig
        print("🧬 Phase 5: Evolution...")
        config = EvolutionConfig(generations=args.generations or 5)
        result = await service.evolve_behavior(behavior, test_cases, config)
        if result.approved:
            behavior = result.evolved_behavior
            print(f"   Evolution improved fitness: {result.original_fitness:.2f} → {result.evolved_fitness:.2f}")
        else:
            print(f"   Evolution not approved (overfitting: {result.overfitting_gap:.2f})")

    if registry:
        print("📦 Phase 6: Register...")
        behavior_id = await service.register_behavior(behavior)
        print(f"   Registered as: {behavior_id}")

    print(f"\n✅ Created behavior: {behavior.name}")
    print(f"   ID: {behavior.behavior_id}")
    print(f"   Actions: {len(behavior.actions)}")
    print(f"   Status: {behavior.status.value}")
    print(f"   Tier: {behavior.tier.value}")

    if args.output:
        output_path = Path(args.output)
        # Export behavior as JSON
        from dataclasses import asdict
        with open(output_path, "w") as f:
            data = {
                "behavior_id": behavior.behavior_id,
                "name": behavior.name,
                "description": behavior.description,
                "actions": [
                    {"name": a.name, "description": a.description}
                    for a in behavior.actions
                ],
                "test_pass_rate": pass_rate,
            }
            json.dump(data, f, indent=2)
        print(f"\n💾 Saved to: {output_path}")


async def list_command(args: argparse.Namespace) -> None:
    """List behaviors in registry."""
    from ..behaviors import BehaviorRegistry

    if not args.storage:
        print("Error: --storage path required")
        sys.exit(1)

    storage_path = Path(args.storage)
    if not storage_path.exists():
        print(f"No behaviors found in {storage_path}")
        return

    registry = BehaviorRegistry(storage_path=storage_path)
    stats = registry.get_stats()

    print(f"\n📚 Behavior Registry: {storage_path}")
    print(f"   Total behaviors: {stats['total_behaviors']}")
    print(f"   Active: {stats['active_count']}")
    print(f"   Total actions: {stats['total_actions']}")

    print("\n   By Tier:")
    for tier, count in stats['by_tier'].items():
        if count > 0:
            print(f"   - {tier}: {count}")

    print("\n   By Status:")
    for status, count in stats['by_status'].items():
        if count > 0:
            print(f"   - {status}: {count}")

    if args.verbose:
        print("\n   Behaviors:")
        for behavior in registry.get_all():
            print(f"   - {behavior.behavior_id}: {behavior.name} ({behavior.status.value})")


def architect_cli() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Behavior Architect - Create AI behaviors from descriptions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Research a domain
  python -m draagon_ai.cli.architect research "smart home management"

  # Create a behavior
  python -m draagon_ai.cli.architect create "A behavior for kitchen timers"

  # Create with evolution
  python -m draagon_ai.cli.architect create "home automation" --evolve --generations 10

  # List behaviors
  python -m draagon_ai.cli.architect list --storage ./behaviors
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Research command
    research_parser = subparsers.add_parser("research", help="Research a domain")
    research_parser.add_argument("description", help="Domain to research")
    research_parser.add_argument("--no-web", action="store_true", help="Skip web search")
    research_parser.add_argument("-o", "--output", help="Output file (JSON)")

    # Create command
    create_parser = subparsers.add_parser("create", help="Create a behavior")
    create_parser.add_argument("description", help="Behavior description")
    create_parser.add_argument("--evolve", action="store_true", help="Evolve the behavior")
    create_parser.add_argument("--generations", type=int, help="Evolution generations")
    create_parser.add_argument("--storage", help="Registry storage path")
    create_parser.add_argument("-o", "--output", help="Output file (JSON)")

    # List command
    list_parser = subparsers.add_parser("list", help="List behaviors")
    list_parser.add_argument("--storage", required=True, help="Registry storage path")
    list_parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Run the appropriate command
    if args.command == "research":
        asyncio.run(research_command(args))
    elif args.command == "create":
        asyncio.run(create_command(args))
    elif args.command == "list":
        asyncio.run(list_command(args))


if __name__ == "__main__":
    architect_cli()
