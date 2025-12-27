"""Prompt Loader for loading prompts into Qdrant.

This module provides functionality to load prompts from domain files
into Qdrant for versioning, evolution, and runtime retrieval.

Prompt Loading Hierarchy (later sources override earlier):
1. Core prompts (draagon-ai framework)
2. Capability prompts (draagon-ai framework, optional)
3. Extension prompts (from Extension.get_prompt_domains())
4. App prompts (application-specific overrides)

Example:
    from draagon_ai.prompts.loader import PromptLoader
    from draagon_ai.prompts.domains import ALL_PROMPTS

    loader = PromptLoader(qdrant_url="http://192.168.168.216:6333")
    await loader.initialize()

    # Load all prompts (with extension support)
    await loader.load_all_prompts(
        extension_manager=ext_manager,
        capabilities=["home_assistant", "calendar"],
    )

    # Get a prompt at runtime
    decision_prompt = await loader.get_prompt("decision", "DECISION_PROMPT")
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any

from .types import Prompt, PromptDomain, PromptMetadata, PromptStatus
from .registry import PromptRegistry

if TYPE_CHECKING:
    from draagon_ai.extensions import ExtensionManager

logger = logging.getLogger(__name__)


class PromptLoader:
    """Loads prompts from domain definitions into Qdrant.

    The loader handles:
    - Initial loading of prompts from domain dictionaries
    - Versioning and change detection
    - Runtime prompt retrieval with caching
    - Support for prompt evolution

    Example:
        loader = PromptLoader()
        await loader.initialize()

        # Load all prompts
        await loader.load_prompts(ALL_PROMPTS)

        # Runtime usage
        prompt = await loader.get_prompt("decision", "DECISION_PROMPT")
        filled = loader.fill_prompt(prompt.content, question="Hello", ...)
    """

    def __init__(
        self,
        qdrant_url: str = "http://192.168.168.216:6333",
        collection_name: str = "draagon_prompts",
        embedding_provider: Any = None,
    ):
        """Initialize the loader.

        Args:
            qdrant_url: Qdrant server URL
            collection_name: Collection name for prompts
            embedding_provider: Provider for generating embeddings
        """
        self._registry = PromptRegistry(
            qdrant_url=qdrant_url,
            collection_name=collection_name,
            embedding_provider=embedding_provider,
        )
        self._loaded_prompts: dict[str, dict[str, str]] = {}

    async def initialize(self) -> None:
        """Initialize the Qdrant connection."""
        await self._registry.initialize()

    async def close(self) -> None:
        """Close the Qdrant connection."""
        await self._registry.close()

    async def load_prompts(
        self,
        prompts: dict[str, dict[str, Any]],
        force: bool = False,
        version: str = "1.0.0",
    ) -> dict[str, int]:
        """Load prompts from domain dictionaries into Qdrant.

        Args:
            prompts: Domain -> {prompt_name: content} dictionary
            force: If True, overwrite existing prompts
            version: Version string for new prompts

        Returns:
            Dict of domain -> count of prompts loaded

        Example:
            from draagon_ai.prompts.domains import ALL_PROMPTS

            stats = await loader.load_prompts(ALL_PROMPTS)
            # {"routing": 2, "decision": 1, ...}
        """
        stats: dict[str, int] = {}

        for domain_name, domain_prompts in prompts.items():
            count = 0

            for prompt_name, content in domain_prompts.items():
                # Skip non-string content (like MODE_MODIFIERS dict)
                if not isinstance(content, str):
                    continue

                # Check if already exists
                existing = await self._registry.get_prompt(domain_name, prompt_name)

                if existing and not force:
                    # Check if content changed
                    if existing.content == content:
                        continue  # No change, skip

                # Store in Qdrant - pass domain_name as string to preserve original
                await self._registry.store_prompt(
                    domain=domain_name,  # Pass string, let registry handle enum mapping
                    name=prompt_name,
                    content=content,
                    description=f"{domain_name} prompt: {prompt_name}",
                    version=version,
                    status=PromptStatus.ACTIVE,
                )

                count += 1

            stats[domain_name] = count

        return stats

    async def load_all_prompts(
        self,
        extension_manager: "ExtensionManager | None" = None,
        capabilities: list[str] | None = None,
        app_prompts: dict[str, dict[str, str]] | None = None,
        force: bool = False,
        version: str = "1.0.0",
    ) -> dict[str, int]:
        """Load prompts from all sources with proper precedence.

        Loads prompts in this order (later overrides earlier):
        1. Core prompts (always loaded)
        2. Capability prompts (if capability in list)
        3. Extension prompts (from extension.get_prompt_domains())
        4. App prompts (application-specific overrides)

        Args:
            extension_manager: Manager for loaded extensions
            capabilities: List of capability domains to load
            app_prompts: App-specific prompt overrides
            force: Overwrite existing prompts
            version: Version for new prompts

        Returns:
            Dict of domain -> count of prompts loaded

        Example:
            from draagon_ai.extensions import get_extension_manager

            stats = await loader.load_all_prompts(
                extension_manager=get_extension_manager(),
                capabilities=["home_assistant", "calendar"],
                app_prompts={"decision": {"CUSTOM_PROMPT": "..."}},
            )
        """
        from .domains import CORE_PROMPTS, CAPABILITY_PROMPTS

        all_prompts: dict[str, dict[str, str]] = {}
        total_stats: dict[str, int] = {}

        # 1. Start with core prompts
        for domain, prompts in CORE_PROMPTS.items():
            all_prompts[domain] = dict(prompts)
        logger.debug(f"Loaded {len(CORE_PROMPTS)} core prompt domains")

        # 2. Add enabled capabilities
        if capabilities:
            for cap in capabilities:
                if cap in CAPABILITY_PROMPTS:
                    if cap not in all_prompts:
                        all_prompts[cap] = {}
                    all_prompts[cap].update(CAPABILITY_PROMPTS[cap])
                    logger.debug(f"Loaded capability: {cap}")
                else:
                    logger.warning(f"Unknown capability: {cap}")

        # 3. Add extension prompts
        if extension_manager:
            ext_domains = extension_manager.get_all_prompt_domains()
            for domain, prompts in ext_domains.items():
                if domain not in all_prompts:
                    all_prompts[domain] = {}
                all_prompts[domain].update(prompts)
            logger.debug(f"Loaded {len(ext_domains)} extension prompt domains")

        # 4. Add app overrides (highest priority)
        if app_prompts:
            for domain, prompts in app_prompts.items():
                if domain not in all_prompts:
                    all_prompts[domain] = {}
                all_prompts[domain].update(prompts)
            logger.debug(f"Applied {len(app_prompts)} app prompt overrides")

        # Load all prompts into Qdrant
        stats = await self.load_prompts(all_prompts, force=force, version=version)

        # Summarize
        total = sum(stats.values())
        logger.info(f"Loaded {total} prompts across {len(stats)} domains")

        return stats

    async def get_prompt(
        self,
        domain: str,
        name: str,
        version: str | None = None,
    ) -> Prompt | None:
        """Get a prompt by domain and name.

        Args:
            domain: Prompt domain
            name: Prompt name
            version: Specific version (None = current)

        Returns:
            Prompt if found, None otherwise
        """
        return await self._registry.get_prompt(domain, name, version)

    async def get_prompt_content(
        self,
        domain: str,
        name: str,
    ) -> str | None:
        """Get prompt content directly.

        Args:
            domain: Prompt domain
            name: Prompt name

        Returns:
            Prompt content string or None
        """
        prompt = await self.get_prompt(domain, name)
        return prompt.content if prompt else None

    def fill_prompt(
        self,
        content: str,
        **kwargs: Any,
    ) -> str:
        """Fill prompt template with values.

        Args:
            content: Prompt template with {variable} placeholders
            **kwargs: Variable values

        Returns:
            Filled prompt string

        Example:
            filled = loader.fill_prompt(
                prompt.content,
                question="What time is it?",
                user_id="doug",
            )
        """
        try:
            return content.format(**kwargs)
        except KeyError as e:
            # Return partial fill if some variables missing
            for key, value in kwargs.items():
                content = content.replace(f"{{{key}}}", str(value))
            return content

    async def list_domains(self) -> list[str]:
        """List all domains with prompts.

        Returns:
            List of domain names
        """
        prompts = await self._registry.list_prompts()
        domains = set(p.domain.value for p in prompts)
        return sorted(domains)

    async def list_prompts(
        self,
        domain: str | None = None,
    ) -> list[Prompt]:
        """List prompts, optionally filtered by domain.

        Args:
            domain: Optional domain filter

        Returns:
            List of prompts
        """
        return await self._registry.list_prompts(domain=domain)

    async def update_metrics(
        self,
        domain: str,
        name: str,
        version: str,
        success: bool,
        latency_ms: float,
    ) -> None:
        """Update usage metrics for a prompt.

        Args:
            domain: Prompt domain
            name: Prompt name
            version: Version string
            success: Whether the interaction was successful
            latency_ms: Response latency
        """
        await self._registry.update_metrics(
            domain=domain,
            name=name,
            version=version,
            success=success,
            latency_ms=latency_ms,
        )

    async def create_shadow_version(
        self,
        domain: str,
        name: str,
        content: str,
        parent_version: str,
    ) -> Prompt:
        """Create a shadow version for A/B testing.

        Args:
            domain: Prompt domain
            name: Prompt name
            content: New prompt content
            parent_version: Version this was derived from

        Returns:
            The created prompt with shadow version
        """
        # Generate next version
        parts = parent_version.split(".")
        minor = int(parts[1]) + 1 if len(parts) > 1 else 1
        new_version = f"{parts[0]}.{minor}.0"

        try:
            prompt_domain = PromptDomain(domain)
        except ValueError:
            prompt_domain = PromptDomain.CUSTOM

        return await self._registry.store_prompt(
            domain=prompt_domain,
            name=name,
            content=content,
            version=new_version,
            status=PromptStatus.SHADOW,
            metadata=PromptMetadata(
                domain=prompt_domain,
                version=new_version,
                parent_version=parent_version,
                created_by="evolution",
            ),
        )

    async def promote_shadow(
        self,
        domain: str,
        name: str,
        version: str,
    ) -> bool:
        """Promote a shadow version to active.

        Args:
            domain: Prompt domain
            name: Prompt name
            version: Shadow version to promote

        Returns:
            True if successful
        """
        prompt = await self.get_prompt(domain, name)
        if not prompt:
            return False

        return prompt.activate_version(version)
