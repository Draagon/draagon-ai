"""Decorator-based tool registration for draagon-ai.

This module provides a @tool decorator for clean, declarative tool registration.
Tools are automatically discovered and registered when their modules are imported.

Usage:
    from draagon_ai.tools import tool

    @tool(
        name="get_time",
        description="Get the current time and date",
    )
    async def get_time(args: dict, **context) -> dict:
        return {"time": datetime.now().isoformat()}

    @tool(
        name="home_assistant",
        description="Control a Home Assistant device",
        parameters={
            "domain": {"type": "string", "required": True, "description": "Device domain"},
            "service": {"type": "string", "required": True, "description": "Service to call"},
        },
        category="smart_home",
    )
    async def home_assistant(args: dict, **context) -> dict:
        ...

Auto-discovery:
    from draagon_ai.tools import discover_tools, get_all_tools

    # Import all tool modules to register decorators
    discover_tools("myapp.tools.handlers")

    # Get all registered tools
    tools = get_all_tools()
"""

from __future__ import annotations

import importlib
import logging
import pkgutil
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable, Coroutine

from draagon_ai.orchestration.registry import Tool, ToolParameter, ToolRegistry

logger = logging.getLogger(__name__)

# Global registry for decorated tools
_global_registry = ToolRegistry()

# Track tool metadata for introspection
_tool_metadata: dict[str, ToolMetadata] = {}


@dataclass
class ToolMetadata:
    """Extended metadata for a tool (beyond what Tool stores).

    Attributes:
        name: Tool name
        description: Tool description
        handler: Async handler function
        parameters: Tool parameters
        returns: Return type description
        category: Tool category for grouping
        requires_confirmation: Whether user confirmation is needed
        tags: Additional tags for filtering
    """

    name: str
    description: str
    handler: Callable[..., Coroutine[Any, Any, Any]]
    parameters: list[ToolParameter] = field(default_factory=list)
    returns: str = "object"
    category: str = "general"
    requires_confirmation: bool = False
    tags: list[str] = field(default_factory=list)


def tool(
    name: str,
    description: str,
    *,
    parameters: dict[str, dict[str, Any]] | list[ToolParameter] | None = None,
    returns: str = "object",
    category: str = "general",
    requires_confirmation: bool = False,
    tags: list[str] | None = None,
    timeout_ms: int = 30000,
) -> Callable[[Callable], Callable]:
    """Decorator to register a function as a tool.

    Args:
        name: Unique tool name (e.g., "get_time", "home_assistant")
        description: Human-readable description of what the tool does
        parameters: Tool parameters, either as:
            - dict[str, dict]: {"param_name": {"type": "string", "description": "..."}}
            - list[ToolParameter]: Direct ToolParameter objects
        returns: Description of return type
        category: Tool category for grouping (e.g., "time", "smart_home", "calendar")
        requires_confirmation: Whether this tool needs user confirmation before execution
        tags: Additional tags for filtering/discovery
        timeout_ms: Execution timeout in milliseconds (default: 30000)

    Returns:
        Decorator function

    Example:
        @tool(
            name="get_time",
            description="Get the current time and date",
        )
        async def get_time(args: dict, **context) -> dict:
            return {"time": datetime.now().isoformat()}

        @tool(
            name="search_web",
            description="Search the web for information",
            parameters={"query": {"type": "string", "description": "Search query"}},
            category="search",
            tags=["web", "research"],
        )
        async def search_web(args: dict, **context) -> dict:
            query = args.get("query", "")
            # ... perform search
            return {"results": [...]}
    """

    def decorator(func: Callable[..., Coroutine[Any, Any, Any]]) -> Callable:
        # Convert parameters to ToolParameter list
        param_list = _convert_parameters(parameters)

        # Create Tool instance
        tool_def = Tool(
            name=name,
            description=description,
            handler=func,
            parameters=param_list,
            returns=returns,
            timeout_ms=timeout_ms,
            requires_confirmation=requires_confirmation,
        )

        # Register in global registry
        _global_registry.register(tool_def)

        # Store extended metadata
        _tool_metadata[name] = ToolMetadata(
            name=name,
            description=description,
            handler=func,
            parameters=param_list,
            returns=returns,
            category=category,
            requires_confirmation=requires_confirmation,
            tags=tags or [],
        )

        logger.debug(f"Registered tool: {name} (category={category})")

        # Return the original function wrapped
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            return await func(*args, **kwargs)

        # Attach metadata to function for introspection
        wrapper._tool_name = name  # type: ignore[attr-defined]
        wrapper._tool_metadata = _tool_metadata[name]  # type: ignore[attr-defined]

        return wrapper

    return decorator


def _convert_parameters(
    parameters: dict[str, dict[str, Any]] | list[ToolParameter] | None,
) -> list[ToolParameter]:
    """Convert parameters from dict or list format to ToolParameter list.

    Args:
        parameters: Parameters in dict or list format

    Returns:
        List of ToolParameter objects
    """
    if parameters is None:
        return []

    if isinstance(parameters, list):
        return parameters

    # Convert dict format to ToolParameter list
    result = []
    for param_name, param_spec in parameters.items():
        result.append(
            ToolParameter(
                name=param_name,
                type=param_spec.get("type", "string"),
                description=param_spec.get("description", ""),
                required=param_spec.get("required", True),
                enum=param_spec.get("enum"),
                default=param_spec.get("default"),
            )
        )
    return result


def get_global_registry() -> ToolRegistry:
    """Get the global tool registry containing all decorated tools.

    Returns:
        The global ToolRegistry instance
    """
    return _global_registry


def get_all_tools() -> list[Tool]:
    """Get all registered tools.

    Returns:
        List of all registered Tool objects
    """
    return _global_registry.get_all()


def get_tool(name: str) -> Tool | None:
    """Get a specific tool by name.

    Args:
        name: Tool name

    Returns:
        Tool object if found, None otherwise
    """
    return _global_registry.get(name)


def get_tool_metadata(name: str) -> ToolMetadata | None:
    """Get extended metadata for a tool.

    Args:
        name: Tool name

    Returns:
        ToolMetadata if found, None otherwise
    """
    return _tool_metadata.get(name)


def get_tools_by_category(category: str) -> list[Tool]:
    """Get all tools in a specific category.

    Args:
        category: Category name

    Returns:
        List of Tool objects in the category
    """
    return [
        _global_registry.get(name)
        for name, meta in _tool_metadata.items()
        if meta.category == category and _global_registry.get(name) is not None
    ]


def get_tools_by_tag(tag: str) -> list[Tool]:
    """Get all tools with a specific tag.

    Args:
        tag: Tag name

    Returns:
        List of Tool objects with the tag
    """
    return [
        _global_registry.get(name)
        for name, meta in _tool_metadata.items()
        if tag in meta.tags and _global_registry.get(name) is not None
    ]


def list_categories() -> list[str]:
    """List all tool categories.

    Returns:
        Sorted list of category names
    """
    return sorted(set(meta.category for meta in _tool_metadata.values()))


def list_tags() -> list[str]:
    """List all tool tags.

    Returns:
        Sorted list of tag names
    """
    all_tags = set()
    for meta in _tool_metadata.values():
        all_tags.update(meta.tags)
    return sorted(all_tags)


def discover_tools(package: str) -> list[str]:
    """Discover and import all tool modules in a package.

    This triggers the @tool decorators to run and register tools.

    Args:
        package: Package name to search for tool modules (e.g., "myapp.tools.handlers")

    Returns:
        List of discovered tool names

    Example:
        # Discover all tools in handlers package
        tools = discover_tools("myapp.tools.handlers")
        print(f"Discovered {len(tools)} tools")
    """
    try:
        pkg = importlib.import_module(package)
    except ImportError as e:
        logger.warning(f"Could not import package {package}: {e}")
        return []

    discovered = []

    # Get the package path
    if not hasattr(pkg, "__path__"):
        logger.warning(f"{package} is not a package (no __path__)")
        return []

    # Import all submodules
    for _, module_name, is_pkg in pkgutil.iter_modules(pkg.__path__):
        full_name = f"{package}.{module_name}"
        try:
            importlib.import_module(full_name)
            logger.debug(f"Imported tool module: {full_name}")

            # If it's a subpackage, recurse into it
            if is_pkg:
                discovered.extend(discover_tools(full_name))
        except ImportError as e:
            logger.warning(f"Could not import {full_name}: {e}")

    # Return names of all tools registered so far
    discovered.extend(_global_registry.list_tools())
    return list(set(discovered))  # Remove duplicates


def clear_registry() -> None:
    """Clear all registered tools.

    Useful for testing to reset state between tests.
    """
    global _global_registry, _tool_metadata
    _global_registry = ToolRegistry()
    _tool_metadata = {}


def set_registry(registry: ToolRegistry) -> None:
    """Set a custom global registry.

    Useful for testing or when you need to share a registry
    between different parts of the application.

    Args:
        registry: ToolRegistry instance to use as the global registry
    """
    global _global_registry
    _global_registry = registry


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Decorator
    "tool",
    # Registry access
    "get_global_registry",
    "get_all_tools",
    "get_tool",
    "get_tool_metadata",
    "get_tools_by_category",
    "get_tools_by_tag",
    "list_categories",
    "list_tags",
    # Discovery
    "discover_tools",
    "clear_registry",
    "set_registry",
    # Types
    "ToolMetadata",
]
