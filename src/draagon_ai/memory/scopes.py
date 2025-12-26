"""Hierarchical scope system for the Temporal Cognitive Graph.

Scopes control visibility and access to nodes in the graph:
- World: Universal facts accessible to all agents
- Context: Shared within a context (household, game session, team)
- Agent: Agent's private memories
- User: Per-user memories within an agent
- Session: Current conversation only

Based on research from:
- Multi-tenant cognitive architectures
- Roxy's existing scope system (extended)
- Netflix's access control patterns
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any
import threading
import uuid


class ScopeType(str, Enum):
    """Types of hierarchical scopes.

    Scopes form a hierarchy: WORLD > CONTEXT > AGENT > USER > SESSION
    Each scope inherits from its parent.
    """

    WORLD = "world"        # Universal (capitals, physics, etc.)
    CONTEXT = "context"    # Shared context (household, game)
    AGENT = "agent"        # Agent-specific
    USER = "user"          # Per-user within agent
    SESSION = "session"    # Single conversation


class Permission(str, Enum):
    """Permissions for scope access."""

    READ = "read"          # Can read nodes in this scope
    WRITE = "write"        # Can create/update nodes
    DELETE = "delete"      # Can delete nodes
    ADMIN = "admin"        # Can manage permissions


@dataclass
class ScopePermission:
    """Permission grant for an agent/user on a scope."""

    grantee_id: str           # Agent or user ID
    grantee_type: str         # "agent" or "user"
    permissions: set[Permission]
    granted_at: datetime = field(default_factory=datetime.now)
    granted_by: str | None = None
    expires_at: datetime | None = None

    def has_permission(self, permission: Permission) -> bool:
        """Check if this grant includes a specific permission."""
        if self.expires_at and datetime.now() > self.expires_at:
            return False
        return permission in self.permissions or Permission.ADMIN in self.permissions


@dataclass
class HierarchicalScope:
    """A scope in the hierarchical memory system.

    Scopes control:
    - Visibility: Who can see nodes in this scope
    - Permissions: Who can read/write/delete
    - Transience: How long nodes persist by default
    - Promotion: When nodes should be promoted to parent scope

    Example hierarchy:
        world:global
        └── context:mealing_home
            └── agent:roxy
                └── user:doug
                    └── session:abc123
    """

    # Identity
    scope_id: str
    scope_type: ScopeType
    name: str = ""

    # Hierarchy
    parent_scope_id: str | None = None
    child_scope_ids: list[str] = field(default_factory=list)

    # Permissions (list of grants)
    permissions: list[ScopePermission] = field(default_factory=list)

    # Transience settings
    default_ttl: timedelta | None = None  # Auto-expire nodes
    max_items: int | None = None          # Capacity limit (LRU eviction)

    # Promotion settings (when to move nodes to parent scope)
    promotion_threshold: float = 0.7      # Importance threshold
    promotion_access_count: int = 5       # Access count threshold
    auto_promote: bool = True             # Enable automatic promotion

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        """Set defaults based on scope type."""
        if self.default_ttl is None:
            self.default_ttl = DEFAULT_TTLS.get(self.scope_type)

    def can_read(self, agent_id: str, user_id: str | None = None) -> bool:
        """Check if an agent/user can read from this scope.

        Args:
            agent_id: Agent requesting access
            user_id: Optional user within agent

        Returns:
            True if read is permitted
        """
        # World scope is always readable
        if self.scope_type == ScopeType.WORLD:
            return True

        return self._check_permission(agent_id, user_id, Permission.READ)

    def can_write(self, agent_id: str, user_id: str | None = None) -> bool:
        """Check if an agent/user can write to this scope."""
        return self._check_permission(agent_id, user_id, Permission.WRITE)

    def can_delete(self, agent_id: str, user_id: str | None = None) -> bool:
        """Check if an agent/user can delete from this scope."""
        return self._check_permission(agent_id, user_id, Permission.DELETE)

    def _check_permission(
        self,
        agent_id: str,
        user_id: str | None,
        permission: Permission,
    ) -> bool:
        """Check if a specific permission is granted on this scope directly."""
        for grant in self.permissions:
            if grant.grantee_type == "agent" and grant.grantee_id == agent_id:
                if grant.has_permission(permission):
                    return True
            elif grant.grantee_type == "user" and user_id and grant.grantee_id == user_id:
                if grant.has_permission(permission):
                    return True
        return False

    def check_permission_with_inheritance(
        self,
        agent_id: str,
        user_id: str | None,
        permission: Permission,
        registry: "ScopeRegistry | None" = None,
    ) -> bool:
        """Check if a permission is granted, including inherited permissions.

        Permissions can be inherited from parent scopes. If a user has READ
        permission on "agent:roxy", they also have READ on child scopes like
        "user:roxy:doug" and "session:roxy:doug:abc123".

        Args:
            agent_id: Agent requesting access
            user_id: Optional user within agent
            permission: Permission to check
            registry: Scope registry for looking up parent scopes

        Returns:
            True if permission is granted directly or inherited
        """
        # Check direct permission on this scope
        if self._check_permission(agent_id, user_id, permission):
            return True

        # Check inherited permissions from parent scopes
        if registry and self.parent_scope_id:
            parent = registry.get(self.parent_scope_id)
            if parent:
                return parent.check_permission_with_inheritance(
                    agent_id, user_id, permission, registry
                )

        return False

    def grant_permission(
        self,
        grantee_id: str,
        grantee_type: str,
        permissions: set[Permission],
        granted_by: str | None = None,
        expires_at: datetime | None = None,
    ) -> ScopePermission:
        """Grant permissions to an agent or user.

        Args:
            grantee_id: Agent or user ID
            grantee_type: "agent" or "user"
            permissions: Set of permissions to grant
            granted_by: Who is granting (for audit)
            expires_at: Optional expiration

        Returns:
            The created permission grant
        """
        grant = ScopePermission(
            grantee_id=grantee_id,
            grantee_type=grantee_type,
            permissions=permissions,
            granted_by=granted_by,
            expires_at=expires_at,
        )
        self.permissions.append(grant)
        self.updated_at = datetime.now()
        return grant

    def revoke_permission(self, grantee_id: str, grantee_type: str) -> bool:
        """Revoke all permissions from an agent or user.

        Args:
            grantee_id: Agent or user ID
            grantee_type: "agent" or "user"

        Returns:
            True if any permissions were revoked
        """
        original_len = len(self.permissions)
        self.permissions = [
            p for p in self.permissions
            if not (p.grantee_id == grantee_id and p.grantee_type == grantee_type)
        ]
        if len(self.permissions) < original_len:
            self.updated_at = datetime.now()
            return True
        return False

    def should_promote(self, importance: float, access_count: int) -> bool:
        """Check if a node should be promoted to parent scope.

        Args:
            importance: Node's importance score
            access_count: Number of times accessed

        Returns:
            True if node should be promoted
        """
        if not self.auto_promote or self.parent_scope_id is None:
            return False

        return importance >= self.promotion_threshold or access_count >= self.promotion_access_count

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "scope_id": self.scope_id,
            "scope_type": self.scope_type.value,
            "name": self.name,
            "parent_scope_id": self.parent_scope_id,
            "child_scope_ids": self.child_scope_ids,
            "permissions": [
                {
                    "grantee_id": p.grantee_id,
                    "grantee_type": p.grantee_type,
                    "permissions": [perm.value for perm in p.permissions],
                    "granted_at": p.granted_at.isoformat(),
                    "granted_by": p.granted_by,
                    "expires_at": p.expires_at.isoformat() if p.expires_at else None,
                }
                for p in self.permissions
            ],
            "default_ttl": self.default_ttl.total_seconds() if self.default_ttl else None,
            "max_items": self.max_items,
            "promotion_threshold": self.promotion_threshold,
            "promotion_access_count": self.promotion_access_count,
            "auto_promote": self.auto_promote,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HierarchicalScope":
        """Create from dictionary."""
        permissions = []
        for p in data.get("permissions", []):
            permissions.append(ScopePermission(
                grantee_id=p["grantee_id"],
                grantee_type=p["grantee_type"],
                permissions={Permission(perm) for perm in p["permissions"]},
                granted_at=datetime.fromisoformat(p["granted_at"]),
                granted_by=p.get("granted_by"),
                expires_at=datetime.fromisoformat(p["expires_at"]) if p.get("expires_at") else None,
            ))

        ttl_seconds = data.get("default_ttl")
        default_ttl = timedelta(seconds=ttl_seconds) if ttl_seconds else None

        return cls(
            scope_id=data["scope_id"],
            scope_type=ScopeType(data["scope_type"]),
            name=data.get("name", ""),
            parent_scope_id=data.get("parent_scope_id"),
            child_scope_ids=data.get("child_scope_ids", []),
            permissions=permissions,
            default_ttl=default_ttl,
            max_items=data.get("max_items"),
            promotion_threshold=data.get("promotion_threshold", 0.7),
            promotion_access_count=data.get("promotion_access_count", 5),
            auto_promote=data.get("auto_promote", True),
            metadata=data.get("metadata", {}),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.now(),
            updated_at=datetime.fromisoformat(data["updated_at"]) if data.get("updated_at") else datetime.now(),
        )


# Default TTLs by scope type
DEFAULT_TTLS: dict[ScopeType, timedelta | None] = {
    ScopeType.WORLD: None,                    # Permanent
    ScopeType.CONTEXT: timedelta(days=365),   # 1 year
    ScopeType.AGENT: timedelta(days=90),      # 3 months
    ScopeType.USER: timedelta(days=30),       # 1 month
    ScopeType.SESSION: timedelta(hours=1),    # 1 hour
}


class ScopeRegistry:
    """Registry for managing hierarchical scopes.

    Provides scope lookup, creation, and hierarchy traversal.
    In production, this would be backed by persistent storage.
    """

    def __init__(self):
        """Initialize empty registry."""
        self._scopes: dict[str, HierarchicalScope] = {}

        # Create default world scope
        self._create_world_scope()

    def _create_world_scope(self) -> None:
        """Create the root world scope."""
        world = HierarchicalScope(
            scope_id="world:global",
            scope_type=ScopeType.WORLD,
            name="Global World",
        )
        self._scopes[world.scope_id] = world

    def get(self, scope_id: str) -> HierarchicalScope | None:
        """Get a scope by ID.

        Args:
            scope_id: Scope identifier

        Returns:
            Scope or None if not found
        """
        return self._scopes.get(scope_id)

    def create_scope(
        self,
        scope_type: ScopeType,
        name: str,
        parent_scope_id: str | None = None,
        scope_id: str | None = None,
        **kwargs: Any,
    ) -> HierarchicalScope:
        """Create a new scope.

        Args:
            scope_type: Type of scope
            name: Human-readable name
            parent_scope_id: Parent scope (uses default based on type if None)
            scope_id: Optional explicit ID (generated if None)
            **kwargs: Additional scope properties

        Returns:
            Created scope

        Raises:
            ValueError: If parent scope doesn't exist
        """
        # Generate scope_id if not provided
        if scope_id is None:
            scope_id = f"{scope_type.value}:{uuid.uuid4().hex[:8]}"

        # Determine parent if not specified
        if parent_scope_id is None:
            parent_scope_id = self._default_parent(scope_type)

        # Validate parent exists
        if parent_scope_id and parent_scope_id not in self._scopes:
            raise ValueError(f"Parent scope not found: {parent_scope_id}")

        scope = HierarchicalScope(
            scope_id=scope_id,
            scope_type=scope_type,
            name=name,
            parent_scope_id=parent_scope_id,
            **kwargs,
        )

        self._scopes[scope_id] = scope

        # Update parent's child list
        if parent_scope_id:
            parent = self._scopes[parent_scope_id]
            if scope_id not in parent.child_scope_ids:
                parent.child_scope_ids.append(scope_id)

        return scope

    def _default_parent(self, scope_type: ScopeType) -> str | None:
        """Get default parent scope ID for a scope type."""
        if scope_type == ScopeType.WORLD:
            return None
        elif scope_type == ScopeType.CONTEXT:
            return "world:global"
        elif scope_type == ScopeType.AGENT:
            # Context-less agent goes under world
            return "world:global"
        elif scope_type == ScopeType.USER:
            # User needs agent parent (must be specified)
            return None
        elif scope_type == ScopeType.SESSION:
            # Session needs user parent (must be specified)
            return None
        return None

    def get_ancestors(self, scope_id: str) -> list[HierarchicalScope]:
        """Get all ancestor scopes (parent, grandparent, etc.).

        Args:
            scope_id: Starting scope

        Returns:
            List of ancestor scopes from immediate parent to root
        """
        ancestors = []
        current = self.get(scope_id)

        while current and current.parent_scope_id:
            parent = self.get(current.parent_scope_id)
            if parent:
                ancestors.append(parent)
                current = parent
            else:
                break

        return ancestors

    def get_descendants(self, scope_id: str) -> list[HierarchicalScope]:
        """Get all descendant scopes (children, grandchildren, etc.).

        Args:
            scope_id: Starting scope

        Returns:
            List of all descendant scopes
        """
        descendants = []
        scope = self.get(scope_id)

        if not scope:
            return descendants

        def collect(s: HierarchicalScope):
            for child_id in s.child_scope_ids:
                child = self.get(child_id)
                if child:
                    descendants.append(child)
                    collect(child)

        collect(scope)
        return descendants

    def get_readable_scopes(self, agent_id: str, user_id: str | None = None) -> list[HierarchicalScope]:
        """Get all scopes readable by an agent/user.

        This includes:
        - World scope (always readable)
        - Scopes with explicit read permission
        - Ancestor scopes of readable scopes

        Args:
            agent_id: Agent ID
            user_id: Optional user ID

        Returns:
            List of readable scopes
        """
        readable = []

        for scope in self._scopes.values():
            if scope.can_read(agent_id, user_id):
                readable.append(scope)

        return readable

    def check_permission(
        self,
        scope_id: str,
        agent_id: str,
        user_id: str | None,
        permission: Permission,
    ) -> bool:
        """Check if an agent/user has a permission on a scope, with inheritance.

        This is the preferred method for checking permissions as it handles:
        - Direct permissions on the scope
        - Inherited permissions from parent scopes
        - World scope always being readable

        Args:
            scope_id: Scope to check
            agent_id: Agent requesting access
            user_id: Optional user within agent
            permission: Permission to check

        Returns:
            True if permission is granted (directly or inherited)
        """
        scope = self.get(scope_id)
        if not scope:
            return False

        # World scope is always readable
        if scope.scope_type == ScopeType.WORLD and permission == Permission.READ:
            return True

        # Check with inheritance
        return scope.check_permission_with_inheritance(
            agent_id, user_id, permission, self
        )

    def build_scope_id(
        self,
        scope_type: ScopeType,
        context_id: str | None = None,
        agent_id: str | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
    ) -> str:
        """Build a scope ID from components.

        Args:
            scope_type: Type of scope
            context_id: Context identifier
            agent_id: Agent identifier
            user_id: User identifier
            session_id: Session identifier

        Returns:
            Scope ID string

        Example:
            build_scope_id(ScopeType.USER, agent_id="roxy", user_id="doug")
            # Returns: "user:roxy:doug"
        """
        parts = [scope_type.value]

        if scope_type == ScopeType.WORLD:
            parts.append("global")
        elif scope_type == ScopeType.CONTEXT:
            parts.append(context_id or "default")
        elif scope_type == ScopeType.AGENT:
            parts.append(agent_id or "default")
        elif scope_type == ScopeType.USER:
            parts.append(agent_id or "default")
            parts.append(user_id or "default")
        elif scope_type == ScopeType.SESSION:
            parts.append(agent_id or "default")
            parts.append(user_id or "default")
            parts.append(session_id or str(uuid.uuid4())[:8])

        return ":".join(parts)

    def ensure_scope_hierarchy(
        self,
        context_id: str | None = None,
        agent_id: str | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
    ) -> str:
        """Ensure the full scope hierarchy exists, creating scopes as needed.

        Creates all scopes in the hierarchy from context down to session,
        with appropriate permissions.

        Args:
            context_id: Context (household, game) ID
            agent_id: Agent ID
            user_id: User ID
            session_id: Session ID

        Returns:
            The leaf scope ID (most specific scope for these parameters)
        """
        current_parent = "world:global"

        # Context scope
        if context_id:
            ctx_scope_id = f"context:{context_id}"
            if ctx_scope_id not in self._scopes:
                self.create_scope(
                    ScopeType.CONTEXT,
                    name=context_id,
                    parent_scope_id=current_parent,
                    scope_id=ctx_scope_id,
                )
            current_parent = ctx_scope_id

        # Agent scope
        if agent_id:
            agent_scope_id = f"agent:{agent_id}"
            if agent_scope_id not in self._scopes:
                scope = self.create_scope(
                    ScopeType.AGENT,
                    name=agent_id,
                    parent_scope_id=current_parent,
                    scope_id=agent_scope_id,
                )
                # Agent has full permissions on their scope
                scope.grant_permission(
                    agent_id,
                    "agent",
                    {Permission.READ, Permission.WRITE, Permission.DELETE, Permission.ADMIN},
                )
            current_parent = agent_scope_id

        # User scope
        if user_id and agent_id:
            user_scope_id = f"user:{agent_id}:{user_id}"
            if user_scope_id not in self._scopes:
                scope = self.create_scope(
                    ScopeType.USER,
                    name=user_id,
                    parent_scope_id=current_parent,
                    scope_id=user_scope_id,
                )
                # User has read/write, agent has admin
                scope.grant_permission(
                    user_id,
                    "user",
                    {Permission.READ, Permission.WRITE},
                )
                scope.grant_permission(
                    agent_id,
                    "agent",
                    {Permission.READ, Permission.WRITE, Permission.DELETE, Permission.ADMIN},
                )
            current_parent = user_scope_id

        # Session scope
        if session_id and agent_id:
            session_scope_id = f"session:{agent_id}:{user_id or 'anon'}:{session_id}"
            if session_scope_id not in self._scopes:
                scope = self.create_scope(
                    ScopeType.SESSION,
                    name=session_id,
                    parent_scope_id=current_parent,
                    scope_id=session_scope_id,
                )
                # Session permissions
                if user_id:
                    scope.grant_permission(
                        user_id,
                        "user",
                        {Permission.READ, Permission.WRITE, Permission.DELETE},
                    )
                scope.grant_permission(
                    agent_id,
                    "agent",
                    {Permission.READ, Permission.WRITE, Permission.DELETE, Permission.ADMIN},
                )
            current_parent = session_scope_id

        return current_parent


# Singleton registry instance with thread-safe initialization
_scope_registry: ScopeRegistry | None = None
_scope_registry_lock = threading.Lock()


def get_scope_registry() -> ScopeRegistry:
    """Get the global scope registry (singleton).

    Thread-safe: Uses double-checked locking to ensure only one
    instance is created even when called from multiple threads.
    """
    global _scope_registry

    # Fast path: registry already exists
    if _scope_registry is not None:
        return _scope_registry

    # Slow path: need to create registry
    with _scope_registry_lock:
        # Double-check after acquiring lock
        if _scope_registry is None:
            _scope_registry = ScopeRegistry()
        return _scope_registry


def reset_scope_registry() -> None:
    """Reset the scope registry (for testing).

    Thread-safe: Acquires lock before modifying singleton.
    """
    global _scope_registry
    with _scope_registry_lock:
        _scope_registry = None
