"""Tests for hierarchical scopes (Phase C.1)."""

import pytest
from datetime import datetime, timedelta

from draagon_ai.memory import (
    HierarchicalScope,
    ScopeType,
    Permission,
    ScopePermission,
    ScopeRegistry,
    get_scope_registry,
    reset_scope_registry,
)


@pytest.fixture(autouse=True)
def reset_registry():
    """Reset scope registry before each test."""
    reset_scope_registry()
    yield
    reset_scope_registry()


class TestScopePermission:
    """Tests for ScopePermission class."""

    def test_create_permission(self):
        """Test basic permission creation."""
        perm = ScopePermission(
            grantee_id="roxy",
            grantee_type="agent",
            permissions={Permission.READ, Permission.WRITE},
        )

        assert perm.grantee_id == "roxy"
        assert perm.grantee_type == "agent"
        assert Permission.READ in perm.permissions
        assert Permission.WRITE in perm.permissions
        assert Permission.DELETE not in perm.permissions

    def test_has_permission(self):
        """Test permission checking."""
        perm = ScopePermission(
            grantee_id="roxy",
            grantee_type="agent",
            permissions={Permission.READ},
        )

        assert perm.has_permission(Permission.READ) is True
        assert perm.has_permission(Permission.WRITE) is False

    def test_admin_includes_all(self):
        """Test that ADMIN permission includes all others."""
        perm = ScopePermission(
            grantee_id="roxy",
            grantee_type="agent",
            permissions={Permission.ADMIN},
        )

        assert perm.has_permission(Permission.READ) is True
        assert perm.has_permission(Permission.WRITE) is True
        assert perm.has_permission(Permission.DELETE) is True
        assert perm.has_permission(Permission.ADMIN) is True

    def test_expired_permission(self):
        """Test expired permissions."""
        past = datetime.now() - timedelta(hours=1)
        perm = ScopePermission(
            grantee_id="roxy",
            grantee_type="agent",
            permissions={Permission.READ},
            expires_at=past,
        )

        assert perm.has_permission(Permission.READ) is False


class TestHierarchicalScope:
    """Tests for HierarchicalScope class."""

    def test_create_scope(self):
        """Test basic scope creation."""
        scope = HierarchicalScope(
            scope_id="agent:roxy",
            scope_type=ScopeType.AGENT,
            name="Roxy Agent",
        )

        assert scope.scope_id == "agent:roxy"
        assert scope.scope_type == ScopeType.AGENT
        assert scope.name == "Roxy Agent"

    def test_world_scope_always_readable(self):
        """Test that world scope is always readable."""
        scope = HierarchicalScope(
            scope_id="world:global",
            scope_type=ScopeType.WORLD,
        )

        # Any agent can read world scope
        assert scope.can_read("any_agent") is True
        assert scope.can_read("another_agent", "user123") is True

    def test_permission_grant(self):
        """Test granting permissions."""
        scope = HierarchicalScope(
            scope_id="agent:roxy",
            scope_type=ScopeType.AGENT,
        )

        grant = scope.grant_permission(
            grantee_id="roxy",
            grantee_type="agent",
            permissions={Permission.READ, Permission.WRITE},
        )

        assert len(scope.permissions) == 1
        assert scope.can_read("roxy") is True
        assert scope.can_write("roxy") is True
        assert scope.can_delete("roxy") is False

    def test_permission_revoke(self):
        """Test revoking permissions."""
        scope = HierarchicalScope(
            scope_id="agent:roxy",
            scope_type=ScopeType.AGENT,
        )

        scope.grant_permission(
            grantee_id="roxy",
            grantee_type="agent",
            permissions={Permission.READ},
        )

        assert scope.can_read("roxy") is True

        result = scope.revoke_permission("roxy", "agent")
        assert result is True
        assert scope.can_read("roxy") is False

    def test_user_permissions(self):
        """Test user-level permissions."""
        scope = HierarchicalScope(
            scope_id="user:roxy:doug",
            scope_type=ScopeType.USER,
        )

        # Grant read to user
        scope.grant_permission(
            grantee_id="doug",
            grantee_type="user",
            permissions={Permission.READ, Permission.WRITE},
        )

        # User can read/write
        assert scope.can_read("roxy", "doug") is True
        assert scope.can_write("roxy", "doug") is True

        # Other user cannot
        assert scope.can_read("roxy", "other_user") is False

    def test_default_ttl_by_scope_type(self):
        """Test default TTLs are set by scope type."""
        session = HierarchicalScope(scope_id="session:1", scope_type=ScopeType.SESSION)
        assert session.default_ttl == timedelta(hours=1)

        user = HierarchicalScope(scope_id="user:1", scope_type=ScopeType.USER)
        assert user.default_ttl == timedelta(days=30)

        world = HierarchicalScope(scope_id="world:1", scope_type=ScopeType.WORLD)
        assert world.default_ttl is None  # Permanent

    def test_should_promote(self):
        """Test promotion threshold logic."""
        scope = HierarchicalScope(
            scope_id="user:roxy:doug",
            scope_type=ScopeType.USER,
            parent_scope_id="agent:roxy",
            promotion_threshold=0.7,
            promotion_access_count=5,
        )

        # Below both thresholds
        assert scope.should_promote(0.5, 3) is False

        # Above importance threshold
        assert scope.should_promote(0.8, 3) is True

        # Above access threshold
        assert scope.should_promote(0.5, 6) is True

    def test_promotion_disabled(self):
        """Test that promotion can be disabled."""
        scope = HierarchicalScope(
            scope_id="user:roxy:doug",
            scope_type=ScopeType.USER,
            parent_scope_id="agent:roxy",
            auto_promote=False,
        )

        # Even with high importance, no promotion
        assert scope.should_promote(1.0, 100) is False

    def test_serialization(self):
        """Test to_dict and from_dict round-trip."""
        original = HierarchicalScope(
            scope_id="agent:roxy",
            scope_type=ScopeType.AGENT,
            name="Roxy Agent",
            parent_scope_id="context:home",
            child_scope_ids=["user:roxy:doug", "user:roxy:maya"],
            promotion_threshold=0.8,
            max_items=1000,
            metadata={"custom": "data"},
        )

        original.grant_permission(
            grantee_id="roxy",
            grantee_type="agent",
            permissions={Permission.ADMIN},
        )

        data = original.to_dict()
        restored = HierarchicalScope.from_dict(data)

        assert restored.scope_id == original.scope_id
        assert restored.scope_type == original.scope_type
        assert restored.name == original.name
        assert restored.parent_scope_id == original.parent_scope_id
        assert restored.child_scope_ids == original.child_scope_ids
        assert restored.promotion_threshold == original.promotion_threshold
        assert restored.max_items == original.max_items
        assert restored.metadata == original.metadata
        assert len(restored.permissions) == 1


class TestScopeRegistry:
    """Tests for ScopeRegistry class."""

    def test_registry_has_world_scope(self):
        """Test that registry is initialized with world scope."""
        registry = ScopeRegistry()
        world = registry.get("world:global")

        assert world is not None
        assert world.scope_type == ScopeType.WORLD

    def test_create_scope(self):
        """Test creating a new scope."""
        registry = ScopeRegistry()

        scope = registry.create_scope(
            scope_type=ScopeType.CONTEXT,
            name="Home",
            scope_id="context:home",
        )

        assert scope.scope_id == "context:home"
        assert scope.scope_type == ScopeType.CONTEXT
        assert scope.parent_scope_id == "world:global"

        # Should be retrievable
        retrieved = registry.get("context:home")
        assert retrieved == scope

    def test_create_scope_auto_id(self):
        """Test that scope IDs are auto-generated if not provided."""
        registry = ScopeRegistry()

        scope = registry.create_scope(
            scope_type=ScopeType.AGENT,
            name="Roxy",
        )

        assert scope.scope_id.startswith("agent:")

    def test_create_scope_validates_parent(self):
        """Test that parent scope must exist."""
        registry = ScopeRegistry()

        with pytest.raises(ValueError, match="Parent scope not found"):
            registry.create_scope(
                scope_type=ScopeType.USER,
                name="Doug",
                parent_scope_id="nonexistent:parent",
            )

    def test_parent_child_relationship(self):
        """Test parent-child relationship is maintained."""
        registry = ScopeRegistry()

        context = registry.create_scope(
            scope_type=ScopeType.CONTEXT,
            name="Home",
            scope_id="context:home",
        )

        agent = registry.create_scope(
            scope_type=ScopeType.AGENT,
            name="Roxy",
            scope_id="agent:roxy",
            parent_scope_id="context:home",
        )

        # Parent should have child
        assert "agent:roxy" in context.child_scope_ids

        # Child should have parent
        assert agent.parent_scope_id == "context:home"

    def test_get_ancestors(self):
        """Test getting ancestor scopes."""
        registry = ScopeRegistry()

        registry.create_scope(ScopeType.CONTEXT, "Home", scope_id="context:home")
        registry.create_scope(
            ScopeType.AGENT, "Roxy",
            scope_id="agent:roxy",
            parent_scope_id="context:home",
        )
        registry.create_scope(
            ScopeType.USER, "Doug",
            scope_id="user:doug",
            parent_scope_id="agent:roxy",
        )

        ancestors = registry.get_ancestors("user:doug")

        assert len(ancestors) == 3  # agent:roxy, context:home, world:global
        assert ancestors[0].scope_id == "agent:roxy"
        assert ancestors[1].scope_id == "context:home"
        assert ancestors[2].scope_id == "world:global"

    def test_get_descendants(self):
        """Test getting descendant scopes."""
        registry = ScopeRegistry()

        registry.create_scope(ScopeType.CONTEXT, "Home", scope_id="context:home")
        registry.create_scope(
            ScopeType.AGENT, "Roxy",
            scope_id="agent:roxy",
            parent_scope_id="context:home",
        )
        registry.create_scope(
            ScopeType.USER, "Doug",
            scope_id="user:doug",
            parent_scope_id="agent:roxy",
        )
        registry.create_scope(
            ScopeType.USER, "Maya",
            scope_id="user:maya",
            parent_scope_id="agent:roxy",
        )

        descendants = registry.get_descendants("context:home")

        assert len(descendants) == 3  # agent:roxy, user:doug, user:maya
        scope_ids = {d.scope_id for d in descendants}
        assert "agent:roxy" in scope_ids
        assert "user:doug" in scope_ids
        assert "user:maya" in scope_ids

    def test_build_scope_id(self):
        """Test building scope IDs from components."""
        registry = ScopeRegistry()

        assert registry.build_scope_id(ScopeType.WORLD) == "world:global"
        assert registry.build_scope_id(ScopeType.CONTEXT, context_id="home") == "context:home"
        assert registry.build_scope_id(ScopeType.AGENT, agent_id="roxy") == "agent:roxy"
        assert registry.build_scope_id(
            ScopeType.USER,
            agent_id="roxy",
            user_id="doug",
        ) == "user:roxy:doug"

    def test_ensure_scope_hierarchy(self):
        """Test ensuring full scope hierarchy exists."""
        registry = ScopeRegistry()

        # Create full hierarchy
        leaf_scope_id = registry.ensure_scope_hierarchy(
            context_id="home",
            agent_id="roxy",
            user_id="doug",
            session_id="session123",
        )

        # All scopes should exist
        assert registry.get("context:home") is not None
        assert registry.get("agent:roxy") is not None
        assert registry.get("user:roxy:doug") is not None
        assert "session:roxy:doug:session123" in leaf_scope_id

        # Agent should have permissions on their scope
        agent_scope = registry.get("agent:roxy")
        assert agent_scope.can_read("roxy") is True
        assert agent_scope.can_write("roxy") is True
        assert agent_scope.can_delete("roxy") is True

    def test_get_readable_scopes(self):
        """Test getting all readable scopes for an agent."""
        registry = ScopeRegistry()

        registry.create_scope(ScopeType.CONTEXT, "Home", scope_id="context:home")
        agent_scope = registry.create_scope(
            ScopeType.AGENT, "Roxy",
            scope_id="agent:roxy",
            parent_scope_id="context:home",
        )
        agent_scope.grant_permission(
            "roxy", "agent",
            {Permission.READ, Permission.WRITE},
        )

        readable = registry.get_readable_scopes("roxy")

        # Should include world (always readable) and agent:roxy (explicitly granted)
        scope_ids = {s.scope_id for s in readable}
        assert "world:global" in scope_ids
        assert "agent:roxy" in scope_ids


class TestSingletonRegistry:
    """Tests for the singleton registry pattern."""

    def test_get_scope_registry_returns_same_instance(self):
        """Test that get_scope_registry returns the same instance."""
        reg1 = get_scope_registry()
        reg2 = get_scope_registry()

        assert reg1 is reg2

    def test_reset_creates_new_instance(self):
        """Test that reset creates a new instance."""
        reg1 = get_scope_registry()
        reg1.create_scope(ScopeType.CONTEXT, "Test", scope_id="context:test")

        reset_scope_registry()

        reg2 = get_scope_registry()
        assert reg2.get("context:test") is None  # Reset cleared it


class TestPermissionInheritance:
    """Tests for permission inheritance from parent scopes.

    Permissions should be inherited from parent scopes. If a user has READ
    permission on "agent:roxy", they should also have READ on child scopes
    like "user:roxy:doug" and "session:roxy:doug:abc123".
    """

    def test_permission_inherited_from_parent(self):
        """Test that permissions are inherited from parent scope."""
        registry = ScopeRegistry()

        # Create hierarchy
        context = registry.create_scope(ScopeType.CONTEXT, "Home", scope_id="context:home")
        agent = registry.create_scope(
            ScopeType.AGENT, "Roxy",
            scope_id="agent:roxy",
            parent_scope_id="context:home",
        )
        user = registry.create_scope(
            ScopeType.USER, "Doug",
            scope_id="user:roxy:doug",
            parent_scope_id="agent:roxy",
        )

        # Grant READ on agent scope
        agent.grant_permission("roxy", "agent", {Permission.READ, Permission.WRITE})

        # Direct permission check (on agent scope)
        assert registry.check_permission("agent:roxy", "roxy", None, Permission.READ)
        assert registry.check_permission("agent:roxy", "roxy", None, Permission.WRITE)

        # Inherited permission check (on user scope)
        assert registry.check_permission("user:roxy:doug", "roxy", None, Permission.READ)
        assert registry.check_permission("user:roxy:doug", "roxy", None, Permission.WRITE)

        # DELETE was not granted
        assert not registry.check_permission("user:roxy:doug", "roxy", None, Permission.DELETE)

    def test_permission_inherited_multiple_levels(self):
        """Test that permissions are inherited through multiple levels."""
        registry = ScopeRegistry()

        # Create hierarchy: context -> agent -> user -> session
        context = registry.create_scope(ScopeType.CONTEXT, "Home", scope_id="context:home")
        agent = registry.create_scope(
            ScopeType.AGENT, "Roxy",
            scope_id="agent:roxy",
            parent_scope_id="context:home",
        )
        user = registry.create_scope(
            ScopeType.USER, "Doug",
            scope_id="user:roxy:doug",
            parent_scope_id="agent:roxy",
        )
        session = registry.create_scope(
            ScopeType.SESSION, "Session1",
            scope_id="session:roxy:doug:123",
            parent_scope_id="user:roxy:doug",
        )

        # Grant permission at context level
        context.grant_permission("roxy", "agent", {Permission.READ})

        # Should be inherited all the way down
        assert registry.check_permission("agent:roxy", "roxy", None, Permission.READ)
        assert registry.check_permission("user:roxy:doug", "roxy", None, Permission.READ)
        assert registry.check_permission("session:roxy:doug:123", "roxy", None, Permission.READ)

    def test_child_can_have_more_permissions_than_parent(self):
        """Test that child scope can grant additional permissions."""
        registry = ScopeRegistry()

        # Create hierarchy
        agent = registry.create_scope(
            ScopeType.AGENT, "Roxy",
            scope_id="agent:roxy",
        )
        user = registry.create_scope(
            ScopeType.USER, "Doug",
            scope_id="user:roxy:doug",
            parent_scope_id="agent:roxy",
        )

        # Grant READ on agent scope
        agent.grant_permission("roxy", "agent", {Permission.READ})

        # Grant DELETE on user scope (additional permission)
        user.grant_permission("roxy", "agent", {Permission.DELETE})

        # Agent has READ on both (inherited) and DELETE only on user
        assert registry.check_permission("agent:roxy", "roxy", None, Permission.READ)
        assert not registry.check_permission("agent:roxy", "roxy", None, Permission.DELETE)

        assert registry.check_permission("user:roxy:doug", "roxy", None, Permission.READ)  # inherited
        assert registry.check_permission("user:roxy:doug", "roxy", None, Permission.DELETE)  # direct

    def test_user_permission_inheritance(self):
        """Test user-level permission inheritance."""
        registry = ScopeRegistry()

        # Create hierarchy
        agent = registry.create_scope(
            ScopeType.AGENT, "Roxy",
            scope_id="agent:roxy",
        )
        user = registry.create_scope(
            ScopeType.USER, "Doug",
            scope_id="user:roxy:doug",
            parent_scope_id="agent:roxy",
        )
        session = registry.create_scope(
            ScopeType.SESSION, "Session1",
            scope_id="session:roxy:doug:123",
            parent_scope_id="user:roxy:doug",
        )

        # Grant user doug permission on user scope
        user.grant_permission("doug", "user", {Permission.READ, Permission.WRITE})

        # User permission should be inherited to session
        assert registry.check_permission("user:roxy:doug", "roxy", "doug", Permission.READ)
        assert registry.check_permission("session:roxy:doug:123", "roxy", "doug", Permission.READ)

        # Other user should not have access
        assert not registry.check_permission("user:roxy:doug", "roxy", "maya", Permission.READ)
        assert not registry.check_permission("session:roxy:doug:123", "roxy", "maya", Permission.READ)

    def test_world_scope_always_readable_with_inheritance(self):
        """Test that world scope is always readable through inheritance check."""
        registry = ScopeRegistry()

        # Any agent can read world scope through inheritance check
        assert registry.check_permission("world:global", "any_agent", None, Permission.READ)
        assert registry.check_permission("world:global", "another_agent", "user123", Permission.READ)

        # But cannot write
        assert not registry.check_permission("world:global", "any_agent", None, Permission.WRITE)

    def test_nonexistent_scope_returns_false(self):
        """Test that checking permission on nonexistent scope returns False."""
        registry = ScopeRegistry()

        assert not registry.check_permission("nonexistent:scope", "agent", None, Permission.READ)

    def test_check_permission_with_inheritance_method(self):
        """Test the scope's check_permission_with_inheritance method directly."""
        registry = ScopeRegistry()

        # Create hierarchy
        agent = registry.create_scope(
            ScopeType.AGENT, "Roxy",
            scope_id="agent:roxy",
        )
        user = registry.create_scope(
            ScopeType.USER, "Doug",
            scope_id="user:roxy:doug",
            parent_scope_id="agent:roxy",
        )

        # Grant permission on agent
        agent.grant_permission("roxy", "agent", {Permission.READ})

        # Direct method call with registry
        assert user.check_permission_with_inheritance("roxy", None, Permission.READ, registry)

        # Without registry, only checks direct permission
        assert not user.check_permission_with_inheritance("roxy", None, Permission.READ, None)


class TestThreadSafeScopeRegistry:
    """Tests for thread-safe singleton implementation of scope registry."""

    def test_concurrent_get_registry(self):
        """Test that concurrent get_scope_registry calls return same instance."""
        import concurrent.futures

        reset_scope_registry()

        results = []
        errors = []

        def get_registry():
            try:
                registry = get_scope_registry()
                results.append(id(registry))
            except Exception as e:
                errors.append(e)

        # Run many threads concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(get_registry) for _ in range(100)]
            concurrent.futures.wait(futures)

        # No errors should occur
        assert len(errors) == 0, f"Errors occurred: {errors}"

        # All threads should get the same instance
        assert len(results) == 100
        assert len(set(results)) == 1, f"Multiple instances created: {set(results)}"

    def test_concurrent_reset_and_get_registry(self):
        """Test thread-safety when resetting and getting concurrently."""
        import concurrent.futures
        import random

        reset_scope_registry()

        errors = []

        def access_registry():
            try:
                if random.choice([True, False]):
                    # Reset registry
                    reset_scope_registry()
                else:
                    # Get registry
                    registry = get_scope_registry()
                    assert registry is not None
            except Exception as e:
                errors.append(e)

        # Run many threads doing mixed operations
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(access_registry) for _ in range(100)]
            concurrent.futures.wait(futures)

        # No errors should occur
        assert len(errors) == 0, f"Errors occurred: {errors}"
