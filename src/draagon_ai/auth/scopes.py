"""Credential scoping for multi-tenant applications.

CredentialScope defines at what level a credential is stored and accessed.
This enables applications to have:
- Global credentials (shared by all users)
- Org-level credentials (shared within an organization)
- User-level credentials (personal to each user)
- Session-level credentials (temporary, single session)
"""

from enum import Enum


class CredentialScope(Enum):
    """Scope at which credentials are stored.

    Credentials cascade: global -> org -> user -> session
    More specific scopes override more general ones.
    """

    GLOBAL = "global"  # Available to all (e.g., system API keys)
    ORG = "org"  # Shared within organization (e.g., team calendar)
    USER = "user"  # Personal to user (e.g., personal calendar OAuth)
    SESSION = "session"  # Single session only (e.g., temporary access)

    def __str__(self) -> str:
        return self.value
